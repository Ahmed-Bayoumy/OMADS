"""Benchmark OMADS' multiobjective MADS on the COCO ``bbob-biobj`` suite.

  python -m pip install -r benchmarks/coco/requirements.txt
  python benchmarks/coco/run_omads_bbob_biobj.py
  python benchmarks/coco/run_omads_bbob_biobj.py -d 2,3 -f 1-10 -i 1-5 -b 200

Each bi-objective COCO problem ``x -> (f1(x), f2(x))`` is handed to OMADS as a
callable blackbox returning ``[[f1, f2], [0]]`` with ``"nobj": 2`` and
``"is_pareto": True``: to ``poll.main`` (poll step only, the default) or, with
``--solver mads``, to ``mads.main`` (search and poll steps, ``--s-method``
sampling search with ``--ns`` samples per step). The search box is COCO's region of
interest ``[-5, 5]^n``, where the Pareto sets of all ``bbob-biobj`` functions
lie.

COCO's bi-objective logger records the hypervolume of all the non-dominated
points found so far, whatever the run that found them, so an instance whose
run stops early (all frame sizes below ``--tol``) is restarted from a new
initial point and seed until its ``budget_multiplier * dimension`` budget is
used; COCO records the restarts of an instance as one trial. The seed and
the initial point of each restart depend only on the restart number, so any
batch split reproduces the same runs.

Results go to ``exdata/<output-folder>``. ``--batches N --batch k`` splits the
suite over N processes; ``reproduce_biobj_results.py`` runs all the batches,
gathers them and post-processes them.
"""

import argparse
import contextlib
import logging
import os
import shutil
import sys
import time
from typing import Dict

import cocoex
import numpy as np

from OMADS import mads, poll

SUITE_NAME = "bbob-biobj"
REGION_OF_INTEREST = (-5.0, 5.0)


def make_blackbox(problem):
  """Wrap a COCO bi-objective problem as an OMADS callable blackbox."""

  def blackbox(x):
    f = problem(np.asarray(x, dtype=float))
    return [[float(v) for v in f], [0.0]]

  return blackbox


def build_omads_input(problem, x0, budget: int, seed: int,
                      args: argparse.Namespace) -> Dict:
  """OMADS input dictionary for one Pareto run on ``problem``."""
  d = problem.dimension
  lo, hi = REGION_OF_INTEREST
  data = {
      "evaluator": {"blackbox": make_blackbox(problem)},
      "param": {
          "name": problem.id,
          "baseline": [float(v) for v in np.clip(x0, lo, hi)],
          "lb": [lo] * d,
          "ub": [hi] * d,
          "var_names": [f"x{i}" for i in range(d)],
          "fun_names": ["f1", "f2"],
          "scaling": [hi - lo] * d,
          "nobj": 2,
          "is_pareto": True,
          "mesh_type": args.mesh_type,
          "post_dir": args.omads_post_dir,
      },
      "options": {
          "seed": seed, "budget": budget, "tol": args.tol,
          "psize_init": args.psize_init, "display": False,
          "check_cache": True, "store_cache": True, "rich_direction": True,
          "opportunistic": args.opportunistic,
          "save_results": False, "is_verbose": False,
      },
  }
  if args.direction_type:   # otherwise OMADS' default for 2 objectives
    data["options"]["direction_type"] = args.direction_type
  if args.solver == "mads":
    # The ACTIVE search draws ns BiTPE proposals per step, as the published
    # bbob-constrained configurations do (OMADS' default is 10).
    data["search"] = {"type": "sampling", "s_method": args.s_method,
                      "ns": args.ns, "batch": args.ns}
  return data


def run_omads(problem, x0, budget: int, seed: int, args: argparse.Namespace):
  """One Pareto run of OMADS on ``problem``."""
  stderr = sys.stderr
  try:
    # Silence OMADS' progress bar so COCO's per-problem summary stays readable.
    with open(os.devnull, "w", encoding="utf-8") as devnull, \
            contextlib.redirect_stdout(devnull):
      solver = mads if args.solver == "mads" else poll
      solver.main(build_omads_input(problem, x0, budget, seed, args))
  except ValueError as exc:
    # After its main loop, OMADS reports the hypervolume of its front and
    # raises when the front is degenerate (one point, or one objective
    # constant). COCO has recorded every evaluation by then, so carry on.
    if "Max and min values" not in str(exc):
      raise
  finally:
    sys.stderr = stderr
    # OMADS logs to <post_dir>/OMADS.log through logging.basicConfig, which
    # keeps the file open; close it so that the folder can be removed.
    root = logging.getLogger()
    for handler in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
      root.removeHandler(handler)
      handler.close()
    for suffix in ("_run", "_ND"):
      shutil.rmtree(os.path.join(args.omads_post_dir, problem.id + suffix),
                    ignore_errors=True)


def solve(problem, args: argparse.Namespace) -> int:
  """Run OMADS on ``problem``, restarted until its budget is used; return the
  number of runs."""
  budget = int(args.budget_multiplier * problem.dimension)
  restart = 0
  while problem.evaluations < budget and restart < args.max_restarts:
    seed = args.seed + restart
    np.random.seed(seed)   # cocoex draws the restart points with numpy
    run_omads(problem, problem.initial_solution_proposal(restart),
              budget=budget - problem.evaluations, seed=seed, args=args)
    restart += 1
  return restart


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--budget-multiplier", "-b", type=float, default=1000,
                      help="evaluation budget per instance = this x dimension"
                      " (default: 1000)")
  parser.add_argument("--dimensions", "-d", default="2,3,5,10",
                      help="e.g. '2,3,5,10' (suite has 2,3,5,10,20,40)")
  parser.add_argument("--functions", "-f", default="",
                      help="function indices, e.g. '1-10' (1-55; default: all)")
  parser.add_argument("--instances", "-i", default="1-15",
                      help="instance indices (default: 1-15)")
  parser.add_argument("--batches", type=int, default=1,
                      help="split the suite into this many batches")
  parser.add_argument("--batch", type=int, default=1,
                      help="which batch (1..batches) this process runs")
  parser.add_argument("--max-restarts", type=int, default=20,
                      help="max runs per instance (default: 20)")
  parser.add_argument("--seed", type=int, default=1234,
                      help="base OMADS seed (offset per restart)")
  parser.add_argument("--psize-init", type=float, default=1.0,
                      help="OMADS initial frame size, in units of 1/10 of the"
                      " region of interest (default: 1)")
  parser.add_argument("--tol", type=float, default=1e-9,
                      help="stop a run when every frame size is below this"
                      " (default: 1e-9)")
  parser.add_argument("--mesh-type", default="ORTHO", choices=["ORTHO", "GMESH"],
                      help="OMADS mesh (default: ORTHO)")
  parser.add_argument("--direction-type", default=None,
                      choices=["ORTHO_2N", "N+1"],
                      help="poll directions (default: OMADS chooses, N+1 for 2 objectives)")
  parser.add_argument("--opportunistic", default=False,
                      action=argparse.BooleanOptionalAction,
                      help="stop a poll at its first success (default: off)")
  parser.add_argument("--solver", default="poll", choices=["mads", "poll"],
                      help="OMADS entry point: poll.main (poll step only,"
                      " default) or mads.main (search and poll steps)")
  parser.add_argument("--s-method", default="ACTIVE",
                      help="mads sampling search method (default: ACTIVE)")
  parser.add_argument("--ns", type=int, default=150,
                      help="mads samples per search step (default: 150)")
  parser.add_argument("--algorithm-name", default=None,
                      help="algorithm name in the COCO data (default:"
                      " OMADS-MADS-ns<ns> for mads, OMADS-MO for poll)")
  parser.add_argument("--output-folder", default=None,
                      help="folder name under exdata/ (default: derived)")
  return parser


def parse_args() -> argparse.Namespace:
  return build_parser().parse_args()


def main():
  args = parse_args()
  if args.algorithm_name is None:
    method = "" if args.s_method == "ACTIVE" else f"{args.s_method}-"
    args.algorithm_name = (f"OMADS-MADS-{method}ns{args.ns}" if args.solver == "mads"
                           else "OMADS-MO")
  if not 1 <= args.batch <= args.batches:
    raise SystemExit("--batch must be in 1..--batches")
  suite_options = " ".join(
      f"{key}: {val}" for key, val in (
          ("dimensions", args.dimensions),
          ("function_indices", args.functions),
          ("instance_indices", args.instances)) if val)
  suite = cocoex.Suite(SUITE_NAME, "", suite_options)
  folder = args.output_folder or "{}_{}x_on_{}{}".format(
      args.algorithm_name, int(args.budget_multiplier), SUITE_NAME,
      f"_b{args.batch}" if args.batches > 1 else "")
  observer = cocoex.Observer(SUITE_NAME, (
      f"result_folder: {folder} algorithm_name: {args.algorithm_name} "
      "log_nondominated: none log_decision_variables: none "
      f"algorithm_info: \"OMADS {args.solver}.main, Pareto, {args.mesh_type} mesh, "
      + (f"search={args.s_method}, ns={args.ns}, " if args.solver == "mads" else "") +
      f"poll={args.direction_type or 'default'}, opportunistic={args.opportunistic}, "
      f"box=[-5,5]^n, budget={int(args.budget_multiplier)}n\""))
  args.omads_post_dir = os.path.join(observer.result_folder, "omads_post")
  os.makedirs(observer.result_folder, exist_ok=True)
  minimal_print = cocoex.utilities.MiniPrint()
  print(f"Benchmarking {args.algorithm_name} on {SUITE_NAME} "
        f"[{suite_options}], results in {observer.result_folder}")
  tic = time.perf_counter()
  with open(os.path.join(observer.result_folder, "runs.txt"), "a",
            encoding="utf-8") as runs_log:
    for problem in suite:
      if problem.index % args.batches != args.batch - 1:
        continue
      problem.observe_with(observer)
      runs = solve(problem, args)
      runs_log.write("{} {} {} {}\n".format(*problem.id_triple, runs))
      runs_log.flush()
      minimal_print(problem)
  shutil.rmtree(args.omads_post_dir, ignore_errors=True)
  print(f"\nDone in {time.perf_counter() - tic:.1f} s")


if __name__ == "__main__":
  main()
