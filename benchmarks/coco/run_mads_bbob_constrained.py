"""Benchmark OMADS' MADS on the COCO ``bbob-constrained`` suite.

Follows the COCO getting-started recipe (https://coco-platform.org/getting-started/):

  python -m pip install -r benchmarks/coco/requirements.txt
  python benchmarks/coco/run_mads_bbob_constrained.py
  python benchmarks/coco/run_mads_bbob_constrained.py --preset halton
  python benchmarks/coco/run_mads_bbob_constrained.py -d 2,3,5 -f 1-12 -i 1-3 \
      --budget-multiplier 50

The defaults reproduce the configuration published in the OMADS documentation
(``--preset A``): ``mads.main`` with the ACTIVE (BiTPE) sampling search,
ns = min((n+1)(n+2)/2, 2n), the N+1 poll and opportunistic evaluation,
on dimensions 2, 3, 5 and 10, instances 1-15, with a budget of 100 x dimension.
``--preset B``, ``--preset halton`` and ``--preset initial`` give the other
published configurations; any explicit option overrides its preset value. To run all of
them and regenerate the documentation figures, use ``reproduce_results.py``.

Each COCO problem ``x -> f(x)`` with constraints ``g(x) <= 0`` is handed to
``OMADS.mads.main`` as a callable blackbox returning ``[f, g]``, with every
constraint treated by the progressive barrier (PB). Runs stop early once the
final COCO target is hit, and an unsolved instance is restarted (from a new
initial point and seed) until its ``budget_multiplier * dimension`` budget is
used up; COCO records the restarts of an instance as one trial. The seed and
the initial point of each restart depend only on the restart number, so any
subset of the suite (``--only``) or batch split reproduces the same runs.

Results go to ``exdata/<output-folder>`` and are post-processed with
``cocopp`` (HTML report in ``ppdata/``) unless ``--no-postprocess`` is given.
Use ``--batches N --batch k`` to split the suite over N parallel processes,
then post-process all batch folders together: ``python -m cocopp exdata/...``.
"""

import argparse
import contextlib
import os
import shutil
import sys
import time
from typing import Dict

import cocoex
import numpy as np

from OMADS import mads, poll

SUITE_NAME = "bbob-constrained"
# bbob-constrained combines 9 raw functions with 6 constraint counts each;
# raw functions 8 and 9 (separable and rotated Rastrigin) are multimodal.
MULTIMODAL_FUNCTIONS = range(43, 55)

# Configurations published in the documentation (Benchmarking > Results on
# bbob-constrained). A is the default.
PRESETS = {
    "A": dict(s_method="ACTIVE", ns="min2n", ns_multimodal="min2n",
              search_every=1, direction_type="N+1", opportunistic=True,
              algorithm_name="OMADS-A"),
    "B": dict(s_method="ACTIVE", ns="quad", ns_multimodal="quad",
              search_every=3, direction_type="N+1", opportunistic=True,
              algorithm_name="OMADS-B"),
    "halton": dict(s_method="HALTON", ns="1", ns_multimodal="56",
                   search_every=1, direction_type="ORTHO_2N",
                   opportunistic=False, algorithm_name="OMADS-Halton"),
    # First ACTIVE configuration, reported in the documentation's table only.
    "initial": dict(s_method="ACTIVE", ns="auto", ns_multimodal="auto",
                    search_every=1, direction_type="ORTHO_2N",
                    opportunistic=False, algorithm_name="OMADS-ACTIVE"),
}


def search_samples(problem, args: argparse.Namespace) -> int:
  """Samples per sampling-search step: a large sample only pays off on the
  multimodal functions, elsewhere it eats most of the budget. For an
  n-variable problem, ``quad`` means (n + 1)(n + 2) / 2 (the size of a full
  quadratic basis), ``auto`` means (n + 1)(n + 2) / 2 + 10 and ``min2n`` means
  min((n + 1)(n + 2) / 2, 2n)."""
  multimodal = problem.id_function in MULTIMODAL_FUNCTIONS
  ns = args.ns_multimodal if multimodal else args.ns
  n = problem.dimension
  if ns == "min2n":
    return min((n + 1) * (n + 2) // 2, 2 * n)
  if ns == "quad":
    return (n + 1) * (n + 2) // 2
  if ns == "auto":
    return (n + 1) * (n + 2) // 2 + 10
  return int(ns)


class TargetHit(Exception):
  """Raised from the blackbox to stop MADS once COCO's final target is hit."""


def make_blackbox(problem):
  """Wrap a COCO constrained problem as an OMADS callable blackbox."""

  def blackbox(x):
    if problem.final_target_hit:
      raise TargetHit
    x = np.asarray(x, dtype=float)
    g = problem.constraint(x)
    f = problem(x)
    return [float(f), [float(gi) for gi in g]]

  return blackbox


def build_omads_input(problem, x0, budget: int, seed: int,
                      args: argparse.Namespace) -> Dict:
  """OMADS input dictionary for one run on ``problem``."""
  d = problem.dimension
  lb = [float(v) for v in problem.lower_bounds]
  ub = [float(v) for v in problem.upper_bounds]
  data = {
      "evaluator": {"blackbox": make_blackbox(problem)},
      "param": {
          "name": problem.id,
          "baseline": [float(v) for v in np.clip(x0, lb, ub)],
          "lb": lb,
          "ub": ub,
          "var_names": [f"x{i}" for i in range(d)],
          "scaling": [hi - lo for lo, hi in zip(lb, ub)],
          "constraints_type": ["PB"] * problem.number_of_constraints,
          "h_max": np.inf,
          "mesh_type": args.mesh_type,
          "min_mesh_size": [args.min_mesh_size] * d,
          "post_dir": args.omads_post_dir,
      },
      "options": {
          "seed": seed, "budget": budget, "tol": args.tol,
          "psize_init": args.psize_init, "display": False,
          "check_cache": True, "store_cache": True, "rich_direction": True,
          "opportunistic": args.opportunistic, "save_results": False, "is_verbose": False,
          "precision": "high", "anisotropy_factor": args.anisotropy_factor,
          "direction_type": args.direction_type,
          "search_every": args.search_every,
          "on_min_mesh": args.on_min_mesh,
      },
      "search": {
          "type": args.search_type,
          "s_method": args.s_method,
          "ns": search_samples(problem, args),
          # The published ACTIVE configurations draw ns BiTPE proposals per
          # search step (OMADS' default is 10).
          "batch": search_samples(problem, args),
          "visualize": False,
      },
  }
  return data


def run_mads(problem, x0, budget: int, seed: int,
             args: argparse.Namespace):
  """Run MADS on ``problem`` until its budget is used or the target is hit."""
  stderr = sys.stderr
  try:
    # Silence OMADS' progress bar so COCO's per-problem summary stays readable.
    with open(os.devnull, "w", encoding="utf-8") as devnull, \
            contextlib.redirect_stdout(devnull):
      solver = poll if args.solver == "poll" else mads
      solver.main(build_omads_input(problem, x0, budget, seed, args))
  except TargetHit:
    pass
  finally:
    # mads.main wraps stderr and only restores it on normal return.
    sys.stderr = stderr
    if not args.keep_omads_logs:
      # OMADS writes every evaluation of the run to <post_dir>/<name>_run/;
      # COCO keeps its own record, so drop it (hundreds of KB per problem).
      shutil.rmtree(os.path.join(args.omads_post_dir, f"{problem.id}_run"),
                    ignore_errors=True)


def solve(problem, args: argparse.Namespace) -> int:
  """Run MADS on ``problem``, restarted until its budget is used or its final
  target is hit; return the number of runs."""
  budget = int(args.budget_multiplier * problem.dimension)
  restart = 0
  while (problem.evaluations < budget and not problem.final_target_hit
         and restart < args.max_restarts):
    seed = args.seed + restart
    np.random.seed(seed)   # cocoex draws the restart points with numpy
    run_mads(problem, problem.initial_solution_proposal(restart),
             budget=budget - problem.evaluations, seed=seed, args=args)
    restart += 1
  return restart


def build_parser(description: str = "Benchmark OMADS MADS on COCO's"
                 " bbob-constrained suite.") -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("--preset", default="A", choices=sorted(PRESETS),
                      help="published OMADS configuration (default: A); the"
                      " options below override its values")
  parser.add_argument("--budget-multiplier", "-b", type=float, default=100,
                      help="f-evaluation budget per instance = this x dimension"
                      " (default: 100)")
  parser.add_argument("--dimensions", "-d", default="2,3,5,10",
                      help="e.g. '2,3,5,10' (suite has 2,3,5,10,20,40; "
                      "default: 2,3,5,10)")
  parser.add_argument("--functions", "-f", default="",
                      help="function indices, e.g. '1-18' (1-54; default: all)")
  parser.add_argument("--instances", "-i", default="1-15",
                      help="instance indices, e.g. '1-15' (default: 1-15)")
  parser.add_argument("--batches", type=int, default=1,
                      help="split the suite into this many batches")
  parser.add_argument("--batch", type=int, default=1,
                      help="which batch (1..batches) this process runs")
  parser.add_argument("--max-restarts", type=int, default=20,
                      help="max runs per instance (default: 20)")
  parser.add_argument("--seed", type=int, default=1234,
                      help="base OMADS seed (offset per restart)")
  parser.add_argument("--psize-init", type=float, default=1.0,
                      help="OMADS initial poll size")
  parser.add_argument("--tol", type=float, default=1e-12,
                      help="OMADS minimum mesh/poll size tolerance (default:"
                      " 1e-12; the mesh size is ~frame^2, and 1e-9 stops"
                      " short of COCO's 1e-8 final target)")
  parser.add_argument("--mesh-type", default="GMESH",
                      choices=["GMESH", "ORTHO"],
                      help="GMESH (default) is the anisotropic granular"
                      " mesh: the frame grows only along successful"
                      " directions, which keeps MADS moving along the flat"
                      " valleys of the ill-conditioned functions")
  parser.add_argument("--anisotropy-factor", type=float, default=0.05,
                      help="GMESH anisotropy factor (default: 0.05)")
  parser.add_argument("--min-mesh-size", type=float, default=1e-13,
                      help="GMESH minimum mesh size (OMADS default 1e-9"
                      " caps the frame at ~5e-5, short of COCO's 1e-8"
                      " final target)")
  parser.add_argument("--search-type", default="sampling",
                      help="MADS search step type (default: sampling)")
  parser.add_argument("--s-method", default=None,
                      help="sampling method of the search step, e.g. ACTIVE"
                      " or HALTON (default: from the preset)")
  parser.add_argument("--ns", default=None,
                      help="samples per sampling-search step on functions"
                      " f1-f42: an integer, 'quad' for (n+1)(n+2)/2, 'auto'"
                      " for (n+1)(n+2)/2+10 or 'min2n' for"
                      " min((n+1)(n+2)/2, 2n) (default: from the preset)")
  parser.add_argument("--ns-multimodal", default=None,
                      help="samples per sampling-search step on the multimodal"
                      " Rastrigin functions f43-f54, same values as --ns"
                      " (default: from the preset)")
  parser.add_argument("--search-every", type=int, default=None,
                      help="run the sampling search only after every k-th"
                      " consecutive failed poll (default: from the preset)")
  parser.add_argument("--opportunistic", default=None,
                      action=argparse.BooleanOptionalAction,
                      help="stop evaluating a poll or search set at its first"
                      " improvement (default: from the preset)")
  parser.add_argument("--direction-type", default=None,
                      choices=["ORTHO_2N", "N+1"],
                      help="poll directions: 2n or n+1 (default: from the"
                      " preset)")
  parser.add_argument("--on-min-mesh", default="stop", choices=["stop", "search"],
                      help="when the poll size falls below --tol with budget"
                      " left: stop the run (COCO then restarts it from a new"
                      " point; default) or keep running the sampling search"
                      " with the initial mesh until it improves")
  parser.add_argument("--only", default=None,
                      help="file listing the problems to run, one"
                      " 'function dimension instance' triple per line")
  parser.add_argument("--folder-per-instance", action="store_true",
                      help="write each instance to its own folder under the"
                      " output folder, so that data sets can be assembled"
                      " instance by instance")
  parser.add_argument("--solver", default="mads", choices=["mads", "poll"],
                      help="OMADS entry point: mads.main (default) or poll.main"
                      " (poll step only; the search options are ignored)")
  parser.add_argument("--algorithm-name", default=None,
                      help="algorithm name in the COCO data (default: from the"
                      " preset, e.g. OMADS-A)")
  parser.add_argument("--output-folder", default=None,
                      help="folder name under exdata/ (default: derived)")
  parser.add_argument("--keep-omads-logs", action="store_true",
                      help="keep OMADS' per-problem evaluation history under"
                      " <output-folder>/omads_post (deleted by default; COCO"
                      " records the runs in its own files)")
  parser.add_argument("--no-postprocess", action="store_true",
                      help="skip running cocopp at the end")
  parser.add_argument("--cocopp-args", default="",
                      help="extra cocopp arguments, e.g. reference algorithms")
  return parser


def observer_options(args: argparse.Namespace, folder: str) -> str:
  """COCO observer options: result folder, algorithm name and description."""
  return (
      f"result_folder: {folder} algorithm_name: {args.algorithm_name} "
      f"algorithm_info: \"OMADS {args.solver}.main, PB constraints, {args.mesh_type} mesh, "
      f"search={args.search_type}/{args.s_method}, "
      f"ns={args.ns} (f43-f54: {args.ns_multimodal}), "
      f"search_every={args.search_every}, poll={args.direction_type}, "
      f"opportunistic={args.opportunistic}, on_min_mesh={args.on_min_mesh}, "
      f"preset={args.preset}\"")


def instance_folder(output_folder: str, key) -> str:
  """Folder of one (function, dimension, instance) under the output folder."""
  return "{}/f{:02d}_d{:02d}_i{:02d}".format(output_folder, *key)


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
  """Fill the options left unset with the values of ``args.preset``."""
  for key, value in PRESETS[args.preset].items():
    if getattr(args, key) is None:
      setattr(args, key, value)
  return args


def parse_args() -> argparse.Namespace:
  return apply_preset(build_parser().parse_args())


def main():  # noqa: C901
  args = parse_args()
  if not 1 <= args.batch <= args.batches:
    raise SystemExit("--batch must be in 1..--batches")

  suite_options = " ".join(
      f"{key}: {val}" for key, val in (
          ("dimensions", args.dimensions),
          ("function_indices", args.functions),
          ("instance_indices", args.instances)) if val)
  suite = cocoex.Suite(SUITE_NAME, "", suite_options)

  output_folder = args.output_folder or "{}_{}x_on_{}{}".format(
      args.algorithm_name, int(args.budget_multiplier), SUITE_NAME,
      f"_batch{args.batch}of{args.batches}" if args.batches > 1 else "")
  observer = cocoex.Observer(SUITE_NAME, observer_options(args, output_folder))
  args.omads_post_dir = os.path.join(observer.result_folder, "omads_post")
  minimal_print = cocoex.utilities.MiniPrint()
  only = None
  if args.only:
    with open(args.only, encoding="utf-8") as f:
      only = {tuple(int(v) for v in line.split()) for line in f if line.strip()}
  observers = {}

  def observer_of(problem):
    """The shared observer, or one per instance with --folder-per-instance."""
    if not args.folder_per_instance:
      return observer
    key = problem.id_triple
    if key not in observers:
      observers[key] = cocoex.Observer(SUITE_NAME, observer_options(
          args, instance_folder(output_folder, key)))
    return observers[key]

  print(f"Benchmarking {args.algorithm_name} on {SUITE_NAME} "
        f"[{suite_options or 'full suite'}], results in {observer.result_folder}")
  tic = time.perf_counter()
  # Runs per instance: an instance run more than once stopped early (e.g. on
  # the poll size), the others ran to their budget or target in one go.
  os.makedirs(observer.result_folder, exist_ok=True)
  runs_log = open(os.path.join(observer.result_folder, "runs.txt"), "a",
                  encoding="utf-8")
  for problem in suite:
    if problem.index % args.batches != args.batch - 1:
      continue
    if only is not None and problem.id_triple not in only:
      continue
    problem.observe_with(observer_of(problem))
    runs = solve(problem, args)
    runs_log.write("{} {} {} {}\n".format(*problem.id_triple, runs))
    runs_log.flush()
    minimal_print(problem)
  runs_log.close()
  print(f"\nDone in {time.perf_counter() - tic:.1f} s")

  if not args.no_postprocess and args.batches == 1:
    import cocopp
    cocopp.main(f"{observer.result_folder} {args.cocopp_args}".strip())


if __name__ == "__main__":
  main()
