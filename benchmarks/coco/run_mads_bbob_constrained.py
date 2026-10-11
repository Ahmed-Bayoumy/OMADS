"""Benchmark OMADS' MADS on the COCO ``bbob-constrained`` suite.

Follows the COCO getting-started recipe (https://coco-platform.org/getting-started/):

  python -m pip install coco-experiment cocopp
  python benchmarks/coco/run_mads_bbob_constrained.py --budget-multiplier 100
  python benchmarks/coco/run_mads_bbob_constrained.py -d 2,3,5 -f 1-12 -i 1-3 \
      --budget-multiplier 50

Each COCO problem ``x -> f(x)`` with constraints ``g(x) <= 0`` is handed to
``OMADS.mads.main`` as a callable blackbox returning ``[f, g]``, with every
constraint treated by the progressive barrier (PB). Runs stop early once the
final COCO target is hit, and ``cocoex.ExperimentRepeater`` restarts unsolved
instances (from different initial points and seeds) until the
``budget_multiplier * dimension`` budget per instance is used up.

Results go to ``exdata/<output-folder>`` and are post-processed with
``cocopp`` (HTML report in ``ppdata/``) unless ``--no-postprocess`` is given.
Use ``--batches N --batch k`` to split the suite over N parallel processes,
then post-process all batch folders together: ``python -m cocopp exdata/...``.
"""

import argparse
import contextlib
import os
import sys
import time
from typing import Dict, Tuple

import cocoex
import numpy as np

from OMADS import mads

SUITE_NAME = "bbob-constrained"
# bbob-constrained combines 9 raw functions with 6 constraint counts each;
# raw functions 8 and 9 (separable and rotated Rastrigin) are multimodal.
MULTIMODAL_FUNCTIONS = range(43, 55)


def search_samples(problem, args: argparse.Namespace) -> int:
  """Samples per sampling-search step: a large sample only pays off on the
  multimodal functions, elsewhere it eats most of the budget."""
  if problem.id_function in MULTIMODAL_FUNCTIONS:
    return args.ns_multimodal
  return args.ns


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
          "opportunistic": False, "save_results": False, "is_verbose": False,
          "precision": "high", "anisotropy_factor": args.anisotropy_factor,
      },
      "search": {
          "type": args.search_type,
          "s_method": args.s_method,
          "ns": search_samples(problem, args),
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
      mads.main(build_omads_input(problem, x0, budget, seed, args))
  except TargetHit:
    pass
  finally:
    # mads.main wraps stderr and only restores it on normal return.
    sys.stderr = stderr


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Benchmark OMADS MADS on COCO's bbob-constrained suite.")
  parser.add_argument("--budget-multiplier", "-b", type=float, default=100,
                      help="f-evaluation budget per instance = this x dimension"
                      " (default: 100)")
  parser.add_argument("--dimensions", "-d", default="",
                      help="e.g. '2,3,5,10' (suite has 2,3,5,10,20,40; "
                      "default: all)")
  parser.add_argument("--functions", "-f", default="",
                      help="function indices, e.g. '1-18' (1-54; default: all)")
  parser.add_argument("--instances", "-i", default="",
                      help="instance indices, e.g. '1-15' (default: suite's)")
  parser.add_argument("--batches", type=int, default=1,
                      help="split the suite into this many batches")
  parser.add_argument("--batch", type=int, default=1,
                      help="which batch (1..batches) this process runs")
  parser.add_argument("--max-sweeps", type=int, default=None,
                      help="max restarts per instance (COCO default: 20)")
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
                      help="GMESH (default) is NOMAD's anisotropic granular"
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
  parser.add_argument("--s-method", default="HALTON",
                      help="sampling method of the search step (default:"
                      " HALTON; ACTIVE retrains pure-Python tree models on"
                      " the whole cache every search step and becomes very"
                      " slow as evaluations accumulate)")
  parser.add_argument("--ns", type=int, default=1,
                      help="samples per sampling-search step on unimodal"
                      " functions (default: 1; larger values spend most of"
                      " the budget on sampling)")
  parser.add_argument("--ns-multimodal", type=int, default=56,
                      help="samples per sampling-search step on the multimodal"
                      " Rastrigin functions f43-f54 (default: 56)")
  parser.add_argument("--algorithm-name", default="OMADS-MADS")
  parser.add_argument("--output-folder", default=None,
                      help="folder name under exdata/ (default: derived)")
  parser.add_argument("--no-postprocess", action="store_true",
                      help="skip running cocopp at the end")
  parser.add_argument("--cocopp-args", default="",
                      help="extra cocopp arguments, e.g. reference algorithms")
  return parser.parse_args()


def main():
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
  observer = cocoex.Observer(
      SUITE_NAME,
      f"result_folder: {output_folder} algorithm_name: {args.algorithm_name} "
      f"algorithm_info: \"OMADS MADS, PB constraints, {args.mesh_type} mesh, "
      f"search={args.search_type}/{args.s_method}, "
      f"ns={args.ns} (f43-f54: {args.ns_multimodal})\"")
  args.omads_post_dir = os.path.join(observer.result_folder, "omads_post")
  repeater = cocoex.ExperimentRepeater(args.budget_multiplier,
                                       max_sweeps=args.max_sweeps)
  minimal_print = cocoex.utilities.MiniPrint()
  used: Dict[Tuple[int, int, int], int] = {}

  print(f"Benchmarking {args.algorithm_name} on {SUITE_NAME} "
        f"[{suite_options or 'full suite'}], results in {observer.result_folder}")
  tic = time.perf_counter()
  while not repeater.done():
    for problem in suite:
      if problem.index % args.batches != args.batch - 1:
        continue
      if repeater.done(problem):
        continue
      problem.observe_with(observer)
      key = problem.id_triple
      remaining = int(repeater.budget_from_dimension(problem)) - used.get(key, 0)
      trial = sum(repeater.trials(problem))
      if remaining > 0:
        run_mads(problem, repeater.initial_solution_proposal(problem),
                 budget=remaining, seed=args.seed + trial, args=args)
      used[key] = used.get(key, 0) + problem.evaluations
      repeater.track(problem)
      minimal_print(problem)
  print(f"\nDone in {time.perf_counter() - tic:.1f} s")

  if not args.no_postprocess and args.batches == 1:
    import cocopp
    cocopp.main(f"{observer.result_folder} {args.cocopp_args}".strip())


if __name__ == "__main__":
  main()
