"""Benchmark OMADS on a single bbob-constrained problem.

The default problem is f1 in 10-D: the sphere with one linear constraint, whose
optimum lies on that constraint. It is the simplest function of the suite, yet
with the published configuration (preset A, 100 n evaluations) OMADS solved
none of its 15 instances to COCO's final target (median f - f_opt = 0.59 in
the 2610 baseline), while COBYLA solved all 15 within the same budget. OMADS
does better on the same sphere with more constraints (f4-f6), which makes f1 a
good case for studying the progress of MADS along a single active constraint.

The problem is formulated and OMADS configured exactly as in
``run_mads_bbob_constrained.py`` (same blackbox wrapper, presets, restarts and
options; every option of that runner is accepted here and overrides the
preset). Each instance is recorded by COCO in its own folder, so that the
results can be post-processed with ``cocopp`` next to the full campaigns, and a
summary of the instances is printed and saved to ``summary.json``.

Usage, from the repository root:

  python benchmarks/coco/run_single_problem.py                  # f1, 10-D, instances 1-15
  python benchmarks/coco/run_single_problem.py -i 1-3 -b 300    # 3 instances, budget 300 n
  python benchmarks/coco/run_single_problem.py --direction-type ORTHO_2N --on-min-mesh search
  python benchmarks/coco/run_single_problem.py -f 25 -d 5       # another problem
  python benchmarks/coco/run_single_problem.py --plot           # + objective trajectory
  python benchmarks/coco/run_single_problem.py --show-input     # print the OMADS input

COCO writes the data to ``exdata/<output folder>/`` under the working
directory (default output folder: ``<algorithm>_f<F>_d<D>_<budget>x``).
"""

import argparse
import json
import os
import re
import sys
import time

import cocoex
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_mads_bbob_constrained import (  # noqa: E402
    SUITE_NAME, apply_preset, build_omads_input, build_parser, instance_folder,
    observer_options, solve)

FINAL_TARGET = 1e-8
# The 41 targets of the runtime ECDFs (10^2 to 10^-6), to report the same
# "(function, target) pairs solved" measure as the documentation.
TARGETS = [10 ** (2 - 0.2 * k) for k in range(41)]
INFO_RUN = re.compile(r"(\d+):(\d+)\|([-+0-9.eE]+|inf|nan)")


def parse_args():
  parser = build_parser(description=__doc__.splitlines()[0])
  parser.set_defaults(functions="1", dimensions="10")
  parser.add_argument("--plot", action="store_true",
                      help="plot the objective trajectory of the runs (5th-95th"
                      " percentile band, mean and median) to"
                      " exdata/<output folder>/trajectory_f<F>_d<D>.svg")
  parser.add_argument("--show-input", action="store_true",
                      help="print the OMADS input of the first instance (the"
                      " blackbox is a COCO callable returning [f, g]) and exit")
  return apply_preset(parser.parse_args())


def final_precision(folder: str) -> float:
  """Best f - f_opt recorded by COCO in the .info file(s) of an instance."""
  best = np.inf
  for root, _, files in os.walk(folder):
    for name in files:
      if name.endswith(".info"):
        with open(os.path.join(root, name), encoding="utf-8") as f:
          for line in f:
            if line.startswith("data_"):
              best = min([best] + [float(df) for _, _, df in INFO_RUN.findall(line)])
  return best


def show_input(problem, args):
  data = build_omads_input(problem, problem.initial_solution,
                           int(args.budget_multiplier * problem.dimension),
                           args.seed, args)
  data["evaluator"]["blackbox"] = (
      f"<COCO {problem.id}: x -> [f(x), g(x)], {problem.number_of_constraints}"
      " constraints g(x) <= 0>")
  print(f"{problem.name}\n")
  print(json.dumps(data, indent=2, default=float))


def main():
  args = parse_args()
  args.omads_post_dir = None
  suite = cocoex.Suite(SUITE_NAME, "", " ".join(
      f"{key}: {val}" for key, val in (
          ("dimensions", args.dimensions), ("function_indices", args.functions),
          ("instance_indices", args.instances)) if val))
  if args.show_input:
    show_input(next(iter(suite)), args)
    return

  output_folder = args.output_folder or "{}_f{}_d{}_{}x".format(
      args.algorithm_name, args.functions.replace(",", "-"),
      args.dimensions.replace(",", "-"), int(args.budget_multiplier))
  # COCO creates the instance folders but not their parent.
  os.makedirs(os.path.join("exdata", output_folder), exist_ok=True)
  rows = []
  print(f"{args.algorithm_name} (preset {args.preset}) on {SUITE_NAME}, f{args.functions}"
        f" in {args.dimensions}-D, instances {args.instances}, budget"
        f" {args.budget_multiplier:g} n")
  print(f"{'problem':32} {'constr.':>7} {'runs':>4} {'f-evals':>7} {'g-evals':>7}"
        f" {'f - f_opt':>10} {'targets':>7} {'time':>7}")
  for problem in suite:
    observer = cocoex.Observer(SUITE_NAME, observer_options(
        args, instance_folder(output_folder, problem.id_triple)))
    args.omads_post_dir = os.path.join(observer.result_folder, "omads_post")
    problem.observe_with(observer)
    tic = time.perf_counter()
    runs = solve(problem, args)
    row = {"problem": problem.id, "function": problem.id_function,
           "dimension": problem.dimension, "instance": problem.id_instance,
           "constraints": problem.number_of_constraints, "runs": runs,
           "f_evaluations": int(problem.evaluations),
           "g_evaluations": int(problem.evaluations_constraints),
           "seconds": round(time.perf_counter() - tic, 2)}
    problem.free()   # COCO completes the instance's files on free
    row["precision"] = final_precision(observer.result_folder)
    row["targets_reached"] = sum(row["precision"] <= t for t in TARGETS) / len(TARGETS)
    rows.append(row)
    print(f"{row['problem']:32} {row['constraints']:7d} {runs:4d}"
          f" {row['f_evaluations']:7d} {row['g_evaluations']:7d}"
          f" {row['precision']:10.2e} {row['targets_reached']:7.0%}"
          f" {row['seconds']:6.1f}s")

  precisions = np.array([r["precision"] for r in rows])
  summary = {
      "configuration": {k: v for k, v in vars(args).items()
                        if isinstance(v, (str, int, float, bool, type(None)))},
      "instances": rows,
      "solved": int(np.sum(precisions <= FINAL_TARGET)),
      "median_precision": float(np.median(precisions)),
      "targets_reached": float(np.mean([r["targets_reached"] for r in rows])),
  }
  print(f"\nsolved to {FINAL_TARGET:g}: {summary['solved']} of {len(rows)};"
        f" median f - f_opt = {summary['median_precision']:.2e};"
        f" (function, target) pairs reached: {summary['targets_reached']:.1%}")
  path = os.path.join("exdata", output_folder, "summary.json")
  with open(path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1)
  print(f"COCO data and summary.json in exdata/{output_folder}"
        f" (post-process with: python -m cocopp exdata/{output_folder})")
  if args.plot:
    import plot_trajectory   # needs cocopp
    for func in sorted({r["function"] for r in rows}):
      for dim in sorted({r["dimension"] for r in rows}):
        plot_trajectory.main(argparse.Namespace(
            series=[[None, os.path.join("exdata", output_folder)]],
            function=func, dimension=dim, output=None, log_x=False))


if __name__ == "__main__":
  main()
