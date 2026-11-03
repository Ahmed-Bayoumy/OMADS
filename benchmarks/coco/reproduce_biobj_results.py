"""Benchmark OMADS' multiobjective mode on COCO ``bbob-biobj`` and draw the figures.

Runs ``run_omads_bbob_biobj.py`` on dimensions 2, 3, 5 and 10, the 55
functions and instances 1-15, with a budget of 1000 x dimension, split into
batches that run in parallel processes; then post-processes the data with
cocopp against archived reference algorithms and draws, under ``--figures``:

  ecdf_<D>D.png, ecdf_grid.png      runtime ECDFs of the hypervolume indicator
  convergence_<D>D.png              hypervolume indicator vs. evaluations, per function
  fronts_<D>D.png                   Pareto fronts found by OMADS on selected functions

Usage, from the repository root:

  python -m pip install -e .
  python -m pip install -r benchmarks/coco/requirements.txt
  python benchmarks/coco/reproduce_biobj_results.py
  python benchmarks/coco/reproduce_biobj_results.py -d 2,3 -i 1-5   # reduced
  python benchmarks/coco/reproduce_biobj_results.py --skip-runs     # figures only
  python benchmarks/coco/reproduce_biobj_results.py --solver mads --ns 150 --compare OMADS-MO

The COCO data go to ``exdata/<algorithm>/`` and the cocopp report to
``ppdata/biobj/``, both under ``--workdir`` (default: the repository root).
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNNER = os.path.join(HERE, "run_omads_bbob_biobj.py")
POSTPROCESS = os.path.join(HERE, "postprocess_bbob_biobj.py")
PLOT = os.path.join(HERE, "plot_biobj.py")
PLOT_FRONTS = os.path.join(HERE, "plot_biobj_fronts.py")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--solver", default="poll", choices=["poll", "mads"],
                      help="passed to the runner: poll.main (default) or mads.main")
  parser.add_argument("--s-method", default="ACTIVE",
                      help="passed to the runner: mads sampling search (default: ACTIVE)")
  parser.add_argument("--ns", default="150",
                      help="passed to the runner: mads samples per search step"
                      " (default: 150)")
  parser.add_argument("--algorithm-name", default=None,
                      help="folder under exdata/ and label (default: OMADS-MO for"
                      " poll, OMADS-MADS-ns<ns> for mads with ACTIVE,"
                      " OMADS-MADS-<METHOD>-ns<ns> otherwise)")
  parser.add_argument("--compare", default="",
                      help="comma-separated folders of earlier OMADS campaigns under"
                      " exdata/ to include in the report and figures, e.g. OMADS-MO")
  parser.add_argument("--budget-multiplier", "-b", default="1000",
                      help="passed to the runner (default: 1000)")
  parser.add_argument("--dimensions", "-d", default="2,3,5,10",
                      help="passed to the runner (default: 2,3,5,10)")
  parser.add_argument("--functions", "-f", default=None,
                      help="passed to the runner (default: all 55)")
  parser.add_argument("--instances", "-i", default=None,
                      help="passed to the runner (default: its 1-15)")
  parser.add_argument("--batches", type=int, default=24,
                      help="batches (default: 24)")
  parser.add_argument("--workers", type=int,
                      default=min(max((os.cpu_count() or 2) // 2, 1), 12),
                      help="parallel processes (default: half the CPU cores,"
                      " at most 12)")
  parser.add_argument("--workdir", default=ROOT,
                      help="folder that receives exdata/ and ppdata/biobj/;"
                      " keep its path short (Windows path limit in cocopp)")
  parser.add_argument("--figures", default=None,
                      help="where to write the PNG figures (default:"
                      " ppdata/biobj/figures)")
  parser.add_argument("--skip-runs", action="store_true",
                      help="reuse exdata/<algorithm>/ and only post-process")
  parser.add_argument("--skip-postprocess", action="store_true",
                      help="only run the benchmark")
  args = parser.parse_args()
  if args.algorithm_name is None:
    method = "" if args.s_method == "ACTIVE" else f"{args.s_method}-"
    args.algorithm_name = ("OMADS-MO" if args.solver == "poll"
                           else f"OMADS-MADS-{method}ns{args.ns}")
  args.compare = [c for c in args.compare.split(",") if c]
  args.workdir = os.path.abspath(args.workdir)
  args.exdata = os.path.join(args.workdir, "exdata")
  args.ppdata = os.path.join(args.workdir, "ppdata", "biobj")
  args.figures = os.path.abspath(args.figures or os.path.join(args.ppdata, "figures"))
  return args


def run_batch(batch: int, args) -> bool:
  """Run one batch; retry once if it fails."""
  folder = f"{args.algorithm_name}_b{batch}"
  log = os.path.join(args.exdata, f"{folder}.log")
  cmd = [sys.executable, "-u", RUNNER, "--batches", str(args.batches),
         "--batch", str(batch), "--output-folder", folder,
         "--algorithm-name", args.algorithm_name, "-b", args.budget_multiplier,
         "-d", args.dimensions, "--solver", args.solver, "--s-method", args.s_method,
         "--ns", args.ns]
  for flag, value in (("-f", args.functions), ("-i", args.instances)):
    if value:
      cmd += [flag, value]
  env = dict(os.environ, PYTHONIOENCODING="utf-8")
  for attempt in (1, 2):
    shutil.rmtree(os.path.join(args.exdata, folder), ignore_errors=True)
    with open(log, "w", encoding="utf-8") as out:
      done = subprocess.run(cmd, cwd=args.workdir, stdout=out, stderr=subprocess.STDOUT,
                            env=env, check=False).returncode == 0
    if done:
      return True
    print(f"  {folder}: attempt {attempt} failed, see {log}", flush=True)
  return False


def merge_batches(args):
  """Gather exdata/<algorithm>_b<k> folders into exdata/<algorithm>/."""
  target = os.path.join(args.exdata, args.algorithm_name)
  shutil.rmtree(target, ignore_errors=True)
  os.makedirs(target)
  for folder in sorted(glob.glob(os.path.join(args.exdata, f"{args.algorithm_name}_b*"))):
    if os.path.isdir(folder):
      shutil.move(folder, target)


def step(cmd, cwd):
  print("$ " + " ".join(os.path.basename(c) if c.endswith(".py") else c for c in cmd),
        flush=True)
  subprocess.run(cmd, cwd=cwd, check=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))


def main():
  args = parse_args()
  os.makedirs(args.exdata, exist_ok=True)
  if not args.skip_runs:
    tic = time.perf_counter()
    print(f"Running {args.batches} batches on {args.workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
      results = list(pool.map(lambda k: run_batch(k, args), range(1, args.batches + 1)))
    failed = [k + 1 for k, ok in enumerate(results) if not ok]
    if failed:
      raise SystemExit(f"batches failed twice: {failed}")
    merge_batches(args)
    print(f"Runs done in {(time.perf_counter() - tic) / 3600:.2f} h", flush=True)
  if args.skip_postprocess:
    return

  # cocopp takes the labels from the folder names, so run it from exdata/.
  shutil.rmtree(args.ppdata, ignore_errors=True)
  algorithms = [args.algorithm_name] + args.compare
  step([sys.executable, POSTPROCESS, args.ppdata] + algorithms, args.exdata)
  step([sys.executable, PLOT, "--ppdata", args.ppdata,
        "--exdata"] + [os.path.join(args.exdata, a) for a in algorithms] +
       ["-d", args.dimensions, "--budget", args.budget_multiplier,
        "--out", args.figures], args.workdir)
  solvers = [args.solver] + [("poll" if c == "OMADS-MO" else "mads") for c in args.compare]
  # the fronts script draws one mads variant: the one of this campaign
  step([sys.executable, PLOT_FRONTS, "-b", args.budget_multiplier,
        "--solvers", ",".join(dict.fromkeys(solvers)), "--ns", args.ns,
        "--s-method", args.s_method, "--out", args.figures], args.workdir)
  print(f"cocopp report: {args.ppdata}\nfigures: {args.figures}")


if __name__ == "__main__":
  main()
