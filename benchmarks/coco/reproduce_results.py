"""Reproduce the OMADS ``bbob-constrained`` results of the documentation.

Runs every published configuration of ``run_mads_bbob_constrained.py``
(presets A, B, halton and initial) on dimensions 2, 3, 5 and 10, instances
1-15, with a budget of 100 x dimension, then post-processes the data with
cocopp against the archived reference algorithms and regenerates the
documentation figures and table:

  docs_src/media/figures/coco/coco_summary.svg   summary chart
  docs_src/media/figures/coco/coco_summary.csv   data table
  docs_src/media/figures/coco/ecdf_<D>D.svg      runtime ECDFs (D = 02, 03, 05, 10)

Usage, from the repository root:

  python -m pip install -e .
  python -m pip install -r benchmarks/coco/requirements.txt
  python benchmarks/coco/reproduce_results.py

Each preset is split into batches that run in parallel processes
(``--workers``, default: half the CPU cores, at most 12); a batch that fails,
for example when the machine runs out of memory, is run again once. The
COCO data go to ``exdata/<algorithm>/`` and the full cocopp report to
``ppdata/repro/``, both under ``--workdir`` (default: the repository root).
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNNER = os.path.join(HERE, "run_mads_bbob_constrained.py")
POSTPROCESS = os.path.join(HERE, "postprocess_bbob_constrained.py")
SUMMARIZE = os.path.join(HERE, "summarize_bbob_constrained.py")

# preset -> algorithm name (folder under exdata/)
PRESETS = {"A": "OMADS-A", "B": "OMADS-B", "halton": "OMADS-Halton",
           "initial": "OMADS-ACTIVE"}
# Configurations drawn in the runtime ECDFs (the summary uses all of them).
ECDF_ALGORITHMS = ["OMADS-A", "OMADS-B", "OMADS-Halton"]
ECDF_DIMENSIONS = ["02", "03", "05", "10"]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--presets", default="A,B,halton,initial",
                      help="comma-separated presets to run (default: all)")
  parser.add_argument("--batches", type=int, default=12,
                      help="batches per preset (default: 12)")
  parser.add_argument("--workers", type=int,
                      default=min(max((os.cpu_count() or 2) // 2, 1), 12),
                      help="parallel processes (default: half the CPU cores,"
                      " at most 12)")
  parser.add_argument("--workdir", default=ROOT,
                      help="folder where the runs write exdata/ (COCO always"
                      " uses exdata/ under the working directory) and the"
                      " cocopp report ppdata/repro/; keep its path short,"
                      " cocopp exceeds the Windows path limit when it is"
                      " deeply nested (default: the repository root)")
  parser.add_argument("--figures", default=os.path.join(
      ROOT, "docs_src", "media", "figures", "coco"),
      help="where to write the documentation figures")
  parser.add_argument("--dimensions", "-d", default=None,
                      help="passed to the runner (default: its 2,3,5,10)")
  parser.add_argument("--functions", "-f", default=None,
                      help="passed to the runner (default: all 54)")
  parser.add_argument("--instances", "-i", default=None,
                      help="passed to the runner (default: its 1-15)")
  parser.add_argument("--skip-runs", action="store_true",
                      help="reuse the existing exdata/<algorithm>/ folders and"
                      " only post-process")
  args = parser.parse_args()
  args.workdir = os.path.abspath(args.workdir)
  args.exdata = os.path.join(args.workdir, "exdata")
  args.ppdata = os.path.join(args.workdir, "ppdata", "repro")
  return args


def run_batch(preset: str, algorithm: str, batch: int, args) -> bool:
  """Run one batch of a preset; retry once if it fails."""
  folder = f"{algorithm}_b{batch}"
  log = os.path.join(args.exdata, f"{folder}.log")
  cmd = [sys.executable, "-u", RUNNER, "--preset", preset,
         "--batches", str(args.batches), "--batch", str(batch),
         "--output-folder", folder, "--no-postprocess"]
  for flag, value in (("-d", args.dimensions), ("-f", args.functions),
                      ("-i", args.instances)):
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


def merge_batches(algorithm: str, args):
  """Gather exdata/<algorithm>_b<k> folders into exdata/<algorithm>/."""
  target = os.path.join(args.exdata, algorithm)
  shutil.rmtree(target, ignore_errors=True)
  os.makedirs(target)
  for folder in sorted(glob.glob(os.path.join(args.exdata, f"{algorithm}_b*"))):
    if os.path.isdir(folder):
      shutil.move(folder, target)


def crop_cocopp_svg(src: str, dst: str, width: float = 560., height: float = 352.):
  """Copy a cocopp runtime ECDF, cropped to the plot and its legend.

  cocopp places an invisible element far to the left of its figures, so the
  saved SVGs are tens of thousands of points wide with the plot at the right
  edge; keep only the rightmost ``width`` x ``height`` points.
  """
  with open(src, encoding="utf-8") as f:
    svg = f.read()
  m = re.search(r'width="([0-9.]+)pt" height="([0-9.]+)pt" '
                r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg)
  if m:
    x0 = max(float(m.group(3)) - width, 0.)
    svg = svg.replace(m.group(0), f'width="{width}pt" height="{height}pt" '
                      f'viewBox="{x0:.3f} 0 {width:.3f} {height:.3f}"', 1)
  with open(dst, "w", encoding="utf-8") as f:
    f.write(svg)


def main():
  args = parse_args()
  presets = [p.strip() for p in args.presets.split(",") if p.strip()]
  unknown = [p for p in presets if p not in PRESETS]
  if unknown:
    raise SystemExit(f"unknown presets {unknown}; choose from {sorted(PRESETS)}")
  os.makedirs(args.exdata, exist_ok=True)
  tic = time.perf_counter()

  if not args.skip_runs:
    jobs = [(p, PRESETS[p], k) for p in presets for k in range(1, args.batches + 1)]
    print(f"Running {len(jobs)} batches ({', '.join(presets)}) on "
          f"{args.workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
      results = list(pool.map(lambda job: run_batch(*job, args), jobs))
    failed = [f"{a}_b{k}" for (_, a, k), ok in zip(jobs, results) if not ok]
    if failed:
      raise SystemExit(f"batches failed twice: {failed}")
    for p in presets:
      merge_batches(PRESETS[p], args)
    print(f"Benchmark runs done in {(time.perf_counter() - tic) / 3600:.1f} h",
          flush=True)

  print("Post-processing with cocopp...", flush=True)
  shutil.rmtree(args.ppdata, ignore_errors=True)
  algorithms = [a for a in ECDF_ALGORITHMS
                if os.path.isdir(os.path.join(args.exdata, a))]
  status = subprocess.run([sys.executable, POSTPROCESS, args.ppdata] + algorithms,
                          cwd=args.exdata, check=False).returncode
  report = glob.glob(os.path.join(args.ppdata, "*", ""))
  report = [r for r in report if all(os.path.isfile(
      os.path.join(r, f"pprldmany_{d}D_noiselessall.svg")) for d in ECDF_DIMENSIONS)]
  if not report:
    raise SystemExit(f"cocopp did not write the runtime ECDFs to {args.ppdata}")
  if status != 0:
    # cocopp can fail in later report sections (e.g. on very small data sets)
    # after the aggregated ECDFs used here are written.
    print(f"  cocopp exited with status {status} after writing the runtime"
          " ECDFs; continuing", flush=True)
  os.makedirs(args.figures, exist_ok=True)
  for d in ECDF_DIMENSIONS:
    crop_cocopp_svg(os.path.join(report[0], f"pprldmany_{d}D_noiselessall.svg"),
                    os.path.join(args.figures, f"ecdf_{d}D.svg"))

  print("Writing the summary table and chart...", flush=True)
  subprocess.run([sys.executable, SUMMARIZE, args.figures, args.exdata],
                 check=True)
  png = os.path.join(args.figures, "coco_summary.png")
  if os.path.exists(png):
    os.remove(png)
  print(f"Done in {(time.perf_counter() - tic) / 3600:.1f} h; figures in "
        f"{args.figures}", flush=True)


if __name__ == "__main__":
  main()
