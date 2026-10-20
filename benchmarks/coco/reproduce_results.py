"""Reproduce the OMADS ``bbob-constrained`` results of the documentation.

Runs every published configuration of ``run_mads_bbob_constrained.py``
(presets A, B, halton and initial) on dimensions 2, 3, 5 and 10, instances
1-15, with a budget of 100 x dimension, then post-processes the data with
cocopp against the archived reference algorithms and regenerates the
documentation figures and table:

  docs_src/media/figures/coco/coco_summary.svg   summary chart
  docs_src/media/figures/coco/coco_summary.csv   data table
  docs_src/media/figures/coco/ecdf_<D>D.svg      runtime ECDFs (D = 02, 03, 05, 10),
                                                 with ecdf_<D>D_dark.svg for dark themes
  docs_src/_static/ecdf_02D[_dark].svg           the 2D ECDF shown in the README

Usage, from the repository root:

  python -m pip install -e .
  python -m pip install -r benchmarks/coco/requirements.txt
  python benchmarks/coco/reproduce_results.py

Each preset is split into batches that run in parallel processes
(``--workers``, default: half the CPU cores, at most 12); a batch that fails,
for example when the machine runs out of memory, is run again once. The
COCO data go to ``exdata/<algorithm>/`` (one folder per instance) and the
full cocopp report to ``ppdata/repro/``, both under ``--workdir`` (default:
the repository root).

``--on-min-mesh search`` runs the presets with the search continuation of
``mads.main`` (algorithms ``<name>-search``; the cocopp report and the figures
then go to ``ppdata/repro-search/``, unless ``--figures`` is given). With ``--reuse
<exdata folder>`` of an earlier campaign of the same code, only the instances
that it had to restart (they stopped before their budget, e.g. on the poll
size) are run again; the trajectories of the other instances do not depend on
``--on-min-mesh``, so their data are copied.
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
PLOT_ECDF = os.path.join(HERE, "plot_ecdf.py")
DOCS_FIGURES = os.path.join(ROOT, "docs_src", "media", "figures", "coco")
README_FIGURES = os.path.join(ROOT, "docs_src", "_static")

# preset -> algorithm name (folder under exdata/)
PRESETS = {"A": "OMADS-A", "B": "OMADS-B", "halton": "OMADS-Halton",
           "initial": "OMADS-ACTIVE"}
# Configurations drawn in the runtime ECDFs (the summary uses all of them).
ECDF_ALGORITHMS = ["OMADS-A", "OMADS-B", "OMADS-Halton"]
ECDF_DIMENSIONS = ["02", "03", "05", "10"]
INSTANCE_FOLDER = re.compile(r"f(\d+)_d(\d+)_i(\d+)")


def algorithm_name(preset: str, args) -> str:
  return PRESETS[preset] + ("-search" if args.on_min_mesh == "search" else "")


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
  parser.add_argument("--figures", default=None,
                      help="where to write the documentation figures (default:"
                      " docs_src/media/figures/coco, or ppdata/repro-search/figures"
                      " with --on-min-mesh search)")
  parser.add_argument("--on-min-mesh", default="stop", choices=["stop", "search"],
                      help="passed to the runner (default: stop)")
  parser.add_argument("--reuse", default=None,
                      help="exdata folder of an earlier campaign with the"
                      " same code: rerun only the instances it restarted and"
                      " copy the others")
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
  args.ppdata = os.path.join(args.workdir, "ppdata", "repro" + (
      "-search" if args.on_min_mesh == "search" else ""))
  if args.figures is None:
    args.figures = DOCS_FIGURES if args.on_min_mesh == "stop" else \
        os.path.join(args.ppdata, "figures")
  if args.reuse:
    args.reuse = os.path.abspath(args.reuse)
  return args


def run_batch(preset: str, algorithm: str, batch: int, args,
              only: str = None) -> bool:
  """Run one batch of a preset (of the instances in ``only``, when given);
  retry once if it fails."""
  folder = f"{algorithm}_b{batch}"
  log = os.path.join(args.exdata, f"{folder}.log")
  cmd = [sys.executable, "-u", RUNNER, "--preset", preset,
         "--batches", str(args.batches), "--batch", str(batch),
         "--output-folder", folder, "--algorithm-name", algorithm,
         "--on-min-mesh", args.on_min_mesh, "--folder-per-instance",
         "--no-postprocess"]
  if only:
    cmd += ["--only", only]
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


def read_runs(folder: str) -> dict:
  """{(function, dimension, instance): runs} from the runs.txt of a campaign."""
  runs = {}
  for log in glob.glob(os.path.join(folder, "**", "runs.txt"), recursive=True):
    with open(log, encoding="utf-8") as f:
      for line in f:
        if line.strip():
          fun, dim, inst, n = (int(v) for v in line.split())
          runs[(fun, dim, inst)] = n
  return runs


def plan_reuse(preset: str, args):
  """Instances of ``preset`` to rerun, written to a file for the runner."""
  base = os.path.join(args.reuse, PRESETS[preset])
  runs = read_runs(base)
  if not runs:
    raise SystemExit(f"no runs.txt found under {base}: run that campaign with"
                     " this version of the scripts first")
  rerun = sorted(key for key, n in runs.items() if n > 1)
  only = os.path.join(args.exdata, f"{algorithm_name(preset, args)}_rerun.txt")
  with open(only, "w", encoding="utf-8") as f:
    f.writelines("{} {} {}\n".format(*key) for key in rerun)
  print(f"  {PRESETS[preset]}: {len(rerun)} of {len(runs)} instances to rerun",
        flush=True)
  return only, set(rerun), runs


def copy_reused(preset: str, rerun: set, runs: dict, args):
  """Copy the instances of the earlier campaign that are not rerun."""
  base = os.path.join(args.reuse, PRESETS[preset])
  target = os.path.join(args.exdata, algorithm_name(preset, args), "reused")
  os.makedirs(target, exist_ok=True)
  kept = []
  for folder in glob.glob(os.path.join(base, "*", "f*_d*_i*")):
    m = INSTANCE_FOLDER.match(os.path.basename(folder))
    key = tuple(int(v) for v in m.groups()) if m else None
    if key in runs and key not in rerun:
      shutil.copytree(folder, os.path.join(target, os.path.basename(folder)))
      kept.append(key)
  with open(os.path.join(target, "runs.txt"), "w", encoding="utf-8") as f:
    f.writelines("{} {} {} {}\n".format(*key, runs[key]) for key in sorted(kept))


def main():  # noqa: C901
  args = parse_args()
  presets = [p.strip() for p in args.presets.split(",") if p.strip()]
  unknown = [p for p in presets if p not in PRESETS]
  if unknown:
    raise SystemExit(f"unknown presets {unknown}; choose from {sorted(PRESETS)}")
  os.makedirs(args.exdata, exist_ok=True)
  tic = time.perf_counter()

  if not args.skip_runs:
    reuse = {p: plan_reuse(p, args) for p in presets} if args.reuse else {}
    jobs = [(p, algorithm_name(p, args), k, reuse[p][0] if p in reuse else None)
            for p in presets for k in range(1, args.batches + 1)
            if p not in reuse or reuse[p][1]]
    print(f"Running {len(jobs)} batches ({', '.join(presets)}) on "
          f"{args.workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
      results = list(pool.map(lambda job: run_batch(*job[:3], args, job[3]), jobs))
    failed = [f"{a}_b{k}" for (_, a, k, _), ok in zip(jobs, results) if not ok]
    if failed:
      raise SystemExit(f"batches failed twice: {failed}")
    for p in presets:
      merge_batches(algorithm_name(p, args), args)
      if p in reuse:
        copy_reused(p, reuse[p][1], reuse[p][2], args)
    print(f"Benchmark runs done in {(time.perf_counter() - tic) / 3600:.1f} h",
          flush=True)

  print("Post-processing with cocopp...", flush=True)
  shutil.rmtree(args.ppdata, ignore_errors=True)
  suffix = "-search" if args.on_min_mesh == "search" else ""
  algorithms = [a + suffix for a in ECDF_ALGORITHMS
                if os.path.isdir(os.path.join(args.exdata, a + suffix))]
  status = subprocess.run([sys.executable, POSTPROCESS, args.ppdata] + algorithms,
                          cwd=args.exdata, check=False).returncode
  ecdf_data = os.path.join(args.ppdata, "ecdf_data")
  if not all(os.path.isfile(os.path.join(ecdf_data, f"ecdf_{d}D.json"))
             for d in ECDF_DIMENSIONS):
    raise SystemExit(f"cocopp did not write the runtime ECDFs to {args.ppdata}")
  if status != 0:
    # cocopp can fail in later report sections (e.g. on very small data sets)
    # after the aggregated ECDFs used here are written.
    print(f"  cocopp exited with status {status} after writing the runtime"
          " ECDFs; continuing", flush=True)
  subprocess.run([sys.executable, PLOT_ECDF, ecdf_data, args.figures] + ECDF_DIMENSIONS,
                 check=True)
  if args.on_min_mesh == "search":
    print(f"Done in {(time.perf_counter() - tic) / 3600:.1f} h; figures in "
          f"{args.figures}", flush=True)
    return
  if os.path.abspath(args.figures) == DOCS_FIGURES:   # the README figure
    os.makedirs(README_FIGURES, exist_ok=True)
    for name in ("ecdf_02D.svg", "ecdf_02D_dark.svg"):
      shutil.copy(os.path.join(args.figures, name), README_FIGURES)

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
