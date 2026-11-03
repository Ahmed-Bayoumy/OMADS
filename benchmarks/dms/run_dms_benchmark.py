"""Benchmark OMADS against NSGA-II on the 100 problems of the DMulti-MADS paper.

The problems are the bound-constrained multiobjective collection of Custodio
et al. (2011) used by Bigeon, Le Digabel and Salomon (DMulti-MADS, COAP 2021),
the method behind NOMAD 4's multiobjective solver; see ``problems.py``. The
protocol is theirs:

  budget        20,000 evaluations per run
  start         n points evenly spread on the diagonal of the box (the
                box centre when n = 1), the "line" initialization of DMS
  OMADS         stops at the budget or when every frame size is below 1e-9
                (no restart); 10 runs, seeds 1, 4734, 6652, ... (the paper's)
  NSGA-II       pymoo, population 100 for 200 generations (20,000
                evaluations); 50 runs with the paper's 50 seeds

Algorithm labels:

  OMADS                      poll.main (speculative, model and middle-point
                             searches, then the poll), the package defaults
  OMADS-MADS-HALTON-ns10     mads.main with a HALTON sampling search of 10
                             points per step
  OMADS-NOMS, ...-NOMS       without the multiobjective model and middle-point
                             searches (mo_search=False, DMulti-MADS as in 2021)
  NSGA-II

Each run's history (objectives of every evaluation, in order) is saved to
``<out>/<problem>/<algorithm>_s<seed>.npz``; ``postprocess_dms.py`` computes
the indicators and draws the profiles.

Usage, from the repository root:

  python benchmarks/dms/run_dms_benchmark.py
  python benchmarks/dms/run_dms_benchmark.py --problems ZDT1,DTLZ2 --runs 2

Finished runs are skipped when the script is run again (``--force`` reruns them).
"""
import os

# One BLAS thread per worker process: the model-search fits otherwise spawn a
# thread per core in every worker and the processes starve each other.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
  os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from problems import PROBLEMS, start_points  # noqa: E402

BUDGET = 20000
# Seeds of the DMulti-MADS experiments (scripts/generate_*_caches of
# DMultiMadsEB): the first 10 for the deterministic-with-random-directions
# solvers, all 50 for NSGA-II.
SEEDS = [1, 4734, 6652, 3507, 1121, 3500, 5816, 2006, 9622, 6117,
         1571, 4117, 3758, 8045, 6554, 8521, 4889, 5893, 9123, 7238,
         3089, 4197, 6489, 8, 7551, 7621, 4809, 8034, 3487, 3680,
         5363, 7786, 104, 8171, 7505, 7075, 9656, 4569, 3121, 4187,
         201, 951, 1093, 3147, 1978, 1475, 2896, 9814, 8803, 1084]
ALGORITHMS = ("OMADS", "OMADS-MADS-HALTON-ns10", "NSGA-II")
VARIANT = re.compile(
    r"^OMADS(?P<mads>-MADS(?:-(?P<method>ACTIVE|HALTON|LH|RS))?(?:-ns(?P<ns>\d+))?)?"
    r"(?P<noms>-NOMS)?$")


def omads_variant(algorithm):
  """Settings of an OMADS label, or None for another algorithm."""
  m = VARIANT.match(algorithm)
  if not m:
    return None
  mads = m.group("mads") is not None
  return dict(solver="mads" if mads else "poll",
              s_method=(m.group("method") or "ACTIVE") if mads else None,
              ns=int(m.group("ns") or 10) if mads else None,
              mo_search=not m.group("noms"))


class History:
  """Objectives of every evaluation, in order, capped at the budget."""

  def __init__(self, f, budget):
    self.f, self.budget, self.F = f, budget, []

  def __call__(self, x):
    y = [float(v) for v in self.f(np.asarray(x, dtype=float))]
    self.F.append(y)
    return y


def run_omads(name, seed, args, solver, s_method, ns, mo_search):
  from OMADS import mads, poll
  f, n, m, lb, ub = PROBLEMS[name]
  hist = History(f, args.budget)

  def blackbox(x):
    if len(hist.F) >= args.budget:
      return [[np.inf] * m, [0.0]]
    return [hist(x), [0.0]]

  with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    data = {
        "evaluator": {"blackbox": blackbox},
        "param": {
            "name": "dms", "baseline": start_points(name), "lb": list(lb), "ub": list(ub),
            "scaling": list(np.subtract(ub, lb)), "nobj": m, "is_pareto": True,
            "fun_names": [f"f{j + 1}" for j in range(m)], "post_dir": tmp,
        },
        "options": {
            "seed": seed, "budget": args.budget, "tol": args.tol, "check_cache": True,
            "store_cache": True, "mo_search": mo_search, "save_results": False,
            "display": False,
        },
    }
    if solver == "mads":
      data["search"] = {"type": "sampling", "s_method": s_method, "ns": ns, "batch": ns}
    stderr = sys.stderr
    try:
      with open(os.devnull, "w", encoding="utf-8") as devnull, \
              contextlib.redirect_stdout(devnull):
        (mads if solver == "mads" else poll).main(data)
    finally:
      sys.stderr = stderr   # the solvers wrap stderr
      root = logging.getLogger()   # OMADS keeps OMADS.log open; close it
      for handler in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
        root.removeHandler(handler)
        handler.close()
  return hist


def run_nsga2(name, seed, budget):
  from pymoo.algorithms.moo.nsga2 import NSGA2
  from pymoo.core.problem import Problem
  from pymoo.optimize import minimize
  f, n, m, lb, ub = PROBLEMS[name]
  hist = History(f, budget)

  class Wrapped(Problem):
    def __init__(self):
      super().__init__(n_var=n, n_obj=m, xl=np.array(lb), xu=np.array(ub))

    def _evaluate(self, X, out, *args, **kwargs):
      out["F"] = np.array([hist(x) for x in X])

  # Population 100 for 200 generations at the default budget.
  minimize(Wrapped(), NSGA2(pop_size=100), ("n_gen", max(budget // 100, 1)), seed=seed,
           verbose=False)
  return hist


def job(name, algorithm, seed, path, args):
  warnings.filterwarnings("ignore")
  np.seterr(all="ignore")
  tic = time.perf_counter()
  variant = omads_variant(algorithm)
  if variant:
    hist = run_omads(name, seed, args, **variant)
  elif algorithm == "NSGA-II":
    hist = run_nsga2(name, seed, args.budget)
  else:
    raise ValueError(f"unknown algorithm {algorithm}")
  _, n, m, _, _ = PROBLEMS[name]
  F = np.array(hist.F[:args.budget], dtype=float).reshape(-1, m)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  np.savez_compressed(path + ".tmp.npz", F=F, n=n, budget=args.budget,
                      seconds=time.perf_counter() - tic)
  os.replace(path + ".tmp.npz", path)
  return f"{name} {algorithm} s{seed}: {len(F)} evals, {time.perf_counter() - tic:.0f} s"


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--problems", default=",".join(PROBLEMS),
                      help="comma-separated names (default: all 100)")
  parser.add_argument("--algorithms", default=",".join(ALGORITHMS),
                      help="comma-separated labels (default: %(default)s)")
  parser.add_argument("--runs", type=int, default=10,
                      help="runs of each OMADS variant (default: 10, as in the paper)")
  parser.add_argument("--nsga-runs", type=int, default=50,
                      help="runs of NSGA-II (default: 50, as in the paper)")
  parser.add_argument("--budget", type=int, default=BUDGET,
                      help="evaluations per run (default: %(default)s, as in the paper)")
  parser.add_argument("--tol", type=float, default=1e-9,
                      help="OMADS minimum frame size (default: 1e-9)")
  parser.add_argument("--workers", type=int,
                      default=min(max((os.cpu_count() or 2) // 2, 1), 12),
                      help="parallel processes (default: half the CPU cores, at most 12)")
  parser.add_argument("--out", default=os.path.join(ROOT, "exdata", "dms"),
                      help="results folder (default: exdata/dms)")
  parser.add_argument("--force", action="store_true", help="rerun finished runs")
  args = parser.parse_args()
  lower = {p.lower(): p for p in PROBLEMS}
  names = [lower.get(p.strip().lower()) for p in args.problems.split(",")]
  if None in names:
    raise SystemExit(f"unknown problems in {args.problems}; choose from {list(PROBLEMS)}")
  jobs = []
  for name in names:
    for algorithm in args.algorithms.split(","):
      runs = args.nsga_runs if algorithm == "NSGA-II" else args.runs
      for seed in SEEDS[:runs]:
        path = os.path.join(args.out, name, f"{algorithm}_s{seed}.npz")
        if args.force or not os.path.exists(path):
          jobs.append((name, algorithm, seed, path))
  # Slowest first (many variables, OMADS), so the pool finishes evenly.
  jobs.sort(key=lambda j: (j[1] == "NSGA-II", -PROBLEMS[j[0]][1]))
  print(f"{len(jobs)} runs on {args.workers} workers, results in {args.out}", flush=True)
  tic = time.perf_counter()
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(job, *j, args): j for j in jobs}
    for k, future in enumerate(as_completed(futures), 1):
      try:
        print(f"[{k}/{len(jobs)}] {future.result()}", flush=True)
      except Exception as exc:   # report and carry on with the other runs
        print(f"[{k}/{len(jobs)}] FAILED {futures[future][:3]}: {exc!r}", flush=True)
  print(f"Done in {(time.perf_counter() - tic) / 60:.1f} min")
  shutil.rmtree(os.path.join(HERE, "__pycache__"), ignore_errors=True)


if __name__ == "__main__":
  main()
