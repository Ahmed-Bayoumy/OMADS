"""Benchmark OMADS against NSGA-II on constrained bi-objective test problems.

COCO has no constrained multiobjective suite, so this benchmark uses the
classical constrained bi-objective problems implemented in pymoo, all with a
known Pareto front:

  BNH, SRN, TNK, OSY                      engineering-style problems (2-6 variables)
  CTP1-CTP8                               Deb's constrained test problems (2 variables)
  MW1-MW13                                Ma and Wang's problems (14-15 variables)
  C1-DTLZ1, C1-DTLZ3, C2-DTLZ2, C3-DTLZ4  constrained DTLZ problems, 2 objectives

Each (problem, algorithm, seed) run gets ``max(min_budget, budget_multiplier
* n)`` evaluations and its whole history (objectives and constraints of every
evaluation) is saved to ``<out>/<problem>/<algorithm>_s<seed>.npz``;
``plot_cmo_benchmark.py`` computes the indicators and draws the figures.

  OMADS       ``poll.main`` (poll step only) with ``"nobj": 2, "is_pareto": True``,
              every constraint on the progressive barrier, a uniformly random
              starting point per seed, restarted from a new random point when it
              stops before its budget (all frame sizes below ``--tol``).
  OMADS-MADS[-<METHOD>][-ns<k>]
              the same with ``mads.main`` (search and poll steps) and the
              <METHOD> sampling search (default ACTIVE) with k samples per step
              (default ``--ns``, 150); e.g. OMADS-MADS (ACTIVE, ns=150),
              OMADS-MADS-HALTON, OMADS-MADS-ns10, OMADS-MADS-HALTON-ns10.
  ...-N1 or ...-2N, ...-OPP
              suffixes of any OMADS label: n+1 or 2n poll directions (without
              either, OMADS chooses: n+1 for several objectives since its
              DMulti-MADS update), and opportunistic evaluation (a poll or search
              set stops at its first success); e.g. OMADS-2N-OPP,
              OMADS-MADS-HALTON-ns5-N1.
  ...-TP, ...-NOMS
              further suffixes: the two-phase extreme barrier
              (``constraint_strategy="TWO_PHASE"``) instead of the progressive
              barrier, and no multiobjective model and middle-point search
              (``mo_search=False``, the speculative search alone).
  NSGA-II  pymoo's NSGA2 (population 100, SBX crossover and polynomial
           mutation, constrained domination), stopped at the same budget.

Usage, from the repository root:

  python -m pip install -r benchmarks/cmo/requirements.txt
  python benchmarks/cmo/run_cmo_benchmark.py
  python benchmarks/cmo/run_cmo_benchmark.py --problems ctp1,mw1 --seeds 3

Finished runs are skipped when the script is run again (``--force`` reruns them).
"""
import argparse
import contextlib
import re
import logging
import os
import shutil
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# pymoo name -> (displayed name, keyword arguments of get_problem)
PROBLEMS = {
    "bnh": ("BNH", {}), "srn": ("SRN", {}), "tnk": ("TNK", {}), "osy": ("OSY", {}),
    **{f"ctp{k}": (f"CTP{k}", {}) for k in range(1, 9)},
    **{f"mw{k}": (f"MW{k}", {"n_obj": 2} if k in (4, 8) else {}) for k in range(1, 14)},
    "c1dtlz1": ("C1-DTLZ1", {"n_obj": 2}), "c1dtlz3": ("C1-DTLZ3", {"n_obj": 2}),
    "c2dtlz2": ("C2-DTLZ2", {"n_obj": 2}), "c3dtlz4": ("C3-DTLZ4", {"n_obj": 2}),
}
ALGORITHMS = ("OMADS", "OMADS-MADS", "OMADS-MADS-HALTON", "NSGA-II")
VARIANT = re.compile(
    r"^OMADS(?P<mads>-MADS(?:-(?P<method>ACTIVE|HALTON|LH|RS|FULLFACTORIAL))?"
    r"(?:-ns(?P<ns>\d+))?)?(?P<dirs>-N1|-2N)?(?P<opp>-OPP)?(?P<tp>-TP)?"
    r"(?P<noms>-NOMS)?$")


def omads_variant(algorithm, default_ns):
  """Settings of an OMADS label (see the module docstring), or None for
  another algorithm: entry point, search method, samples per search step,
  poll directions, opportunistic evaluation, constraint strategy and
  multiobjective model search."""
  m = VARIANT.match(algorithm)
  if not m:
    return None
  mads = m.group("mads") is not None
  return dict(solver="mads" if mads else "poll",
              s_method=(m.group("method") or "ACTIVE") if mads else None,
              ns=int(m.group("ns") or default_ns) if mads else None,
              direction_type={"-N1": "N+1", "-2N": "ORTHO_2N"}.get(m.group("dirs")),
              opportunistic=bool(m.group("opp")),
              constraint_strategy="TWO_PHASE" if m.group("tp") else "PB",
              mo_search=not m.group("noms"))
NSGA2_POPULATION = 100


def get_problem(name):
  from pymoo.problems import get_problem as pymoo_problem
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return pymoo_problem(name, **PROBLEMS[name][1])


def budget_of(problem, args) -> int:
  return int(max(args.min_budget, args.budget_multiplier * problem.n_var))


class History:
  """Objectives and constraints of every evaluation of a pymoo problem."""

  def __init__(self, problem, budget):
    self.problem, self.budget = problem, budget
    self.F, self.G = [], []

  def evaluate(self, X):
    F, G = self.problem.evaluate(np.atleast_2d(X), return_values_of=["F", "G"])
    G = np.zeros((len(F), 0)) if G is None else np.atleast_2d(G)
    self.F.extend(F)
    self.G.extend(G)
    return F, G

  def arrays(self):
    m = self.problem.n_ieq_constr
    return (np.array(self.F[:self.budget]).reshape(-1, 2),
            np.array(self.G[:self.budget]).reshape(-1, m))


def run_omads(problem, budget, seed, args, solver="poll", s_method=None, ns=None,
              direction_type=None, opportunistic=False, constraint_strategy="PB",
              mo_search=True):
  from OMADS import mads, poll
  hist = History(problem, budget)
  xl, xu = problem.xl.astype(float), problem.xu.astype(float)

  def blackbox(x):
    if len(hist.F) >= budget:   # restarts never exceed the budget
      return [[np.inf, np.inf], [np.inf] * problem.n_ieq_constr]
    F, G = hist.evaluate(np.asarray(x, dtype=float))
    return [[float(v) for v in F[0]], [float(g) for g in G[0]]]

  rng = np.random.default_rng(seed)
  runs = 0
  with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    while len(hist.F) < budget and runs < args.max_restarts:
      data = {
          "evaluator": {"blackbox": blackbox},
          "param": {
              "name": "cmo", "baseline": list(rng.uniform(xl, xu)),
              "lb": list(xl), "ub": list(xu), "scaling": list(xu - xl),
              "nobj": 2, "is_pareto": True, "fun_names": ["f1", "f2"],
              "constraints_type": ["PB"] * problem.n_ieq_constr,
              "post_dir": tmp,
          },
          "options": {
              "seed": seed + runs, "budget": budget - len(hist.F), "tol": args.tol,
              "check_cache": True, "store_cache": True, "rich_direction": True,
              "opportunistic": opportunistic,
              "constraint_strategy": constraint_strategy, "mo_search": mo_search,
              "save_results": False, "display": False,
          },
      }
      if direction_type is not None:   # otherwise OMADS' default for 2 objectives
        data["options"]["direction_type"] = direction_type
      if solver == "mads":
        data["search"] = {"type": "sampling", "s_method": s_method,
                          "ns": ns, "batch": ns}
      stderr = sys.stderr
      try:
        with open(os.devnull, "w", encoding="utf-8") as devnull, \
                contextlib.redirect_stdout(devnull):
          (mads if solver == "mads" else poll).main(data)
      except ValueError as exc:
        # After its main loop, OMADS reports the hypervolume of its front
        # and raises when the front is degenerate (one point, or one objective
        # constant). All evaluations are recorded by then, so carry on.
        if "Max and min values" not in str(exc):
          raise
      finally:
        sys.stderr = stderr   # mads.main wraps stderr
        root = logging.getLogger()   # OMADS keeps OMADS.log open; close it
        for handler in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
          root.removeHandler(handler)
          handler.close()
      runs += 1
  return hist, runs


def run_nsga2(problem, budget, seed):
  from pymoo.algorithms.moo.nsga2 import NSGA2
  from pymoo.core.problem import Problem
  from pymoo.optimize import minimize
  hist = History(problem, budget)

  class Wrapped(Problem):
    def __init__(self):
      super().__init__(n_var=problem.n_var, n_obj=2, n_ieq_constr=problem.n_ieq_constr,
                       xl=problem.xl, xu=problem.xu)

    def _evaluate(self, X, out, *args, **kwargs):
      out["F"], G = hist.evaluate(X)
      if problem.n_ieq_constr:
        out["G"] = G

  minimize(Wrapped(), NSGA2(pop_size=NSGA2_POPULATION), ("n_eval", budget),
           seed=seed, verbose=False)
  return hist, 1


def job(name, algorithm, seed, path, args):
  warnings.filterwarnings("ignore")
  problem = get_problem(name)
  budget = budget_of(problem, args)
  tic = time.perf_counter()
  variant = omads_variant(algorithm, args.ns)
  if variant:
    hist, runs = run_omads(problem, budget, seed, args, **variant)
  elif algorithm == "NSGA-II":
    hist, runs = run_nsga2(problem, budget, seed)
  else:
    raise ValueError(f"unknown algorithm {algorithm}")
  F, G = hist.arrays()
  os.makedirs(os.path.dirname(path), exist_ok=True)
  np.savez_compressed(path + ".tmp.npz", F=F, G=G, budget=budget, runs=runs,
                      n_var=problem.n_var, seconds=time.perf_counter() - tic)
  os.replace(path + ".tmp.npz", path)
  return f"{PROBLEMS[name][0]} {algorithm} s{seed}: {len(F)} evals, {runs} run(s), " \
         f"{(G <= 0).all(axis=1).sum()} feasible, {time.perf_counter() - tic:.0f} s"


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--problems", default=",".join(PROBLEMS),
                      help="comma-separated pymoo names (default: all 29)")
  parser.add_argument("--algorithms", default=",".join(ALGORITHMS),
                      help="comma-separated: OMADS (poll.main), OMADS-MADS[-<METHOD>]"
                      "[-ns<k>] (mads.main) and NSGA-II (default: OMADS, OMADS-MADS,"
                      " OMADS-MADS-HALTON, NSGA-II)")
  parser.add_argument("--ns", type=int, default=150,
                      help="samples per search step of the mads variants whose"
                      " label has no -ns<k> (default: 150)")
  parser.add_argument("--seeds", type=int, default=11,
                      help="runs per problem and algorithm, seeds 1..N (default: 11)")
  parser.add_argument("--budget-multiplier", "-b", type=float, default=1000,
                      help="budget = max(min_budget, this x n) (default: 1000)")
  parser.add_argument("--min-budget", type=int, default=2000,
                      help="smallest budget (default: 2000)")
  parser.add_argument("--tol", type=float, default=1e-9,
                      help="OMADS minimum frame size (default: 1e-9)")
  parser.add_argument("--max-restarts", type=int, default=20,
                      help="maximum OMADS runs per seed (default: 20)")
  parser.add_argument("--workers", type=int,
                      default=min(max((os.cpu_count() or 2) // 2, 1), 12),
                      help="parallel processes (default: half the CPU cores, at most 12)")
  parser.add_argument("--out", default=os.path.join(ROOT, "exdata", "cmo"),
                      help="results folder (default: exdata/cmo)")
  parser.add_argument("--force", action="store_true", help="rerun finished runs")
  args = parser.parse_args()
  names = [p.strip().lower() for p in args.problems.split(",")]
  unknown = [p for p in names if p not in PROBLEMS]
  if unknown:
    raise SystemExit(f"unknown problems: {unknown}; choose from {list(PROBLEMS)}")
  jobs = []
  for name in names:
    for algorithm in args.algorithms.split(","):
      for seed in range(1, args.seeds + 1):
        path = os.path.join(args.out, name, f"{algorithm}_s{seed}.npz")
        if args.force or not os.path.exists(path):
          jobs.append((name, algorithm, seed, path))
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
  shutil.rmtree(os.path.join(args.out, "__pycache__"), ignore_errors=True)


if __name__ == "__main__":
  main()
