"""Pareto fronts of OMADS on six constrained bi-objective test problems.

The problems are classical constrained bi-objective functions (see e.g. the
list of test functions for optimization on Wikipedia):

  Binh and Korn       n = 2, 2 constraints
  Chankong and Haimes n = 2, 2 constraints
  CONSTR              n = 2, 2 constraints
  CTP1                n = 2, 2 constraints
  Osyczka and Kundu   n = 6, 6 constraints
  Test function 4     n = 2, 3 constraints

Each problem is solved once by ``mads.main`` (progressive barrier for every
constraint, multiobjective model searches, a Halton sampling search of 10
points after each failed poll), from the centre of the box, with a fixed
seed and budget. Every evaluation is plotted (nondominated feasible,
dominated feasible, infeasible) with the true Pareto front: computed from a
dense grid of the box for the two-variable problems, taken from pymoo for
Osyczka and Kundu. The panel titles give the hypervolume of the OMADS front
relative to that of the best known front, the nondominated union of the true
front and the OMADS front (a grid front can be slightly dominated by points
off the grid); objectives are scaled by the ideal and nadir points of the
best known front and the reference point is 1.1.

The problem definitions are checked against pymoo (BNH, SRN, OSY, CTP1) before
the runs.

Usage, from the repository root:

  python benchmarks/cmo/plot_constrained_mo.py [--budget 5000] [--seed 0]
"""
import argparse
import contextlib
import logging
import os
import sys
import tempfile
import warnings

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


# --- Problems: f(x) -> (f1, f2), g(x) -> constraints, feasible when g <= 0 -----

def binh_korn(x):
  f = (4 * x[0] ** 2 + 4 * x[1] ** 2, (x[0] - 5) ** 2 + (x[1] - 5) ** 2)
  g = ((x[0] - 5) ** 2 + x[1] ** 2 - 25, 7.7 - (x[0] - 8) ** 2 - (x[1] + 3) ** 2)
  return f, g


def chankong_haimes(x):
  f = (2 + (x[0] - 2) ** 2 + (x[1] - 1) ** 2, 9 * x[0] - (x[1] - 1) ** 2)
  g = (x[0] ** 2 + x[1] ** 2 - 225, x[0] - 3 * x[1] + 10)
  return f, g


def constr(x):
  f = (x[0], (1 + x[1]) / x[0])
  g = (6 - x[1] - 9 * x[0], 1 + x[1] - 9 * x[0])
  return f, g


def _ctp1_constants(J=2):
  # Deb, Pratap & Meyarivan's recurrence; rounded, a = (0.858, 0.728) and
  # b = (0.541, 0.295), the values usually quoted.
  a, b, alpha = [1.0], [1.0], 1 / (J + 1)
  for j in range(J):
    beta = a[j] * np.exp(-b[j] * alpha)
    a.append((a[j] + beta) / 2)
    b.append(-np.log(beta / a[j + 1]) / alpha)
    alpha += 1 / (J + 1)
  return a[1:], b[1:]


_CTP1_A, _CTP1_B = _ctp1_constants()


def ctp1(x):
  f1 = x[0]
  f2 = (1 + x[1]) * np.exp(-x[0] / (1 + x[1]))
  g = tuple(1 - f2 / (a * np.exp(-b * f1)) for a, b in zip(_CTP1_A, _CTP1_B))
  return (f1, f2), g


def osyczka_kundu(x):
  f1 = -(25 * (x[0] - 2) ** 2 + (x[1] - 2) ** 2 + (x[2] - 1) ** 2 + (x[3] - 4) ** 2
         + (x[4] - 1) ** 2)
  f2 = sum(v ** 2 for v in x)
  g = (2 - x[0] - x[1], x[0] + x[1] - 6, x[1] - x[0] - 2, x[0] - 3 * x[1] - 2,
       (x[2] - 3) ** 2 + x[3] - 4, 4 - (x[4] - 3) ** 2 - x[5])
  return (f1, f2), g


def test_function_4(x):
  f = (x[0] ** 2 - x[1], -0.5 * x[0] - x[1] - 1)
  g = (x[0] / 6 + x[1] - 6.5, 0.5 * x[0] + x[1] - 7.5, 5 * x[0] + x[1] - 30)
  return f, g


# name: (function, lb, ub, pymoo name or None)
PROBLEMS = {
    "Binh and Korn": (binh_korn, [0, 0], [5, 3], "bnh"),
    "Chankong and Haimes": (chankong_haimes, [-20, -20], [20, 20], "srn"),
    "CONSTR": (constr, [0.1, 0], [1, 5], None),
    "CTP1": (ctp1, [0, 0], [1, 1], "ctp1"),
    "Osyczka and Kundu": (osyczka_kundu, [0, 0, 1, 0, 1, 0], [10, 10, 5, 6, 5, 10], "osy"),
    "Test function 4": (test_function_4, [-7, -7], [4, 4], None),
}


def check_against_pymoo(npts=2000, seed=0):
  """Objectives and feasibility agree with pymoo's versions of the problems."""
  from pymoo.problems import get_problem
  rng = np.random.default_rng(seed)
  for name, (fun, lb, ub, key) in PROBLEMS.items():
    if key is None:
      continue
    p = get_problem(key)
    assert np.allclose(p.xl, lb) and np.allclose(p.xu, ub), f"{name}: bounds differ from pymoo"
    X = rng.uniform(lb, ub, (npts, len(lb)))
    out = p.evaluate(X, return_values_of=["F", "G"])
    F = np.array([fun(x)[0] for x in X])
    feas = np.array([np.all(np.asarray(fun(x)[1]) <= 0) for x in X])
    assert np.allclose(F, out[0], rtol=1e-12, atol=1e-12), f"{name}: objectives differ from pymoo"
    assert np.array_equal(feas, np.all(out[1] <= 0, axis=1)), f"{name}: feasibility differs from pymoo"
    print(f"  {name}: matches pymoo '{key}' on {npts} random points")


# --- Reference fronts ----------------------------------------------------------

def nondominated(F):
  import moocore
  F = np.unique(F[np.all(np.isfinite(F), axis=1)], axis=0)
  return F[moocore.is_nondominated(F)]


def true_front(name, grid=1500):
  fun, lb, ub, key = PROBLEMS[name]
  if len(lb) > 2:
    from pymoo.problems import get_problem
    return nondominated(get_problem(key).pareto_front())
  # Dense grid of the box, vectorized through the scalar definitions.
  a, b = np.meshgrid(np.linspace(lb[0], ub[0], grid), np.linspace(lb[1], ub[1], grid))
  with np.errstate(all="ignore"):
    f, g = fun(np.array([a.ravel(), b.ravel()]))
  F = np.column_stack([np.broadcast_to(v, a.size) for v in f])
  feas = np.all(np.column_stack([np.broadcast_to(v, a.size) for v in g]) <= 0, axis=1)
  return nondominated(F[feas])


def hv_ratio(front, true):
  import moocore
  ref_front = nondominated(np.vstack([true, front]))
  ideal, nadir = ref_front.min(axis=0), ref_front.max(axis=0)

  def scale(Y):
    return (Y - ideal) / (nadir - ideal)

  ref = np.full(2, 1.1)
  return moocore.hypervolume(scale(front), ref=ref) / moocore.hypervolume(scale(ref_front), ref=ref)


# --- OMADS run -------------------------------------------------------------------

def run_omads(name, budget, seed, strategy):
  from OMADS import mads
  fun, lb, ub, _ = PROBLEMS[name]
  lb, ub = np.asarray(lb, float), np.asarray(ub, float)
  F, G = [], []

  def blackbox(x):
    f, g = fun(np.asarray(x, dtype=float))
    F.append([float(v) for v in f])
    G.append([float(v) for v in g])
    return [F[-1], G[-1]]

  ncon = len(fun(lb)[1])
  with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    data = {
        "evaluator": {"blackbox": blackbox},
        "param": {
            "name": "cmo", "baseline": list((lb + ub) / 2), "lb": list(lb), "ub": list(ub),
            "scaling": list(ub - lb), "nobj": 2, "is_pareto": True,
            "fun_names": ["f1", "f2"], "constraints_type": ["PB"] * ncon, "post_dir": tmp,
        },
        "options": {
            "seed": seed, "budget": budget, "tol": 1e-9, "check_cache": True,
            "store_cache": True, "constraint_strategy": strategy,
            "save_results": False, "display": False,
        },
        "search": {"type": "sampling", "s_method": "HALTON", "ns": 10, "batch": 10},
    }
    stderr = sys.stderr
    try:
      with open(os.devnull, "w", encoding="utf-8") as devnull, \
              contextlib.redirect_stdout(devnull):
        mads.main(data)
    finally:
      sys.stderr = stderr   # mads.main wraps stderr
      root = logging.getLogger()   # OMADS keeps OMADS.log open; close it
      for handler in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
        root.removeHandler(handler)
        handler.close()
  return np.array(F), np.array(G)


# --- Figure ----------------------------------------------------------------------

def plot(results, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
  for ax, (name, (F, G, ref_front, ratio)) in zip(axes.ravel(), results.items()):
    feas = np.all(G <= 0, axis=1)
    front = nondominated(F[feas]) if feas.any() else np.empty((0, 2))
    on_front = np.zeros(len(F), dtype=bool)
    if len(front):
      on_front = feas & (np.isin(F[:, 0], front[:, 0]) & np.isin(F[:, 1], front[:, 1]))
    # Axis limits: the true front with a margin, so stray points do not hide it.
    lo, hi = ref_front.min(axis=0), ref_front.max(axis=0)
    pad = 0.12 * (hi - lo)
    ax.scatter(*F[~feas].T, s=5, color="#f2b134", alpha=0.6, lw=0, label="infeasible")
    ax.scatter(*F[feas & ~on_front].T, s=5, color="#9a9a9a", alpha=0.6, lw=0,
               label="dominated feasible")
    order = np.argsort(ref_front[:, 0])
    ax.scatter(*front.T, s=11, color="#1c5cab", lw=0, label="OMADS front", zorder=3)
    ax.plot(*ref_front[order].T, ".", ms=1.2, color="black", label="true Pareto front",
            zorder=4)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    n = len(PROBLEMS[name][1])
    ax.set_title(f"{name} ($n={n}$): HV ratio {ratio:.3f}", fontsize=10)
    ax.set_xlabel("$f_1$")
    ax.set_ylabel("$f_2$")
    ax.grid(alpha=0.3)
  handles, labels = axes.ravel()[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, markerscale=3)
  fig.tight_layout(rect=(0, 0.05, 1, 1))
  fig.savefig(path + ".png", dpi=150)
  fig.savefig(path + ".pdf")
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--budget", type=int, default=5000, help="evaluations per problem")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--strategy", default="PB", choices=("PB", "TWO_PHASE"),
                      help="OMADS constraint_strategy (default: %(default)s)")
  parser.add_argument("--out", default=os.path.join(ROOT, "ppdata", "cmo", "constrained_mo"),
                      help="figure path without extension (default: %(default)s)")
  args = parser.parse_args()
  warnings.filterwarnings("ignore")
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  print("Checking the problems against pymoo")
  check_against_pymoo()
  results = {}
  for name in PROBLEMS:
    F, G = run_omads(name, args.budget, args.seed, args.strategy)
    ref_front = true_front(name)
    feas = np.all(G <= 0, axis=1)
    ratio = hv_ratio(nondominated(F[feas]), ref_front) if feas.any() else 0.0
    results[name] = (F, G, ref_front, ratio)
    print(f"  {name:20s} {len(F)} evaluations, {feas.sum()} feasible, "
          f"HV ratio {ratio:.4f}")
  plot(results, args.out)
  print("wrote", args.out + ".png", "and .pdf")


if __name__ == "__main__":
  main()
