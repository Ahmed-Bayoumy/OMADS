"""Indicators and profiles of the DMulti-MADS benchmark (``run_dms_benchmark.py``).

Two sets of measures, each from the article that introduced them:

1. Hypervolume data profiles (Bigeon, Le Digabel & Salomon, DMulti-MADS,
   COAP 2021, Sect. 6.2), the measure of the NOMAD 4 multiobjective
   experiments. For each problem, the reference front Y^p gathers the
   nondominated points of all runs of all solvers; objectives are scaled so
   that its ideal and nadir points map to 0 and 1 and the reference point is
   1.1 in every objective. A run solves the problem with accuracy eps after e
   evaluations when HV(front after e evaluations) >= (1 - eps) HV(Y^p). The
   data profile is the fraction of problems solved within k groups of n + 1
   evaluations; for each solver, the profile of every run (one seed across
   all problems) is computed, and their average is drawn with the band
   between the lowest and the highest (the paper's lower and upper bound
   profiles).

2. The metrics of Custodio, Madeira, Vaz & Vicente (DMS, SIAM J. Optim.
   2011, Sect. 5.3), all summarized by performance profiles (Dolan & More):
   - Purity, solvers compared in pairs: the fraction of a solver's front
     that is nondominated in the union of the two fronts, with t = 1/purity
     (+inf when no point survives). For stochastic solvers, the best and the
     worst of 10 runs, selected by their purity against the union of the 10
     runs of that solver.
   - Spread Gamma (largest gap) and Delta (uniformity) of each front, with
     the 'extreme points' of every objective taken from the nondominated
     union of all runs of all solvers; averaged over 10 runs.

Every front is the set of nondominated points among all evaluations of a run.
Custodio's measures use the first 10 runs of each solver (10 is what both
papers ran for the deterministic-direction solvers), the data profiles all
the runs (50 for NSGA-II, as in the DMulti-MADS paper).

Usage, from the repository root:

  python benchmarks/dms/postprocess_dms.py [--data exdata/dms] [--out ppdata/dms]
"""
import argparse
import csv
import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import moocore
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from problems import PROBLEMS  # noqa: E402
from run_dms_benchmark import SEEDS  # noqa: E402

EPSILONS = (1e-2, 5e-2, 1e-1)
CUSTODIO_RUNS = 10
# Legend names, as in the paper (OMADS-mads: Halton sampling search, ns = 10).
NAMES = {"OMADS": "OMADS-poll", "OMADS-MADS-HALTON-ns10": "OMADS-mads",
         "OMADS-NOMS": "OMADS-poll (no model search)",
         "OMADS-MADS-HALTON-ns10-NOMS": "OMADS-mads (no model search)",
         "NSGA-II": "NSGA-II"}
# Okabe-Ito colours (distinguishable with colour-vision deficiencies), each solver
# also with its own line style.
COLORS = {"OMADS": "#0072B2", "OMADS-MADS-HALTON-ns10": "#009E73", "NSGA-II": "#D55E00",
          "OMADS-NOMS": "#56B4E9", "OMADS-MADS-HALTON-ns10-NOMS": "#E69F00"}
LINESTYLES = {"OMADS": "-", "OMADS-MADS-HALTON-ns10": "--", "NSGA-II": "-.",
              "OMADS-NOMS": ":", "OMADS-MADS-HALTON-ns10-NOMS": (0, (5, 1, 1, 1))}
# Figure widths in inches, drawn at print size for an ACM single-column page
# (text width about 5.5 in), so that 8 pt text stays 8 pt in the paper.
FULL_WIDTH, NARROW_WIDTH = 5.6, 4.8


# --- Nondominated sets -------------------------------------------------------

def nondominated(F):
  """Nondominated rows of F (duplicates kept once)."""
  F = np.unique(np.asarray(F, dtype=float), axis=0)
  F = F[np.all(np.isfinite(F), axis=1)]
  if len(F) == 0:
    return F
  return F[moocore.is_nondominated(F)]


def hypervolume(Z, ref):
  """Exact hypervolume (moocore); dominated points and points that do not
  strictly dominate `ref` add nothing, so `Z` needs no filtering."""
  Z = np.asarray(Z, dtype=float)
  Z = Z[np.all(np.isfinite(Z), axis=1)]
  return float(moocore.hypervolume(Z, ref=ref)) if len(Z) else 0.0


# --- Hypervolume data profiles ----------------------------------------------

def scaler(Yp):
  ideal, nadir = Yp.min(axis=0), Yp.max(axis=0)
  span = np.where(nadir > ideal, nadir - ideal, 1.0)
  return lambda Y: (np.asarray(Y) - ideal) / span


def solve_times(F, T, hv_ref):
  """Evaluations needed to reach (1 - eps) HV(Y^p), for each eps (inf if
  never), by bisection on the number of evaluations (the hypervolume of the
  front grows with it)."""
  ref = np.full(F.shape[1], 1.1)
  Z = T(F)
  cache = {}

  def hv(k):   # hypervolume of the first k evaluations
    if k not in cache:
      cache[k] = hypervolume(Z[:k], ref) if k else 0.0
    return cache[k]

  out = []
  for eps in EPSILONS:
    target = (1 - eps) * hv_ref
    if hv_ref <= 0 or hv(len(Z)) < target:
      out.append(np.inf)
      continue
    lo, hi = 0, len(Z)   # hv(lo) < target <= hv(hi)
    while hi - lo > 1:
      mid = (lo + hi) // 2
      lo, hi = (mid, hi) if hv(mid) < target else (lo, mid)
    out.append(hi)   # evaluations used
  return out


# --- Custodio et al. metrics -------------------------------------------------

def purity(front, union_front):
  if len(front) == 0:
    return 0.0
  ref = {tuple(y) for y in union_front}
  return sum(tuple(y) in ref for y in front) / len(front)


def spreads(front, lo, hi):
  """Gamma and Delta of `front` given the extreme values lo, hi of every
  objective (Custodio et al. 2011, eqs. 5.1 and 5.2)."""
  gam, dlt = 0.0, 0.0
  for j in range(front.shape[1]):
    v = np.sort(np.clip(front[:, j], lo[j], hi[j]))
    d = np.diff(np.concatenate([[lo[j]], v, [hi[j]]]))   # delta_0 .. delta_N
    gam = max(gam, float(d.max()))
    inner = d[1:-1]
    mean = float(inner.mean()) if len(inner) else 0.0
    num = d[0] + d[-1] + float(np.sum(np.abs(inner - mean)))
    den = d[0] + d[-1] + len(inner) * mean
    dlt = max(dlt, num / den if den > 0 else 0.0)
  return gam, dlt


# --- Per-problem computation ------------------------------------------------

def load_problem(data, name):
  runs = {}
  for path in glob.glob(os.path.join(data, name, "*_s*.npz")):
    alg, seed = os.path.basename(path)[:-4].rsplit("_s", 1)
    runs.setdefault(alg, {})[int(seed)] = np.load(path)["F"]
  # Runs in seed order of the paper's list.
  return {a: [r[s] for s in SEEDS if s in r] for a, r in runs.items()}


def process_problem(args):
  data, name = args
  runs = load_problem(data, name)
  if not runs:
    return name, None
  n = PROBLEMS[name][1]
  fronts = {a: [nondominated(F) for F in Fs] for a, Fs in runs.items()}
  Yp = nondominated(np.vstack([f for fs in fronts.values() for f in fs if len(f)]))
  T = scaler(Yp)
  hv_ref = hypervolume(T(Yp), np.full(Yp.shape[1], 1.1))
  res = {"n": n, "m": Yp.shape[1], "hv_ref": hv_ref, "algs": {}}
  lo, hi = Yp.min(axis=0), Yp.max(axis=0)
  for a, Fs in runs.items():
    times, hvs, gam, dlt, pur_self = [], [], [], [], []
    own = nondominated(np.vstack([f for f in fronts[a][:CUSTODIO_RUNS] if len(f)]))
    for k, F in enumerate(Fs):
      times.append(solve_times(F, T, hv_ref))
      fr = fronts[a][k]
      hvs.append(hypervolume(T(fr), np.full(Yp.shape[1], 1.1)) / hv_ref
                 if hv_ref > 0 else 0.0)
      if k < CUSTODIO_RUNS:
        g, d = spreads(fr, lo, hi)
        gam.append(g)
        dlt.append(d)
        pur_self.append(purity(fr, own))
    best = int(np.argmax(pur_self))
    worst = int(np.argmin(pur_self))
    res["algs"][a] = {"times": np.array(times), "hv": np.array(hvs),
                      "gamma": float(np.mean(gam)), "delta": float(np.mean(dlt)),
                      "best": fronts[a][best], "worst": fronts[a][worst],
                      "evals": [len(F) for F in Fs]}
  return name, res


# --- Profiles and figures ----------------------------------------------------

def perf_profile(T, taus):
  """Dolan-More performance profiles: T[p, s] (lower is better, inf =
  failure). Returns rho[s, k] for each tau in taus."""
  best = np.min(T, axis=1, keepdims=True)
  with np.errstate(divide="ignore", invalid="ignore"):
    R = np.where(np.isfinite(T), T / np.where(best > 0, best, 1e-300), np.inf)
    R[(T == best) & np.isfinite(T)] = 1.0
  return np.array([[np.mean(R[:, s] <= t) for t in taus] for s in range(T.shape[1])])


def data_profiles(results, algs, problems, eps_i, kappas):
  """Average, lowest and highest data profile over the runs of each solver."""
  out = {}
  for a in algs:
    nruns = min(len(results[p]["algs"][a]["times"]) for p in problems)
    D = np.zeros((nruns, len(kappas)))
    for j in range(nruns):
      groups = np.array([results[p]["algs"][a]["times"][j, eps_i] / (results[p]["n"] + 1)
                         for p in problems])
      D[j] = [np.mean(groups <= k) for k in kappas]
    out[a] = (D.mean(axis=0), D.min(axis=0), D.max(axis=0), nruns)
  return out


def plot_all(results, algs, out):  # noqa: C901
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  plt.rcParams.update({"font.size": 8, "legend.fontsize": 7.5, "axes.titlesize": 8,
                       "lines.linewidth": 1.3, "pdf.fonttype": 42})
  files = []
  for label, sel in (("all", lambda r: True), ("m2", lambda r: r["m"] == 2)):
    problems = [p for p, r in results.items() if sel(r)]
    kappas = np.linspace(0, 650, 651)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.4), sharey=True)
    for i, (ax, eps) in enumerate(zip(axes, EPSILONS)):
      for a, (mean, lo, hi, nruns) in data_profiles(results, algs, problems, i, kappas).items():
        ax.plot(kappas, 100 * mean, color=COLORS.get(a), ls=LINESTYLES.get(a, "-"),
                label=f"{NAMES.get(a, a)} ({nruns} runs)")
        ax.fill_between(kappas, 100 * lo, 100 * hi, color=COLORS.get(a), alpha=0.15, lw=0)
      ax.set_title(f"$\\varepsilon_\\tau$ = {eps:g}")
      ax.set_xlabel("Groups of $n+1$ evaluations")
      ax.grid(alpha=0.3)
    axes[0].set_ylabel("Problems solved (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = os.path.join(out, f"dms_data_profiles_{label}.png")
    fig.savefig(path, dpi=150)
    fig.savefig(path[:-4] + ".pdf")   # vector copy for the paper
    plt.close(fig)
    files.append(path)

  problems = list(results)
  taus = np.concatenate([np.linspace(1, 2, 101), np.linspace(2, 20, 181)[1:]])
  # Purity, pairwise, best and worst runs.
  pairs = [(a, b) for i, a in enumerate(algs) for b in algs[i + 1:]]
  fig, axes = plt.subplots(len(pairs), 2, figsize=(NARROW_WIDTH, 1.75 * len(pairs) + 0.2),
                           squeeze=False)
  for row, (a, b) in enumerate(pairs):
    for col, which in enumerate(("best", "worst")):
      T = np.zeros((len(problems), 2))
      for i, p in enumerate(problems):
        fa, fb = results[p]["algs"][a][which], results[p]["algs"][b][which]
        union = nondominated(np.vstack([f for f in (fa, fb) if len(f)]))
        for s, f in enumerate((fa, fb)):
          pu = purity(f, union)
          T[i, s] = 1 / pu if pu > 0 else np.inf
      rho = perf_profile(T, taus)
      ax = axes[row, col]
      for s, alg in enumerate((a, b)):
        ax.step(taus, rho[s], where="post", color=COLORS.get(alg), ls=LINESTYLES.get(alg, "-"),
                label=NAMES.get(alg, alg))
      k = min(CUSTODIO_RUNS, *(len(results[p]["algs"][x]["hv"]) for p in problems
                                for x in (a, b)))
      ax.set_title(f"{which.capitalize()} of {k} runs")
      ax.set_xlabel(r"$\tau$")
      ax.set_ylim(0, 1.02)
      ax.grid(alpha=0.3)
      ax.legend(loc="lower right")
  fig.tight_layout()
  path = os.path.join(out, "dms_purity_profiles.png")
  fig.savefig(path, dpi=150)
  fig.savefig(path[:-4] + ".pdf")   # vector copy for the paper
  plt.close(fig)
  files.append(path)
  # Spread metrics, all solvers together.
  fig, axes = plt.subplots(1, 2, figsize=(NARROW_WIDTH, 2.3))
  for ax, metric, title in zip(axes, ("gamma", "delta"), (r"$\Gamma$", r"$\Delta$")):
    T = np.array([[results[p]["algs"][a][metric] for a in algs] for p in problems])
    rho = perf_profile(T, taus)
    for s, a in enumerate(algs):
      ax.step(taus, rho[s], where="post", color=COLORS.get(a), ls=LINESTYLES.get(a, "-"),
              label=NAMES.get(a, a))
    ax.set_title(f"{title} (average of {CUSTODIO_RUNS} runs)")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
  fig.tight_layout()
  path = os.path.join(out, "dms_spread_profiles.png")
  fig.savefig(path, dpi=150)
  fig.savefig(path[:-4] + ".pdf")   # vector copy for the paper
  plt.close(fig)
  files.append(path)
  return files


def write_tables(results, algs, out):
  problems = list(results)
  path = os.path.join(out, "dms_summary.csv")
  with open(path, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["problem", "n", "m"] + [f"{h} {a}" for a in algs for h in (
        "median HV ratio", "solved 1e-2", "solved 5e-2", "solved 1e-1", "Gamma", "Delta")])
    for p in problems:
      r = results[p]
      row = [p, r["n"], r["m"]]
      for a in algs:
        d = r["algs"][a]
        row += [f"{np.median(d['hv']):.4f}"] + \
            [f"{np.mean(np.isfinite(d['times'][:, i])):.2f}" for i in range(3)] + \
            [f"{d['gamma']:.4g}", f"{d['delta']:.4g}"]
      w.writerow(row)
  lines = ["solver                                     runs  solved at budget (eps = 1e-2 / 5e-2 / 1e-1)"
           "   median HV ratio   Gamma wins  Delta wins"]
  gam = np.array([[results[p]["algs"][a]["gamma"] for a in algs] for p in problems])
  dlt = np.array([[results[p]["algs"][a]["delta"] for a in algs] for p in problems])
  for s, a in enumerate(algs):
    solved = [np.mean([np.mean(np.isfinite(results[p]["algs"][a]["times"][:, i]))
                       for p in problems]) for i in range(3)]
    hv = np.median([np.median(results[p]["algs"][a]["hv"]) for p in problems])
    nruns = min(len(results[p]["algs"][a]["hv"]) for p in problems)
    lines.append(f"{NAMES.get(a, a):42s} {nruns:4d}  "
                 + " / ".join(f"{100 * v:5.1f}%" for v in solved)
                 + f"          {hv:.3f}        {int(np.sum(gam[:, s] == gam.min(axis=1))):3d}"
                 + f"        {int(np.sum(dlt[:, s] == dlt.min(axis=1))):3d}")
  return path, "\n".join(lines)


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--data", default=os.path.join(ROOT, "exdata", "dms"))
  parser.add_argument("--out", default=os.path.join(ROOT, "ppdata", "dms"))
  parser.add_argument("--algorithms", default=None,
                      help="comma-separated labels (default: every label with runs on all problems)")
  parser.add_argument("--workers", type=int, default=min(max((os.cpu_count() or 2) // 2, 1), 12))
  parser.add_argument("--reuse", action="store_true",
                      help="reuse the indicators cached by a previous run in <out>/indicators.pkl")
  args = parser.parse_args()
  os.makedirs(args.out, exist_ok=True)
  cache = os.path.join(args.out, "indicators.pkl")
  if args.reuse and os.path.exists(cache):
    results = pickle.load(open(cache, "rb"))
  else:
    names = [p for p in PROBLEMS if os.path.isdir(os.path.join(args.data, p))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
      results = {name: r for name, r in pool.map(process_problem, [(args.data, p) for p in names])
                 if r is not None}
    pickle.dump(results, open(cache, "wb"))
  if args.algorithms:
    algs = args.algorithms.split(",")
  else:
    common = set.intersection(*[set(r["algs"]) for r in results.values()])
    algs = [a for a in NAMES if a in common] + sorted(common - set(NAMES))
  missing = [p for p in PROBLEMS if p not in results]
  print(f"{len(results)} problems ({len(missing)} without data), solvers: {algs}")
  for f in plot_all(results, algs, args.out):
    print("wrote", f)
  path, table = write_tables(results, algs, args.out)
  print("wrote", path)
  print(table)


if __name__ == "__main__":
  main()
