"""Indicators, figures and summary table of the constrained bi-objective benchmark.

Reads the histories written by ``run_cmo_benchmark.py`` and computes, after a
growing number of evaluations, two indicators of the *feasible* non-dominated
points found so far, in the objective space normalized by the true Pareto
front (ideal point at 0, nadir point at 1):

  HV     hypervolume with respect to the reference point (1.1, 1.1); higher is
         better (the dashed line is the hypervolume of pymoo's sampled front);
  IGD+   inverted generational distance plus to the true front; lower is better
         (infinite while no feasible point has been found).

Writes to ``--out`` (default ``ppdata/cmo``):

  cmo_hv.png, cmo_igd.png   median and interquartile range over the seeds against
                            the evaluations, one panel per problem;
  cmo_ecdf.png              fraction of (problem, seed, IGD+ target) triples reached
                            against the evaluations / n, over all problems
                            (31 targets from 1 to 1e-3), as in COCO's runtime ECDFs;
  cmo_fronts.png            final feasible fronts of the run with the median final
                            HV of each algorithm, with the true front;
  cmo_summary.csv           per problem: medians at the budget, runs with a feasible
                            point, runs with HV > 0 (some feasible point inside the
                            reference box, i.e. near the true front; medians hide
                            bimodal results) and a two-sided Mann-Whitney test on
                            the final HV.

Usage, from the repository root:

  python benchmarks/cmo/plot_cmo_benchmark.py
"""
import argparse
import csv
import glob
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_cmo_benchmark import PROBLEMS, ROOT, get_problem, omads_variant  # noqa: E402

DEFAULT_NS = 150   # samples per search step of the labels without -ns<k>
# Algorithms found in the data (set by load()), in drawing order.
ALGORITHMS = []
FIXED_COLORS = {"OMADS": "#1c5cab", "NSGA-II": "#eb6834",
                ("ACTIVE", 150): "#00a6a6", ("HALTON", 150): "#7b3fa0",
                ("ACTIVE", 10): "#2e9e3e", ("HALTON", 10): "#c79a00"}
EXTRA_COLORS = ["#d55181", "#8a5a2b", "#52514e", "#3987e5"]


class _ByLabel(dict):
  """Display name or color of an algorithm label, computed on first use."""

  def __init__(self, kind):
    super().__init__()
    self.kind = kind

  def __missing__(self, label):
    v = omads_variant(label, DEFAULT_NS)
    poll_opts = "" if v is None else "".join(
        [{"N+1": " n+1", "ORTHO_2N": " 2n"}.get(v["direction_type"], ""),
         " opp" if v["opportunistic"] else "",
         " two-phase" if v["constraint_strategy"] == "TWO_PHASE" else "",
         " no-mo-search" if not v["mo_search"] else ""])
    if self.kind == "name":
      if v is None:
        value = label
      elif v["solver"] == "poll":
        value = "OMADS poll" + poll_opts
      else:
        value = f"OMADS mads {v['s_method']} ns={v['ns']}" + poll_opts
    else:
      if v is None or (v["solver"] == "poll" and not poll_opts):
        key = label
      else:
        key = (v["s_method"], v["ns"]) if not poll_opts else None
      value = FIXED_COLORS.get(key) or EXTRA_COLORS[len(self) % len(EXTRA_COLORS)]
    self[label] = value
    return value


NAMES, COLORS = _ByLabel("name"), _ByLabel("color")


def discover(data, wanted=None):
  """Algorithm labels with results under ``data``: OMADS poll, the mads
  variants, then NSGA-II."""
  labels = {os.path.basename(p).rsplit("_s", 1)[0]
            for p in glob.glob(os.path.join(data, "*", "*_s*.npz"))}
  if wanted:
    labels &= set(wanted)
  def order(label):
    v = omads_variant(label, DEFAULT_NS)
    if v is None:
      return (2, label)
    if v["solver"] == "poll":
      return (0, label)
    return (1, f"{v['s_method']}{v['ns']:06d}{label}")
  return sorted(labels, key=order)
CHECKPOINTS = 60
REF_POINT = 1.1
IGD_TARGETS = np.logspace(0, -3, 31)


def non_dominated(F):
  """Non-dominated rows of a 2-D objective array, sorted by f1."""
  if not len(F):
    return F.reshape(0, 2)
  F = F[np.lexsort((F[:, 1], F[:, 0]))]
  keep = np.r_[True, np.minimum.accumulate(F[:, 1])[:-1] > F[1:, 1]]
  return F[keep]


def hypervolume(front, ref=REF_POINT):
  front = front[(front[:, 0] < ref) & (front[:, 1] < ref)]
  if not len(front):
    return 0.0
  front = non_dominated(front)
  right = np.r_[front[1:, 0], ref]
  return float(np.sum((right - front[:, 0]) * (ref - front[:, 1])))


def igd_plus(front, pf, chunk=256):
  if not len(front):
    return np.inf
  dmin = np.empty(len(pf))
  for i in range(0, len(pf), chunk):
    z = pf[i:i + chunk, None, :]
    dmin[i:i + chunk] = np.sqrt((np.maximum(front[None] - z, 0) ** 2).sum(-1)).min(1)
  return float(dmin.mean())


def evaluate_run(path, pf, ideal, scale):
  """Indicators of one run at the checkpoints."""
  d = np.load(path)
  F, G, budget = d["F"], d["G"], int(d["budget"])
  feasible = (G <= 0).all(axis=1) if G.shape[1] else np.ones(len(F), bool)
  Fn = (F - ideal) / scale
  ks = np.unique(np.geomspace(10, budget, CHECKPOINTS).astype(int))
  hv, igd = [], []
  for k in ks:
    front = non_dominated(Fn[:k][feasible[:k]])
    hv.append(hypervolume(front))
    igd.append(igd_plus(front, pf))
  final = non_dominated(Fn[feasible])
  return dict(k=ks, hv=np.array(hv), igd=np.array(igd), front=final, n=int(d["n_var"]),
              budget=budget, feasible=bool(feasible.any()))


def load(args):
  ALGORITHMS[:] = discover(args.data, args.algorithms)
  print("algorithms: " + ", ".join(f"{a} ({NAMES[a]})" for a in ALGORITHMS))
  results = {}
  for name in PROBLEMS:
    folder = os.path.join(args.data, name)
    if not os.path.isdir(folder):
      continue
    problem = get_problem(name)
    pf = problem.pareto_front()
    ideal, nadir = pf.min(axis=0), pf.max(axis=0)
    scale = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    pfn = (pf - ideal) / scale
    runs = {a: [evaluate_run(p, pfn, ideal, scale)
                for p in sorted(glob.glob(os.path.join(folder, f"{a}_s*.npz")))
                if int(p.rsplit("_s", 1)[1][:-4]) <= args.max_seed]
            for a in ALGORITHMS}
    if any(runs.values()):
      results[name] = dict(pf=pfn, hv_pf=hypervolume(pfn), m=problem.n_ieq_constr, runs=runs)
      print(f"  {PROBLEMS[name][0]}: " + ", ".join(
          f"{a} {len(r)} runs" for a, r in runs.items()), flush=True)
  return results


def place_legend(fig, axes, n, **kwargs):
  """Legend in the first empty panel of the grid, or below it."""
  handles, labels = axes[0, 0].get_legend_handles_labels()
  if n < axes.size:
    list(axes.flat)[n].legend(handles, labels, loc="center", fontsize=10,
                              frameon=False, **kwargs)
  else:
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               frameon=False, **kwargs)


def grid(n):
  cols = 5
  rows = -(-n // cols)
  fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.7 * rows), squeeze=False)
  for ax in list(axes.flat)[n:]:
    ax.axis("off")
  return fig, axes


def plot_indicator(results, key, path, title, ylabel, log):
  fig, axes = grid(len(results))
  for ax, (name, r) in zip(axes.flat, results.items()):
    for a, runs in r["runs"].items():
      if not runs:
        continue
      k = runs[0]["k"]
      vals = np.array([run[key] for run in runs])
      if log:
        vals = np.where(np.isfinite(vals), vals, np.nan)
      with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # no feasible point yet
        q1, med, q3 = np.nanpercentile(vals, [25, 50, 75], axis=0)
      ax.fill_between(k, q1, q3, color=COLORS[a], alpha=0.18, lw=0)
      ax.plot(k, med, color=COLORS[a], lw=1.6, label=NAMES[a])
    if key == "hv":
      ax.axhline(r["hv_pf"], color="#0b0b0b", lw=0.8, ls="--", label="true front")
    ax.set_xscale("log")
    # explicit limits: a panel without any feasible point has no data to scale on
    ax.set_xlim(10, max(run["budget"] for runs in r["runs"].values() for run in runs))
    if log:
      shown = [v for ln in ax.get_lines() for v in ln.get_ydata()]
      if any(np.isfinite(v) and v > 0 for v in shown):
        ax.set_yscale("log")
      else:
        ax.text(0.5, 0.5, "no feasible point", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#52514e")
    n = next((runs[0]["n"] for runs in r["runs"].values() if runs), "?")
    m = r["m"]
    ax.set_title(f"{PROBLEMS[name][0]}  (n={n}, {m} constr.)", fontsize=9)
    ax.grid(True, color="#ebeae6", lw=0.5)
    ax.tick_params(labelsize=7)
  place_legend(fig, axes, len(results))
  fig.supxlabel("evaluations", y=0.03)
  fig.supylabel(ylabel)
  fig.suptitle(title, fontsize=12)
  fig.tight_layout(rect=(0.01, 0.045, 1, 0.975))
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print("wrote", path)


def plot_ecdf(results, path):
  fig, ax = plt.subplots(figsize=(7.5, 4.6))
  xgrid = np.logspace(0, 4, 400)
  for a in ALGORITHMS:
    hits = []   # evaluations / n at which each (problem, seed, target) is reached
    for r in results.values():
      for run in r["runs"][a]:
        for t in IGD_TARGETS:
          reached = np.nonzero(run["igd"] <= t)[0]
          hits.append(run["k"][reached[0]] / run["n"] if len(reached) else np.inf)
    if not hits:
      continue
    hits = np.sort(np.array(hits))
    frac = np.searchsorted(hits, xgrid, side="right") / len(hits)
    ax.plot(xgrid, frac, color=COLORS[a], lw=2,
            label=f"{NAMES[a]} ({frac[-1]:.0%} at the end)")
  ax.set_xscale("log")
  ax.set_ylim(0, 1)
  ax.set_xlim(1, 1e4)
  ax.grid(True, color="#ebeae6", lw=0.6)
  ax.set_xlabel("evaluations / n")
  ax.set_ylabel("fraction of (problem, seed, target) triples")
  ax.set_title(f"Constrained bi-objective problems ({len(results)}): IGD+ targets "
               "from 1 to $10^{-3}$ reached", fontsize=10)
  ax.legend(loc="upper left", fontsize=9)
  fig.savefig(path, dpi=200, bbox_inches="tight")
  plt.close(fig)
  print("wrote", path)


def plot_fronts(results, path):
  fig, axes = grid(len(results))
  for ax, (name, r) in zip(axes.flat, results.items()):
    pf = r["pf"]
    ax.plot(pf[:, 0], pf[:, 1], ".", ms=2, c="#0b0b0b", label="true front", zorder=1)
    for a, runs in r["runs"].items():
      if not runs:
        continue
      final = np.array([run["hv"][-1] for run in runs])
      run = runs[int(np.argsort(final)[len(final) // 2])]   # median final HV
      f = run["front"]
      ax.plot(f[:, 0], f[:, 1], ".", ms=3, c=COLORS[a], label=f"{NAMES[a]} (median run)",
              zorder=3 if a.startswith("OMADS") else 2)
    # widen the view to fronts that stay away from the true one (at most 10)
    pts = np.vstack([pf] + [ln.get_xydata() for ln in ax.get_lines()])
    hi = float(np.clip(np.nanmax(pts) * 1.05, 1.3, 10.0))
    ax.set_xlim(-0.05, hi)
    ax.set_ylim(-0.05, hi)
    ax.set_title(PROBLEMS[name][0], fontsize=9)
    ax.grid(True, color="#ebeae6", lw=0.5)
    ax.tick_params(labelsize=7)
  place_legend(fig, axes, len(results), markerscale=4)
  fig.supxlabel("normalized $f_1$", y=0.03)
  fig.supylabel("normalized $f_2$")
  fig.suptitle("Final feasible non-dominated points (run with the median final HV)",
               fontsize=12)
  fig.tight_layout(rect=(0.01, 0.045, 1, 0.975))
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print("wrote", path)


def summary(results, path):
  """Per problem: medians and run counts of every algorithm, and a two-sided
  Mann-Whitney test of the final HV of each OMADS variant against NSGA-II."""
  algs = [a for a in ALGORITHMS if any(r["runs"][a] for r in results.values())]
  variants = [a for a in algs if a.startswith("OMADS")]
  header = ["problem", "n", "constraints", "budget", "HV true front"]
  header += [f"HV {NAMES[a]}" for a in algs] + [f"IGD+ {NAMES[a]}" for a in algs]
  header += [f"feasible runs {NAMES[a]}" for a in algs]
  header += [f"HV>0 runs {NAMES[a]}" for a in algs]
  for v in variants:
    header += [f"p {NAMES[v]} vs NSGA-II", f"better: {NAMES[v]} vs NSGA-II"]
  rows, verdicts = [], {v: [] for v in variants}
  ranks = {a: [] for a in algs}
  for name, r in results.items():
    runs = {a: r["runs"][a] for a in algs}
    if not all(runs.values()):
      continue
    hv = {a: [run["hv"][-1] for run in runs[a]] for a in algs}
    igd = {a: [run["igd"][-1] for run in runs[a]] for a in algs}
    first = next(iter(runs.values()))[0]
    row = [PROBLEMS[name][0], first["n"], r["m"], first["budget"], f"{r['hv_pf']:.4f}"]
    row += [f"{np.median(hv[a]):.4f}" for a in algs]
    row += [f"{np.median(igd[a]):.3e}" for a in algs]
    row += [f"{sum(run['feasible'] for run in runs[a])}/{len(runs[a])}" for a in algs]
    row += [f"{sum(v > 0 for v in hv[a])}/{len(runs[a])}" for a in algs]
    # rank of the median final HV among the algorithms (1 = best, ties averaged)
    for a, rk in zip(algs, rankdata([-np.median(hv[a]) for a in algs])):
      ranks[a].append(rk)
    for v in variants:
      if "NSGA-II" not in hv:
        continue
      a, b = hv[v], hv["NSGA-II"]
      p = mannwhitneyu(a, b).pvalue if len(set(a + b)) > 1 else 1.0
      verdict = "=" if p >= 0.05 else (NAMES[v] if np.median(a) > np.median(b) else "NSGA-II")
      verdicts[v].append(verdict)
      row += [f"{p:.3g}", verdict]
    rows.append(row)
  with open(path, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([header] + rows)
  print("wrote", path)
  print(f"\n{'algorithm':40s} {'mean rank':>9s}  better/worse/= vs NSGA-II (p<0.05)")
  for a in sorted(algs, key=lambda a: np.mean(ranks[a])):
    ws = verdicts.get(a)
    vs = (f"{ws.count(NAMES[a])}/{ws.count('NSGA-II')}/{ws.count('=')}" if ws else "-")
    print(f"{NAMES[a]:40s} {np.mean(ranks[a]):9.2f}  {vs}")
  print(f"(mean rank of the median final HV over {len(rows)} problems and "
        f"{len(algs)} algorithms; 1 = best)")


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--data", default=os.path.join(ROOT, "exdata", "cmo"),
                      help="results of run_cmo_benchmark.py (default: exdata/cmo)")
  parser.add_argument("--max-seed", type=int, default=10**6,
                      help="use only seeds 1..N of every algorithm (default: all)")
  parser.add_argument("--algorithms", default=None,
                      help="comma-separated labels to include (default: all in --data)")
  parser.add_argument("--out", default=os.path.join(ROOT, "ppdata", "cmo"),
                      help="output folder (default: ppdata/cmo)")
  args = parser.parse_args()
  if args.algorithms:
    args.algorithms = args.algorithms.split(",")
  os.makedirs(args.out, exist_ok=True)
  plt.rcParams.update({"font.family": "DejaVu Sans"})
  print("computing indicators...", flush=True)
  results = load(args)
  if not results:
    raise SystemExit(f"no results in {args.data}")
  plot_indicator(results, "hv", os.path.join(args.out, "cmo_hv.png"),
                 "Hypervolume of the feasible non-dominated points (median and IQR "
                 "over seeds; dashed: true front)", "normalized hypervolume", log=False)
  plot_indicator(results, "igd", os.path.join(args.out, "cmo_igd.png"),
                 "IGD+ of the feasible non-dominated points to the true front (median "
                 "and IQR over seeds)", "IGD+ (normalized)", log=True)
  plot_ecdf(results, os.path.join(args.out, "cmo_ecdf.png"))
  plot_fronts(results, os.path.join(args.out, "cmo_fronts.png"))
  summary(results, os.path.join(args.out, "cmo_summary.csv"))


if __name__ == "__main__":
  main()
