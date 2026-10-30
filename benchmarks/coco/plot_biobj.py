"""Draw the performance figures of OMADS on COCO ``bbob-biobj`` as PNG files.

Two kinds of figures, both in the hypervolume indicator of COCO (the
difference between the hypervolume of the reference set and that of all the
non-dominated points found so far, in the normalized objective space; lower
is better, negative values beat the reference set):

  ecdf_<D>D.png, ecdf_grid.png   runtime ECDFs aggregated over the 55 functions
                                 and the 58 indicator targets, redrawn from the
                                 curves saved by ``postprocess_bbob_biobj.py``;
  convergence_<D>D.png           median indicator over the instances against
                                 the evaluations, one panel per function, read
                                 with ``cocopp.load`` from the OMADS data and
                                 from the reference algorithms of the archive.

Usage, from the repository root, after the runs and ``postprocess_bbob_biobj.py``:

  python benchmarks/coco/plot_biobj.py --ppdata ppdata/biobj --exdata exdata/OMADS-MO
  python benchmarks/coco/plot_biobj.py --exdata exdata/OMADS-MADS-ns150 exdata/OMADS-MO
"""
import argparse
import json
import os
import re
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

import cocoex
import cocopp

from postprocess_bbob_biobj import REFERENCES

XMAX = 1e6
FLOOR = 1e-6    # indicator values below this (or negative) are drawn at it
# Reference algorithms: (substring of the cocopp label, displayed name, color).
STYLES = [("DMS", "DMS", "#eb6834"),
          ("MultiGLODS", "MultiGLODS", "#d55181"),
          ("NSGA-II", "NSGA-II (archive)", "#1baf7a"),
          ("SMS-EMOA", "SMS-EMOA", "#6250d6"),
          ("HMO-CMA-ES", "HMO-CMA-ES", "#8a5a2b"),
          ("MOTPE", "MOTPE", "#c79a00"),
          ("RS-100", "Random search", "#a3a29c")]
OMADS_COLOR = "#1c5cab"        # poll.main (OMADS-MO)
OMADS_OLD_COLOR = "#86b6ef"    # poll.main, earlier OMADS version (compared with -v2)
OMADS_MADS_COLOR = "#00a6a6"   # mads.main, ACTIVE search (OMADS-MADS-ns<ns>)
OMADS_HALTON_COLOR = "#7b3fa0"  # mads.main, HALTON search (OMADS-MADS-HALTON-ns<ns>)
BBOB_NAMES = {1: "Sphere", 2: "Ellipsoid", 6: "Attr. sector", 8: "Rosenbrock",
              13: "Sharp ridge", 14: "Diff. powers", 15: "Rastrigin",
              17: "Schaffer F7", 20: "Schwefel", 21: "Gallagher"}


def is_omads(label):
  return "OMADS" in label


def style(label):
  """Displayed name, color, line width and z-order of a cocopp label."""
  if is_omads(label):
    base = os.path.basename(label.rstrip("/\\"))
    # "-v2" campaigns ran a later OMADS version; earlier ones are drawn lighter
    new = base.endswith("-v2")
    tag = " (new)" if new else ""
    m = re.search(r"OMADS-MADS-(?:([A-Z]+)-)?ns(\d+)", base)
    if m:
      method = m.group(1) or "ACTIVE"
      color = OMADS_HALTON_COLOR if method == "HALTON" else OMADS_MADS_COLOR
      return f"OMADS mads {method} (ns={m.group(2)}){tag}", color, 2.4, 6
    if base.startswith("OMADS-MO"):
      if new:
        return "OMADS poll (new)", OMADS_COLOR, 2.6, 7
      return "OMADS poll", OMADS_OLD_COLOR, 1.6, 5
    return base, OMADS_COLOR, 2.4, 5
  if label.startswith("best"):   # cocopp's virtual best algorithm of a year
    return label + " (virtual)", "#0b0b0b", 0.9, 4
  for key, name, color in STYLES:
    if key in label:
      return name, color, 1.2, 3
  return os.path.basename(label), "#52514e", 1.0, 2


def ecdf_axes(ax, curves, dim):
  """One runtime ECDF panel; returns the OMADS budget (evaluations / n)."""
  budget = max((c["budget"] for c in curves if is_omads(c["label"])), default=None)
  if budget:
    ax.axvspan(1, budget, color=OMADS_COLOR, alpha=0.08, lw=0, zorder=0)
  for c in curves:
    name, color, lw, z = style(c["label"])
    x, y = np.asarray(c["x"]), np.asarray(c["y"])
    run = x <= c["budget"]
    ls = "--" if c["label"].startswith("best") else "-"
    ax.plot(x[run], y[run], color=color, lw=lw, ls=ls, zorder=z, label=name)
    if (~run).any():   # cocopp's simulated restarts beyond the median budget
      ax.plot(x[~run], y[~run], color=color, lw=lw * 0.7, ls=":", zorder=z - 0.5)
  ax.set_xscale("log")
  ax.set_xlim(1, XMAX)
  ax.set_ylim(0, 1)
  ax.xaxis.set_major_locator(FixedLocator([1, 1e2, 1e4, 1e6]))
  ax.set_xticklabels(["1", "$10^2$", "$10^4$", "$10^6$"])
  ax.xaxis.set_minor_locator(NullLocator())
  ax.grid(True, color="#ebeae6", lw=0.6)
  ax.set_axisbelow(True)
  ax.set_title(f"{dim}-D", fontsize=10)
  return budget


def value_at(curve, x):
  xs = np.asarray(curve["x"])
  i = np.searchsorted(xs, x, side="right") - 1
  return curve["y"][i] if i >= 0 else 0.0


def plot_ecdfs(args):
  dims = [d for d in args.dimensions if os.path.exists(
      os.path.join(args.ppdata, "ecdf_data", f"ecdf_{d:02d}D.json"))]
  data = {}
  for d in dims:
    with open(os.path.join(args.ppdata, "ecdf_data", f"ecdf_{d:02d}D.json"),
              encoding="utf-8") as f:
      data[d] = json.load(f)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    budget = ecdf_axes(ax, data[d], d)
    ax.set_xlabel("evaluations / n")
    ax.set_ylabel("fraction of (function, target) pairs")
    ax.set_title(f"bbob-biobj, {d}-D: runtime ECDF of the hypervolume indicator, "
                 "55 functions, 58 targets", fontsize=10)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(os.path.join(args.out, f"ecdf_{d:02d}D.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{d}-D ECDF (fraction of pairs solved at the OMADS budget, {budget:.0f}"
          " evaluations / n, and at the end of each curve):")
    for c in data[d]:
      name = style(c["label"])[0]
      print(f"  {name:<14} {value_at(c, budget):6.1%}   {c['y'][-1]:6.1%}")
  if not dims:
    return
  cols = 2 if len(dims) > 1 else 1
  rows = -(-len(dims) // cols)
  fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows),
                           sharex=True, sharey=True, squeeze=False)
  for ax, d in zip(axes.flat, dims):
    ecdf_axes(ax, data[d], d)
  for ax in list(axes.flat)[len(dims):]:
    ax.axis("off")
  for ax in axes[-1]:
    ax.set_xlabel("evaluations / n")
  fig.supylabel("fraction of (function, target) pairs", fontsize=10)
  handles, labels = axes[0, 0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
  fig.suptitle("OMADS on bbob-biobj: runtime ECDFs of the hypervolume indicator "
               "(shaded: OMADS budget; dotted: simulated restarts)", fontsize=10)
  fig.tight_layout(rect=(0, 0.07, 1, 0.97))
  fig.savefig(os.path.join(args.out, "ecdf_grid.png"), dpi=200)
  plt.close(fig)


def median_trajectory(ds, xgrid):
  """Median over instances of the best indicator value after x evaluations / n."""
  fv = ds.funvals
  evals = fv[:, 0] / ds.dim
  runs = fv[:, 1:].copy()
  for j in range(runs.shape[1]):   # carry the last value past the end of a run
    col = runs[:, j]
    ok = ~np.isnan(col)
    if ok.any():
      last = np.where(ok)[0][-1]
      col[last:] = col[last]
  idx = np.searchsorted(evals, xgrid, side="right") - 1
  valid = (idx >= 0) & (xgrid <= ds.maxevals.max() / ds.dim)
  med = np.full(xgrid.shape, np.nan)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    med[valid] = np.nanmedian(runs[idx[valid]], axis=1)
  return med


def function_titles():
  """bbob-biobj function -> 'f<k> A/B' from the names of its bbob components."""
  suite = cocoex.Suite("bbob-biobj", "", "dimensions: 2 instance_indices: 1")
  titles = {}
  for fid in range(1, 56):
    p = suite.get_problem_by_function_dimension_instance(fid, 2, 1)
    a, b = (int(part.split("_")[1][1:]) for part in p.name.split("__"))
    titles[fid] = f"f{fid} {BBOB_NAMES[a]}/{BBOB_NAMES[b]}"
  return titles


def trajectories(path, xgrid, cache=None):
  """{(algId, dim, funcId): median trajectory} of a COCO data set; the
  reference archives are cached in ``cache`` (loading them takes minutes)."""
  if cache and os.path.exists(cache):
    with open(cache, encoding="utf-8") as f:
      return {tuple(json.loads(k)): np.array(v, dtype=float)
              for k, v in json.load(f).items()}
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    out = {(ds.algId, ds.dim, ds.funcId): median_trajectory(ds, xgrid)
           for ds in cocopp.load(path)}
  if cache:
    with open(cache, "w", encoding="utf-8") as f:
      json.dump({json.dumps(list(k)): [None if np.isnan(v) else float(v) for v in t]
                 for k, t in out.items()}, f)
  return out


def plot_convergence(args):
  xgrid = np.logspace(0, np.log10(XMAX), 300)
  cache_dir = os.path.join(os.path.expanduser("~"), ".cocopp", "omads_trajectories")
  os.makedirs(cache_dir, exist_ok=True)
  data = {}
  for path in args.exdata:
    data.update(trajectories(path, xgrid))
  for ref in REFERENCES:
    data.update(trajectories(ref, xgrid, os.path.join(
        cache_dir, os.path.basename(ref).replace(".tgz", ".json"))))
  titles = function_titles()
  for d in args.dimensions:
    if not any(is_omads(k[0]) and k[1] == d for k in data):
      continue
    fig, axes = plt.subplots(11, 5, figsize=(15, 28), sharex=True, sharey=True)
    for fid, ax in zip(range(1, 56), axes.flat):
      for key in sorted((k for k in data if k[1] == d and k[2] == fid),
                        key=lambda k: is_omads(k[0])):
        name, color, lw, z = style(key[0])
        ax.plot(xgrid, np.maximum(data[key], FLOOR), color=color, lw=lw * 0.8,
                zorder=z, label=name)
      ax.axvline(args.budget, color=OMADS_COLOR, lw=0.6, ls="--", alpha=0.6)
      ax.set_xscale("log")
      ax.set_yscale("log")
      ax.set_xlim(1, XMAX)
      ax.set_ylim(FLOOR, 2)
      ax.set_title(titles[fid], fontsize=8)
      ax.grid(True, color="#ebeae6", lw=0.5)
      ax.tick_params(labelsize=7)
    handles, labels = {}, {}
    for ax in axes.flat:
      for h, lab in zip(*ax.get_legend_handles_labels()):
        handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=8,
               fontsize=10, frameon=False)
    fig.supxlabel("evaluations / n", y=0.025)
    fig.supylabel("median hypervolume indicator (reference HV $-$ HV), floored at "
                  f"{FLOOR:g}")
    fig.suptitle(f"bbob-biobj, {d}-D: hypervolume indicator vs. evaluations "
                 "(median over instances; dashed: OMADS budget)", fontsize=13)
    fig.tight_layout(rect=(0.01, 0.035, 1, 0.985))
    path = os.path.join(args.out, f"convergence_{d:02d}D.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--ppdata", default=os.path.join("ppdata", "biobj"),
                      help="output folder of postprocess_bbob_biobj.py")
  parser.add_argument("--exdata", nargs="+", default=[os.path.join("exdata", "OMADS-MO")],
                      help="COCO data folders of OMADS campaigns (default: exdata/OMADS-MO)")
  parser.add_argument("--dimensions", "-d", default="2,3,5,10",
                      help="dimensions to draw (default: 2,3,5,10)")
  parser.add_argument("--budget", type=float, default=1000,
                      help="OMADS budget in evaluations / n (default: 1000)")
  parser.add_argument("--out", default=None,
                      help="output folder (default: <ppdata>/figures)")
  args = parser.parse_args()
  args.dimensions = [int(v) for v in args.dimensions.split(",")]
  args.out = args.out or os.path.join(args.ppdata, "figures")
  os.makedirs(args.out, exist_ok=True)
  plt.rcParams.update({"font.family": "DejaVu Sans"})
  plot_ecdfs(args)
  plot_convergence(args)


if __name__ == "__main__":
  main()
