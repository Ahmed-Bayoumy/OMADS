"""Design history of OMADS and the COCO reference algorithms on one problem.

Plots, for one ``bbob-constrained`` function and dimension, the convergence
of every algorithm: the best precision f - f_opt found so far against the
number of objective evaluations, within OMADS' evaluation budget. Each curve is
the mean over the runs (one per COCO instance) of log10(f - f_opt), that is the
geometric mean, and the shaded band is +/- one standard deviation of
log10(f - f_opt) across the runs.

Usage (from ``exdata/``, after ``reproduce_results.py`` or a preset run):

  python ../benchmarks/coco/plot_design_history.py OMADS-A 43 2 \\
      ../docs_src/_static/f43_2d_design_history.svg
"""
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cocopp

warnings.filterwarnings("ignore")

REFERENCES = {"COBYLA": "bbob-constrained/2022/COBYLA_Dufosse.tgz",
              "AL1-CMA-ES": "bbob-constrained/2022/AL1-CMA-ES_Dufosse.tgz",
              "fmincon": "bbob-constrained/2022/fmincon_Hellwig.tgz",
              "random search": "bbob-constrained/2022/RandomSearch-5_Dufosse.tgz"}
# Reference palette (dataviz skill), validated for these five adjacent series.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FLOOR = 1e-8   # COCO's final target; precisions below are drawn at the floor


def dataset(source, func, dim):
  for ds in cocopp.load(source):
    if ds.funcId == func and ds.dim == dim:
      return ds
  raise SystemExit(f"no data for f{func} in {dim}D in {source}")


def histories(ds, grid):
  """Best precision so far of every run, at each evaluation count of grid."""
  evals, runs = ds.funvals[:, 0], ds.funvals[:, 1:]
  rows = np.searchsorted(evals, grid, side="right") - 1
  out = np.full((len(grid), runs.shape[1]), np.nan)
  out[rows >= 0] = runs[rows[rows >= 0]]
  return np.maximum(out, FLOOR)


def main(omads, func, dim, output):
  func, dim = int(func), int(dim)
  ds_omads = dataset(omads, func, dim)
  budget = int(ds_omads.funvals[-1, 0])
  grid = np.arange(1, budget + 1)
  series = {"OMADS": ds_omads}
  for name, path in REFERENCES.items():
    series[name] = dataset(cocopp.archives.bbob_constrained.get(path), func, dim)

  plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                       "svg.fonttype": "path", "axes.edgecolor": GRID,
                       "axes.labelcolor": INK2, "xtick.color": INK2,
                       "ytick.color": INK2, "text.color": INK})
  fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
  ends = []
  nruns = {}
  for (name, ds), color in zip(series.items(), COLORS):
    logh = np.log10(histories(ds, grid))
    nruns[name] = logh.shape[1]
    mean, std = np.nanmean(logh, axis=1), np.nanstd(logh, axis=1)
    is_omads = name == "OMADS"
    ax.fill_between(grid, 10 ** (mean - std), 10 ** (mean + std), color=color,
                    alpha=0.18 if is_omads else 0.10, lw=0,
                    zorder=1.5 if is_omads else 1)
    ax.plot(grid, 10 ** mean, color=color, lw=2.6 if is_omads else 1.6,
            label=name, zorder=3 if is_omads else 2)
    ends.append((10 ** mean[-1], name))

  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim(1, budget)
  ax.set_xlabel("objective evaluations")
  runs = min(nruns.values())
  ax.set_ylabel(rf"best $f - f_{{\mathrm{{opt}}}}$ so far (mean $\pm$ std, {runs} runs)")
  ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)
  ax.tick_params(length=0)

  # Direct labels at the line ends, spread vertically to avoid overlaps.
  ends.sort()
  logs = [np.log10(v) for v, _ in ends]
  for i in range(1, len(logs)):
    logs[i] = max(logs[i], logs[i - 1] + 0.35)
  colors = dict(zip(series, COLORS))
  for (_, name), y in zip(ends, logs):
    ax.annotate(name, xy=(budget, 10 ** y), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9.5,
                color=INK, fontweight="bold" if name == "OMADS" else "normal")
    ax.plot([budget], [10 ** y], marker="s", ms=5, color=colors[name],
            clip_on=False, zorder=4)
  ax.legend(loc="lower left", frameon=False, fontsize=9)
  ax.set_title(f"bbob-constrained f{func}, {dim}D: equal budget of {budget} evaluations",
               fontsize=10.5, color=INK, loc="left")
  fig.tight_layout()
  fig.savefig(output)
  print(f"wrote {output}")


if __name__ == "__main__":
  if len(sys.argv) != 5:
    raise SystemExit(__doc__)
  main(*sys.argv[1:])
