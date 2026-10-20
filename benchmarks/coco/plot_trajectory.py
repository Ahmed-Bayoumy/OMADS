"""Plot the objective trajectory of OMADS runs on one bbob-constrained problem.

For each series, the best precision f - f_opt found so far (COCO adds the sum
of the constraint violations, so an infeasible point counts as worse) is taken
from the COCO data of every run (one per instance and folder) at each number
of objective evaluations. The plot shows the band between the 5th and 95th
percentiles of the runs, their mean and their median.

Usage, from the folder that holds exdata/:

  python benchmarks/coco/plot_trajectory.py exdata/OMADS-A_f1_d10_100x
  python benchmarks/coco/plot_trajectory.py exdata/OMADS-A_f1_d10_s* -o f1_seeds.svg
  python benchmarks/coco/plot_trajectory.py -f 1 -d 10 \\
      -s "preset A" exdata/OMADS-A_f1_d10_100x \\
      -s "ORTHO 2N" exdata/OMADS-A_f1_d10_2n

Positional folders are pooled into one series; ``-s LABEL FOLDER...`` adds a
labelled series (repeatable) to compare configurations. ``-f`` and ``-d`` pick
the problem when the folders hold more than one.
"""
import argparse
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
import cocopp  # noqa: E402

FLOOR = 1e-9   # precisions below are drawn here (COCO's final target is 1e-8)
FINAL_TARGET = 1e-8
COLORS = ["#2a78d6", "#eb6834", "#6250d6", "#1baf7a", "#d55181"]
INK, INK2, AXIS, GRID = "#0b0b0b", "#52514e", "#8a8984", "#ebeae6"


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("folders", nargs="*",
                      help="COCO data folders, pooled into one series")
  parser.add_argument("-s", "--series", nargs="+", action="append", default=[],
                      metavar=("LABEL", "FOLDER"),
                      help="a labelled series: label followed by its folders")
  parser.add_argument("-f", "--function", type=int, default=None,
                      help="function index (default: the only one in the data)")
  parser.add_argument("-d", "--dimension", type=int, default=None,
                      help="dimension (default: the only one in the data)")
  parser.add_argument("--log-x", action="store_true",
                      help="logarithmic evaluation axis (default: linear)")
  parser.add_argument("-o", "--output", default=None,
                      help="figure file, .svg or .png (default:"
                      " trajectory_f<F>_d<D>.svg in the first folder)")
  args = parser.parse_args()
  if args.folders:
    args.series.insert(0, [None] + args.folders)
  if not args.series or any(len(s) < 2 for s in args.series):
    parser.error("give data folders, or -s LABEL FOLDER... for each series")
  return args


def load_runs(folders, func, dim):
  """Data sets of the folders for one problem; picks it when unique."""
  datasets = []
  for folder in folders:
    datasets += list(cocopp.load(folder))
  problems = sorted({(ds.funcId, ds.dim) for ds in datasets})
  if func is None or dim is None:
    candidates = [p for p in problems
                  if (func is None or p[0] == func) and (dim is None or p[1] == dim)]
    if len(candidates) != 1:
      raise SystemExit(f"the data hold {problems}: choose one with -f and -d")
    func, dim = candidates[0]
  datasets = [ds for ds in datasets if (ds.funcId, ds.dim) == (func, dim)]
  if not datasets:
    raise SystemExit(f"no data for f{func} in {dim}-D in {folders}")
  return datasets, func, dim


def histories(datasets, grid):
  """Best precision so far of every run (columns) at each count of grid."""
  columns = []
  for ds in datasets:
    evals, runs = ds.funvals[:, 0], ds.funvals[:, 1:]
    rows = np.searchsorted(evals, grid, side="right") - 1
    h = np.full((len(grid), runs.shape[1]), np.nan)
    h[rows >= 0] = runs[rows[rows >= 0]]
    # after its last record a run keeps its final value
    columns.append(h)
  return np.maximum(np.hstack(columns), FLOOR)


def plot(series, func, dim, output, log_x=False):
  plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
  fig, ax = plt.subplots(figsize=(8.2, 4.8))
  budget = max(ds.funvals[-1, 0] for _, dsl in series for ds in dsl)
  grid = np.unique(np.round(np.logspace(0, np.log10(budget), 400) if log_x
                            else np.linspace(1, budget, 400)))
  ends = []
  for k, (label, datasets) in enumerate(series):
    col = COLORS[k % len(COLORS)]
    h = histories(datasets, grid)
    nruns = h.shape[1]
    lo, med, hi = np.nanpercentile(h, [5, 50, 95], axis=1)
    mean = np.nanmean(h, axis=1)
    name = label or datasets[0].algId.split("/")[-1]
    prefix = f"{name}: " if len(series) > 1 else ""
    ax.fill_between(grid, lo, hi, color=col, alpha=0.16, lw=0, zorder=1,
                    label=f"{prefix}5th–95th percentile ({nruns} runs)")
    ax.plot(grid, mean, color=col, lw=1.6, ls=(0, (4, 2.5)), zorder=3,
            label=f"{prefix}mean")
    ax.plot(grid, med, color=col, lw=2.4, zorder=4, label=f"{prefix}median")
    ends.append((med[-1], mean[-1], name, col))

  ax.axhline(FINAL_TARGET, color=AXIS, lw=1, ls=(0, (1, 2)), zorder=2)
  ax.text(budget, FINAL_TARGET * 1.6, "COCO final target 10⁻⁸", fontsize=8.5,
          color=INK2, ha="right")
  if log_x:
    ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim(1, budget)
  ax.set_ylim(bottom=FLOOR)
  ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  ax.spines["bottom"].set_color(AXIS)
  ax.spines["left"].set_color(AXIS)
  ax.tick_params(colors=INK2, length=0, pad=6)
  ax.set_xlabel("objective evaluations", color=INK2, labelpad=8)
  ax.set_ylabel("best f − f_opt so far  (+ constraint violation)", color=INK2,
                labelpad=8)
  ax.legend(loc="lower left", bbox_to_anchor=(0, 0.06), frameon=False, fontsize=8.5, ncol=1 if len(series) == 1 else 2)
  fig.text(0.08, 0.97, f"bbob-constrained f{func}, {dim}-D: objective trajectory",
           fontsize=12.5, fontweight="bold", color=INK, ha="left", va="top")
  summary = "; ".join(f"{name}: median {m:.2g}, mean {a:.2g} at {int(budget)} evaluations"
                      for m, a, name, _ in ends)
  fig.text(0.08, 0.915, summary, fontsize=8.8, color=INK2, ha="left", va="top")
  fig.subplots_adjust(left=0.1, right=0.97, top=0.85, bottom=0.13)
  fig.savefig(output, facecolor="white", bbox_inches="tight", pad_inches=0.15,
              dpi=160)
  plt.close(fig)
  print(f"wrote {output}")


def main(argv=None):
  args = parse_args() if argv is None else argv
  func, dim = args.function, args.dimension
  series = []
  for label, *folders in args.series:
    datasets, func, dim = load_runs(folders, func, dim)
    series.append((label, datasets))
  output = args.output or os.path.join(args.series[0][1],
                                       f"trajectory_f{func}_d{dim}.svg")
  plot(series, func, dim, output, args.log_x)


if __name__ == "__main__":
  main()
