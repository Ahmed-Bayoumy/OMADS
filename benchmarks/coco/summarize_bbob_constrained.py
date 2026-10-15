"""Summarize OMADS bbob-constrained runs from COCO .info files.

For every (variant, dimension), computes the fraction of the 54 x 15
(function, instance) runs whose final recorded precision (f - f_opt) reached
1e-1, 1e-3 and 1e-6 within the budget, writes a CSV table and a grouped-bar
figure (one panel per precision).

Usage (from the repository root, after the runs and with each variant's batch
folders gathered in one folder per variant under ``exdata/``):

  python benchmarks/coco/summarize_bbob_constrained.py <output-folder> [exdata]
"""
import csv
import math
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
EXDATA = sys.argv[2] if len(sys.argv) > 2 else "exdata"
# (folder under EXDATA, label, shown in the figure); every variant is in the table.
VARIANTS = [("OMADS-A", "mads.main, ACTIVE (A)", True),
            ("OMADS-B", "mads.main, ACTIVE (B)", True),
            ("OMADS-Halton", "mads.main, HALTON search", True),
            ("OMADS-ACTIVE", "mads.main, ACTIVE, ns = (n+1)(n+2)/2 + 10",
             False)]
DIMS = [2, 3, 5, 10]
PRECISIONS = [1e-1, 1e-3, 1e-6]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # reference palette, slots 1-3
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

head_re = re.compile(r"funcId = (\d+), DIM = (\d+)")
run_re = re.compile(r"(\d+):(\d+)\|([-+0-9.eE]+|inf|nan)")


def final_precisions(folder):
  """{(dim, func): {instance: best df}} over all .info files of a variant."""
  out = defaultdict(dict)
  for info in glob.glob(os.path.join(EXDATA, folder, "**", "*.info"), recursive=True):
    key = None
    with open(info, encoding="utf-8") as f:
      for line in f:
        m = head_re.search(line)
        if m:
          key = (int(m.group(2)), int(m.group(1)))
          continue
        if key and line.startswith("data_"):
          for inst, _, df in run_re.findall(line):
            df = float(df)
            runs = out[key]
            runs[int(inst)] = min(df, runs.get(int(inst), float("inf")))
  return out


rows = []
fractions = {}
VARIANTS = [v for v in VARIANTS if os.path.isdir(os.path.join(EXDATA, v[0]))]
for folder, label, _ in VARIANTS:
  data = final_precisions(folder)
  for d in DIMS:
    dfs = [df for (dim, _), runs in data.items() if dim == d for df in runs.values()]
    n = len(dfs)
    fr = [sum(df <= p for df in dfs) / n if n else float("nan") for p in PRECISIONS]
    fractions[(folder, d)] = fr
    rows.append([label, d, n] + [f"{x:.3f}" for x in fr])

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "coco_summary.csv"), "w", newline="", encoding="utf-8") as f:
  w = csv.writer(f)
  w.writerow(["algorithm", "dimension", "runs"] + [f"reached {p:g}" for p in PRECISIONS])
  w.writerows(rows)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "svg.fonttype": "path"})
fig, axes = plt.subplots(1, len(PRECISIONS), figsize=(10.5, 3.6), sharey=True,
                         facecolor=SURFACE)
width = 0.26
for ax, (pi, p) in zip(axes, enumerate(PRECISIONS)):
  shown = [(folder, label) for folder, label, plot in VARIANTS if plot]
  for vi, ((folder, label), color) in enumerate(zip(shown, COLORS)):
    xs = [i + (vi - 1) * width for i in range(len(DIMS))]
    ys = [fractions[(folder, d)][pi] for d in DIMS]
    ax.bar(xs, ys, width=width - 0.02, color=color, label=label, zorder=2,
           edgecolor=SURFACE, linewidth=1)
  ax.set_xticks(range(len(DIMS)), [f"{d}D" for d in DIMS])
  ax.set_title(f"precision $10^{{{int(round(math.log10(p)))}}}$",
               fontsize=10, color=INK)
  ax.set_ylim(0, 1)
  ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
  ax.tick_params(length=0)
  for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
axes[0].set_ylabel("fraction of runs reaching the precision")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(OUT, "coco_summary.svg"))
fig.savefig(os.path.join(OUT, "coco_summary.png"), dpi=150)
for r in rows:
  print(r)
