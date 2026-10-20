"""Post-process OMADS bbob-constrained runs with cocopp against archived references.

Usage (from ``exdata/``, so that the folder names become the legend labels):

  python ../benchmarks/coco/postprocess_bbob_constrained.py ../ppdata/out OMADS-A OMADS-B OMADS-Halton

Besides the cocopp report, the curves of the runtime ECDFs aggregated over all
functions are written to ``<out>/ecdf_data/ecdf_<D>D.json`` (one entry per
algorithm: label, curve and median budget), which ``plot_ecdf.py`` redraws.

Keep the output folder path short: cocopp's file paths exceed the Windows
260-character limit when it is nested deeply.
"""
import json
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import cocopp

matplotlib.rcParams["savefig.bbox"] = "tight"   # keep legends outside the axes in the figure
matplotlib.rcParams["savefig.pad_inches"] = 0.05

out_dir, algs = sys.argv[1], sys.argv[2:]
refs = ["bbob-constrained/2022/COBYLA_Dufosse.tgz",
        "bbob-constrained/2022/AL1-CMA-ES_Dufosse.tgz",
        "bbob-constrained/2022/fmincon_Hellwig.tgz",
        "bbob-constrained/2022/RandomSearch-5_Dufosse.tgz"]
ecdf_dir = os.path.join(out_dir, "ecdf_data")
ECDF_FIGURE = re.compile(r"pprldmany_(\d+)D_noiselessall")
_savefig = matplotlib.figure.Figure.savefig


def ecdf_curves(ax):
  """Curves of a cocopp runtime ECDF axis.

  For each algorithm cocopp draws, in this order: a '.' marker, the ECDF
  curve, its markers, a legend proxy at (-1, -1) carrying the label, and the
  'x' and '+' markers at the median budget.
  """
  lines = ax.get_lines()
  curves = []
  for i, proxy in enumerate(lines):
    label = proxy.get_label()
    if label.startswith("_") or list(proxy.get_xdata()) != [-1]:
      continue
    curve, budget = lines[i - 2], lines[i + 1]
    curves.append({"label": os.path.basename(label).removesuffix(".tgz"),
                   "x": [float(v) for v in curve.get_xdata()],
                   "y": [float(v) for v in curve.get_ydata()],
                   "budget": float(budget.get_xdata()[0])})
  return curves


def savefig(fig, fname, *args, **kwargs):
  """Save as cocopp asks, keeping the data of the aggregated runtime ECDFs."""
  m = ECDF_FIGURE.search(str(fname))
  if m and fig.axes:
    os.makedirs(ecdf_dir, exist_ok=True)
    with open(os.path.join(ecdf_dir, f"ecdf_{int(m.group(1)):02d}D.json"), "w",
              encoding="utf-8") as f:
      json.dump(ecdf_curves(fig.axes[0]), f)
  return _savefig(fig, fname, *args, **kwargs)


matplotlib.figure.Figure.savefig = savefig
cocopp.main(" ".join(["-o", out_dir] + algs + refs))
