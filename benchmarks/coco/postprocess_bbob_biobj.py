"""Post-process OMADS bbob-biobj runs with cocopp against archived references.

Usage (from ``exdata/``, so that the folder names become the legend labels):

  python ../benchmarks/coco/postprocess_bbob_biobj.py ../ppdata/biobj OMADS-MO

Besides the cocopp report, the curves of the runtime ECDFs aggregated over all
functions are written to ``<out>/ecdf_data/ecdf_<D>D.json`` (one entry per
algorithm: label, curve and median budget), which ``plot_biobj.py`` redraws.
The runtime is the number of evaluations to reach a target of the
hypervolume indicator (the difference between the hypervolume of a reference
set and that of the non-dominated points found so far, in the normalized
objective space).

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

matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.pad_inches"] = 0.05

# Reference algorithms from the COCO data archive: direct-search methods
# closest to MADS (DMS, MultiGLODS), classical and state-of-the-art EMO
# algorithms, a TPE-based method and random search.
REFERENCES = ["bbob-biobj/2021/DMS_Brockhoff_bbob-biobj.tgz",
              "bbob-biobj/2021/MultiGLODS_Brockhoff_bbob-biobj.tgz",
              "bbob-biobj/2016/NSGA-II-MATLAB_Auger_bbob-biobj.tgz",
              "bbob-biobj/2016/SMS-EMOA-PM_Auger_bbob-biobj.tgz",
              "bbob-biobj/2016/HMO-CMA-ES_Loshchilov_bbob-biobj.tgz",
              "bbob-biobj/2022/MOTPE_Tanabe.tgz",
              "bbob-biobj/2016/RS-100.tgz"]
ECDF_FIGURE = re.compile(r"pprldmany_(\d+)D_[a-z]*all")
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


def main():
  out_dir, algs = sys.argv[1], sys.argv[2:]
  ecdf_dir = os.path.join(out_dir, "ecdf_data")

  def savefig(fig, fname, *args, **kwargs):
    """Save as cocopp asks, keeping the data of the aggregated runtime ECDFs."""
    m = ECDF_FIGURE.search(os.path.basename(str(fname)))
    if m and fig.axes:
      os.makedirs(ecdf_dir, exist_ok=True)
      with open(os.path.join(ecdf_dir, f"ecdf_{int(m.group(1)):02d}D.json"), "w",
                encoding="utf-8") as f:
        json.dump(ecdf_curves(fig.axes[0]), f)
    return _savefig(fig, fname, *args, **kwargs)

  matplotlib.figure.Figure.savefig = savefig
  cocopp.main(" ".join(["-o", out_dir] + algs + REFERENCES))


if __name__ == "__main__":
  main()
