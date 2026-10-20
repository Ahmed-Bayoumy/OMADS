"""Draw the runtime ECDFs of OMADS against the COCO reference algorithms.

Reads the curves written by ``postprocess_bbob_constrained.py`` to
``<ppdata>/ecdf_data/ecdf_<D>D.json`` and writes, for each dimension, a light
and a dark figure ``ecdf_<D>D.svg`` and ``ecdf_<D>D_dark.svg``.

OMADS runs with a budget of 100 n evaluations, the reference algorithms with up
to 10^7 n, so a single number per curve either flatters OMADS (the final value
includes cocopp's simulated restarts) or hides its early progress. Each curve is
therefore labelled with its value at the OMADS budget and at 10^7 n, and drawn
solid over its recorded runs and dotted where cocopp simulates restarts.

Usage:

  python benchmarks/coco/plot_ecdf.py <ecdf_data folder> <output folder> [dims...]
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

# (substring of the cocopp label, displayed name, style key); drawing order.
ALGORITHMS = [("OMADS-A", "OMADS-A", "omadsA"),
              ("OMADS-B", "OMADS-B", "omadsB"),
              ("OMADS-Halton", "OMADS-Halton", "omadsH"),
              ("COBYLA", "COBYLA", "cobyla"),
              ("AL1-CMA-ES", "AL1-CMA-ES", "cma"),
              ("fmincon", "fmincon", "fmincon"),
              ("RandomSearch", "Random search", "random")]
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", axis="#8a8984",
                  grid="#ebeae6", band="#2a78d6", band_alpha=0.08,
                  omadsA="#1c5cab", omadsB="#3987e5", omadsH="#86b6ef",
                  cobyla="#eb6834", cma="#6250d6", fmincon="#d55181", random="#a3a29c"),
    "dark": dict(surface="#0d1117", ink="#f0efec", ink2="#c3c2b7", axis="#8f8e88",
                 grid="#262b33", band="#3987e5", band_alpha=0.14,
                 omadsA="#5598e7", omadsB="#86b6ef", omadsH="#b7d3f6",
                 cobyla="#d95926", cma="#9085e9", fmincon="#d55181", random="#6b6a66"),
}
WIDTH = {"omadsA": 2.6, "omadsB": 1.7, "omadsH": 1.7}
XMAX = 1e7
XTICKS = {1: "1", 10: "10", 100: "100", 1e3: "1k", 1e4: "10k", 1e5: "100k",
          1e6: "1M", 1e7: "10M"}


def load(path):
  """Curves of the known algorithms, in drawing order."""
  with open(path, encoding="utf-8") as f:
    entries = json.load(f)
  curves = []
  for key, name, style in ALGORITHMS:
    entry = next((e for e in entries if key in e["label"]), None)
    if entry:
      if style.startswith("omads"):   # keep variants apart, e.g. OMADS-A-search
        name = entry["label"]
      curves.append(dict(name=name, style=style, budget=entry["budget"],
                         x=np.array(entry["x"]), y=np.array(entry["y"])))
  return curves


def value_at(curve, x):
  """ECDF value of a curve after x evaluations per variable."""
  i = np.searchsorted(curve["x"], x, side="right") - 1
  return curve["y"][i] if i >= 0 else 0.


def resample(x, y, n=800):
  """Step-hold the monotone ECDF on a log grid to keep the SVG small."""
  grid = np.logspace(np.log10(max(x.min(), 0.5)), np.log10(min(x.max(), XMAX)), n)
  idx = np.searchsorted(x, grid, side="right") - 1
  keep = idx >= 0
  return grid[keep], y[idx[keep]]


def spread(ys, gap):
  """Move label positions apart by at least gap, within [0, 1]."""
  order = np.argsort(ys)
  pos = np.array(ys, float)[order]
  for k in range(1, len(pos)):
    pos[k] = max(pos[k], pos[k - 1] + gap)
  if pos[-1] > 1.:
    pos[-1] = 1.
    for k in range(len(pos) - 2, -1, -1):
      pos[k] = min(pos[k], pos[k + 1] - gap)
  out = np.empty_like(pos)
  out[order] = pos
  return out


def draw(curves, dim, theme, path):
  c = THEMES[theme]
  plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
  fig, ax = plt.subplots(figsize=(9.2, 4.9))
  fig.patch.set_facecolor(c["surface"])
  ax.set_facecolor(c["surface"])

  # OMADS budget: 100 n objective evaluations; on the axis COCO adds the
  # constraint evaluations, hence about 200 n.
  budget = max(cv["budget"] for cv in curves if cv["style"].startswith("omads"))
  ax.axvspan(1, budget, color=c["band"], alpha=c["band_alpha"], lw=0, zorder=0)
  ax.text(1.35, 0.975, "OMADS budget\n100·n objective evaluations", ha="left", va="top",
          fontsize=8.5, color=c["ink2"], linespacing=1.3)

  for cv in curves[::-1]:   # OMADS last, on top
    x, y = resample(cv["x"], cv["y"])
    col, lw = c[cv["style"]], WIDTH.get(cv["style"], 1.4)
    z = 5 if cv["style"] == "omadsA" else 4 if cv["style"].startswith("omads") else 3
    run = x <= cv["budget"]
    yb = value_at(cv, cv["budget"])
    ax.plot(np.r_[x[run], cv["budget"]], np.r_[y[run], yb], color=col, lw=lw, zorder=z,
            solid_capstyle="round")
    ext = x > cv["budget"]
    if ext.any():   # beyond the median budget cocopp simulates restarts
      ax.plot(np.r_[cv["budget"], x[ext]], np.r_[yb, y[ext]], color=col, lw=lw * 0.8,
              ls=(0, (1.2, 1.8)), dash_capstyle="round", zorder=z - 0.5)
      ax.plot([cv["budget"]], [yb], "o", ms=5.5, mfc=col, mec=c["surface"], mew=1.4,
              zorder=z + 1)

  # labels at the right edge: value at the OMADS budget, at 10^7 n, name
  tr = ax.get_yaxis_transform()   # x in axes fraction, y in data
  ends = [value_at(cv, XMAX) for cv in curves]
  for cv, yend, ly in zip(curves, ends, spread(ends, 0.064)):
    bold = cv["style"] == "omadsA"
    weight = "bold" if bold else "normal"
    ax.plot([1., 1.03, 1.05], [yend, ly, ly], color=c[cv["style"]], lw=2. if bold else 1.1,
            transform=tr, clip_on=False, solid_capstyle="round")
    ax.text(1.135, ly, f"{value_at(cv, budget):.0%}", ha="right", va="center",
            fontsize=9.5, color=c["ink2"], fontweight=weight, transform=tr)
    ax.text(1.225, ly, f"{yend:.0%}", ha="right", va="center", fontsize=9.5,
            color=c["ink"], fontweight=weight, transform=tr)
    ax.text(1.25, ly, cv["name"], ha="left", va="center", fontsize=9.5,
            color=c["ink"] if bold else c["ink2"], fontweight=weight, transform=tr)
  for xh, head in ((1.135, "at OMADS\nbudget"), (1.225, "at\n10⁷·n")):
    ax.text(xh, 1.035, head, ha="right", va="bottom", fontsize=8, color=c["ink2"],
            linespacing=1.2, transform=tr)

  ax.set_xscale("log")
  ax.set_xlim(1, XMAX)
  ax.set_ylim(0, 1)
  ax.xaxis.set_major_locator(FixedLocator(list(XTICKS)))
  ax.set_xticklabels(list(XTICKS.values()))
  ax.xaxis.set_minor_locator(NullLocator())
  ticks = np.linspace(0, 1, 6)
  ax.set_yticks(ticks)
  ax.set_yticklabels([f"{v:.0%}" for v in ticks])
  ax.grid(True, color=c["grid"], lw=0.8)
  ax.set_axisbelow(True)
  for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
  ax.spines["bottom"].set_color(c["axis"])
  ax.tick_params(colors=c["ink2"], length=0, pad=6)
  ax.set_xlabel("Evaluations per variable  (objective + constraint evaluations, ÷ n)",
                color=c["ink2"], labelpad=8)
  ax.set_ylabel("(function, target) pairs solved", color=c["ink2"], labelpad=8)

  fig.text(0.085, 0.965, f"COCO bbob-constrained · 54 functions · {dim}-D",
           fontsize=13, fontweight="bold", color=c["ink"], ha="left", va="top")
  fig.text(0.085, 0.905, "Runtime ECDF over 41 targets from 10² to 10⁻⁶.  Solid: recorded "
           "runs, up to each solver's median budget (●).\nDotted: cocopp's simulated "
           "restarts beyond it.  OMADS used 100·n evaluations; the others up to 10⁷·n.",
           fontsize=8.8, color=c["ink2"], ha="left", va="top", linespacing=1.4)
  fig.subplots_adjust(left=0.085, right=0.72, top=0.75, bottom=0.14)
  fig.savefig(path, facecolor=c["surface"], bbox_inches="tight", pad_inches=0.15)
  plt.close(fig)


def main():
  data, out = sys.argv[1], sys.argv[2]
  dims = sys.argv[3:] or ["02", "03", "05", "10"]
  os.makedirs(out, exist_ok=True)
  for d in dims:
    curves = load(os.path.join(data, f"ecdf_{d}D.json"))
    if not any(cv["style"].startswith("omads") for cv in curves):
      print(f"no OMADS data in {d}D, skipped")
      continue
    for theme in THEMES:
      suffix = "" if theme == "light" else "_dark"
      draw(curves, int(d), theme, os.path.join(out, f"ecdf_{d}D{suffix}.svg"))


if __name__ == "__main__":
  main()
