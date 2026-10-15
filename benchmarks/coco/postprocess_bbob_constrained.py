"""Post-process OMADS bbob-constrained runs with cocopp against archived references.

Usage (from ``exdata/``, so that the folder names become the legend labels):

  python ../benchmarks/coco/postprocess_bbob_constrained.py ../ppdata/out OMADS-hybrid OMADS-BiTPE

Keep the output folder path short: cocopp's file paths exceed the Windows
260-character limit when it is nested deeply.
"""
import sys
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.bbox"] = "tight"   # keep legends outside the axes in the figure
matplotlib.rcParams["savefig.pad_inches"] = 0.05
import cocopp

out_dir, algs = sys.argv[1], sys.argv[2:]
refs = ["bbob-constrained/2022/COBYLA_Dufosse.tgz",
        "bbob-constrained/2022/AL1-CMA-ES_Dufosse.tgz",
        "bbob-constrained/2022/fmincon_Hellwig.tgz",
        "bbob-constrained/2022/RandomSearch-5_Dufosse.tgz"]
cocopp.main(" ".join(["-o", out_dir] + algs + refs))
