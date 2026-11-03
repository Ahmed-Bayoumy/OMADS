"""Reproduce the table of runs on two-dimensional test functions (paper, Table 8).

Runs ``mads.main`` with seed 0 on Rosenbrock, Alpine and Ackley 3 (ACTIVE
sampling search, budget 1,000) and on Eggholder (Latin hypercube search with 30
samples per step, budget 2,000), with the settings of the tutorial scripts in
``docs_src/tutorials/scripts``, and prints the best value and the number of
evaluations of each run. Takes about a minute.

Usage, from the repository root:

  python benchmarks/toy/run_toy_problems.py
"""
import contextlib
import logging
import os
import sys
import tempfile
import warnings

import numpy as np


def rosenbrock(x):
  x = np.asarray(x)
  return [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0), [0]]


def alpine(x):
  return [sum(abs(xi * np.sin(xi) + 0.1 * xi) for xi in x), [0]]


def ackley3(x):
  return [-200.0 * np.exp(-0.2 * np.sqrt(x[0] ** 2 + x[1] ** 2))
          + 5.0 * np.exp(np.cos(3.0 * x[0]) + np.sin(3.0 * x[1])), [0]]


def eggholder(x):
  a, b = x
  return [-(b + 47.0) * np.sin(np.sqrt(abs(a / 2.0 + (b + 47.0))))
          - a * np.sin(np.sqrt(abs(a - (b + 47.0)))), [0]]


# name, blackbox, global minimum, start, lower and upper bounds, scaling, budget, search
PROBLEMS = [
    ("Rosenbrock", rosenbrock, 0.0, [-2.0, -2.0], [-5, -5], [10, 10], [15.0, 15.0], 1000,
     {"type": "sampling", "s_method": "ACTIVE", "ns": 10}),
    ("Alpine", alpine, 0.0, [5.0, 5.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0], 1000,
     {"type": "sampling", "s_method": "ACTIVE", "ns": 10}),
    ("Ackley 3", ackley3, -186.4112, [5.0, 5.0], [-32.0, -32.0], [32.0, 32.0],
     [64.0, 64.0], 1000, {"type": "sampling", "s_method": "ACTIVE", "ns": 10}),
    ("Eggholder", eggholder, -959.6407, [50.0, 280.0], [-512.0, -512.0], [512.0, 512.0],
     [1024.0, 1024.0], 2000, {"type": "sampling", "s_method": "LH", "ns": 30}),
]


def run(name, f, x0, lb, ub, scaling, budget, search):
  from OMADS import mads
  with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    data = {
        "evaluator": {"blackbox": f},
        "param": {"name": name.lower().replace(" ", ""), "baseline": x0, "lb": lb,
                  "ub": ub, "scaling": scaling, "post_dir": tmp},
        "options": {"seed": 0, "budget": budget, "tol": 1e-9, "check_cache": True,
                    "store_cache": True, "rich_direction": True, "display": False,
                    "save_results": False},
        "search": dict(search),
    }
    stderr = sys.stderr
    try:
      with open(os.devnull, "w", encoding="utf-8") as devnull, \
              contextlib.redirect_stdout(devnull):
        out = mads.main(data)[0]
    finally:
      sys.stderr = stderr   # the solvers wrap stderr
      root = logging.getLogger()   # OMADS keeps OMADS.log open; close it
      for handler in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
        root.removeHandler(handler)
        handler.close()
  return out


def main():
  warnings.filterwarnings("ignore")
  print(f"{'Function':<12}{'Global minimum':>16}{'OMADS fmin':>16}{'Evaluations':>13}  Search")
  for name, f, fstar, x0, lb, ub, scaling, budget, search in PROBLEMS:
    out = run(name, f, x0, lb, ub, scaling, budget, search)
    label = search["s_method"] + (f", ns = {search['ns']}" if search["s_method"] != "ACTIVE" else "")
    print(f"{name:<12}{fstar:>16.4f}{float(out['fmin'][0]):>16.7g}{out['nbb_evals']:>13d}  {label}")


if __name__ == "__main__":
  main()
