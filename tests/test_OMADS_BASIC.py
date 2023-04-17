from OMADS import POLL, SEARCH, MADS

import copy
import os
from BMDFO import toy

import pandas as pd
import numpy as np

from typing import Dict, List


def rosen(x, *argv):
  x = np.asarray(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
        axis=0), [0]]
  return y


def test_omads_callable_quick():
  eval = {"blackbox": rosen}
  param = {"baseline": [-2.0, -2.0],
       "lb": [-5, -5],
       "ub": [10, 10],
       "var_names": ["x1", "x2"],
       "scaling": 10.0,
       "post_dir": "./post"}
  sampling = {
                    "method": SEARCH.explore.SAMPLING_METHOD.LH.value,
                    "ns": int((2+1)*(2+2)/2)+50,
                    "visualize": False,
                    "criterion": None
                  }
  options = {"seed": 0, "budget": 100000, "tol": 1e-12, "display": True, "check_cache": True, "store_cache": True, "rich_direction": True,}

  data = {"evaluator": eval, "param": param, "options": options, "sampling": sampling}

  out: Dict = MADS.main(data)
  print(out)

def test_omads_callable_quick_parallel():
  eval = {"blackbox": rosen}
  param = {"baseline": [-2.0, -2.0],
       "lb": [-5, -5],
       "ub": [10, 10],
       "var_names": ["x1", "x2"],
       "scaling": 10.0,
       "post_dir": "./post"}
  options = {"seed": 0, "budget": 1000, "tol": 1e-6, "display": True, "parallel_mode": True}
  sampling = {
                    "method": SEARCH.explore.SAMPLING_METHOD.LH.value,
                    "ns": int((2+1)*(2+2)/2)+50,
                    "visualize": False,
                    "criterion": None
                  }
  data = {"evaluator": eval, "param": param, "options": options, "sampling": sampling}

  out: Dict = MADS.main(data)
  print(out)

def test_omads_toy_quick():
  assert POLL.DType
  assert POLL.Options
  assert POLL.Parameters
  assert POLL.Evaluator
  assert POLL.Point
  assert POLL.OrthoMesh
  assert POLL.Cache
  assert POLL.Dirs2n
  assert POLL.PreMADS
  assert POLL.Output
  assert POLL.PostMADS
  assert POLL.main

  p_file_1 = os.path.abspath("./tests/bm/unconstrained/rosenbrock.json")
  POLL.main(p_file_1)

  p_file_2 = os.path.abspath("./tests/bm/constrained/geom_prog.json")
  POLL.main(p_file_2)

  # p_file_3 = os.path.abspath("./tests/Rosen/param.json")
  # main(p_file_3)

  data = {
    "evaluator":
      {
        "blackbox": "rosenbrock",
        "internal": "uncon",
        "path": os.path.abspath(".\\bm"),
        "input": "input.inp",
        "output": "output.out"},

    "param":
      {
        "baseline": [-2.0, -2.0],
        "lb": [-5, -5],
        "ub": [10, 10],
        "var_names": ["x1", "x2"],
        "scaling": 10.0,
        "post_dir": "./tests/bm/unconstrained/post"
      },

    "options":
      {
        "seed": 0,
        "budget": 100000,
        "tol": 1e-12,
        "psize_init": 1,
        "display": False,
        "opportunistic": False,
        "check_cache": True,
        "store_cache": True,
        "collect_y": False,
        "rich_direction": True,
        "precision": "high",
        "save_results": False,
        "save_coordinates": False,
        "save_all_best": False,
        "parallel_mode": False
      },

    "sampling" : {
                  "method": SEARCH.explore.SAMPLING_METHOD.LH.value,
                  "ns": int((2+1)*(2+2)/2)+50,
                  "visualize": False,
                  "criterion": None
                }
  }

  MADS.main(data)


def test_omads_toy_extended():
  uncon_test_names = ["ackley", "beale", "dixonprice", "griewank", "levy", "michalewicz", "perm", "powell",
            "powersum", "rastrigin", "rosenbrock", "schwefel", "sphere", "trid", "zakharov"]

  con_test_names = ["g1", "g2", "g3", "geom_prog", "himmelblau", "pressure_vessel", "tc_spring",
            "speed_reducer", "wbeam"]

  for name in uncon_test_names:
    POLL.main(os.path.abspath(os.path.join("./tests/bm/unconstrained", name + ".json")))

  for name in con_test_names:
    POLL.main(os.path.abspath(os.path.join("./tests/bm/constrained", name + ".json")))


def test_omads_toy_uncon_bm():
  p_files = []
  runs: int = 2
  # Remove existing BM log files (if any)
  file = os.path.abspath(os.path.join('./tests/bm/unconstrained/post', 'BM_report.csv'))
  if os.path.exists(file) and os.path.isfile(file):
    os.remove(file)

  df = pd.DataFrame(list())
  df.to_csv(file)

  bm: toy.Run = toy.Run(os.path.abspath('./tests/bm/unconstrained/post'))
  bm.test_suite = "uncon"
  bm_root = os.path.abspath('./tests/bm/unconstrained')

  # get BM parameters file names
  for p, _, filename in os.walk(bm_root):
    if p == bm_root:
      p_files = copy.deepcopy(filename)
  ms: bool
  sl: List[int] = []
  if runs > 1:
    sl = list(range(runs))
    ms = True
  else:
    ms = False
  for run in range(runs):
    for i in range(0, len(p_files)):
      try:
        _, file_exe = os.path.splitext(p_files[i])
        print(f"Solving {p_files[i]}: run# {run:.0f}: seed is {sl[run]:.0f}")
        if file_exe == '.json':
          if ms:
            POLL.main(os.path.join(bm_root, p_files[i]), bm, run, sl[run])
          else:
            POLL.main(os.path.join(bm_root, p_files[i]), bm, run)
      except RuntimeError:
        print("An error occured while running" + p_files[i])

  # Show box plot for the BM stats as an indicator
  # for measuring various algorithmic performance
  # bm.BM_statistics()


def test_omads_toy_con_bm():
  p_files = []
  runs: int = 2
  # Remove existing BM log files (if any)
  file = os.path.abspath(os.path.join('./tests/bm/constrained/post', 'BM_report.csv'))
  if os.path.exists(file) and os.path.isfile(file):
    os.remove(file)

  df = pd.DataFrame(list())
  df.to_csv(file)

  bm: toy.Run = toy.Run(os.path.abspath('./tests/bm/constrained/post'))
  bm.test_suite = "con"
  bm_root = os.path.abspath('./tests/bm/constrained')
  # get BM parameters file names
  for p, _, filename in os.walk(bm_root):
    if p == bm_root:
      p_files = copy.deepcopy(filename)

  ms: bool
  sl: List[int] = []
  if runs > 1:
    sl = list(range(runs))
    ms = True
  else:
    ms = False
  for run in range(runs):
    for i in range(0, len(p_files)):
      try:
        _, file_exe = os.path.splitext(p_files[i])
        if file_exe == '.json':
          if ms:
            POLL.main(os.path.join(bm_root, p_files[i]), bm, run, sl[run])
          else:
            POLL.main(os.path.join(bm_root, p_files[i]), bm, run)
      except RuntimeError:
        print("An error occured while running" + p_files[i])
  # Show box plot for the BM stats as an indicator
  # for measuring various algorithmic performance
  # bm.BM_statistics()

# MADS.main(os.path.abspath(os.path.join("./tests/bm/constrained", "tc_spring.json")))
# test_omads_callable_quick()