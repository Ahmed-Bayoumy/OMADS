from OMADS import POLL, SEARCH, MADS

import copy
import os
from BMDFO import toy

import pandas as pd
import numpy as np

from typing import Dict, List
from multiprocessing import freeze_support


def rosen(x, *argv):
  x = np.asarray(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
        axis=0), [0]]
  return y

def test_MADS_callable_quick_40d():
  d = 40
  eval = {"blackbox": rosen}
  param = {"name": "RB","baseline": [-2.5]*d,
       "lb": [-5]*d,
       "ub": [10]*d,
       "var_names": [f"x{i}" for i in range(d)],
       "scaling": [15.0]*d,
       "post_dir": "./post"}
  sampling = {
              "method": 'ACTIVE',
              "ns": int((d+1)*(d+2)/2)+50,
              "visualize": False,
              "criterion": None
            }
  options = {"seed": 10000, "budget": 20000, "tol": 1e-9, "display": False, "check_cache": True, "store_cache": True, "rich_direction": True, "opportunistic": False, "save_results": False, "isVerbose": False}
  search = {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": int((d+1)*(d+2)/2)+50,
      "visualize": False
    }
  sampling = {
              "method": 'sampling',
              "ns": int((d+1)*(d+2)/2),
              "visualize": False,
              "criterion": None
            }
  data = {"evaluator": eval, "param": param, "options": options, "sampling": sampling, "search": search}
  outS: Dict = SEARCH.main(data)

  if (outS[0]["fmin"] > 0.0006):
    raise ValueError(f"Search: fmin > {0.0006}")
  
  outP: Dict = POLL.main(data)
  if (outP[0]["fmin"] > 2.7):
    raise ValueError(f"POLL: fmin > {2.7}")
  
  outM: Dict = MADS.main(data)
  if (outM[0]["fmin"] > 0.0006):
    raise ValueError(f"MADS: fmin > {0.0006}")

def test_omads_toy_extended():
  uncon_test_names = ["ackley", "beale", "dixonprice", "griewank", "levy", "michalewicz", "perm", "powell",
            "powersum", "rastrigin", "rosenbrock", "schwefel", "sphere", "trid", "zakharov"]

  con_test_names = ["g1", "g2", "g3", "geom_prog", "himmelblau", "pressure_vessel", "tc_spring",
            "speed_reducer", "wbeam"]

  for name in uncon_test_names:
    POLL.main(os.path.abspath(os.path.join("./tests/bm/unconstrained", name + ".json")))
    SEARCH.main(os.path.abspath(os.path.join("./tests/bm/unconstrained", name + ".json")))
    MADS.main(os.path.abspath(os.path.join("./tests/bm/unconstrained", name + ".json")))

  for name in con_test_names:
    POLL.main(os.path.abspath(os.path.join("./tests/bm/constrained", name + ".json")))
    SEARCH.main(os.path.abspath(os.path.join("./tests/bm/constrained", name + ".json")))
    MADS.main(os.path.abspath(os.path.join("./tests/bm/constrained", name + ".json")))

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
            MADS.main(os.path.join(bm_root, p_files[i]), bm, run, sl[run])
          else:
            MADS.main(os.path.join(bm_root, p_files[i]), bm, run)
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

def test_omads_toy_con_GP():
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

  outP = POLL.main(os.path.join(bm_root, 'geom_prog.json'), bm, 0)

  if (outP[0]["fmin"] > 23.5):
    raise ValueError(f"Search: fmin > {23.5}")
  
  outM = MADS.main(os.path.join(bm_root, 'geom_prog.json'), bm, 0)

  if (outM[0]["fmin"] > 17.7):
    raise ValueError(f"Search: fmin > {17.7}")
  
  outS = SEARCH.main(os.path.join(bm_root, 'geom_prog.json'), bm, 0)

  if (outS[0]["fmin"] > 17.7):
    raise ValueError(f"Search: fmin > {17.7}")

def test_omads_toy_con_WB():
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

  outP = POLL.main(os.path.join(bm_root, 'wbeam.json'), bm, 0)

  if (outP[0]["fmin"] > 2.42):
    raise ValueError(f"Search: fmin > {2.42}")
  
  outM = MADS.main(os.path.join(bm_root, 'wbeam.json'), bm, 0)

  if (outM[0]["fmin"] > 2.21):
    raise ValueError(f"Search: fmin > {2.21}")
  
  outS = SEARCH.main(os.path.join(bm_root, 'wbeam.json'), bm, 0)

  if (outS[0]["fmin"] > 2.81):
    raise ValueError(f"Search: fmin > {2.81}")


if __name__ == "__main__":
  test_MADS_callable_quick_40d()
  test_omads_toy_con_GP()
  test_omads_toy_extended()
  test_omads_toy_uncon_bm()
  test_omads_toy_con_bm()
  test_omads_toy_con_WB()
  freeze_support()
