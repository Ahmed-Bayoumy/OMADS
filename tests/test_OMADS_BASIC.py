from OMADS import POLL, SEARCH, MADS

import copy
import os
from BMDFO import toy

import pandas as pd
import numpy as np

from typing import Dict, List
from multiprocessing import freeze_support
import platform


def rosen(x, *argv):
  x = np.asarray(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
        axis=0), [0]]
  return y

def test_MADS_callable_quick_2d():
  d = 2
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
  options = {"seed": 10000, "budget": 10000, "tol": 1e-9, "display": False, "check_cache": True, "store_cache": True, "rich_direction": True, "opportunistic": True, "save_results": False, "isVerbose": False}
  search = {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": int((d+1)*(d+2)/2)+50,
      "visualize": False
    }
  data = {"evaluator": eval, "param": param, "options": options, "sampling": sampling, "search": search}

  outM: Dict = MADS.main(data)
  outP: Dict = POLL.main(data)
  outS: Dict = SEARCH.main(data)
  if (outM[0]["fmin"] > 0.0006):
    raise ValueError(f"MADS: fmin > {0.0006}")
  
  if (outP[0]["fmin"] > 0.0006):
    raise ValueError(f"POLL: fmin > {0.0006}")
  
  if (outS[0]["fmin"] > 0.0006):
    raise ValueError(f"Search: fmin > {0.0006}")

def test_MADS_callable_quick_10d():
  d = 10
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

  options = {"seed": 10000, "budget": 10000, "tol": 1e-9, "display": False, "check_cache": True, "store_cache": True, "rich_direction": True, "opportunistic": False, "save_results": False, "isVerbose": False}
  search = {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": int((d+1)*(d+2)/2)+50,
      "visualize": False
    }
  sampling = {
              "method": 'sampling',
              "ns": int((d+1)*(d+2)/2)+50,
              "visualize": False,
              "criterion": None
            }
  data = {"evaluator": eval, "param": param, "options": options, "sampling": sampling, "search": search}
  outS: Dict = SEARCH.main(data)

  if (outS[0]["fmin"] > 0.0006):
    raise ValueError(f"Search: fmin > {0.0006}")
  
  outP: Dict = POLL.main(data)
  if (outP[0]["fmin"] > 0.25):
    raise ValueError(f"POLL: fmin > {0.25}")
  
  outM: Dict = MADS.main(data)
  if (outM[0]["fmin"] > 0.0006):
    raise ValueError(f"MADS: fmin > {0.0006}")

def test_MADS_callable_quick_20d():
  d = 20
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
  options = {"seed": 10000, "budget": 10000, "tol": 1e-9, "display": False, "check_cache": True, "store_cache": True, "rich_direction": True, "opportunistic": False, "save_results": False, "isVerbose": False}
  search = {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": int((d+1)*(d+2)/2)+50,
      "visualize": False
    }
  sampling = {
              "method": 'sampling',
              "ns": int((d+1)*(d+2)/2)+50,
              "visualize": False,
              "criterion": None
            }
  data = {"evaluator": eval, "param": param, "options": options, "sampling": sampling, "search": search}
  outS: Dict = SEARCH.main(data)

  if (outS[0]["fmin"] > 0.0006 and platform.platform().split('-')[0] == 'Windows'):
    raise ValueError(f"Search: fmin > {0.0006}")
  
  outP: Dict = POLL.main(data)
  if (outP[0]["fmin"] > 2.7 and platform.platform().split('-')[0] == 'Windows'):
    raise ValueError(f"POLL: fmin > {2.7}")
  
  outM: Dict = MADS.main(data)
  if (outM[0]["fmin"] > 0.0006 and platform.platform().split('-')[0] == 'Windows'):
    raise ValueError(f"MADS: fmin > {0.0006}")

def test_omads_callable_quick_parallel():
  eval = {"blackbox": rosen}
  param = {"baseline": [-2.0, -2.0],
       "lb": [-5, -5],
       "ub": [10, 10],
       "var_names": ["x1", "x2"],
       "scaling": 10.0,
       "post_dir": "./post"}
  options = {"seed": 0, "budget": 100, "tol": 1e-6, "display": True, "parallel_mode": True, "save_results": True, "isVerbose": True}
  search = {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": 10,
      "visualize": False
    }
  data = {"evaluator": eval, "param": param, "options": options, "search": search}

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

  p_file_3 = os.path.abspath("./tests/bm/unconstrained/rosenbrock.json")
  SEARCH.main(p_file_3)

  p_file_5 = os.path.abspath("./tests/bm/unconstrained/rosenbrock.json")
  MADS.main(p_file_5)

  p_file_2 = os.path.abspath("./tests/bm/constrained/geom_prog.json")
  outP = POLL.main(p_file_2)
  res = outP[0]["fmin"]
  if (outP[0]["fmin"] > 23.8 and platform.platform().split('-')[0] == 'Windows'):
    raise ValueError(f"GP: Poll: fmin = {res} > {23.8}")
 


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
        "budget": 1000,
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

    "search": {
      "type": "VNS",
      "s_method": "LH",
      "ns": 10,
      "visualize": False
    }
  }

  MADS.main(data)

if __name__ == "__main__":
  freeze_support()
