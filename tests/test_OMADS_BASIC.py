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
  search = {
      "type": "sampling",
      "s_method": "LH",
      "ns": 10,
      "visualize": False
    }
  data = {"evaluator": eval, "param": param, "options": options, "sampling": sampling, "search": search}

  

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
  options = {"seed": 0, "budget": 100, "tol": 1e-6, "display": True, "parallel_mode": True}
  sampling = {
                    "method": SEARCH.explore.SAMPLING_METHOD.LH.value,
                    "ns": int((2+1)*(2+2)/2)+50,
                    "visualize": False,
                    "criterion": None
                  }
  search = {
      "type": "sampling",
      "s_method": "LH",
      "ns": 10,
      "visualize": False
    }
  data = {"evaluator": eval, "param": param, "options": options, "sampling": sampling, "search": search}

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

  p_file_3 = os.path.abspath("./tests/bm/unconstrained/rosenbrock.json")
  SEARCH.main(p_file_3)

  p_file_4 = os.path.abspath("./tests/bm/constrained/geom_prog.json")
  SEARCH.main(p_file_4)

  p_file_5 = os.path.abspath("./tests/bm/unconstrained/rosenbrock.json")
  MADS.main(p_file_5)

  p_file_6 = os.path.abspath("./tests/bm/constrained/geom_prog.json")
  MADS.main(p_file_6)

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