import importlib
from OMADS import POLL, SEARCH, MADS
from matplotlib import pyplot as plt
import copy
import os
import numpy as np

from typing import Dict, List
from multiprocessing import freeze_support
import platform

def geom_prog(x, *argv):
  xx = x
  x2 = np.sqrt(xx[3] ** 2 + xx[4] ** -2 + xx[5] ** -2 + xx[6] ** 2)
  x5 = np.sqrt(xx[6] ** 2 + xx[7] ** 2 + xx[8] ** 2 + xx[9] ** 2)
  x0 = np.sqrt(x2 ** 2 + xx[0] ** -2 + xx[1] ** 2)
  x1 = np.sqrt(xx[1] ** 2 + x5 ** 2 + xx[2] ** 2)

  f = x0 ** 2 + x1 ** 2
  c = [x2 ** -2 + xx[0] ** 2 - xx[1] ** 2, xx[1] ** 2 + x5 ** -2 - xx[2] ** 2,
        xx[3] ** 2 + xx[4] ** 2 - xx[6] ** 2, xx[3] ** -2 + xx[5] ** 2 - xx[6] ** 2,
        xx[6] ** 2 + xx[7] ** -2 - xx[8] ** 2, xx[6] ** 2 + xx[7] ** 2 - xx[9] ** 2]
  return [f, c]

def rosen(x, *argv):
  x = np.asarray(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
        axis=0), [0]]
  return y


def thin_con(x):
  f = np.sqrt((x[0]-20)**2 + (x[1]-1)**2)
  c1 = np.sin(x[0])-0.1-x[1]
  c2 = x[1] - np.sin(x[0])
  y = [[f], [c1, c2]]
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
  OM = outM[0]["fmin"][0]
  OP = outP[0]["fmin"][0]
  OS = outS[0]["fmin"][0]
  if (outM[0]["fmin"][0] > 0.0006):
    raise ValueError(f"\nMADS: fmin: {OM} > {0.0006} \nPoll: fmin = {OP}\nSearch: fmin = {OS}")
  
  if (outP[0]["fmin"][0] > 0.0006):
    raise ValueError(f"\nPOLL: fmin: {OP} > {0.0006} \nMADS: fmin = {OM}\nSearch: fmin = {OS}")
  
  if (outS[0]["fmin"][0] > 0.0006):
    raise ValueError(f"\nSearch: fmin {OS} > {0.0006} \nMADS: fmin = {OM}\nPoll: fmin = {OP}")

def test_MADS_callable_quick_const_2d():
  d = 2
  eval = {"blackbox": thin_con}
  param = {"name": "thin_con","baseline": [0, -10],
       "lb": [0, -10],
       "ub": [25, 10],
       "var_names": [f"x{i}" for i in range(d)],
       "constraints_type": ["PB", "PB"],
       "scaling": [20.0]*d,
       "post_dir": "./post"}
  sampling = {
                    "method": 'LH',
                    "ns": 100,
                    "visualize": False,
                    "criterion": None
                  }
  options = {"seed": 1234, "budget": 2000, "tol": 1e-9, "display": False, "check_cache": True, "store_cache": True, "rich_direction": True, "opportunistic": False, "save_results": False, "isVerbose": False}
  search = {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": 100,
      "visualize": False
    }
  data = {"evaluator": eval, "param": param, "options": options, "sampling": sampling, "search": search}

  outM: Dict = MADS.main(data)
  OM = outM[0]["fmin"][0] 

  if (outM[0]["fmin"][0] > 0.098):
    raise ValueError(f"MADS: fmin: {OM} > {0.098}")

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

  if (outS[0]["fmin"][0] > 0.0006):
    raise ValueError(f"Search: fmin > {0.0006}")
  
  outP: Dict = POLL.main(data)
  if (outP[0]["fmin"][0] > 0.25):
    raise ValueError(f"POLL: fmin > {0.25}")
  
  outM: Dict = MADS.main(data)
  if (outM[0]["fmin"][0] > 0.0006):
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
  isWin = platform.platform().split('-')[0] == 'Windows'

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

  data = {"evaluator": eval, "param": param, "options": options, "sampling": sampling, "search": search}
  outS: Dict = SEARCH.main(data)
  SR = outS[0]["fmin"][0]
  if (SR > 0.0006 and platform.platform().split('-')[0] == 'Windows'):
    raise ValueError(f"Search: fmin {SR} > {0.0006}")
  
  outP: Dict = POLL.main(data)
  PR = outP[0]["fmin"][0]
  if (PR > 2.7 and platform.platform().split('-')[0] == 'Windows'):
    raise ValueError(f"POLL: fmin {PR} > {2.7}")
  
  outM: Dict = MADS.main(data)
  MR = outM[0]["fmin"][0]
  if (MR > 0.0006 and platform.platform().split('-')[0] == 'Windows'):
    raise ValueError(f"MADS: fmin {MR} > {0.0006}")

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
  assert POLL.CandidatePoint
  assert POLL.OrthoMesh
  assert POLL.Cache
  assert POLL.Dirs2n
  assert POLL.PrePoll
  assert POLL.Output
  assert POLL.PostMADS
  assert POLL.main

  if importlib.util.find_spec('BMDFO'):
    from BMDFO import toy
    p_file = os.path.abspath("./tests/bm/unconstrained/rosenbrock.json")
    p_file_2 = os.path.abspath("./tests/bm/constrained/geom_prog.json")
  else:
    p_file = {
    "evaluator":
      {
        "blackbox": rosen,
      },
    "param":
      {
        "baseline": [-2.0,-2.0],
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
      "search": {
        "type": "VNS",
        "s_method": "LH",
        "ns": 100,
        "visualize": False
      }
  }
    p_file_2 = {
  "evaluator":
    {
      "blackbox": geom_prog,
    },

  "param":
    {
      "name": "GP",
      "baseline": [1E5,1E5,1E5,1E5,1E5,1E5,1E5,1E5,1E5,1E5],
      "lb": [1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6,1e-6],
      "ub": [1e6,1e6,1e6,1e6,1e6,1e6,1e6,1e6,1e6,1e6],
      "var_names": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"],
      "scaling": 10,
      "constraints_type": ["PB", "PB", "PB","PB", "PB", "PB"],
      "LAMBDA": [1E5, 1E5, 1E5, 1E5, 1E5, 1E5],
      "RHO": 1.0,
      "post_dir": "./tests/bm/constrained/post",
      "h_max": 0.0
    },

  "options":
    {
      "seed": 10000,
      "budget": 100000,
      "tol": 1e-12,
      "psize_init": 2.0,
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
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": 500,
      "visualize": False
    }
}
  
  POLL.main(p_file)
  SEARCH.main(p_file)
  MADS.main(p_file)

  outP = POLL.main(p_file_2)
  res = outP[0]["fmin"][0]
  if (outP[0]["fmin"][0] > 23.8 and platform.platform().split('-')[0] == 'Windows'):
    raise ValueError(f"GP: Poll: fmin = {res} > {23.8}")
 


  data = {
    "evaluator":
      {
        "blackbox": rosen
      },

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
