import importlib
import time
from OMADS import POLL, SEARCH, MADS
from matplotlib import pyplot as plt
import copy
import os
import numpy as np

from typing import Dict, List
from multiprocessing import freeze_support
import platform

import logging
if importlib.util.find_spec('BMDFO'):
  from BMDFO import toy
# Configure the logging
# Create a custom logger


logger = logging.getLogger('OMADS_SO_BBO_unit_tests')
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Only log INFO and above to console

# Create a file handler
file_handler = logging.FileHandler(filename='tests/OMADS_BBO_unit_test.log', mode = 'a')
file_handler.setLevel(logging.DEBUG)  # Log all messages to file

# Create a formatter and set it for handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Example filter to exclude messages from the root logger
class NoRootMessagesFilter(logging.Filter):
    def filter(self, record):
        return record.name != 'root'

# Add the filter to handlers
console_handler.addFilter(NoRootMessagesFilter())
file_handler.addFilter(NoRootMessagesFilter())

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
  x = np.array(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
        axis=0), [0]]
  return y

def thin_con(x):
  f = np.sqrt((x[0]-20)**2 + (x[1]-1)**2)
  c1 = np.sin(x[0])-0.1-x[1]
  c2 = x[1] - np.sin(x[0])
  y = [[f], [c1, c2]]
  return y

def test_create_out_file():
  open('tests/OMADS_BBO_unit_test.log', 'w').close()
  

def test_callable_quick_2d():
  logger.info('\nStarted running bbo_2d_rosenbrock test... \n')
  tic = time.perf_counter()
  d = 2
  eval_callable = {"blackbox": rosen}
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
  options = {"seed": 10000, "budget": 1100, "tol": 1e-9, "display": False, "check_cache": True, "store_cache": True, "rich_direction": True, "opportunistic": False, "save_results": False, "isVerbose": False}
  search = {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": int((d+1)*(d+2)/2)+55,
      "visualize": False
    }
  data = {"evaluator": eval_callable, "param": param, "options": options, "sampling": sampling, "search": search}
  data["param"]["lhs_search_initialization"] = True
  logger.info('\nStarted running MADS on bbo_2d_rosenbrock serial exectution ...')
  ticms = time.perf_counter()
  out_mads: Dict = MADS.main(data)
  tocms = time.perf_counter()
  logger.info(f'Completed serial MADS run on bbo_2d_rosenbrock in {tocms - ticms:.4f} seconds.\n')
  
  ticps = time.perf_counter()
  logger.info('\nStarted running POLL on bbo_2d_rosenbrock serial exectution ...')
  out_poll: Dict = POLL.main(data)
  tocps = time.perf_counter()
  logger.info(f'Completed serial POLL run on bbo_2d_rosenbrock in {tocps - ticps:.4f} seconds.\n')

  ticss = time.perf_counter()
  logger.info('\nStarted running SEARCH on bbo_2d_rosenbrock serial exectution ...')
  out_search: Dict = SEARCH.main(data)
  tocss = time.perf_counter()
  logger.info(f'Completed serial SEARCH run on bbo_2d_rosenbrock in {tocss - ticss:.4f} seconds.\n')

  OMS = out_mads[0]["fmin"][0]
  OPS = out_poll[0]["fmin"][0]
  OSS = out_search[0]["fmin"][0]
  
  
  data["options"]["parallel_mode"] = True
  data["options"]["np"] = 4
  logger.info('\nStarted running MADS on bbo_2d_rosenbrock parallel exectution ...')
  ticmp = time.perf_counter()
  out_mads: Dict = MADS.main(data)
  tocmp = time.perf_counter()
  logger.info(f'Completed parallel MADS run on bbo_2d_rosenbrock in {tocmp - ticmp:.4f} seconds.\n')
  
  ticpp = time.perf_counter()
  logger.info('\nStarted running POLL on bbo_2d_rosenbrock parallel exectution ...')
  out_poll: Dict = POLL.main(data)
  tocpp = time.perf_counter()
  logger.info(f'Completed parallel POLL run on bbo_2d_rosenbrock in {tocpp - ticpp:.4f} seconds.\n')

  ticsp = time.perf_counter()
  logger.info('\nStarted running SEARCH on bbo_2d_rosenbrock parallel exectution ...')
  out_search: Dict = SEARCH.main(data)
  tocsp = time.perf_counter()
  logger.info(f'Completed parallel SEARCH run on bbo_2d_rosenbrock in {tocsp - ticsp:.4f} seconds.\n')
  
  OMP = out_mads[0]["fmin"][0]
  OPP = out_poll[0]["fmin"][0]
  OSP = out_search[0]["fmin"][0]
  
  
  toc = time.perf_counter()
  logger.info(f'Completed bbo_2d_rosenbrock serial test in {toc - tic:.4f} seconds.')
  logger.info(f"\nBest known solution: fmin = {0.}")
  logger.info(f"\nSequential Exec: MADS: fmin = {OMS} \nPoll: fmin = {OPS} \nSearch: fmin = {OSS}")
  logger.info(f"\nParallel Exec: MADS: fmin: {OMP} \nPoll: fmin = {OPP}\nSearch: fmin = {OSP}")

  if (OMS > 0.0006):
    logger.error(f"Sequential Exec: MADS: fmin: {OMS} > {0.0006}")
    raise ValueError(f"\nSequential Exec: MADS: fmin: {OMS} > {0.0006}")
  
  if (OPS > 0.008):
    logger.error(f"Sequential Exec: POLL: fmin: {OPS} > {0.008}")
    raise ValueError(f"\nSequential Exec: POLL: fmin: {OPS} > {0.008}")
  
  if (OSS > 0.0006):
    logger.error(f"Sequential Exec: Search: fmin {OSS} > {0.0006}")
    raise ValueError(f"\nSequential Exec: Search: fmin {OSS} > {0.0006}")

  if (OMP > 0.05):
    logger.error(f"Parallel Exec: MADS: fmin: {OMP} > {0.05}")
    raise ValueError(f"\nParallel Exec: MADS: fmin: {OMP} > {0.05}")
  
  if (OPP > 0.008):
    logger.error(f"Parallel Exec: POLL: fmin: {OPP} > {0.008}")
    raise ValueError(f"\nParallel Exec: POLL: fmin: {OPP} > {0.008}")
  
  if (OSP > 0.001):
    logger.error(f"Parallel Exec: Search: fmin {OSP} > {0.001}")
    raise ValueError(f"\nParallel Exec: Search: fmin {OSP} > {0.001}")

def test_callable_2d_sin_const():
  logger.info('\nStarted running bbo_2d_sin_const test...')
  tic = time.perf_counter()
  d = 2
  eval_callable = {"blackbox": thin_con}
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
  options = {"seed": 0, "budget": 2000, "tol": 1e-9, "display": False, "check_cache": True, "store_cache": True, "rich_direction": True, "opportunistic": False, "save_results": False, "isVerbose": False, "precision": "high"}
  search = {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": 100,
      "visualize": False
    }
  data = {"evaluator": eval_callable, "param": param, "options": options, "sampling": sampling, "search": search}

  out_mads: Dict = MADS.main(data)
  OMS = out_mads[0]["fmin"][0] 
  
  toc = time.perf_counter()
  logger.info(f'Completed bbo_2d_sin_const run in {toc - tic:.4f} seconds.\n')
  logger.info(f"\nBest known solution: fmin = {0.0989}")
  logger.info(f"\nSequential Exec: MADS: fmin = {OMS}")

  if (out_mads[0]["fmin"][0] > 0.0989):
    logger.error(f"Sequential Exec: MADS: fmin: {OMS} > {0.0989}")
    raise ValueError(f"\nSequential Exec: MADS: fmin: {OMS} > {0.0989}")

def test_callable_quick_10d():
  logger.info('\nStarted running bbo_10d_rosenbrock test...')
  tic = time.perf_counter()
  d = 10
  eval_callable = {"blackbox": rosen}
  param = {"name": "RB","baseline": [-2.5]*d,
       "lb": [-5]*d,
       "ub": [10]*d,
       "var_names": [f"x{i}" for i in range(d)],
       "scaling": [15.0]*d,
       "post_dir": "./post"}

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
  data = {"evaluator": eval_callable, "param": param, "options": options, "sampling": sampling, "search": search}
  logger.info('\nStarted running MADS on bbo_10d_rosenbrock serial exectution ...')
  ticms = time.perf_counter()
  out_mads: Dict = MADS.main(data)
  tocms = time.perf_counter()
  logger.info(f'Completed serial MADS run on bbo_10d_rosenbrock in {tocms - ticms:.4f} seconds.\n')
  
  ticps = time.perf_counter()
  logger.info('\nStarted running POLL on bbo_10d_rosenbrock serial exectution ...')
  out_poll: Dict = POLL.main(data)
  tocps = time.perf_counter()
  logger.info(f'Completed serial POL run on bbo_10d_rosenbrock in {tocps - ticps:.4f} seconds.\n')

  ticss = time.perf_counter()
  logger.info('\nStarted running SEARCH on bbo_10d_rosenbrock serial exectution ...')
  out_search: Dict = SEARCH.main(data)
  tocss = time.perf_counter()
  logger.info(f'Completed serial SEARCH run on bbo_10d_rosenbrock in {tocss - ticss:.4f} seconds.\n')

  OSS = out_search[0]["fmin"][0]
  OPS = out_poll[0]["fmin"][0]
  OMS = out_mads[0]["fmin"][0]

  toc = time.perf_counter()
  logger.info(f'Completed bbo_10d_rosenbrock run in {toc - tic:.4f} seconds.')
  logger.info(f"\nBest known solution: fmin = {0.}")
  logger.info(f"\nSequential Exec: MADS: fmin = {OMS} \nPoll: fmin = {OPS} \nSearch: fmin = {OSS}")

  if (out_mads[0]["fmin"][0] > 0.0006):
    logger.error(f"Sequential Exec: MADS: fmin: {OMS} > {0.0006}")
    raise ValueError(f"\nSequential Exec: MADS: fmin: {OMS} > {0.0006}")
  
  if (out_poll[0]["fmin"][0] > 0.25):
    logger.error(f"Sequential Exec: POLL: fmin: {OPS} > {0.25}")
    raise ValueError(f"\nSequential Exec: POLL: fmin: {OPS} > {0.25}")
  
  if (out_search[0]["fmin"][0] > 0.0006):
    logger.error(f"Sequential Exec: Search: fmin {OSS} > {0.0006}")
    raise ValueError(f"\nSequential Exec: Search: fmin {OSS} > {0.0006}")

def test_callable_quick_20d():
  logger.info('\nStarted running bbo_20d_rosenbrock test...')
  tic = time.perf_counter()
  d = 20
  eval_callable = {"blackbox": rosen}
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
  options = {"seed": 12345, "budget": 5500, "tol": 1e-12, "display": False, "check_cache": True, "store_cache": True, "rich_direction": True, "opportunistic": False, "save_results": False, "isVerbose": False, "precision": "high"}
  search = {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": 250,
      "visualize": False
    }

  data = {"evaluator": eval_callable, "param": param, "options": options, "sampling": sampling,"search": search}
  logger.info('\nStarted running MADS on bbo_20d_rosenbrock serial exectution ...')
  ticms = time.perf_counter()
  out_mads: Dict = MADS.main(data)
  tocms = time.perf_counter()
  logger.info(f'Completed serial MADS run on bbo_20d_rosenbrock in {tocms - ticms:.4f} seconds.\n')
  
  ticps = time.perf_counter()
  logger.info('\nStarted running POLL on bbo_20d_rosenbrock serial exectution ...')
  out_poll: Dict = POLL.main(data)
  tocps = time.perf_counter()
  logger.info(f'Completed serial POL run on bbo_20d_rosenbrock in {tocps - ticps:.4f} seconds.\n')

  ticss = time.perf_counter()
  logger.info('\nStarted running SEARCH on bbo_20d_rosenbrock serial exectution ...')
  out_search: Dict = SEARCH.main(data)
  tocss = time.perf_counter()
  logger.info(f'Completed serial SEARCH run on bbo_20d_rosenbrock in {tocss - ticss:.4f} seconds.\n')

  OSS = out_search[0]["fmin"][0]
  OPS = out_poll[0]["fmin"][0]
  OMS = out_mads[0]["fmin"][0]

  
  toc = time.perf_counter()
  logger.info(f'Completed bbo_20d_rosenbrock run in {toc - tic:.4f} seconds.')
  logger.info(f"\nBest known solution: fmin = {0.}")
  logger.info(f"\nSequential Exec: MADS: fmin = {OMS} \nPoll: fmin = {OPS} \nSearch: fmin = {OSS}")

  if (out_mads[0]["fmin"][0] > 0.0006):
    logger.error(f"Sequential Exec: MADS: fmin: {OMS} > {0.0006}")
    raise ValueError(f"\nSequential Exec: MADS: fmin: {OMS} > {0.0006}")
  
  if (out_poll[0]["fmin"][0] > 2.7):
    logger.error(f"Sequential Exec: POLL: fmin: {OPS} > {2.7}")
    raise ValueError(f"\nSequential Exec: POLL: fmin: {OPS} > {2.7}")
  
  if (out_search[0]["fmin"][0] > 0.0006):
    logger.error(f"Sequential Exec: Search: fmin {OSS} > {0.0006}")
    raise ValueError(f"\nSequential Exec: Search: fmin {OSS} > {0.0006}")

def test_omads_toy_quick():
  assert POLL.CandidatePoint
  assert POLL.PrePoll
  assert POLL.main

  if importlib.util.find_spec('BMDFO'):
    from BMDFO import toy
    p_file = os.path.abspath("./tests/bm/unconstrained/rosenbrock.json")
    p_file_2 = os.path.abspath("./tests/bm/constrained/geom_prog.json")
  else:
    is_win = platform.platform().split('-')[0] == 'Windows'
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
        "ns": 50,
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
      "h_max": 0.0,
      "lhs_search_initialization": True

    },
  
  "options":
    {
      "seed": 10000,
      "budget": 100000,
      "tol": 1e-12,
      "psize_init": 2.0 if is_win else 1.0,
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
  
  logger.info('\nStarted running bbo_2d_rosenbrock_VNS test...')
  tic = time.perf_counter()
  out_search: Dict = SEARCH.main(p_file)
  OSS = out_search[0]["fmin"][0]
  out_poll: Dict = POLL.main(p_file)
  OPS = out_poll[0]["fmin"][0]
  out_mads: Dict = MADS.main(p_file)
  OMS = out_mads[0]["fmin"][0]

  toc = time.perf_counter()
  logger.info(f'Completed bbo_2d_rosenbrock_VNS run in {toc - tic:.4f} seconds.')
  logger.info(f"\nBest known solution: fmin = {0.}")
  logger.info(f"\nSequential Exec: MADS: fmin = {OMS} \nPoll: fmin = {OPS} \nSearch: fmin = {OSS}")


  logger.info('\nStarted running bbo_GP_POLL test...')
  tic = time.perf_counter()
  out_poll = POLL.main(p_file_2)
  res = out_poll[0]["fmin"][0]
  
  toc = time.perf_counter()
  logger.info(f'Completed bbo_GP_POLL run in {toc - tic:.4f} seconds.')
  logger.info(f"\nBest known solution: {15} < fmin <= {25}")
  logger.info(f"\nSequential Exec: Poll: fmin = {res}")

  if (res > 23.8 ):
    logger.error(f"\nSequential Exec: POLL: fmin: {res} > {23.8 }")
    raise ValueError(f"\nSequential Exec: POLL: fmin: {res} > {23.8 }")

  

if __name__ == "__main__":
  freeze_support()
