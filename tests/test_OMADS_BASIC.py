"""
Pytest for single objective optimization studies
"""
import time
import platform

import logging
from typing import Dict

import numpy as np
from OMADS import poll, search, mads
from multiprocessing import freeze_support


# Configure the logging
# Create a custom logger
logger = logging.getLogger('Omads_SO_BBO_unit_tests')
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Only log INFO and above to console

# Create a file handler
file_handler = logging.FileHandler(
    filename='tests/Omads_BBO_unit_test.log', mode='a')
file_handler.setLevel(logging.DEBUG)  # Log all messages to file

# Create a formatter and set it for handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Example filter to exclude messages from the root logger


class NoRootMessagesFilter(logging.Filter):
  """Exclude logging messages from the root filter
  """

  def filter(self, record):
    return record.name != 'root'


# Add the filter to handlers
console_handler.addFilter(NoRootMessagesFilter())
file_handler.addFilter(NoRootMessagesFilter())


def geom_prog(x):
  """Geometric programming problem
  """
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


def rosen(x):
  """Rosenbrock function
  """
  x = np.array(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
              axis=0), [0]]
  return y


def thin_con(x):
  """Highly infeasible problem (narrow feasible region)
  """
  f = np.sqrt((x[0] - 20)**2 + (x[1] - 1)**2)
  c1 = np.sin(x[0]) - 0.1 - x[1]
  c2 = x[1] - np.sin(x[0])
  y = [f, [c1, c2]]
  return y


def test_create_out_file():
  """Test creating the output file
  """
  open('tests/Omads_BBO_unit_test.log', 'w', encoding='utf-8').close()


def test_callable_quick_2d():
  """Run quick 2D test
  """
  logger.info('\nStarted running bbo_2d_rosenbrock test... \n')
  tic = time.perf_counter()
  d = 2
  eval_callable = {"blackbox": rosen}
  param = {"name": "RB", "baseline": [-2.5] * d,
           "lb": [-5] * d,
           "ub": [10] * d,
           "var_names": [f"x{i}" for i in range(d)],
           "scaling": [15.0] * d,
           "post_dir": "./post"
           }
  options = {"seed": 10000, "budget": 2500, "tol": 1e-12, "display": False,
             "check_cache": True, "store_cache": True, "rich_direction": True,
             "opportunistic": False, "save_results": False, "is_verbose": False, "precision": "high"}
  search_conf = {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": int((d + 1) * (d + 2) / 2) + 50,
      "visualize": False
  }
  data = {"evaluator": eval_callable, "param": param,
          "options": options, "search": search_conf}
  # data["param"]["lhs_search_initialization"] = True
  logger.info(
      '\nStarted running mads on bbo_2d_rosenbrock serial exectution ...')
  ticms = time.perf_counter()
  out_mads: Dict = mads.main(data)
  tocms = time.perf_counter()
  logger.info(
      'Completed serial mads run on bbo_2d_rosenbrock in %s seconds.\n',
      f'{tocms - ticms:.4f}')

  ticps = time.perf_counter()
  logger.info(
      '\nStarted running poll on bbo_2d_rosenbrock serial exectution ...')
  out_poll: Dict = poll.main(data)
  tocps = time.perf_counter()
  logger.info(
      'Completed serial poll run on bbo_2d_rosenbrock in %s seconds.\n',
      f'{tocps - ticps:.4f}')

  ticss = time.perf_counter()
  logger.info(
      '\nStarted running search on bbo_2d_rosenbrock serial exectution ...')
  out_search: Dict = search.main(data)
  tocss = time.perf_counter()
  logger.info(
      'Completed serial search run on bbo_2d_rosenbrock in %s seconds.\n',
      f'{tocss - ticss:.4f}')

  oms = out_mads[0]["fmin"][0]
  ops = out_poll[0]["fmin"][0]
  oss = out_search[0]["fmin"][0]

  data["options"]["parallel_mode"] = True
  data["options"]["np"] = 4
  logger.info(
      '\nStarted running mads on bbo_2d_rosenbrock parallel exectution ...')
  ticmp = time.perf_counter()
  out_mads: Dict = mads.main(data)
  tocmp = time.perf_counter()
  logger.info(
      'Completed parallel mads run on bbo_2d_rosenbrock in %s seconds.\n',
      f'{tocmp - ticmp:.4f}')

  ticpp = time.perf_counter()
  logger.info(
      '\nStarted running poll on bbo_2d_rosenbrock parallel exectution ...')
  out_poll: Dict = poll.main(data)
  tocpp = time.perf_counter()
  logger.info(
      'Completed parallel poll run on bbo_2d_rosenbrock in %s seconds.\n',
      f'{tocpp - ticpp:.4f}')

  ticsp = time.perf_counter()
  logger.info(
      '\nStarted running search on bbo_2d_rosenbrock parallel exectution ...')
  data["search"]["ns"] = int((d + 1) * (d + 2) / 2) + 250
  out_search: Dict = search.main(data)
  tocsp = time.perf_counter()
  logger.info(
      'Completed parallel search run on bbo_2d_rosenbrock in %s seconds.\n',
      f'{tocsp - ticsp: .4f}')

  omp = out_mads[0]["fmin"][0]
  opp = out_poll[0]["fmin"][0]
  osp = out_search[0]["fmin"][0]

  toc = time.perf_counter()
  logger.info(
      'Completed bbo_2d_rosenbrock serial test in %s seconds.',
      f'{toc - tic:.4f} ')
  logger.info("\nBest known solution: fmin = {%s}", 0.)
  logger.info(
      "\nSequential Exec: mads: fmin = %s \npoll: fmin = %s \nsearch: fmin = %s",
      oms, ops, oss)
  logger.info(
      "\nParallel Exec: mads: fmin: %s \npoll: fmin = %s\nsearch: fmin = %s",
      omp, opp, osp)

  if oms > 0.005:
    logger.error(f"Sequential Exec: mads: fmin: %s > {0.0006}", oms)
    raise ValueError(f"\nSequential Exec: mads: fmin: {oms} > {0.0006}")

  if ops > 0.008:
    logger.error("Sequential Exec: poll: fmin: %s > %s", ops, 0.008)
    raise ValueError(f"\nSequential Exec: poll: fmin: {ops} > {0.008}")

  if oss > 0.07:
    logger.error("Sequential Exec: search: fmin %s > %s", oss, 0.07)
    raise ValueError(f"\nSequential Exec: search: fmin {oss} > {0.07}")

  if omp > 0.005:
    logger.error("Parallel Exec: mads: fmin: %s > %s", omp, 0.004)
    raise ValueError(f"\nParallel Exec: mads: fmin: {omp} > {0.004}")

  if opp > 0.008:
    logger.error("Parallel Exec: poll: fmin: %s > %s", opp, 0.008)
    raise ValueError(f"\nParallel Exec: poll: fmin: {opp} > {0.008}")

  if osp > 0.065:
    logger.error("Parallel Exec: search: fmin %s > %s", osp, 0.065)
    raise ValueError(
        f"\nParallel Exec: search: fmin {osp} > {0.065}", osp, 0.065)


def test_callable_2d_sin_const():
  """Highly constrained 2d problem
  """
  logger.info('\nStarted running bbo_2d_sin_const test...')
  tic = time.perf_counter()
  d = 2
  eval_callable = {"blackbox": thin_con}
  param = {"name": "thin_con", "baseline": [0, -10],
           "lb": [0, -10],
           "ub": [25, 10],
           "var_names": [f"x{i}" for i in range(d)],
           "constraints_type": ["PB", "PB"],
           "scaling": [20.0] * d,
           "post_dir": "./post",
           "rho": 1,
           "lambda_multipliers": 1,
           "h_max": np.inf}
  options = {
      "seed": 1234, "budget": 2000, "tol": 1e-9, "display": True,
      "check_cache": True, "store_cache": True, "rich_direction": True,
      "opportunistic": False, "save_results": False, "is_verbose": False,
      "precision": "high"}
  search_conf = {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": 70,
      "visualize": False
  }

  data = {"evaluator": eval_callable, "param": param, "options": options,
          "search": search_conf}

  out_mads: Dict = mads.main(data)
  oms = out_mads[0]["fmin"][0]

  toc = time.perf_counter()
  logger.info('Completed bbo_2d_sin_const run in %s seconds.\n',
              f'{toc - tic:.4f}')
  logger.info("\nBest known solution: fmin = %s", 0.0989)
  logger.info("\nSequential Exec: mads: fmin = %s", oms)

  if out_mads[0]["fmin"][0] > 0.0989:
    logger.error("Sequential Exec: mads: fmin: %s > %s", oms, 0.0989)
    raise ValueError(f"\nSequential Exec: mads: fmin: {oms} > {0.0989}")


def test_callable_quick_10d():
  """Ten dimensional Rosenbrock
  """
  logger.info('\nStarted running bbo_10d_rosenbrock test...')
  tic = time.perf_counter()
  d = 10
  eval_callable = {"blackbox": rosen}
  param = {"name": "RB", "baseline": [-2.5] * d,
           "lb": [-5] * d,
           "ub": [10] * d,
           "var_names": [f"x{i}" for i in range(d)],
           "scaling": [15.0] * d,
           "post_dir": "./post"}

  options = {
      "seed": 12345, "budget": 1200, "tol": 1e-12, "display": False,
      "check_cache": True, "store_cache": True, "rich_direction": True,
      "opportunistic": False, "save_results": False, "is_verbose": False,
      "precision": "medium"}
  search_conf = {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": int((d + 1) * (d + 2) / 2) + 150,
      "visualize": False
  }
  data = {"evaluator": eval_callable, "param": param,
          "options": options, "search": search_conf}
  logger.info(
      '\nStarted running mads on bbo_10d_rosenbrock serial exectution ...')
  ticms = time.perf_counter()
  out_mads: Dict = mads.main(data)
  tocms = time.perf_counter()
  logger.info(
      'Completed serial mads run on bbo_10d_rosenbrock in %s seconds.\n',
      f'{tocms - ticms:.4f}')

  ticps = time.perf_counter()
  logger.info(
      '\nStarted running poll on bbo_10d_rosenbrock serial exectution ...')
  data_poll = data
  data_poll["options"]["budget"] = 3000
  out_poll: Dict = poll.main(data_poll)
  tocps = time.perf_counter()
  logger.info(
      'Completed serial POL run on bbo_10d_rosenbrock in %s seconds.\n',
      f'{tocps - ticps: .4f}')

  ticss = time.perf_counter()
  logger.info(
      '\nStarted running search on bbo_10d_rosenbrock serial exectution ...')
  data_search = data
  data_search["options"]["budget"] = 1000
  out_search: Dict = search.main(data_search)
  tocss = time.perf_counter()
  logger.info(
      'Completed serial search run on bbo_10d_rosenbrock in %s seconds.\n',
      f'{tocss - ticss:.4f}')

  oss = out_search[0]["fmin"][0]
  ops = out_poll[0]["fmin"][0]
  oms = out_mads[0]["fmin"][0]

  toc = time.perf_counter()
  logger.info('Completed bbo_10d_rosenbrock run in %s seconds.',
              f'{toc - tic:.4f}')
  logger.info("\nBest known solution: fmin = %s", 0.)
  logger.info(
      "\nSequential Exec: mads: fmin = %s \npoll: fmin = %s \nsearch: fmin = %s",
      oms, ops, oss)

  if out_mads[0]["fmin"][0] > 0.0006:
    logger.error("Sequential Exec: mads: fmin: %s > %s", oms, 0.0006)
    raise ValueError(f"\nSequential Exec: mads: fmin: {oms} > {0.0006}")

  if out_poll[0]["fmin"][0] > 8:
    logger.error("Sequential Exec: poll: fmin: %s > %s", ops, 0.25)
    raise ValueError(f"\nSequential Exec: poll: fmin: {ops} > {0.25}")

  if out_search[0]["fmin"][0] > 0.0006:
    logger.error("Sequential Exec: search: fmin %s > %s", oss, 0.0006)
    raise ValueError(f"\nSequential Exec: search: fmin {oss} > {0.0006}")


def test_callable_quick_20d():
  """Twenty dimensional Rosenbrock  """
  logger.info('\nStarted running bbo_20d_rosenbrock test...')
  tic = time.perf_counter()
  d = 20
  eval_callable = {"blackbox": rosen}
  param = {"name": "RB", "baseline": [-2.5] * d,
           "lb": [-5] * d,
           "ub": [10] * d,
           "var_names": [f"x{i}" for i in range(d)],
           "scaling": [15] * d,
           "post_dir": "./post"}

  sampling = {
      "method": 'ACTIVE',
      "ns": int((d + 1) * (d + 2) / 2) + 50,
      "visualize": False,
      "criterion": None
  }
  options = {
      "seed": 12345, "budget": 3000, "tol": 1e-12, "display": False,
      "check_cache": True, "store_cache": True, "rich_direction": True,
      "opportunistic": False, "save_results": False, "is_verbose": False, 
      "precision": "medium"}
  search_conf = {
      "type": "sampling",
      "s_method": "ACTIVE",
      "ns": int((d + 1) * (d + 2) / 2) + 50,
      "visualize": False
  }

  data = {"evaluator": eval_callable, "param": param,
          "options": options, "sampling": sampling, "search": search_conf}
  logger.info(
      '\nStarted running mads on bbo_20d_rosenbrock serial exectution ...')
  ticms = time.perf_counter()
  out_mads: Dict = mads.main(data)
  tocms = time.perf_counter()
  logger.info(
      'Completed serial mads run on bbo_20d_rosenbrock in %s seconds.\n',
      f'{tocms - ticms:.4f}')

  ticps = time.perf_counter()
  logger.info(
      '\nStarted running poll on bbo_20d_rosenbrock serial exectution ...')
  data_poll = data
  data_poll["options"]["budget"] = 10000
  out_poll: Dict = poll.main(data_poll)
  tocps = time.perf_counter()
  logger.info(
      'Completed serial POL run on bbo_20d_rosenbrock in %s seconds.\n',
      f'{tocps - ticps:.4f}')

  ticss = time.perf_counter()
  logger.info(
      '\nStarted running search on bbo_20d_rosenbrock serial exectution ...')
  data["options"]["budget"] = 3000
  out_search: Dict = search.main(data)
  tocss = time.perf_counter()
  logger.info(
      'Completed serial search run on bbo_20d_rosenbrock in %s seconds.\n',
      f'{tocss - ticss:.4f}')

  oss = out_search[0]["fmin"][0]
  ops = out_poll[0]["fmin"][0]
  oms = out_mads[0]["fmin"][0]

  toc = time.perf_counter()
  logger.info('Completed bbo_20d_rosenbrock run in %s seconds.',
              f'{toc - tic:.4f}')
  logger.info("\nBest known solution: fmin = %s", 0.)
  logger.info(
      "\nSequential Exec: mads: fmin = %s \npoll: fmin = %s \nsearch: fmin = %s",
      oms, ops, oss)

  if out_mads[0]["fmin"][0] > 0.0006:
    logger.error("Sequential Exec: mads: fmin: %s > %s", oms, 0.0006)
    raise ValueError(f"\nSequential Exec: mads: fmin: {oms} > {0.0006}")

  if out_poll[0]["fmin"][0] > 20:
    logger.error("Sequential Exec: poll: fmin: %s > %s", ops, 2.7)
    raise ValueError(f"\nSequential Exec: poll: fmin: {ops} > {2.7}")

  if out_search[0]["fmin"][0] > 0.0006:
    logger.error("Sequential Exec: search: fmin %s > %s", oss, 0.0006)
    raise ValueError(f"\nSequential Exec: search: fmin {oss} > {0.0006}")


def test_omads_toy_quick():
  """GP and RB
  """
  assert poll.CandidatePoint
  # assert poll.PrePoll
  assert poll.main

  is_win = platform.platform().split('-')[0] == 'Windows'
  p_file = {
      "evaluator":
      {
          "blackbox": rosen,
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
          "baseline": [1E5, 1E5, 1E5, 1E5, 1E5, 1E5, 1E5, 1E5, 1E5, 1E5],
          "lb": [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
          "ub": [1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6],
          "var_names": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"],
          "scaling": 1e6,
          "constraints_type": ["PB", "PB", "PB", "PB", "PB", "PB"],
          "lambda_multipliers": [1000, 1000, 1000, 1000, 1000, 1000],
          "rho": 0.01,
          "post_dir": "./tests/bm/constrained/post",
          "h_max": np.inf,
          # "lhs_search_initialization": True,
          # "mesh_type": "GMESH"

      },

      "options":
      {
          "seed": 12345,
          "budget": 10000,
          "tol": 1e-12,
          "psize_init": 2.0 if is_win else 1.0,
          "display": False,
          "opportunistic": False,
          "check_cache": True,
          "store_cache": True,
          "collect_y": False,
          "rich_direction": True,
          "precision": "high",
          "save_results": True,
          "save_coordinates": False,
          "save_all_best": False,
          "parallel_mode": False
      },
      "search": {
          "type": "sampling",
          "s_method": "ACTIVE",
          "ns": 100,
          "visualize": False
      }
  }

  logger.info('\nStarted running bbo_2d_rosenbrock test...')
  tic = time.perf_counter()
  out_search: Dict = search.main(p_file)
  oss = out_search[0]["fmin"][0]
  out_poll: Dict = poll.main(p_file)
  ops = out_poll[0]["fmin"][0]
  out_mads: Dict = mads.main(p_file)
  oms = out_mads[0]["fmin"][0]

  toc = time.perf_counter()
  logger.info(
      'Completed bbo_2d_rosenbrock run in %s seconds.',
      f'{toc - tic: .4f}')
  logger.info("\nBest known solution: fmin = %s", 0.)
  logger.info(
      "\nSequential Exec: mads: fmin = %s \npoll: fmin = %s \nsearch: fmin = %s",
      oms, ops, oss)

  logger.info('\nStarted running bbo_GP_poll test...')
  tic = time.perf_counter()
  out_poll = poll.main(p_file_2)
  res = out_poll[0]["fmin"][0]

  toc = time.perf_counter()
  logger.info('Completed bbo_GP_poll run in %s seconds.', f'{toc - tic: .4f}')
  logger.info("\nBest known solution: %s < fmin <= %s", 15, 25)
  logger.info("\nSequential Exec: poll: fmin = %s", res)

  if res > 23.8:
    logger.error("\nSequential Exec: poll: fmin: %s > %s", res, 23.8)
    raise ValueError(f"\nSequential Exec: poll: fmin: {res} > {23.8}")


if __name__ == "__main__":
  freeze_support()
