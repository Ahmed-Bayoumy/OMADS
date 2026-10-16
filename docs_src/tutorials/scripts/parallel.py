"""Evaluate poll and search points in parallel processes."""
import time

import numpy as np
from OMADS import mads


def slow_sphere(x):
  time.sleep(0.05)                        # pretend each evaluation is expensive
  return [float(np.sum(np.square(x))), [0]]


def make_data(parallel):
  return {
      "evaluator": {"blackbox": slow_sphere},
      "param": {"name": "sphere", "baseline": [3.0] * 4, "lb": [-5.0] * 4,
                "ub": [5.0] * 4, "scaling": [10.0] * 4, "post_dir": "./post"},
      "options": {"seed": 0, "budget": 120, "tol": 1e-9,
                  "check_cache": True, "store_cache": True, "rich_direction": True,
                  "parallel_mode": parallel, "np": 4},
      "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 8},
  }


if __name__ == "__main__":                 # required with parallel_mode
  for parallel in (False, True):
    tic = time.perf_counter()
    out, *_ = mads.main(make_data(parallel))
    print(f"parallel={parallel!s:5}  fmin={out['fmin'][0]:.3e}  "
          f"evals={out['nbb_evals']}  wall time={time.perf_counter() - tic:.1f}s")
