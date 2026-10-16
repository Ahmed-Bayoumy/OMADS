"""2D Ackley 3 function."""
import numpy as np
from OMADS import mads


def ackley3(x):
  f = (-200.0 * np.exp(-0.2 * np.sqrt(x[0] ** 2 + x[1] ** 2))
       + 5.0 * np.exp(np.cos(3.0 * x[0]) + np.sin(3.0 * x[1])))
  return [f, [0]]


data = {
    "evaluator": {"blackbox": ackley3},
    "param": {"name": "ackley3", "baseline": [5.0, 5.0], "lb": [-32.0, -32.0],
              "ub": [32.0, 32.0], "scaling": [64.0, 64.0], "post_dir": "./post"},
    "options": {"seed": 0, "budget": 1000, "tol": 1e-9,
                "check_cache": True, "store_cache": True, "rich_direction": True},
    "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 10},
}

if __name__ == "__main__":
  out, *_ = mads.main(data)
  print("xmin =", out["xmin"])
  print("fmin =", out["fmin"], "after", out["nbb_evals"], "evaluations")
