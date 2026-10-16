"""Pass fixed parameters to the blackbox through evaluator["constants"]."""
import numpy as np
from OMADS import mads


def rosenbrock(x, p):
  """p = [a, b]: f = sum(a (x[i+1] - x[i]^2)^2 + (b - x[i])^2)."""
  a, b = p
  x = np.asarray(x)
  f = np.sum(a * (x[1:] - x[:-1] ** 2) ** 2 + (b - x[:-1]) ** 2)
  return [f, [0]]


data = {
    "evaluator": {"blackbox": rosenbrock, "constants": [100.0, 1.0]},
    "param": {"name": "rosen_p", "baseline": [-2.0, -2.0], "lb": [-5, -5],
              "ub": [10, 10], "scaling": [15.0, 15.0], "post_dir": "./post"},
    "options": {"seed": 0, "budget": 1000, "tol": 1e-9,
                "check_cache": True, "store_cache": True, "rich_direction": True},
    "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 10},
}

if __name__ == "__main__":
  out, *_ = mads.main(data)
  print("xmin =", out["xmin"], " fmin =", out["fmin"])
