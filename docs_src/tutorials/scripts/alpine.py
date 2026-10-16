"""2D Alpine function (non-smooth, multimodal)."""
import numpy as np
from OMADS import mads


def alpine(x):
  f = sum(abs(xi * np.sin(xi) + 0.1 * xi) for xi in x)
  return [f, [0]]


data = {
    "evaluator": {"blackbox": alpine},
    "param": {"name": "alpine", "baseline": [5.0, 5.0], "lb": [0.0, 0.0],
              "ub": [10.0, 10.0], "scaling": [10.0, 10.0], "post_dir": "./post"},
    "options": {"seed": 0, "budget": 1000, "tol": 1e-9,
                "check_cache": True, "store_cache": True, "rich_direction": True},
    "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 10},
}

if __name__ == "__main__":
  out, *_ = mads.main(data)
  print("xmin =", out["xmin"])
  print("fmin =", out["fmin"], "after", out["nbb_evals"], "evaluations")
