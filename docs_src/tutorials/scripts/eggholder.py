"""2D Eggholder function: compare an ACTIVE and a Latin hypercube search."""
import copy
import numpy as np
from OMADS import mads


def eggholder(x):
  a, b = x
  f = (-(b + 47.0) * np.sin(np.sqrt(abs(a / 2.0 + (b + 47.0))))
       - a * np.sin(np.sqrt(abs(a - (b + 47.0)))))
  return [f, [0]]


data = {
    "evaluator": {"blackbox": eggholder},
    "param": {"name": "eggholder", "baseline": [50.0, 280.0],
              "lb": [-512.0, -512.0], "ub": [512.0, 512.0],
              "scaling": [1024.0, 1024.0], "post_dir": "./post"},
    "options": {"seed": 0, "budget": 2000, "tol": 1e-9,
                "check_cache": True, "store_cache": True, "rich_direction": True},
    "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 10},
}

if __name__ == "__main__":
  out, *_ = mads.main(data)
  print(f"ACTIVE search: fmin = {out['fmin'][0]:.4f} at {out['xmin']}")

  lhs = copy.deepcopy(data)
  lhs["search"] = {"type": "sampling", "s_method": "LH", "ns": 30}
  out, *_ = mads.main(lhs)
  print(f"LH search:     fmin = {out['fmin'][0]:.4f} at {out['xmin']}")
