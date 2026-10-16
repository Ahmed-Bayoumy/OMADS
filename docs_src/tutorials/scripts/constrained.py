"""A thin, curved feasible region handled with the progressive barrier."""
import numpy as np
from OMADS import mads


def thin_con(x):
  f = np.sqrt((x[0] - 20.0) ** 2 + (x[1] - 1.0) ** 2)
  c1 = np.sin(x[0]) - 0.1 - x[1]          # x2 >= sin(x1) - 0.1
  c2 = x[1] - np.sin(x[0])                # x2 <= sin(x1)
  return [f, [c1, c2]]


data = {
    "evaluator": {"blackbox": thin_con},
    "param": {
        "name": "thin_con",
        "baseline": [0.0, -10.0],          # infeasible start
        "lb": [0.0, -10.0],
        "ub": [25.0, 10.0],
        "scaling": [25.0, 20.0],
        "constraints_type": ["PB", "PB"],
        "post_dir": "./post",
    },
    "options": {"seed": 0, "budget": 2000, "tol": 1e-9,
                "check_cache": True, "store_cache": True, "rich_direction": True},
    "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 10},
}

if __name__ == "__main__":
  out, *_ = mads.main(data)
  print("xmin =", out["xmin"])
  print("fmin =", out["fmin"], " hmin =", out["hmin"])
  print("evaluations:", out["nbb_evals"])
