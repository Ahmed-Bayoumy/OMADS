"""2D Rosenbrock with the three OMADS solvers."""
import numpy as np
from OMADS import mads, poll, search


def rosenbrock(x):
  x = np.asarray(x)
  f = np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
  return [f, [0]]


data = {
    "evaluator": {"blackbox": rosenbrock},
    "param": {
        "name": "rosen",
        "baseline": [-2.0, -2.0],
        "lb": [-5, -5],
        "ub": [10, 10],
        "var_names": ["x1", "x2"],
        "scaling": [15.0, 15.0],
        "post_dir": "./post",
    },
    "options": {
        "seed": 0, "budget": 1000, "tol": 1e-9,
        "check_cache": True, "store_cache": True, "rich_direction": True,
    },
    "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 10},
}

if __name__ == "__main__":
  for solver in (mads, poll, search):
    out = solver.main(data)[0]
    print(f"{solver.__name__.split('.')[-1]:>6}: xmin = {out['xmin']}, "
          f"fmin = {out['fmin'][0]:.3e}, iterations = {out['niterations']}")
