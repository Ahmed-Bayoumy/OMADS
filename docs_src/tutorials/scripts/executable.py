"""Optimize an external program by wrapping it in a Python callable."""
import os
import subprocess
import sys

import numpy as np
from OMADS import mads

SIM_DIR = os.path.dirname(os.path.abspath(__file__))


def run_simulator(x):
  """Write the input file, run the program, read the output file."""
  np.savetxt(os.path.join(SIM_DIR, "input.inp"), x)
  try:
    subprocess.run([sys.executable, "simulator.py"], cwd=SIM_DIR,
                   check=True, timeout=60, capture_output=True)
    f, *c = np.loadtxt(os.path.join(SIM_DIR, "output.out"), ndmin=1)
  except (subprocess.SubprocessError, OSError, ValueError):
    return [np.inf, [np.inf]]          # a failed run is simply rejected
  return [float(f), [float(ci) for ci in c]]


data = {
    "evaluator": {"blackbox": run_simulator},
    "param": {"name": "simulator", "baseline": [0.0, 0.0], "lb": [-3.0, -3.0],
              "ub": [3.0, 3.0], "scaling": [6.0, 6.0],
              "constraints_type": ["PB"], "post_dir": "./post"},
    "options": {"seed": 0, "budget": 200, "tol": 1e-6,
                "check_cache": True, "store_cache": True, "rich_direction": True},
    "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 4},
}

if __name__ == "__main__":
  out, *_ = mads.main(data)
  print("xmin =", np.round(out["xmin"], 4), " fmin =", np.round(out["fmin"], 5),
        " evaluations =", out["nbb_evals"])
