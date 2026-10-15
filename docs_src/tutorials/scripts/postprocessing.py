"""Save the evaluation history and plot the convergence."""
import numpy as np
import pandas as pd
from OMADS import mads, poll


def rosenbrock(x):
  x = np.asarray(x)
  return [np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2), [0]]


data = {
    "evaluator": {"blackbox": rosenbrock},
    "param": {"name": "rosen_hist", "baseline": [-2.0, -2.0], "lb": [-5, -5],
              "ub": [10, 10], "var_names": ["x1", "x2"], "scaling": [15.0, 15.0],
              "post_dir": "./post"},
    "options": {"seed": 0, "budget": 1000, "tol": 1e-9,
                "check_cache": True, "store_cache": True, "rich_direction": True,
                "save_results": True},
    "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 10},
}

if __name__ == "__main__":
  mads.main(data)

  df = pd.read_csv("post/rosen_hist_run/rosen_hist_all.out", skipinitialspace=True)
  df.columns = df.columns.str.strip()
  df["best"] = df["fobj"].cummin()
  print(df[["Iteration", "Step", "Status", "fobj", "best", "x1", "x2"]]
        .head(6).to_string(index=False))
  print("rows:", len(df), "| evaluations by step:",
        df["Step"].str.strip().value_counts().to_dict())

  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(np.arange(1, len(df) + 1), df["best"] + 1e-16)
    ax.set_xlabel("evaluation")
    ax.set_ylabel("best f")
    fig.tight_layout()
    fig.savefig("post/rosen_convergence.png", dpi=120)
  except ImportError:
    pass

  # Poll trajectory for animations (poll.main only)
  data["options"]["save_coordinates"] = True
  poll.main(data)
  print("coords.json written to post/rosen_hist_run/")
