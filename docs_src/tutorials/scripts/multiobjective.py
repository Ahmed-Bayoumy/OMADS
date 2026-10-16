"""Bi-objective Binh and Korn problem, solved with a Pareto poll study."""
import numpy as np
import pandas as pd
from OMADS import poll


def binh_korn(x):
  f1 = 4.0 * x[0] ** 2 + 4.0 * x[1] ** 2
  f2 = (x[0] - 5.0) ** 2 + (x[1] - 5.0) ** 2
  return [[f1, f2], [0]]


data = {
    "evaluator": {"blackbox": binh_korn},
    "param": {
        "name": "binh_korn",
        "baseline": [2.5, 1.5],
        "lb": [0.0, 0.0],
        "ub": [5.0, 3.0],
        "scaling": [5.0, 3.0],
        "var_names": ["x1", "x2"],
        "fun_names": ["f1", "f2"],
        "nobj": 2,
        "is_pareto": True,
        "ref_point": [140.0, 50.0],
        "post_dir": "./post",
    },
    "options": {"seed": 0, "budget": 1500, "tol": 1e-9,
                "check_cache": True, "store_cache": True, "rich_direction": True,
                "save_results": True},
}


def pareto_front(csv_path, objectives):
  """Non-dominated feasible rows of an OMADS results file."""
  df = pd.read_csv(csv_path, skipinitialspace=True)
  df.columns = df.columns.str.strip()
  df = df[df["Status"].str.strip() == "FEASIBLE"].drop_duplicates(objectives)
  F = df[objectives].to_numpy()
  keep = [not np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F]
  return df[keep].sort_values(objectives[0])


if __name__ == "__main__":
  out, _ = poll.main(data)
  print("hypervolume =", round(out["HV"], 4))

  front = pareto_front("post/binh_korn_run/binh_korn_all.out", ["f1", "f2"])
  print(len(front), "non-dominated points, e.g.")
  print(front[["x1", "x2", "f1", "f2"]].iloc[::len(front) // 5].to_string(index=False))
