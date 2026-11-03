"""Plot the Pareto fronts found by OMADS on selected COCO ``bbob-biobj`` problems.

Runs ``poll.main`` on each problem with the settings and restarts of
``run_omads_bbob_biobj.py`` (same budget, seeds and box), records every
evaluation and draws, in COCO's normalized objective space (ideal point at
(0, 0), nadir point at (1, 1)):

  * every point evaluated by OMADS (grey) and its non-dominated points (blue);
  * the non-dominated points of NSGA-II (pymoo, population 100, SBX and
    polynomial mutation) in [-5, 5]^n with the same budget (orange);
  * the hypervolume of both sets and COCO's reference hypervolume.

Writes ``fronts_<D>D.png`` (one panel per problem) and one PNG per problem in
``<out>/fronts/``. Usage, from the repository root:

  python benchmarks/coco/plot_biobj_fronts.py
  python benchmarks/coco/plot_biobj_fronts.py -d 2,5 -f 1,4,28,46,53,55 -i 1 -b 1000
"""
import argparse
import contextlib
import glob
import os
import re
import sys
import tempfile

import cocoex
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_omads_bbob_biobj as runner  # noqa: E402

BBOB_NAMES = {1: "Sphere", 2: "Ellipsoid", 6: "Attractive sector",
              8: "Rosenbrock", 13: "Sharp ridge", 14: "Sum of diff. powers",
              15: "Rastrigin", 17: "Schaffer F7", 20: "Schwefel",
              21: "Gallagher 101"}
COLORS = dict(points="#c9c8c3", omads="#1c5cab", mads="#00a6a6", nsga2="#eb6834",
              box="#52514e")
NSGA2_POPULATION = 100


class Recorder:
  """A COCO problem that keeps the objective vectors of all its evaluations."""

  def __init__(self, problem):
    self._problem = problem
    self.f = []

  def __call__(self, x):
    f = self._problem(x)
    self.f.append(np.array(f, dtype=float))
    return f

  def __getattr__(self, name):
    return getattr(self._problem, name)


def components(problem):
  """The two single-objective bbob problems of a bi-objective problem."""
  out = []
  for name in problem.name.split("__"):
    fid, iid, dim = map(int, re.match(r"bbob_f(\d+)_i(\d+)_d(\d+)", name).groups())
    suite = cocoex.Suite("bbob", f"instances: {iid}",
                         f"dimensions: {dim} function_indices: {fid}")
    out.append((fid, suite, suite.get_problem_by_function_dimension_instance(fid, dim, iid)))
  return out


def best_parameter(problem):
  """Optimal solution of a single-objective bbob problem (cocoex writes it to
  a file in the working directory)."""
  cwd = os.getcwd()
  with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)
    try:
      problem._best_parameter("print")
      return np.loadtxt("._bbob_problem_best_parameter.txt", ndmin=1)
    finally:
      os.chdir(cwd)


def ideal_nadir(problem):
  """Ideal and nadir points of a bbob-biobj problem, as COCO defines them."""
  (fa, _, pa), (fb, _, pb) = components(problem)
  xa, xb = best_parameter(pa), best_parameter(pb)
  ideal = np.array([pa(xa), pb(xb)])
  nadir = np.array([pa(xb), pb(xa)])
  return ideal, nadir, (fa, fb)


def non_dominated(F):
  """Non-dominated rows of F (minimization), sorted by the first objective."""
  F = F[np.argsort(F[:, 0], kind="stable")]
  keep, best = [], np.inf
  for f in F:
    if f[1] < best:
      keep.append(f)
      best = f[1]
  return np.array(keep).reshape(-1, 2)


def hypervolume(front):
  """Hypervolume of a normalized 2-D front with respect to (1, 1)."""
  front = front[(front[:, 0] < 1) & (front[:, 1] < 1)]
  if not len(front):
    return 0.0
  front = non_dominated(front)
  right = np.r_[front[1:, 0], 1.0]
  return float(np.sum((right - front[:, 0]) * (1.0 - front[:, 1])))


def coco_indicator(folder):
  """COCO's reference hypervolume and final indicator from a _hyp.dat file."""
  path = glob.glob(os.path.join(folder, "**", "*_hyp.dat"), recursive=True)[0]
  with open(path, encoding="utf-8") as f:
    text = f.read()
  ref = float(re.search(r"reference value = ([-+0-9.eE]+)", text).group(1))
  rows = [line.split() for line in text.splitlines() if line and line[0] != "%"]
  return ref, float(rows[-1][1]) if rows else np.nan


def run_nsga2(problem, budget, seed):
  """Objective vectors of all the evaluations of NSGA-II on ``problem``."""
  from pymoo.algorithms.moo.nsga2 import NSGA2
  from pymoo.core.problem import Problem
  from pymoo.optimize import minimize

  history = []

  class Wrapped(Problem):
    def __init__(self):
      lo, hi = runner.REGION_OF_INTEREST
      super().__init__(n_var=problem.dimension, n_obj=2, xl=lo, xu=hi)

    def _evaluate(self, X, out, *args, **kwargs):
      F = np.array([problem(x) for x in X], dtype=float)
      history.extend(F)
      out["F"] = F

  minimize(Wrapped(), NSGA2(pop_size=NSGA2_POPULATION), ("n_eval", budget),
           seed=seed, verbose=False)
  return np.array(history[:budget])


def solver_label(solver, args):
  return f"OMADS mads {args.s_method} (ns={args.ns})" if solver == "mads" else "OMADS poll"


def run_problem(fid, dim, iid, args):
  """Run each OMADS solver (observed by COCO) and NSGA-II on one problem."""
  cwd = os.getcwd()
  fronts, runs, deltas = {}, {}, {}
  with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)   # COCO writes exdata/ in the working directory
    try:
      suite = cocoex.Suite("bbob-biobj", "", f"dimensions: {dim}")
      for solver in args.solvers:
        observer = cocoex.Observer("bbob-biobj", f"result_folder: F_{solver} "
                                   "algorithm_name: F log_nondominated: none "
                                   "log_decision_variables: none")
        problem = suite.get_problem_by_function_dimension_instance(fid, dim, iid)
        ideal, nadir, pair = ideal_nadir(problem)
        problem.observe_with(observer)
        rec = Recorder(problem)
        opts = runner.build_parser().parse_args(
            ["-b", str(args.budget_multiplier), "--seed", str(args.seed),
             "--solver", solver, "--ns", str(args.ns), "--s-method", args.s_method])
        opts.omads_post_dir = os.path.join(tmp, "omads_post")
        with contextlib.redirect_stderr(open(os.devnull, "w", encoding="utf-8")):
          runs[solver] = runner.solve(rec, opts)
        title = problem.id
        problem.free()   # flush COCO's files
        ref, deltas[solver] = coco_indicator(observer.result_folder)
        fronts[solver_label(solver, args)] = np.array(rec.f)
      budget = max(len(f) for f in fronts.values())
      ns = run_nsga2(suite.get_problem_by_function_dimension_instance(fid, dim, iid),
                     budget, args.seed)
    finally:
      os.chdir(cwd)
  scale = nadir - ideal
  return dict(id=title, pair=pair, runs=runs, budget=budget, ref=ref, delta=deltas,
              omads={k: (F - ideal) / scale for k, F in fronts.items()},
              nsga2=(ns - ideal) / scale)


def draw(ax, r, legend=False):
  """Fronts of one problem; returns {algorithm: hypervolume}."""
  ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec=COLORS["box"], lw=0.8,
                             ls="--", zorder=1))
  first, F = next(iter(r["omads"].items()))
  ax.scatter(F[:, 0], F[:, 1], s=3, c=COLORS["points"], lw=0, zorder=2,
             label=f"{first}, all evaluations", rasterized=True)
  hvs = {}
  rfront = non_dominated(r["nsga2"])
  ax.plot(rfront[:, 0], rfront[:, 1], ".", ms=4, c=COLORS["nsga2"], zorder=3,
          label="NSGA-II, non-dominated")
  hvs["NSGA-II"] = hypervolume(rfront)
  for k, (label, F) in enumerate(r["omads"].items()):
    front = non_dominated(F)
    color = COLORS["mads"] if "mads" in label else COLORS["omads"]
    ax.plot(front[:, 0], front[:, 1], ".", ms=4, c=color, zorder=5 - k,
            label=f"{label}, non-dominated")
    hvs[label] = hypervolume(front)
  text = "\n".join(f"HV {name} {v:.4f}" for name, v in hvs.items())
  ax.text(0.98, 0.98, f"{text}\nHV reference {r['ref']:.4f}", transform=ax.transAxes,
          ha="right", va="top", fontsize=6.5, color="#0b0b0b",
          bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
  a, b = r["pair"]
  ax.set_title(f"{r['id']}\n{BBOB_NAMES.get(a, a)} / {BBOB_NAMES.get(b, b)}",
               fontsize=8)
  ax.set_xlim(-0.05, 1.3)
  ax.set_ylim(-0.05, 1.3)
  ax.set_aspect("equal")
  ax.grid(True, color="#ebeae6", lw=0.6)
  ax.set_axisbelow(True)
  ax.tick_params(labelsize=7)
  if legend:
    ax.legend(fontsize=6.5, loc="lower left", markerscale=2, framealpha=0.9)
  return hvs


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--dimensions", "-d", default="2",
                      help="comma-separated dimensions (default: 2)")
  parser.add_argument("--functions", "-f", default="1,4,11,28,46,55",
                      help="bbob-biobj functions (default: 1,4,11,28,46,55)")
  parser.add_argument("--instance", "-i", type=int, default=1,
                      help="instance (default: 1)")
  parser.add_argument("--budget-multiplier", "-b", type=float, default=1000,
                      help="budget = this x dimension (default: 1000)")
  parser.add_argument("--seed", type=int, default=1234,
                      help="OMADS base seed, as in the runner (default: 1234)")
  parser.add_argument("--solvers", default="poll",
                      help="comma-separated OMADS solvers to draw: poll, mads or both"
                      " (default: poll)")
  parser.add_argument("--ns", type=int, default=150,
                      help="mads samples per search step (default: 150)")
  parser.add_argument("--s-method", default="ACTIVE",
                      help="mads sampling search method (default: ACTIVE)")
  parser.add_argument("--out", default=os.path.join("ppdata", "biobj", "figures"),
                      help="output folder (default: ppdata/biobj/figures)")
  args = parser.parse_args()
  args.out = os.path.abspath(args.out)
  args.solvers = [v for v in args.solvers.split(",") if v]
  os.makedirs(os.path.join(args.out, "fronts"), exist_ok=True)
  functions = [int(v) for v in args.functions.split(",")]
  plt.rcParams.update({"font.family": "DejaVu Sans"})
  for dim in (int(v) for v in args.dimensions.split(",")):
    results = []
    for fid in functions:
      r = run_problem(fid, dim, args.instance, args)
      results.append(r)
      fig, ax = plt.subplots(figsize=(4.2, 4.2))
      hvs = draw(ax, r, legend=True)
      ax.set_xlabel("normalized $f_1$", fontsize=8)
      ax.set_ylabel("normalized $f_2$", fontsize=8)
      fig.savefig(os.path.join(args.out, "fronts", f"{r['id']}.png"), dpi=200,
                  bbox_inches="tight")
      plt.close(fig)
      print(f"{r['id']}: {r['budget']} evaluations, runs {r['runs']}, HV "
            + ", ".join(f"{k} {v:.4f}" for k, v in hvs.items())
            + f" (reference {r['ref']:.4f})", flush=True)
    cols = 3
    rows = -(-len(results) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.9 * rows), squeeze=False)
    for k, (ax, r) in enumerate(zip(axes.flat, results)):
      draw(ax, r, legend=False)
      if k % cols == 0:
        ax.set_ylabel("normalized $f_2$", fontsize=8)
      if k >= len(results) - cols:
        ax.set_xlabel("normalized $f_1$", fontsize=8)
    for ax in list(axes.flat)[len(results):]:
      ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               markerscale=2, frameon=False)
    fig.suptitle(f"OMADS on bbob-biobj, {dim}-D, instance {args.instance}, "
                 f"{int(args.budget_multiplier)}·n evaluations "
                 "(dashed box: ideal to nadir)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97), h_pad=1.5)
    path = os.path.join(args.out, f"fronts_{dim:02d}D.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
  main()
