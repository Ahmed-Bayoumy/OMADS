[![pages-build-deployment](https://github.com/Ahmed-Bayoumy/OMADS/actions/workflows/pages/pages-build-deployment/badge.svg?branch=DEV)](https://github.com/Ahmed-Bayoumy/OMADS/actions/workflows/pages/pages-build-deployment)
[![lx-build-and-pytest](https://github.com/Ahmed-Bayoumy/OMADS/actions/workflows/lx-build-and-tests.yml/badge.svg?branch=DEV)](https://github.com/Ahmed-Bayoumy/OMADS/actions/workflows/lx-build-and-tests.yml)
[![win-build-and-pytest](https://github.com/Ahmed-Bayoumy/OMADS/actions/workflows/win-build-and-pytest.yml/badge.svg?branch=DEV)](https://github.com/Ahmed-Bayoumy/OMADS/actions/workflows/win-build-and-pytest.yml)
[![macos-build-and-pytest](https://github.com/Ahmed-Bayoumy/OMADS/actions/workflows/macos-build-and-pytest.yml/badge.svg?branch=DEV)](https://github.com/Ahmed-Bayoumy/OMADS/actions/workflows/macos-build-and-pytest.yml)

<img width="870" alt="OMADS logo" src="https://ahmed-bayoumy.github.io/OMADS/_static/OMADS_logo.png">

# OMADS

OMADS (Optimization using Mesh Adaptive Direct Search) is a Python implementation of the MADS
algorithm for blackbox optimization. It solves single- and multi-objective problems, with or
without constraints.

Documentation, tutorials and algorithms: [ahmed-bayoumy.github.io/OMADS](https://ahmed-bayoumy.github.io/OMADS/)

---

**Version 2610**

---

## Why OMADS

**Optimize what you can only run.** Your simulation, CAD pipeline or legacy code gives you a
number, not a gradient: it may be noisy, non-smooth, slow, or crash at some inputs. OMADS
optimizes it directly from `pip install` and a ten-line Python dictionary, with no
derivatives, no surrogate to train and no solver to compile.

- **Built on a convergence theory, not a heuristic.** OMADS implements Mesh Adaptive Direct
  Search (MADS), whose iterates converge, under standard assumptions and from any starting
  point, to points satisfying Clarke stationarity conditions.
- **Strong where evaluations are scarce.** On the 54 constrained problems of the COCO
  `bbob-constrained` suite in 2D, the published benchmark configuration solves about
  **98% of the (function, target) pairs within 100 × n evaluations**, ahead of fmincon
  (92%), and close to COBYLA and AL1-CMA-ES, which were run with up to 10⁷ × n evaluations.
- **Constraints the way engineers have them.** Relaxable constraints are steered towards
  feasibility with a progressive barrier; points that violate hard constraints (where the
  code cannot even run) are rejected; failed evaluations are discarded and the search
  carries on.
- **One input, many problems.** The same dictionary or JSON file handles constrained and
  unconstrained, single- and multi-objective (Pareto) problems, serial or parallel
  evaluation, and any Python callable, including wrappers around external programs.
- **Results you can check.** Every published benchmark is reproducible with a single command
  (`python benchmarks/coco/reproduce_results.py`), and a regression suite runs on Linux,
  Windows and macOS in continuous integration.

<p align="center">
  <img width="560" alt="Runtime distributions on COCO bbob-constrained in 2D: OMADS against COBYLA, AL1-CMA-ES, fmincon and random search" src="https://ahmed-bayoumy.github.io/OMADS/_images/ecdf_02D.svg">
</p>

*Fraction of (function, target) pairs solved versus evaluations per variable, over the 54
COCO `bbob-constrained` functions in 2D. `OMADS-A` is the published benchmark configuration
(`OMADS-B` and `OMADS-Hal` are two alternatives); OMADS used at most 100 × n evaluations
(more on the axis, since COCO also counts constraint evaluations), the reference algorithms
up to 10⁷ × n. Details in the
[benchmarking documentation](https://ahmed-bayoumy.github.io/OMADS/benchmarking.html).*

## Features

- Three solvers with one input format: full MADS (`mads`), poll only (`poll`) and search only (`search`)
- Orthogonal Householder poll directions, `2n` (`ORTHO_2N`) or `n+1` (`ORTHO_N+1`)
- Orthogonal mesh or anisotropic granular mesh (`GMESH`)
- Progressive (`PB`) and extreme (`EB`) barriers for constraints
- A cheap speculative step plus a constraint-aware quadratic model search at every iteration
- An `ACTIVE` sampling search (BiTPE density-ratio sampling with Halton, LHS, simulated annealing and particle swarm fallbacks)
- Multi-objective Pareto studies with the hypervolume indicator
- Cache, opportunistic evaluation, parallel evaluation, CSV logs and reproducible seeds

## Installation

```commandline
pip install OMADS
```

## Quick start

```python
import numpy as np
from OMADS import mads

def rosenbrock(x):
    x = np.asarray(x)
    f = np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    return [f, [0]]              # [objective, list of constraints c(x) <= 0]

data = {
    "evaluator": {"blackbox": rosenbrock},
    "param": {"name": "rosen", "baseline": [-2.0, -2.0], "lb": [-5, -5],
              "ub": [10, 10], "scaling": [15, 15], "post_dir": "./post"},
    "options": {"seed": 0, "budget": 1000, "tol": 1e-9, "rich_direction": True,
                "store_cache": True, "check_cache": True},
    "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 10},
}

if __name__ == "__main__":
    out, poll_step, search_step = mads.main(data)
    print(out["xmin"], out["fmin"], out["nbb_evals"])   # [1.0, 1.0] [0.0] 223
```

`poll.main(data)` and `search.main(data)` take the same input and return `(out, sampler)`.
Each solver also accepts the path to a JSON file holding the same dictionary.

## Input at a glance

| Section | Purpose | Main keys |
|---|---|---|
| `evaluator` | How to evaluate a point | `blackbox` (callable or executable), `constants`, `path`, `input`, `output`, `timeout` |
| `param` | The problem | `name`, `baseline`, `lb`, `ub`, `var_names`, `scaling`, `post_dir`, `constraints_type`, `h_max`, `mesh_type`, `nobj`, `is_pareto` |
| `options` | Algorithm and run options | `seed`, `budget`, `tol`, `psize_init`, `store_cache`, `check_cache`, `rich_direction`, `direction_type`, `opportunistic`, `parallel_mode`, `np`, `save_results` |
| `search` | The search step (`mads`, `search`) | `type` (`sampling` or `VNS`), `s_method` (`ACTIVE`, `LH`, `RS`, `HALTON`, `FULLFACTORIAL`), `ns` |

The blackbox receives the design vector and returns `[f, [c1, c2, ...]]`, with constraints
feasible when `c_j(x) <= 0`. Every key, with its default, is listed in the
[configuration reference](https://ahmed-bayoumy.github.io/OMADS/user_guide/configuration.html).

> **Note:** keep `store_cache` and `check_cache` set to `True`; the `mads` and `search` solvers
> need the cache. The `internal` evaluator suites (`"con"`, `"uncon"`) of older releases have been
> removed.

## Multi-objective MADS

Return a list of objectives from the blackbox and set `"nobj"` and `"is_pareto": True` in
`param` to run a Pareto study (use `poll.main` in this release). See the
[multi-objective tutorial](https://ahmed-bayoumy.github.io/OMADS/tutorials/multiobjective.html).
The figures below show fronts found on test problems from the literature.

### Constrained test functions

![BM_res_ConstrainedMOO](https://github.com/user-attachments/assets/033012c6-8e4f-4f8f-b6e3-982ef4ec1f47)

### Unconstrained test functions

![BM_res_UnconstrainedMOO](https://github.com/user-attachments/assets/4b856656-e294-41b0-89ab-ea79818ecf76)

## Benchmarking

- `pytest tests` runs the regression suite (Rosenbrock in 2, 10 and 20 dimensions, constrained
  and geometric-programming problems), with a target value for each solver.
- `benchmarks/coco/run_mads_bbob_constrained.py` runs OMADS on the COCO `bbob-constrained` suite.

See the [benchmarking page](https://ahmed-bayoumy.github.io/OMADS/benchmarking.html).

## Building the documentation

```commandline
pip install -r docs_src/requirements.txt
sphinx-build -b html docs_src docs
```

The sources live in `docs_src/`; the rendered site in `docs/` is served by GitHub Pages.

## License & copyright

© Ahmed H. Bayoumy 2022-2026

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version. See the [LICENSE](LICENSE) file for the full
license text.

## Citation

If you use this code, please cite it as follows.

```bibtex
@software{OMADS_AB,
  author    = {Bayoumy, Ahmed H.},
  title     = {OMADS: Optimization using Mesh Adaptive Direct Search},
  year      = {2022},
  publisher = {GitHub},
  version   = {2610},
  doi       = {10.5281/zenodo.7212701},
  url       = {https://github.com/Ahmed-Bayoumy/OMADS}
}
```

https://github.com/Ahmed-Bayoumy/OMADS/assets/22842095/5dc72c34-4722-4d93-8c84-d17f5595556d
