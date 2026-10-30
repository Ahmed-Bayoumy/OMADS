"""
# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                               #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the BSD 3-Clause License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the BSD 3-Clause License for more details.   #
#                                                                                     #
#  You should have received a copy of the BSD 3-Clause License along     #
#  with this program. If not, see <https://opensource.org/license/bsd-3-clause/>.     #
#                                                                                     #
#  You can find information on OMADS at                                               #
#  https://github.com/Ahmed-Bayoumy/OMADS                                             #
#  Copyright (C) 2026  Ahmed H. Bayoumy                                               #
# ------------------------------------------------------------------------------------#

Two-phase extreme barrier (``options["constraint_strategy"] = "TWO_PHASE"``),
the constraint handling of DMulti-MADS-TEB (Bigeon, Le Digabel & Salomon):

1. minimize the constraint violation h(x) = sum(max(c_i(x), 0)^2), a
   single-objective problem, until a feasible point is found;
2. from that point, solve the original problem with every constraint
   treated as an extreme barrier.

Both phases run the same solver (``poll.main`` or ``mads.main``) and share
the evaluation budget.
"""

import copy
import sys
from typing import Any, Callable, Dict

import numpy as np


class _FeasiblePointFound(Exception):
  """Raised by the phase-one blackbox at the first feasible point, to end
  phase one."""

  def __init__(self, x, output):
    super().__init__("feasible point found")
    self.x = x
    self.output = output


def requested(data: Dict[str, Any]) -> bool:
  """Whether the problem `data` asks for the two-phase strategy and has
  constraints."""
  strategy = str(data.get("options", {}).get(
      "constraint_strategy", "PB")).upper()
  ctypes = data.get("param", {}).get("constraints_type")
  return strategy == "TWO_PHASE" and bool(ctypes)


def run(main: Callable, data: Dict[str, Any], *args) -> Any:
  """Solve `data` with the solver entry point `main` in two phases (see the
  module docstring); `args` are the extra arguments of `main`. Returns the
  output of phase two, or of phase one when it ends (budget or minimum mesh
  size) without a feasible point."""
  blackbox = data["evaluator"].get("blackbox")
  if not callable(blackbox):
    raise NotImplementedError(
        'constraint_strategy="TWO_PHASE" needs a callable blackbox')
  constants = data["evaluator"].get("constants")

  def call(x):
    out = blackbox(x) if constants is None else blackbox(x, constants)
    if isinstance(out, (float, int)):
      out = [[out], [0]]
    return out

  budget = int(data.get("options", {}).get("budget", 1000))
  n_evals = [0]

  def phase_one_blackbox(x):
    out = call(x)
    n_evals[0] += 1
    f = np.ravel(np.asarray(out[0], dtype=float))
    c = np.ravel(np.asarray(out[1], dtype=float))
    h = float(np.sum(np.maximum(c, 0.) ** 2))
    if h == 0. and np.all(np.isfinite(f)) and np.all(np.isfinite(c)):
      raise _FeasiblePointFound(list(np.asarray(x, dtype=float)), out)
    return [[h if np.isfinite(h) else np.inf], [0.]]

  one = copy.copy(data)
  one["evaluator"] = dict(data["evaluator"], blackbox=phase_one_blackbox)
  one["evaluator"].pop("constants", None)
  one["param"] = dict(data["param"], nobj=1, is_pareto=False,
                      fun_names=["h"], constraints_type=["PB"])
  one["param"].pop("ref_point", None)
  one["options"] = dict(data["options"], constraint_strategy="PB")

  stderr = sys.stderr
  try:
    return main(one, *args)
  except _FeasiblePointFound as found:
    x_feas, out_feas = found.x, found.output
  finally:
    sys.stderr = stderr   # the solvers wrap stderr

  # Phase two: the feasible point is the starting point, and its outputs
  # are replayed instead of evaluated again.
  replay = [True]

  def phase_two_blackbox(x):
    if replay[0] and np.allclose(np.asarray(x, dtype=float), x_feas,
                                 rtol=0., atol=1e-12):
      replay[0] = False
      return out_feas
    return call(x)

  n_cons = len(np.ravel(np.asarray(out_feas[1], dtype=float)))
  two = copy.copy(data)
  two["evaluator"] = dict(data["evaluator"], blackbox=phase_two_blackbox)
  two["evaluator"].pop("constants", None)
  two["param"] = dict(data["param"], baseline=x_feas,
                      constraints_type=["EB"] * n_cons)
  two["options"] = dict(data["options"], constraint_strategy="PB",
                        budget=max(budget - n_evals[0] + 1, 1))
  return main(two, *args)
