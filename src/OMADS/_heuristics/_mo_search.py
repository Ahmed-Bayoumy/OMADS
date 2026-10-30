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

Cheap search steps of DMulti-MADS for multiobjective problems, in the
spirit of the search strategies of NOMAD 4 (Bigeon, Le Digabel & Salomon):

- quadratic model search: quadratics of every objective and inequality
  constraint around the frame center, then the point predicted to dominate
  the frame center the most (the "dominance move"), plus, when the frame
  center is an extreme point of the front, the minimizer of that objective's
  model, which extends the front;
- middle-point search: the point halfway, in the variable space, between
  the frame center and its neighbour across the largest gap of the front.

The helpers below work on plain arrays so that both the poll (``poll.main``)
and the search sampler (``mads.main``) can use them.
"""

from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize


def fit_quads(S: np.ndarray, Y: np.ndarray, ridge: float = 1e-3
              ) -> "Optional[list[tuple[float, np.ndarray, np.ndarray]]]":
  """Quadratic regressions y ~ c + g.s + s.H.s/2 in scaled coordinates of
  every column y of `Y` (same regression points `S`), or None when the fit
  fails. Ridge penalty on the quadratic terms only: with few points this
  behaves like a minimum-Frobenius-norm model, with many it is a lightly
  regularized regression.

  Solved through the normal equations (M'M + ridge P) c = M'y with one
  Cholesky factorization for all columns: an SVD of the regression matrix
  costs about a second per model in 30 variables."""
  Y = np.asarray(Y, dtype=float).reshape(len(S), -1)
  n = S.shape[1]
  iu = np.triu_indices(n, k=1)
  M = np.hstack([np.ones((len(S), 1)), S, 0.5 * S ** 2,
                 S[:, iu[0]] * S[:, iu[1]]])
  n_q = M.shape[1] - n - 1
  A = M.T @ M
  diag = np.concatenate([np.zeros(n + 1), np.full(n_q, ridge)])
  # A tiny relative shift keeps the linear block positive definite when the
  # points span less than the whole space (e.g. a line of starting points).
  A[np.diag_indices_from(A)] += diag + 1e-12 * max(float(np.trace(A)), 1.)
  try:
    coef = np.linalg.solve(A, M.T @ Y)
  except np.linalg.LinAlgError:
    return None
  if not np.all(np.isfinite(coef)):
    return None
  fits = []
  for c in coef.T:
    c_Q = c[n + 1:]
    H = np.diag(c_Q[:n])
    H[iu] = c_Q[n:]
    H[(iu[1], iu[0])] = c_Q[n:]
    fits.append((float(c[0]), c[1:n + 1], H))
  return fits


def fit_quad(S: np.ndarray, y: np.ndarray, ridge: float = 1e-3
             ) -> "Optional[tuple[float, np.ndarray, np.ndarray]]":
  """`fit_quads` for a single response `y`."""
  fits = fit_quads(S, y, ridge)
  return None if fits is None else fits[0]


def convexify(H: np.ndarray, floor: float = 1e-3) -> np.ndarray:
  """Saddle-free Hessian (|eigenvalues|, floored at `floor` times the
  largest): an indefinite model always sends its minimizer to a corner of
  the box; this turns it into a damped Newton step instead."""
  w, V = np.linalg.eigh(H)
  w = np.maximum(np.abs(w), floor * max(float(np.max(np.abs(w))), 1e-12))
  return (V * w) @ V.T


def scaling_vector(scaling, n: int) -> np.ndarray:
  """The per-variable scaling (ub - lb) / factor of a sampler, stored as a
  vector by the search sampler and as a diagonal matrix by the poll."""
  if scaling is None:
    return np.ones(n)
  s = np.asarray(scaling, dtype=float)
  return np.diag(s).copy() if s.ndim == 2 else np.ravel(s)


def mo_model_points(  # noqa: C901
        X: np.ndarray, F: np.ndarray, C: np.ndarray, xk: np.ndarray,
        fk: Sequence[float], fc_feasible: bool, radius: np.ndarray,
        lb: np.ndarray, ub: np.ndarray, half_mesh: np.ndarray,
        extreme_objs: Sequence[int] = (), ridge: float = 1e-3,
        eig_floor: float = 1e-3, reg_dist: float = 2.0, reg_mult: int = 2
) -> List[np.ndarray]:
  """Multiobjective quadratic model search around the frame center `xk`.

  :param X: evaluated points (rows), :param F: their objective values
    (one column per objective), :param C: their inequality constraint
    values (one column per constraint, may have no column)
  :param fk: objective values of the frame center
  :param fc_feasible: whether the frame center is feasible
  :param radius: trust-region half-widths (original units)
  :param half_mesh: half the mesh size per variable (original units)
  :param extreme_objs: objectives in which the frame center is the best
    point of the front; each gets the minimizer of its own model
  :return: up to ``1 + len(extreme_objs)`` points (original units)

  The dominance move solves, over the trust region and subject to the
  constraint models,

    min_{s, t} t   s.t.   m_j(s) - m_j(0) <= t * |scale_j|   for every j,

  and is kept when it predicts a strict decrease of every objective
  (t < 0). An infeasible frame center without such a point gets the
  minimizer of the predicted violation instead.
  """
  n = len(xk)
  m = F.shape[1] if F.ndim == 2 else 0
  if len(X) < n + 2 or m == 0:
    return []
  S = (X - xk) / radius
  dist = np.max(np.abs(S), axis=1)
  n_quad = (n + 1) * (n + 2) // 2
  near = np.argsort(dist, kind="stable")
  near = near[dist[near] <= reg_dist][:reg_mult * n_quad]
  if len(near) < n + 2:
    return []
  S, F = S[near], F[near]
  C = C[near] if C.size else np.zeros((len(near), 0))

  # One regression for every objective and constraint, each scaled by its
  # magnitude on the regression set (objectives relative to the frame center).
  Yf = F - np.asarray(fk, dtype=float)
  Yf = Yf / np.maximum(np.max(np.abs(Yf), axis=0), 1e-12)
  Yc = C / np.maximum(np.max(np.abs(C), axis=0), 1e-12) if C.shape[1] else C
  fits = fit_quads(S, np.hstack([Yf, Yc]), ridge)
  if fits is None:
    return []
  objs = [(g, convexify(H, eig_floor)) for _, g, H in fits[:m]]
  cons = fits[m:]

  def m_obj(j, s):
    g, H = objs[j]
    return float(g @ s + 0.5 * s @ H @ s), g + H @ s

  def m_c(s):
    return np.array([c0 + gc @ s + 0.5 * s @ Hc @ s for c0, gc, Hc in cons])

  def jac_c(s):
    return np.array([gc + Hc @ s for _, gc, Hc in cons]).reshape(-1, n)

  lo = (np.maximum(xk - radius, lb) - xk) / radius
  hi = (np.minimum(xk + radius, ub) - xk) / radius
  bounds = list(zip(lo, hi))
  s0 = np.zeros(n)
  # Projected on the mesh, a model minimizer on the boundary of an active
  # constraint is infeasible about half the time: aim inside by twice the
  # worst-case change of each constraint model over half a mesh step.
  margin = 2. * np.abs(jac_c(s0)) @ (half_mesh / radius) if cons else 0.

  def model_feasible(s):
    return not cons or bool(np.all(m_c(s) <= 1e-9))

  c_cons = [{"type": "ineq", "fun": lambda z: -m_c(z[:n]) - margin,
             "jac": lambda z: np.hstack([-jac_c(z[:n]),
                                         np.zeros((len(cons), 1))])}] \
      if cons else []
  points = []

  # Dominance move, z = (s, t).
  dom_cons = c_cons + [
      {"type": "ineq",
       "fun": lambda z: np.array([z[n] - m_obj(j, z[:n])[0] for j in range(m)]),
       "jac": lambda z: np.array([np.append(-m_obj(j, z[:n])[1], 1.)
                                  for j in range(m)])}]
  try:
    res = minimize(lambda z: (z[n], np.append(np.zeros(n), 1.)),
                   np.zeros(n + 1), jac=True, method="SLSQP",
                   bounds=bounds + [(None, None)], constraints=dom_cons,
                   options={"maxiter": 100})
    s_dom = res.x[:n]
    t_dom = max(m_obj(j, s_dom)[0] for j in range(m))
    if np.all(np.isfinite(s_dom)) and t_dom < -1e-8 and model_feasible(s_dom):
      points.append(s_dom)
  except (ValueError, np.linalg.LinAlgError):
    pass

  if not points and not fc_feasible and cons:
    def m_h(s):
      viol = np.maximum(m_c(s), 0.)
      return float(viol @ viol), 2. * viol @ jac_c(s)
    res = minimize(m_h, s0, jac=True, method="L-BFGS-B", bounds=bounds)
    if np.all(np.isfinite(res.x)) and res.fun < m_h(s0)[0] - 1e-12:
      points.append(res.x)

  # Front extension from an extreme point: minimize that objective alone.
  s_cons = [{"type": "ineq", "fun": lambda s: -m_c(s) - margin,
             "jac": lambda s: -jac_c(s)}] if cons else []
  for j in extreme_objs:
    try:
      res = minimize(lambda s, j=j: m_obj(j, s), s0, jac=True,
                     method="SLSQP", bounds=bounds, constraints=s_cons,
                     options={"maxiter": 100})
    except (ValueError, np.linalg.LinAlgError):
      continue
    s = res.x
    if np.all(np.isfinite(s)) and m_obj(j, s)[0] < -1e-8 and \
            model_feasible(s) and \
            all(np.max(np.abs(s - p)) > 1e-6 for p in points):
      points.append(s)

  return [np.clip(xk + s * radius, lb, ub) for s in points]


def middle_point(front_x: np.ndarray, front_f: np.ndarray,
                 xk: np.ndarray) -> Optional[np.ndarray]:
  """Middle-point search: the point halfway (in the variable space) between
  the frame center `xk` and its neighbour along the front (sorted by the
  first objective) across the larger normalized gap, or None when `xk` is
  not on the front or the front has a single point."""
  if len(front_x) < 2:
    return None
  hit = np.where(np.all(np.isclose(front_x, xk, rtol=0., atol=1e-12), axis=1))[0]
  if len(hit) == 0:
    return None
  order = np.argsort(front_f[:, 0], kind="stable")
  pos = int(np.where(order == hit[0])[0][0])
  span = np.ptp(front_f, axis=0)
  span[span <= 0] = 1.
  best, gap = None, -1.
  for nb in (pos - 1, pos + 1):
    if 0 <= nb < len(order):
      g = float(np.sum(np.abs(front_f[order[nb]] - front_f[order[pos]]) / span))
      if g > gap:
        best, gap = order[nb], g
  if best is None or gap <= 0.:
    return None
  return 0.5 * (front_x[best] + xk)


def mo_cheap_points(hashtable, front: Sequence, fc, frame_size: np.ndarray,
                    mesh_size: np.ndarray, scaling: np.ndarray,
                    lb: Sequence[float], ub: Sequence[float],
                    models: bool = True, middle: bool = True,
                    tr_factor: float = 2.0) -> List[np.ndarray]:
  """Model and middle-point search points of a multiobjective frame center
  `fc` (a CandidatePoint), given the cache, the feasible front (a sequence
  of CandidatePoints) and the frame and mesh sizes of `fc` (scaled units,
  converted with `scaling`)."""
  lb = np.asarray(lb, dtype=float)
  ub = np.asarray(ub, dtype=float)
  xk = np.asarray(fc.coordinates, dtype=float)
  n = len(xk)
  points: List[np.ndarray] = []
  fx = np.asarray([p.coordinates for p in front], dtype=float).reshape(-1, n)
  ff = np.asarray([p.fobj for p in front], dtype=float)
  fc_feasible = fc.is_feasible()
  if models:
    radius = np.maximum(tr_factor * np.asarray(frame_size, dtype=float) * scaling,
                        np.finfo(float).eps)
    half_mesh = 0.5 * np.asarray(mesh_size, dtype=float) * scaling
    X, F, C = hashtable.get_points_objectives_and_c()
    extreme = []
    if fc_feasible and len(ff) >= 2:
      fk_row = np.asarray(fc.fobj, dtype=float)
      extreme = [j for j in range(ff.shape[1])
                 if fk_row[j] <= np.min(ff[:, j])]
    if len(X):
      points.extend(mo_model_points(
          X, F, C, xk, np.asarray(fc.fobj, dtype=float), fc_feasible, radius,
          lb, ub, half_mesh, extreme_objs=extreme))
  if middle and fc_feasible and len(ff) >= 2:
    x_mid = middle_point(fx, ff, xk)
    if x_mid is not None:
      points.append(x_mid)
  return points
