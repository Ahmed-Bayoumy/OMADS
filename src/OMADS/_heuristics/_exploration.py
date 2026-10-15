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
"""
import copy
import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import random

from samplersLib.samplers import LHS
import samplersLib as explore
from .._include import Cache

import numpy as np
from scipy.optimize import minimize

from .._include import BARRIER_TYPES, DType, VAR_TYPE, \
    SUCCESS_TYPES, DESIGN_STATUS, \
    MSG_TYPE, SAMPLING_METHOD, SEARCH_TYPE, DIST_TYPE, STOP_TYPE
from .._include import Mesh
from .._include import CandidatePoint
from .._include import Point
from .._include import AdaptiveBarrier
from .._directions._directions import Dirs2n
from .._setup._parameters import Parameters
from .._templates._optimizer import GenericSamplerBase, GenericSamplerBaseData
from .._analytics._metadata import MadsStatistics


def _bitpe_supports_reduction() -> bool:
  """Whether the installed samplersLib's BiTPE takes `reduction_method` /
  `n_reduced_dims`. Released samplersLib (<= 2604.0) does not: passing them
  raised a TypeError that BiTPE's construction fallback silently turned into
  trust-region Halton sampling at every search step."""
  try:
    params = inspect.signature(explore.samplers.BiTPE.__init__).parameters
  except (TypeError, ValueError):
    return False
  return "reduction_method" in params and "n_reduced_dims" in params


_BITPE_REDUCES = _bitpe_supports_reduction()


@dataclass(slots=True)
class VNSData:
  fixed_vars: Optional[List[CandidatePoint]] = None
  nb_search_pts: int = 0
  stop: bool = False
  stop_reason: STOP_TYPE = STOP_TYPE.NO_STOP
  success: SUCCESS_TYPES = SUCCESS_TYPES.US
  count_search: bool = False
  new_feas_inc: Optional[CandidatePoint] = None
  new_infeas_inc: Optional[CandidatePoint] = None
  params: Optional[Parameters] = None


@dataclass(slots=True)
class VNS(VNSData):
  """ 
  """
  _k: int = 1
  _k_max: int = 100
  _old_x: Optional[CandidatePoint] = None
  _dist: Optional[List[DIST_TYPE]] = None
  _ns_dist: Optional[List[int]] = None
  _rho: float = 0.1
  _seed: int = 12345
  x_inc: CandidatePoint = None

  def __init__(
          self, stop: bool = False, params=None, x_inc: CandidatePoint = None, seed: int = 12345, rho: int = 0.1):
    self.stop = stop
    self.count_search = not self.stop
    self._dist = [DIST_TYPE.GAUSS, DIST_TYPE.GAMMA,
                  DIST_TYPE.EXPONENTIAL, DIST_TYPE.POISSON]
    self.params = params
    self.x_inc = x_inc
    self._seed = seed
    self._rho = rho

  @property
  def dist(self):
    return self._dist

  @dist.setter
  def dist(self, value: List[DIST_TYPE]) -> List[DIST_TYPE]:
    self._dist = value

  @property
  def ns_dist(self):
    return self._ns_dist

  @ns_dist.setter
  def ns_dist(self, value: List[int]) -> List[int]:
    self._ns_dist = value

  def draw_from_gauss(self, mean: CandidatePoint) -> List[CandidatePoint]:
    """_summary_
    """
    np.random.seed(self._seed)
    cs = np.zeros((self._ns_dist[0], mean.n_dimensions))
    for i in range(mean.n_dimensions):
      if mean.var_type is not None:
        if mean.var_type[i] == VAR_TYPE.REAL:
          cs[:, i] = np.random.normal(
              loc=mean.coordinates[i], scale=self._rho, size=(self._ns_dist[0],))
        elif mean.var_type[i] == VAR_TYPE.INTEGER or mean.var_type[i] == VAR_TYPE.DISCRETE:
          cs[:, i] = np.random.randint(low=int(
              mean.coordinates[i] - self._rho),
              high=int(
                  np.ceil(
                      mean.coordinates[i] +
                      (self._rho if self._rho > 0 else 0.001))),
              size=(self._ns_dist[0],))
        elif mean.var_type[i] == VAR_TYPE.CATEGORICAL:
          stemp = np.linspace(self.params.lb[i], self.params.ub[i], int(
              self.params.ub[i]-self.params.lb[i])+1).tolist()
          cs[:len(stemp), i] = random.sample(stemp, len(stemp))
        else:
          cs[:, i] = [mean.coordinates[i]]*self._ns_dist[0]
      else:
        cs[:, i] = np.random.normal(
            loc=mean.coordinates[i], scale=self._rho, size=(self._ns_dist[0],))

    return cs

  def draw_from_gamma(self, mean: CandidatePoint) -> List[CandidatePoint]:
    """_summary_
    """
    np.random.seed(self._seed)
    cs = np.zeros((self._ns_dist[1], mean.n_dimensions))
    for i in range(mean.n_dimensions):
      val = mean.coordinates
      delta = 0.
      if val[i] <= 0. or 0 < val[i] <= 0.5:
        delta = 5 - val[i]
      if mean.var_type is not None:
        if mean.var_type[i] == VAR_TYPE.REAL:
          cs[:, i] = np.random.gamma(shape=(
              mean.coordinates[i]+delta)/self._rho, scale=self._rho, size=(self._ns_dist[1],))-delta
        elif mean.var_type[i] == VAR_TYPE.INTEGER or \
            mean.var_type[i] == VAR_TYPE.CATEGORICAL or \
                mean.var_type[i] == VAR_TYPE.DISCRETE:
          cs[:, i] = np.random.randint(low=int(
              mean.coordinates[i] - self._rho),
              high=int(
                  np.ceil(
                      mean.coordinates[i] +
                      (self._rho if self._rho > 0 else 0.001))),
              size=(self._ns_dist[1],))
        else:
          cs[:, i] = [mean.coordinates[i]]*self._ns_dist[1]
      else:
        cs[:, i] = np.random.gamma(shape=(
            mean.coordinates[i]+delta)/self._rho, scale=self._rho, size=(self._ns_dist[1],))-delta

    return cs

  def draw_from_exp(self, mean: CandidatePoint) -> List[CandidatePoint]:
    """_summary_
    """
    np.random.seed(self._seed)
    cs = np.zeros((self._ns_dist[2], mean.n_dimensions))
    # pts: List[Point] = [Point()] * self._ns_dist[2]
    for i in range(mean.n_dimensions):
      if mean.var_type is not None:
        if mean.var_type[i] == VAR_TYPE.REAL:
          cs[:, i] = (np.random.exponential(scale=self._rho,
                      size=self._ns_dist[2]))+mean.coordinates[i]
        elif mean.var_type[i] == VAR_TYPE.INTEGER or \
            mean.var_type[i] == VAR_TYPE.CATEGORICAL or \
                mean.var_type[i] == VAR_TYPE.DISCRETE:
          cs[:, i] = np.random.randint(low=int(
              mean.coordinates[i] - self._rho),
              high=int(
                  np.ceil(
                      mean.coordinates[i] +
                      (self._rho if self._rho > 0 else 0.001))),
              size=(self._ns_dist[2],))
        else:
          cs[:, i] = [mean.coordinates[i]]*self._ns_dist[2]
      else:
        cs[:, i] = (np.random.exponential(scale=self._rho,
                    size=self._ns_dist[2]))+mean.coordinates[i]

    return cs

  def draw_from_poisson(self, mean: CandidatePoint) -> List[CandidatePoint]:
    """_summary_
    """
    np.random.seed(self._seed)
    cs = np.zeros((self._ns_dist[3], mean.n_dimensions))
    for i in range(mean.n_dimensions):
      val = mean.coordinates
      delta = 0.
      if val[i] <= 0. or 0 < val[i] <= 0.5:
        delta = 5 - val[i]
      if mean.var_type is not None:
        if mean.var_type[i] == VAR_TYPE.REAL:
          cs[:, i] = (np.random.poisson(
              lam=(mean.coordinates[i] + delta),
              size=(self._ns_dist[3],)) - delta) * self._rho
        elif mean.var_type[i] == VAR_TYPE.INTEGER or \
            mean.var_type[i] == VAR_TYPE.CATEGORICAL or \
                mean.var_type[i] == VAR_TYPE.DISCRETE:
          cs[:, i] = np.random.randint(low=int(
              mean.coordinates[i] - self._rho),
              high=int(
                  np.ceil(
                      mean.coordinates[i] +
                      (self._rho if self._rho > 0 else 0.001))),
              size=(self._ns_dist[3],))
        else:
          cs[:, i] = [mean.coordinates[i]]*self._ns_dist[3]
      else:
        cs[:, i] = (np.random.poisson(
            lam=(mean.coordinates[i] + delta),
            size=(self._ns_dist[3],)) - delta) * self._rho

    return cs

  def draw_from_binomial(self, mean: CandidatePoint) -> List[CandidatePoint]:
    """_summary_
    """
    np.random.seed(self._seed)
    cs = np.zeros((self._ns_dist[4], mean.n_dimensions))
    for i in range(mean.n_dimensions):
      val = mean.coordinates
      delta = 0.
      if val[i] <= 0. or 0 < val[i] <= 0.5:
        delta = 5 - val[i]
      if mean.var_type is not None:
        if mean.var_type[i] == VAR_TYPE.REAL:
          cs[:, i] = (np.random.binomial(
              n=(mean.coordinates[i] + delta) /
              ((1 / self._rho) if self._rho > 1. else self._rho),
              p=(1 / self._rho) if self._rho > 1. else self._rho,
              size=(self._ns_dist[4],)) - delta)
        elif mean.var_type[i] == VAR_TYPE.INTEGER or \
                mean.var_type[i] == VAR_TYPE.CATEGORICAL or \
            mean.var_type[i] == VAR_TYPE.DISCRETE:
          cs[:, i] = np.random.randint(low=int(
              mean.coordinates[i] - self._rho),
              high=int(
                  np.ceil(
                      mean.coordinates[i] +
                      (self._rho if self._rho > 0 else 0.001))),
              size=(self._ns_dist[4],))
        else:
          cs[:, i] = [mean.coordinates[i]]*self._ns_dist[4]
      else:
        cs[:, i] = (np.random.binomial(
            n=(mean.coordinates[i] + delta) /
            ((1 / self._rho) if self._rho > 1. else self._rho),
            p=(1 / self._rho) if self._rho > 1. else self._rho,
            size=(self._ns_dist[4],)) - delta)

    return cs

  def generate_samples(
          self, active_barrier: AdaptiveBarrier = None,
          dist: DIST_TYPE = None) -> Optional[
          List[float]]:
    """_summary_
    """
    if self.x_inc is None:
      if isinstance(active_barrier, AdaptiveBarrier):
        self.x_inc = active_barrier.elements()[-1]
      else:
        raise IOError("Unrecognized barrier type has been used!")
    if self.x_inc is None or not self.x_inc.evaluated:
      return None
    else:
      if dist == DIST_TYPE.GAUSS:
        return self.draw_from_gauss(self.x_inc)

      if dist == DIST_TYPE.GAMMA:
        return self.draw_from_gamma(self.x_inc)

      if dist == DIST_TYPE.EXPONENTIAL:
        return self.draw_from_exp(self.x_inc)

      if dist == DIST_TYPE.POISSON:
        return self.draw_from_poisson(self.x_inc)

      if dist == DIST_TYPE.BIONOMIAL:
        return self.draw_from_binomial(self.x_inc)

    return None

  def run(self, x: CandidatePoint, n_dist: List[int]):
    if self.stop:
      return
    self._ns_dist = n_dist
    samples = np.zeros((sum(self._ns_dist), len(self.params.baseline)))
    c = 0
    self._seed += np.random.randint(0, 10000)

    if x.status is DESIGN_STATUS.FEASIBLE:
      for i, _ in enumerate((self._dist)):
        temp = self.generate_samples(dist=self._dist[i])
        if temp is None:
          continue
        temp = np.unique(temp, axis=0)
        for p in temp:
          if p not in samples:
            samples[c, :] = p
            c += 1

    return samples


class EfficientExploration(GenericSamplerBaseData, GenericSamplerBase):
  """Efficient exploration class

  :param GenericSamplerBase: Generic sampler base class
  :type GenericSamplerBase: Class object
  """

  def __init__(self):
    super().__init__()
    self._xmin = CandidatePoint()
    self.bb_eval = int
    self._dtype = DType()
    self.explore_new = False
    self.nds = 0
    self.search_trial: int = 0
    self.diverse_intense_trial_period = 2
    self.nvars = None
    self.ns = 0
    self.psize = None
    # --- BiTPE confidence/stance policy state (see _bitpe_readiness,
    # _exploitation_stance, _update_bitpe_trust below) -----------------
    self.last_success_trial: int = 0
    self._last_bitpe_coords: Optional[np.ndarray] = None
    self._bitpe_recent_hits: List[bool] = []

  def _bitpe_readiness(
          self, n_good: int, n_bad: int, good_f: np.ndarray,
          bad_f: np.ndarray, n_d_eff: int) -> "tuple[bool, str]":
    """Confidence gate: is there enough *separable* data for BiTPE's
    good/bad KDE split to mean anything, for the dimensionality it will
    actually fit in (`n_d_eff` is the post-reduction dimension, not the
    raw problem dimension)?

    Two independent checks, both must pass:
    - sample sufficiency scaled to `n_d_eff` (a flat "5 points" threshold
      is fine at 2D but meaningless once BiTPE is fitting a 7-14D KDE);
    - class separation: "good" only means something if the good/bad
      f-value buckets are actually distinguishable, else the density
      ratio BiTPE resamples from is noise, not signal.
    """
    min_pts = max(5, 3 * max(n_d_eff, 1))
    if n_good < min_pts or n_bad < 2:
      return False, f"insufficient data ({n_good} good pts, need >= {min_pts})"
    all_f = np.concatenate([np.ravel(good_f), np.ravel(bad_f)])
    spread = float(np.std(all_f))
    if spread <= 0:
      return False, "no spread in objective values yet"
    gap = float(np.median(bad_f) - np.median(good_f))
    if gap < 0.05 * spread:
      return False, f"good/bad not separable (gap={gap:.3g}, spread={spread:.3g})"
    return True, "confident"

  def _exploitation_stance(self, last_success: SUCCESS_TYPES) -> str:
    """Aggressive-local vs conservative-global exploitation stance, from
    two signals that are already tracked nearby: how coarse the mesh
    still is, and how long it's been since the last actual improvement
    (stagnation). Aggressive narrows in on the current incumbent;
    conservative widens the net instead of continuing to dig in a spot
    that has stopped paying off.
    """
    if last_success != SUCCESS_TYPES.US:
      self.last_success_trial = self.search_trial
    stagnation = self.search_trial - self.last_success_trial
    mesh_coarse = np.linalg.norm(
        self.mesh.get_delta_frame_size().coordinates) > 1
    if mesh_coarse or stagnation >= self.diverse_intense_trial_period:
      return "conservative"
    return "aggressive"

  def _update_bitpe_trust(
          self, center_points: np.ndarray, window: int = 20,
          min_samples: int = 10) -> float:
    """Rolling track record for BiTPE's *previous* proposal: of the points
    it drew last time this ran, what fraction ended up in the current
    "good" bucket once evaluated? This is a track-record signal on top of
    (not instead of) `_bitpe_readiness` -- the data can look separable and
    sufficient while BiTPE's actual picks keep missing, e.g. right after a
    kernel/bandwidth mismatch for the current mesh scale.

    This only *tunes* how BiTPE gets used (kernel width, alongside
    `_exploitation_stance`) -- deliberately not a hard on/off gate.
    `_bitpe_readiness` alone decides whether BiTPE runs at all; if low
    trust also veto'd BiTPE outright, a single unlucky round would starve
    the model of the very data it needs to earn trust back, since it
    would never run again to refresh `_last_bitpe_coords`.

    Each previous proposal is checked exactly once (against whatever
    "good" bucket exists the next time this is called), not on every
    subsequent call -- the top-30% "good" bucket is relative and shrinks
    as more points accumulate, so re-checking the same stale coordinates
    against an ever-tightening bar would silently decay trust toward zero
    regardless of how good those points actually were when proposed.

    Returns 1.0 (fully trusted) until there are at least `min_samples` of
    real history to judge -- an untested or barely-tested model isn't
    distrusted, just unproven.
    """
    if self._last_bitpe_coords is not None and len(self._last_bitpe_coords) > 0:
      good_set = {tuple(np.round(row, 12)) for row in center_points} \
          if len(center_points) > 0 else set()
      hits = [tuple(np.round(row, 12)) in good_set
              for row in self._last_bitpe_coords]
      self._bitpe_recent_hits.extend(hits)
      self._bitpe_recent_hits = self._bitpe_recent_hits[-window:]
    self._last_bitpe_coords = None
    if len(self._bitpe_recent_hits) < min_samples:
      return 1.0
    return sum(self._bitpe_recent_hits) / len(self._bitpe_recent_hits)

  def _trust_region(
          self, fc: CandidatePoint, v: np.ndarray,
          factor: float = 2.0, min_size: float = 0.
  ) -> "tuple[np.ndarray, np.ndarray]":
    """Trust region centred on the incumbent `fc` (TuRBO-style, Eriksson et
    al. 2019), sized from the MADS frame so it grows/shrinks with the same
    success/failure logic the poll already uses -- no separate radius
    bookkeeping. The frame size lives in scaled units; `self.scaling`
    ((ub - lb) / scaling, set by the preprocessor) converts it to the
    original units the samplers work in, exactly as the poll directions do.

    `min_size` (scaled units, like the frame) lets a caller keep the region
    larger than the frame, e.g. a model-search radius adapted by a ratio
    test.

    Returns the box clipped to the global bounds `v`, and the unclipped
    per-variable half-widths (used as the distance metric / bandwidth).
    """
    frame = np.asarray(
        self.mesh.get_delta_frame_size().coordinates, dtype=float).ravel()
    scaling = np.ravel(np.asarray(
        getattr(self, "scaling", np.ones_like(frame)), dtype=float))
    radius = np.maximum(np.maximum(factor * frame, min_size) * scaling,
                        np.finfo(float).eps)
    x = np.asarray(fc.coordinates, dtype=float)
    box = np.column_stack([np.maximum(x - radius, v[:, 0]),
                           np.minimum(x + radius, v[:, 1])])
    return box, radius

  def _local_split(
          self, x_better: List, f_better: List, x_worse: List,
          f_worse: List, fc: CandidatePoint, radius: np.ndarray,
          n_d_eff: int, gamma: float = 0.3) -> "tuple[np.ndarray, ...]":
    """Good/bad split restricted to the incumbent's neighbourhood.

    The global top-30% split spans most of the box early on (and in 10D+
    stays that way for the whole budget), so BiTPE's base points -- drawn
    from the good set -- never concentrate near the incumbent, whatever
    `scale` is passed. Instead keep the k points nearest `fc` (inf-norm in
    trust-region units) and split those, with k sized so the good share
    meets `_bitpe_readiness`'s minimum. The cache's rank order (feasible
    by f, then infeasible by h) is preserved, so the split stays
    constraint-aware.
    """
    xs = list(x_better) + list(x_worse)
    if not xs:
      empty = np.empty((0, self._n))
      return empty, np.empty(0), empty, np.empty(0)
    X = np.asarray(xs, dtype=float)
    F = np.ravel(np.asarray(list(f_better) + list(f_worse), dtype=float))
    dist = np.max(np.abs(X - np.asarray(fc.coordinates, dtype=float)) / radius,
                  axis=1)
    min_good = max(5, 3 * max(n_d_eff, 1))
    k = min(len(X), max(int(np.ceil(min_good / gamma)), 2 * (self.dim + 1)))
    idx = np.sort(np.argsort(dist, kind="stable")[:k])
    n_good = max(int(np.ceil(gamma * len(idx))), 1)
    good, bad = idx[:n_good], idx[n_good:]
    return X[good], F[good], X[bad], F[bad]

  def generate_quad_model_points(
          self, hashtable: Cache, fc: CandidatePoint,
          x_parent: Optional[List[float]] = None):
    """Cheap search, meant to run first every iteration (at most two
    evaluations):

    - speculative search: right after a success (`x_parent` is the point
      the incumbent was generated from), repeat that successful step once
      more from the incumbent;
    - quadratic model search (see `_quad_model_point`).

    Single-objective only. With constraints (`constraints_type` set), the
    constraints are modelled too and the frame center may be infeasible
    (see `_constrained_quad_model_point`); otherwise the incumbent must be
    feasible. Leaves the candidate set empty when there is nothing worth
    evaluating.
    """
    self._candidate_points_set = []
    constrained = bool(self.prob_params.constraints_type)
    if self.prob_params.is_pareto or fc is None or fc.status not in (
            (DESIGN_STATUS.FEASIBLE, DESIGN_STATUS.INFEASIBLE) if constrained
            else (DESIGN_STATUS.FEASIBLE,)):
      return
    v = np.column_stack([np.asarray(self.prob_params.lb, dtype=float),
                         np.asarray(self.prob_params.ub, dtype=float)])
    xk = np.asarray(fc.coordinates, dtype=float)
    points = []
    if x_parent is not None:
      step = xk - np.asarray(x_parent, dtype=float)
      if np.any(step != 0):
        points.append(np.clip(xk + step, v[:, 0], v[:, 1]))
    x_model = self._constrained_quad_model_point(hashtable, fc, v) \
        if constrained else self._quad_model_point(hashtable, fc, v)
    if x_model is not None:
      points.append(x_model)
    if points:
      self.map_samples_from_coords_to_points(np.asarray(points), fc=fc)

  def _quad_model_point(  # noqa: C901
          self, hashtable: Cache, fc: CandidatePoint, v: np.ndarray,
          ridge: float = 1e-3) -> Optional[np.ndarray]:
    """Quadratic model search (Conn & Le Digabel 2013; NOMAD's default
    search step): fit a quadratic to feasible cached points around the
    incumbent and return the model's minimizer over the trust region, or
    None when the model can't be built or predicts no decrease.

    Ridge-regularized regression on the quadratic terms (MFN-like when
    points are scarce), convexified before minimizing.
    """
    X, F = hashtable.get_feasible_points_and_f()
    n = self.dim
    if len(X) < n + 2:
      return
    tr_v, radius = self._trust_region(fc, v)
    xk = np.asarray(fc.coordinates, dtype=float)
    S = (X - xk) / radius
    dist = np.max(np.abs(S), axis=1)
    # Regression set: nearest points within twice the trust region, up to
    # twice what a fully determined quadratic needs -- exact interpolation
    # of poll points (which lie on near-orthogonal lines) identifies the
    # cross terms poorly.
    n_quad = (n + 1) * (n + 2) // 2
    near = np.argsort(dist, kind="stable")
    near = near[dist[near] <= 2.0][:2 * n_quad]
    if len(near) < n + 2:
      return
    S, F = S[near], F[near]
    fk = float(np.min(F))
    f_scale = max(float(np.max(np.abs(F - fk))), 1e-12)
    y = (F - fk) / f_scale
    fit = self._fit_quad(S, y, ridge)
    if fit is None:
      return
    g, H = fit[1], self._convexify(fit[2])

    # Minimize over the trust region, which is [-1, 1] per variable in
    # these scaled coordinates, intersected with the global bounds.
    lo = (tr_v[:, 0] - xk) / radius
    hi = (tr_v[:, 1] - xk) / radius
    s_best = S[np.argmin(y)]

    def model(s):
      return float(g @ s + 0.5 * s @ H @ s), g + H @ s

    best_s, best_m = None, model(np.zeros(n))[0]
    for s0 in (np.zeros(n), np.clip(s_best, lo, hi)):
      res = minimize(model, s0, jac=True, method="L-BFGS-B",
                     bounds=list(zip(lo, hi)))
      if res.fun < best_m - 1e-12:
        best_s, best_m = res.x, res.fun
    if best_s is None:
      return
    return np.clip(xk + best_s * radius, v[:, 0], v[:, 1])

  # Constraint-aware quadratic model search: ridge on the quadratic terms,
  # eigenvalue floor (relative to the largest) of the convexified objective
  # Hessian, and the regression set (points within `_model_reg_dist` trust
  # region radii, at most `_model_reg_mult` times a full quadratic's size).
  _model_ridge: float = 1e-3
  _model_eig_floor: float = 1e-3
  _model_reg_dist: float = 2.0
  _model_reg_mult: int = 2

  @staticmethod
  def _fit_quad(S: np.ndarray, y: np.ndarray, ridge: float = 1e-3
                ) -> "Optional[tuple[float, np.ndarray, np.ndarray]]":
    """Quadratic regression y ~ c + g.s + s.H.s/2 in scaled coordinates,
    or None when it fails. Ridge penalty on the quadratic terms only: with
    few points this behaves like a minimum-Frobenius-norm model, with many
    it is a lightly regularized regression."""
    n = S.shape[1]
    iu = np.triu_indices(n, k=1)
    M_L = np.hstack([np.ones((len(S), 1)), S])
    M_Q = np.hstack([0.5 * S ** 2, S[:, iu[0]] * S[:, iu[1]]])
    n_q = M_Q.shape[1]
    A = np.vstack([np.hstack([M_L, M_Q]),
                   np.hstack([np.zeros((n_q, n + 1)),
                              np.sqrt(ridge) * np.eye(n_q)])])
    try:
      coef = np.linalg.lstsq(A, np.concatenate([y, np.zeros(n_q)]),
                             rcond=None)[0]
    except np.linalg.LinAlgError:
      return None
    if not np.all(np.isfinite(coef)):
      return None
    c_Q = coef[n + 1:]
    H = np.diag(c_Q[:n])
    H[iu] = c_Q[n:]
    H[(iu[1], iu[0])] = c_Q[n:]
    return float(coef[0]), coef[1:n + 1], H

  @staticmethod
  def _convexify(H: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Saddle-free Hessian (|eigenvalues|, floored at `floor` times the
    largest): an indefinite model always sends its minimizer to a corner of
    the box; this turns it into a damped Newton step instead."""
    w, V = np.linalg.eigh(H)
    w = np.maximum(np.abs(w), floor * max(float(np.max(np.abs(w))), 1e-12))
    return (V * w) @ V.T

  def _constrained_quad_model_point(  # noqa: C901
          self, hashtable: Cache, fc: CandidatePoint, v: np.ndarray,
          ridge: Optional[float] = None) -> Optional[np.ndarray]:
    """Constraint-aware quadratic model search (NOMAD's QUAD_MODEL_SEARCH
    with constraints): fit quadratics to the objective and to each
    inequality constraint around the frame center, then return

    - the minimizer of the objective model subject to the constraint models
      (c_i <= 0) over the trust region, when it predicts a decrease from the
      frame center (or a feasible point when the frame center is not);
    - otherwise, for an infeasible frame center, the minimizer of the
      predicted violation sum(max(c_i, 0)^2) when it predicts a reduction.

    Without constraint models, the f-model minimizer nearly always falls on
    the infeasible side of an active constraint, leaving only the poll to
    crawl along it.
    """
    X, F, C = hashtable.get_points_f_and_c()
    n = self.dim
    self._update_model_radius(X, F, C)
    if len(X) < n + 2 or C.shape[1] == 0:
      return
    tr_v, radius = self._trust_region(fc, v, min_size=self._model_radius)
    if not np.all(np.isfinite(radius)):
      return
    xk = np.asarray(fc.coordinates, dtype=float)
    S = (X - xk) / radius
    dist = np.max(np.abs(S), axis=1)
    n_quad = (n + 1) * (n + 2) // 2
    near = np.argsort(dist, kind="stable")
    near = near[dist[near] <= self._model_reg_dist][:self._model_reg_mult * n_quad]
    if len(near) < n + 2:
      return
    S, F, C = S[near], F[near], C[near]

    fk = float(np.min(F))
    f_scale = max(float(np.max(np.abs(F - fk))), 1e-12)
    if ridge is None:
      ridge = self._model_ridge
    fit = self._fit_quad(S, (F - fk) / f_scale, ridge)
    if fit is None:
      return
    g, H = fit[1], self._convexify(fit[2], self._model_eig_floor)
    # Constraint models keep their curvature (no convexification) and their
    # zero level; each is scaled by its magnitude on the regression set.
    c_models = []
    for i in range(C.shape[1]):
      c_scale = max(float(np.max(np.abs(C[:, i]))), 1e-12)
      fit_c = self._fit_quad(S, C[:, i] / c_scale, ridge)
      if fit_c is None:
        return
      c_models.append(fit_c)

    def m_f(s):
      return float(g @ s + 0.5 * s @ H @ s), g + H @ s

    def m_c(s):
      return np.array([c0 + gc @ s + 0.5 * s @ Hc @ s
                       for c0, gc, Hc in c_models])

    def jac_c(s):
      return np.array([gc + Hc @ s for _, gc, Hc in c_models])

    def m_h(s):
      viol = np.maximum(m_c(s), 0.)
      return float(viol @ viol), 2. * viol @ jac_c(s)

    lo = (tr_v[:, 0] - xk) / radius
    hi = (tr_v[:, 1] - xk) / radius
    bounds = list(zip(lo, hi))
    s0 = np.zeros(n)
    starts = [s0]
    feasible_rows = np.all(C <= 0., axis=1)
    if np.any(feasible_rows):
      starts.append(np.clip(S[feasible_rows][np.argmin(F[feasible_rows])],
                            lo, hi))
    fc_feasible = fc.status == DESIGN_STATUS.FEASIBLE
    best_s, best_m = None, m_f(s0)[0] if fc_feasible else np.inf
    # The model minimizer sits on the boundary of an active constraint, so
    # projecting it on the mesh makes it infeasible about half the time.
    # Aim inside by twice the worst-case change of each constraint model
    # over half a mesh step.
    scaling = np.ravel(np.asarray(
        getattr(self, "scaling", np.ones(n)), dtype=float))
    half_mesh_s = 0.5 * np.ravel(np.asarray(
        self.mesh.get_delta_mesh_size().coordinates, dtype=float)) \
        * scaling / radius
    margin = 2. * np.abs(jac_c(s0)) @ half_mesh_s
    cons = [{"type": "ineq", "fun": lambda s: -m_c(s) - margin,
             "jac": lambda s: -jac_c(s)}]
    for start in starts:
      try:
        res = minimize(m_f, start, jac=True, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 100})
      except (ValueError, np.linalg.LinAlgError):
        continue
      if np.all(np.isfinite(res.x)) and np.all(m_c(res.x) <= 1e-9) and \
              res.fun < best_m - 1e-12:
        best_s, best_m = res.x, res.fun
    if best_s is None and not fc_feasible:
      h0 = m_h(s0)[0]
      res = minimize(m_h, s0, jac=True, method="L-BFGS-B", bounds=bounds)
      if np.all(np.isfinite(res.x)) and res.fun < h0 - 1e-12:
        best_s = res.x
    if best_s is None:
      return
    x_model = np.clip(xk + best_s * radius, v[:, 0], v[:, 1])
    if fc_feasible:
      self._last_model_step = (
          x_model, float(np.sum(fc.fobj)), -best_m * f_scale,
          float(np.max(radius / scaling)),
          bool(np.max(np.abs(best_s)) >= 0.9), radius)
    return x_model

  # Model-search radius (scaled units) kept on top of the frame-sized trust
  # region, and the last model step awaiting its ratio test.
  _model_radius: float = 0.
  _last_model_step: Optional[tuple] = None

  def _update_model_radius(self, X: np.ndarray, F: np.ndarray,
                           C: np.ndarray):
    """Classic trust-region ratio test on the previous constraint-aware
    model step: actual over predicted decrease of f. A well-predicted step
    that reached the edge of the region doubles the model radius, a poorly
    predicted (or infeasible) one halves it.

    The frame shrinks whenever the poll fails, which along the long, flat
    valleys of ill-conditioned problems is most of the time; tying the
    model region to the frame alone then caps model steps far below the
    distance left to the optimum.
    """
    last, self._last_model_step = self._last_model_step, None
    if last is None or len(X) == 0:
      return
    x_prop, f_old, pred, r_used, at_edge, radius = last
    # The evaluated point is the proposal projected on the mesh.
    dev = np.max(np.abs(X - x_prop) / radius, axis=1)
    i = int(np.argmin(dev))
    if dev[i] > 1e-2 or pred <= 0.:
      return
    ratio = (f_old - F[i]) / pred if np.all(C[i] <= 0.) else -np.inf
    if ratio >= 0.75 and at_edge:
      self._model_radius = 2. * r_used
    elif ratio < 0.1:
      self._model_radius = 0.5 * min(self._model_radius, r_used)

  def generate_candidate_points(self):
    pass

  def postprocess_evaluated_candidates(self):
    pass

  def update(self):
    pass

  def frame_size(self):
    pass

  @property
  def iter(self):
    return self._iter

  @property
  def candidate_points_set(self):
    return self._candidate_points_set

  @property
  def n(self):
    return self._n

  @n.setter
  def n(self, value: Any) -> Any:
    self._n = value

  @candidate_points_set.setter
  def candidate_points_set(
          self, value: List[CandidatePoint]) -> List[CandidatePoint]:
    if isinstance(value, list) and len(value) == 0:
      self._candidate_points_set = value
    elif not isinstance(self._candidate_points_set, list):
      self._candidate_points_set = []
      self._candidate_points_set.append(value)
    else:
      self._candidate_points_set.append(value)

  @candidate_points_set.deleter
  def candidate_points_set(self):
    del self._candidate_points_set
    self._candidate_points_set = []

  @iter.setter
  def iter(self, other: int):
    self._iter = other

  @property
  def opportunistic(self):
    return self._opportunistic

  @opportunistic.setter
  def opportunistic(self, other: bool):
    self._opportunistic = other

  @property
  def store_cache(self):
    return self._store_cache

  @store_cache.setter
  def store_cache(self, other: bool):
    self._store_cache = other

  @property
  def check_cache(self):
    return self._check_cache

  @check_cache.setter
  def check_cache(self, other: bool):
    self._check_cache = other

  @property
  def eval_budget(self):
    return self._eval_budget

  @eval_budget.setter
  def eval_budget(self, other: int):
    self._eval_budget = other

  @property
  def display(self):
    return self._display

  @display.setter
  def display(self, other: bool):
    self._display = other

  @property
  def bb_output(self) -> List[float]:
    return self._bb_output

  @bb_output.setter
  def bb_output(self, other: List[float]):
    self._bb_output = other

  @property
  def nb_success(self):
    return self._nb_success

  @nb_success.setter
  def nb_success(self, other: int):
    self._nb_success = other

  @property
  def bb_eval(self):
    return self._bb_eval

  @bb_eval.setter
  def bb_eval(self, other: int):
    self._bb_eval = other

  @property
  def type(self):
    return self._type

  @type.setter
  def type(self, value: Any) -> Any:
    self._type = value

  @property
  def save_results(self):
    return self._save_results

  @save_results.setter
  def save_results(self, value: bool) -> bool:
    self._save_results = value

  @property
  def dim(self):
    return self._n

  @dim.setter
  def dim(self, value: Any) -> Any:
    self._n = value

  @dim.deleter
  def dim(self):
    self._n = 0

  @property
  def xmin(self):
    return self._xmin

  @xmin.setter
  def xmin(self, value: Any) -> Any:
    self._xmin = value

  @property
  def success(self):
    return self._success

  @success.setter
  def success(self, value: Any) -> Any:
    self._success = value

  @property
  def seed(self):
    return self._seed

  @seed.setter
  def seed(self, value: Any) -> Any:
    self._seed = value

  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, value: Any) -> Any:
    self._dtype = value

  def scale(self, ub: List[float], lb: List[float], factor: float = 10.0):
    self.scaling = np.divide(np.subtract(ub, lb, dtype=self._dtype.dtype),
                             factor, dtype=self._dtype.dtype)
    if any(np.isinf(self.scaling)):
      for k, x in enumerate(np.isinf(self.scaling)):
        if x:
          self.scaling[k][0] = 1.0

  def get_list_of_coords_from_list_of_points(
          self, xps: List[CandidatePoint] = None) -> np.ndarray:
    coords_array = np.zeros((len(xps), self.dim))
    for i, _ in enumerate((xps)):
      coords_array[i, :] = xps[i].coordinates

    return coords_array

  def generate_2ngrid(
          self, vlim: np.ndarray = None, x_incumbent: CandidatePoint = None,
          p_in: List[float] = [0.01],
          m_in: List[float] = [0.01]) -> np.ndarray:
    grid = Dirs2n()
    grid.mesh = copy.deepcopy(
        self.mesh) if x_incumbent.mesh is None else x_incumbent.mesh
    # """ 5- Assign optional algorithmic parameters to the constructed poll instant  """
    grid.seed = int(self.seed + self.iter)
    grid.mesh.dtype.precision = "medium"
    grid.mesh.psize = p_in
    grid.mesh.msize = m_in
    grid.scaling = self.scaling
    grid.dim = self.dim
    grid.n = self.dim
    grid.center = x_incumbent
    grid.scale(ub=vlim[:, 0], lb=vlim[:, 1], factor=self.prob_params.scaling)
    # hhm = grid.create_housholder(
    #     True if self.success == SUCCESS_TYPES.FS else False,
    #     domain=self.xmin.var_type)
    grid.lb = vlim[:, 0]
    grid.ub = vlim[:, 1]
    grid.hmax = self.xmin.h_max

    grid.create_poll_set(
        ub=grid.ub, lb=grid.lb, it=self.iter, var_type=self.xmin.var_type,
        var_sets=self.xmin.sets, var_link=self.xmin.var_link, c_types=None,
        is_prim=True, rich_direction=True
        if self.success == SUCCESS_TYPES.FS else False)

    return self.get_list_of_coords_from_list_of_points(
        grid.candidate_points_set)

  def HD_grid(self, hashtable: Cache, n: int = 3, vlim: np.ndarray = None) -> np.ndarray:
    grid_points = None

    if n <= 2 * self.dim:
      x_inc = CandidatePoint()
      x_inc = hashtable.get_all_improving_candidates(nsamples=n)[0]
      grid_points = self.generate_2ngrid(vlim=vlim, x_incumbent=x_inc,
                                         p_in=self.prob_params.scaling, m_in=[
                                             x/100 for x in self.prob_params.scaling])[:n]
    else:
      grid_points: np.ndarray
      for i in range(int(n/(2*self.dim))+1):
        x_inc = CandidatePoint()
        x_inc = hashtable.get_all_improving_candidates(nsamples=n)[i]
        if isinstance(x_inc, CandidatePoint):
          temp = self.generate_2ngrid(
              vlim=vlim, x_incumbent=x_inc, p_in=self.prob_params.scaling,
              m_in=[x / 100 for x in self.prob_params.scaling])
          if i == 0:
            grid_points = temp
          else:
            grid_points = np.vstack((grid_points, temp))
        else:
          continue

    return grid_points[:n, :]

  def project_on_mesh_and_snap_to_bounds(  # noqa: C901
          self, m: Mesh, x_center: List[float],
          lb: List[float],
          ub: List[float], hashtable: Cache, stats: MadsStatistics):
    # Length check equivalent in Python
    for k, _ in enumerate((self._candidate_points_set)):
      if len(lb) != m.n or len(ub) != m.n:
        raise ValueError(
            f"Expected vectors of length {m.n}, but got lengths {len(lb)} and {len(ub)}")

      if not np.all(lb < ub):
        raise ValueError("Wrong bound constraints")

      if not np.all(lb <= x_center) or not np.all(ub >= x_center):
        raise ValueError("mesh center values must satisfy: lb <= x^mesh <= ub")

      # 1. Project on the mesh
      px = Point(self.mesh.n)
      px.coordinates = self._candidate_points_set[k].coordinates
      candidate = m.project_on_mesh(point=px, frame_center=x_center)

      # 2. Snap to bounds if necessary
      δ = m.get_delta_mesh_size()
      # snapped_candidate = np.zeros(m.n)

      for i in range(m.n):
        if lb[i] <= candidate[i] <= ub[i]:
          self._candidate_points_set[k].coordinates[i] = candidate[i]
        else:
          if candidate[i] < lb[i]:
              # Mesh center is supposed to be >= lb; normally,
              # this rounding is supposed to be in the box constraints
            self._candidate_points_set[k].coordinates[i] = δ[i] * np.ceil(
                (lb[i] - x_center[i]) / δ[i]) + x_center[i]
          else:
              # Ref value is supposed to be <= ub; normally,
              # this rounding is supposed to be in the box constraints
            self._candidate_points_set[k].coordinates[i] = δ[i] * np.floor(
                (ub[i] - x_center[i]) / δ[i]) + x_center[i]

          # snapRandGen = np.random.default_rng(seed=self.seed+self.iter)
          if self._candidate_points_set[k].coordinates[i] < lb[i]:
            print(f"Warning: snap_to_bounds: Error snapping {candidate[i]} to lower bound {lb[i]}")
            print(f"frameCenter = {x_center[i]}, δ = {δ[i]} : it gave \
                  {self._candidate_points_set[k]} which is still lower than {lb[i]}")
            # TODO: Force the snapping?
            # diff = lb[i] - self._candidate_points_set[k].coordinates[i]
            # self._candidate_points_set[k].coordinates[i] = lb[i] + snapRandGen.random() * diff
            stats.noutbound_hits+=1

          if self._candidate_points_set[k].coordinates[i] > ub[i]:
            print(f"Warning: snap_to_bounds: Error snapping {candidate[i]} to upper bound {ub[i]}")
            print(f"frameCenter = {x_center[i]}, δ = {δ[i]} : it gave \
                  {self._candidate_points_set[k]} which is still higher than {ub[i]}")
            # TODO: Force the snapping?
            # diff = self._candidate_points_set[k].coordinates[i] - ub[i]
            # self._candidate_points_set[k].coordinates[i] = ub[i] - snapRandGen.random() * diff
            stats.noutbound_hits+=1
            
    filtered = [x for x in self._candidate_points_set
                if not hashtable.is_duplicate(x, False)]
                

    self.candidate_points_set = []
    for fi, f in enumerate(filtered):
      self.candidate_points_set = f

  def generate_sample_points(  # noqa: C901
          self, active_barrier: AdaptiveBarrier, hashtable: Cache,
          ub: List[float],
          lb: List[float],
          fc: CandidatePoint, fci: int, it: int, var_type: List,
      var_sets: Dict, var_link: List[str],
          c_types: List[BARRIER_TYPES] = None,
          last_success: SUCCESS_TYPES = SUCCESS_TYPES.US, nsamples: int = None):
    """ Generate the sample points """
    self.nvars = len(self.prob_params.baseline)
    is_active_sampling = False
    self.search_trial += 1
    is_sas: bool = False
    is_pss: bool = False
    sampling_sas = None
    sampling_pss = None
    sampling_quasi_random = None
    if self.prob_params.lhs_search_initialization and self.iter == 1:
      nsamples = self.ns = (self.dim+1)*(self.dim+2)
    v = np.empty((self.nvars, 2))
    if self.bb_eval + nsamples > self.eval_budget:
      nsamples = self.eval_budget - self.bb_eval
    # Local but not active sampling
    if fc and self.iter > 1 and self.sampling_t != SAMPLING_METHOD.ACTIVE.name and self.vicinity_ratio is not None:
      for i, _ in enumerate((self.prob_params.lb)):
        d_uc = abs(self.prob_params.ub[i] - self.prob_params.lb[i])
        # lb/ub are plain floats here: assignment is enough, deepcopy is pure overhead
        lb = fc.coordinates[i] - (d_uc * self.vicinity_ratio[i][0])
        ub = fc.coordinates[i] + (d_uc * self.vicinity_ratio[i][0])
        if lb <= self.prob_params.lb[i]:
          lb = self.prob_params.lb[i]
        elif lb >= self.prob_params.ub[i]:
          lb = fc.coordinates[i]
        if ub >= self.prob_params.ub[i]:
          ub = self.prob_params.ub[i]
        elif ub <= self.prob_params.lb[i]:
          ub = fc.coordinates[i]
        v[i] = [lb, ub]
    else:
      for i, _ in enumerate((self.prob_params.lb)):
        v[i] = [self.prob_params.lb[i], self.prob_params.ub[i]]
    # Rule of thumb if the number of samples is not provided
    if nsamples is None:
      nsamples = int((self.nvars+1)*(self.nvars+2)/2)
    is_lhs = False
    sampling = None
    self.ns = nsamples
    resize = False
    clipping = True
    if self.type == SEARCH_TYPE.VNS.name:
      sampling = VNS(params=self.prob_params, x_inc=fc)
    elif self.sampling_t == SAMPLING_METHOD.FULLFACTORIAL.name:
      sampling = explore.samplers.FullFactorial(
          ns=nsamples, vlim=v, w=self.weights, c=clipping)
      resize = True
    elif self.sampling_t == SAMPLING_METHOD.RS.name:
      sampling = explore.samplers.RS(ns=nsamples, vlim=v)
      sampling.options["randomness"] = self.seed
    elif self.sampling_t == SAMPLING_METHOD.HALTON.name:
      sampling = explore.samplers.Halton(ns=nsamples, vlim=v, is_ham=True)
    elif self.sampling_t == SAMPLING_METHOD.LH.name:
      sampling = explore.samplers.LHS(ns=nsamples, vlim=v)
      sampling.options["randomness"] = self.seed+self.iter
      sampling.options["criterion"] = self.sampling_criter if self.iter > 1 else "corr"
      sampling.options["msize"] = self.mesh.get_delta_mesh_size().coordinates
      is_lhs = True
    else:
      if (not self.explore_new):
        self.explore_new = last_success == SUCCESS_TYPES.US
      switch_to_global_sampling: bool = (
          self.search_trial % self.diverse_intense_trial_period) == 0 or self.explore_new
      self.nds = len(hashtable.get_all_nd_candidates()) if self.prob_params.is_pareto and self.nds < len(
          hashtable.get_all_nd_candidates()) else self.nds

      # TODO: Checking if n_non_errors is better
      if self.iter == 1 or (
              self.prob_params.is_pareto and len(
                  hashtable.get_all_nd_candidates) < 5) or hashtable.n_all_non_error_candidates() < 5:
        sampling = explore.samplers.Halton(
            ns=nsamples, vlim=v) if active_barrier is None or (
            not self.explore_new and not switch_to_global_sampling and self.
            iter > 1) else explore.samplers.LHS(
            ns=nsamples, vlim=v)
        sampling.options["randomness"] = self.seed + self.iter
        sampling.options["criterion"] = self.sampling_criter
        sampling.options["msize"] = self.mesh.get_delta_mesh_size().coordinates
        sampling.options["varLimits"] = v
        self.explore_new = False
      else:
        # Enough data exists to try something smarter than pure space-
        # filling. Which "something" -- BiTPE's local density-guided
        # sampling, the TunableSA/TunablePSS global diversifiers, or a
        # plain Halton fallback -- is decided below by the good/bad
        # split's own statistical confidence (_bitpe_readiness), BiTPE's
        # recent track record (_update_bitpe_trust), and the current
        # exploitation stance (_exploitation_stance), rather than the flat
        # trial-count/period checks this replaced.
        if hashtable.is_pareto:
          nsamples = len(hashtable.get_all_nd_candidates()) if self.prob_params.is_pareto and self.nds >= len(
              hashtable.get_all_nd_candidates()) else hashtable.n_centers()  # len(self.hashtable.nd_points)
        self.best_samples = hashtable.n_centers()
        x_better   = []          # list of np.ndarray rows
        x_worse    = []
        f_better   = []
        f_worse    = []

        hashtable.get_splitted_sorted_candidates(x_better,
                                                 x_worse,
                                                 f_better,
                                                 f_worse,
                                                 ratio=0.3)

        n_reduced_dims = int(self.dim*0.7) if self.dim > 3 else 3
        n_d_eff = min(self.dim, n_reduced_dims) if _BITPE_REDUCES else self.dim
        # BiTPE works inside a trust region around the incumbent, on the
        # incumbent's neighbourhood only (see _trust_region/_local_split).
        tr_v, tr_radius = self._trust_region(fc, v)
        center_points, center_points_f, non_improving, non_improving_f = \
            self._local_split(x_better, f_better, x_worse, f_worse, fc,
                              tr_radius, n_d_eff)

        bitpe_ready, bitpe_reason = self._bitpe_readiness(
            len(center_points), len(non_improving),
            center_points_f, non_improving_f, n_d_eff)
        stance = self._exploitation_stance(last_success)
        bitpe_trust = self._update_bitpe_trust(center_points)
        # _bitpe_readiness alone gates BiTPE on/off; low trust widens the
        # kernel instead of vetoing BiTPE outright (see _update_bitpe_trust
        # for why a hard veto here would be a feedback deadlock).
        use_bitpe = bitpe_ready
        use_wide_kernel = stance == "conservative" or self.dim > 15 or bitpe_trust < 0.5
        if self.log is not None and self.log.is_verbose:
          self.log.log_msg(
              msg=f"BiTPE policy: ready={bitpe_ready} ({bitpe_reason}), "
                  f"trust={bitpe_trust:.2f}, stance={stance}, "
                  f"use_bitpe={use_bitpe}",
              msg_type=MSG_TYPE.INFO)

        if not use_bitpe and switch_to_global_sampling and hashtable.n_centers() > 100:
          # Not enough confidence/trust in BiTPE's local model right now --
          # widen the net with the global TunableSA/TunablePSS diversifiers
          # instead of forcing a possibly-noisy local model.
          # max_iter = 10000
          initial_temp = max(1000-self.search_trial, 100)
          cooling_rate = 0.95
          sampling_sas = explore.samplers.TunableSA(
              data=np.array(hashtable.get_all_center_points(nsamples=hashtable.n_centers())),
              y=np.array(
                  [xnd.fobj + [xnd.h]
                   if self.prob_params.is_pareto else xnd.f + [xnd.h]
                   for xnd in hashtable.get_all_center_candidates(
                       nsamples=hashtable.n_centers())]),
              x_inc=fc.coordinates, it=self.iter, vlim=v, seed=self.seed +
              self.iter, max_iter=10000, initial_temp=initial_temp,
              cooling_rate=cooling_rate)
          sampling_pss = explore.samplers.TunablePSS(
              data=np.array(hashtable.get_all_center_points(
                  nsamples=hashtable.n_centers())),
              y=np.array(
                  [xnd.fobj + [xnd.h]
                   if self.prob_params.is_pareto else xnd.f + [xnd.h]
                   for xnd in hashtable.get_all_center_candidates(
                       nsamples=hashtable.n_centers())]),
              x_inc=fc.coordinates, it=self.iter, vlim=v,
              inertia_weight=max(
                  0.9 -
                  (0.5 * self.search_trial / self.diverse_intense_trial_period),
                  0),
              social_weight=max(
                  0.1 +
                  (0.1 * self.search_trial / self.diverse_intense_trial_period),
                  1),
              cognitive_weight=max(
                  0.1 +
                  (0.1 * self.search_trial / self.diverse_intense_trial_period),
                  1),
              seed=self.seed + self.iter, num_particles=max(
                  int(self.ns / 3),
                  3),
              max_iter=100)
          sampling_quasi_random = explore.samplers.LHS(
              ns=max(int(self.ns/3), 3), vlim=v)
          is_sas = True
          is_pss = True
        elif not use_bitpe:
          # Not ready/trusted for BiTPE, and not eligible (or not due) for
          # the global diversifiers either -- fall back to plain
          # space-filling of the trust region rather than force a
          # low-confidence local model.
          sampling = explore.samplers.Halton(ns=nsamples, vlim=tr_v)
          sampling.options["randomness"] = self.seed + self.iter
          sampling.options["criterion"] = self.sampling_criter
          sampling.options["msize"] = self.mesh.get_delta_mesh_size().coordinates
          sampling.options["varLimits"] = tr_v
        else:
          # BiTPE's good/bad split is both statistically confident
          # (_bitpe_readiness) and, per its recent track record, worth
          # trusting (_update_bitpe_trust). Kernel width still follows the
          # exploitation stance: narrow/Gaussian to dig in locally,
          # wider Cauchy/Multiquadric-RBF to keep exploring broadly once
          # stagnating or the mesh is still coarse, or once the space is
          # large enough (>15D) that a narrow kernel would barely cover it.
          # BiTPE's obfuscated bandwidth validation/indexing for `h` has
          # shown inconsistent shape expectations once the problem
          # dimensionality crosses its internal reduction threshold: the
          # same per-dim `h` that works below that threshold can raise
          # OSError/IndexError above it. Rather than risk aborting the
          # whole run over a third-party edge case, fall back to Halton
          # sampling for this iteration if construction (or the later
          # resample()) fails.
          # `vlim=tr_v` keeps candidates inside the trust region: BiTPE
          # clips to it directly in full space, and on back-projection
          # when the space is POD-reduced. BiTPE draws base points with
          # probability proportional to their *position* in `good_data`,
          # so pass the good set best-last to favour the best points.
          try:
            self.active_sampling = explore.samplers.BiTPE(
                good_data=center_points[::-1], good_f_values=center_points_f[::-1],
                bad_data=non_improving, bad_f_values=non_improving_f,
                n_r=self.ns, vlim=tr_v,
                kernel_type={"Gaussian": 0.5, "Gaussian_RBF": 0.1,
                             "Multiquadric_RBF": 0.1, "Laplace": 0.1,
                             "cosine": 0.1, "logistic": 0.05,
                             'InverseMultiquadric_RBF': 0.05}
                if self.prob_params.is_pareto
                else
                {"Cauchy": 0.5, "Multiquadric_RBF": 0.5}
                if use_wide_kernel
                else {"Gaussian": 1},
                bw_method="SCOTT", seed=int(self.seed + self.iter),
                h=tr_radius.tolist(), gamma=0.1,
                **(dict(reduction_method="pca", n_reduced_dims=n_reduced_dims)
                   if _BITPE_REDUCES else {}))

            for ki, _ in enumerate((self.active_sampling.kernel)):
              self.active_sampling.kernel[ki].bw_method = "SCOTT"
              if self.active_sampling.kernel[ki].type == "PARAMETRIC":
                self.active_sampling.kernel[ki].h = float(np.mean(tr_radius))
            is_active_sampling = True
          except (IOError, OSError, IndexError, TypeError, ValueError) as bitpe_err:
            if self.log is not None and self.log.is_verbose:
              self.log.log_msg(
                  msg=f"BiTPE sampler construction failed ({bitpe_err}); "
                      "falling back to Halton sampling for this iteration.",
                  msg_type=MSG_TYPE.INFO)
            sampling = explore.samplers.Halton(ns=nsamples, vlim=tr_v)
            sampling.options["randomness"] = self.seed + self.iter
            sampling.options["criterion"] = self.sampling_criter
            sampling.options["msize"] = self.mesh.get_delta_mesh_size().coordinates
            sampling.options["varLimits"] = tr_v
            is_active_sampling = False

    # A single-sample LHS has nothing to space-fill, and samplersLib's ExactSE
    # optimizer (the default criterion, also used inside expand_lhs) loops
    # forever with fewer than two free rows in releases up to 2604.0.
    single_lhs = isinstance(sampling, explore.samplers.LHS) and nsamples < 2
    if single_lhs and sampling.options.get("criterion") in (None, "ExactSE"):
      sampling.options["criterion"] = "maximin"

    if self.iter > 1 and is_lhs and not single_lhs and \
            len(self.candidate_points_set) > 0:
      # Freshly-returned arrays from the sampler: a fast native copy is all
      # the isolation we need, copy.deepcopy's generic object-graph walk buys
      # nothing extra here and is far slower for numpy arrays.
      ps = np.array(
          sampling.expand_lhs(
              x=self.map_samples_from_points_to_coords(),
              n_points=nsamples, method="basic"), copy=True)
    else:
      if is_active_sampling:
        try:
          # No extra scaling (`scale=1.0`): the kernels' bandwidths are already
          # tuned on trust-region-local data, so they carry the local length
          # scale. Passed explicitly because released samplersLib (<= 2604.0)
          # requires it; leaving it out raised a TypeError that the fallback
          # below silently turned into Halton sampling.
          # The batch is the configured number of samples (`search["ns"]`,
          # capped by the remaining budget above).
          ps = np.array(
              self.active_sampling.resample(
                  size=max(int(self.ns), 1), seed=int(self.seed + self.iter),
                  scale=1.0),
              copy=True)
          # Plus a few cheap line probes from the incumbent toward the
          # trust-region faces (these used to aim at the global bounds,
          # from the *worst* good point).
          rng = np.random.default_rng(seed=self.seed+self.iter)
          rval = rng.random()
          x_inc = np.asarray(fc.coordinates, dtype=float)
          ps = np.vstack((ps, x_inc + rval * (tr_v[:, 1] - x_inc)))
          ps = np.vstack((ps, x_inc - rval * (x_inc - tr_v[:, 0])))
          ps = np.vstack((ps, np.clip(ps[0] + rval * tr_radius, tr_v[:, 0], tr_v[:, 1])))
        except (IOError, OSError, IndexError, TypeError, ValueError) as bitpe_err:
          # Same rationale as the construction-time fallback above: don't
          # let an obfuscated-library edge case abort the whole run.
          if self.log is not None and self.log.is_verbose:
            self.log.log_msg(
                msg=f"BiTPE resample failed ({bitpe_err}); "
                    "falling back to Halton sampling for this iteration.",
                msg_type=MSG_TYPE.INFO)
          fallback_sampling = explore.samplers.Halton(ns=nsamples, vlim=tr_v)
          fallback_sampling.options["randomness"] = self.seed + self.iter
          fallback_sampling.options["criterion"] = self.sampling_criter
          fallback_sampling.options["msize"] = self.mesh.get_delta_mesh_size().coordinates
          fallback_sampling.options["varLimits"] = tr_v
          ps = np.array(fallback_sampling.generate_samples(), copy=True)

      elif is_pss and is_sas:
        ps1 = np.array(
            sampling_sas.resample(
                size=max(int(self.ns / 3),
                         3),
                seed=int(self.seed + self.iter)), copy=True)
        ps2 = np.array(
            sampling_pss.resample(
                size=max(int(self.ns / 3),
                         3),
                seed=int(self.seed + self.iter)), copy=True)
        ps = np.concatenate((ps1, ps2), axis=0)
        ps4 = np.array(sampling_quasi_random.generate_samples(), copy=True)
        ps = np.concatenate((ps, ps4), axis=0)
      else:
        ps = np.array(sampling.generate_samples(), copy=True) if self.type != SEARCH_TYPE.VNS.name else np.array(
            sampling.run(x=fc, n_dist=[int(nsamples/4)]*4), copy=True)

    if resize:
      self.ns = len(ps)
      nsamples = len(ps)

    if self.iter > 1 and is_lhs:
      self.map_samples_from_coords_to_points(
          samples=ps[len(ps)-self.ns:], fc=fc)
    else:
      self.map_samples_from_coords_to_points(ps, fc=fc)

    if is_active_sampling:
      # Remember what BiTPE actually proposed (post mesh-projection/bounds
      # clipping -- matching what will actually get evaluated and cached)
      # so the *next* call's _update_bitpe_trust can check how many of
      # these landed in the "good" bucket once evaluated.
      self._last_bitpe_coords = np.array(
          [c.coordinates for c in self._candidate_points_set])

  def project_coords_to_mesh(self, x: List[float], ref: List[float] = None):
    pref = Point(self.mesh.n)
    pref.coordinates = ref
    px = Point(self.mesh.n)
    px.coordinates = x
    x_projected: Point = self.mesh.project_on_mesh(px, pref)

    return x_projected.coordinates

  def map_samples_from_coords_to_points(
          self, samples: np.ndarray, add_to_list: bool = False,
          fc: CandidatePoint = None):
    for i in range(len(samples)):
      samples[i, :] = self.project_coords_to_mesh(samples[i, :], ref=np.subtract(
          self.prob_params.ub, self.prob_params.lb).tolist())
      for s in range(len(samples[i, :])):
        if self.prob_params.lb[s] > samples[i, s]:
          samples[i, s] = self.prob_params.lb[s]
        if self.prob_params.ub[s] < samples[i, s]:
          samples[i, s] = self.prob_params.ub[s]
    samples = np.unique(samples, axis=0)
    if not add_to_list:
      self._candidate_points_set: List[CandidatePoint] = [0] * len(samples)
      for i in range(len(samples)):
        self._candidate_points_set[i] = CandidatePoint()
        if fc.var_type is not None:
          self._candidate_points_set[i].var_type = fc.var_type
        else:
          self._candidate_points_set[i].var_type = None
        self._candidate_points_set[i].sets = fc.sets
        self._candidate_points_set[i].var_link = fc.var_link
        self._candidate_points_set[i].n_dimensions = len(samples[i, :])
        # the coordinates setter already does `list(coords)`, its own copy,
        # so wrapping the ndarray row in copy.deepcopy here is redundant
        self._candidate_points_set[i].coordinates = samples[i, :]
        self._candidate_points_set[i].direction = np.subtract(
            fc.coordinates, self._candidate_points_set[i].coordinates)
        self._candidate_points_set[i].incumbent_signature = fc.signature
    else:
      for i in range(len(samples)):
        self._candidate_points_set += [CandidatePoint()]
        if fc.var_type is not None:
          self._candidate_points_set[-1].var_type = fc.var_type
        else:
          self._candidate_points_set[-1].var_type = None
        self._candidate_points_set[-1].sets = fc.sets
        self._candidate_points_set[-1].var_link = fc.var_link
        self._candidate_points_set[-1].n_dimensions = len(samples[i, :])
        self._candidate_points_set[-1].coordinates = samples[i, :]
        self._candidate_points_set[-1].direction = np.subtract(
            fc.coordinates, self._candidate_points_set[-1].coordinates)
        self._candidate_points_set[-1].incumbent_signature = fc.signature

  def map_samples_from_points_to_coords(self):
    return np.array([x.coordinates for x in self._candidate_points_set])

  def gauss_perturbation(
          self, p: CandidatePoint, npts: int = 5) -> List[CandidatePoint]:
    lb = self.prob_params.lb
    ub = self.prob_params.ub
    cs = np.zeros((npts, p.n_dimensions))
    pts: List[CandidatePoint] = [0] * npts
    for k in range(p.n_dimensions):
      if p.var_type[k] == VAR_TYPE.REAL:
        cs[:, k] = np.random.normal(loc=p.coordinates[k],
                                    scale=self.mesh.get_delta_mesh_size().coordinates[k],
                                    size=(npts,))
      elif p.var_type[k] == VAR_TYPE.INTEGER or \
              p.var_type[k] == VAR_TYPE.DISCRETE or \
      p.var_type[k] == VAR_TYPE.CATEGORICAL:
        cs[:, k] = np.random.randint(low=lb[k], high=ub[k], size=(npts,))
      else:
        cs[:, k] = [p.coordinates[k]]*npts
      for i in range(npts):
        if cs[i, k] < lb[k]:
          cs[i, k] = lb[k]
        if cs[i, k] > ub[k]:
          cs[i, k] = ub[k]

    for i in range(npts):
      pts[i] = p
      pts[i].coordinates = cs[i, :]

    return pts

  def unique_lhs(self, n_samples: int, n_dimensions: int, trial: int):
    unique_samples = set()

    while len(unique_samples) < n_samples:
      # Generate a batch of LHS samples
      samples = LHS(
          n_dimensions, samples=n_samples - len(unique_samples),
          random_state=self.seed + trial * 2)

      # Add unique samples to the set
      for sample in samples:
        # Convert to a tuple for immutability and uniqueness
        sample_tuple = tuple(np.round(sample, decimals=5))
        unique_samples.add(sample_tuple)

        # Break if we have enough unique samples
        if len(unique_samples) == n_samples:
          break
    out = np.zeros((n_samples, n_dimensions))
    for i in range(n_samples):
      out[i] = list(list(unique_samples)[i])

    return out

  def omit_duplicates(self, n_total_evals: int = 0, stats: MadsStatistics = None,
                      hashtable: Cache = None):
    temp: List[CandidatePoint] = []
    npts = 1
    if stats is None:
      stats = MadsStatistics()
    # Coordinates already walked in this pass, so a same-set duplicate check
    # is an O(1) membership test instead of rescanning every prior candidate
    # (same result as the previous pairwise-list-equality scan, just O(k)
    # instead of O(k^2) over the candidate set).
    seen_coords = set()
    for xi, xtry in enumerate(self.candidate_points_set):
      if n_total_evals+npts > self.eval_budget:
        break
      is_dup = xtry is None or hashtable.is_duplicate(
          xtry, add=False)
      coord_key = None if xtry is None else tuple(xtry.coordinates)
      is_dup_in_the_set = xtry is None or coord_key in seen_coords
      if xtry is not None:
        seen_coords.add(coord_key)
      is_duplicate: bool = (
          (self.check_cache and hashtable.last_index >= 0 and is_dup)
          or is_dup_in_the_set)
      if is_duplicate:
        if self.log is not None and self.log.is_verbose:
          self.log.log_msg(
              msg="Cache hit ... Failed to find a non-duplicate alternative.",
              msg_type=MSG_TYPE.INFO)
        if self.display:
          print("Cache hit ... Failed to find a non-duplicate alternative.")
        stats.ncache_hits += 1
      else:
        # self.hashtable.add_to_cache(xtry)
        if n_total_evals+npts == self.eval_budget:
          temp.append(xtry)
          break
        else:
          npts += 1
          temp.append(xtry)
    del self.candidate_points_set
    for t in temp:
      self.candidate_points_set = t.clone()

  def master_updates(
          self, x: List[CandidatePoint],
          peval, save_all_best: bool = False, save_all: bool = False):
    if peval >= self.eval_budget:
      self.terminate = True
    x_post: List[CandidatePoint] = []
    c = 0
    for xtry in x:
      c += 1
      # """ Check success conditions """
      is_infeas_dom: bool = (
          xtry.status == DESIGN_STATUS.INFEASIBLE and (xtry.h < self.xmin.h))
      is_feas_dom: bool = (
          xtry.status == DESIGN_STATUS.FEASIBLE and xtry.fobj < self.xmin.fobj)

      success = SUCCESS_TYPES.US
      if (is_infeas_dom or is_feas_dom):
        self.success = SUCCESS_TYPES.FS
        success = SUCCESS_TYPES.US  # <- This redundant variable is important
        # for managing concurrent parallel execution
        self.nb_success += 1
        # """ Update the post instant """
        del self._xmin
        self._xmin = CandidatePoint()
        self._xmin = copy.deepcopy(xtry)
        if self.display:
          if self._dtype.dtype == np.float64:
            print(f"Success: fmin = {self.xmin.f} (hmin = {self.xmin.h:.15})")
          elif self._dtype.dtype == np.float32:
            print(f"Success: fmin = {self.xmin.f} (hmin = {self.xmin.h:.6})")
          else:
            print(f"Success: fmin = {self.xmin.f} (hmin = {self.xmin.h:.18})")

        self.mesh.psize_success = copy.deepcopy(
            self.mesh.get_delta_frame_size().coordinates)
        self.mesh.psize_max = copy.deepcopy(
            max(self.mesh.get_delta_frame_size().coordinates))

      if (save_all_best and success == SUCCESS_TYPES.FS) or (save_all):
        x_post.append(xtry)

    if self.success == SUCCESS_TYPES.FS:
      self.n_successes += 1
    return x_post

  def update_local_region(self, region="expand"):
    if self.vicinity_ratio is None:
      self.vicinity_ratio = np.ones((len(self.prob_params.baseline), 1))
    if region == "expand":
      for i, _ in enumerate((self.vicinity_ratio)):
        if self.vicinity_ratio[i] * 2 < self.prob_params.ub[i]:
          self.vicinity_ratio[i] *= 2
    elif region == "contract":
      for i, _ in enumerate((self.vicinity_ratio)):
        if self.vicinity_ratio[i] / 2 > self.prob_params.lb[i] and \
                self.vicinity_ratio[i] > self.vicinity_min:
          self.vicinity_ratio[i] /= 2
    else:
      raise IOError(f"Unrecognized {region} local region operation")


@dataclass(slots=True)
class search_sampling:
  """Options of the search step (the ``search`` section of the input).

  :param s_method: Sampling method: "ACTIVE", "LH", "RS", "HALTON" or
    "FULLFACTORIAL"
  :param ns: Number of samples per search step (None: (n+1)(n+2)/4)
  :param visualize: Reserved
  :param criterion: Space-filling criterion of the Latin hypercube sampler
  :param weights: Level weights of the full-factorial sampler
  :param type: "VNS" for the variable neighbourhood search, any other value
    for the sampling search
  """
  s_method: str = SAMPLING_METHOD.LH.name
  ns: int = 3
  visualize: bool = False
  criterion: Optional[str] = None
  weights: Optional[List[float]] = None
  type: str = SEARCH_TYPE.SAMPLING.name
