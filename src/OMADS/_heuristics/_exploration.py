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
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import random

from samplersLib.samplers import LHS
import samplersLib as explore
from .._include import Cache

import numpy as np

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
          ub: List[float], hashtable: Cache):
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

          # Warnings as defined in Nomad 3
          if self._candidate_points_set[k].coordinates[i] < lb[i]:
            print(
                f"Warning: snap_to_bounds: Error snapping {candidate[i]} to lower bound {lb[i]}")
            print(
                f"frameCenter = {x_center[i]}, δ = {δ[i]} : it gave \
                  {self._candidate_points_set[k]} which is still lower than {lb[i]}")
            # TODO: Force the snapping?

          if self._candidate_points_set[k].coordinates[i] > ub[i]:
            print(
                f"Warning: snap_to_bounds: Error snapping {candidate[i]} to upper bound {ub[i]}")
            print(
                f"frameCenter = {x_center[i]}, δ = {δ[i]} : it gave \
                  {self._candidate_points_set[k]} which is still higher than {ub[i]}")
            # TODO: Force the snapping?
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
        lb = copy.deepcopy(
            fc.coordinates[i]-(d_uc * self.vicinity_ratio[i][0]))
        ub = copy.deepcopy(
            fc.coordinates[i]+(d_uc * self.vicinity_ratio[i][0]))
        if lb <= self.prob_params.lb[i]:
          lb = copy.deepcopy(self.prob_params.lb[i])
        elif lb >= self.prob_params.ub[i]:
          lb = fc.coordinates[i]
        if ub >= self.prob_params.ub[i]:
          ub = copy.deepcopy(self.prob_params.ub[i])
        elif ub <= self.prob_params.lb[i]:
          ub = fc.coordinates[i]
        v[i] = [lb, ub]
    else:
      for i, _ in enumerate((self.prob_params.lb)):
        lb = copy.deepcopy(self.prob_params.lb[i])
        ub = copy.deepcopy(self.prob_params.ub[i])
        v[i] = [lb, ub]
    # Rule of thumb if the number of samples is not provided
    if nsamples is None:
      nsamples = int((self.nvars+1)*(self.nvars+2)/2)
    is_lhs = False
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
      sampling = explore.samplers.halton(ns=nsamples, vlim=v, is_ham=True)
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
      self.nds = len(hashtable.get_all_nd_candidates) if self.prob_params.is_pareto and self.nds < len(
          hashtable.get_all_nd_candidates()) else self.nds

      # TODO: Checking if n_non_errors is better
      if self.iter == 1 or (
              self.prob_params.is_pareto and len(
                  hashtable.get_all_nd_candidates) < 5) or hashtable.n_all_non_error_candidates() < 5:
        sampling = explore.samplers.halton(
            ns=nsamples, vlim=v) if active_barrier is None or (
            not self.explore_new and not switch_to_global_sampling and self.
            iter > 1) else explore.samplers.LHS(
            ns=nsamples, vlim=v)
        sampling.options["randomness"] = self.seed + self.iter
        sampling.options["criterion"] = self.sampling_criter
        sampling.options["msize"] = self.mesh.get_delta_mesh_size().coordinates
        sampling.options["varLimits"] = v
        self.explore_new = False
      elif switch_to_global_sampling and hashtable.n_centers() > 100:
        # max_iter = 10000
        initial_temp = max(1000-self.search_trial, 100)
        cooling_rate = 0.95
        sampling_sas = explore.samplers.TunableSA(
            data=hashtable.get_all_center_candidates(nsamples=hashtable.n_centers()),
            y=np.array(
                [xnd.fobj + [xnd.h]
                 if self.prob_params.is_pareto else xnd.f + [xnd.h]
                 for xnd in hashtable.get_all_center_candidates(
                     nsamples=hashtable.n_centers())]),
            x_inc=fc.coordinates, it=self.iter, vlim=v, seed=self.seed +
            self.iter, max_iter=10000, initial_temp=initial_temp,
            cooling_rate=cooling_rate)
        sampling_pss = explore.samplers.TunablePSS(
            data=hashtable.get_all_center_candidates(
                nsamples=hashtable.n_centers()),
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
      else:
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
        
        center_points   = np.array(x_better) if x_better else np.empty((0, self._n))
        non_improving    = np.array(x_worse)  if x_worse  else np.empty((0, self._n))
        center_points_f   = np.array(f_better) if f_better else np.empty((0, self.prob_params.nobj))
        non_improving_f    = np.array(f_worse)  if f_worse  else np.empty((0, self.prob_params.nobj))
        self.active_sampling = explore.samplers.BiTPE(
            good_data=center_points, good_f_values=center_points_f, bad_data=non_improving, \
              bad_f_values=non_improving_f, n_r=self.ns, vlim=v,
            kernel_type={"Gaussian": 0.5, "Gaussian_RBF": 0.1,
                         "Multiquadric_RBF": 0.1, "Laplace": 0.1,
                         "cosine": 0.1, "logistic": 0.05,
                         'InverseMultiquadric_RBF': 0.05}
            if self.prob_params.is_pareto
            else
            {"Cauchy": 0.5, "Multiquadric_RBF": 0.5}
            if np.linalg.norm(self.mesh.get_delta_frame_size().coordinates) > 1
            # else {'Cauchy': 0.8, "Laplace": 0.2},
            else {"Gaussian": 1},
            bw_method="SCOTT", seed=int(self.seed + self.iter),
            h=[np.linalg.norm(
                self.mesh.get_delta_frame_size().coordinates)] * self.dim, gamma=0.1)

        for ki, _ in enumerate((self.active_sampling.kernel)):
          self.active_sampling.kernel[ki].bw_method = "SCOTT" if np.linalg.norm(
              self.mesh.get_delta_frame_size().coordinates) > 1 else "SCOTT"
          if self.active_sampling.kernel[ki].type == "PARAMETRIC":
            self.active_sampling.kernel[ki].h = np.linalg.norm(
                self.mesh.get_delta_frame_size().coordinates) if np.linalg.norm(
                self.mesh.get_delta_frame_size().coordinates) > 1 else np.maximum(
                np.linalg.norm(self.mesh.get_delta_frame_size().coordinates),
                0.1)
        is_active_sampling = True

    if self.iter > 1 and is_lhs and len(self.candidate_points_set) > 0:
      ps = copy.deepcopy(
          sampling.expand_lhs(
              x=self.map_samples_from_points_to_coords(),
              n_points=nsamples, method="basic"))
    else:
      if is_active_sampling:
        s = self.mesh.get_delta_frame_size().coordinates
        ps = copy.deepcopy(
            self.active_sampling.resample(
                size=10, seed=int(self.seed + self.iter),
                scale=s))
      elif is_pss and is_sas:
        ps1 = copy.deepcopy(
            sampling_sas.resample(
                size=max(int(self.ns / 3),
                         3),
                seed=int(self.seed + self.iter)))
        ps2 = copy.deepcopy(
            sampling_pss.resample(
                size=max(int(self.ns / 3),
                         3),
                seed=int(self.seed + self.iter)))
        ps = np.concatenate((ps1, ps2), axis=0)
        ps4 = copy.deepcopy(sampling_quasi_random.generate_samples())
        ps = np.concatenate((ps, ps4), axis=0)
      else:
        ps = copy.deepcopy(sampling.generate_samples()) if self.type != SEARCH_TYPE.VNS.name else copy.deepcopy(
            sampling.run(x=fc, n_dist=[int(nsamples/4)]*4))

    if resize:
      self.ns = len(ps)
      nsamples = len(ps)

    if self.iter > 1 and is_lhs:
      self.map_samples_from_coords_to_points(
          samples=ps[len(ps)-self.ns:], fc=fc)
    else:
      self.map_samples_from_coords_to_points(ps, fc=fc)

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
        self._candidate_points_set[i].coordinates = copy.deepcopy(
            samples[i, :])
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
        self._candidate_points_set[-1].coordinates = copy.deepcopy(
            samples[i, :])
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
      pts[i].coordinates = copy.deepcopy(cs[i, :])

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
    for xi, xtry in enumerate(self.candidate_points_set):
      if n_total_evals+npts > self.eval_budget:
        break
      is_dup = xtry is None or hashtable.is_duplicate(
          xtry, add=False)
      is_dup_in_the_set = xtry is None or sum(
          [x.coordinates == xtry.coordinates
           for x in self.candidate_points_set[0: xi]]) >= 1
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
      self.candidate_points_set = copy.deepcopy(t)

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
  s_method: str = SAMPLING_METHOD.LH.name
  ns: int = 3
  visualize: bool = False
  criterion: Optional[str] = None
  weights: Optional[List[float]] = None
  type: str = SEARCH_TYPE.SAMPLING.name
