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
from dataclasses import dataclass
from typing import List, Dict
import copy

import numpy as np

from .._include import DType, VAR_TYPE, BARRIER_TYPES, MSG_TYPE
from .._include import CandidatePoint
from .._include import Point
from .._templates._optimizer import GenericSamplerBase, GenericSamplerBaseData
from .._include import Mesh
from .._include import Cache
from .._analytics._metadata import MadsStatistics


class Dirs2n(GenericSamplerBaseData, GenericSamplerBase):
  """This is the orthognal 2n-directions class used for the poll step

    :param _poll_dirs: Poll set (list of points)
    :param _point_index: List of point indices
    :param _n: Number of directions
    :param _defined: A boolean that indicate if the poll points are defined
  """

  def __init__(self):
    super().__init__()

  @property
  def rng(self):
    if self._rng is None:
      self._rng = np.random.default_rng(seed=self.seed + (self.iter) * 123)
    return self._rng

  @rng.setter
  def rng(self, value: np.random) -> np.random:
    self._rng = value

  @property
  def n(self):
    return self._n

  @n.setter
  def n(self, value: int) -> int:
    self._n = value

  @property
  def bb_output(self) -> List[float]:
    """Evaluated blackbox functions

    :return: List of evaluated blackbox functions
    :rtype: List[float]
    """
    return self._bb_output

  @bb_output.setter
  def bb_output(self, other: List[float]):
    self._bb_output = other

  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, other: DType):
    self._dtype = other

  @property
  def point_index(self):
    return self._point_index

  @point_index.setter
  def point_index(self, other: int):
    if other == -1:
      self._point_index = []
    else:
      self._point_index.append(other)

  @property
  def terminate(self):
    return self._terminate

  @terminate.setter
  def terminate(self, other: bool):
    self._terminate = other

  @property
  def seed(self):
    return self._seed

  @seed.setter
  def seed(self, other: int):
    self._seed = other

  @property
  def success(self):
    return self._success

  @success.setter
  def success(self, other: bool):
    self._success = other

  @property
  def eval_budget(self):
    return self._eval_budget

  @eval_budget.setter
  def eval_budget(self, other: int):
    self._eval_budget = other

  @property
  def opportunistic(self):
    return self._opportunistic

  @opportunistic.setter
  def opportunistic(self, other: bool):
    self._opportunistic = other

  @property
  def save_results(self):
    return self._save_results

  @save_results.setter
  def save_results(self, other: bool):
    self._save_results = other

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
  def display(self):
    return self._display

  @display.setter
  def display(self, other: bool):
    self._display = other

  @property
  def bb_eval(self):
    return self._bb_eval

  @bb_eval.setter
  def bb_eval(self, other: int):
    self._bb_eval = other

  @property
  def frame_size(self):
    return self._frame_size

  @frame_size.setter
  def frame_size(self, other: float):
    self._frame_size = other

  @property
  def iter(self):
    return self._iter

  @iter.setter
  def iter(self, other: int):
    self._iter = other

  @property
  def candidate_points_set(self):
    return self._candidate_points_set

  @candidate_points_set.setter
  def candidate_points_set(self, p: CandidatePoint):
    if not isinstance(self._candidate_points_set, list):
      self._candidate_points_set = []
    self._candidate_points_set.append(p)

  @candidate_points_set.deleter
  def candidate_points_set(self):
    del self._candidate_points_set
    self._candidate_points_set = []

  @property
  def poll_dirs(self):
    return self._directions_set

  @poll_dirs.setter
  def poll_dirs(self, dirs: Point):
    self._directions_set.append(dirs)

  @poll_dirs.deleter
  def poll_dirs(self):
    del self._directions_set
    self._directions_set = []

  @property
  def dim(self):
    return self._n

  @dim.setter
  def dim(self, n: int):
    self._n = n

  @dim.deleter
  def dim(self):
    self._n = 0

  @property
  def defined(self):
    return any(self._defined)

  @defined.setter
  def defined(self, defined):
    self._defined = defined

  @defined.deleter
  def defined(self):
    del self._defined

  @property
  def center(self) -> CandidatePoint:
    return self._center

  @center.setter
  def center(self, other: CandidatePoint):
    self._center = other

  @property
  def nb_success(self):
    return self._nb_success

  @nb_success.setter
  def nb_success(self, other: int):
    self._nb_success = other

  def generate_candidate_points(self):
    pass

  def postprocess_evaluated_candidates(self):
    pass

  def update(self):
    pass

  def generate_random_dir(self):
    # np.random.seed(seed=self.seed+(self.iter)*123)
    # np.random.seed(seed=self.seed)
    return self.rng.rand(self._n).tolist()

  def ran(self):
    # np.random.seed(seed=self.seed+(self.iter)*123)
    # np.random.seed(seed=self.seed)
    return self.rng.standard_normal(self._n).astype(dtype=self._dtype.dtype)

  def create_housholder(self, is_rich: bool, domain: List[int] = None,  # noqa: C901
                        is_one_dir: bool = False) -> np.ndarray:
    """Create householder matrix

    :param is_rich:  A flag that indicates if the rich direction option is enabled
    :type is_rich: bool
    :return: The householder matrix
    :rtype: np.ndarray
    """
    if domain is None:
      domain = [VAR_TYPE.REAL] * self._n
    elif len(domain) != self._n:
      raise IOError(
          "Number of dimensions doesn't match the size of \
          the variables type list invoked to Dirs2n::create_householder.")
    elif not isinstance(domain, list):
      raise IOError(
          "The variables domain type input invoked \
            to Dirs2n::create_householder should be of type list.")

    hhm: np.ndarray
    if is_rich:
      v_dir = copy.deepcopy(self.ran())
      v_dir_array = np.array(v_dir, dtype=self._dtype.dtype)
      v_dir_array = np.divide(
          v_dir_array,
          (np.linalg.norm(v_dir_array, 2).astype(dtype=self._dtype.dtype)),
          dtype=self._dtype.dtype)
      hhm = np.subtract(np.eye(self.dim, dtype=self._dtype.dtype),
                        np.multiply(2.0, np.outer(v_dir_array, v_dir_array.T),
                                    dtype=self._dtype.dtype),
                        dtype=self._dtype.dtype)
    else:
      hhm = np.eye(self.dim, dtype=self._dtype.dtype)
    hhm = np.dot(hhm, np.diag(
        (np.abs(hhm, dtype=self._dtype.dtype)).max(1) ** (-1)))
    # Rounding( and transpose)
    tmp = np.multiply(self.mesh.get_rho(), hhm, dtype=self._dtype.dtype)
    hhm = np.transpose(
        np.multiply(
            self.mesh.get_delta_mesh_size().coordinates, np.ceil(tmp),
            dtype=self._dtype.dtype))
    hhm = np.dot(hhm, self.scaling)

    for i, _ in enumerate((domain)):
      if domain[i] == VAR_TYPE.DISCRETE or \
              domain[i] == VAR_TYPE.BINARY or domain[i] == VAR_TYPE.INTEGER:
        hhm[i][i] = int(np.floor((-1 if i % 2 else 1) - 2 **
                        self.mesh.get_delta_mesh_size().coordinates[i]))
      elif domain[i] == VAR_TYPE.CATEGORICAL:
        hhm[i][i] = np.ceil(self.rng.random(
            1).astype(dtype=self._dtype.dtype))
      else:
        for j, _ in enumerate((domain)):
          if domain[j] != VAR_TYPE.REAL.name:
            hhm[i][j] = int(
                np.floor(-1 + 2**self.mesh.get_delta_mesh_size().coordinates[i]))

    if is_one_dir:
      return hhm
    else:
      hhm = np.vstack((hhm, -hhm))

    return hhm

  def create_poll_set(
          self, ub: List[float],
          lb: List[float], fc: CandidatePoint, fci: int,
          it: int, var_type: List, var_sets: Dict, var_link: List[str],
          c_types: List[BARRIER_TYPES] = None,
          rich_direction: bool = True):
    """Create the poll directions

    :param hhm: Householder matrix
    :type hhm: np.ndarray
    :param ub: Variables upper bound
    :type ub: List[float]
    :param lb: Variables lower bound
    :type lb: List[float]
    :param it: iteration
    :type it: int
    """
    del self.candidate_points_set
    del self.poll_dirs
    hhm = self.create_housholder(
        rich_direction,
        domain=self.prob_params.var_type)
    temp = np.add(
        hhm, np.array(
            fc.coordinates).T,
        dtype=self._dtype.dtype)

    temp = self.rng.permutation(temp)
    temp = np.minimum(temp, ub, dtype=self._dtype.dtype)
    temp = np.maximum(temp, lb, dtype=self._dtype.dtype)
    temp = np.unique(temp, axis=0)
    if isinstance(temp, list) or isinstance(temp, np.ndarray):
      ndirs = len(temp) if isinstance(
          temp[0],
          list) or isinstance(
          temp[0],
          np.ndarray) else 1
    else:
      ndirs = 0

    for k in range(ndirs):
      tmp = CandidatePoint()
      tmp.create_candidate_point_from_coords(
          temp=temp[k],
          var_type=var_type, var_sets=var_sets, var_link=var_link,
          c_types=c_types, fc_index=fci, other=fc)
      self.candidate_points_set = tmp
      # tmp.mesh = self.mesh
      if fc.is_feasible():
        tmp.direction = tmp - fc
        tmp.fc_index = fc
        self.poll_dirs = tmp - fc
      else:
        tmp.fc_index = fc
        tmp.direction = tmp - fc
        self.poll_dirs = tmp - fc
      del tmp
    del temp

    self.iter = it

  def scale(self, ub: List[float], lb: List[float], factor: float = 10.0):
    self.scaling = np.divide(np.subtract(ub, lb, dtype=self._dtype.dtype),
                             factor, dtype=self._dtype.dtype)
    if any(np.isinf(self.scaling)):
      for k, x in enumerate(np.isinf(self.scaling)):
        if x:
          self.scaling[k][0] = 1.0
    s_array = np.diag(self.scaling)

    self.scaling = []
    for k, _ in enumerate((s_array)):
      temp: List[float] = []
      for j in range(len(s_array[k])):
        temp.append(s_array[k][j])
      self.scaling.append(temp)
      del temp

  def directional_scaling(
          self, p: CandidatePoint) -> List[CandidatePoint]:
    lb = self.prob_params.lb
    ub = self.prob_params.ub
    scaling = self.mesh.get_delta_mesh_size().coordinates
    p_trials: List[CandidatePoint] = [0] * len(scaling)
    for k, _ in enumerate((scaling)):
      p_trials[k] = copy.deepcopy(p)
      p_trials[k].coordinates = copy.deepcopy(
          np.subtract(p_trials[k].coordinates, scaling[k]))
      for i in range(p_trials[k].n_dimensions):
        if p_trials[k].coordinates[i] < lb[k]:
          p_trials[k].coordinates[i] = copy.deepcopy(lb[k])
        if p_trials[k].coordinates[i] > ub[k]:
          p_trials[k].coordinates[i] = copy.deepcopy(ub[k])

    return p_trials

  def gauss_perturbation(
          self, p: CandidatePoint, npts: int = 5) -> List[CandidatePoint]:
    lb = self.prob_params.lb
    ub = self.prob_params.ub
    # np.random.seed(self.seed)
    cs = np.zeros((npts, p.n_dimensions))
    pts: List[CandidatePoint] = [0] * npts
    for k in range(p.n_dimensions):
      if p.var_type[k] == VAR_TYPE.REAL:
        cs[:, k] = self.rng.normal(loc=p.coordinates[k],
                                   scale=self.mesh.get_delta_mesh_size().coordinates[k],
                                   size=(npts,))
      elif p.var_type[k] == VAR_TYPE.INTEGER or \
              p.var_type[k] == VAR_TYPE.CATEGORICAL or p.var_type[k] == VAR_TYPE.DISCRETE:
        cs[:, k] = self.rng.randint(low=lb[k], high=ub[k], size=(npts,))
        for i in range(npts):
          cs[i, k] = int(cs[i, k])
      for i in range(npts):
        if cs[i, k] < lb[k]:
          cs[i, k] = lb[k]
        if cs[i, k] > ub[k]:
          cs[i, k] = ub[k]

    for i in range(npts):
      pts[i] = p
      pts[i].coordinates = copy.deepcopy(cs[i, :])

    return pts

  def project_on_mesh_and_snap_to_bounds(
          self, m: Mesh, x_center: List[float],
          lb: List[float],
          ub: List[float], hashtable: Cache):
    # Length check equivalent in Python
    for k, _ in enumerate((self.candidate_points_set)):
      if len(lb) != m.n or len(ub) != m.n:
        raise ValueError(
            f"Expected vectors of length {m.n}, but got lengths {len(lb)} and {len(ub)}")

      if not np.all(lb < ub):
        raise ValueError("Wrong bound constraints")

      if not np.all(lb <= x_center) or not np.all(ub >= x_center):
        raise ValueError("mesh center values must satisfy: lb <= x^mesh <= ub")

      # 1. Project on the mesh
      px = Point(self.mesh.n)
      px.coordinates = self.candidate_points_set[k].coordinates
      candidate = m.project_on_mesh(point=px, frame_center=x_center)

      # 2. Snap to bounds if necessary
      δ = m.get_delta_mesh_size()
      # snapped_candidate = np.zeros(m.n)

      for i in range(m.n):
        if lb[i] <= candidate[i] <= ub[i]:
          self.candidate_points_set[k].coordinates[i] = candidate[i]
        else:
          if candidate[i] < lb[i]:
              # Mesh center is supposed to be >= lb; normally,
              # this rounding is supposed to be in the box constraints
            self.candidate_points_set[k].coordinates[i] = δ[i] * np.ceil(
                (lb[i] - x_center[i]) / δ[i]) + x_center[i]
          else:
              # Ref value is supposed to be <= ub; normally,
              # this rounding is supposed to be in the box constraints
            self.candidate_points_set[k].coordinates[i] = δ[i] * np.floor(
                (ub[i] - x_center[i]) / δ[i]) + x_center[i]

          # Warnings as defined in Nomad 3
          if self.candidate_points_set[k].coordinates[i] < lb[i]:
            print(
                f"Warning: snap_to_bounds: Error snapping {candidate[i]} to lower bound {lb[i]}")
            print(
                f"frameCenter = {x_center[i]}, δ = {δ[i]} : it gave \
                  {self.candidate_points_set[k]} which is still lower than {lb[i]}")
            # TODO: Force the snapping?

          if self.candidate_points_set[k].coordinates[i] > ub[i]:
            print(
                f"Warning: snap_to_bounds: Error snapping {candidate[i]} to upper bound {ub[i]}")
            print(
                f"frameCenter = {x_center[i]}, δ = {δ[i]} : it gave \
                  {self.candidate_points_set[k]} which is still higher than {ub[i]}")

    filtered = [x for x in self._candidate_points_set
                if not hashtable.is_duplicate(x, False)]
    del self._candidate_points_set
    self._candidate_points_set = filtered
    # TODO: Force the snapping?

    # return snapped_candidate

  def omit_duplicates(
          self, n_total_evals: int = 0, stats: MadsStatistics = None,
          hashtable: Cache = None):
    temp: List[CandidatePoint] = []
    npts = 1
    if stats is None:
      stats = MadsStatistics()
    for xi, xtry in enumerate(self.candidate_points_set):
      if n_total_evals + npts > self.eval_budget:
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
        if n_total_evals + npts == self.eval_budget:
          temp.append(xtry)
          break
        else:
          npts += 1
          temp.append(xtry)
    del self.candidate_points_set
    for t in temp:
      self.candidate_points_set = copy.deepcopy(t)


@dataclass
class DirsNP1(GenericSamplerBase):
  def __post_init__(self):
    self._dtype = DType()
    self._xmin: CandidatePoint = CandidatePoint()
    self._x_sc: CandidatePoint = CandidatePoint()
    self._x_secondary_fc = CandidatePoint()
    self._x_primary_fc = CandidatePoint()
