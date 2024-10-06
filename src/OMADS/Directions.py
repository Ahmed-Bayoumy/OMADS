# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - ORTHO-MADS (MADS)                                    #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU Lesser General Public License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.   #
#                                                                                     #
#  You should have received a copy of the GNU Lesser General Public License along     #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on OMADS at                                               #
#  https://github.com/Ahmed-Bayoumy/OMADS                                             #
#  Copyright (C) 2022  Ahmed H. Bayoumy                                               #
# ------------------------------------------------------------------------------------#


import copy
from .CandidatePoint import CandidatePoint
from .Point import Point
from dataclasses import dataclass
from typing import List, Dict
from ._globals import DType, VAR_TYPE, BARRIER_TYPES, SUCCESS_TYPES, DESIGN_STATUS, MSG_TYPE
from .Optimizer import GenericSamplerBase
import numpy as np


@dataclass
class Dirs2n(GenericSamplerBase):
  """This is the orthognal 2n-directions class used for the poll step

    :param _poll_dirs: Poll set (list of points)
    :param _point_index: List of point indices
    :param _n: Number of directions
    :param _defined: A boolean that indicate if the poll points are defined
  """

  def __post_init__(self):
    self._dtype = DType()
    self._xmin: CandidatePoint = CandidatePoint()
    self._x_sc: CandidatePoint = CandidatePoint()

  @property
  def x_sc(self) -> CandidatePoint:
    return self._x_sc
  
  @x_sc.setter
  def x_sc(self, value: CandidatePoint):
    self._x_sc = value
  
  @property
  def bb_output(self) -> List[float]:
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
  def psize(self):
    return self._psize

  @psize.setter
  def psize(self, other: float):
    self._psize = other

  @property
  def iter(self):
    return self._iter

  @iter.setter
  def iter(self, other: int):
    self._iter = other

  @property
  def poll_set(self):
    return self._candidate_points_set 

  @poll_set.setter
  def poll_set(self, p: CandidatePoint):
    self._candidate_points_set.append(p)

  @poll_set.deleter
  def poll_set(self):
    del self._candidate_points_set 
    self._candidate_points_set  = []
  
  @property
  def poll_dirs(self):
    return self._directions_set
  
  @poll_dirs.setter
  def poll_dirs(self, dir: Point):
    self._directions_set.append(dir)
  
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
  def xmin(self)->CandidatePoint:
    return self._xmin

  @xmin.setter
  def xmin(self, other: CandidatePoint):
    self._xmin = other

  @property
  def nb_success(self):
    return self._nb_success

  @nb_success.setter
  def nb_success(self, other: int):
    self._nb_success = other

  def generate_dir(self):
    return np.random.rand(self._n).tolist()

  def ran(self):
    return np.random.random(self._n).astype(dtype=self._dtype.dtype)

  def create_housholder(self, is_rich: bool, domain: List[int] = None, is_one_dir: bool=False) -> np.ndarray:
    """Create householder matrix

    :param is_rich:  A flag that indicates if the rich direction option is enabled
    :type is_rich: bool
    :return: The householder matrix
    :rtype: np.ndarray
    """
    if domain is None:
      domain = [VAR_TYPE.REAL] * self._n
    elif len(domain) != self._n:
      raise IOError("Number of dimensions doesn't match the size of the variables type list invoked to Dirs2n::create_householder.")
    elif not isinstance(domain, list):
      raise IOError("The variables domain type input invoked to Dirs2n::create_householder should be of type list.")
    
    hhm: np.ndarray
    if is_rich:
      v_dir = copy.deepcopy(self.ran())
      v_dir_array = np.array(v_dir, dtype=self._dtype.dtype)
      v_dir_array = np.divide(v_dir_array,
                  (np.linalg.norm(v_dir_array,
                          2).astype(dtype=self._dtype.dtype)),
                  dtype=self._dtype.dtype)
      hhm = np.subtract(np.eye(self.dim, dtype=self._dtype.dtype),
                np.multiply(2.0, np.outer(v_dir_array, v_dir_array.T),
                      dtype=self._dtype.dtype),
                dtype=self._dtype.dtype)
    else:
      hhm = np.eye(self.dim, dtype=self._dtype.dtype)
    hhm = np.dot(hhm, np.diag((np.abs(hhm, dtype=self._dtype.dtype)).max(1) ** (-1)))
    # Rounding( and transpose)
    tmp = np.multiply(self.mesh.getRho(), hhm, dtype=self._dtype.dtype)
    hhm = np.transpose(np.multiply(self.mesh.getdeltaMeshSize().coordinates, np.ceil(tmp), dtype=self._dtype.dtype))
    hhm = np.dot(hhm, self.scaling)

    for i in range(len(domain)):
      if domain[i] == VAR_TYPE.DISCRETE or domain[i] == VAR_TYPE.BINARY or domain[i] == VAR_TYPE.INTEGER:
        hhm[i][i] = int(np.floor((-1 if i%2 else 1) - 2**self.mesh.getdeltaMeshSize().coordinates[i]))
      elif domain[i] == VAR_TYPE.CATEGORICAL:
        hhm[i][i] = np.ceil(np.random.random(1).astype(dtype=self._dtype.dtype))
      else:
        for j in range(len(domain)):
          if domain[j] != VAR_TYPE.REAL:
            hhm[i][j] = int(np.floor(-1 + 2**self.mesh.getdeltaMeshSize().coordinates[i]))
    
    if is_one_dir:
      return hhm
    else:
      hhm = np.vstack((hhm, -hhm))


    return hhm

  def create_poll_set(self, hhm: np.ndarray, ub: List[float], lb: List[float], it: int, var_type: List, var_sets: Dict, var_link: List[str], c_types: List[BARRIER_TYPES]=None, is_prim: bool = True):
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
    del self.poll_set
    del self.poll_dirs
    if is_prim:
      temp = np.add(hhm, np.array(self.xmin.coordinates), dtype=self._dtype.dtype)
    else:
      temp = np.add(hhm, np.array(self.x_sc.coordinates), dtype=self._dtype.dtype)
    # np.random.seed(self._seed)
    np.random.seed(seed=self.seed+self.iter*123)
    temp = np.random.permutation(temp)
    temp = np.minimum(temp, ub, dtype=self._dtype.dtype)
    temp = np.maximum(temp, lb, dtype=self._dtype.dtype)
    temp = np.unique(temp, axis=0)
    if isinstance(temp, list) or isinstance(temp, np.ndarray):
      ndirs = len(temp) if isinstance(temp[0], list) or isinstance(temp[0], np.ndarray) else 1
    else:
      ndirs = 0
    for k in range(ndirs):
      tmp = CandidatePoint()
      tmp.constraints_type = copy.deepcopy([xb for xb in c_types] if isinstance(c_types, list) else [c_types])
      tmp.sets = copy.deepcopy(var_sets)
      tmp.var_type = copy.deepcopy(var_type)
      tmp.var_link = copy.deepcopy(var_link)
      tmp.coordinates = temp[k]
      tmp.dtype.precision = self.dtype.precision
      tmp.mesh = copy.deepcopy(self.mesh)
      self.poll_set = tmp
      if is_prim:
        tmp.direction = Point(self.mesh._n)
        tmp.direction.coordinates = tmp - self.xmin
        self.poll_dirs = tmp - self.xmin
      else:
        tmp.direction = Point(self.mesh._n)
        tmp.direction.coordinates = tmp - self.x_sc
        self.poll_dirs = tmp - self.x_sc
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
    for k in range(len(s_array)):
      temp: List[float] = []
      for j in range(len(s_array[k])):
        temp.append(s_array[k][j])
      self.scaling.append(temp)
      del temp
  
  def directional_scaling(self, p: CandidatePoint, npts: int = 5) -> List[CandidatePoint]:
    lb = self.lb
    ub = self.ub
    scaling = self.mesh.getdeltaMeshSize().coordinates
    p_trials: List[CandidatePoint] = [0]*len(scaling)
    for k in range(len(scaling)):
      p_trials[k] = copy.deepcopy(p)
      p_trials[k].coordinates = copy.deepcopy(np.subtract(p_trials[k].coordinates, scaling[k]))
      for i in range(p_trials[k].n_dimensions):
        if p_trials[k].coordinates[i] < lb[k]:
          p_trials[k].coordinates[i] = copy.deepcopy(lb[k])
        if p_trials[k].coordinates[i] > ub[k]:
          p_trials[k].coordinates[i] = copy.deepcopy(ub[k])
    
    return p_trials
  
  def gauss_perturbation(self, p: CandidatePoint, npts: int = 5) -> List[CandidatePoint]:
    lb = self.lb
    ub = self.ub
    # np.random.seed(self.seed)
    cs = np.zeros((npts, p.n_dimensions))
    pts: List[CandidatePoint] = [0] * npts
    for k in range(p.n_dimensions):
      if p.var_type[k] == VAR_TYPE.REAL:
        cs[:, k] = np.random.normal(loc=p.coordinates[k], scale=self.mesh.getdeltaMeshSize().coordinates[k], size=(npts,))
      elif p.var_type[k] == VAR_TYPE.INTEGER or p.var_type[k] == VAR_TYPE.CATEGORICAL or p.var_type[k] == VAR_TYPE.DISCRETE:
        cs[:, k] = np.random.randint(low=lb[k], high=ub[k], size=(npts,))
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

  def postprocess_evaluated_candidates(self, x_cps: List[CandidatePoint] = None):
    #   self.hashtable._best_hash_ID.append(self.xmin.signature)
    for xtry in x_cps:
      if self.log is not None and self.log.is_verbose:
        self.log.log_msg(msg=f"Completed evaluation of point # {xtry.eval_no} in {xtry.eval_time} seconds, ftry={xtry.f}, status={xtry.status.name} and htry={xtry.h}. \n", msg_type=MSG_TYPE.INFO)
      """ Add to the cache memory """
      self.hashtable.add_to_cache(xtry)
      if not self.hashtable._is_pareto:
        self.hashtable.add_to_best_cache(xtry)
      if self.store_cache and xtry.signature not in self.hashtable.hash_id:
        self.hashtable.hash_id = xtry

      self.bb_eval = self.bb_handle.bb_eval
      self.psize = copy.deepcopy(self.mesh.getDeltaFrameSize().coordinates)

  def omit_duplicates(self):
    temp: List[CandidatePoint] = []
    for xtry in self.poll_set:
      is_dup = xtry.signature in self.hashtable.hash_id if not self.hashtable._is_pareto else self.hashtable.is_duplicate(xtry)
      is_duplicate: bool = (self.check_cache and self.hashtable.size > 0 and is_dup)
      # COMPLETED: The commented logic below needs more investigation to make sure that it doesn't hurt.
      # while is_duplicate and unique_p_trials < 5:
      #   if self.display:
      #     print(f'Cache hit. Trial# {unique_p_trials}: Looking for a non-duplicate along the poll direction where the duplicate point is located...')
      #   if xtry.var_type is None:
      #     if self.xmin.var_type is not None:
      #       xtry.var_type = self.xmin.var_type
      #     else:
      #       xtry.var_type = [VAR_TYPE.CONTINUOUS] * len(self.xmin.coordinates)
      #   xtries: List[Point] = self.directional_scaling(p=xtry, npts=len(self.poll_dirs)*2)
      #   for tr in range(len(xtries)):
      #     is_duplicate = self.hashtable.is_duplicate(xtries[tr])
      #     if is_duplicate:
      #        continue 
      #     else:
      #       xtry = copy.deepcopy(xtries[tr])
      #       break
      #   unique_p_trials += 1
      if (is_duplicate):
        if self.log is not None and self.log.is_verbose:
          self.log.log_msg(msg="Cache hit ... Failed to find a non-duplicate alternative.", msg_type=MSG_TYPE.INFO)
        if self.display:
          print("Cache hit ... Failed to find a non-duplicate alternative.")
      else:
        # self.hashtable.add_to_cache(xtry)
        temp.append(xtry)
    del self.poll_set
    for t in temp:
      self.poll_set = copy.deepcopy(t)


  def master_updates(self, x: List[CandidatePoint], peval, save_all_best: bool = False, save_all:bool = False):
    if peval >= self.eval_budget:
      self.terminate = True
    x_post: List[CandidatePoint] = []
    for xtry in x:
      """ Check success conditions """
      is_infeas_dom: bool = (xtry.status == DESIGN_STATUS.INFEASIBLE and (xtry.h < self.xmin.h) )
      is_feas_dom: bool = (xtry.status == DESIGN_STATUS.FEASIBLE and xtry.fobj < self.xmin.fobj)
      success = SUCCESS_TYPES.US
      if (is_infeas_dom or is_feas_dom):
        self.success = SUCCESS_TYPES.FS
        self.n_successes += 1
        success = SUCCESS_TYPES.FS  # <- This redundant variable is important
        # for managing concurrent parallel execution
        self.nb_success += 1
        """ Update the post instant """
        del self._xmin
        self._xmin = CandidatePoint()
        self._xmin = copy.deepcopy(xtry)
        self.constraints_RP.hmax = copy.deepcopy(xtry.h_max)
        if self.display:
          if self._dtype.dtype == np.float64:
            print(f"Success: fmin = {self.xmin.f} (hmin = {self.xmin.h:.15})")
          elif self._dtype.dtype == np.float32:
            print(f"Success: fmin = {self.xmin.f} (hmin = {self.xmin.h:.6})")
          else:
            print(f"Success: fmin = {self.xmin.f} (hmin = {self.xmin.h:.18})")

        self.mesh.psize_success = copy.deepcopy(self.mesh.getDeltaFrameSize().coordinates)
        self.mesh.psize_max = copy.deepcopy(max(self.mesh.getDeltaFrameSize().coordinates))
      if (save_all_best and success == SUCCESS_TYPES.FS) or (save_all):
        x_post.append(xtry)

    return x_post
