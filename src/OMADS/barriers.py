"""
#-------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                               #
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
"""
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from ._globals import DType, VAR_TYPE, BARRIER_TYPES, \
    SUCCESS_TYPES, DESIGN_STATUS, COMPARE_TYPE, INSERTION_FLAG
from .candidate_point import CandidatePoint
from .point import Point
from .parameters import Parameters
from .barrier import BarrierBase
from .options import Options
from .gmesh import Gmesh


def parent_indices(
        candidates: List[CandidatePoint],
        subcandidates: List[CandidatePoint]) -> List[int]:
  """
  Get the list of indices of points in 'candidates' found in 'subcandidates'

  :param candidates: List of cndidate solutions.
  :type candidates: List[CandidatePoint]
  :param subcandidates: List of subcandidate solutions that 
  might intersect with the list of 'candidates'
  :type subcandidates: List[CandidatePoint]
  :return: List of indices.
  :rtype: _type_
  """
  indices = []
  vector = [v.coordinates for v in candidates if v is not None]
  subvector = [v.coordinates for v in subcandidates if v is not None]
  if isinstance(vector, np.ndarray):
    vin = vector.tolist()
  else:
    vin = vector

  if isinstance(subvector, np.ndarray):
    svin = subvector.tolist()
  else:
    svin = subvector

  for v in svin:
    indices.append(vin.index(v))

  return indices


@dataclass
class Barrier:
  """_summary_

  :raises IOError: _description_
  :raises RuntimeError: _description_
  :raises RuntimeError: _description_
  :return: _description_
  :rtype: _type_
  """
  _params: Optional[Parameters] = None
  _eval_type: int = 1
  _h_max: float = 0
  _best_feasible: Optional[CandidatePoint] = None
  _ref: Optional[CandidatePoint] = None
  _filter: Optional[List[CandidatePoint]] = None
  _prefilter: int = 0
  _rho_leaps: float = 0.1
  _prim_poll_center: Optional[CandidatePoint] = None
  _sec_poll_center: Optional[CandidatePoint] = None
  _peb_changes: int = 0
  _peb_filter_reset: int = 0
  _peb_lop: Optional[List[CandidatePoint]] = None
  _all_inserted: Optional[List[CandidatePoint]] = None
  _one_eval_succ: Optional[SUCCESS_TYPES] = None
  _success: Optional[SUCCESS_TYPES] = None

  def __init__(self, p: Parameters, eval_type: int = 1):
    self._h_max = p.get_h_max_0()
    self._params = p
    self._eval_type = eval_type

  @property
  def sec_poll_center(self) -> CandidatePoint:
    return self._sec_poll_center

  @property
  def all_inserted(self) -> List[CandidatePoint]:
    return self._all_inserted

  def is_filtered_list_nonempty(self) -> bool:
    return self._filter is not None

  def is_all_inserted_nonempty(self) -> bool:
    return self._all_inserted is not None

  def get_best_feasible(self) -> CandidatePoint:
    return self._best_feasible

  def insert_feasible(self, x: CandidatePoint) -> SUCCESS_TYPES:
    """
    Try to insert a feasible candidate to the barrier as a new incumbent.

    :param x: _description_
    :type x: CandidatePoint
    :raises IOError:  One point has no f value
    :return: _description_
    :rtype: SUCCESS_TYPES
    """
    fx: float
    fx_bf: float
    if self._best_feasible is not None:
      fx_bf = self._best_feasible.fobj
    else:
      self._best_feasible = copy.deepcopy(x)
      return SUCCESS_TYPES.FS
    fx = x.fobj

    if (fx is None or fx_bf is None):
      raise IOError("insert_feasible(): one point has no f value")

    if fx < fx_bf:
      self._best_feasible = copy.deepcopy(x)
      return SUCCESS_TYPES.FS

    return SUCCESS_TYPES.US

  def filter_insertion(self, x: CandidatePoint) -> bool:
    """
    Insert apoint (if possible) to the filtered list of points.

    :param x: _description_
    :type x: CandidatePoint
    :return: _description_
    :rtype: bool
    """
    if not x.is_extreme_barrier_passed:
      return False
    if self._filter is None:
      self._filter = []
      self._filter.append(x)
      insert = True
    else:
      insert = False
      it = 0
      while it != len(self._filter):
        if x < self._filter[it]:
          del self._filter[it]
          insert = True
          continue
        it += 1

      if not insert:
        insert = True
        for it, _ in enumerate(self._filter):
          if self._filter[it].fobj < x.fobj:
            insert = False
            break

      if insert:
        self._filter.append(x)

    return insert

  def insert_infeasible(self, x: CandidatePoint):
    """_summary_

    :param x: _description_
    :type x: CandidatePoint
    :return: _description_
    :rtype: _type_
    """
    _ = self.filter_insertion(x=x)
    if not self._ref:
      return SUCCESS_TYPES.PS

    hx = x.h
    fx = x.fobj
    hr = self._ref.h
    fr = self._ref.fobj

    # Failure
    if hx > hr or (hx == hr and fx >= fr):
      return SUCCESS_TYPES.US

    # Partial success
    if fx > fr:
      return SUCCESS_TYPES.PS

    #  FULL success
    return SUCCESS_TYPES.FS

  def get_best_infeasible(self):
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    if self._filter:
      return self._filter[-1]
    else:
      return None

  def get_best_infeasible_min_viol(self):
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    return self._filter[0]

  def select_poll_center(self):
    """_summary_
    """
    best_infeasible: CandidatePoint = self.get_best_infeasible()
    self._sec_poll_center = None
    if not self._best_feasible and not best_infeasible:
      self._prim_poll_center = None
      return
    if not best_infeasible:
      self._prim_poll_center = self._best_feasible
      return

    if not self._best_feasible:
      self._prim_poll_center = best_infeasible
      return

    last_poll_center: Optional[CandidatePoint] = None
    if self._params.get_barrier_type() == BARRIER_TYPES.PB:
      last_poll_center = self._prim_poll_center
      if best_infeasible.fobj[0] < (
              self._best_feasible.fobj[0] - self._rho_leaps):
        self._prim_poll_center = best_infeasible
        self._sec_poll_center = self._best_feasible
      else:
        self._prim_poll_center = self._best_feasible
        self._sec_poll_center = best_infeasible

      if last_poll_center is None or self._prim_poll_center != last_poll_center:
        self._rho_leaps += 1

  def set_h_max(self, h_max):
    """_summary_

    :param h_max: _description_
    :type h_max: _type_
    """
    self._h_max = np.round(h_max, 2)
    if self._filter is not None and self._filter[0].h > self._h_max:
      self._filter = None
      return
    if self._filter is not None:
      it = 0
      while it != len(self._filter):
        if self._filter[it].h > self._h_max:
          del self._filter[it]
          continue
        it += 1

  def insert(self, x: CandidatePoint):
    """
    Insertion of a candidate point in the barrier    
    """
    if not x.evaluated:
      raise RuntimeError(
          "This point hasn't been evaluated yet and cannot be inserted into the barrier object!")

    if x.status == DESIGN_STATUS.ERROR:
      self._one_eval_succ = SUCCESS_TYPES.US
    if self._all_inserted is None:
      self._all_inserted = []
    self._all_inserted.append(x)
    if x.status == DESIGN_STATUS.INFEASIBLE and (
            not x.is_extreme_barrier_passed or x.h > self._h_max):
      self._one_eval_succ = SUCCESS_TYPES.US
      return

    # insert_feasible or insert_infeasible:
    self._one_eval_succ = self.insert_feasible(
        x) if x.status == DESIGN_STATUS.FEASIBLE else self.insert_infeasible(x)

    if self._success is None or self._one_eval_succ.value > self._success.value:
      self._success = self._one_eval_succ

  def insert_vns(self):
    """ 
    Not required here
    """
    ...

  def update_and_reset_success(self):
    """
    barrier update: invoked by Evaluator_Control::eval_lop()  
    """
    if self._params.get_barrier_type() == BARRIER_TYPES.PB and self._success != SUCCESS_TYPES.US:
      if self._success == SUCCESS_TYPES.PS:
        if self._filter is None:
          raise RuntimeError("filter empty after a partial success")
        it = len(self._filter)-1
        while True:
          if self._filter[it].h < self._h_max:
            self.set_h_max(self._filter[it].h)
            break
          if it == 0:
            break
          it -= 1
      if self._filter is not None:
        self._ref = self.get_best_infeasible()
      if self._ref is not None:
        self.set_h_max(self._ref.h)
        if self._ref.status is DESIGN_STATUS.INFEASIBLE:
          self.insert_infeasible(self._ref)

        if self._ref.status is DESIGN_STATUS.FEASIBLE:
          self.insert_feasible(self._ref)

        if not (self._ref.status is DESIGN_STATUS.INFEASIBLE or
                self._ref.status is DESIGN_STATUS.FEASIBLE):
          self.insert(self._ref)

    # reset success types:
    self._one_eval_succ = self._success = SUCCESS_TYPES.US

  def reset(self):
    """
    Reset the barrier
    """

    self._prefilter = None
    self._filter = None
    self._best_feasible = None
    self._ref = None
    self._rho_leaps = 0
    # self._poll_center = None
    self._sec_poll_center = None

    #     self._params.reset_PEB_changes()

    self._peb_changes = 0
    self._peb_filter_reset = 0

    self._peb_lop = None
    self._all_inserted = None

    self._one_eval_succ = _success = SUCCESS_TYPES.US


@dataclass
class BarrierMO(BarrierBase):
  """_summary_

  :param BarrierBase: _description_
  :type BarrierBase: _type_
  :raises RuntimeWarning: _description_
  :raises IOError: _description_
  :raises IOError: _description_
  :raises IOError: _description_
  :raises IOError: _description_
  :raises IOError: _description_
  :raises IOError: _description_
  :raises IOError: _description_
  :raises IOError: _description_
  :raises ValueError: _description_
  :raises IOError: _description_
  :raises ValueError: _description_
  :raises ValueError: _description_
  :raises ValueError: _description_
  :raises IOError: _description_
  :return: _description_
  :rtype: _type_
  """
  ims: int = None
  h_max: float = None
  elements: List[CandidatePoint] = None
  meshes: List[Gmesh] = None
  parent_indexes: List[int] = None
  within_fk: List[bool] = None
  within_uk: List[bool] = None  # Filtered points
  within_ik: List[bool] = None  # best infeasible points
  max_size: int = 30000
  last_index: int = None

  _current_incumbent_feas: Optional[CandidatePoint] = None
  _current_incumbent_inf: Optional[CandidatePoint] = None
  _fixed_variables: Optional[CandidatePoint] = None
  _x_filter_inf: Optional[List[CandidatePoint]] = None
  _nobj: int = 1
  _bb_inputs_type: Optional[List[VAR_TYPE]] = None
  _incumbent_selection_param: int = 1

  def __init__(self, param: Parameters, options: Options,
               eval_point_list: Optional[List[CandidatePoint]] = None):
    super(BarrierBase, self).__init__(hMax=param.h_max)

    self._nobj = param.nobj
    self._fixed_variables = param.fixed_variables
    self._bb_inputs_type = param.var_type
    self._incumbent_selection_param = param.incumbent_selection_param
    self.barrier_initialized_from_cache = param.barrier_initialized_from_cache
    self._dtype = DType(options.precision)
    self._x_feas = []
    self._x_inf = []
    self._x_filter_inf = []
    self._h_max = param.h_max
    self.last_index = -1
    self.elements = [None] * self.max_size  # placeholder for OVector
    self.meshes = [None] * self.max_size    # placeholder for GranularMesh
    # Assuming parent indexes are integers
    self.parent_indexes = [0] * self.max_size
    self.within_fk = [False] * self.max_size
    self.within_uk = [False] * self.max_size
    self.within_ik = [False] * self.max_size

    self.check_h_max()
    if eval_point_list:
      self.init(eval_point_list=eval_point_list)

  @property
  def x_filter_inf(self) -> List[CandidatePoint]:
    return self._x_filter_inf

  @property
  def current_incumbent_feas(self) -> CandidatePoint:
    return self._current_incumbent_feas

  @property
  def current_incumbent_inf(self) -> CandidatePoint:
    return self._current_incumbent_inf

  @property
  def nobj(self):
    # pylint: disable=missing-docstring
    return self._nobj

  @nobj.setter
  def nobj(self, value: int) -> int:
    self._nobj = value

  def init(self, eval_point_list: Optional[List[Point]] = None):
    _, _, _, _ = self.update_with_points(eval_point_list)

  def check_is_full(self):
    """_summary_

    :raises RuntimeWarning: _description_
    """
    if self.last_index == self.max_size:
      raise RuntimeWarning(
          "Maximum size of barrier reached: cannot add element")

  def check_mesh_parameters(self, x: CandidatePoint = None):
    """_summary_

    :param x: _description_, defaults to None
    :type x: CandidatePoint, optional
    :raises IOError: _description_
    :raises IOError: _description_
    :raises IOError: _description_
    :raises IOError: _description_
    """
    mesh = copy.deepcopy(x.mesh)

    mesh_size_correction: int = 0

    if mesh.get_delta_mesh_size().size != x.n_dimensions:
      mesh_size_correction = sum(self._fixed_variables.defined)

    if (mesh.get_delta_mesh_size().size + mesh_size_correction != x.n_dimensions
        or mesh.get_delta_frame_size().size + mesh_size_correction != x.n_dimensions
            or mesh.getMeshIndex().size + mesh_size_correction != x.n_dimensions):
      raise IOError(
          "Error: Mesh parameters dimensions are not compatible with EvalPoint dimension.")

    if not mesh.get_delta_mesh_size().is_all_defined():
      raise IOError(
          "Error: some MeshSize components of EvalPoint passed to MO Barrier are not defined.")

    if not mesh.get_delta_frame_size().is_all_defined():
      raise IOError(
          "Error: some FrameSize components of EvalPoint passed to MO Barrier are not defined.")

    if not mesh.getMeshIndex().is_all_defined():
      raise IOError(
          "Error: some MeshIndex components of EvalPoint passed to MO Barrier ")

  def get_fk(self) -> np.ndarray[CandidatePoint]:
    """_summary_

    :return: _description_
    :rtype: np.ndarray[CandidatePoint]
    """
    fk_i = np.where(self.within_fk == np.array(True))[0]
    return np.array(self.elements)[fk_i] if len(fk_i) > 0 else np.empty((0,))

  def get_ik(self) -> np.ndarray[CandidatePoint]:
    """_summary_

    :return: _description_
    :rtype: np.ndarray[CandidatePoint]
    """
    ik_i = np.where(self.within_ik == np.array(True))[0]
    return np.array(self.elements)[ik_i] if len(ik_i) > 0 else np.empty((0,))

  def get_uk(self) -> np.ndarray[CandidatePoint]:
    """_summary_

    :return: _description_
    :rtype: np.ndarray[CandidatePoint]
    """
    uk_i = np.where(self.within_uk == np.array(True))[0]
    return np.array(self.elements)[uk_i] if len(uk_i) > 0 else np.empty((0,))

  def get_all_points(self) -> List[CandidatePoint]:
    """_summary_

    :return: _description_
    :rtype: List[CandidatePoint]
    """
    all_points: List[CandidatePoint] = []
    if self._x_feas is None:
      self._x_feas = []
    for cp in self._x_feas:
      all_points.append(cp)
    # all_points = self.non_dominated_sort(all_points)

    # if self._xInf is None:
    #   self._xInf = []
    # xInf = self.getFilteredBestInfPoints()
    # xInf = self.non_dominated_sort(xInf)
    for cp in self.get_ik():
      all_points.append(cp)

    all_points = self.non_dominated_sort(all_points)

    return all_points

  def update_with_points(
          self, eval_point_list: List[CandidatePoint] = None):
    """_summary_

    :param eval_point_list: _description_, defaults to None
    :type eval_point_list: List[CandidatePoint], optional
    :raises IOError: _description_
    :return: _description_
    :rtype: _type_
    """
    updated = False
    updated_feas = False
    updated_inf = False
    insertion_flag: List[INSERTION_FLAG] = [
        INSERTION_FLAG.NO_IMPROVEMENT] * len(eval_point_list)
    ci = 0
    for cp in eval_point_list:
      self.check_mesh_parameters(cp)

      if not cp.evaluated or cp.status == DESIGN_STATUS.ERROR:
        continue

      if cp.fs.size != self._nobj:
        raise IOError(
            f"Barrier update: number of objectives is equal to {self._nobj}.\
            Trying to add this point with number of objectives {cp.fs.size}")
      if cp.status == DESIGN_STATUS.FEASIBLE:
        updated_feas, insertion_flag[ci] = self.update_feas_with_point(
            eval_point=cp) or updated_feas
      elif cp.status == DESIGN_STATUS.INFEASIBLE:
        updated_inf, insertion_flag[ci] = self.update_inf_with_point(
            eval_point=cp) or updated_feas

      # // Do separate loop on evalPointList
    # // Second loop update the bestInfeasible.
    # // Use the flag oneFeasEvalFullSuccess.
    # // If the flag is true hmax will not change.
    # A point improving the best infeasible should not replace it.

      if not cp.evaluated or cp.status == DESIGN_STATUS.ERROR:
        continue

      ci += 1

    updated = updated or updated_feas or updated_inf

    if updated:
      self.set_n()
      self.update_current_incumbents()

    return updated, updated_feas, updated_inf, insertion_flag

  def update_current_incumbents(self):
    """_summary_
    """
    self.update_current_incumbent_feas()
    self.update_current_incumbent_inf()

  def set_h_max(self, h_max=None):
    """_summary_

    :param h_max: _description_, defaults to None
    :type h_max: _type_, optional
    """
    old_h_max = self._h_max
    self._h_max = h_max
    self.check_h_max()
    if h_max < old_h_max and h_max > 0:
      self.update_x_inf_and_filter_inf_after_h_max_set()
    self.update_current_incumbent_inf()

  def update_x_inf_and_filter_inf_after_h_max_set(self):
    """_summary_
    """
    if len(self._x_inf) == 0:
      return

    current_ind = 0

    is_in_x_inf = [True] * len(self._x_inf)
    for x_inf in self._x_inf:
      h = x_inf.h
      if h > self._h_max:
        is_in_x_inf[current_ind] = False
      current_ind += 1

    current_ind = 0
    for _ in self._x_inf:
      if not is_in_x_inf[current_ind]:
        self._x_inf.pop(current_ind)
      current_ind += 1

    current_ind = 0
    is_in_x_filter_inf = [True] * len(self._x_filter_inf)

    for x_filter_inf in self._x_filter_inf:
      h = x_filter_inf.h
      if h > self._h_max:
        is_in_x_filter_inf[current_ind] = False
      current_ind += 1

    current_ind = 0
    for _ in self._x_filter_inf:
      if not is_in_x_filter_inf[current_ind]:
        self._x_filter_inf.pop(current_ind)
      current_ind += 1

    self._x_filter_inf = self.non_dominated_sort(self._x_filter_inf)

    # // And reinsert potential infeasible non dominated points into the set of infeasible
    # // solutions.
    current_ind = 0
    is_in_x_inf = [False] * len(self._x_filter_inf)
    for eval_point in self._x_filter_inf:
      if len(self._x_inf) > 0 and self.find_eval_point(self._x_filter_inf,
                                                       eval_point)[1] == self._x_inf[-1]:
        current_ind_tmp = 0
        insert = True
        for eval_point_inf in self._x_filter_inf:
          if current_ind_tmp != current_ind:
            comp_flag = eval_point.__compare__(eval_point_inf, True)
            if comp_flag == COMPARE_TYPE.DOMINATED:
              insert = False
              break
            elif comp_flag == COMPARE_TYPE.DOMINATING:
              is_in_x_inf[current_ind_tmp] = False
          current_ind_tmp += 1
        is_in_x_inf[current_ind] = insert
      current_ind += 1

    for i in enumerate(is_in_x_inf):
      if is_in_x_inf[i]:
        self._x_inf.append(self._x_filter_inf[i])

    self._x_inf = self.non_dominated_sort(self._x_inf)

  def clear_x_feas(self):
    self._x_feas.clear()

    self.update_current_incumbents()

  def clear_x_inf(self):
    self._x_inf.clear()
    self._x_filter_inf.clear()
    # Update the current incumbent inf.
    # Only the infeasible one depends on XInf (not the case for the feasible one).
    self.update_current_incumbent_inf()

  def compute_success_type(
          self, eval1: CandidatePoint = None, eval2: CandidatePoint = None,
          h_max: int = np.inf):
    """_summary_

    :param eval1: _description_, defaults to None
    :type eval1: CandidatePoint, optional
    :param eval2: _description_, defaults to None
    :type eval2: CandidatePoint, optional
    :param h_max: _description_, defaults to np.inf
    :type h_max: int, optional
    :return: _description_
    :rtype: _type_
    """
    success: SUCCESS_TYPES = SUCCESS_TYPES.US
    if eval1 is not None:
      if eval2 is None:
        h = eval1.h
        if h > h_max or h == np.inf:
          success = SUCCESS_TYPES.US
        else:
          if eval1.status == DESIGN_STATUS.FEASIBLE:
            success = SUCCESS_TYPES.FS
          else:
            success = SUCCESS_TYPES.PS
      else:
        if eval1.__compare__(eval2, self._h_max) == COMPARE_TYPE.DOMINATING:
          # Whether eval1 and eval2 are both feasible, or both
          # infeasible, dominance means FULL_SUCCESS.
          success = SUCCESS_TYPES.FS
        elif eval1.status == DESIGN_STATUS.FEASIBLE and eval2.status == DESIGN_STATUS.FEASIBLE:
          success = SUCCESS_TYPES.US
        elif eval1.status != DESIGN_STATUS.FEASIBLE and eval2.status != DESIGN_STATUS.FEASIBLE:
          if eval1.h <= h_max and eval1.h < eval2.h and eval1.f > eval2.f:
            success = SUCCESS_TYPES.PS
        else:
          success = SUCCESS_TYPES.US
    return success

  def default_compute_success_type(
          self, eval_point1: CandidatePoint, eval_point2: CandidatePoint,
          h_max: float):
    """_summary_

    :param eval_point1: _description_
    :type eval_point1: CandidatePoint
    :param eval_point2: _description_
    :type eval_point2: CandidatePoint
    :param h_max: _description_
    :type h_max: float
    :return: _description_
    :rtype: _type_
    """
    success: SUCCESS_TYPES = SUCCESS_TYPES.US
    if eval_point1 and eval_point2:
      h = eval_point1.h
      if h > h_max or h == np.inf:
        # // Even if evalPoint2 is NULL, this case is still
        # // not a success.
        success = SUCCESS_TYPES.US
      elif eval_point1.status == DESIGN_STATUS.FEASIBLE:
        success = SUCCESS_TYPES.FS
      else:
        success = self.default_compute_success_type(
            eval_point1, eval_point2, h_max)
    return success

  def get_success_type_of_points(
          self, x_feas: CandidatePoint = None, x_inf: CandidatePoint = None):
    success_type = SUCCESS_TYPES.US
    success_type2 = SUCCESS_TYPES.US

    if self._current_incumbent_feas is not None or self._current_incumbent_inf is not None:
      if not self._current_incumbent_feas:
        success_type = self.default_compute_success_type(
            x_feas, self._current_incumbent_feas, self._h_max)
      if not self._current_incumbent_inf:
        success_type = self.default_compute_success_type(
            x_inf, self._current_incumbent_inf, self._h_max)
      if success_type2.value > success_type.value:
        success_type = success_type2

    return success_type

  def check_x_feas_is_feas(self, x_feas: CandidatePoint = None):
    if x_feas.evaluated and x_feas.status != DESIGN_STATUS.ERROR:
      h = x_feas.h
      if h != 0:
        raise IOError("Error: DMultiMadsBarrier: xFeas' h value must be 0.0")
      if x_feas.fs.size != self._nobj:
        raise IOError("Error: DMultiMadsBarrier: xFeas' F must be of size")

  def get_mesh_max_frame_size(self, pt: CandidatePoint):
    """_summary_

    :param pt: _description_
    :type pt: CandidatePoint
    :return: _description_
    :rtype: _type_
    """
    max_real_val = -1.0
    max_integer_val = -1.0

    # Detect if mesh is sub dimension and pt are in full dimension.
    mesh_is_in_subdimension = False
    mesh = pt.mesh
    if pt.mesh.n < pt.n_dimensions:
      mesh_is_in_subdimension = True

    shift = 0
    for i in range(pt.n_dimensions):
      # Do not use access the frame size for fixed variables.
      if mesh_is_in_subdimension and self._fixed_variables.defined[i]:
        shift += 1
      if self._bb_inputs_type[i] == VAR_TYPE.REAL:
        max_real_val = max(max_real_val, mesh.get_delta_frame_size(i-shift))
      elif self._bb_inputs_type[i] == VAR_TYPE.INTEGER:
        max_integer_val = max(
            max_integer_val, mesh.get_delta_frame_size(i-shift))
    if max_real_val > 0.0:
      # Some values are real: get norm inf on these values only.
      return max_real_val
    elif max_integer_val > 0.0:
      # No real value but some integer values: get norm inf on these values only
      return max_integer_val
    else:
      return 1.0  # Only binary variables: any elements of the iterate list can be chosen

  def update_current_incumbent_feas(self):
    """_summary_

    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """
    if len(self._x_feas) == 0:
      self._current_incumbent_feas = None
      return

    if len(self._x_feas) == 1:
      self._current_incumbent_feas = self._x_feas[0]
      return

    max_frame_size_feas_elts = -1.0

    # Set max frame size of all elements
    for xf in self._x_feas:
      max_frame_size_feas_elts = max(
          self.get_mesh_max_frame_size(xf),
          max_frame_size_feas_elts)

    # Select candidates
    can_be_frame_center: List[bool] = [False] * len(self._x_feas)
    nb_selected_candidates = 0

    # see article DMultiMads Algorithm 4.
    for i, _ in enumerate(self._x_feas):
      max_frame_size_elt = self.get_mesh_max_frame_size(self._x_feas[i])

      if (10**(-float(self._incumbent_selection_param)) * max_frame_size_feas_elts) \
              <= max_frame_size_elt and not self._x_feas[i].mesh.checkMeshForStopping():
        can_be_frame_center[i] = True
        nb_selected_candidates += 1

    if nb_selected_candidates == 0:
      return self._x_feas[0]

    # Only one point in the barrier.
    if nb_selected_candidates == 1:
      # for it in range(len(can_be_frame_center)):
      #   if can_be_frame_center[it]:
      #     break
      # if it == len(can_be_frame_center):
      #   raise IOError("Error: DMultiMadsBarrier, should not reach this condition")
      # else:
      #   selected_ind = it
      #   self._currentIncumbentFeas = self._xFeas[selected_ind]
      try:
        selected_ind = can_be_frame_center.index(True)
        self._current_incumbent_feas = self._x_feas[selected_ind]
      except ValueError as exc:
        raise ValueError(
            "Error: BarrierMO, should not reach this condition") from exc
    # Only two points in the barrier.
    elif ((nb_selected_candidates == 2) and (len(self._x_feas) == 2)):
      eval1 = self._x_feas[0]
      eval2 = self._x_feas[1]

      objv1 = eval1.f
      objv2 = eval2.f

      if max(np.abs(objv1.coordinates)) > max(np.abs(objv2.coordinates)):
        self._current_incumbent_feas = self._x_feas[0]
      else:
        self._current_incumbent_feas = self._x_feas[1]
      # 2 * self._incumbentSelectionParam + 2
      self._incumbent_selection_param = self._incumbent_selection_param
    # More than three points in the barrier.
    else:
      # First case: biobjective optimization. Points are already ranked by lexicographic order.
      if self._nobj:
        current_best_ind = 0
        max_gap = -1.0
        current_gap: float
        for obj in range(self._nobj):
          # Get extreme values value according to one objective
          fmin: float = self._x_feas[0].f[obj]
          fmax: float = self._x_feas[len(self._x_feas)-1].f[obj]
          # In this case, it means all elements of _xFeas are equal (return the first one)
          if fmin == fmax:
            break

          # Intermediate points
          for i in range(1, len(self._x_feas)-1):
            current_gap = self._x_feas[i+1].f[obj]-self._x_feas[i-1].f[obj]
            # self._x_feas[i-1].f[obj]
            current_gap /= (fmax-fmin)
            if (can_be_frame_center[i] and current_gap >= max_gap):
              max_gap = current_gap
              current_best_ind = i

          # Extreme points
          current_gap = 2 * (self._x_feas[len(self._x_feas)-1]
                             ).f[obj] - (self._x_feas[len(self._x_feas)-2]).f[obj]
          current_gap /= (fmax - fmin)
          if can_be_frame_center[len(self._x_feas)-1] and current_gap > max_gap:
            max_gap = current_gap
            current_best_ind = len(self._x_feas)-1

          current_gap = 2 * (self._x_feas[1]).f[obj] - (self._x_feas[0]).f[obj]
          current_gap /= (fmax - fmin)

          if can_be_frame_center[0] and current_gap > max_gap:
            max_gap = current_gap
            current_best_ind = 0
        self._current_incumbent_feas = self._x_feas[current_best_ind]
        # 2 * self._incumbentSelectionParam + 2
        self._incumbent_selection_param = self._incumbent_selection_param

      # // More than 2 objectives
      else:
        tmp_x_feas_p_ind: List[Tuple[CandidatePoint, int]] = [
            (CandidatePoint(), 0)]*len(self._x_feas)
        for i, _ in enumerate(tmp_x_feas_p_ind):
          tmp_x_feas_p_ind[i] = (self._x_feas[i], i)
        current_best_ind = 0
        max_gap = -1.0
        current_gap: float

        for obj, _ in enumerate(self.nobj):
          # Sort elements of tmpXFeasPInd according to objective obj (in ascending order)
          tmp_x_feas_p_ind = sorted(
              tmp_x_feas_p_ind, key=lambda x, obj=obj: x[0].f[obj])

          # Get extreme values value according to one objective
          fmin = tmp_x_feas_p_ind[0][0].f[obj]
          fmax = tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-1][0].f[obj]

          # Can happen for example when we have several minima or for more than three objectives
          if fmin == fmax:
            fmin = 0.
            fmax = 1.

          # Intermediate points
          for i in range(1, len(tmp_x_feas_p_ind)-1):
            current_gap = tmp_x_feas_p_ind[i +
                                           1][0].f[obj]-tmp_x_feas_p_ind[i-1][0].f[obj]
            current_gap /= (fmax - fmin)
            if can_be_frame_center[tmp_x_feas_p_ind[i][1]] and current_gap >= max_gap:
              max_gap = current_gap
              current_best_ind = tmp_x_feas_p_ind[i][1]

          # Extreme points
          current_gap = 2*(tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-1][0].f[obj]
                           ) - tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-2][0].f[obj]
          current_gap /= (fmax - fmin)

          if (can_be_frame_center[tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-1][1]]
                  and current_gap > max_gap):
            max_gap = current_gap
            current_best_ind = tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-1][1]

          current_gap = 2 * \
              tmp_x_feas_p_ind[1][0].f[obj] - tmp_x_feas_p_ind[0][0].f[obj]
          current_gap /= (fmax - fmin)

          if (can_be_frame_center[tmp_x_feas_p_ind[0][1]] and current_gap > max_gap):
            max_gap = current_gap
            current_best_ind = tmp_x_feas_p_ind[0][1]
        self._current_incumbent_feas = self._x_feas[current_best_ind]

  def infeasible_frame_center(self):
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    # Get the infeasible points (Uk)
    uk = self.get_uk()
    if len(uk) == 0:
      return 0

    ik_indexes = parent_indices(self.elements, self.get_ik())

    # Find the minimum mesh size value (Ik_Δ_min)
    xi_ind = np.argmin([elt.h for elt in np.array(self.elements)[ik_indexes]])
    ik_Δ_min = np.linalg.norm(  # pylint: disable=non-ascii-name
        self.meshes[ik_indexes[xi_ind]].get_frame_size_parameter(),
        np.inf)

    # Select candidates based on mesh size
    ik_selected_indexes = []
    for ind in ik_indexes:
      Δ_val = np.linalg.norm(
          self.meshes[ind].get_frame_size_parameter(), np.inf)
      if ik_Δ_min <= Δ_val:
        ik_selected_indexes.append(ind)

    if len(ik_selected_indexes) == 0:
      return 0

    # Only one point
    if len(ik_selected_indexes) == 1:
      return ik_selected_indexes[0]

    # Two points in the barrier
    if len(ik_selected_indexes) == 2 and len(ik_indexes) == 2:
      v1f = self.elements[ik_selected_indexes[0]].f
      v2f = self.elements[ik_selected_indexes[1]].f
      if np.linalg.norm(v1f, np.inf) > np.linalg.norm(v2f, np.inf):
        return ik_selected_indexes[0]
      else:
        return ik_selected_indexes[1]

    # More than two points
    frame_ind = 0
    maximum_gap = -1.0  # negative to handle the case where two points in Ik_selected_indexes

    for obj in range(self._nobj):
      fvalues = np.array([(self.elements[ind].f[obj], ind)
                         for ind in ik_indexes])
      sorted_indices = np.argsort(fvalues[:, 0])
      fvalues = fvalues[sorted_indices]

      # Get extreme points according to one objective
      fmin = fvalues[0][0]
      fmax = fvalues[-1][0]

      # Handle cases where there are multiple minima or more than three objectives
      if fmin == fmax:
        fmin = 0.0
        fmax = 1.0

      # Intermediate points
      for i in range(1, len(fvalues) - 1):
        current_gap = (fvalues[i+1][0] - fvalues[i-1][0]) / (fmax - fmin)
        if fvalues[i][1] in ik_selected_indexes and current_gap >= maximum_gap:
          maximum_gap = current_gap
          frame_ind = fvalues[i][1]

      # Extreme points
      current_gap = 2 * (fvalues[-1][0] - fvalues[-2][0]) / (fmax - fmin)
      if fvalues[-1][1] in ik_selected_indexes and current_gap > maximum_gap:
        maximum_gap = current_gap
        frame_ind = fvalues[-1][1]

      current_gap = 2 * (fvalues[1][0] - fvalues[0][0]) / (fmax - fmin)
      if fvalues[0][1] in ik_selected_indexes and current_gap > maximum_gap:
        maximum_gap = current_gap
        frame_ind = fvalues[0][1]
    self._current_incumbent_inf = copy.deepcopy(self.elements[int(frame_ind)])
    # return int(frame_ind)

  def feasible_frame_center(self, w: int):
    """_summary_

    :param w: _description_
    :type w: int
    :return: _description_
    :rtype: _type_
    """
    # Get the feasible points (Fk)
    fk = self.get_fk()
    if len(fk) == 0:
      return -1

    fk_indexes = parent_indices(self.elements, fk)

    # Get maximum mesh size value
    fk_Δ_max = 0.0  # pylint: disable=non-ascii-name
    for ind in fk_indexes:
      Δ_val = np.linalg.norm(
          (self.meshes[ind].get_delta_frame_size()).coordinates, np.inf)
      fk_Δ_max = max(fk_Δ_max, Δ_val)

    # Select candidates
    fk_selected_indexes = []
    for ind in fk_indexes:
      fk_elt_mesh = self.meshes[ind]
      Δ_val = np.linalg.norm(
          (fk_elt_mesh.get_delta_frame_size()).coordinates, np.inf)
      if (10.0**(-w) * fk_Δ_max) <= Δ_val and not fk_elt_mesh.checkMeshForStopping():
        fk_selected_indexes.append(ind)

    # The selection must always work
    if len(fk_selected_indexes) == 0:
      return fk_indexes[0]

    # Only one point
    if len(fk_selected_indexes) == 1:
      return fk_selected_indexes[0]

    # Two points in the barrier
    if len(fk_selected_indexes) == 2 and len(fk_indexes) == 2:
      v1f = self.elements[fk_selected_indexes[0]].f
      v2f = self.elements[fk_selected_indexes[1]].f
      if np.linalg.norm(v1f, np.inf) > np.linalg.norm(v2f, np.inf):
        return fk_selected_indexes[0]
      else:
        return fk_selected_indexes[1]

    # More than two points
    frame_ind = -1
    maximum_gap = -1.0  # negative to deal with the case where two points in Fk_selected_indexes

    for obj in range(len(self.elements[0].f)):
      fvalues = np.array([(self.elements[ind].f[obj], ind)
                         for ind in fk_indexes])
      sorted_indices = np.argsort(fvalues[:, 0])
      fvalues = fvalues[sorted_indices]

      # Get extreme points according to one objective
      fmin = fvalues[0][0]
      fmax = fvalues[-1][0]

      # Can happen for example when we have several minima or for more than three objectives
      if fmin == fmax:
        fmin = 0.0
        fmax = 1.0

      if self._nobj == 1:
        frame_ind = fvalues[0][1]
        return int(frame_ind)

      # Intermediate points
      for i in range(1, len(fvalues) - 1):
        current_gap = (fvalues[i+1][0] - fvalues[i-1][0]) / (fmax - fmin)
        if fvalues[i][1] in fk_selected_indexes and current_gap >= maximum_gap:
          maximum_gap = current_gap
          frame_ind = fvalues[i][1]

      # Extreme points
      current_gap = 2 * (fvalues[-1][0] - fvalues[-2][0]) / (fmax - fmin)
      if fvalues[-1][1] in fk_selected_indexes and current_gap > maximum_gap:
        maximum_gap = current_gap
        frame_ind = fvalues[-1][1]

      current_gap = 2 * (fvalues[1][0] - fvalues[0][0]) / (fmax - fmin)
      if fvalues[0][1] in fk_selected_indexes and current_gap > maximum_gap:
        maximum_gap = current_gap
        frame_ind = fvalues[0][1]

    return int(frame_ind)

  # Return the extent of the Pareto front (a customized one)
  def extent(self):
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    # Get the Pareto front (Fk)
    fk = self.get_fk()
    if len(fk) == 0:
      return 0

    fk_indexes = parent_indices(self.elements, fk)

    extent_val = 0

    for obj in range(len(self.elements[0].f)):
      fvalues = np.array([self.elements[ind].f[obj] for ind in fk_indexes])
      sorted_indices = np.argsort(fvalues)
      fvalues = fvalues[sorted_indices]

      # Get extreme points according to one objective
      fmin = fvalues[0]
      fmax = fvalues[-1]

      # If all values are the same, use abs(fmin)
      if fmin == fmax:
        extent_val += abs(fmin)
      else:
        extent_val += abs(fmax - fmin)

    return extent_val

  def update_barrier(self, h_max: float):
    """_summary_

    :param h_max: _description_
    :type h_max: float
    :raises ValueError: _description_
    """
    if h_max < 0:
      raise ValueError("h_max cannot be negative")
    self.h_max = h_max

    filter_flags = [False] * self.last_index

    # Mark Uk elements that exceed h_max
    for index, is_in_uk in enumerate(self.within_uk[:self.last_index]):
      if is_in_uk and self.elements[index].h > self.h_max:
        filter_flags[index] = True

    # Remove all Uk elements above the threshold
    for i in range(self.last_index):
      if filter_flags[i]:
        self.within_uk[i] = False

    # Remove all Ik elements above the threshold
    for i in range(self.last_index):
      if filter_flags[i]:
        self.within_ik[i] = False

    # Reinsert potential new non-dominated points into the set Ik
    self._update_ik_after_h_max_setting()

  def frame_centers(self, w: int, use_dom_selection=True):
    """_summary_

    :param w: _description_
    :type w: int
    :param use_dom_selection: _description_, defaults to True
    :type use_dom_selection: bool, optional
    :return: _description_
    :rtype: _type_
    """
    feasible_index = self.feasible_frame_center(w)

    infeasible_index = None
    if feasible_index == -1:
      infeasible_index = self.infeasible_frame_center()
    else:
      # Define the distance function
      def dom_distance(z1, z2):
        if any(z1 < z2):
          return np.linalg.norm(z2 - z1)
        else:
          return -np.linalg.norm(z2 - z1)

      if use_dom_selection:
        tmp_ind = -1
        tmp_dist = -float('inf')
        for ind in range(0, self.last_index+1):
          if self.within_ik[ind]:
            v = self.elements[ind]
            tmp_dom_distance = min(
                [sum(elt.f - np.minimum(v.f, elt.f)) for elt in self.get_fk()])
            if v.h <= self.h_max and tmp_dom_distance > tmp_dist:
              tmp_ind = ind
              tmp_dist = tmp_dom_distance

        if tmp_dist == 0:
          tmp_ind = 0
          tmp_dist = float('inf')
          for ind in range(0, self.last_index+1):
            if self.within_ik[ind]:
              v = self.elements[ind]
              tmp_dom_distance = min(
                  [sum(v.f - np.minimum(v.f, elt.f)) for elt in self.get_fk()])
              if v.h <= self.h_max and tmp_dom_distance < tmp_dist:
                tmp_ind = ind
                tmp_dist = tmp_dom_distance
        infeasible_index = tmp_ind
      else:
        tmp_ind = 0
        tmp_dist = float('inf')
        for ind in range(0, self.last_index+1):
          if self.within_ik[ind]:
            v = self.elements[ind]
            tmp_dom_distance = dom_distance(
                self.elements[feasible_index].f, v.f)
            if v.h <= self.h_max and tmp_dom_distance < tmp_dist:
              tmp_ind = ind
              tmp_dist = tmp_dom_distance
        infeasible_index = tmp_ind

    return {"feasible": feasible_index, "infeasible": infeasible_index}

  def _update_ik_after_h_max_setting(self):
    for index, is_in_uk in enumerate(self.within_uk[:self.last_index]):
      if is_in_uk and not self.within_ik[index]:
        # Check if the element indexed by 'index' can be inserted in the set Ik
        insert = True
        for index_2, is_in_uk_2 in enumerate(self.within_uk[:self.last_index]):
          if is_in_uk_2 and index != index_2:
            comp_flag = self._private_compare_ik_elements(
                self.elements[index], self.elements[index_2])
            if comp_flag == "dominating":
              self.within_ik[index_2] = False
            elif comp_flag in ["dominated", "equal"]:
              insert = False
              break
        self.within_ik[index] = insert

  def _private_compare_ik_elements(self, v1, v2):
    isbetter = False
    isworse = False

    # Iterate over corresponding elements in f attributes of v1 and v2
    for f1, f2 in zip(v1.f, v2.f):
      if f1 < f2:
        isbetter = True
      if f2 < f1:
        isworse = True
      if isworse and isbetter:
        break  # No need to continue once we know the result

    if isworse:
      if isbetter:
        return "nondominated"
      else:
        return "dominated"
    else:
      if isbetter:
        return "dominating"
      else:
        return "equal"

  def update_current_incumbent_inf(self):
    """_summary_
    """
    self.infeasible_frame_center()
    return
    # self._current_incumbent_inf = None
    # if len(self._x_feas) > 0 and len(self._x_inf) > 0:
    #   # // Get the infeasible solution with maximum dominance move below the _hMax threshold,
    #   # // according to the set of best feasible incumbent solutions.
    #   current_ind = 0
    #   max_dom_move = -np.inf

    #   for j in range(len(self._x_inf)):
    #     # // Compute dominance move
    #     # // = min \sum_{1}^m max(fi(y) - fi(x), 0)
    #     # //   y \in Fk

    #     tmp_dom_move = np.inf
    #     eval_inf = self._x_inf[j]
    #     h = eval_inf.h

    #     if h <= self._h_max:
    #       for x_feas in self._x_feas:
    #         sum_val = 0.
    #         eval_feas = x_feas
    #         for i in range(self._nobj):
    #           sum_val += max(eval_feas.f[i]-eval_inf.f[i], 0)
    #         if tmp_dom_move > sum_val:
    #           tmp_dom_move = sum_val

    #       # Get the maximum dominance move index
    #       if max_dom_move < tmp_dom_move:
    #         max_dom_move = tmp_dom_move
    #         current_ind = j

    #   # // In this case, all infeasible solutions are "dominated" in terms of fvalues
    #   # // by at least one element of Fk
    #   if np.isclose(max_dom_move, 0., rtol=1e-09, atol=1e-09):
    #     # // In this case, get the infeasible solution below the _hMax threshold which has
    #     # // minimal dominance move, when considered a maximization problem.
    #     min_dom_move = np.inf
    #     current_ind = 0
    #     for j in range(len(self._x_inf)):
    #       # // Compute dominance move
    #       # // = min \sum_{1}^m max(fi(x) - fi(y), 0)
    #       # //   y \in Fk
    #       tmp_dom_move = np.inf
    #       eval_inf = self._x_inf[j]
    #       h = eval_inf.h
    #       if h <= self._h_max:
    #         for x_feas in self._x_feas:
    #           sum_val = 0.
    #           eval_feas = x_feas
    #           # Compute \sum_{1}^m max (fi(x) - fi(y), 0)
    #           for i in range(self._nobj):
    #             sum_val += max(eval_inf.f[i] - eval_feas.f[i], 0.)
    #           if tmp_dom_move > sum_val:
    #             tmp_dom_move = sum_val

    #         # Get the minimal dominance move index
    #         if min_dom_move > tmp_dom_move:
    #           min_dom_move = tmp_dom_move
    #           current_ind = j
    #   self._current_incumbent_inf = self._x_inf[current_ind]
    # else:
    #   self._current_incumbent_inf = self.getFirstXIncInfNoXFeas() if len(
    #       self._x_inf) > 0 else None

  def get_x_inf_min_h(self):
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    ind_x_inf_min_h = 0
    h_min_val = np.inf

    for i, _ in enumerate(self._x_inf):
      my_eval = self._x_inf[i]
      h = my_eval.h

      # // By definition, all elements of _xInf or _xFilterInf have a well-defined
      # // h value. So, no need to check.

      if h < h_min_val:
        h_min_val = h
        ind_x_inf_min_h = i
    return self._x_inf[ind_x_inf_min_h]

  def get_first_x_inc_inf_no_x_feas(self):
    """_summary_

    :raises IOError: _description_
    :return: _description_
    :rtype: _type_
    """
    x_inf = None
    if len(self._x_filter_inf) == 0:
      return x_inf

    # Select candidates
    min_frame_size_inf_elts: float = self.get_mesh_max_frame_size(
        self.get_x_inf_min_h())
    can_be_frame_center: List[bool] = [False] * len(self._x_inf)
    nb_selected_candidates = 0

    for i, _ in enumerate(self._x_inf):
      max_frame_size_elt = self.get_mesh_max_frame_size(self._x_inf[i])
      if min_frame_size_inf_elts <= max_frame_size_elt:
        can_be_frame_center[i] = True
        nb_selected_candidates += 1

    # The selection must always work
    if nb_selected_candidates == 0:
      x_inf = self._x_inf[0]
    elif nb_selected_candidates == 1:
      for it, _ in enumerate(can_be_frame_center):
        if can_be_frame_center[it]:
          break
        it_index = it
      if it_index == len(can_be_frame_center):
        raise IOError(
            "Error: DMultiMadsBarrier, should not reach this condition")
      else:
        selected_ind = it_index
        x_inf = self._x_inf[selected_ind]
    elif ((nb_selected_candidates == 2) and (len(self._x_inf) == 2)):
      eval1 = self._x_inf[0]
      eval2 = self._x_inf[1]

      objv1 = eval1.f
      objv2 = eval2.f

      if max(np.abs(objv1.coordinates)) > max(np.abs(objv2.coordinates)):
        x_inf = self._x_inf[0]
      else:
        x_inf = self._x_inf[1]
    else:
      if self._nobj == 2:
        current_best_ind = 0
        max_gap = -1.
        current_gap: float

        for obj in range(self._nobj):
          # Get extreme values value according to one objective
          fmin: float = self._x_inf[0].f[obj]
          fmax: float = self._x_inf[len(self._x_inf)-1].f[obj]

          # In this case, it means all elements of _xFeas are equal (return the first one)
          if fmin == fmax:
            break

          # Intermediate points
          for i in range(1, self._x_inf-1):
            current_gap = self._x_inf[i+1].f[obj]-self._x_inf[i-1].f[obj]
            # self._x_inf[i-1].f[obj]
            current_gap /= (fmax-fmin)
            if (can_be_frame_center[i] and current_gap >= max_gap):
              max_gap = current_gap
              current_best_ind = i

          # Extreme points
          current_gap = 2 * (self._x_inf[len(self._x_inf)-1]
                             ).f[obj] - (self._x_inf[len(self._x_inf)-2]).f[obj]
          current_gap /= (fmax - fmin)
          if can_be_frame_center[len(self._x_inf)-1] and current_gap > max_gap:
            max_gap = current_gap
            current_best_ind = len(self._x_inf)-1

          current_gap = 2 * (self._x_inf[1]).f[obj] - (self._x_inf[0]).f[obj]
          current_gap /= (fmax - fmin)

          if can_be_frame_center[0] and current_gap > max_gap:
            max_gap = current_gap
            current_best_ind = 0
        x_inf = self._x_inf[current_best_ind]
      # // More than 2 objectives
      else:
        tmp_x_inf_p_ind: List[Tuple[CandidatePoint, int]] = [
            (CandidatePoint(), 0)]*len(self._x_inf)
        for i, _ in enumerate(tmp_x_inf_p_ind):
          tmp_x_inf_p_ind[i] = (self._x_inf[i], i)
        current_best_ind = 0
        max_gap = -1.0
        current_gap: float

        for obj, _ in range(self._nobj):
          # Sort elements of tmpXFeasPInd according to objective obj (in ascending order)
          tmp_x_inf_p_ind = sorted(
              tmp_x_inf_p_ind, key=lambda x, obj=obj: x[0].f[obj])

          # Get extreme values value according to one objective
          fmin = tmp_x_inf_p_ind[0][0].f[obj]
          fmax = tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-1][0].f[obj]

          # Can happen for exemple when we have several minima or for more than three objectives
          if fmin == fmax:
            fmin = 0.
            fmax = 1.

          # Intermediate points
          for i in range(1, len(tmp_x_inf_p_ind)-1):
            current_gap = tmp_x_inf_p_ind[i+1][0].f[obj]-tmp_x_inf_p_ind[i-1][0].f[obj]
            current_gap /= (fmax - fmin)
            if can_be_frame_center[tmp_x_inf_p_ind[i][1]] and current_gap >= max_gap:
              max_gap = current_gap
              current_best_ind = tmp_x_inf_p_ind[i][1]

          # Extreme points
          current_gap = 2*(tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-1][0].f[obj]
                           ) - tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-2][0].f[obj]
          current_gap /= (fmax - fmin)

          if (can_be_frame_center[tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-1][1]]
                  and current_gap > max_gap):
            max_gap = current_gap
            current_best_ind = tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-1][1]

          current_gap = 2 * \
              tmp_x_inf_p_ind[1][0].f[obj] - tmp_x_inf_p_ind[0][0].f[obj]
          current_gap /= (fmax - fmin)

          if (can_be_frame_center[tmp_x_inf_p_ind[0][1]] and current_gap > max_gap):
            max_gap = current_gap
            current_best_ind = tmp_x_inf_p_ind[0][1]
        x_inf = self._x_inf[current_best_ind]

    return x_inf

  def update_inf_with_point(self, eval_point: CandidatePoint = None):
    """_summary_

    :param eval_point: _description_, defaults to None
    :type eval_point: CandidatePoint, optional
    :raises ValueError: _description_
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """
    updated = False
    insertion_flag: INSERTION_FLAG = None
    insertion_best_flag: INSERTION_FLAG = INSERTION_FLAG.IS_DOMINATED
    prev_nb_best_inf_pts = len(self._x_inf)

    if not eval_point.evaluated:
      raise ValueError(
          "Cannot update the barrier elements based on unevaluated candidate.")

    if eval_point.is_feasible():
      raise ValueError(
          "Trying to insert a feasible point into the set of infeasible points.")

    if eval_point.h >= self._h_max:
      self.last_index += 1
      self.elements[self.last_index] = copy.deepcopy(eval_point)
      self.meshes[self.last_index] = copy.deepcopy(eval_point.mesh)
      insertion_flag = INSERTION_FLAG.REJECTED
      updated = False
      self.within_uk[self.last_index] = True
      self.within_ik[self.last_index] = True
      return updated, insertion_flag

    current_filtered_elements = [self.elements[i]
                                 for i in range(self.max_size) if self.within_uk[i]]
    nb_filtered_elements = len(current_filtered_elements)
    prev_nb_best_inf_pts = sum(
        1 for x in self.within_ik[:self.last_index] if x)

    if self._x_inf is None:
      self._x_inf = []

    if self._x_filter_inf is None:
      self._x_filter_inf = []

    # Empty set case (possibly new infeasible point is the first point)
    if len(self._x_inf) <= 0:
      self.last_index += 1
      self.elements[self.last_index] = copy.deepcopy(eval_point)
      self.meshes[self.last_index] = copy.deepcopy(eval_point.mesh)
      self._x_inf.append(eval_point)
      self._x_filter_inf.append(eval_point)
      self._current_incumbent_inf = self._x_inf[0]
      updated = True
      insertion_flag = INSERTION_FLAG.IMPROVES

      return updated, insertion_flag

    # Insertion into the two sets of infeasible non-dominated points
    # Try to insert into _x_inf_filter
    insert = True

    is_in_x_inf_filter: List[bool] = [True] * nb_filtered_elements
    # Check if v dominates an element of Uk
    for index in range(nb_filtered_elements):
      corresponding_index = parent_indices(
          self.elements, current_filtered_elements)[index]
      comp_flag = eval_point.__compare__(self.elements[corresponding_index])
      if comp_flag == COMPARE_TYPE.DOMINATING:
        self.within_uk[corresponding_index] = False
        self.within_ik[corresponding_index] = False
        insertion_best_flag = INSERTION_FLAG.DOMINATES
        updated = True
        is_in_x_inf_filter[index] = False
      elif comp_flag in [COMPARE_TYPE.DOMINATED, COMPARE_TYPE.EQUAL]:
        insert = False
        break

    insertion_flag = INSERTION_FLAG.IMPROVES

    # is_in_x_inf_filter: List[bool] = [True] * len(self._xFilterInf)
    # current_ind = 0
    # for x_filter_inf in self._xFilterInf:
    #   comp_flag = eval_point.__compare__(x_filter_inf)
    #   if comp_flag == COMPARE_TYPE.DOMINATED:
    #     insert = False
    #     break
    #   elif comp_flag == COMPARE_TYPE.DOMINATING:
    #     insertion_best_flag = INSERTION_FLAG.DOMINATES
    #     updated = True
    #     is_in_x_inf_filter[current_ind] = False
    #   elif comp_flag == COMPARE_TYPE.EQUAL:
    #     if (not keep_all_points):
    #       insert = False
    #       break
    #     if self.findEvalPoint(self._xFilterInf, eval_point)[0]:
    #       insert = False
    #     else:
    #       updated = True
    #       break
    #   current_ind += 1

    if self.extends_pf(eval_point, False):
      insertion_flag = INSERTION_FLAG.EXTENDS

    self.last_index += 1
    self.elements[self.last_index] = copy.deepcopy(eval_point)
    self.meshes[self.last_index] = copy.deepcopy(eval_point.mesh)

    # Add the point into the set of filter points
    if insert:
      self.within_uk[self.last_index] = True
      # Remove all dominated elements of _xInfFilter
      current_ind = 0
      indices_to_remove = []
      for i in range(nb_filtered_elements):
        if not is_in_x_inf_filter[i]:
          indices_to_remove.append(i)
        current_ind += 1

      self._x_filter_inf.append(eval_point)

      for index in sorted(indices_to_remove, reverse=True):
        del self._x_filter_inf[index]
      # self._xFilterInf = self.non_dominated_sort(self._xFilterInf)
    else:
      return updated, INSERTION_FLAG.IS_DOMINATED

    insertion_best_flag = INSERTION_FLAG.IS_DOMINATED

    if insert:
      current_ind = 0
      is_in_x_inf = [True] * len(self._x_inf)
      domination_flags = [False] * self.last_index

      for index, x_inf in enumerate(self._x_inf):
        comp_flag = eval_point.__compare__(x_inf)
        if comp_flag == COMPARE_TYPE.DOMINATING:
          domination_flags[index] = True
          insertion_best_flag = INSERTION_FLAG.DOMINATES
          insert = True
          is_in_x_inf[index] = False
        elif comp_flag in [COMPARE_TYPE.DOMINATED, COMPARE_TYPE.EQUAL]:
          insertion_best_flag = INSERTION_FLAG.IS_DOMINATED
          updated = False

      # Update the set of best infeasible points (Ik)
      for index, flag in enumerate(domination_flags):
        if flag:
          self.within_ik[index] = False
      self.within_ik[self.last_index] = insert

      if insert:
        indices_to_remove = []
        for i in range(len(self._x_inf)):
          if not is_in_x_inf[i]:
            indices_to_remove.append(i)
        updated = True
        self._x_inf.append(eval_point)

        for index in sorted(indices_to_remove, reverse=True):
          del self._x_inf[index]

        self._x_inf = self.non_dominated_sort(self._x_inf)

      if insertion_flag == INSERTION_FLAG.EXTENDS:
        return updated, insertion_flag

      if insertion_best_flag == INSERTION_FLAG.IS_DOMINATED:
        return updated, INSERTION_FLAG.NO_IMPROVEMENT
      else:
        if len(self._x_inf) <= prev_nb_best_inf_pts:
          return updated, INSERTION_FLAG.DOMINATES
        else:
          return updated, insertion_best_flag

    return updated, insertion_best_flag

  def non_dominated_sort(
          self, points: List[CandidatePoint] = None, only_f: bool = False):
    """ Perform biobjective nondominated sorting """
    fronts = [[]]  # List to store different fronts
    # Array to count number of points dominating each point
    dominated_count = [0] * len(points)

    for i, p in enumerate(points):
      for j, q in enumerate(points):
        if i != j and p.__compare__(q) == COMPARE_TYPE.DOMINATED:
          dominated_count[i] += 1

      if dominated_count[i] == 0:
        fronts[0].append(p)

    # Sort each front lexicographically
    for front in fronts:
      front.sort()

    # Flatten the fronts into a single list
    sorted_points = [point for front in fronts for point in front]

    return sorted_points

  def update_feas_with_point(
          self, eval_point: CandidatePoint = None, keep_all_points: bool = None):
    """_summary_

    :param eval_point: _description_, defaults to None
    :type eval_point: CandidatePoint, optional
    :param keep_all_points: _description_, defaults to None
    :type keep_all_points: bool, optional
    :raises ValueError: _description_
    :raises IOError: _description_
    :return: _description_
    :rtype: _type_
    """
    updated = False
    insertion_flag: INSERTION_FLAG = None
    if not eval_point.is_feasible():
      raise ValueError(
          "Trying to insert an infeasible element into the set of feasible points.")
    if eval_point.evaluated and eval_point.status == DESIGN_STATUS.FEASIBLE:
      if eval_point.fs.size != self._nobj:
        raise IOError(
            f"Barrier update: number of objectives is equal to {self._nobj}. \
            Trying to add this point with number of objectives {eval_point.fs.size}")

      if self._x_feas is None:
        self._x_feas = []

      if len(self._x_feas) == 0:
        self.last_index += 1
        self._x_feas.append(eval_point)
        updated = True
        self.meshes[self.last_index] = copy.deepcopy(eval_point.mesh)
        insertion_flag = INSERTION_FLAG.IMPROVES
        self._current_incumbent_feas = self._x_feas[0]
        self.elements[self.last_index] = copy.deepcopy(eval_point)
        self.within_fk[self.last_index] = True
        return updated, insertion_flag

      insert = True
      insertion_flag = INSERTION_FLAG.IMPROVES

      keep_in_x_feas = [True] * len(self._x_feas)
      current_ind = 0
      for xf in self._x_feas:
        comp_flag: COMPARE_TYPE = eval_point.__compare__(xf)
        if comp_flag == COMPARE_TYPE.DOMINATED:
          insert = False
          insertion_flag = INSERTION_FLAG.IS_DOMINATED
          break
        elif comp_flag == COMPARE_TYPE.DOMINATING:
          updated = True
          keep_in_x_feas[current_ind] = False
          # INSERTION_FLAG.DOMINATES
        elif comp_flag == COMPARE_TYPE.EQUAL or comp_flag == COMPARE_TYPE.DOMINATED:
          insertion_flag = INSERTION_FLAG.IS_DOMINATED
          if not keep_all_points:
            insert = False
            break

          if self.find_eval_point(self._x_feas, eval_point)[0]:
            insert = False
          else:
            updated = True
          break
        current_ind += 1
      if insert:
        current_ind = 0
        for cp in self._x_feas:
          if cp.__compare__(eval_point) == COMPARE_TYPE.DOMINATED:
            self._x_feas.pop(current_ind)
          current_ind += 1
        updated = True
        my_dir = copy.deepcopy(eval_point.direction)
        if my_dir is not None:
          eval_point.mesh.enlarge_delta_frame_size(direction=my_dir)

        self._x_feas.append(eval_point)

        # Sort according to lexicographic order.
        self._x_feas = self.non_dominated_sort(self._x_feas)

      if self.extends_pf(eval_point, True):
        insertion_flag = INSERTION_FLAG.EXTENDS

    self.last_index += 1
    self.elements[self.last_index] = copy.deepcopy(eval_point)
    self.meshes[self.last_index] = copy.deepcopy(eval_point.mesh)
    if insert:
      self.within_fk[self.last_index] = True
    return updated, insertion_flag

  def extends_pf(self, cp: CandidatePoint, is_feasible: bool):
    """
    Extend Pareto front

    :param cp: _description_
    :type cp: CandidatePoint
    :param is_feasible: _description_
    :type is_feasible: bool
    :return: _description_
    :rtype: _type_
    """
    temp_ndp: List[CandidatePoint] = []
    if is_feasible:
      temp_ndp = copy.deepcopy(self._x_feas)
    else:
      temp_ndp = copy.deepcopy(self._x_inf)

    if len(temp_ndp) <= 0:
      return False

    ideal_v = [np.inf] * self._nobj
    for elt in temp_ndp:
      ideal_v = [min(elt.f[i], ideal_v[i])
                 for i in range(self._nobj)]  # min(elt.fobj, ideal_v)

    return any([cp.fobj[i] < ideal_v[i] for i in range(self._nobj)])

  # def update_barrier(self, h_max: float):
  #   if h_max < 0:
  #     raise ValueError("h_max cannot be negative")

  #   self.h_max = h_max
  #   filter_flags = [False] * self.last_index


# @dataclass
# class BarrierMO(BarrierBase):
#   """ """

#   ims: int = None
#   h_max: float = None
#   elements: List[CandidatePoint] = None
#   meshes: List[Gmesh] = None
#   parent_indexes: List[int] = None
#   within_Fk: List[bool] = None
#   within_Uk: List[bool] = None
#   within_Ik: List[bool] = None
#   max_size: int = 30000
#   last_index: int = None

#   _currentIncumbentFeas: Optional[CandidatePoint] = None
#   _currentIncumbentInf: Optional[CandidatePoint] = None
#   _fixedVariables: Optional[CandidatePoint] = None
#   _xFilterInf: Optional[List[CandidatePoint]] = None
#   _nobj: int = 0
#   _bbInputsType: Optional[List[VAR_TYPE]] = None
#   _incumbentSelectionParam: int = 1

#   def __init__(self, param: Parameters, options: Options,
# eval_point_list: Optional[List[CandidatePoint]]= None):
#     super(BarrierBase, self).__init__(hMax=param.h_max)

#     self._nobj = param.nobj
#     self._fixedVariables = param.fixed_variables
#     self._bbInputsType = param.var_type
#     self._incumbentSelectionParam = param.incumbentincumbentSelectionParam
#     self.barrierInitializedFromCache = param.barrierInitializedFromCache
#     self._dtype = DType(options.precision)
#     self._xFeas = []
#     self._xInf = []
#     self._xFilterInf = []
#     self._h_max = param.h_max
#     self.last_index = -1
#     self.elements = [None] * self.max_size  # placeholder for OVector
#     self.meshes = [None] * self.max_size    # placeholder for GranularMesh
#     self.parent_indexes = [0] * self.max_size  # Assuming parent indexes are integers
#     self.within_Fk = [False] * self.max_size
#     self.within_Uk = [False] * self.max_size
#     self.within_Ik = [False] * self.max_size


#     self.checkHMax()
#     if eval_point_list:
#       self.init(eval_point_list=eval_point_list)


#   def init(self, eval_point_list: Optional[List[Point]] = None):
#     _, _, _, _ = self.updateWithPoints(eval_point_list)

#   def check_is_full(self):
#     if self.last_index == self.max_size:
#       raise RuntimeWarning("Maximum size of barrier reached: cannot add element")

#   def checkMeshParameters(self, x: CandidatePoint = None):
#     mesh = copy.deepcopy(x.mesh)


#     mesh_size_correction: int = 0

#     if mesh.getdeltaMeshSize().size != x._n:
#       mesh_size_correction = sum(self._fixedVariables.defined)

#     if (mesh.getdeltaMeshSize().size + mesh_size_correction != x._n
#         or mesh.getDeltaFrameSize().size + mesh_size_correction != x._n
#         or mesh.getMeshIndex().size + mesh_size_correction != x._n):
#       raise IOError("Error: Mesh parameters dimensions
#  are not compatible with EvalPoint dimension.")

#     if not mesh.getdeltaMeshSize().is_all_defined():
#       raise IOError("Error: some MeshSize components of
# EvalPoint passed to MO Barrier are not defined.")

#     if not mesh.getDeltaFrameSize().is_all_defined():
#       raise IOError("Error: some FrameSize components
# of EvalPoint passed to MO Barrier are not defined.")

#     if not mesh.getMeshIndex().is_all_defined():
#       raise IOError("Error: some MeshIndex components of EvalPoint passed to MO Barrier ")

#   def get_Fk(self) -> np.ndarray[CandidatePoint]:
#     Fk_i = np.where(self.within_Fk==np.array(True))[0]
#     return np.array(self.elements)[Fk_i] if len(Fk_i) > 0 else np.empty((0,))

#   def get_Ik(self) -> np.ndarray[CandidatePoint]:
#     Ik_i = np.where(self.within_Ik==np.array(True))[0]
#     return np.array(self.elements)[Ik_i] if len(Ik_i) > 0 else np.empty((0,))

#   def get_Uk(self) -> np.ndarray[CandidatePoint]:
#     Uk_i = np.where(self.within_Uk==np.array(True))[0]
#     return np.array(self.elements)[Uk_i] if len(Uk_i) > 0 else np.empty((0,))

#   def updateWithPoints(self, eval_point_list: List[CandidatePoint]=
# None, keep_all_points: bool = None):
#     updated = False
#     updated_feas = False
#     updated_inf = False
#     insertion_flag: List[INSERTION_FLAG] = [INSERTION_FLAG.NO_IMPROVEMENT] * len(eval_point_list)
#     ci = 0
#     for cp in eval_point_list:
#       self.checkMeshParameters(cp)

#       if not cp.evaluated or cp.status == DESIGN_STATUS.ERROR:
#         continue

#       if cp.fs.size != self._nobj:
#         raise IOError(f"Barrier update: number of objectives is equal to {self._nobj}.
# Trying to add this point with number of objectives {cp.fs.size}")
#       if cp.status == DESIGN_STATUS.FEASIBLE:
#         updated_feas, insertion_flag[ci] = self.updateFeasWithPoint(eval_point=cp,
# keep_all_points=keep_all_points) or updated_feas


#       # // Do separate loop on evalPointList
#     # // Second loop update the bestInfeasible.
#     # // Use the flag oneFeasEvalFullSuccess.
#     # // If the flag is true hmax will not change.
# A point improving the best infeasible should not replace it.

#       if not cp.evaluated or cp.status == DESIGN_STATUS.ERROR:
#         continue
#       updated_inf, insertion_flag[ci] = self.updateInfWithPoint(eval_point=
# cp, keep_all_points=keep_all_points)  or updated_feas

#       ci += 1

#     updated = updated or updated_feas or updated_inf

#     if updated:
#       self.setN()
#       self.updateCurrentIncumbents()


#     return updated, updated_feas, updated_inf, insertion_flag

#   def updateCurrentIncumbents(self):
#     self.updateCurrentIncumbentFeas()
#     self.updateCurrentIncumbentInf()

#   def setHMax(self, h_max=None):
#     old_h_max = self._h_max
#     self._h_max = h_max
#     self.checkHMax()
#     if h_max < old_h_max and h_max > 0:
#       self.updateXInfAndFilterInfAfterHMaxSet()
#     self.updateCurrentIncumbentInf()

#   def updateXInfAndFilterInfAfterHMaxSet(self):
#     """ """
#     if len(self._xInf) == 0:
#       return

#     current_ind = 0

#     is_in_x_inf = [True] * len(self._xInf)
#     for x_inf in self._xInf:
#       h  = x_inf.h
#       if h > self._h_max:
#         is_in_x_inf[current_ind] = False
#       current_ind += 1

#     current_ind = 0
#     for _ in self._xInf:
#       if not is_in_x_inf[current_ind]:
#         self._xInf.pop(current_ind)
#       current_ind += 1

#     current_ind = 0
#     is_in_x_filter_inf = [True] *len(self._xFilterInf)

#     for  x_filter_inf in self._xFilterInf:
#       h = x_filter_inf.h
#       if h >self._h_max:
#         is_in_x_filter_inf[current_ind] = False
#       current_ind += 1

#     current_ind = 0
#     for _ in self._xFilterInf:
#       if not is_in_x_filter_inf[current_ind]:
#         self._xFilterInf.pop(current_ind)
#       current_ind += 1

#     self._xFilterInf = self.non_dominated_sort(self._xFilterInf)

#     # // And reinsert potential infeasible non dominated points into the set of infeasible
#     # // solutions.
#     current_ind = 0
#     is_in_x_inf = [False] * len(self._xFilterInf)
#     for eval_point in self._xFilterInf:
#       if len(self._xInf) > 0 and
# self.findEvalPoint(self._xFilterInf, eval_point)[1] == self._xInf[-1]:
#         current_ind_tmp = 0
#         insert = True
#         for eval_point_inf in self._xFilterInf:
#           if current_ind_tmp != current_ind:
#             comp_flag = eval_point.__compare__(eval_point_inf, True)
#             if comp_flag == COMPARE_TYPE.DOMINATED:
#               insert = False
#               break
#             elif comp_flag == COMPARE_TYPE.DOMINATING:
#               is_in_x_inf[current_ind_tmp] = False
#           current_ind_tmp += 1
#         is_in_x_inf[current_ind] = insert
#       current_ind += 1

#     for i in range(len(is_in_x_inf)):
#       if is_in_x_inf[i]:
#         self._xInf.append(self._xFilterInf[i])

#     self._xInf = self.non_dominated_sort(self._xInf)

#   def clearXFeas(self):
#     self._xFeas.clear()

#     self.updateCurrentIncumbents()

#   def clearXInf(self):
#     self._xInf.clear()
#     self._xFilterInf.clear()
#     # Update the current incumbent inf. Only the infeasible one
# depends on XInf (not the case for the feasible one).
#     self.updateCurrentIncumbentInf()

#   def computeSuccessType(self, eval1: CandidatePoint=None, eval2:
# CandidatePoint=None, h_max: int=np.inf):
#     """ """
#     success: SUCCESS_TYPES = SUCCESS_TYPES.US
#     if eval1 is not None:
#       if eval2 is None:
#         h = eval1.h
#         if h > h_max or h == np.inf:
#           success = SUCCESS_TYPES.US
#         else:
#           if eval1.status == DESIGN_STATUS.FEASIBLE:
#             success = SUCCESS_TYPES.FS
#           else:
#             success = SUCCESS_TYPES.PS
#       else:
#         if eval1.__compare__(eval2, self._h_max) == COMPARE_TYPE.DOMINATING:
#           # // Whether eval1 and eval2 are both feasible, or both
#           # // infeasible, dominance means FULL_SUCCESS.
#           success = SUCCESS_TYPES.FS
#         elif eval1.status == DESIGN_STATUS.FEASIBLE and eval2.status == DESIGN_STATUS.FEASIBLE:
#           success = SUCCESS_TYPES.US
#         elif eval1.status != DESIGN_STATUS.FEASIBLE and eval2.status != DESIGN_STATUS.FEASIBLE:
#           if eval1.h <= h_max and eval1.h < eval2.h and eval1.f > eval2.f:
#             success = SUCCESS_TYPES.PS
#         else:
#           success = SUCCESS_TYPES.US
#     return success


#   def defaultComputeSuccessType(self, eval_point1: CandidatePoint,
# eval_point2: CandidatePoint, h_max: float):
#     success: SUCCESS_TYPES = SUCCESS_TYPES.US
#     if eval_point1 and eval_point2:
#       h = eval_point1.h
#       if h > h_max or h == np.inf:
#         # // Even if evalPoint2 is NULL, this case is still
#         # // not a success.
#         success = SUCCESS_TYPES.US
#       elif eval_point1.status == DESIGN_STATUS.FEASIBLE:
#         success = SUCCESS_TYPES.FS
#       else:
#         success = self.defaultComputeSuccessType(eval_point1, eval_point2, h_max)
#     return success

#   def getSuccessTypeOfPoints(self, x_feas: CandidatePoint = None, x_inf: CandidatePoint = None):
#     success_type = SUCCESS_TYPES.US
#     success_type2 = SUCCESS_TYPES.US

#     if self._currentIncumbentFeas != None or self._currentIncumbentInf != None:
#       if not self._currentIncumbentFeas:
#         success_type = self.defaultComputeSuccessType(x_feas,
# self._currentIncumbentFeas, self._h_max)
#       if not self._currentIncumbentInf:
#         success_type = self.defaultComputeSuccessType(x_inf,
# self._currentIncumbentInf, self._h_max)
#       if success_type2.value > success_type.value:
#         success_type = success_type2

#     return success_type

#   def checkXFeasIsFeas(self, x_feas: CandidatePoint = None, eval_type: DESIGN_STATUS = None):
#     if x_feas.evaluated and x_feas.status != DESIGN_STATUS.ERROR:
#       h = x_feas.h
#       if h != 0:
#         raise IOError("Error: DMultiMadsBarrier: xFeas' h value must be 0.0")
#       if x_feas.fs.size != self._nobj:
#         raise IOError("Error: DMultiMadsBarrier: xFeas' F must be of size")


#   def getMeshMaxFrameSize(self, pt:CandidatePoint):
#     max_real_val = -1.0
#     max_integer_val = -1.0

#     # Detect if mesh is sub dimension and pt are in full dimension.
#     mesh_is_in_subdimension = False
#     mesh = pt.mesh
#     if pt.mesh._n < pt._n:
#       mesh_is_in_subdimension = True

#     shift = 0
#     for i in range(pt._n):
#       # Do not use access the frame size for fixed variables.
#       if mesh_is_in_subdimension and self._fixedVariables.defined[i]:
#         shift += 1
#       if self._bbInputsType[i] == VAR_TYPE.REAL:
#         max_real_val = max(max_real_val, mesh.getDeltaFrameSize(i-shift))
#       elif self._bbInputsType[i] == VAR_TYPE.INTEGER:
#         max_integer_val = max(max_integer_val, mesh.getDeltaFrameSize(i-shift))
#     if max_real_val > 0.0:
#       return max_real_val # Some values are real: get norm inf on these values only.
#     elif max_integer_val > 0.0:
#       return max_integer_val # No real value but
# some integer values: get norm inf on these values only
#     else:
#       return 1.0 # Only binary variables: any elements of the iterate list can be chosen


#   def updateCurrentIncumbentFeas(self):
#     if len(self._xFeas) == 0:
#       self._currentIncumbentFeas = None
#       return

#     if len(self._xFeas) == 1:
#       self._currentIncumbentFeas = self._xFeas[0]
#       return

#     max_frame_size_feas_elts = -1.0

#     # Set max frame size of all elements
#     for xf in self._xFeas:
#       max_frame_size_feas_elts = max(self.getMeshMaxFrameSize(xf), max_frame_size_feas_elts)

#     # Select candidates
#     can_be_frame_center: List[bool] = [False] * len(self._xFeas)
#     nb_selected_candidates = 0

#     # see article DMultiMads Algorithm 4.
#     for i in range(len(self._xFeas)):
#       max_frame_size_elt = self.getMeshMaxFrameSize(self._xFeas[i])

#       if (10**(-float(self._incumbentSelectionParam)) *
# max_frame_size_feas_elts) <= max_frame_size_elt:
#         can_be_frame_center[i] = True
#         nb_selected_candidates += 1

#     # Only one point in the barrier.
#     if (nb_selected_candidates == 1):
#       # for it in range(len(can_be_frame_center)):
#       #   if can_be_frame_center[it]:
#       #     break
#       # if it == len(can_be_frame_center):
#       #   raise IOError("Error: DMultiMadsBarrier, should not reach this condition")
#       # else:
#       #   selected_ind = it
#       #   self._currentIncumbentFeas = self._xFeas[selected_ind]
#       try:
#         selected_ind = can_be_frame_center.index(True)
#         self._currentIncumbentFeas = self._xFeas[selected_ind]
#       except ValueError:
#         raise IOError("Error: DMultiMadsBarrier, should not reach this condition")
#     # Only two points in the barrier.
#     elif ((nb_selected_candidates == 2) and (len(self._xFeas) == 2)):
#       eval1 = self._xFeas[0]
#       eval2 = self._xFeas[1]

#       objv1 = eval1.f
#       objv2 = eval2.f

#       if max(np.abs(objv1.coordinates)) > max(np.abs(objv2.coordinates)):
#         self._currentIncumbentFeas = self._xFeas[0]
#       else:
#         self._currentIncumbentFeas = self._xFeas[1]
#       self._incumbentSelectionParam = self._incumbentSelectionParam#2 *
# self._incumbentSelectionParam + 2
#     # More than three points in the barrier.
#     else:
#       # First case: biobjective optimization. Points are already ranked by lexicographic order.
#       if self._nobj:
#         current_best_ind = 0
#         max_gap = -1.0
#         current_gap: float
#         for obj in range(self._nobj):
#           # Get extreme values value according to one objective
#           fmin: float = self._xFeas[0].f[obj]
#           fmax: float = self._xFeas[len(self._xFeas)-1].f[obj]
#           # In this case, it means all elements of _xFeas are equal (return the first one)
#           if fmin == fmax:
#             break

#           # Intermediate points
#           for i in range(1, len(self._xFeas)-1):
#             current_gap = self._xFeas[i+1].f[obj]-self._xFeas[i-1].f[obj]
#             self._xFeas[i-1].f[obj]
#             current_gap /= (fmax-fmin)
#             if (can_be_frame_center[i] and current_gap >= max_gap):
#               max_gap = current_gap
#               current_best_ind = i


#           # Extreme points
#           current_gap = 2 * (self._xFeas[len(self._xFeas)-1]).f[obj] -
# (self._xFeas[len(self._xFeas)-2]).f[obj]
#           current_gap /= (fmax - fmin)
#           if can_be_frame_center[len(self._xFeas)-1] and current_gap > max_gap:
#             max_gap = current_gap
#             current_best_ind = len(self._xFeas)-1

#           current_gap = 2 * (self._xFeas[1]).f[obj] - (self._xFeas[0]).f[obj]
#           current_gap /= (fmax -fmin)

#           if can_be_frame_center[0] and current_gap > max_gap:
#             max_gap = current_gap
#             current_best_ind = 0
#         self._currentIncumbentFeas = self._xFeas[current_best_ind]
#         self._incumbentSelectionParam = self._incumbentSelectionParam #2 *
# self._incumbentSelectionParam + 2

#       # // More than 2 objectives
#       else:
#         tmp_x_feas_p_ind: List[Tuple[CandidatePoint, int]] =
# [(CandidatePoint(), 0)]*len(self._xFeas)
#         for i in range(len(tmp_x_feas_p_ind)):
#           tmp_x_feas_p_ind[i] = (self._xFeas[i], i)
#         current_best_ind = 0
#         max_gap = -1.0
#         current_gap: float

#         for obj in range(self._nobj):
#           # Sort elements of tmpXFeasPInd according to objective obj (in ascending order)
#           tmp_x_feas_p_ind = sorted(tmp_x_feas_p_ind, key=lambda x: x[0].f[obj])

#           # Get extreme values value according to one objective
#           fmin = tmp_x_feas_p_ind[0][0].f[obj]
#           fmax = tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-1][0].f[obj]

#           # Can happen for example when we have several minima or for more than three objectives
#           if fmin == fmax:
#             fmin = 0.
#             fmax = 1.

#           # Intermediate points
#           for i in range(1, len(tmp_x_feas_p_ind)-1):
#             current_gap = tmp_x_feas_p_ind[i+1][0].f[obj]-tmp_x_feas_p_ind[i-1][0].f[obj]
#             current_gap /= (fmax - fmin)
#             if can_be_frame_center[tmp_x_feas_p_ind[i][1]] and current_gap >= max_gap:
#               max_gap = current_gap
#               current_best_ind = tmp_x_feas_p_ind[i][1]

#           # Extreme points
#           current_gap = 2*(tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-1][0].f[obj]) -
# tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-2][0].f[obj]
#           current_gap /= (fmax - fmin)

#           if (can_be_frame_center[tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-1][1]]
# and current_gap > max_gap):
#             max_gap = current_gap
#             current_best_ind = tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-1][1]

#           current_gap = 2 * tmp_x_feas_p_ind[1][0].f[obj] - tmp_x_feas_p_ind[0][0].f[obj]
#           current_gap /= (fmax -fmin)

#           if (can_be_frame_center[tmp_x_feas_p_ind[0][1]] and current_gap > max_gap):
#             max_gap = current_gap
#             current_best_ind = tmp_x_feas_p_ind[0][1]
#         self._currentIncumbentFeas = self._xFeas[current_best_ind]

#   def updateCurrentIncumbentInf(self):
#     if len(self._xInf) == 0:
#       self._currentIncumbentInf = None
#       return

#     if len(self._xInf) == 1:
#       self._currentIncumbentInf = self._xInf[0]
#       return

#     max_frame_size_infeas_elts = -1.0

#     # Set max frame size of all elements
#     for xf in self._xInf:
#       max_frame_size_infeas_elts = max(self.getMeshMaxFrameSize(xf), max_frame_size_infeas_elts)

#     # Select candidates
#     # Select candidates
#     can_be_frame_center: List[bool] = [False] * len(self._xInf)
#     nb_selected_candidates = 0

#     # see article DMultiMads Algorithm 4.
#     for i in range(len(self._xInf)):
#       max_frame_size_elt = self.getMeshMaxFrameSize(self._xInf[i])

#       if (10**(-float(self._incumbentSelectionParam)) *
# max_frame_size_infeas_elts) <= max_frame_size_elt:
#         can_be_frame_center[i] = True
#         nb_selected_candidates += 1

#     # Only one point in the barrier.
#     if (nb_selected_candidates == 1):
#       # for it in range(len(can_be_frame_center)):
#       #   if can_be_frame_center[it]:
#       #     break
#       # if it == len(can_be_frame_center):
#       #   raise IOError("Error: DMultiMadsBarrier, should not reach this condition")
#       # else:
#       #   selected_ind = it
#       #   self._currentIncumbentInf = self._xInf[selected_ind]
#       try:
#         selected_ind = can_be_frame_center.index(True)
#         self._currentIncumbentInf = self._xInf[selected_ind]
#       except ValueError:
#         raise IOError("Error: DMultiMadsBarrier, should not reach this condition")
#     # Only two points in the barrier.
#     elif ((nb_selected_candidates == 2) and (len(self._xInf) == 2)):
#       eval1 = self._xInf[0]
#       eval2 = self._xInf[1]

#       objv1 = eval1.f
#       objv2 = eval2.f

#       if max(np.abs(objv1.coordinates)) > max(np.abs(objv2.coordinates)):
#         self._currentIncumbentInf = self._xInf[0]
#       else:
#         self._currentIncumbentInf = self._xInf[1]
#       self._incumbentSelectionParam = self._incumbentSelectionParam#2 * \
# self._incumbentSelectionParam + 2
#     # More than three points in the barrier.
#     else:
#       # First case: biobjective optimization. Points are already ranked by lexicographic order.
#       if self._nobj:
#         current_best_ind = 0
#         max_gap = -1.0
#         current_gap: float
#         for obj in range(self._nobj):
#           # Get extreme values value according to one objective
#           fmin: float = self._xInf[0].f[obj]
#           fmax: float = self._xInf[len(self._xInf)-1].f[obj]
#           # In this case, it means all elements of _xInf are equal (return the first one)
#           if fmin == fmax:
#             break

#           # Intermediate points
#           for i in range(1, len(self._xInf)-1):
#             current_gap = self._xInf[i+1].f[obj]-self._xInf[i-1].f[obj]
#             self._xInf[i-1].f[obj]
#             current_gap /= (fmax-fmin)
#             if (can_be_frame_center[i] and current_gap >= max_gap):
#               max_gap = current_gap
#               current_best_ind = i


#           # Extreme points
#           current_gap = 2 * (self._xInf[len(self._xInf)-1]).f[obj] -
# (self._xInf[len(self._xInf)-2]).f[obj]
#           current_gap /= (fmax - fmin)
#           if can_be_frame_center[len(self._xInf)-1] and current_gap > max_gap:
#             max_gap = current_gap
#             current_best_ind = len(self._xInf)-1

#           current_gap = 2 * (self._xInf[1]).f[obj] - (self._xInf[0]).f[obj]
#           current_gap /= (fmax -fmin)

#           if can_be_frame_center[0] and current_gap > max_gap:
#             max_gap = current_gap
#             current_best_ind = 0
#         self._currentIncumbentInf = self._xInf[current_best_ind]
#         self._incumbentSelectionParam = self._incumbentSelectionParam #2 *
# self._incumbentSelectionParam + 2

#       # // More than 2 objectives
#       else:
#         tmp_x_infeas_p_ind: List[Tuple[CandidatePoint, int]] =
# [(CandidatePoint(), 0)]*len(self._xInf)
#         for i in range(len(tmp_x_infeas_p_ind)):
#           tmp_x_infeas_p_ind[i] = (self._xInf[i], i)
#         current_best_ind = 0
#         max_gap = -1.0
#         current_gap: float

#         for obj in range(self._nobj):
#           # Sort elements of tmpXFeasPInd according to objective obj (in ascending order)
#           tmp_x_infeas_p_ind = sorted(tmp_x_infeas_p_ind, key=lambda x: x[0].f[obj])

#           # Get extreme values value according to one objective
#           fmin = tmp_x_infeas_p_ind[0][0].f[obj]
#           fmax = tmp_x_infeas_p_ind[len(tmp_x_infeas_p_ind)-1][0].f[obj]

#           # Can happen for example when we have several minima or for more than three objectives
#           if fmin == fmax:
#             fmin = 0.
#             fmax = 1.

#           # Intermediate points
#           for i in range(1, len(tmp_x_infeas_p_ind)-1):
#             current_gap = tmp_x_infeas_p_ind[i+1][0].f[obj]-tmp_x_infeas_p_ind[i-1][0].f[obj]
#             current_gap /= (fmax - fmin)
#             if can_be_frame_center[tmp_x_infeas_p_ind[i][1]] and current_gap >= max_gap:
#               max_gap = current_gap
#               current_best_ind = tmp_x_infeas_p_ind[i][1]

#           # Extreme points
#           current_gap = 2*(tmp_x_infeas_p_ind[len(tmp_x_infeas_p_ind)-1][0].f[obj])
# - tmp_x_infeas_p_ind[len(tmp_x_infeas_p_ind)-2][0].f[obj]
#           current_gap /= (fmax - fmin)

#           if (can_be_frame_center[tmp_x_infeas_p_ind[len(tmp_x_infeas_p_ind)-1][1]]
# and current_gap > max_gap):
#             max_gap = current_gap
#             current_best_ind = tmp_x_infeas_p_ind[len(tmp_x_infeas_p_ind)-1][1]

#           current_gap = 2 * tmp_x_infeas_p_ind[1][0].f[obj] - tmp_x_infeas_p_ind[0][0].f[obj]
#           current_gap /= (fmax -fmin)

#           if (can_be_frame_center[tmp_x_infeas_p_ind[0][1]] and current_gap > max_gap):
#             max_gap = current_gap
#             current_best_ind = tmp_x_infeas_p_ind[0][1]
#         self._currentIncumbentInf = self._xInf[current_best_ind]


#   def getXInfMinH(self):
#     ind_x_inf_min_h = 0
#     h_min_val = np.inf

#     for i in range(len(self._xInf)):
#       my_eval = self._xInf[i]
#       h = my_eval.h

#       # // By definition, all elements of _xInf or _xFilterInf have a well-defined
#       # // h value. So, no need to check.

#       if h <h_min_val:
#         h_min_val = h
#         ind_x_inf_min_h = i
#     return self._xInf[ind_x_inf_min_h]

#   def getFirstXIncInfNoXFeas(self):
#     """ """
#     x_inf = None
#     if len(self._xFilterInf) == 0:
#       return x_inf

#     # Select candidates
#     min_frame_size_inf_elts: float = self.getMeshMaxFrameSize(self.getXInfMinH())
#     can_be_frame_center: List[bool] = [False] * len(self._xInf)
#     nb_selected_candidates = 0

#     for i in range(len(self._xInf)):
#       max_frame_size_elt = self.getMeshMaxFrameSize(self._xInf[i])
#       if min_frame_size_inf_elts <= max_frame_size_elt:
#         can_be_frame_center[i] = True
#         nb_selected_candidates += 1

#     # The selection must always work
#     if nb_selected_candidates == 0:
#       x_inf = self._xInf[0]
#     elif nb_selected_candidates == 1:
#       for it in range(len(can_be_frame_center)):
#         if can_be_frame_center[it]:
#           break
#       if it == len(can_be_frame_center):
#         raise IOError("Error: DMultiMadsBarrier, should not reach this condition")
#       else:
#         selected_ind = it
#         x_inf = self._xInf[selected_ind]
#     elif ((nb_selected_candidates == 2) and (len(self._xInf) == 2)):
#       eval1 = self._xInf[0]
#       eval2 = self._xInf[1]

#       objv1 = eval1.f
#       objv2 = eval2.f

#       if max(np.abs(objv1.coordinates)) > max(np.abs(objv2.coordinates)):
#         x_inf = self._xInf[0]
#       else:
#         x_inf = self._xInf[1]
#     else:
#       if self._nobj == 2:
#         current_best_ind = 0
#         max_gap = -1.
#         current_gap: float

#         for obj in range(self._nobj):
#           # Get extreme values value according to one objective
#           fmin: float = self._xInf[0].f[obj]
#           fmax: float = self._xInf[len(self._xInf)-1].f[obj]

#           # In this case, it means all elements of _xFeas are equal (return the first one)
#           if fmin == fmax:
#             break

#           # Intermediate points
#           for i in range(1, self._xInf-1):
#             current_gap = self._xInf[i+1].f[obj]-self._xInf[i-1].f[obj]
#             self._xInf[i-1].f[obj]
#             current_gap /= (fmax-fmin)
#             if (can_be_frame_center[i] and current_gap >= max_gap):
#               max_gap = current_gap
#               current_best_ind = i

#           # Extreme points
#           current_gap = 2 * (self._xInf[len(self._xInf)-1]).f[obj] -
# (self._xInf[len(self._xInf)-2]).f[obj]
#           current_gap /= (fmax - fmin)
#           if can_be_frame_center[len(self._xInf)-1] and current_gap > max_gap:
#             max_gap = current_gap
#             current_best_ind = len(self._xInf)-1

#           current_gap = 2 * (self._xInf[1]).f[obj] - (self._xInf[0]).f[obj]
#           current_gap /= (fmax -fmin)

#           if can_be_frame_center[0] and current_gap > max_gap:
#             max_gap = current_gap
#             current_best_ind = 0
#         x_inf = self._xInf[current_best_ind]
#       # // More than 2 objectives
#       else:
#         tmp_x_inf_p_ind: List[Tuple[CandidatePoint, int]] =
# [(CandidatePoint(), 0)]*len(self._xInf)
#         for i in range(len(tmp_x_inf_p_ind)):
#           tmp_x_inf_p_ind[i] = (self._xInf[i], i)
#         current_best_ind = 0
#         max_gap = -1.0
#         current_gap: float

#         for obj in range(self._nobj):
#           # Sort elements of tmpXFeasPInd according to objective obj (in ascending order)
#           tmp_x_inf_p_ind = sorted(tmp_x_inf_p_ind, key=lambda x: x[0].f[obj])

#           # Get extreme values value according to one objective
#           fmin = tmp_x_inf_p_ind[0][0].f[obj]
#           fmax = tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-1][0].f[obj]

#           # Can happen for exemple when we have several minima or for more than three objectives
#           if fmin == fmax:
#             fmin = 0.
#             fmax = 1.

#           # Intermediate points
#           for i in range(1, len(tmp_x_inf_p_ind)-1):
#             current_gap = tmp_x_inf_p_ind[i+1][0].f[obj]-tmp_x_inf_p_ind[i-1][0].f[obj]
#             current_gap /= (fmax - fmin)
#             if can_be_frame_center[tmp_x_inf_p_ind[i][1]] and current_gap >= max_gap:
#               max_gap = current_gap
#               current_best_ind = tmp_x_inf_p_ind[i][1]

#           # Extreme points
#           current_gap = 2*(tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-1][0].f[obj]) -
# tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-2][0].f[obj]
#           current_gap /= (fmax - fmin)

#           if (can_be_frame_center[tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-1][1]]
# and current_gap > max_gap):
#             max_gap = current_gap
#             current_best_ind = tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-1][1]

#           current_gap = 2 * tmp_x_inf_p_ind[1][0].f[obj] - tmp_x_inf_p_ind[0][0].f[obj]
#           current_gap /= (fmax -fmin)

#           if (can_be_frame_center[tmp_x_inf_p_ind[0][1]] and current_gap > max_gap):
#             max_gap = current_gap
#             current_best_ind = tmp_x_inf_p_ind[0][1]
#         x_inf = self._xInf[current_best_ind]

#     return x_inf

#   def updateInfWithPoint(self, eval_point: CandidatePoint = None, keep_all_points: bool = None):
#     updated = False
#     insertion_flag: INSERTION_FLAG = None
#     insertion_best_flag: INSERTION_FLAG = INSERTION_FLAG.IS_DOMINATED
#     prev_nb_best_inf_pts = len(self._xInf)

#     if eval_point.evaluated and eval_point.status != DESIGN_STATUS.FEASIBLE:
#       s: str
#       h = eval_point.h

#       if h > self._h_max:
#         insertion_flag = INSERTION_FLAG.REJECTED
#         return False, insertion_best_flag
#       else:
#         self._h_max = h
#         # self.setHMax(h)

#       if self._xInf is None:
#         self._xInf = []

#       if self._xFilterInf is None:
#         self._xFilterInf = []
#       # New infeasible point is the first point
#       if len(self._xInf) <= 0:
#         self._xInf.append(eval_point)
#         self._xFilterInf.append(eval_point)
#         self._currentIncumbentInf = self._xInf[0]
#         updated = True
#         insertion_flag = INSERTION_FLAG.IMPROVES
#       # Insertion into the two sets of infeasible non-dominated points
#       else:
#         # Try to insert into _x_inf_filter
#         insert = True
#         insertion_flag = INSERTION_FLAG.IMPROVES
#         is_in_x_inf_filter: List[bool] = [True] * len(self._xFilterInf)
#         current_ind = 0
#         for x_filter_inf in self._xFilterInf:
#           comp_flag = eval_point.__compare__(x_filter_inf)
#           if comp_flag == COMPARE_TYPE.DOMINATED:
#             insert = False
#             break
#           elif comp_flag == COMPARE_TYPE.DOMINATING:
#             insertion_best_flag = INSERTION_FLAG.DOMINATES
#             updated = True
#             is_in_x_inf_filter[current_ind] = False
#           elif comp_flag == COMPARE_TYPE.EQUAL:
#             if (not keep_all_points):
#               insert = False
#               break
#             if self.findEvalPoint(self._xFilterInf, eval_point)[0]:
#               insert = False
#             else:
#               updated = True
#               break
#           current_ind += 1

#         if self.extends_PF(eval_point, False):
#           insertion_flag = INSERTION_FLAG.EXTENDS

#         if insert:
#           # Remove all dominated elements of _xInfFilter
#           current_ind = 0
#           indices_to_remove = []
#           for i in range(len(self._xFilterInf)):
#             if not is_in_x_inf_filter[i]:
#               indices_to_remove.append(i)
#             current_ind += 1

#           self._xFilterInf.append(eval_point)

#           for index in sorted(indices_to_remove, reverse=True):
#             del self._xFilterInf[index]

#           self._xFilterInf = self.non_dominated_sort(self._xFilterInf)

#           insert = True
#           current_ind = 0
#           is_in_x_inf = [True] * len(self._xInf)

#           for x_inf in self._xInf:
#             comp_flag = eval_point.__compare__(x_inf)
#             if comp_flag == COMPARE_TYPE.DOMINATED:
#               insert = False
#               break
#             elif comp_flag == COMPARE_TYPE.DOMINATING or
# eval_point.__compare__(x_inf) == COMPARE_TYPE.DOMINATING:
#               updated = True
#               is_in_x_inf[current_ind] = False
#             current_ind += 1

#           if insert:
#             indices_to_remove = []
#             for i in range(len(self._xInf)):
#               if not is_in_x_inf[i]:
#                 indices_to_remove.append(i)
#             updated = True
#             self._xInf.append(eval_point)

#             for index in sorted(indices_to_remove, reverse=True):
#               del self._xInf[index]

#             self._xInf = self.non_dominated_sort(self._xInf)

#           if insertion_flag == INSERTION_FLAG.EXTENDS:
#             return updated, insertion_flag

#           if insertion_best_flag == INSERTION_FLAG.IS_DOMINATED:
#             return updated, INSERTION_FLAG.NO_IMPROVEMENT
#           else:
#             if len(self._xInf) <= prev_nb_best_inf_pts:
#               return updated, INSERTION_FLAG.DOMINATES
#             else:
#               return updated, insertion_best_flag

#     return updated, insertion_best_flag


#   def non_dominated_sort(self, points: List[CandidatePoint] = None):
#     """ Perform biobjective nondominated sorting """
#     fronts = [[]]  # List to store different fronts
#     dominated_count = [0] * len(points)  # Array to count number of points dominating each point

#     for i, p in enumerate(points):
#       for j, q in enumerate(points):
#         if i != j and p.__compare__(q) == COMPARE_TYPE.DOMINATED:
#           dominated_count[i] += 1

#       if dominated_count[i] == 0:
#         fronts[0].append(p)

#     # Sort each front lexicographically
#     for front in fronts:
#       front.sort()

#     # Flatten the fronts into a single list
#     sorted_points = [point for front in fronts for point in front]

#     return sorted_points

#   def updateFeasWithPoint(self, eval_point: CandidatePoint = None, keep_all_points: bool = None):
#     updated = False
#     insertion_flag: INSERTION_FLAG = None
#     if not eval_point.is_feasible():
#       raise ValueError("Trying to insert an infeasible element into the set of feasible points.")
#     if eval_point.evaluated and eval_point.status == DESIGN_STATUS.FEASIBLE:
#       if eval_point.fs.size != self._nobj:
#         raise IOError(f"Barrier update: number of objectives is equal to {self._nobj}.
# Trying to add this point with number of objectives {eval_point.fs.size}")

#       if self._xFeas is None:
#         self._xFeas = []

#       if len(self._xFeas) == 0:
#         self._xFeas.append(eval_point)
#         updated = True
#         insertion_flag = INSERTION_FLAG.IMPROVES
#         self._currentIncumbentFeas = self._xFeas[0]
#       else:
#         insert = True
#         insertion_flag = INSERTION_FLAG.IMPROVES
#         keep_in_x_feas = [True] * len(self._xFeas)
#         current_ind = 0
#         for xf in self._xFeas:
#           comp_flag: COMPARE_TYPE = eval_point.__compare__(xf)
#           if comp_flag == COMPARE_TYPE.DOMINATED:
#             insert = False
#             insertion_flag = INSERTION_FLAG.IS_DOMINATED
#             break
#           elif comp_flag == COMPARE_TYPE.DOMINATING:
#             updated = True
#             keep_in_x_feas[current_ind] = False
#             INSERTION_FLAG.DOMINATES
#           elif comp_flag == COMPARE_TYPE.EQUAL or comp_flag == COMPARE_TYPE.DOMINATED:
#             insertion_flag = INSERTION_FLAG.IS_DOMINATED
#             if not keep_all_points:
#               insert = False
#               break

#             if self.findEvalPoint(self._xFeas, eval_point)[0]:
#               insert = False
#             else:
#               updated = True
#             break
#           current_ind += 1
#         if insert:
#           current_ind = 0
#           for cp in self._xFeas:
#             if cp.__compare__(eval_point) == COMPARE_TYPE.DOMINATED:
#               self._xFeas.pop(current_ind)
#             current_ind += 1
#           updated = True
#           my_dir = copy.deepcopy(eval_point.direction)
#           if my_dir is not None:
#             eval_point.mesh.enlargeDeltaFrameSize(direction=my_dir)

#           self._xFeas.append(eval_point)

#           # Sort according to lexicographic order.
#           self._xFeas = self.non_dominated_sort(self._xFeas)

#         if self.extends_PF(eval_point, True):
#           insertion_flag = INSERTION_FLAG.EXTENDS

#     return updated, insertion_flag

#   def extends_PF(self, cp: CandidatePoint, is_feasible: bool):
#     temp_ndp: List[CandidatePoint] = []
#     if is_feasible:
#       temp_ndp = copy.deepcopy(self._xFeas)
#     else:
#       temp_ndp = copy.deepcopy(self._xInf)

#     if len(temp_ndp) <= 0:
#       return False

#     ideal_v = [np.inf] * self._nobj
#     for elt in temp_ndp:
#       ideal_v = [min(elt.f[i], ideal_v[i]) for i in range(self._nobj)] #min(elt.fobj, ideal_v)

#     return any([cp.fobj[i] < ideal_v[i] for i in range(self._nobj)])
