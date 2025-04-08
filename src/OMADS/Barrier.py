"""
# ------------------------------------------------------------------------------------#
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
from typing import Protocol
from typing import List, Optional
import numpy as np
from ._globals import COMPARE_TYPE, SUCCESS_TYPES, DType, DESIGN_STATUS
from .candidate_point import CandidatePoint
from .point import Point
from .cache import Cache
from .mesh import Mesh


@dataclass
class BarrierData(Protocol):
  """
  A data class protocol for constraints violation barrier

  :param Protocol: _description_
  :type Protocol: _type_
  """
  _x_feas: Optional[List[CandidatePoint]] = None
  _x_inf: Optional[List[CandidatePoint]] = None

  _x_inc_feas: Optional[List[CandidatePoint]] = None
  _x_inc_inf: Optional[List[CandidatePoint]] = None

  _x_filter_inf: Optional[List[CandidatePoint]] = None
  _ref_best_feas: Optional[CandidatePoint] = None

  _ref_best_inf: Optional[CandidatePoint] = None
  _meshes: Optional[Mesh] = None

  _dtype: Optional[DType] = None

  def init(self, eval_point_list: Optional[List[Point]] = None):
    """
    Initialize an instance of BarrierData.

    :param eval_point_list: List of points, defaults to None
    :type eval_point_list: Optional[List[Point]], optional
    """
    ...

  def get_all_feas_points(self) -> List[CandidatePoint]:
    """
    Getter for all feasible points in the barrier list of points

    :return: List of candidate points
    :rtype: List[CandidatePoint]
    """
    ...

  def clone(self):
    """
    Clone from another BarrierData instant.
    """
    ...

  def get_current_incumbent_feas(self) -> CandidatePoint:
    """
    Getter for the incumbent infeasible candidate solution.
    """
    ...

  def get_all_x_inc_feas(self) -> List[CandidatePoint]:
    """
    Getter for all nondominated feasible incumbents.

    :return: List of candidate points.
    :rtype: List[CandidatePoint]
    """
    ...

  def get_ref_best_feas(self) -> CandidatePoint:
    """
    Getter for the best reference feasible candidate solution.

    :return: A candidate solution point.
    :rtype: CandidatePoint
    """
    ...

  def set_ref_best_feas(self, x: CandidatePoint = None):
    """Setter for the best reference feasible candidate solution.

    :param x: best candidate solution, defaults to None
    :type x: CandidatePoint, optional
    """
    ...

  def update_ref_bests(self):
    """
    Update reference incumbents.
    """
    ...

  def nb_x_feas(self) -> int:
    """
    Get number of feasible solutions.

    :return: number
    :rtype: int
    """
    ...

  def clear_x_feas(self):
    """
    Clear the set of feasible solutions in the barrier instant.
    """
    ...

  def get_all_x_inf(self) -> List[CandidatePoint]:
    """
    Getter for all infeasible points in the barrier list of points

    :return: List of candidate points
    :rtype: List[CandidatePoint]
    """
    ...

  def get_all_x_inc_inf(self) -> List[CandidatePoint]:
    """
    Get all nondominated infeasible incumbents.

    :return: List of candidate points.
    :rtype: List[CandidatePoint]
    """
    ...

  def get_current_incumbent_inf(self) -> CandidatePoint:
    """
    Get the current incumbent infeasible solution.

    :return: Candidate point.
    :rtype: CandidatePoint
    """
    ...

  def get_ref_best_inf(self) -> CandidatePoint:
    """Get the best infeasible reference point.

    :return: Candidate point.
    :rtype: CandidatePoint
    """
    ...

  def set_ref_best_inf(self, x: CandidatePoint = None):
    """Setter for the best reference infeasible candidate solution.

    :param x: best infeasible candidate solution, defaults to None
    :type x: CandidatePoint, optional
    """
    ...

  def nb_x_inf(self) -> int:
    """
    Get number of infeasible solutions.

    :return: number
    :rtype: int
    """
    ...

  def clear_x_inf(self):
    """
    Clear the set of infeasible solutions.
    """
    ...

  def get_all_points(self) -> List[CandidatePoint]:
    """
    Get all the points evaluated within the barrier class instant (dom/nondom, feasinfeas)

    :return: List of candidate solutions.
    :rtype: List[CandidatePoint]
    """
    ...

  def get_first_point(self) -> CandidatePoint:
    """
    Get the first point in the list of points.

    :return: _description_
    :rtype: CandidatePoint
    """
    ...

  def get_h_max(self):
    """
    Get the maximum value of evaluated constraints violation function.
    """
    ...

  def set_h_max(self, h_max: float):
    """
    Update the maximum value of constraints violation function.

    :param value: The value of evaluated constraints violation function.
    :type value: float
    """
    ...

  def get_success_type_of_points(self) -> SUCCESS_TYPES:
    """
    Get the success type given two candidate points in the args.

    :return: Success type.
    :rtype: SUCCESS_TYPES
    """
    ...

  def update_with_points(self):
    """
    Update the barrier evaluated points list with points.
    """
    ...

  def find_point(self, point: CandidatePoint):
    """
    Look for an evaluated point.

    :param point: point
    :type point: Point
    """
    ...

  def set_n(self):
    """
    Set the number of points.

    :param n: number
    :type n: int
    """
    ...

  def check_h_max(self):
    """
    Check on the updated constraints violation margine.
    """
    ...

  def check_cache(self):
    """
    Check the cache memory.
    """
    ...

  def find_eval_point(
          self, cps: List[CandidatePoint] = None, cp: CandidatePoint = None):
    """
    Look for an evaluated candidate point in a given list of candidate points.

    :param cps: List of candidate solutions, defaults to None
    :type cps: List[CandidatePoint], optional
    :param cp: The candidate solution to look for, defaults to None
    :type cp: CandidatePoint, optional
    """
    ...


@dataclass
class BarrierBase(BarrierData):
  """_summary_
  :param BarrierData: _The base template that can be used to create 
  class objects for different barrier types_
  :type BarrierData: _Dataclass_
  """
  _h_max: float = np.inf
  _n: int = 0

  def __init__(self, h_max: float = np.inf):
    self._h_max = h_max
    self._n = 0
    self._dtype = DType()
    self._x_inf = []
    self._x_feas = []
    self._x_inc_feas = []
    self._x_inc_inf = []
    self._meshes = []
    self._x_filter_inf = []

  def set_n(self):
    is_set: bool = False
    s: str

    for cp in self.get_all_points():
      if not is_set:
        self._n = cp.n_dimensions
        is_set = True
      elif cp.n_dimensions != self._n:
        s = f"Barrier has points of size {self._n} and of size {cp.n_dimensions}"
        raise IOError(s)
    if not is_set:
      raise IOError("Barrier could not set point size")

  def check_cache(self, cache: Cache = None):
    if cache is None:
      raise IOError("Cache must be instantiated before initializing Barrier.")

  def check_h_max(self):
    if self._h_max is None or self._h_max < 0:  # self._dtype.zero:
      raise IOError("Barrier: hMax must be positive.")

  def clear_x_feas(self):
    del self._x_feas

  def clear_x_inf(self):
    del self._x_inf
    del self._x_inc_inf

  def non_dominated_sort(
          self, points: List[CandidatePoint] = None, only_f: bool = False):
    """ Perform biobjective nondominated sorting """
    fronts = [[]]  # List to store different fronts
    # Array to count number of points dominating each point
    dominated_count = [0] * len(points)

    for i, p in enumerate(points):
      for j, q in enumerate(points):
        if i != j and p.__compare__(
                other=q, onlyfvalues=only_f) == COMPARE_TYPE.DOMINATED:
          dominated_count[i] += 1

      if dominated_count[i] == 0:
        fronts[0].append(p)

    # Sort each front lexicographically
    for front in fronts:
      front.sort()

    # Flatten the fronts into a single list
    sorted_points = [point for front in fronts for point in front]

    return sorted_points

  def get_filtered_best_inf_points(self):
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    if len(self._x_filter_inf) == 0:
      return []
    elif len(self._x_filter_inf) == 1:
      return self._x_filter_inf
    else:
      points: List[CandidatePoint] = []
      points = copy.deepcopy(self._x_filter_inf)
      h = []
      for x in self._x_filter_inf:
        h.append(x.h)
      hmin = max(min(h), 0.01)
      remove_index = []
      for i, _ in enumerate(h):
        if h[i] > 1.5 * hmin:
          remove_index.append(i)

      for index in sorted(remove_index, reverse=True):
        del points[index]

      return points

    # # Get sorted indices (original indices) of the sorted list
    # sorted_indices = [i for i, _ in sorted(enumerate(h), key=lambda x: x[1])]

  def get_all_points(self) -> List[CandidatePoint]:
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
    for cp in self._x_filter_inf:
      all_points.append(cp)

    all_points = self.non_dominated_sort(all_points)

    return all_points

  def get_first_point(self) -> Optional[CandidatePoint]:
    if self._x_inc_feas and len(self._x_inc_feas) > 0:
      return self._x_inc_feas[0]
    elif self._x_feas and len(self._x_feas) > 0:
      return self._x_feas[0]
    elif self._x_inc_inf and len(self._x_inc_inf) > 0:
      return self._x_inc_inf[0]
    elif self._x_inf and len(self._x_inf) > 0:
      return self._x_inf[0]
    else:
      return None

  def find_eval_point(
          self, cps: List[CandidatePoint] = None, cp: CandidatePoint = None):
    for p in cps:
      if p.signature == cp.signature:
        return True, p

    return False, cp

  def find_point(self, point: Point) -> bool:
    found: bool = False

    eval_point_list: List[CandidatePoint] = self.get_all_points()
    for cp in eval_point_list:
      if cp.n_dimensions != point.n:
        raise IOError("Error: Eval points have different dimensions")
      if point == cp.coordinates:
        found = True
        break

    return found

  def check_x_feas(
          self, x_feas: CandidatePoint = None):
    """_summary_

    :param x_feas: _description_, defaults to None
    :type x_feas: CandidatePoint, optional
    :type eval_type: EVAL_TYPE, optional
    """
    if x_feas.evaluated:
      self.check_x_feas_is_feas(x_feas=x_feas)

  def get_all_feas_points(self):
    return self._x_feas

  def check_x_feas_is_feas(self, x_feas: CandidatePoint = None):
    """_summary_

    :param x_feas: _description_, defaults to None
    :type x_feas: CandidatePoint, optional
    :param eval_type: _description_, defaults to None
    :raises IOError: _description_
    """
    if x_feas.evaluated and x_feas.status != DESIGN_STATUS.ERROR:
      h = x_feas.h
      if h is None or not np.isclose(h, 0.0, rtol=1e-09, atol=1e-09):
        raise IOError(f"Error: Barrier: xFeas' h value must be 0.0, got: {h}")

  def check_x_inf(
          self, x_inf: CandidatePoint = None):
    """_summary_

    :param x_inf: _description_, defaults to None
    :type x_inf: CandidatePoint, optional
    :param eval_type: _description_, defaults to None
    :raises IOError: _description_
    """
    if not x_inf.evaluated:
      raise IOError("Barrier: xInf must be evaluated before being set.")
