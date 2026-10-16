"""
#-------------------------------------------------------------------------------------#
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
from typing import Dict, List, Optional
import numpy as np

from .._include import COMPARE_TYPE, INSERTION_FLAG
from .._points._candidate_point import CandidatePoint
from .._points._cache import Cache

from .._setup._parameters import Parameters
from .._setup._options import Options
from .._mesh._mesh import Mesh


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


class Elements:
  """
  Elements to candidates mapper.

  Backed by a plain, insertion-ordered list of candidates (position i is the
  integer "index" the rest of the barrier logic already threads around) plus
  a signature -> position dict for O(1) duplicate checks/updates -- replacing
  the previous pandas-DataFrame-as-hashtable implementation, which paid for a
  full index-list rebuild on every insertion and a query-engine round trip on
  every single-element access.
  """

  def __init__(self, capacity: int):
    self._elements: List[Optional[CandidatePoint]] = []
    self._sig_to_index: Dict[int, int] = {}
    self._capacity = capacity
    self._last_index = -1

  def __getitem__(self, index) -> CandidatePoint:
    if isinstance(index, list) or isinstance(index, np.ndarray):
      return [self.get_candidate_from_cache_by_index(int(i)) for i in index]
    else:
      return self.get_candidate_from_cache_by_index(index)

  def __setitem__(self, index: int, candidate: CandidatePoint):
    self.add_to_candidates_table_index(index=index, x=candidate)

  def append(self, candidate: CandidatePoint):
    self.add_to_candidates_table(x=candidate)

  def index(self, x: CandidatePoint):
    return self._sig_to_index[x.signature]

  def get_by_eval_no(self, n: int):
    # Matches the previous `_candidates_table[0:self._last_index]` slice:
    # excludes the most recently inserted element, same as Cache's query
    # methods (see _points/_cache.py docstring for why).
    for c in self._elements[:self._last_index]:
      if c is not None and c._eval_no == n:
        return c.clone()
    return None

  def get_index_loc_by_signature(self, hash_id: int) -> int:
    return self._sig_to_index[hash_id]

  def __deleteitem__(self, index: int):
    pass

  def __iter__(self):
    x = self.get_cache_candidate_points()
    return iter(x)

  def get_cache_candidate_points(self) -> List[CandidatePoint]:
    return [c.clone() for c in self._elements[:self._last_index + 1]
            if c is not None]

  def get_candidate_from_elements_by_signature(self, hash_id: int):
    index: int = self._sig_to_index[hash_id]
    return self._elements[index].clone()

  def get_candidate_from_cache_by_index(self, index: int):
    # Callers (e.g. the progress bar) probe indices across the whole budget
    # range, not just filled slots; the previous DataFrame, preallocated up
    # to capacity, tolerated that silently, so mirror the same "not yet
    # filled" -> None contract here instead of raising IndexError.
    if index < 0 or index >= len(self._elements):
      return None
    c = self._elements[index]
    return c.clone() if c is not None else None

  def get_raw(self, index: int) -> Optional[CandidatePoint]:
    """Internal accessor for AdaptiveBarrier's own read-only bookkeeping
    (dominance comparisons, f/h lookups): returns the stored candidate by
    reference instead of paying for a clone(). Never mutate the result --
    external callers (evaluator, search/poll steps, progress bar) must keep
    using __getitem__/get_candidate_from_cache_by_index, which still clone."""
    if index < 0 or index >= len(self._elements):
      return None
    return self._elements[index]

  def add_to_candidates_table(self, x: CandidatePoint):
    if x.signature is None or x.signature != hash(tuple((x._coords))):
      x.signature = hash(tuple((x._coords)))
    # Safeguard to avoid saving duplicates
    if x.signature in self._sig_to_index:
      self.update_candidate_in_cache(x)
    else:
      clone = x.clone()
      index = len(self._elements)
      self._elements.append(clone)
      self._sig_to_index[x.signature] = index
      self._last_index = index

  def add_to_candidates_table_index(self, index: int, x: CandidatePoint):
    if x.signature is None or x.signature != hash(tuple((x._coords))):
      x.signature = hash(tuple((x._coords)))
    # Safeguard to avoid saving duplicates
    if x.signature in self._sig_to_index:
      self.update_candidate_in_cache(x)
    else:
      clone = x.clone()
      if index < len(self._elements):
        self._elements[index] = clone
      else:
        self._elements.extend([None] * (index - len(self._elements)))
        self._elements.append(clone)
      self._sig_to_index[x.signature] = index
      self._last_index = index

  def update_candidate_in_cache(self, x: CandidatePoint):
    if x.signature is None or x.signature != hash(tuple((x._coords))):
      x.signature = hash(tuple((x._coords)))
    index = self._sig_to_index.get(x.signature)
    if index is not None and self._elements[index] is not None:
      self._elements[index].__dict__.update(x.__dict__)

  def _get_candidate_class_attr(self):
    cp: CandidatePoint = CandidatePoint()
    out = {
        attr: getattr(cp, attr) for attr in dir(cp)
        if attr.title() and attr.startswith('_') and
        not attr.startswith('__') and
        not
        callable(getattr(cp, attr)) and
        not isinstance(getattr(type(cp),
                               attr, None),
                       property)}
    del cp
    return out

  def _set_candidate_attr(self, attr: dict):
    x: CandidatePoint = CandidatePoint()
    for key, value in attr.items():
      setattr(x, key, value)
    return x

  @property
  def ht(self) -> Cache:
    return self._ht

  @ht.setter
  def ht(self, value: Cache):
    self._ht = value

  @ht.deleter
  def ht(self):
    del self._ht


class AdaptiveBarrier:
  dims: List[int]
  h_max: float
  elements: Elements
  meshes: List[Mesh]
  parent_indexes: List[int]
  within_fk: List[bool]
  within_uk: List[bool]
  within_ik: List[bool]
  max_size: int
  last_index: int
  param: Parameters
  options: Options

  def __init__(
          self, param: Parameters, options: Options, max_size=30000):
    # Check for valid dimensions
    if param.n <= 0 or param.nobj <= 0:
      raise ValueError("Non coherent dimensions")
    if param.h_max < 0:
      raise ValueError("h_max must be positive")
    if max_size <= 0:
      raise ValueError(
          "The barrier cannot contain a negative number of elements")

    # Initialize the attributes
    self.dims = [param.n, param.nobj]
    self.h_max = param.h_max
    # self._elements = np.empty((options.budget,))
    self.elements = Elements(capacity=30000)
    # self.elements = [None] * max_size  # placeholder for CandidatePoint
    self.meshes = [None] * max_size    # placeholder for Mesh
    # Assuming parent indexes are integers
    self.parent_indexes = [0] * max_size
    self.within_fk = [False] * max_size
    self.within_uk = [False] * max_size
    self.within_ik = [False] * max_size
    self.max_size = max_size
    self.last_index = -1
    self.param = param
    self.options = options

  # def get_candidate_element(
  #         self, index: int, hashtable: Cache) -> CandidatePoint:
  #   return hashtable.get_candidate_from_cache_by_signature(
  #       hash_id=self._elements[index])

  # def set_candidate_element(self, index: int, x: CandidatePoint):
  #   self._elements[index] = x.signature

  def check_dimension(self, v: CandidatePoint, m: Mesh):
    if self.dims[0] != m.n:
      raise ValueError(
          "Granular mesh and barrier do not have compatible dimensions")
    if self.dims[1] != len(v.f):
      raise ValueError(
          "Objective vector and barrier do not have compatible dimensions")

  def check_is_full(self):
    if self.last_index == self.max_size:
      raise ValueError("Maximum size of barrier reached: cannot add element")

  def get_fk(self) -> np.ndarray[CandidatePoint]:
    # within_fk/uk/ik are preallocated to max_size but only the first
    # last_index+1 entries can ever be True, so bound the scan there instead
    # of converting the full (often much larger) preallocated list every call.
    n = self.last_index + 1
    fk_i = np.where(np.array(self.within_fk[:n]) == True)[0]  # noqa: E712
    return self.elements[fk_i] if len(fk_i) > 0 else np.empty((0,))

  def get_ik(self) -> np.ndarray[CandidatePoint]:
    n = self.last_index + 1
    ik_i = np.where(np.array(self.within_ik[:n]) == True)[0]  # noqa: E712
    return self.elements[ik_i] if len(ik_i) > 0 else np.empty((0,))

  def get_uk(self) -> np.ndarray[CandidatePoint]:
    n = self.last_index + 1
    uk_i = np.where(np.array(self.within_uk[:n]) == True)[0]  # noqa: E712
    return self.elements[uk_i] if len(uk_i) > 0 else np.empty((0,))

  def get_nd_elements(self):
    fk = self.get_fk()
    ik = self.get_ik()
    fk_list = fk.tolist() if hasattr(fk, "tolist") else list(fk)
    ik_list = ik.tolist() if hasattr(ik, "tolist") else list(ik)
    return fk_list + ik_list

  def get_filled_elements(self):
    out = []
    for x in self.elements:
      if x is not None:
        x.h_max = self.h_max
        out.append(x)
      else:
        break
    return out

  def add_feasible(self, v: CandidatePoint, m: Mesh):
    # Preliminary checks
    self.check_dimension(v, m)
    self.check_is_full()

    if not v.is_feasible():
      raise ValueError(
          "Trying to insert an infeasible element into the set of feasible points.")

    # Get current elements in fh (index kept alongside its element so the
    # dominance loop below doesn't have to re-derive it via parentindices --
    # that used to recompute an O(n) lookup on *every* iteration, making this
    # whole method O(n^2) in the number of accumulated feasible points, and
    # get progressively slower every iteration as the barrier grows).
    fh_indices = [
        i for i in range(self.last_index + 1)
        if self.within_fk[i] and self.elements.get_raw(i) is not None]
    nb_elements_in_fh = len(fh_indices)

    # Empty set case
    if nb_elements_in_fh == 0:
      self.last_index += 1
      self.elements[self.last_index] = v
      self.meshes[self.last_index] = copy.deepcopy(m)
      self.within_fk[self.last_index] = True
      return INSERTION_FLAG.IMPROVES

    insert = True
    insertion_flag = INSERTION_FLAG.IMPROVES

    # Check if v dominates an element of fh
    for corresponding_index in fh_indices:
      comp_flag = v.__compare__(self.elements.get_raw(corresponding_index))
      if comp_flag == COMPARE_TYPE.DOMINATING:
        self.within_fk[corresponding_index] = False
        insertion_flag = INSERTION_FLAG.DOMINATES
      elif comp_flag in [COMPARE_TYPE.DOMINATED, COMPARE_TYPE.EQUAL]:
        insertion_flag = INSERTION_FLAG.IS_DOMINATED
        insert = False
        break

    # Check extension
    if self.extends(v, m, True):
      insertion_flag = INSERTION_FLAG.EXTENDS

    # Add the point into the barrier ...
    self.last_index += 1
    self.elements[self.last_index] = v
    self.meshes[self.last_index] = copy.deepcopy(m)

    # ... and into the set of feasible points
    if insert:
      self.within_fk[self.last_index] = True

    return insertion_flag

  def add_infeasible(self, v: CandidatePoint, m: Mesh):  # noqa: C901
    # Preliminary checks
    self.check_dimension(v, m)
    self.check_is_full()

    if v.is_feasible():
      raise ValueError(
          "Trying to insert a feasible point into the set of infeasible points")

    # Reject the point if above the threshold.
    if v.h >= self.h_max:
      self.last_index += 1
      self.elements[self.last_index] = v
      self.meshes[self.last_index] = copy.deepcopy(m)
      return INSERTION_FLAG.REJECTED

    # Get current elements in uk (see add_feasible for why indices are kept
    # alongside the filter instead of recovered later via parentindices).
    uk_indices = [
        i for i in range(self.last_index + 1)
        if self.within_uk[i] and self.elements.get_raw(i) is not None]
    nb_elements_in_uk = len(uk_indices)
    prev_nb_best_inf_pts = sum(
        1 for x in self.within_ik[:self.last_index] if x)

    # Empty set case
    if nb_elements_in_uk == 0:
      self.last_index += 1
      self.elements[self.last_index] = v
      self.meshes[self.last_index] = copy.deepcopy(m)
      self.within_uk[self.last_index] = True
      self.within_ik[self.last_index] = True
      return INSERTION_FLAG.IMPROVES

    insert = True

    # Check if v dominates an element of uk
    for corresponding_index in uk_indices:
      comp_flag = v.__compare__(self.elements.get_raw(corresponding_index))
      if comp_flag == COMPARE_TYPE.DOMINATING:
        self.within_uk[corresponding_index] = False
        self.within_ik[corresponding_index] = False
      elif comp_flag in [COMPARE_TYPE.DOMINATED, COMPARE_TYPE.EQUAL]:
        insert = False
        break

    insertion_flag = INSERTION_FLAG.IMPROVES

    # Check extension
    if self.extends(v, m, False):
      insertion_flag = INSERTION_FLAG.EXTENDS

    # Add the point into the barrier
    self.last_index += 1
    self.elements[self.last_index] = v
    self.meshes[self.last_index] = copy.deepcopy(m)

    # Add the point into the set of filter points
    if insert:
      self.within_uk[self.last_index] = True
    else:
      return INSERTION_FLAG.IS_DOMINATED

    insertion_best_flag = INSERTION_FLAG.IS_DOMINATED

    # Update the set of best infeasible points
    if insert:
      insertion_best_flag = self._update_ik_set_after_insertion()

    # Set corresponding flags according to the situation
    if insertion_flag == INSERTION_FLAG.EXTENDS:
      return insertion_flag

    if insertion_best_flag == INSERTION_FLAG.IS_DOMINATED:
      return INSERTION_FLAG.NO_IMPROVEMENT
    else:
      # The number of best infeasible points may have been reduced
      # by the insertion into the filter, so this check is needed.
      # (within_ik is preallocated to max_size; entries past last_index are
      # always False, so bound the sum instead of scanning the whole array.)
      if sum(self.within_ik[:self.last_index + 1]) <= prev_nb_best_inf_pts:
        return INSERTION_FLAG.DOMINATES
      else:
        return insertion_best_flag

  def _private_compare_ik_elements(self, v1, v2):
    isbetter = False
    isworse = False

    # Iterate over corresponding elements in f attributes of v1 and v2
    for f1, f2 in zip(v1.fobj, v2.fobj):
      if f1 < f2:
        isbetter = True
      if f2 < f1:
        isworse = True
      if isworse and isbetter:
        break  # No need to continue once we know the result

    if isworse:
      if isbetter:
        return COMPARE_TYPE.INDIFFERENT
      else:
        return COMPARE_TYPE.DOMINATED
    else:
      if isbetter:
        return COMPARE_TYPE.DOMINATING
      else:
        return COMPARE_TYPE.EQUAL

  def _update_ik_set_after_insertion(self):
    candidate = self.elements.get_raw(self.last_index)
    domination_flags = [False] * self.last_index  # Initialize with False

    insert = True
    insertion_flag = INSERTION_FLAG.IMPROVES

    # Check if the candidate dominates any element in ik
    for index, is_in_ik in enumerate(self.within_ik[:self.last_index]):
      if is_in_ik:
        comp_flag = self._private_compare_ik_elements(
            candidate, self.elements.get_raw(index))
        if comp_flag == COMPARE_TYPE.DOMINATING:
          domination_flags[index] = True
          insertion_flag = INSERTION_FLAG.DOMINATES
        elif comp_flag in [COMPARE_TYPE.DOMINATED, COMPARE_TYPE.EQUAL]:
          insertion_flag = INSERTION_FLAG.IS_DOMINATED
          insert = False
          break

    # Update the set of best infeasible points (ik)
    for index, flag in enumerate(domination_flags):
      if flag:
        self.within_ik[index] = True
      else:
        self.within_ik[index] = False
    self.within_ik[self.last_index] = insert

    return insertion_flag

  def extends(self, v: CandidatePoint, m: Mesh, is_feasible: bool):
    # Preliminary check
    self.check_dimension(v, m)

    # Get best points based on feasibility
    tmp_best_pts = [
        self.elements.get_raw(i) for i in range(self.last_index + 1)
        if (self.within_fk[i] if is_feasible else self.within_uk[i])]

    if len(tmp_best_pts) == 0:
      return False

    # Ideal vector of the best points
    ideal_v = [float('inf')] * self.dims[1]
    for elt in tmp_best_pts:
      ideal_v = [min(elt.f[i], ideal_v[i]) for i in range(self.dims[1])]

    return any(v.f[i] < ideal_v[i] for i in range(self.dims[1]))

  def update_barrier(self, h_max: float):
    if h_max < 0:
      raise ValueError("h_max cannot be negative")
    self.h_max = h_max

    filter_flags = [False] * self.last_index

    # Mark uk elements that exceed h_max
    for index, is_in_uk in enumerate(self.within_uk[:self.last_index]):
      if is_in_uk and self.elements.get_raw(index).h > self.h_max:
        filter_flags[index] = True

    # Remove all uk elements above the threshold
    for i in range(self.last_index):
      if filter_flags[i]:
        self.within_uk[i] = False

    # Remove all ik elements above the threshold
    for i in range(self.last_index):
      if filter_flags[i]:
        self.within_ik[i] = False

    # Reinsert potential new non-dominated points into the set ik
    self._update_ik_after_h_max_setting()

  def _update_ik_after_h_max_setting(self):
    for index, is_in_uk in enumerate(self.within_uk[:self.last_index]):
      if is_in_uk and not self.within_ik[index]:
        # Check if the element indexed by 'index' can be inserted in the set ik
        insert = True
        for index_2, is_in_uk_2 in enumerate(self.within_uk[:self.last_index]):
          if is_in_uk_2 and index != index_2:
            comp_flag = self._private_compare_ik_elements(
                self.elements.get_raw(index), self.elements.get_raw(index_2))
            if comp_flag == COMPARE_TYPE.DOMINATING:
              self.within_ik[index_2] = False
            elif comp_flag in [COMPARE_TYPE.DOMINATED, COMPARE_TYPE.EQUAL]:
              insert = False
              break
        self.within_ik[index] = insert

  def frame_centers(self, w: int, use_dom_selection=True):  # noqa: C901
    # TODO: Syncing the mesh updates conducted on active barrier elements last index with the center index selected below
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
        for ind in range(0, self.last_index + 1):
          if self.within_ik[ind]:
            v = self.elements.get_raw(ind)
            tmp_dom_distance = min(
                [sum(elt.f - np.minimum(v.f, elt.f)) for elt in self.get_fk()])
            if v.h <= self.h_max and tmp_dom_distance > tmp_dist:
              tmp_ind = ind
              tmp_dist = tmp_dom_distance

        if tmp_dist == 0:
          tmp_ind = 0
          tmp_dist = float('inf')
          for ind in range(0, self.last_index + 1):
            if self.within_ik[ind]:
              v = self.elements.get_raw(ind)
              tmp_dom_distance = min(
                  [sum(v.f - np.minimum(v.f, elt.f)) for elt in self.get_fk()])
              if v.h <= self.h_max and tmp_dom_distance < tmp_dist:
                tmp_ind = ind
                tmp_dist = tmp_dom_distance
        infeasible_index = tmp_ind
      else:
        tmp_ind = 0
        tmp_dist = float('inf')
        for ind in range(0, self.last_index + 1):
          if self.within_ik[ind]:
            v = self.elements.get_raw(ind)
            tmp_dom_distance = dom_distance(
                self.elements.get_raw(feasible_index).f, v.f)
            if v.h <= self.h_max and tmp_dom_distance < tmp_dist:
              tmp_ind = ind
              tmp_dist = tmp_dom_distance
        infeasible_index = tmp_ind

    return {"feasible": feasible_index, "infeasible": infeasible_index}

  def feasible_frame_center(self, w: int):  # noqa: C901
    # Get the feasible points (fh)
    fh = self.get_fk()
    if len(fh) == 0:
      return -1

    fk_indexes = self.parentindices(self.elements, fh)

    # Get maximum mesh size value
    fh_Δ_max = 0.0
    for ind in fk_indexes:
      Δ_val = np.linalg.norm(
          self.meshes[ind].get_frame_size_parameter(), np.inf)
      fh_Δ_max = max(fh_Δ_max, Δ_val)

    # Select candidates
    fh_selected_indexes = []
    for ind in fk_indexes:
      fh_elt_mesh = self.meshes[ind]
      Δ_val = np.linalg.norm(fh_elt_mesh.get_frame_size_parameter(), np.inf)
      if (10.0**(-w) * fh_Δ_max) <= Δ_val and not fh_elt_mesh.check_mesh_for_stopping():
        fh_selected_indexes.append(ind)

    # The selection must always work
    if len(fh_selected_indexes) == 0:
      return fk_indexes[0]

    # Only one point
    if len(fh_selected_indexes) == 1:
      return fh_selected_indexes[0]

    # Two points in the barrier
    if len(fh_selected_indexes) == 2 and len(fk_indexes) == 2:
      v1f = self.elements.get_raw(fh_selected_indexes[0]).f
      v2f = self.elements.get_raw(fh_selected_indexes[1]).f
      if np.linalg.norm(v1f, np.inf) > np.linalg.norm(v2f, np.inf):
        return fh_selected_indexes[0]
      else:
        return fh_selected_indexes[1]

    # More than two points
    frame_ind = -1
    maximum_gap = -1.0  # negative to deal with the case where two points in fh_selected_indexes

    for obj in range(self.dims[1]):
      fvalues = np.array([(self.elements.get_raw(ind).f[obj], ind)
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

      # Intermediate points
      for i in range(1, len(fvalues) - 1):
        current_gap = (fvalues[i + 1][0] - fvalues[i - 1][0]) / (fmax - fmin)
        if fvalues[i][1] in fh_selected_indexes and current_gap >= maximum_gap:
          maximum_gap = current_gap
          frame_ind = fvalues[i][1]

      # Extreme points
      current_gap = 2 * (fvalues[-1][0] - fvalues[-2][0]) / (fmax - fmin)
      if fvalues[-1][1] in fh_selected_indexes and current_gap > maximum_gap:
        maximum_gap = current_gap
        frame_ind = fvalues[-1][1]

      current_gap = 2 * (fvalues[1][0] - fvalues[0][0]) / (fmax - fmin)
      if fvalues[0][1] in fh_selected_indexes and current_gap > maximum_gap:
        maximum_gap = current_gap
        frame_ind = fvalues[0][1]

    return int(frame_ind)

  def infeasible_frame_center(self):  # noqa: C901
    # Get the infeasible points (uk)
    uk = self.get_uk()
    if len(uk) == 0:
      return 0

    ik = self.get_ik()

    if len(ik) == 0:
      return 0

    ik_indexes = self.parentindices(self.elements, ik)

    # Find the minimum mesh size value (ik_Δ_min)
    xI_ind = np.argmin([elt.h for elt in self.elements[ik_indexes]])
    ik_Δ_min = np.linalg.norm(
        self.meshes[ik_indexes[xI_ind]].get_frame_size_parameter(),
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
      v1f = self.elements.get_raw(ik_selected_indexes[0]).f
      v2f = self.elements.get_raw(ik_selected_indexes[1]).f
      if np.linalg.norm(v1f, np.inf) > np.linalg.norm(v2f, np.inf):
        return ik_selected_indexes[0]
      else:
        return ik_selected_indexes[1]

    # More than two points
    frame_ind = 0
    maximum_gap = -1.0  # negative to handle the case where two points in ik_selected_indexes

    for obj in range(self.dims[1]):
      fvalues = np.array([(self.elements.get_raw(ind).f[obj], ind)
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
        current_gap = (fvalues[i + 1][0] - fvalues[i - 1][0]) / (fmax - fmin)
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

    return int(frame_ind)

  # Return the extent of the Pareto front (a customized one)
  def extent(self):
    # Get the Pareto front (fh)
    fh = self.get_fk()
    if len(fh) == 0:
      return 0

    fk_indexes = self.parentindices(self.elements, fh)

    extent_val = 0

    for obj in range(self.dims[1]):
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

  # Save Pareto front values to a file
  def save_pf_values(self, filename):
    with open(filename, 'w') as file:
      # Write values of the Pareto front
      for elt in self.get_fk():
        np.savetxt(file, np.transpose(elt.f), delimiter=' ')

  def parentindices(self,
                    vector: List[CandidatePoint],
                    subvector: List[CandidatePoint]):
    indices = []
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
