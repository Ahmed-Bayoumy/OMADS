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

from typing import Dict, List, Optional
import os
import pandas as pd
import numpy as np

from .._include import DESIGN_STATUS
from ._candidate_point import CandidatePoint


class Cache:
  """ In computing, a hash table (hash map) is a data structure that implements an associative array
  abstract data type, a structure that can map keys to values. A hash table uses a hash function to
  compute an index, also called a hash code, into an array of buckets or slots, from which the
  desired value can be found. During lookup, the key is hashed and the resulting hash indicates
  where the corresponding value is stored.

  Internally the cache is a plain, insertion-ordered list of candidates (position i is the
  historical integer "index" that the rest of the solver already threads around) plus a
  signature -> position dict for O(1) duplicate checks/updates, and small incrementally
  maintained index lists for the "was_center"/"improving" flags so the hot sampling path
  never has to rescan the whole cache to answer "how many centers do I have so far".

  Note: every query/aggregate method below deliberately excludes the most recently inserted
  candidate (indices are compared with `< self._last_index`, not `<=`), mirroring the
  `iloc[0:self._last_index]` slicing the previous pandas-backed implementation used
  throughout -- that is existing, load-bearing behavior (the newest candidate only becomes
  visible to these aggregates once a further candidate is added), not something introduced
  here, and changing it would shift which sampling branch/parameters the solver picks."""

  def __init__(self, capacity: int = 1):
    self._candidates: List[CandidatePoint] = []
    self._sig_to_index: Dict[int, int] = {}
    self._center_indices: List[int] = []
    self._center_index_set: set = set()
    self._improving_indices: List[int] = []
    self._improving_index_set: set = set()
    self._capacity: int = capacity + 100
    self._is_pareto: bool = False
    self._last_index: int = -1

  # ---------------------------------------------------------------- helpers
  def _sync_flag_indices(self, index: int):
    """Keep the incremental was_center/improving index lists exactly in sync
    with the candidate's live flags (self-healing both ways), rather than
    assuming the flags are monotonic -- update_candidate_in_cache overwrites
    a candidate's whole __dict__, so a flag can in principle flip back to
    False, and this must mirror that instead of leaving a stale entry
    behind (which the previous re-query-every-time implementation could
    never do)."""
    c = self._candidates[index]
    if c._was_center:
      if index not in self._center_index_set:
        self._center_index_set.add(index)
        self._center_indices.append(index)
    elif index in self._center_index_set:
      self._center_index_set.discard(index)
      self._center_indices.remove(index)
    if c._improving:
      if index not in self._improving_index_set:
        self._improving_index_set.add(index)
        self._improving_indices.append(index)
    elif index in self._improving_index_set:
      self._improving_index_set.discard(index)
      self._improving_indices.remove(index)

  def _center_indices_visible(self) -> List[int]:
    # _center_indices is in flag-set order, not candidate-insertion order (an
    # older candidate can be marked a center after newer ones already were).
    # The previous DataFrame query returned rows in positional (insertion
    # index) order regardless of when the flag was set, so sort to match.
    n = self._last_index
    return sorted(i for i in self._center_indices if i < n)

  def _improving_indices_visible(self) -> List[int]:
    n = self._last_index
    return sorted(i for i in self._improving_indices if i < n)

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

  def n_centers(self):
    return len(self._center_indices_visible())

  def get_maximum_cstr_violation(self) -> float:
    hs = [c._h for c in self._candidates[:self._last_index]
          if c._status == DESIGN_STATUS.INFEASIBLE and c._h >= 0]
    return min(hs) if hs else 0.001

  def n_all_non_error_candidates(self):
    h_max: float = self.get_maximum_cstr_violation()
    return sum(
        1 for c in self._candidates[:self._last_index]
        if c._status not in (DESIGN_STATUS.ERROR, DESIGN_STATUS.UNEVALUATED)
        and c._h <= h_max)

  def get_splitted_sorted_candidates(  # noqa: C901
          self, x_better: List[CandidatePoint],
          x_worse: List[CandidatePoint],
          f_better: np.ndarray,
          f_worse: np.ndarray,
          ratio: float = 0.2):
    h_max: float = self.get_maximum_cstr_violation()

    prim: List[CandidatePoint] = []
    sec: List[CandidatePoint] = []
    inf: List[CandidatePoint] = []
    for c in self._candidates[:self._last_index]:
      if c._status in (DESIGN_STATUS.ERROR, DESIGN_STATUS.UNEVALUATED):
        continue
      if c._status != DESIGN_STATUS.INFEASIBLE:
        prim.append(c)
      elif c._h <= h_max:
        sec.append(c)
      else:
        inf.append(c)

    # Stable sort (ties keep insertion order), ascending by objective/violation
    prim = sorted(prim, key=lambda c: c._f)
    sec = sorted(sec, key=lambda c: c._h)
    inf = sorted(inf, key=lambda c: c._h)

    if len(prim) <= 5:
      ratio = 1
    split_idx = int(ratio * len(prim))

    df_better = prim[:split_idx]
    df_worse = prim[split_idx:] if ratio < 1 else []

    for c in df_better:
      x_better.append(c._coords)
    for c in sec:
      x_better.append(c._coords)

    if ratio < 1:
      for c in df_worse:
        x_worse.append(c._coords)
    for c in inf:
      x_worse.append(c._coords)

    for c in df_better:
      f_better.append(sum(c._f))
    for c in sec:
      f_better.append(sum(c._f))
    if ratio < 1:
      for c in df_worse:
        f_worse.append(sum(c._f))
    for c in inf:
      f_worse.append(sum(c._f))

  def n_improving(self):
    return len(self._improving_indices_visible())

  def n_non_improving(self):
    return sum(
        1 for c in self._candidates[:self._last_index]
        if not c._improving and not c._was_center)

  def best_hash_id(self):
    idxs = self._center_indices_visible()
    if not idxs:
      raise IndexError("Cache.best_hash_id: no center candidates recorded yet.")
    return self._candidates[idxs[-1]]

  def add_to_cache(self, x: CandidatePoint):
    if x.signature is None or x.signature != hash(tuple((x._coords))):
      x.signature = hash(tuple((x._coords)))
    index = self._sig_to_index.get(x.signature)
    if index is not None:
      self.update_candidate_in_cache(x)
    else:
      clone = x.clone()
      index = len(self._candidates)
      self._candidates.append(clone)
      self._sig_to_index[x.signature] = index
      self._last_index = index
      self._sync_flag_indices(index)

  def update_candidate_in_cache(self, x: CandidatePoint):
    if x.signature is None or x.signature != hash(tuple((x._coords))):
      x.signature = hash(tuple((x._coords)))
    index = self._sig_to_index.get(x.signature)
    if index is not None:
      self._candidates[index].__dict__.update(x.__dict__)
      self._sync_flag_indices(index)

  def get_candidate_from_cache_by_signature(self, hash_id: int):
    index: int = self._sig_to_index[hash_id]
    return self._candidates[index].clone()

  def get_candidate_from_cache_by_index(self, index: int):
    return self._candidates[index].clone()

  def _set_candidate_attr(self, attr: dict):
    x: CandidatePoint = CandidatePoint()
    for key, value in attr.items():
      setattr(x, key, value)
    return x

  def get_cache_candidate_points(self) -> List[CandidatePoint]:
    # Unlike the query-style methods above, this one is (and was) inclusive
    # of the most recently inserted candidate.
    return [c.clone() for c in self._candidates[:self._last_index + 1]]

  def get_all_cache_points(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = [c._coords for c in self._candidates[:self._last_index]
         if c._evaluated and (hmax is None or c._h <= hmax)]
    return np.array(x) if nsamples is None or nsamples > len(x) else np.array(
        x[len(x) - nsamples:])

  def get_all_cache_points_f(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = [c._f[0] for c in self._candidates[:self._last_index]
         if c._evaluated and (hmax is None or c._h <= hmax)]
    return np.array(x) if nsamples is None or nsamples > len(x) else np.array(
        x[len(x) - nsamples:])

  def set_improving_candidate(self, x: CandidatePoint):
    # NOTE: the pandas-backed implementation this replaced set this flag via
    # `self.data_cache.iloc[index]["_improving"] = True`, a chained-indexing
    # write that pandas documents as never propagating back to the original
    # frame -- confirmed with a standalone repro (pandas raises
    # ChainedAssignmentError on exactly this pattern). So previously only the
    # single candidate seeded with the flag already set at insertion time
    # (the initial baseline point) was ever actually counted as
    # improving/center; every later set_improving_candidate/set_center_candidate
    # call was silently a no-op. This implementation applies the flag for
    # real, which is strictly more correct and only ever measured as
    # equal-or-better in regression testing (never worse) -- see PR notes.
    index: int = self._sig_to_index[x.signature]
    self._candidates[index]._improving = True
    self._sync_flag_indices(index)

  def set_center_candidate(self, x: CandidatePoint):
    # See the note in set_improving_candidate: same previously-inert flag write.
    index: int = self._sig_to_index[x.signature]
    self._candidates[index]._was_center = True
    self._sync_flag_indices(index)

  def get_all_improving_candidates(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = [self._candidates[i] for i in self._improving_indices_visible()
         if hmax is None or self._candidates[i]._h <= hmax]
    x = [c.clone() for c in x]
    return x if nsamples is None or nsamples > len(x) else x[len(x) - nsamples:]

  def get_all_improving_points(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = [self._candidates[i]._coords for i in self._improving_indices_visible()
         if hmax is None or self._candidates[i]._h <= hmax]
    return np.array(x) if nsamples is None or nsamples > len(x) else np.array(
        x[len(x) - nsamples:])

  def get_all_improving_points_f(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = [self._candidates[i]._f[0] for i in self._improving_indices_visible()
         if hmax is None or self._candidates[i]._h <= hmax]
    return np.array(x) if nsamples is None or nsamples > len(x) else np.array(
        x[len(x) - nsamples:])

  def get_all_non_improving_candidates(
          self, nsamples: int = None) -> List[CandidatePoint]:
    x = [c.clone() for c in self._candidates[:self._last_index]
         if not c._improving and not c._was_center]
    return x if nsamples is None or nsamples > len(x) else x[len(x) - nsamples:]

  def get_all_non_improving_points(
          self, nsamples: int = None) -> List[CandidatePoint]:
    x = [c._coords for c in self._candidates[:self._last_index]
         if not c._improving and not c._was_center]
    return np.array(x) if nsamples is None or nsamples > len(x) else np.array(
        x[len(x) - nsamples:])

  def get_all_non_improving_points_f(
          self, nsamples: int = None) -> List[CandidatePoint]:
    x = [c._f[0] for c in self._candidates[:self._last_index]
         if not c._improving and not c._was_center]
    return np.array(x) if nsamples is None or nsamples > len(x) else np.array(
        x[len(x) - nsamples:])

  def get_all_center_candidates(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = [self._candidates[i] for i in self._center_indices_visible()
         if hmax is None or self._candidates[i]._h <= hmax]
    x = [c.clone() for c in x]
    if nsamples is None or nsamples >= len(x):
      return x
    return x[len(x) - nsamples:]

  def get_all_center_points(
          self, nsamples: int = None, hmax: float = None) -> List[float]:
    x = [self._candidates[i]._coords for i in self._center_indices_visible()
         if hmax is None or self._candidates[i]._h <= hmax]
    return x if nsamples is None or nsamples > len(x) else x[len(x) - nsamples:]

  def get_all_center_points_f(
          self, nsamples: int = None, hmax: float = None) -> List[float]:
    x = [self._candidates[i]._f[0] for i in self._center_indices_visible()
         if hmax is None or self._candidates[i]._h <= hmax]
    return x if nsamples is None or nsamples > len(x) else x[len(x) - nsamples:]

  def get_all_center_points_h(
          self, nsamples: int = None, hmax: float = None) -> List[float]:
    x = [self._candidates[i]._h for i in self._center_indices_visible()
         if hmax is None or self._candidates[i]._h <= hmax]
    return x if nsamples is None or nsamples > len(x) else x[len(x) - nsamples:]

  def get_all_primary_center_candidates(
          self, nsamples: int = None) -> List[CandidatePoint]:
    x = [self._candidates[i].clone() for i in self._center_indices_visible()
         if self._candidates[i]._status == DESIGN_STATUS.FEASIBLE]
    return x if nsamples is None or nsamples > len(x) else x[len(x) - nsamples:]

  def get_all_secondary_center_candidates(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = [self._candidates[i].clone() for i in self._center_indices_visible()
         if self._candidates[i]._status == DESIGN_STATUS.INFEASIBLE
         and (hmax is None or self._candidates[i]._h <= hmax)]
    return x if nsamples is None or nsamples > len(x) else x[len(x) - nsamples:]

  def get_all_nd_candidates(self) -> List[CandidatePoint]:
    return [c.clone() for c in self._candidates[:self._last_index]
            if c._is_nondominated]

  def get_all_feasible_nd_candidates(
          self, nsamples: int = None) -> List[CandidatePoint]:
    x = [c.clone() for c in self._candidates[:self._last_index]
         if c._is_nondominated and c._status == DESIGN_STATUS.FEASIBLE]
    return x if nsamples is None or nsamples > len(x) else x[len(x) - nsamples:]

  def get_all_infeasible_nd_candidates(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = [c.clone() for c in self._candidates[:self._last_index]
         if c._is_nondominated and c._status == DESIGN_STATUS.INFEASIBLE
         and (hmax is None or c._h <= hmax)]
    return x if nsamples is None or nsamples > len(x) else x[len(x) - nsamples:]

  @property
  def capacity(self) -> int:
    return self._capacity

  @capacity.setter
  def capacity(self, value: int):
    self._capacity = value

  @property
  def data_cache(self) -> pd.DataFrame:
    """A pandas view of the cache, materialized on demand for backward
    compatibility with external consumers; not used on any internal hot path.

    :rtype: pd.DataFrame
    """
    cols = list(self._get_candidate_class_attr().keys())
    n = self._last_index + 1
    df = pd.DataFrame(
        index=[c.signature for c in self._candidates[:n]], columns=cols)
    for i, c in enumerate(self._candidates[:n]):
      for key in cols:
        df.iat[i, cols.index(key)] = getattr(c, key, None)
    return df

  @property
  def is_pareto(self) -> bool:
    return self._is_pareto

  @property
  def last_index(self):
    return self._last_index

  @last_index.setter
  def last_index(self, value: int) -> int:
    self._last_index = value

  @property
  def size(self) -> int:
    """A getter of the size of the hash ID list

    :rtype: int
    """
    return len(self._candidates)

  def is_duplicate(self, x: CandidatePoint, add: bool = True) -> bool:
    """Check if the point is in the cache memory

    :param x: Design point
    :type x: Point
    :return: A boolean flag indicate whether the point exist in the cache memory
    :rtype: bool
    """
    if x.signature is None or x.signature != hash(tuple((x._coords))):
      x.signature = hash(tuple((x._coords)))
    is_dup = x.signature in self._sig_to_index
    if not is_dup and add:
      self.add_to_cache(x)

    return is_dup

  def is_duplicate_in_set(
          self, xp: CandidatePoint, x_pool: List[CandidatePoint]) -> bool:
    """Check if the point is in the cache memory

    :param x: Design point
    :type x: Point
    :return: A boolean flag indicate whether the point exist in the cache memory
    :rtype: bool
    """
    is_dup: bool = False
    if xp.signature is None or xp.signature != hash(tuple((xp._coords))):
      xp.signature = hash(tuple((xp._coords)))
    for i, x in enumerate(x_pool):
      if x.signature is None or x.signature != hash(tuple((x._coords))):
        x.signature = hash(tuple((x._coords)))
      if xp.signature == x.signature:
        is_dup = True
        break

    return is_dup

  def update(self, x: List[CandidatePoint]):
    """Update the cache points with evaluated design criteria

    :param x: _description_
    :type x: List[CandidatePoint]
    """
    for xt in x:
      if not isinstance(xt.signature, int) or xt.signature is None:
        xt.signature = hash(tuple((xt.coordinates)))
      if xt.signature in self._sig_to_index:
        self.update_candidate_in_cache(xt)
      else:
        self.add_to_cache(xt)

  def remove_candidates_from_hash(self, x: List[CandidatePoint]):
    signatures_to_remove = {
        xt.signature for xt in x if xt.signature in self._sig_to_index}
    if not signatures_to_remove:
      return

    self._candidates = [
        c for c in self._candidates if c.signature not in signatures_to_remove]
    self._sig_to_index = {c.signature: i for i, c in enumerate(self._candidates)}
    self._last_index = len(self._candidates) - 1
    self._center_indices = [
        i for i, c in enumerate(self._candidates) if c._was_center]
    self._center_index_set = set(self._center_indices)
    self._improving_indices = [
        i for i, c in enumerate(self._candidates) if c._improving]
    self._improving_index_set = set(self._improving_indices)

  def save_cache(self, path: str = None, table_name: str = 'data_cache'):
    if path is None:
      path = os.path.join(os.getcwd(), "cache.hd5")
    with pd.HDFStore('cache_data.h5', mode='a') as store:
      # format='table' enables appending/querying
      store.append(table_name, self.data_cache, format='table')

  def load_cache(self, path: str = None, table_name: str = "data_cache"):
    if path is None:
      raise FileNotFoundError("Could not find the cache file for loading!")

    df = pd.read_hdf('cache_data.h5', key=table_name)
    self._candidates = []
    self._sig_to_index = {}
    self._center_indices = []
    self._center_index_set = set()
    self._improving_indices = []
    self._improving_index_set = set()
    self._last_index = -1
    for _, row in df.iterrows():
      cp = CandidatePoint()
      for key, value in row.to_dict().items():
        setattr(cp, key, value)
      self.add_to_cache(cp)
