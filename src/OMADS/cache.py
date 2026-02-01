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

from typing import List
import pandas as pd
import numpy as np

from ._globals import DESIGN_STATUS
from .candidate_point import CandidatePoint


class Cache:
  """ In computing, a hash table (hash map) is a data structure that implements an associative array
  abstract data type, a structure that can map keys to values. A hash table uses a hash function to
  compute an index, also called a hash code, into an array of buckets or slots, from which the
  desired value can be found. During lookup, the key is hashed and the resulting hash indicates
  where the corresponding value is stored."""
  # _hash_id: List[int] = field(default_factory=list)
  # _best_hash_id: List[int] = field(default_factory=list)
  # _cache_dict: Dict[Any, Any] = field(default_factory=lambda: {})
  # _n_dim: int = 0
  # _is_pareto: bool = False
  # _last_index: int = -1
  # nd_points: Optional[List[CandidatePoint]] = None

  def __init__(self, capacity: int = 1):
    self._data_cache: pd.DataFrame = pd.DataFrame(
        index=range(capacity + 100),
        columns=self._get_candidate_class_attr())
    self._capacity: int = capacity + 100
    self._is_pareto: bool = False
    self._last_index: int = -1

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
    return self._data_cache.iloc[0:self._last_index].query('_was_center').shape[0]

  def get_maximum_cstr_violation(self) -> float:
    s1 = DESIGN_STATUS.ERROR
    s2 = DESIGN_STATUS.UNEVALUATED
    s3 = DESIGN_STATUS.INFEASIBLE
    filtered: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
        '_status != @s1 & _status != @s2 & _status == @s3 & _h >= 0')
    if '_h' in filtered.columns and filtered.shape[0] > 0:
      sorted_slice = filtered.sort_values(by='_h', ascending=True)
      return sorted_slice['_h'].min()
    else:
      return 0.001

  def n_all_non_error_candidates(self):
    s1 = DESIGN_STATUS.ERROR
    s2 = DESIGN_STATUS.UNEVALUATED
    h_max: float = self.get_maximum_cstr_violation()
    return self._data_cache.iloc[0: self._last_index].query(
        '_status != @s1 & _status != @s2 & _h <= @h_max').shape[0]

  def get_splitted_sorted_candidates(
          self, x_better: List[CandidatePoint],
          x_worse: List[CandidatePoint],
          f_better: np.ndarray,
          f_worse: np.ndarray,
          ratio: float = 0.2):
    s1 = DESIGN_STATUS.ERROR
    s2 = DESIGN_STATUS.UNEVALUATED
    s3 = DESIGN_STATUS.INFEASIBLE
    h_max: float = self.get_maximum_cstr_violation()

    filtered_prim: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
        '_status != @s1 & _status != @s2 & _status != @s3')

    filtered_sec: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
        '_status != @s1 & _status != @s2 & _status == @s3 & _h <= @h_max')

    filtered_inf: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
        '_status != @s1 & _status != @s2 & _status == @s3 & _h > @h_max')

    # Sort by the `_f` column (ascending)
    sorted_slice = filtered_prim.sort_values(by='_f', ascending=True)
    sorted_slice_sec = filtered_sec.sort_values(by='_h', ascending=True)
    sorted_slice_inf = filtered_inf.sort_values(by='_h', ascending=True)

    # Determine split point at 30 % of the original DataFrame’s rows
    if filtered_prim.shape[0] <= 5:
      ratio = 1
    split_idx = int(ratio * len(filtered_prim))

    # Split into two frames: first part (≤ split_idx) and remainder
    df_better = sorted_slice.iloc[:split_idx]

    if ratio < 1:
      df_worse = sorted_slice.iloc[split_idx:]

    for i, h_id in enumerate(df_better.index):
      x_better.append(df_better.at[h_id, '_coords'])

    for i, h_id in enumerate(sorted_slice_sec.index):
      x_better.append(sorted_slice_sec.at[h_id, '_coords'])

    if ratio < 1:
      for i, h_id in enumerate(df_worse.index):
        x_worse.append(df_worse.at[h_id, '_coords'])

    for i, h_id in enumerate(sorted_slice_inf.index):
      x_worse.append(sorted_slice_inf.at[h_id, '_coords'])

    for i, h_id in enumerate(df_better.index):
      f_better.append(sum(df_better.at[h_id, '_f']))

    for i, h_id in enumerate(sorted_slice_sec.index):
      f_better.append(sum(sorted_slice_sec.at[h_id, '_f']))
    if ratio < 1:
      for i, h_id in enumerate(df_worse.index):
        f_worse.append(sum(df_worse.at[h_id, '_f']))

    for i, h_id in enumerate(sorted_slice_inf.index):
      f_worse.append(sum(sorted_slice_inf.at[h_id, '_f']))

  def n_improving(self):
    return self._data_cache.iloc[0:self._last_index].query('_improving').shape[0]

  def n_non_improving(self):
    return self._data_cache.iloc[0: self._last_index].query(
        '~_improving & ~_was_center').shape[0]

  def best_hash_id(self):
    return self._data_cache.iloc[0:self._last_index].query('_was_center')[-1]

  def add_to_cache(self, x: CandidatePoint):
    if x.signature is None or x.signature != hash(tuple((x._coords))):
      x.signature = hash(tuple((x._coords)))
    if x.signature in self._data_cache.index:
      self.update_candidate_in_cache(x)
    else:
      self._last_index += 1
      new_index = self.data_cache.index.tolist()
      new_index[self._last_index] = x.signature
      self.data_cache.index = new_index
      # self.data_cache.at[x.signature, 'candidate'] = x
      for key, value in vars(x).items():
        if key in self.data_cache.columns:
          self.data_cache.loc[x.signature, key] = value

  def update_candidate_in_cache(self, x: CandidatePoint):
    if x.signature is None or x.signature != hash(tuple((x._coords))):
      x.signature = hash(tuple((x._coords)))
    if x.signature in self.data_cache.index:
      for key, value in vars(x).items():
        if key in self.data_cache.columns:
          self.data_cache.loc[x.signature, key] = value

  def get_candidate_from_cache_by_signature(self, hash_id: int):
    index: int = self._data_cache.index.get_loc(hash_id)
    row = self.data_cache.iloc[index].to_dict()
    return self._set_candidate_attr(row)

  def get_candidate_from_cache_by_index(self, index: int):
    # return self.data_cache.iloc[index]['candidate']
    row = self.data_cache.iloc[index].to_dict()
    return self._set_candidate_attr(row)

  def _set_candidate_attr(self, attr: dict):
    x: CandidatePoint = CandidatePoint()
    for key, value in attr.items():
      setattr(x, key, value)
    return x

  def get_cache_candidate_points(self) -> List[CandidatePoint]:
    x = []
    for i, h_id in enumerate(self._data_cache.index):
      if i > self._last_index:
        break
      index: int = self._data_cache.index.get_loc(h_id)
      x.append(self._set_candidate_attr(
          self._data_cache.iloc[index].to_dict()))
    return x

  # def get_all_cache_points(
  #         self, nsamples: int = None, hmax: float = None) -> List[float]:
  #   x = []
  #   for i, h_id in enumerate(self._data_cache.index):
  #     x.append(self._data_cache.at[h_id, '_coords'])
  #   return x

  # def get_all_cache_points_f(
  #         self, nsamples: int = None, hmax: float = None) -> List[float]:
  #   x = []
  #   for i, h_id in enumerate(self._data_cache.index):
  #     x.append(self._data_cache.at[h_id, '_f'])
  #   return x if nsamples is None or nsamples > len(x) else x[-1:-nsamples]

  def get_all_cache_points(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = []
    if hmax is None:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_evaluated')
    else:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_evaluated & _h <= @hmax')
    for i, h_id in enumerate(df.index):
      x.append(df.at[h_id, '_coords'])
    return np.array(x) if nsamples is None or nsamples > len(x) else np.array(
        x[len(x) - nsamples:])

  def get_all_cache_points_f(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = []
    if hmax is None:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_evaluated')
    else:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_evaluated & _h <= @hmax')
    for i, h_id in enumerate(df.index):
      x.append(df.at[h_id, '_f'][0])
    return np.array(x) if nsamples is None or nsamples > len(x) else np.array(
        x[len(x) - nsamples:])

  def set_improving_candidate(self, x: CandidatePoint):
    index: int = self._data_cache.index.get_loc(x.signature)
    self.data_cache.iloc[index]["_improving"] = True

  def set_center_candidate(self, x: CandidatePoint):
    index: int = self._data_cache.index.get_loc(x.signature)
    self.data_cache.iloc[index]["_was_center"] = True

  def get_all_improving_candidates(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = []
    if hmax is None:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_improving')
    else:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_improving & _h <= @hmax')
    for i, h_id in enumerate(df.index):
      index: int = df.index.get_loc(h_id)
      x.append(self._set_candidate_attr(df.iloc[index].to_dict()))
    return x if nsamples is None or nsamples > len(x) else x[len(x) - nsamples:]

  def get_all_improving_points(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = []
    if hmax is None:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_improving')
    else:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_improving & _h <= @hmax')
    for i, h_id in enumerate(df.index):
      x.append(df.at[h_id, '_coords'])
    return np.array(x) if nsamples is None or nsamples > len(x) else np.array(
        x[len(x) - nsamples:])

  def get_all_improving_points_f(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = []
    if hmax is None:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_improving')
    else:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_improving & _h <= @hmax')
    for i, h_id in enumerate(df.index):
      x.append(df.at[h_id, '_f'][0])
    return np.array(x) if nsamples is None or nsamples > len(x) else np.array(
        x[len(x) - nsamples:])

  def get_all_non_improving_candidates(
          self, nsamples: int = None) -> List[CandidatePoint]:
    x = []
    df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
        '~_improving & ~_was_center')
    for i, h_id in enumerate(df.index):
      index: int = df.index.get_loc(h_id)
      x.append(self._set_candidate_attr(df.iloc[index].to_dict()))
    return x if nsamples is None or nsamples > len(x) else x[len(x) - nsamples:]

  def get_all_non_improving_points(
          self, nsamples: int = None) -> List[CandidatePoint]:
    x = []
    df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
        '~_improving & ~_was_center')
    for i, h_id in enumerate(df.index):
      x.append(df.at[h_id, '_coords'])
    return np.array(x) if nsamples is None or nsamples > len(x) else np.array(
        x[len(x) - nsamples:])

  def get_all_non_improving_points_f(
          self, nsamples: int = None) -> List[CandidatePoint]:
    x = []
    df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
        '~_improving & ~_was_center')
    for i, h_id in enumerate(df.index):
      x.append(df.at[h_id, '_f'][0])
    return np.array(x) if nsamples is None or nsamples > len(x) else np.array(
        x[len(x) - nsamples:])

  def get_all_center_candidates(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = []
    if hmax is None:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_was_center')
    else:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_was_center & _h <= @hmax')
    for i, h_id in enumerate(df.index):
      index: int = df.index.get_loc(h_id)
      x.append(self._set_candidate_attr(df.iloc[index].to_dict()))
    return x if nsamples is None or x[:nsamples] > len(x) else x[-1:-nsamples]

  def get_all_center_points(
          self, nsamples: int = None, hmax: float = None) -> List[float]:
    x = []
    if hmax is None:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_was_center')
    else:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_was_center & _h <= @hmax')
    for i, h_id in enumerate(df.index):
      x.append(df.at[h_id, '_coords'])
    return x if nsamples is None or nsamples > len(x) else x[-1:-nsamples]

  def get_all_center_points_f(
          self, nsamples: int = None, hmax: float = None) -> List[float]:
    x = []
    if hmax is None:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_was_center')
    else:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_was_center & _h <= @hmax')
    for i, h_id in enumerate(df.index):
      x.append(df.at[h_id, '_f'][0])
    return x if nsamples is None or nsamples > len(x) else x[-1:-nsamples]

  def get_all_center_points_h(
          self, nsamples: int = None, hmax: float = None) -> List[float]:
    x = []
    if hmax is None:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_was_center')
    else:
      df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
          '_was_center & _h <= @hmax')
    for i, h_id in enumerate(df.index):
      x.append(df.at[h_id, '_h'])
    return x if nsamples is None or nsamples > len(x) else x[-1:-nsamples]

  def get_all_primary_center_candidates(
          self, nsamples: int = None) -> List[CandidatePoint]:
    x = []
    s = DESIGN_STATUS.FEASIBLE
    df: pd.DataFrame = self._data_cache.iloc[0:self._last_index].query(
        '_was_center & status == @s')
    for i, h_id in enumerate(df.index):
      index: int = df.index.get_loc(h_id)
      x.append(self._set_candidate_attr(df.iloc[index].to_dict()))
    return x if nsamples is None or nsamples > len(x) else x[-1:-nsamples]

  def get_all_secondary_center_candidates(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = []
    s = DESIGN_STATUS.INFEASIBLE
    if hmax is None:
      df: pd.DataFrame = self._data_cache.iloc[0:self._last_index].query(
          '_was_center & status == @s')
    else:
      df: pd.DataFrame = self._data_cache.iloc[0:self._last_index].query(
          '_was_center & status == @s  & _h <= @hmax')
    for i, h_id in enumerate(df.index):
      index: int = df.index.get_loc(h_id)
      x.append(self._set_candidate_attr(df.iloc[index].to_dict()))
    return x if nsamples is None or nsamples > len(x) else x[-1:-nsamples]

  def get_all_nd_candidates(self) -> List[CandidatePoint]:
    x = []
    df: pd.DataFrame = self._data_cache.iloc[0: self._last_index].query(
        'is_nondominated')
    for i, h_id in enumerate(df.index):
      index: int = df.index.get_loc(h_id)
      x.append(self._set_candidate_attr(df.iloc[index].to_dict()))
    return x

  def get_all_feasible_nd_candidates(
          self, nsamples: int = None) -> List[CandidatePoint]:
    x = []
    s = DESIGN_STATUS.FEASIBLE
    df: pd.DataFrame = self._data_cache.iloc[0:self._last_index].query(
        'is_nondominated & status == @s')
    for i, h_id in enumerate(df.index):
      index: int = df.index.get_loc(h_id)
      x.append(self._set_candidate_attr(df.iloc[index].to_dict()))
    return x if nsamples is None or nsamples > len(x) else x[-1:-nsamples]

  def get_all_infeasible_nd_candidates(
          self, nsamples: int = None, hmax: float = None) -> List[CandidatePoint]:
    x = []
    s = DESIGN_STATUS.INFEASIBLE
    if hmax is None:
      df: pd.DataFrame = self._data_cache.iloc[0:self._last_index].query(
          'is_nondominated & status == @s')
    else:
      df: pd.DataFrame = self._data_cache.iloc[0:self._last_index].query(
          'is_nondominated & status == @s & _h <= @hmax')
    for i, h_id in enumerate(df.index):
      index: int = df.index.get_loc(h_id)
      x.append(self._set_candidate_attr(df.iloc[index].to_dict()))
    return x if nsamples is None or nsamples > len(x) else x[-1:-nsamples]

  def _extend_database(self):
    # Create new NaN rows
    new_data_rows = pd.DataFrame(
        np.nan, index=range(self._capacity),
        columns=self._data_cache.columns)

    # Extend the original DataFrame
    self._data_cache = pd.concat(
        [self._data_cache, new_data_rows],
        ignore_index=True)

  @property
  def capacity(self) -> int:
    return self._capacity

  @capacity.setter
  def capacity(self, value: int):
    self._capacity = value

  @property
  def data_cache(self) -> pd.DataFrame:
    """A getter of the cache memory dictionary

    :rtype: Dict
    """
    return self._data_cache

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
    return self._data_cache.shape[0]

  def is_duplicate(self, x: CandidatePoint, add: bool = True) -> bool:
    """Check if the point is in the cache memory

    :param x: Design point
    :type x: Point
    :return: A boolean flag indicate whether the point exist in the cache memory
    :rtype: bool
    """
    if x.signature is None or x.signature != hash(tuple((x._coords))):
      x.signature = hash(tuple((x._coords)))
    is_dup = x.signature in self._data_cache.index
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
    found = 0
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
      is_dup = xt.signature in self._data_cache.index

      if not is_dup:
        self.add_to_cache(xt)
      else:
        self.update_candidate_in_cache(xt)
