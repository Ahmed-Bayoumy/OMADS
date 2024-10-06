import copy
from dataclasses import dataclass, field
import operator
from typing import List, Dict, Any, Optional
import numpy as np
from .CandidatePoint import CandidatePoint
from ._globals import DESIGN_STATUS

@dataclass
class Cache:
  """ In computing, a hash table (hash map) is a data structure that implements an associative array abstract data type, a structure that can map keys to values. A hash table uses a hash function to compute an index, also called a hash code, into an array of buckets or slots, from which the desired value can be found. During lookup, the key is hashed and the resulting hash indicates where the corresponding value is stored."""
  _hash_id: List[int] = field(default_factory=list)
  _best_hash_id: List[int] = field(default_factory=list)
  _cache_dict: Dict[Any, Any] = field(default_factory=lambda: {})
  _n_dim: int = 0
  _is_pareto: bool = False
  nd_points: Optional[List[CandidatePoint]] = None

  @property
  def cache_dict(self)->Dict:
    """A getter of the cache memory dictionary

    :rtype: Dict
    """
    return self._cache_dict

  @property
  def hash_id(self)->List[int]:
    """A getter to return the list of hash IDs

    :rtype: List[int]
    """
    return self._hash_id

  @hash_id.setter
  def hash_id(self, other: CandidatePoint):
    if hash(tuple(other.coordinates)) not in self._hash_id:
      self._hash_id.append(hash(tuple(other.coordinates)))
  
  @property
  def best_hash_id(self)->List[int]:
    """A getter to return the list of hash IDs

    :rtype: List[int]
    """
    return self._best_hash_id

  @best_hash_id.setter
  def best_hash_id(self, id: int):
    self._best_hash_id.append(id)

  @property
  def size(self)->int:
    """A getter of the size of the hash ID list

    :rtype: int
    """
    return len(self.hash_id)

  def is_duplicate(self, x: CandidatePoint) -> bool:
    """Check if the point is in the cache memory

    :param x: Design point
    :type x: Point
    :return: A boolean flag indicate whether the point exist in the cache memory
    :rtype: bool
    """
    is_dup = x.signature in self.hash_id
    if not is_dup:
      self.add_to_cache(x)

    return is_dup

  def get_index(self, x: CandidatePoint)->int:
    """Get the index of hash value, associated with the point x, if that point was saved in the cach memory

    :param x: Input point
    :type x: Point
    :return: The index of the point in the hash ID list
    :rtype: int
    """
    hash_value: int = hash(tuple(x.coordinates))
    if hash_value in self.hash_id:
      return self.hash_id.index(hash_value)
    return -1

  def add_to_cache(self, x: CandidatePoint):
    """Save the point x to the cache memory

    :param x: Evaluated point to be saved in the cache memory
    :type x: Point
    """
    if not isinstance(x, list):
      hash_value: int = hash(tuple(x.coordinates))
      self._cache_dict[hash_value] = x
      self._hash_id.append(hash(tuple(x.coordinates)))
    else:
      for i in range(len(x)):
        hash_value: int = hash(tuple(x[i].coordinates))
        self._cache_dict[hash_value] = x[i]
        self._hash_id.append(hash(tuple(x[i].coordinates)))
    
  
  def add_to_best_cache(self, x: CandidatePoint):
    if not self._is_pareto:
      if x.signature in self._best_hash_id:
        return
      if len(self._best_hash_id) <= 0 and len(self._cache_dict) >= 1:
        self._best_hash_id.append(list(self.cache_dict.keys())[0])
      if not isinstance(x, list):
        if len(self._cache_dict) > 1:
          is_infeas_dom: bool = (x.status == DESIGN_STATUS.INFEASIBLE and (x.h < self._cache_dict[self._best_hash_id[-1]].h) )
          is_feas_dom: bool = (x.status == DESIGN_STATUS.FEASIBLE and x.fobj < self._cache_dict[self._best_hash_id[-1]].fobj)
        else:
          is_infeas_dom: bool = False
          is_feas_dom: bool = False
        if is_infeas_dom or is_feas_dom:
          self._n_dim = len(x.coordinates)
          self._best_hash_id.append(x.signature)
      else:
        for i in range(len(x)):
          is_infeas_dom: bool = (x[i].status == DESIGN_STATUS.INFEASIBLE and (x[i].h < self._cache_dict[self._best_hash_id[0]].h) )
          is_feas_dom: bool = (x[i].status == DESIGN_STATUS.FEASIBLE and x[i].fobj < self._cache_dict[self._best_hash_id[0]].fobj)
          if len(self._cache_dict) == 1 or is_infeas_dom or is_feas_dom:
            self._n_dim = len(x[i].coordinates)
            self._best_hash_id.append(self._hash_id[-1])
    else:
      self.nd_points = copy.deepcopy(x)
      self._best_hash_id = []
      for i in range(len(self.nd_points)):
        self._best_hash_id.append(self.nd_points[i].signature)
  
  def get_best_cache_points(self, nsamples):
    """ Get best points """
    temp = np.zeros((nsamples, self._n_dim))
    index = 0
    if not self._is_pareto:
      cache_temp = dict(sorted(self._cache_dict.items(), key=operator.itemgetter(1)))

      for k in cache_temp:
        if index < len(temp):
          temp[index, :] = cache_temp[k].coordinates
          index += 1
        else:
          break
    else:
      for k in self.nd_points:
        # if index < len(temp):
        temp[index, :] = k.coordinates
        index += 1
        # else:
        #   break
    
    return temp
  
  def get_cache_points(self):
    """ Get best points """
    temp = np.zeros((len(self._hash_id)-1, self._n_dim))
    for i in range(1, len(self._hash_id)):
      temp[i-1, :] = self._cache_dict[self._hash_id[i]].coordinates
    return temp
  
  def get_point(self, key):
    return self._cache_dict[key]
