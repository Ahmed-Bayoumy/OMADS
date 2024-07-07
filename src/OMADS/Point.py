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
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from numpy import sum, subtract, add, maximum, power, inf
import numpy as np
from ._globals import *

@dataclass
class Point:
  """ A class for the poll point
    
    :param _n: # Dimension of the point
    :param _coords: Coordinates of the point
    :param _defined: Coordinates definition boolean
    :param _signature: hash signature; facilitate looking for duplicates and storing coordinates, hash signature, in the cache memory
    :param _dtype:  numpy double data type precision
  """
  # Dimension of the point
  _n: int = 0
  # Coordinates of the point
  _coords: List[float] = None
  # Coordinates definition boolean
  _defined: List[bool] = None
  # Evaluation boolean
  _evaluated: bool = False
  # hash signature, in the cache memory
  _signature: int = 0
  # numpy double data type precision
  _dtype: DType = None
  # Variables type
  _var_type: List[int] = None
  # Discrete set
  _sets: Dict = None

  source: str = "Current run"

  Model: str = "Simulation"

  def __post_init__(self):
    self._dtype = DType()
  
  @property
  def var_type(self) -> List[int]:
    return self._var_type
  
  @var_type.setter
  def var_type(self, value: List[int]):
    self._var_type = value
  

  @property
  def sets(self):
    return self._sets
  
  @sets.setter
  def sets(self, value: Any) -> Any:
    self._sets = value

  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, other: DType):
    self._dtype = other

  @property
  def signature(self):
    return self._signature

  @property
  def size(self):
    return self._n

  @size.setter
  def size(self, n: int):
    if n < 0:
      del self.size
      if len(self.coordinates) > 0:
        del self.coordinates
      if self.defined:
        del self.defined
    else:
      self._n = n

  @size.deleter
  def size(self):
    self._n = 0

  @property
  def coordinates(self):
    """Get the coordinates of the point."""
    return self._coords

  @coordinates.setter
  def coordinates(self, coords: List[float]):
    """ Get the coordinates of the point. """
    self._n = len(coords)
    self._coords = list(coords)
    self._signature = hash(tuple(self._coords))
    self._defined = [True] * self._n

  @coordinates.deleter
  def coordinates(self):
    del self._coords
  
  def fill(self, val: Any):
    if isinstance(val, list):
      self.coordinates = val
    else:
      self.coordinates = [val]*self._n
  
  def push_back(self, val: Any):
    if isinstance(val, list):
      self.coordinates = self._coords + val
    else:
      self.coordinates = self._coords + [val]*self._n
  
  def checkForGranularity(self, g: Any, name: str) -> bool:
    for i in range(self._n):
      if not self.isMult(self.coordinates[i], g[i]):
        raise IOError("Check: Invalid granularity of parameter " + name + f"at index {i} : {self.coordinates[i]} vs granularity value {g[i]} found a non-zero remainder of {self.coordinates[i] % g[i]}.")

    return True

  @property
  def defined(self) -> List[bool]:
    return self._defined

  @defined.setter
  def defined(self, value: List[bool]):
    self._defined = copy.deepcopy(value)

  @defined.deleter
  def defined(self):
    del self._defined

  def is_any_defined(self) -> bool:
    """Check if at least one coordinate is defined."""
    if self.size > 0:
      return any(self.defined)
    else:
      return False
  
  def is_all_defined(self) -> bool:
    """Check if at least one coordinate is defined."""
    if self.size > 0:
      return all(self.defined)
    else:
      return False
  
  def is_complete(self) -> bool:
    if self.is_all_defined() and self.size > 0:
      return True
    return False

  def reset(self, n: int = 0, d: Optional[float] = 0):
    """ Sets all coordinates to d. """
    if n <= 0:
      self._n = 0
      del self.coordinates
    else:
      if self._n != n:
        del self.coordinates
        self.size = n
      self.coordinates = [d] * n if d is not None else []
      if d != 0:
        self.defined = [True] * n
      else:
        self.defined = [False] * n
      
  
  def nextMult(self, g: float = None, i: int = 0) -> float:
    d: float
    # Calculate the remainder when number is divided by multiple_of
    # Calculate the ratio to find next multiple_of
    # ratio = math.ceil(g / self.coordinates[i])
    
    # # Calculate the next multiple_of
    # next_multiple = ratio * self.coordinates[i]
    value = self.coordinates[i]
    if g is None or not self.defined[i] or g <= 0. or self.isMult(value, g):
      d = value
    else:
      # granularity > 0, and _value is not a multiple of granularity.
      # Adjust value with granularity
      granMult = round(abs(value)/g)
      if value > 0:
        granMult += 1
      # if abs(value) > 0:
      #   granMult += granMult
      d = granMult*g

      if not self.isMult(d, g):
        raise IOError("nextMult(gran): cannot get a multiple of granularity")
    # trials = 0
    # while (not self.isMult(d, g)):
    # # if :
    #   d = d + (d % g)
    #   trials+=1
    #   if trials > 10:
    #     raise IOError("nextMult(gran): cannot get a multiple of granularity")
    # if value < 0:
    #   d *= -1
    
    return d
  
  def previousMult(self, g: float, i: int):
    d: float
    if g is not None or not self.is_all_defined() or g <= 0. or self.isMult(self.coordinates[i], g):
      d = self.coordinates[i]
    else:
      granMult: int = int(self.coordinates[i]/g)
      if self.coordinates[i] < 0:
        granMult-= 1
      bigGranExp: int = 10 ** self.nDecimals(g)
      bigGran: int = int(g*bigGranExp)
      d = granMult * bigGran/bigGranExp
    return d
  
  def isMult(self, v1: float, v2: float):
    isMult: bool = True
    if abs(v1) <= self.dtype.zero:
      isMult = True
    elif (abs(v2) > 0):
      mult = round(v1/v2)
      verif_value = mult * v2
      if abs(v1-verif_value) < abs(mult)*self.dtype.zero:
        isMult = True
      
    elif v2 < 0:
      isMult = False
    else:
      isMult = True

    return isMult

    # return ((v1%v2) <= self.dtype.zero) if v2 > 0.0 else True
  
  def nDecimals(self, n: float):
    return len(n.rsplit('.')[-1]) if '.' in n else 0


  def __eq__(self, other) -> bool:
    return self.size is other.size and other.coordinates is self.coordinates \
         and self.is_any_defined() is other.is_any_defined()
  
  def __le__(self, other) -> bool:
    if self.size is other._n and self.is_all_defined() is other.is_all_defined():
      return all(self.coordinates[i] <= other.coordinates[i] for i in range(self._n))
    else:
      return None
  
  def __lt__(self, other) -> bool:
    if self.size is other._n and self.is_all_defined() is other.is_all_defined():
      return all(self.coordinates[i] < other.coordinates[i] for i in range(self._n))
    else:
      return None

  def __str__(self) -> str:
    return f'{self.coordinates}'

  def __sub__(self, other) -> List[float]:
    dcoord: List[float] = []
    for k in range(self.size):
      dcoord.append(subtract(self.coordinates[k],
                   other.coordinates[k], dtype=self._dtype.dtype))
    return dcoord

  def __add__(self, other) -> List[float]:
    dcoord: List[float] = []
    for k in range(self.size):
      dcoord.append(add(self.coordinates[k], other.coordinates[k], dtype=self._dtype.dtype))
    return dcoord

  def __truediv__(self, s: float):
    return np.divide(self.coordinates, s, dtype=self._dtype.dtype)

  def __is_duplicate__(self, other) -> bool:
    return other.signature is self._signature
  
  def __getitem__(self, idx: int):
    return self.coordinates[idx]

  def __setitem__(self, idx, value):
    self.coordinates[idx] = value
    self.defined[idx] = True
