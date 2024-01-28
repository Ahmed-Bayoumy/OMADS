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
    :param _evaluated: Evaluation boolean
    :param _f: Objective function
    :param _freal: Realistic target value of the objective function
    :param _c_ineq: Inequality constraints
    :param _c_eq: Equality constraints
    :param _h: Aggregated constraints; active set
    :param _signature: hash signature; facilitate looking for duplicates and storing coordinates, hash signature, in the cache memory
    :param _dtype:  numpy double data type precision
  """
  # Dimension of the point
  _n: int = 0
  # Coordinates of the point
  _coords: List[float] = field(default_factory=list)
  # Coordinates definition boolean
  _defined: List[bool] = field(default_factory=lambda: [False])
  # Evaluation boolean
  _evaluated: bool = False
  # Objective function
  _f: float = inf
  _freal: float = inf
  # Inequality constraints
  _c_ineq: List[float] = field(default_factory=list)
  # Equality constraints
  _c_eq: List[float] = field(default_factory=list)
  # Aggregated constraints; active set
  _h: float = inf
  # hash signature; facilitate looking for duplicates and storing coordinates,
  # hash signature, in the cache memory
  _signature: int = 0
  # numpy double data type precision
  _dtype: DType = DType()
  # Variables type
  _var_type: List[int] = None
  # Discrete set
  _sets: Dict = None

  _var_link: List[str] = None

  _status: DESIGN_STATUS = DESIGN_STATUS.UNEVALUATED

  _constraints_type: List[BARRIER_TYPES] = None

  _is_EB_passed: bool = False

  _LAMBDA: List[float] = None
  _RHO: float = MPP.RHO.value

  _hmax: float = 1.

  _hmin: float = inf

  Eval_time: float = 0.

  source: str = "Current run"

  Model: str = "Simulation"

  _hzero: float = None

  @property
  def hzero(self):
    if self._hzero is None:
      return self._dtype.zero
    else:
      return self._hzero
  
  @hzero.setter
  def hzero(self, value: Any) -> Any:
    self._hzero = value
  

  @property
  def hmax(self) -> float:
    if self._hmax == 0.:
      return self._dtype.zero
    return self._hmax
  
  @hmax.setter
  def hmax(self, value: float):
    self._hmax = value
  

  @property
  def RHO(self) -> float:
    return self._RHO
  
  @RHO.setter
  def RHO(self, value: float):
    self._RHO = value
  
  @property
  def LAMBDA(self) -> float:
    return self._LAMBDA
  
  @LAMBDA.setter
  def LAMBDA(self, value: float):
    self._LAMBDA = value
  

  @property
  def var_link(self) -> Any:
    return self._var_link
  
  @var_link.setter
  def var_link(self, value: Any):
    self._var_link = value
  
  @property
  def status(self) -> DESIGN_STATUS:
    return self._status
  
  @status.setter
  def status(self, value: DESIGN_STATUS):
    self._status = value
  
  @property
  def is_EB_passed(self) -> bool:
    return self._is_EB_passed
  
  @is_EB_passed.setter
  def is_EB_passed(self, value: bool):
    self._is_EB_passed = value
  
  @property
  def var_type(self) -> List[int]:
    return self._var_type
  
  @var_type.setter
  def var_type(self, value: List[int]):
    self._var_type = value
  
  @property
  def constraints_type(self) -> List[BARRIER_TYPES]:
    return self._constraints_type
  
  @constraints_type.setter
  def constraints_type(self, value: List[BARRIER_TYPES]):
    self._constraints_type = value
  

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
  def evaluated(self):
    return self._evaluated

  @evaluated.setter
  def evaluated(self, other: bool):
    self._evaluated = other

  @property
  def signature(self):
    return self._signature

  @property
  def n_dimensions(self):
    return self._n

  @n_dimensions.setter
  def n_dimensions(self, n: int):
    if n < 0:
      del self.n_dimensions
      if len(self.coordinates) > 0:
        del self.coordinates
      if self.defined:
        del self.defined
    else:
      self._n = n

  @n_dimensions.deleter
  def n_dimensions(self):
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
    if self.n_dimensions > 0:
      return any(self.defined)
    else:
      return False

  @property
  def f(self):
    return self._f

  @f.setter
  def f(self, val: float):
    self._f = val

  @f.deleter
  def f(self):
    del self._f

  @property
  def fobj(self):
    return self._freal

  @fobj.setter
  def fobj(self, other: float):
    self._freal = other

  @property
  def hmin(self):
    return self._hmin

  @hmin.setter
  def hmin(self, other: float):
    self._hmin = other

  @property
  def c_ineq(self):
    return self._c_ineq

  @c_ineq.setter
  def c_ineq(self, vals: List[float]):
    self._c_ineq = vals

  @c_ineq.deleter
  def c_ineq(self):
    del self._c_ineq

  @property
  def c_eq(self):
    return self._c_eq

  @c_eq.setter
  def c_eq(self, other: List[float]):
    self._c_eq = other

  @property
  def h(self):
    return self._h

  @h.setter
  def h(self, val: float):
    self._h = val

  @h.deleter
  def h(self):
    del self._h

  def reset(self, n: int = 0, d: Optional[float] = None):
    """ Sets all coordinates to d. """
    if n <= 0:
      self._n = 0
      del self.coordinates
    else:
      if self._n != n:
        del self.coordinates
        self.n_dimensions = n
      self.coordinates = [d] * n if d is not None else []

  def __eq__(self, other) -> bool:
    return self.n_dimensions is other.n_dimensions and other.coordinates is self.coordinates \
         and self.is_any_defined() is other.is_any_defined() \
         and self.f is other.f and self.h is other.h

  def __lt__(self, other):
    return (other.h > (self.hmax if self._is_EB_passed else self._dtype.zero) > self.__dh__(other=other)) or \
         (((self.hmax if self._is_EB_passed else self._dtype.zero) >= self.h >= 0.0) and
        self.__df__(other=other) < 0)

  def __le__(self, other):
    return self.__eq_f__(other) or self.f == other.f

  def __gt__(self, other):
    return not self.__lt__(other=other)

  def __str__(self) -> str:
    return f'{self.coordinates}'

  def __sub__(self, other) -> List[float]:
    dcoord: List[float] = []
    for k in range(self.n_dimensions):
      dcoord.append(subtract(self.coordinates[k],
                   other.coordinates[k], dtype=self._dtype.dtype))
    return dcoord

  def __add__(self, other) -> List[float]:
    dcoord: List[float] = []
    for k in range(self.n_dimensions):
      dcoord.append(add(self.coordinates[k], other.coordinates[k], dtype=self._dtype.dtype))
    return dcoord

  def __truediv__(self, s: float):
    return np.divide(self.coordinates, s, dtype=self._dtype.dtype)

  def __dominate__(self, other) -> bool:
    """ x dominates y, if f(x)< f(y) """
    if self.__le__(other):
      return True
    return False

  def __eval__(self, bb_output):
    """ Evaluate point """
    """ Objective function """
    self.f = bb_output[0]
    self.fobj = bb_output[0]
    """ Inequality constraints (can be an empty vector) """
    self.c_ineq = bb_output[1]
    if not isinstance(self.c_ineq, list):
      self.c_ineq = [self.c_ineq]
    self.evaluated = True
    """ Check the multiplier matrix """
    if self.LAMBDA is None:
      self.LAMBDA = []
      for _ in range(len(self.c_ineq)):
        self.LAMBDA.append(MPP.LAMBDA.value)
    else:
      if len(self.c_ineq) != len(self.LAMBDA):
        for _ in range(len(self.LAMBDA), len(self.c_ineq)):
          self.LAMBDA.append(MPP.LAMBDA.value)
    """ Check and adapt the barriers matrix"""
    if self.constraints_type is not None:
      if len(self.c_ineq) != len(self.constraints_type):
        if len(self.c_ineq) > len(self.constraints_type):
          for _ in range(len(self.constraints_type), len(self.c_ineq)):
            self.constraints_type.append(BARRIER_TYPES.EB)
        else:
          for i in range(len(self.c_ineq), len(self.constraints_type)):
            del self.constraints_type[-1]
    else:
      self.constraints_type = []
      for _ in range(len(self.c_ineq)):
        self.constraints_type.append(BARRIER_TYPES.EB)
    """ Check if all extreme barriers are satisfied """
    cEB = []
    cPB = []
    self.cPB = []
    for i in range(len(self.c_ineq)):
      if self.constraints_type[i] == BARRIER_TYPES.EB:
        cEB.append(self.c_ineq[i])
      else:
        cPB.append(self.c_ineq[i])
    if isinstance(cEB, list) and len(cEB) >= 1:
      hEB = sum(power(maximum(cEB, self._dtype.zero,
                   dtype=self._dtype.dtype), 2, dtype=self._dtype.dtype))
    else:
      hEB = self._dtype.zero
    if isinstance(cPB, list) and len(cPB) >= 1:
      hPB = sum(power(maximum(cPB, self._dtype.zero,
                   dtype=self._dtype.dtype), 2, dtype=self._dtype.dtype))
      self.cPB = cPB
    else:
      hPB = self._dtype.zero
    if hEB <= self.hzero:
      self.is_EB_passed = True
      if hPB > self.hzero:
        self.status = DESIGN_STATUS.INFEASIBLE
      self.h = copy.deepcopy(hPB)
      if hPB < self.hmax:
        self.hmax = copy.deepcopy(hPB)
    else:
      self.is_EB_passed = False
      self.status = DESIGN_STATUS.INFEASIBLE
      self.h = copy.deepcopy(hEB)
      self.__penalize__(extreme= True)
      return
    """ Aggregate all constraints """
    # self.h = sum(power(maximum(self.c_ineq, self._dtype.zero,
    #                dtype=self._dtype.dtype), 2, dtype=self._dtype.dtype))
    if np.isnan(self.h) or np.any(np.isnan(self.c_ineq)):
      self.h = inf
      self.status = DESIGN_STATUS.ERROR

    """ Penalize relaxable constraints violation """
    if np.isnan(self.f) or self.h > self.hzero:
      if self.h > np.round(self.hmax, 2):
        self.__penalize__(extreme=False)
      self.status = DESIGN_STATUS.INFEASIBLE
    else:
      self.hmax = copy.deepcopy(self.h)
      self.status = DESIGN_STATUS.FEASIBLE

  def __penalize__(self, extreme: bool=True):
    if len(self.cPB) > len(self.LAMBDA):
      self.LAMBDA += [self.LAMBDA[-1]] * abs(len(self.LAMBDA)-len(self.cPB))
    if 0 < len(self.cPB) < len(self.LAMBDA):
      del self.LAMBDA[len(self.cPB):]
    if extreme:
      self.f = inf
      self.hmin = inf
    else:
      self.hmin = np.dot(self.LAMBDA, self.cPB) + ((1/(2*self.RHO)) * self.h if self.RHO > 0. else np.inf)
      self.f = self.fobj + self.hmin

  def __is_duplicate__(self, other) -> bool:
    return other.signature is self._signature

  def __eq_f__(self, other):
    return self.__df__(other=other) < self._dtype.zero

  def __eq_h__(self, other):
    return self.__dh__(other=other) < self._dtype.zero

  def __df__(self, other):
    return subtract(self.f, other.f, dtype=self._dtype.dtype)

  def __dh__(self, other):
    return subtract(self.h, other.h, dtype=self._dtype.dtype)

