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
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from numpy import subtract, add, maximum, power, inf
import numpy as np


from ._globals import DType, BARRIER_TYPES, MPP, DESIGN_STATUS, COMPARE_TYPE
from .gmesh import Gmesh
from .point import Point


@dataclass
class CandidatePoint:
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
    :param _signature: hash signature; facilitate looking for duplicates 
    and storing coordinates, hash signature, in the cache memory
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
  _f: List[float] = field(default_factory=lambda: [inf])
  _freal: List[float] = field(default_factory=lambda: [inf])
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
  _dtype: Optional[DType] = None
  # Variables type
  _var_type: Optional[List[int]] = None
  # Discrete set
  _sets: Optional[Dict] = None

  _var_link: Optional[List[str]] = None

  _status: DESIGN_STATUS = DESIGN_STATUS.UNEVALUATED

  _constraints_type: Optional[List[BARRIER_TYPES]] = None

  _is_eb_passed: bool = False

  _lambda: Optional[List[float]] = None
  _rho: float = MPP.RHO.value

  _hmax: float = 1.

  _hmin: float = inf

  eval_time: float = 0.

  source: str = "Current run"

  model: str = "Simulation"

  _hzero: Optional[float] = None

  _mesh: Optional[Gmesh] = None

  _direction: Optional[Point] = None

  _fs: Optional[Point] = None

  eval_no: int = 0

  _incumbent_signature: int = None

  cpb: List[float] = None

  def __post_init__(self, f=None, h=None, coords=None):
    if f is not None:
      if len(f) == 0:
        raise ValueError("Objective function cannot be empty")
      self.f = copy.deepcopy(f)
    if h is not None:
      if h < 0:
        raise ValueError("h value cannot be negative")
      self.h = copy.deepcopy(h)
    if coords is not None:
      self.coordinates = copy.deepcopy(coords)
    self._dtype = DType()

  # def __post_init__(self):

  @property
  def n(self):
    return self._n

  @n.setter
  def n(self, value: int) -> int:
    self._n = value

  @property
  def is_extreme_barrier_passed(self) -> bool:
    """Check whether constraints violation function passes the extreme violation margin threshold

    :return: Whether passed an extreme barrier
    :rtype: booleane_
    """
    return self._is_eb_passed

  @property
  def incumbent_signature(self) -> int:
    """The hash ID of the incumbent candidate solution used to generate the current point

    :return: Incumbent hash ID
    :rtype: int
    """
    return self._incumbent_signature

  @incumbent_signature.setter
  def incumbent_signature(self, value: int) -> Any:
    self._incumbent_signature = value

  @property
  def fs(self) -> Point:
    """Point coordinates on the functions space

    :return: Functions Point
    :rtype: Point
    """
    if self._fs is None:
      self._fs = Point(len(self.f))
      self._fs.coordinates = self.f
    return self._fs

  @fs.setter
  def fs(self, value: Point) -> Any:
    self._fs = value

  @property
  def mesh(self) -> Gmesh:
    """The mesh settings the current point has been generated on

    :return: A mesh instant
    :rtype: Gmesh
    """
    return self._mesh

  @mesh.setter
  def mesh(self, value: Any) -> Any:
    self._mesh = value

  @property
  def direction(self) -> Point:
    """Point instant of the direction from the incumbent to the current point

    :return: Direction Point
    :rtype: Point
    """
    return self._direction

  @direction.setter
  def direction(self, value: Any) -> Any:
    self._direction = value

  @property
  def hzero(self) -> float:
    """Violation margin

    :return: _description_
    :rtype: _type_
    """
    if self._hzero is None:
      return self._dtype.zero
    else:
      return self._hzero

  @hzero.setter
  def hzero(self, value: Any) -> Any:
    self._hzero = value

  @property
  def h_max(self) -> float:
    """Maximum constraints violation

    :return: The maximum value of the evaluated constraints violation function
    :rtype: float
    """
    if np.isclose(self._hmax, 0., rtol=1e-09, atol=1e-09):
      return self._dtype.zero
    return self._hmax

  @h_max.setter
  def h_max(self, value: float):
    self._hmax = value

  @property
  def rho(self) -> float:
    """Mesh density

    :return: mesh parameter
    :rtype: float
    """
    # TODO: Remove abd get it from the mesh instant saved already
    return self._rho

  @rho.setter
  def rho(self, value: float):
    self._rho = value

  @property
  def lambda_multipliers(self) -> List[float]:
    """Multipliers values

    :return: LMs values
    :rtype: List[float]
    """
    return self._lambda

  @lambda_multipliers.setter
  def lambda_multipliers(self, value: float):
    self._lambda = value

  @property
  def var_link(self) -> List[str]:
    """Linked sets or components to the variables configuration

    :return: List of component names
    :rtype: List[str]
    """
    return self._var_link

  @var_link.setter
  def var_link(self, value: Any):
    self._var_link = value

  @property
  def status(self) -> DESIGN_STATUS:
    """Returns the design status of the current candidate solution

    :return: Design status (feasible, infeasible, error, or pending)
    :rtype: DESIGN_STATUS
    """
    return self._status

  @status.setter
  def status(self, value: DESIGN_STATUS):
    self._status = value

  @is_extreme_barrier_passed.setter
  def is_extreme_barrier_passed(self, value: bool):
    self._is_eb_passed = value

  @property
  def var_type(self) -> List[int]:
    """List of variables type

    :return: List of variables type
    :rtype: List[int]
    """
    return self._var_type

  @var_type.setter
  def var_type(self, value: List[int]):
    self._var_type = value

  @property
  def constraints_type(self) -> List[BARRIER_TYPES]:
    """List of constraints barrier type

    :return: List of constraints barrier type
    :rtype: List[BARRIER_TYPES]
    """
    return self._constraints_type

  @constraints_type.setter
  def constraints_type(self, value: List[BARRIER_TYPES]):
    self._constraints_type = value

  @property
  def sets(self) -> Dict:
    """Dictionary of the discrete sets used in design configurations 
    associated with the current candidate solution

    :return: Dictionary of the discrete sets 
    :rtype: Dict
    """
    return self._sets

  @sets.setter
  def sets(self, value: Any) -> Any:
    self._sets = value

  @property
  def dtype(self) -> DType:
    """Numerical precision type

    :return: Numerical precision type
    :rtype: DType
    """
    return self._dtype

  @dtype.setter
  def dtype(self, other: DType):
    self._dtype = other

  @property
  def evaluated(self) -> bool:
    """Check whther the current point has been evaluated

    :return: Check whther the current point has been evaluated
    :rtype: boolean
    """
    return self._evaluated

  @evaluated.setter
  def evaluated(self, other: bool):
    self._evaluated = other

  @property
  def signature(self) -> int:
    """Hash ID of the current candidate solution

    :return: Hash ID of the current candidate solution
    :rtype: int
    """
    return self._signature

  @property
  def n_dimensions(self) -> int:
    """Number of dimensions

    :return: Number of design variables
    :rtype: int
    """
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
    """Set of booleans that verify that the point is well defined

    :return: Set of booleans that verify that the point is well defined

    :rtype: List[bool]
    """
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
  def f(self) -> List[float]:
    """List of objective functions values

    :return: List of objective functions values
    :rtype: List[float]
    """
    return self._f

  @f.setter
  def f(self, val: Any):
    if isinstance(val, list):
      self._f = val
    else:
      self._f = [val]

  @f.deleter
  def f(self):
    del self._f

  @property
  def fobj(self) -> List[float]:
    """List of objective functions values

    :return: List of evaluated objective functions
    :rtype: List[float]
    """
    return self._freal

  @fobj.setter
  def fobj(self, other: Any):
    if isinstance(other, list):
      self._freal = other
    else:
      self._freal = [other]

    if self.fs is None or len(
            self._fs.coordinates) < 0 or not isinstance(
            self._freal, list):
      self.fs = Point(len(self._freal))
      self.fs.coordinates = self._freal
    else:
      self.fs.coordinates = self._freal

  @property
  def hmin(self) -> float:
    """Minimum constraints violation

    :return: min constraints violation
    :rtype: float
    """
    return self._hmin

  @hmin.setter
  def hmin(self, other: float):
    self._hmin = other

  @property
  def c_ineq(self) -> List[float]:
    """List of inequality constraints

    :return: List of values of inequality constraints
    :rtype: List[float]
    """
    return self._c_ineq

  @c_ineq.setter
  def c_ineq(self, vals: List[float]):
    self._c_ineq = vals

  @c_ineq.deleter
  def c_ineq(self):
    del self._c_ineq

  @property
  def c_eq(self) -> List[float]:
    """List of equality constraints

    :return: List of values for evaluated list of constraints
    :rtype: List[float]
    """
    return self._c_eq

  @c_eq.setter
  def c_eq(self, other: List[float]):
    self._c_eq = other

  @property
  def h(self):
    """Constrains violation function

    :return: The value of the constraints violation function
    :rtype: float
    """
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
    """Equality check: satisfied if current candidate (self) is equal to other."""
    return self.n_dimensions is other.n_dimensions \
        and self.is_any_defined() is other.is_any_defined() \
        and self.__compare__(other) == COMPARE_TYPE.EQUAL

  def __lt__(self, other):
    """Strict dominance: satisfied if current candidate (self) strictly dominates other."""
    if len(self.f) != len(other.f):
      return False
    strict_dom = False
    if self.is_feasible() and other.is_feasible():
      strict_dom = all(f1 < f2 for f1, f2 in zip(self.f, other.f))
    elif not self.is_feasible() and not other.is_feasible():
      strict_dom = all(f1 < f2 for f1, f2 in zip(
          self.f, other.f)) and self.h < other.h
    return strict_dom
    # return (other.h > (self.h_max if self._is_EB_passed else ..
    # self._dtype.zero) > self.__dh__(other=other)) or \
    #      (((self.h_max if self._is_EB_passed else self._dtype.zero) >= self.h >= 0.0) and
    #     max(self.__df__(other=other)) < 0)

  def __le__(self, other):
    """Weak dominance: satisfied if current candidate (self) is equal or dominating other."""
    return self.__compare__(other) in [COMPARE_TYPE.EQUAL, COMPARE_TYPE.DOMINATING]

  def __gt__(self, other):
    return not self.__lt__(other=other)

  def __str__(self) -> str:
    """For printing the candidate meta information."""
    return f'f = {self.f}, h = {self.h}, x= {self.coordinates}'

  def __sub__(self, other) -> List[float]:
    dcoord: List[float] = []
    for k in range(self.n_dimensions):
      dcoord.append(subtract(self.coordinates[k],
                             other.coordinates[k], dtype=self._dtype.dtype))
    return dcoord

  def __add__(self, other) -> List[float]:
    dcoord: List[float] = []
    for k in range(self.n_dimensions):
      dcoord.append(
          add(
              self.coordinates[k],
              other.coordinates[k],
              dtype=self._dtype.dtype))
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
    # """ Objective function """
    self.f = bb_output[0]
    self.fobj = bb_output[0]
    # """ Inequality constraints (can be an empty vector) """
    self.c_ineq = bb_output[1]
    if not isinstance(self.c_ineq, list):
      self.c_ineq = [self.c_ineq]
    self.evaluated = True
    # """ Check the multiplier matrix """
    if self.lambda_multipliers is None:
      self.lambda_multipliers = []
      for _ in range(len(self.c_ineq)):
        self.lambda_multipliers.append(MPP.LAMBDA.value)
    else:
      if len(self.c_ineq) != len(self.lambda_multipliers):
        for _ in range(len(self.lambda_multipliers), len(self.c_ineq)):
          self.lambda_multipliers.append(MPP.LAMBDA.value)
    # """ Check and adapt the barriers matrix"""
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
    # """ Check if all extreme barriers are satisfied """
    ceb = []
    cpb = []
    self.cpb = []
    for i, _ in enumerate((self.c_ineq)):
      if self.constraints_type[i] == BARRIER_TYPES.EB:
        ceb.append(self.c_ineq[i])
      else:
        cpb.append(self.c_ineq[i])
    if isinstance(ceb, list) and len(ceb) >= 1:
      ceb_max = maximum(ceb, self._dtype.zero, dtype=self._dtype.dtype)
      if max(ceb_max) != self._dtype.zero:
        heb = np.sum(power(ceb_max, 2, dtype=self._dtype.dtype))
      else:
        heb = self._dtype.zero
    else:
      heb = self._dtype.zero
    if isinstance(cpb, list) and len(cpb) >= 1:
      cpbmax = maximum(cpb, self._dtype.zero, dtype=self._dtype.dtype)
      if max(cpbmax) != self._dtype.zero:
        hpb = np.sum(power(cpbmax, 2, dtype=self._dtype.dtype))
      else:
        hpb = self._dtype.zero
      self.cpb = cpb
    else:
      hpb = self._dtype.zero
    if heb <= self.hzero:
      self.is_extreme_barrier_passed = True
      if hpb > self.hzero:
        self.status = DESIGN_STATUS.INFEASIBLE
      self.h = copy.deepcopy(hpb)
      if hpb < self.h_max:
        self.h_max = copy.deepcopy(hpb)
    else:
      self.is_extreme_barrier_passed = False
      self.status = DESIGN_STATUS.INFEASIBLE
      self.h = copy.deepcopy(heb)
      self.__penalize__(extreme=True)
      return
    # """ Aggregate all constraints """
    if np.isnan(self.h) or np.any(np.isnan(self.c_ineq)):
      self.h = inf
      self.status = DESIGN_STATUS.ERROR

    # """ Penalize relaxable constraints violation """
    if any(np.isnan(self.f)):
      for fi, _ in enumerate((self.f)):
        if np.isnan(self.f[fi]):
          self.f[fi] = np.inf
          self.fobj[fi] = np.inf

    if self.h > self.hzero:
      if self.h > np.round(self.h_max, 2):
        self.__penalize__(extreme=False)
      self.status = DESIGN_STATUS.INFEASIBLE
    else:
      self.h_max = copy.deepcopy(self.h)
      self.status = DESIGN_STATUS.FEASIBLE

  def __penalize__(self, extreme: bool = True):
    if len(self.cpb) > len(self.lambda_multipliers):
      self.lambda_multipliers += [self.lambda_multipliers[-1]
                                  ] * abs(len(self.lambda_multipliers)-len(self.cpb))
    if 0 < len(self.cpb) < len(self.lambda_multipliers):
      del self.lambda_multipliers[len(self.cpb):]
    if extreme:
      self.hmin = inf
    else:
      # np.dot(self.lambda_multipliers, self.cPB) + ((1/(2*self.rho)) ..
      # * self.h if self.rho > 0. else np.inf)
      self.hmin = self.h
      self.f = [
          self.fobj[i] * (1. / len(self.fobj)) + self.hmin
          for i in range(len(self.fobj))]

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

  def __compare__(self, other):
    """Return dominance comparison between two candidates."""
    if len(
            self.f) != len(
            other.f) or not (
            self.is_feasible() and other.is_feasible()) or (
            not self.is_feasible() and not other.is_feasible()):
      return COMPARE_TYPE.UNDEFINED
    isbetter = False
    isworse = False
    for f1, f2 in zip(self.f, other.f):
      if f1 < f2:
        isbetter = True
      if f2 < f1:
        isworse = True
      if isworse and isbetter:
        break

    # // The comparison code has been adapted from
    # // Jaszkiewicz, A., & Lust, T. (2018).
    # // ND-tree-based update: a fast algorithm for the dynamic nondominance problem.
    # // IEEE Transactions on Evolutionary Computation, 22(5), 778-791.

    if not (isworse and isbetter):
      if self.h < other.h:
        isbetter = True
      if other.h < self.h:
        isworse = True

    if isworse:
      if isbetter:
        # TODO: check whether we need to add a "nondominated" type
        return COMPARE_TYPE.INDIFFERENT
      else:
        return COMPARE_TYPE.DOMINATED
    else:
      if isbetter:
        return COMPARE_TYPE.DOMINATING
      else:
        return COMPARE_TYPE.EQUAL

  def print_info(self):
    """Prints detailed information about the candidate."""
    print(f"f = {self.f}, h = {self.h}")

  def is_feasible(self):
    """Check if the candidate is feasible."""
    return self.status == DESIGN_STATUS.FEASIBLE
