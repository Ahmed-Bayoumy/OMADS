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
from typing import List, Dict, Any, Optional
from numpy import subtract, add, maximum, power, inf
import numpy as np


from ._globals import VAR_TYPE, DType, BARRIER_TYPES, MPP, DESIGN_STATUS, COMPARE_TYPE


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

  def __init__(
      self,
      _fc_index: Optional[int] = None,
      _n: int = 0,
      _coords: Optional[List[float]] = None,
      _defined: Optional[List[bool]] = None,
      _evaluated: bool = False,
      _f: Optional[List[float]] = None,
      _freal: Optional[List[float]] = None,
      _c_ineq: Optional[List[float]] = None,
      _c_eq: Optional[List[float]] = None,
      _h: float = inf,
      _signature: int = 0,
      _dtype: Optional[DType] = None,
      _var_type: Optional[List[int]] = None,
      _sets: Optional[Dict] = None,
      _var_link: Optional[List[str]] = None,
      _status: DESIGN_STATUS = DESIGN_STATUS.UNEVALUATED,
      _constraints_type: Optional[List[BARRIER_TYPES]] = None,
      _is_eb_passed: bool = False,
      _lambda: Optional[List[float]] = None,
      _rho: float = MPP.RHO.value,
      _hmax: float = 1.0,
      _hmin: float = inf,
      _eval_time: float = 0.0,
      _source: str = "Current run",
      _model: str = "Simulation",
      _hzero: Optional[float] = None,
      _direction: Optional[List[Any]] = None,
      _fs: Optional[List[Any]] = None,
      _eval_no: int = 0,
      _incumbent_signature: Optional[int] = None,
      _improving: bool = False,
      _cpb: Optional[List[float]] = None,
      _mapped_coords: Optional[List[Any]] = None,
      _is_nondominated: bool = False,
      _was_center: bool = False
  ):
    self._fc_index = _fc_index
    self._n = _n
    self._coords = _coords if _coords is not None else []
    self._defined = _defined if _defined is not None else []
    self._evaluated = _evaluated
    self._f = _f if _f is not None else []
    self._freal = _freal if _freal is not None else []
    self._c_ineq = _c_ineq if _c_ineq is not None else []
    self._c_eq = _c_eq if _c_eq is not None else []
    self._h = _h
    self._signature = _signature
    self._dtype = _dtype if _dtype is not None else DType()
    self._var_type = _var_type
    self._sets = _sets
    self._var_link = _var_link
    self._status = _status
    self._constraints_type = _constraints_type
    self._is_eb_passed = _is_eb_passed
    self._lambda = _lambda
    self._rho = _rho
    self._hmax = _hmax
    self._hmin = _hmin
    self._eval_time = _eval_time
    self._source = _source
    self._model = _model
    self._hzero = _hzero
    self._direction = _direction
    self._fs = _fs
    self._eval_no = _eval_no
    self._incumbent_signature = _incumbent_signature
    self._improving = _improving
    self._cpb = _cpb
    self._mapped_coords = _mapped_coords if _mapped_coords is not None else []
    self._is_nondominated = _is_nondominated
    self._was_center = _was_center

  @property
  def cpb(self):
    return self._cpb

  @cpb.setter
  def cpb(self, value: List[float]) -> List[float]:
    self._cpb = value

  @property
  def eval_no(self):
    return self._eval_no

  @eval_no.setter
  def eval_no(self, value: int) -> int:
    self._eval_no = value

  @property
  def model(self):
    return self._model

  @model.setter
  def model(self, value: str) -> str:
    self._model = value

  @property
  def source(self):
    return self._source

  @source.setter
  def source(self, value: str) -> str:
    self._source = value

  @property
  def eval_time(self):
    return self._eval_time

  @eval_time.setter
  def eval_time(self, value: float) -> float:
    self._eval_time = value

  @property
  def fc_index(self):
    return self._fc_index

  @fc_index.setter
  def fc_index(self, value: int) -> int:
    self._fc_index = value

  @property
  def was_center(self):
    return self._was_center

  @was_center.setter
  def was_center(self, value: Any) -> Any:
    self._was_center = value

  @property
  def improving(self):
    return self._improving

  @improving.setter
  def improving(self, value: bool) -> bool:
    self._improving = value

  @property
  def is_nondominated(self):
    return self._is_nondominated

  @is_nondominated.setter
  def is_nondominated(self, value: bool) -> bool:
    self._is_nondominated = value

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
  def mapped_coords(self):
    return self._mapped_coords

  @mapped_coords.setter
  def mapped_coords(self, value: List[Any]) -> Any:
    self._mapped_coords = list(value)

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
  def fs(self) -> List[Any]:
    """Point coordinates on the functions space

    :return: Functions Point
    :rtype: Point
    """
    if self._fs is None:
      self._fs = self.f
    return self._fs

  @fs.setter
  def fs(self, value: List[Any]) -> Any:
    self._fs = value

  # @property
  # def mesh(self) -> Gmesh:
  #   """The mesh settings the current point has been generated on

  #   :return: A mesh instant
  #   :rtype: Gmesh
  #   """
  #   return self._mesh

  # @mesh.setter
  # def mesh(self, value: Any) -> Any:
  #   self._mesh = value

  @property
  def direction(self) -> List[Any]:
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
  def var_link(self, value: List[str]):
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

  @signature.setter
  def signature(self, value: int) -> int:
    self._signature = value

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
    self.map_coordinates()
    self._defined = [True] * self._n

  @coordinates.deleter
  def coordinates(self):
    del self._coords

  def map_coordinates(self):
    self._mapped_coords = [None] * self._n
    if self.sets is not None and isinstance(self.sets, dict):
      for i, t in enumerate((self.var_type)):
        if (t == VAR_TYPE.DISCRETE or t == VAR_TYPE.CATEGORICAL) \
                and self.var_link[i] is not None:
          self._mapped_coords[i] = self.sets[self.var_link[i]][
              int(self._coords[i])]
        else:
          self._mapped_coords[i] = self._coords[i]
    else:
      self._mapped_coords = self._coords

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

    self.fs = self._freal

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
    return self.__compare__(other) in [COMPARE_TYPE.EQUAL, COMPARE_TYPE.
                                       DOMINATING]

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

  def _update_candidate_design_criteria(self, bb_output):
    """ Objective function, equality constraints and inequality constraints (can be an empty vector) """
    self.f = bb_output[0]
    self.fobj = bb_output[0]
    self.c_ineq = bb_output[1]
    if not isinstance(self.c_ineq, list):
      self.c_ineq = [self.c_ineq]
    self.evaluated = True

  def _check_and_initialize_multipliers_if_needed(self):
    """ Check the multiplier matrix """
    if self.lambda_multipliers is None:
      self.lambda_multipliers = []
      for _ in range(len(self.c_ineq)):
        self.lambda_multipliers.append(MPP.LAMBDA.value)
    else:
      if len(self.c_ineq) != len(self.lambda_multipliers):
        for _ in range(len(self.lambda_multipliers), len(self.c_ineq)):
          self.lambda_multipliers.append(MPP.LAMBDA.value)

  def _check_and_update_barriers_type_if_needed(self):
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

  def _calculate_barriers_based_violations(self):
    """ Check if all extreme barriers are satisfied """
    ceb = []
    cpb = []
    self.cpb = []
    for i, _ in enumerate((self.c_ineq)):
      if self.constraints_type[i] == BARRIER_TYPES.EB:
        ceb.append(self.c_ineq[i])
      else:
        cpb.append(self.c_ineq[i] if self.c_ineq[i] > 0. else 0.)
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
    return heb, hpb

  def _calculate_constraints_violation_function(self):
    heb: float = 0.
    hpb: float = 0.
    heb, hpb = self._calculate_barriers_based_violations()
    if heb <= self.hzero:
      self.is_extreme_barrier_passed = True
      if hpb > self.hzero:
        self.status = DESIGN_STATUS.INFEASIBLE
      self.h = copy.deepcopy(hpb)
      # if hpb < self.h_max:
      #   self.h_max = copy.deepcopy(hpb)
    else:
      self.is_extreme_barrier_passed = False
      self.status = DESIGN_STATUS.INFEASIBLE
      self.h = copy.deepcopy(heb)
      return
    # """ Aggregate all constraints """
    if np.isnan(self.h) or np.any(np.isnan(self.c_ineq)):
      self.h = inf
      self.status = DESIGN_STATUS.ERROR

  def _penalize_relaxable_constraints_violation(self):
    """ Penalize relaxable constraints violation """
    if (not self.is_extreme_barrier_passed):
      self.__penalize__(extreme=True)
      return
    if any(np.isnan(self.f)):
      for fi, _ in enumerate((self.f)):
        if np.isnan(self.f[fi]):
          self.f[fi] = np.inf
          self.fobj[fi] = np.inf

    if self.h > self.hzero:
      if self.h >= self.h_max and self.h_max < np.inf:
        self.__penalize__(extreme=False)
      self.status = DESIGN_STATUS.INFEASIBLE
    else:
      # self.h_max = copy.deepcopy(self.h)
      self.status = DESIGN_STATUS.FEASIBLE

  def _update_multipliers_and_penalty_param(self):
    for i, _ in enumerate(self.lambda_multipliers):
      self.lambda_multipliers[i] = np.max(
          [self.hzero, self.lambda_multipliers[i] + (1 / self.rho) * self.c_ineq[i]])

    if self.h < self.h_max:
      self.h_max = self.h
    self.rho *= 0.5

  def __eval__(self, bb_output):
    """ Evaluate point """
    self._update_candidate_design_criteria(bb_output)
    self._check_and_initialize_multipliers_if_needed()
    self._check_and_update_barriers_type_if_needed()
    self._calculate_constraints_violation_function()
    self._penalize_relaxable_constraints_violation()
    # self._update_multipliers_and_penalty_param()
    if self.h < self.h_max:
      self.h_max = self.h

  def __penalize__(self, extreme: bool = True):
    if len(self.cpb) > len(self.lambda_multipliers):
      self.lambda_multipliers += [self.lambda_multipliers[-1]
                                  ] * abs(len(self.lambda_multipliers) - len(self.cpb))
    if 0 < len(self.cpb) < len(self.lambda_multipliers):
      del self.lambda_multipliers[len(self.cpb):]
    if extreme:
      self.hmin = inf
    else:
      # np.dot(self.lambda_multipliers, self.cpb) + ((1/(2*self.rho))
      #                                              * self.h if self.rho > 0. else np.inf)
      # self.hmin = self.h
      # TODO: Check the following for MOO
      self.f = [
          self.fobj[i] * (1. / len(self.fobj)) + np.dot(
              self.lambda_multipliers, self.cpb) +
          ((1 / (2 * self.rho)) * self.h if self.rho > 0. else np.inf)
          for i in range(len(self.fobj))] if len(self.fobj) == 1 else self.fobj

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
        self.fobj) != len(
            other.fobj) or (self.status != other.status):  # not (
      # self.is_feasible() and other.is_feasible()) or (
      # not self.is_feasible() and not other.is_feasible()):
      return COMPARE_TYPE.UNDEFINED

    isbetter = False
    isworse = False
    for f1, f2 in zip(self.fobj, other.fobj):
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
    if (self.status == DESIGN_STATUS.INFEASIBLE and other.status == DESIGN_STATUS.INFEASIBLE) or DESIGN_STATUS.INFEASIBLE in [self.status, other.status]:
      if not (isworse and isbetter):
        if self.h < other.h:
          isbetter = True
        if other.h < self.h:
          isworse = True
    # elif (self.status == DESIGN_STATUS.INFEASIBLE and other.status == DESIGN_STATUS.FEASIBLE):
    #   if not (isworse and isbetter):
    #     if self.h <= self.h_max:
    #       isbetter = True
    #     if self.h_max < self.h:
    #       isworse = True
    # elif (self.status == DESIGN_STATUS.FEASIBLE and other.status == DESIGN_STATUS.INFEASIBLE):
    #   if not (isworse and isbetter):
    #     if other.h <= self.h_max:
    #       isbetter = True
    #     if self.h_max < other.h:
    #       isworse = True

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

  def create_candidate_point_from_coords(
          self, temp, var_type: List, var_sets: Dict, var_link: List[str],
          c_types: List[BARRIER_TYPES] = None, fc_index: int = None, other=None):
    self.constraints_type = copy.deepcopy(
        [xb for xb in c_types] if isinstance(c_types, list) else [c_types])
    self.sets = copy.deepcopy(var_sets)
    self.var_type = copy.deepcopy(var_type)
    self.var_link = copy.deepcopy(var_link)
    self.coordinates = temp
    self.dtype.precision = self.dtype.precision
    # tmp.mesh = copy.deepcopy(self.mesh)
    self.fc_index = fc_index
    self.rho = other.rho
    self.lambda_multipliers = other.lambda_multipliers
    self.incumbent_signature = other.signature

  def to_dict(self) -> Dict[str, Any]:
    return {
        key: value
        for key, value in self.__dict__.items()
    }

  def __dir__(self):
      # Collect instance attributes
    instance_attrs = list(self.__dict__.keys())

    # Collect class attributes and methods
    class_attrs = list(set(dir(self.__class__)))

    # Combine and deduplicate
    return sorted(set(instance_attrs + class_attrs))
