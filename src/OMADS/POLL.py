
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

"""
  This is a python implementation of the orothognal mesh adaptive direct search method (OMADS)
"""
from inspect import signature
import warnings
import logging
import copy
import csv
import json
from multiprocessing import freeze_support, cpu_count
import os
import platform
import subprocess
import sys
import numpy as np
import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any
from BMDFO import toy
from numpy import sum, subtract, add, maximum, minimum, power, inf
from enum import Enum, auto
import random



@dataclass
class DType:
  """A numpy data type delegator for decimal precision control

    :param prec: Default precision option can be set to "high", "medium" or "low" precision resolution, defaults to "medium"
    :type prec: str, optional
    :param dtype: Numpy double data type precision, defaults to np.float64
    :type dtype: np.dtype, optional
    :param itype: Numpy integer data type precision, defaults to np.int
    :type itype: np.dtype, optional
    :param zero: Zero value resolution, defaults to np.finfo(np.float64).resolution
    :type zero: float, optional
  """
  _prec: str = "medium"
  _dtype: np.dtype = np.float64
  _itype: np.dtype = np.int_
  _zero: float = np.finfo(np.float64).resolution
  _warned: bool = False


  @property
  def zero(self)->float:
    """This is the mantissa value of the machine zero

    :return: machine precision zero resolution
    :rtype: float
    """
    return self._zero

  @property
  def precision(self):
    """Set/get precision resolution level

    :return: float precision value
    :rtype: numpy float
    :note: MS Windows does not support precision with the {1e-18} high resolution of the python
            numerical library (numpy) so high precision will be
            changed to medium precision which supports {1e-15} resolution
            check: https://numpy.org/doc/stable/user/basics.types.html 
    """
    return self._prec

  @precision.setter
  def precision(self, val: str):
    self._prec = val
    self._prec = val
    isWin = platform.platform().split('-')[0] == 'Windows'
    if val == "high":
      if (isWin or not hasattr(np, 'float128')):
        # COMPLETE: pop up this warning during initialization
        'Warning: MS Windows does not support precision with the {1e-18} high resolution of the python numerical library (numpy) so high precision will be changed to medium precision which supports {1e-15} resolution check: https://numpy.org/doc/stable/user/basics.types.html '
        self.dtype = np.float64
        self._zero = np.finfo(np.float64).resolution
        self.itype = np.int_
        if not self._warned:
          warnings.warn("MS Windows does not support precision with the {1e-18} high resolution of the python numerical library (numpy) so high precision will be changed to medium precision which supports {1e-15} resolution check: https://numpy.org/doc/stable/user/basics.types.html")
        self._warned = True
      else:
        self.dtype = np.float128
        self._zero = np.finfo(np.float128).resolution
        self.itype = np.int_
    elif val == "medium":
      self.dtype = np.float64
      self._zero = np.finfo(np.float64).resolution
      self.itype = np.intc
    elif val == "low":
      self.dtype = np.float32
      self._zero = np.finfo(np.float32).resolution
      self.itype = np.short
    else:
      raise Exception("JASON parameters file; unrecognized textual"
              " input for the defined precision type. "
              "Please enter one of these textual values (high, medium, low)")

  @property
  def dtype(self):
    """Set/get the double formate type

    :return: float precision value
    :rtype: numpy float 
    """
    return self._dtype

  @dtype.setter
  def dtype(self, other: np.dtype):
    self._dtype = other

  @property
  def itype(self):
    """Set/Get for the integer formate type

    :return: integer precision value
    :rtype: numpy float 
    """
    return self._itype

  @itype.setter
  def itype(self, other: np.dtype):
    self._itype = other


@dataclass
class Options:
  """ The running study and algorithmic options of OMADS
  
    :param seed: Random generator seed
    :param budget: The evaluation budget
    :param tol: The threshold of the minimum poll size at which the run will be terminated
    :param psize_init: Initial poll size
    :param dispaly: Print the study progress during the run
    :param opportunistic: Loop on the points populated in the poll set until a better minimum found, then stop the evaluation loop
    :param check_cache: Check the hash table before points evaluation to avoid duplicates
    :param store_cache: Enable storing evaluated points into the hash table
    :param collect_y: Collect dependent design variables (required for DMDO)
    :param rich_direction: Go with the rich direction (Impact the mesh size update)
    :param precision: Define the precision level
    :param save_results: A boolean flag that indicates saving results in a csv file
    :param save_coordinates: A boolean flag that indicates saving coordinates of the poll set in a JSON file (required to generate animations of the spinner)
    :param save_all_best: A boolean used to check whether saving best points only in the MADS.out file
    :param parallel_mode: A boolean to check whether evaluating the poll set in parallel multiprocessing
    :param np: The number of CPUs
  """
  seed: int = 0
  budget: int = 1000
  tol: float = 1e-9
  psize_init: float = 1.0
  display: bool = False
  opportunistic: bool = False
  check_cache: bool = False
  store_cache: bool = False
  collect_y: bool = False
  rich_direction: bool = False
  precision: str = "high"
  save_results: bool = False
  save_coordinates: bool = False
  save_all_best: bool = False
  parallel_mode: bool = False
  np: int = 1
  extend: Any = None

class VAR_TYPE(Enum):
  CONTINUOUS = auto()
  INTEGER = auto()
  DISCRETE = auto()
  BINARY = auto()
  CATEGORICAL = auto()
  ORDINAL = auto()

class BARRIER_TYPES(Enum):
  EB = auto()
  PB = auto()
  PEB = auto()
  RB = auto()

class SUCCESS_TYPES(Enum):
  US = auto()
  PS = auto()
  FS = auto()
  
class MPP(Enum):
  LAMBDA = 0.
  RHO = 0.00005

class DESIGN_STATUS(Enum):
  FEASIBLE = auto()
  INFEASIBLE = auto()
  ERROR = auto()
  UNEVALUATED = auto()

@dataclass
class Parameters:
  """ Variables and algorithmic parameters 
  
    :param baseline: Baseline design point (initial point ``x0``)
    :param lb: The variables lower bound
    :param ub: The variables upper bound
    :param var_names: The variables name
    :param scaling: Scaling factor (can be defined as a list (assigning a factor for each variable) or a scalar value that will be applied on all variables)
    :param post_dir: The location and name of the post directory where the output results file will live in (if any)
  """
  baseline: List[float] = field(default_factory=lambda: [0.0, 0.0])
  lb: List[float] = field(default_factory=lambda: [-5.0, -5.0])
  ub: List[float] = field(default_factory=lambda: [10.0, 10.0])
  var_names: List[str] = field(default_factory=lambda: ["x1", "x2"])
  scaling: float = 10.0
  post_dir: str = os.path.abspath(".\\")
  var_type: List[str] = None
  var_sets: Dict = None
  constants: List = None
  constants_name: List = None
  Failure_stop: bool = None
  problem_name: str = "unknown"
  best_known: List[float] = None
  constraints_type: List[BARRIER_TYPES] = None
  h_max: float = 0
  RHO: float = 0.00005
  LAMBDA: List[float] = None
  name: str = "undefined"
  # COMPLETE: support more variable types
  # TODO: give better control on variabls' resolution (mesh granularity)
  # var_type: List[str] = field(default_factory=["cont", "cont"])
  # resolution: List[int] = field(default_factory=["cont", "cont"])

  def get_barrier_type(self):
    if self.constraints_type is not None:
      if isinstance(self.constraints_type, list):
        for i in range(len(self.constraints_type)):
          if self.constraints_type[i] == BARRIER_TYPES.PB:
            return BARRIER_TYPES.PB
      else:
        if self.constraints_type == BARRIER_TYPES.PB:
            return BARRIER_TYPES.PB

    
    return BARRIER_TYPES.EB
  
  def get_h_max_0 (self):
    return self.h_max


@dataclass
class Evaluator:
  """ Define the evaluator attributes and settings
  
    :param blackbox: The blackbox name (it can be a callable function or an executable file)
    :param commandOptions: Define options that will be added to the execution command of the executable file. Command options should be defined as a string in one line
    :param internal: If the blackbox callable function is part of the internal benchmarking library
    :param path: The path of the executable file (if any) 
    :param input: The input file name -- should include the file extension
    :param output: The output file name -- should include the file extension 
    :param constants: Define constant parameters list, see the documentation in Tutorials->Blackbox evaluation->User parameters
    :param _dtype: The precision delegator of the numpy library
    :param timeout: The time out of the evaluation process
  """
  blackbox: Any = "rosenbrock"
  commandOptions: Any = None
  internal: Optional[str] = None
  path: str = "..\\tests\\Rosen"
  input: str = "input.inp"
  output: str = "output.out"
  constants: List = None
  bb_eval: int = 0
  _dtype: DType = DType()
  timeout: float = 1000000.

  @property
  def dtype(self):
    return self._dtype

  def eval(self, values: List[float]):
    """ Evaluate the poll point

    :param values: Poll point coordinates (design vector)
    :type values: List[float]
    :raises IOError: Incorrect number of input arguments introduced to the callable function
    :raises IOError: Incorrect number of input arguments introduced to the callable function
    :raises IOError: The blackbox file is not an executable file (if not a callable function)
    :raises IOError: Incorrect benchmarking keyword category
    :return: Evaluated optimization functions
    :rtype: List[float, List[float]]
    """
    self.bb_eval += 1
    if self.internal is None or self.internal == "None" or self.internal == "none":
      if callable(self.blackbox):
        is_object = False
        try:
          sig = signature(self.blackbox)
        except:
          is_object = True
          pass
        if not is_object:
          npar = len(sig.parameters) 
          # Get input arguments defined for the callable 
          inputs = str(sig).replace("(", "").replace(")", "").replace(" ","").split(',')
          # Check if user constants list is defined and if the number of input args of the callable matches what OMADS expects 
          if self.constants is None:
            if (npar == 1 or (npar> 0 and npar <= 3 and ('*argv' in inputs))):
              try:
                f_eval = self.blackbox(values)
              except:
                evalerr = True
                logging.error(f"Callable {str(self.blackbox)} evaluation returned an error at the poll point {values}")
                f_eval = [inf, [inf]]
            else:
              raise IOError(f'The callable {str(self.blackbox)} requires {npar} input args, but only one input can be provided! You can introduce other input parameters to the callable function using the constants list.')
          else:
            if (npar == 2 or (npar> 0 and npar <= 3 and ('*argv' in inputs))):
              try:
                f_eval = self.blackbox(values, self.constants)
              except:
                evalerr = True
                logging.error(f"Callable {str(self.blackbox)} evaluation returned an error at the poll point {values}")
            else:
              raise IOError(f'The callable {str(self.blackbox)} requires {npar} input args, but only two input args can be provided as the constants list is defined!')
        else:
          try:
            f_eval = self.blackbox(values)
          except:
            evalerr = True
            logging.error(f"Callable {str(self.blackbox)} evaluation returned an error at the poll point {values}")
            f_eval = [inf, [inf]]
        if isinstance(f_eval, list):
          return f_eval
        elif isinstance(f_eval, float) or isinstance(f_eval, int):
          return [f_eval, [0]]
      else:
        self.write_input(values)
        pwd = os.getcwd()
        os.chdir(self.path)
        isWin = platform.platform().split('-')[0] == 'Windows'
        evalerr = False
        timouterr = False
        #  Check if the file is executable
        executable = os.access(self.blackbox, os.X_OK)
        if not executable:
          raise IOError(f"The blackbox file {str(self.blackbox)} is not an executable! Please provide a valid executable file.")
        # Prepare the execution command based on the running machine's OS
        if isWin and self.commandOptions is None:
          cmd = self.blackbox
        elif isWin:
          cmd = f'{self.blackbox} {self.commandOptions}'
        elif self.commandOptions is None:
          cmd = f'./{self.blackbox}'
        else:
          cmd =  f'./{self.blackbox} {self.commandOptions}'
        try:
          p = subprocess.run(cmd, shell=True, timeout=self.timeout)
          if p.returncode != 0:
            evalerr = True
            logging.error("Evaluation # {self.bb_eval} is errored at the poll point {values}")
        except subprocess.TimeoutExpired:
          timouterr = True 
          logging.error(f'Timeout for {cmd} ({self.timeout}s) expired at evaluation # {self.bb_eval} at the poll point {values}')

        os.chdir(pwd)
        
        if evalerr or timouterr:
          out = [np.inf, [np.inf]]
        else:
          out = [self.read_output()[0], [self.read_output()[1:]]]
        return out
    elif self.internal == "uncon":
      f_eval = toy.UnconSO(values)
    elif self.internal == "con":
      f_eval = toy.ConSO(values)
    else:
      raise IOError(f"Input dict:: evaluator:: internal:: "
              f"Incorrect internal method :: {self.internal} :: "
              f"it should be a a BM library name, "
              f"or None.")
    f_eval.dtype.dtype = self._dtype.dtype
    f_eval.name = self.blackbox
    f_eval.dtype.dtype = self._dtype.dtype
    return getattr(f_eval, self.blackbox)()

  def write_input(self, values: List[float]):
    """_summary_

    :param values: Write the variables in the input file
    :type values: List[float]
    """
    inp = os.path.join(self.path, self.input)
    with open(inp, 'w+') as f:
      for c, value in enumerate(values, start=1):
        if c == len(values):
          f.write(str(value))
        else:
          f.write(str(value) + "\n")

  def read_output(self) -> List[float]:
    """_summary_

    :return: Read the output values from the output file 
    :rtype: List[float]
    """
    out = os.path.join(self.path, self.output)
    f_eval = []
    f = open(out)
    for line in f:  # read rest of lines
      f_eval.append(float(line))
    f.close()
    if len(f_eval) == 1:
      f_eval.append(0.0)
    return f_eval


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
         (((self.hmax if self._is_EB_passed else self._dtype.zero) > self.h >= 0.0) and
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
    for i in range(len(self.c_ineq)):
      if self.constraints_type[i] == BARRIER_TYPES.EB:
        cEB.append(self.c_ineq[i])
    if isinstance(cEB, list) and len(cEB) >= 1:
      hEB = sum(power(maximum(cEB, self._dtype.zero,
                   dtype=self._dtype.dtype), 2, dtype=self._dtype.dtype))
    else:
      hEB = self._dtype.zero
    if hEB <= self.hzero:
      self.is_EB_passed = True
    else:
      self.is_EB_passed = False
      self.status = DESIGN_STATUS.INFEASIBLE
      self.__penalize__(extreme= True)
      return
    """ Aggregate all constraints """
    self.h = sum(power(maximum(self.c_ineq, self._dtype.zero,
                   dtype=self._dtype.dtype), 2, dtype=self._dtype.dtype))
    if np.isnan(self.h) or np.any(np.isnan(self.c_ineq)):
      self.h = inf
      self.status = DESIGN_STATUS.ERROR

    """ Penalize relaxable constraints violation """
    if np.isnan(self.f) or self.h > self.hzero:
      if self.h > np.round(self.hmax, 2):
        self.__penalize__(extreme=False)
      self.status = DESIGN_STATUS.INFEASIBLE
    else:
      self.status = DESIGN_STATUS.FEASIBLE

  def __penalize__(self, extreme: bool=True):
    if len(self.c_ineq) > len(self.LAMBDA):
      self.LAMBDA += [self.LAMBDA[-1]] * abs(len(self.LAMBDA)-len(self.c_ineq))
    if len(self.c_ineq) < len(self.LAMBDA):
      del self.LAMBDA[len(self.c_ineq):]
    if extreme:
      self.f = inf
    else:
      self.f = self.fobj + np.dot(self.LAMBDA, self.c_ineq) + ((1/(2*self.RHO)) * self.h if self.RHO > 0. else np.inf)

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


@dataclass
class Barrier:
  _params: Parameters = None
  _eval_type: int = 1
  _h_max: float = 0
  _best_feasible: Point = None
  _ref: Point = None
  _filter: List[Point] = None
  _prefilter: int = 0
  _rho_leaps: float = 0.1
  _prim_poll_center: Point = None
  _sec_poll_center: Point = None
  _peb_changes: int = 0
  _peb_filter_reset: int = 0
  _peb_lop: List[Point] = None
  _all_inserted: List[Point] = None
  _one_eval_succ: int = None
  _success: int = None

  def __init__(self, p: Parameters, eval_type: int = 1):
    self._h_max = p.get_h_max_0()
    self._params = p
    self._eval_type = eval_type


  def insert_feasible(self, x: Point) -> SUCCESS_TYPES:
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
    
    if (fx < fx_bf):
      self._best_feasible = copy.deepcopy(x)
      return SUCCESS_TYPES.FS
    
    return SUCCESS_TYPES.US
  
  def filter_insertion(self, x:Point) -> bool:
    if not x._is_EB_passed:
      return
    if self._filter is None:
      self._filter = []
      self._filter.append(x)
      insert = True
    else:
      insert = False
      it = 0
      while it != len(self._filter):
        if (x<self._filter[it]):
          del self._filter[it]
          insert = True
          continue
        it += 1
      
      if not insert:
        insert = True
        for it in range(len(self._filter)):
          if self._filter[it].fobj < x.fobj:
            insert = False
            break
      
      if insert:
        self._filter.append(x)
    
    return insert


  def insert_infeasible(self, x: Point):
    insert: bool = self.filter_insertion(x=x)
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
    if (fx > fr):
      return SUCCESS_TYPES.PS
    
    #  FULL success
    return SUCCESS_TYPES.FS

  def get_best_infeasible(self):
    return self._filter[-1]
  
  def get_best_infeasible_min_viol(self):
    return self._filter[0]
  
  def select_poll_center(self):
    best_infeasible: Point = self.get_best_infeasible()
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
    
    last_poll_center: Point = Point()
    if self._params.get_barrier_type() == BARRIER_TYPES.PB:
      last_poll_center = self._prim_poll_center
      if best_infeasible.fobj < (self._best_feasible.fobj-self._rho_leaps):
        self._prim_poll_center = best_infeasible
        self._sec_poll_center = self._best_feasible
      else:
        self._prim_poll_center = self._best_feasible
        self._sec_poll_center = best_infeasible

      if last_poll_center is None or self._prim_poll_center != last_poll_center:
        self._rho_leaps += 1

  def set_h_max(self, h_max):
    self._h_max = np.round(h_max, 2)
    if self._filter is not None:
      if self._filter[0].h > self._h_max:
        self._filter = None
        return
    if self._filter is not None:
      it = 0
      while it != len(self._filter):
        if (self._filter[it].h>self._h_max):
          del self._filter[it]
          continue
        it += 1

  def insert(self, x: Point):
    """/*---------------------------------------------------------*/
      /*         insertion of an Eval_Point in the barrier       */
      /*---------------------------------------------------------*/
    """
    if not x.evaluated:
      raise RuntimeError("This points hasn't been evaluated yet and cannot be inserted to the barrier object!")
    
    if (x.status == DESIGN_STATUS.ERROR):
      self._one_eval_succ = SUCCESS_TYPES.US
    if self._all_inserted is None:
      self._all_inserted = []
    self._all_inserted.append(x)
    h = x.h
    if x.status == DESIGN_STATUS.INFEASIBLE and (not x.is_EB_passed or x.h > self._h_max):
      self._one_eval_succ = SUCCESS_TYPES.US
      return
    
    # insert_feasible or insert_infeasible:
    self._one_eval_succ = self.insert_feasible(x) if x.status == DESIGN_STATUS.FEASIBLE else self.insert_infeasible(x)

    if self._success is None or self._one_eval_succ.value > self._success.value:
      self._success = self._one_eval_succ


  def insert_VNS(self):
    pass

  def update_and_reset_success(self):
    """/*------------------------------------------------------------*/
      /*  barrier update: invoked by Evaluator_Control::eval_lop()  */
      /*------------------------------------------------------------*/
    """
    if self._params.get_barrier_type() == BARRIER_TYPES.PB and self._success != SUCCESS_TYPES.US:
      if self._success == SUCCESS_TYPES.PS:
        if self._filter is None:
          raise RuntimeError("filter empty after a partial success")
        it = len(self._filter)-1
        while True:
          if (self._filter[it].h<self._h_max):
            self.set_h_max(self._filter[it].h)
            break
          if it == 0:
            break
            # raise RuntimeError("could not find a filter point with h < h_max after a partial success")
          it -= 1
      if self._filter is not None:
        self._ref = self.get_best_infeasible()
      if self._ref is not None:
        self.set_h_max(self._ref.h)
        if self._ref.status is DESIGN_STATUS.INFEASIBLE:
          self.insert_infeasible(self._ref)
        
        if self._ref.status is DESIGN_STATUS.FEASIBLE:
          self.insert_feasible(self._ref)
        
        if not (self._ref.status is DESIGN_STATUS.INFEASIBLE or self._ref.status is DESIGN_STATUS.INFEASIBLE):
          self.insert(self._ref)

        
    
    # reset success types:
    self._one_eval_succ = self._success = SUCCESS_TYPES.US

    

  def reset(self):
    """/*---------------------------------------------------------*/
      /*                    reset the barrier                    */
      /*---------------------------------------------------------*/"""

    self._prefilter = None
    self._filter = None
    # self._h_max = self._params._h_max_0()
    self._best_feasible   = None
    self._ref             = None
    self._rho_leaps       = 0
    self._poll_center     = None
    self._sec_poll_center = None
    
    # if ( self._peb_changes > 0 ):
    #     self._params.reset_PEB_changes()
    
    self._peb_changes      = 0
    self._peb_filter_reset = 0
    
    self._peb_lop = None
    self._all_inserted = None
    
    self._one_eval_succ = _success = SUCCESS_TYPES.US

@dataclass
class OrthoMesh:
  """ Mesh coarsness update class

  :param _delta: mesh size
  :param _Delta: poll size
  :param _rho: poll size to mesh size ratio
  :param _exp:  manage the poll size granularity for discrete variables. See Audet et. al, The mesh adaptive direct search algorithm for granular and discrete variable
  :param _mantissa: Same as ``_exp``
  :param psize_max: Maximum poll size
  :param psize_success: Poll size at successful evaluation
  :param _dtype: numpy double data type precision

  """
  _delta: float = 1.0  # mesh size
  _Delta: float = 1.0  # poll size
  _rho: float = 1.0  # poll size to mesh size ratio
  # TODO: manage the poll size granularity for discrete variables
  # See: Audet et. al, The mesh adaptive direct search algorithm for
  # granular and discrete variable
  _exp: int = 0
  _mantissa: int = 1
  psize_max: float = 0.0
  psize_success: float = 0.0
  # numpy double data type precision
  _dtype: DType = DType()

  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, other: DType):
    self._dtype = other

  @property
  def msize(self):
    return self._delta

  @msize.setter
  def msize(self, size):
    self._delta = size

  @msize.deleter
  def msize(self):
    del self._delta

  @property
  def psize(self):
    return self._Delta

  @psize.setter
  def psize(self, size):
    self._Delta = size

  @psize.deleter
  def psize(self):
    del self._Delta

  @property
  def rho(self):
    return self._rho

  @rho.setter
  def rho(self, size):
    self._rho = size

  @rho.deleter
  def rho(self):
    del self._rho

  def update(self):
    self.msize = minimum(np.power(self._Delta, 2.0, dtype=self.dtype.dtype),
               self._Delta, dtype=self.dtype.dtype)
    self.rho = np.divide(self._Delta, self._delta, dtype=self.dtype.dtype)


@dataclass
class Cache:
  """ In computing, a hash table (hash map) is a data structure that implements an associative array abstract data type, a structure that can map keys to values. A hash table uses a hash function to compute an index, also called a hash code, into an array of buckets or slots, from which the desired value can be found. During lookup, the key is hashed and the resulting hash indicates where the corresponding value is stored."""
  _hash_ID: List[int] = field(default_factory=list)
  _cache_dict: Dict[Any, Any] = field(default_factory=lambda: {})

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
    return self._hash_ID

  @hash_id.setter
  def hash_id(self, other: Point):
    self._hash_ID.append(hash(tuple(other.coordinates)))

  @property
  def size(self)->int:
    """A getter of the size of the hash ID list

    :rtype: int
    """
    return len(self.hash_id)

  def is_duplicate(self, x: Point) -> bool:
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

  def get_index(self, x: Point)->int:
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

  def add_to_cache(self, x: Point):
    """Save the point x to the cache memory

    :param x: Evaluated point to be saved in the cache memory
    :type x: Point
    """
    if not isinstance(x, list):
      hash_value: int = hash(tuple(x.coordinates))
      self._cache_dict[hash_value] = x
      self._hash_ID.append(hash(tuple(x.coordinates)))
    else:
      for i in range(len(x)):
        hash_value: int = hash(tuple(x[i].coordinates))
        self._cache_dict[hash_value] = x[i]
        self._hash_ID.append(hash(tuple(x[i].coordinates)))


@dataclass
class Dirs2n:
  """This is the orthognal 2n-directions class used for the poll step

    :param _poll_dirs: Poll set (list of points)
    :param _point_index: List of point indices
    :param _n: Number of directions
    :param _defined: A boolean that indicate if the poll points are defined
  """
  _poll_dirs: List[Point] = field(default_factory=list)
  _point_index: List[int] = field(default_factory=list)
  _n: int = 0
  _defined: List[bool] = field(default_factory=lambda: [False])
  scaling: List[List[float]] = field(default_factory=list)
  _xmin: Point = Point()
  _x_sc: Point = Point()
  _nb_success: int = 0
  _bb_eval: int = field(default_factory=int)
  _psize: float = field(default_factory=float)
  _iter: int = field(default_factory=int)
  hashtable: Cache = field(default_factory=Cache)
  _check_cache: bool = True
  _display: bool = True
  _store_cache: bool = True
  _save_results = True
  mesh: OrthoMesh = field(default_factory=OrthoMesh)
  _opportunistic: bool = False
  _eval_budget: int = 100
  _dtype: DType = DType()
  bb_handle: Evaluator = field(default_factory=Evaluator)
  _success: bool = False
  _seed: int = 0
  _terminate: bool = False
  _bb_output: List[float] = field(default_factory=list)
  Failure_stop: bool = None
  RHO: float = MPP.RHO
  LAMBDA: List[float] = None
  hmax: float = 1.

  @property
  def x_sc(self) -> Point:
    return self._x_sc
  
  @x_sc.setter
  def x_sc(self, value: Point):
    self._x_sc = value
  

  @property
  def bb_output(self) -> List[float]:
    return self._bb_output

  @bb_output.setter
  def bb_output(self, other: List[float]):
    self._bb_output = other

  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, other: DType):
    self._dtype = other

  @property
  def point_index(self):
    return self._point_index

  @point_index.setter
  def point_index(self, other: int):
    if other == -1:
      self._point_index = []
    else:
      self._point_index.append(other)

  @property
  def terminate(self):
    return self._terminate

  @terminate.setter
  def terminate(self, other: bool):
    self._terminate = other

  @property
  def seed(self):
    return self._seed

  @seed.setter
  def seed(self, other: int):
    self._seed = other

  @property
  def success(self):
    return self._success

  @success.setter
  def success(self, other: bool):
    self._success = other

  @property
  def eval_budget(self):
    return self._eval_budget

  @eval_budget.setter
  def eval_budget(self, other: int):
    self._eval_budget = other

  @property
  def opportunistic(self):
    return self._opportunistic

  @opportunistic.setter
  def opportunistic(self, other: bool):
    self._opportunistic = other

  @property
  def save_results(self):
    return self._save_results

  @save_results.setter
  def save_results(self, other: bool):
    self._save_results = other

  @property
  def store_cache(self):
    return self._store_cache

  @store_cache.setter
  def store_cache(self, other: bool):
    self._store_cache = other

  @property
  def check_cache(self):
    return self._check_cache

  @check_cache.setter
  def check_cache(self, other: bool):
    self._check_cache = other

  @property
  def display(self):
    return self._display

  @display.setter
  def display(self, other: bool):
    self._display = other

  @property
  def bb_eval(self):
    return self._bb_eval

  @bb_eval.setter
  def bb_eval(self, other: int):
    self._bb_eval = other

  @property
  def psize(self):
    return self._psize

  @psize.setter
  def psize(self, other: float):
    self._psize = other

  @property
  def iter(self):
    return self._iter

  @iter.setter
  def iter(self, other: int):
    self._iter = other

  @property
  def poll_dirs(self):
    return self._poll_dirs

  @poll_dirs.setter
  def poll_dirs(self, p: Point):
    self._poll_dirs.append(p)

  @poll_dirs.deleter
  def poll_dirs(self):
    del self._poll_dirs
    self._poll_dirs = []

  @property
  def dim(self):
    return self._n

  @dim.setter
  def dim(self, n: int):
    self._n = n

  @dim.deleter
  def dim(self):
    self._n = 0

  @property
  def defined(self):
    return any(self._defined)

  @defined.setter
  def defined(self, defined):
    self._defined = defined

  @defined.deleter
  def defined(self):
    del self._defined

  @property
  def xmin(self):
    return self._xmin

  @xmin.setter
  def xmin(self, other: Point):
    self._xmin = other

  @property
  def nb_success(self):
    return self._nb_success

  @nb_success.setter
  def nb_success(self, other: int):
    self._nb_success = other

  def generate_dir(self):
    return np.random.rand(self._n).tolist()

  def ran(self):
    return np.random.random(self._n).astype(dtype=self._dtype.dtype)

  def create_housholder(self, is_rich: bool, domain: List[int] = None, is_oneDir: bool=False) -> np.ndarray:
    """Create householder matrix

    :param is_rich:  A flag that indicates if the rich direction option is enabled
    :type is_rich: bool
    :return: The householder matrix
    :rtype: np.ndarray
    """
    if domain is None:
      domain = [VAR_TYPE.CONTINUOUS] * self._n
    elif len(domain) != self._n:
      raise IOError("Number of dimensions doesn't match the size of the variables type list invoked to Dirs2n::create_householder.")
    elif not isinstance(domain, list):
      raise IOError("The variables domain type input invoked to Dirs2n::create_householder should be of type list.")
    
    hhm: np.ndarray
    if is_rich:
      v_dir = copy.deepcopy(self.ran())
      v_dir_array = np.array(v_dir, dtype=self._dtype.dtype)
      v_dir_array = np.divide(v_dir_array,
                  (np.linalg.norm(v_dir_array,
                          2).astype(dtype=self._dtype.dtype)),
                  dtype=self._dtype.dtype)
      hhm = np.subtract(np.eye(self.dim, dtype=self._dtype.dtype),
                np.multiply(2.0, np.outer(v_dir_array, v_dir_array.T),
                      dtype=self._dtype.dtype),
                dtype=self._dtype.dtype)
    else:
      hhm = np.eye(self.dim, dtype=self._dtype.dtype)
    hhm = np.dot(hhm, np.diag((np.abs(hhm, dtype=self._dtype.dtype)).max(1) ** (-1)))
    # Rounding( and transpose)
    tmp = np.multiply(self.mesh.rho, hhm, dtype=self._dtype.dtype)
    hhm = np.transpose(np.multiply(self.mesh.msize, np.ceil(tmp), dtype=self._dtype.dtype))
    hhm = np.dot(hhm, self.scaling)

    for i in range(len(domain)):
      if domain[i] == VAR_TYPE.DISCRETE or domain[i] == VAR_TYPE.BINARY or domain[i] == VAR_TYPE.INTEGER:
        hhm[i][i] = int(np.floor((-1 if i%2 else 1) - 2**self.mesh.msize))
      elif domain[i] == VAR_TYPE.CATEGORICAL:
        hhm[i][i] = np.ceil(np.random.random(1).astype(dtype=self._dtype.dtype))
      else:
        for j in range(len(domain)):
          if domain[j] != VAR_TYPE.CONTINUOUS:
            hhm[i][j] = int(np.floor(-1 + 2**self.mesh.msize))
    
    if is_oneDir:
      return hhm
    else:
      hhm = np.vstack((hhm, -hhm))


    return hhm

  def create_poll_set(self, hhm: np.ndarray, ub: List[float], lb: List[float], it: int, var_type: List, var_sets: Dict, var_link: List[str], c_types: List[BARRIER_TYPES]=None, is_prim: bool = True):
    """Create the poll directions

    :param hhm: Householder matrix
    :type hhm: np.ndarray
    :param ub: Variables upper bound
    :type ub: List[float]
    :param lb: Variables lower bound
    :type lb: List[float]
    :param it: iteration
    :type it: int
    """
    if is_prim:
      del self.poll_dirs
      temp = np.add(hhm, np.array(self.xmin.coordinates), dtype=self._dtype.dtype)
    else:
      temp = np.add(hhm, np.array(self.x_sc.coordinates), dtype=self._dtype.dtype)
    # np.random.seed(self._seed)
    temp = np.random.permutation(temp)
    temp = np.minimum(temp, ub, dtype=self._dtype.dtype)
    temp = np.maximum(temp, lb, dtype=self._dtype.dtype)
    temp = np.unique(temp, axis=0)
    if isinstance(temp, list) or isinstance(temp, np.ndarray):
      ndirs = len(temp) if isinstance(temp[0], list) or isinstance(temp[0], np.ndarray) else 1
    else:
      ndirs = 0
    for k in range(ndirs):
      tmp = Point()
      tmp.constraints_type = copy.deepcopy([xb for xb in c_types] if isinstance(c_types, list) else [c_types])
      tmp.sets = copy.deepcopy(var_sets)
      tmp.var_type = copy.deepcopy(var_type)
      tmp.var_link = copy.deepcopy(var_link)
      tmp.coordinates = temp[k]
      tmp.dtype.precision = self.dtype.precision
      self.poll_dirs = tmp
      del tmp
    del temp

    self.iter = it

  def scale(self, ub: List[float], lb: List[float], factor: float = 10.0):
    self.scaling = np.divide(subtract(ub, lb, dtype=self._dtype.dtype),
                 factor, dtype=self._dtype.dtype)
    if any(np.isinf(self.scaling)):
      for k, x in enumerate(np.isinf(self.scaling)):
        if x:
          self.scaling[k][0] = 1.0
    s_array = np.diag(self.scaling)

    self.scaling = []
    for k in range(len(s_array)):
      temp: List[float] = []
      for j in range(len(s_array[k])):
        temp.append(s_array[k][j])
      self.scaling.append(temp)
      del temp
  
  def gauss_perturbation(self, p: Point, npts: int = 5) -> List[Point]:
    lb = self.lb
    ub = self.ub
    # np.random.seed(self.seed)
    cs = np.zeros((npts, p.n_dimensions))
    pts: List[Point] = [0] * npts
    mp = 1.
    for k in range(p.n_dimensions):
      if p.var_type[k] == VAR_TYPE.CONTINUOUS:
        cs[:, k] = np.random.normal(loc=p.coordinates[k], scale=self.mesh.msize, size=(npts,))
      elif p.var_type[k] == VAR_TYPE.INTEGER or p.var_type[k] == VAR_TYPE.CATEGORICAL or p.var_type[k] == VAR_TYPE.DISCRETE:
        cs[:, k] = np.random.randint(low=lb[k], high=ub[k], size=(npts,))
        for i in range(npts):
          cs[i, k] = int(cs[i, k])
      for i in range(npts):
        if cs[i, k] < lb[k]:
          cs[i, k] = lb[k]
        if cs[i, k] > ub[k]:
          cs[i, k] = ub[k]
    
    for i in range(npts):
      pts[i] = p
      pts[i].coordinates = copy.deepcopy(cs[i, :])
    
    return pts


  def eval_poll_point(self, index: int):
    """ Evaluate the point i on the poll set """
    """ Set the dynamic index for this point """
    tic = time.perf_counter()
    self.point_index = index
    """ Initialize stopping and success conditions"""
    stop: bool = False
    """ Copy the point i to a trial one """
    xtry: Point = self.poll_dirs[index]
    """ This is a success bool parameter used for
     filtering out successful designs to be printed
    in the output results file"""
    success = False

    """ Check the cache memory; check if the trial point
     is a duplicate (it has already been evaluated) """
    unique_p_trials: int = 0
    is_duplicate: bool = (self.check_cache and self.hashtable.size > 0 and self.hashtable.is_duplicate(xtry))
    while is_duplicate and unique_p_trials < 5:
      if self.display:
        print(f'Cache hit. Trial# {unique_p_trials}: Looking for a non-duplicate in the vicinity of the duplicate point ...')
      if xtry.var_type is None:
        xtry.var_type = self.xmin.var_type
      xtries: List[Point] = self.gauss_perturbation(p=xtry, npts=len(self.poll_dirs)*2)
      for tr in range(len(xtries)):
        is_duplicate = self.hashtable.is_duplicate(xtries[tr])
        if is_duplicate:
           continue 
        else:
          xtry = copy.deepcopy(xtries[tr])
          break
      unique_p_trials += 1

    if (is_duplicate):
      if self.display:
        print("Cache hit ... Failed to find a non-duplicate alternative.")
      stop = True
      bb_eval = copy.deepcopy(self.bb_eval)
      psize = copy.deepcopy(self.mesh.psize)
      return [stop, index, self.bb_handle.bb_eval, success, psize, xtry]

    """ Evaluation of the blackbox; get output responses """
    if xtry.sets is not None and isinstance(xtry.sets,dict):
      p: List[Any] = []
      for i in range(len(xtry.var_type)):
        if (xtry.var_type[i] == VAR_TYPE.DISCRETE or xtry.var_type[i] == VAR_TYPE.CATEGORICAL) and xtry.var_link[i] is not None:
          p.append(xtry.sets[xtry.var_link[i]][int(xtry.coordinates[i])])
        else:
          p.append(xtry.coordinates[i])
      self.bb_output = self.bb_handle.eval(p)
    else:
      self.bb_output = self.bb_handle.eval(xtry.coordinates)

    """
      Evaluate the poll point:
        - Set multipliers and penalty
        - Evaluate objective function
        - Evaluate constraint functions (can be an empty vector)
        - Aggregate constraints
        - Penalize the objective (extreme barrier)
    """
    xtry.LAMBDA = copy.deepcopy(self.LAMBDA)
    xtry.RHO = copy.deepcopy(self.RHO)
    xtry.hmax = copy.deepcopy(self.hmax)
    xtry.__eval__(self.bb_output)
    toc = time.perf_counter()
    xtry.Eval_time = (toc - tic)

    """ Update multipliers and penalty """
    if self.LAMBDA == None:
      self.LAMBDA = self.xmin.LAMBDA
    if len(xtry.c_ineq) > len(self.LAMBDA):
      self.LAMBDA += [self.LAMBDA[-1]] * abs(len(self.LAMBDA)-len(xtry.c_ineq))
    if len(xtry.c_ineq) < len(self.LAMBDA):
      del self.LAMBDA[len(xtry.c_ineq):]
    for i in range(len(xtry.c_ineq)):
      if self.RHO == 0.:
        self.RHO = 0.001
      self.LAMBDA[i] = copy.deepcopy(max(self.dtype.zero, self.LAMBDA[i] + (1/self.RHO)*xtry.c_ineq[i]))
    
    if xtry.status == DESIGN_STATUS.FEASIBLE:
      self.RHO *= copy.deepcopy(0.5)

    # if xtry < self.xmin:
    #   self.success = True
    #   success = True

    """ Add to the cache memory """
    if self.store_cache:
      self.hashtable.hash_id = xtry

    if self.save_results or self.display:
      self.bb_eval = self.bb_handle.bb_eval
      self.psize = copy.deepcopy(self.mesh.psize)
    psize = copy.deepcopy(self.mesh.psize)



    if self.success and self.opportunistic and self.iter > 1:
      stop = True

    """ Check stopping criteria """
    if self.bb_eval >= self.eval_budget:
      self.terminate = True
      stop = True
      return [stop, index, self.bb_handle.bb_eval, success, psize, xtry]

    return [stop, index, self.bb_handle.bb_eval, success, psize, xtry]

  def master_updates(self, x: List[Point], peval, save_all_best: bool = False, save_all:bool = False):
    if peval >= self.eval_budget:
      self.terminate = True
    x_post: List[Point] = []
    for xtry in x:
      """ Check success conditions """
      is_infeas_dom: bool = (self.xmin.status == xtry.status == DESIGN_STATUS.INFEASIBLE and (xtry.h < self.xmin.h and xtry.h <= xtry.hmax and xtry.fobj <= self.xmin.fobj) )
      is_feas_dom: bool = (self.xmin.status == xtry.status == DESIGN_STATUS.FEASIBLE and xtry < self.xmin)
      is_infea_improving: bool = (self.xmin.status == DESIGN_STATUS.FEASIBLE and xtry.status == DESIGN_STATUS.INFEASIBLE and (xtry.fobj < self.xmin.fobj and xtry.h <= xtry.hmax))
      is_feas_improving: bool = (self.xmin.status == DESIGN_STATUS.INFEASIBLE and xtry.status == DESIGN_STATUS.FEASIBLE and (xtry.fobj < self.xmin.fobj))
      success = False
      if (xtry.is_EB_passed and (is_infeas_dom or is_feas_dom or is_infea_improving or is_feas_improving)):
        self.success = True
        success = True  # <- This redundant variable is important
        # for managing concurrent parallel execution
        self.nb_success += 1
        """ Update the post instant """
        del self._xmin
        self._xmin = Point()
        self._xmin = copy.deepcopy(xtry)
        if self.display:
          if self._dtype.dtype == np.float64:
            print(f"Success: fmin = {self.xmin.f:.15f} (hmin = {self.xmin.h:.15})")
          elif self._dtype.dtype == np.float32:
            print(f"Success: fmin = {self.xmin.f:.6f} (hmin = {self.xmin.h:.6})")
          else:
            print(f"Success: fmin = {self.xmin.f:.18f} (hmin = {self.xmin.h:.18})")

        self.mesh.psize_success = copy.deepcopy(self.mesh.psize)
        self.mesh.psize_max = copy.deepcopy(np.maximum(self.mesh.psize,
                              self.mesh.psize_max,
                              dtype=self._dtype.dtype))
      if (save_all_best and success) or (save_all):
        x_post.append(xtry)

    return x_post




# TODO: More methods and parameters will be added to the
#  'pre_mads' class when OMADS is used for solving MDO
#  subproblems
@dataclass
class PreMADS:
  """ Preprocessor for setting up optimization settings and parameters"""
  data: Dict[Any, Any]

  def initialize_from_dict(self, xs: Point):
    """ MADS initialization """
    """ 1- Construct the following classes by unpacking
     their respective dictionaries from the input JSON file """
    options = Options(**self.data["options"])
    param = Parameters(**self.data["param"])
    B = Barrier(param)
    ev = Evaluator(**self.data["evaluator"])
    ev.dtype.precision = options.precision
    if param.constants != None:
      ev.constants = copy.deepcopy(param.constants)
    
    if param.constraints_type is not None and isinstance(param.constraints_type, list):
      for i in range(len(param.constraints_type)):
        if param.constraints_type[i] == BARRIER_TYPES.PB.name or param.constraints_type[i] == BARRIER_TYPES.PB:
          param.constraints_type[i] = BARRIER_TYPES.PB
        elif param.constraints_type[i] == BARRIER_TYPES.RB.name or param.constraints_type[i] == BARRIER_TYPES.RB:
          param.constraints_type[i] = BARRIER_TYPES.RB
        elif param.constraints_type[i] == BARRIER_TYPES.PEB.name or param.constraints_type[i] == BARRIER_TYPES.PEB:
          param.constraints_type[i] = BARRIER_TYPES.PEB
        else:
          param.constraints_type[i] = BARRIER_TYPES.EB
    elif param.constraints_type is not None:
      param.constraints_type = BARRIER_TYPES(param.constraints_type)
  
    iteration: int =  0
    """ 2- Initialize iteration number and construct a point instant for the starting point """
    extend = options.extend is not None and isinstance(options.extend, Dirs2n)
    is_xs = False
    if xs is None or not isinstance(xs, Point) or not xs.evaluated:
      x_start = Point()
    else:
      x_start = xs
      is_xs = True

    if not extend:
      """ 3- Construct an instant for the poll 2n orthogonal directions class object """
      poll = Dirs2n()
      if param.Failure_stop != None and isinstance(param.Failure_stop, bool):
        poll.Failure_stop = param.Failure_stop
      poll.dtype.precision = options.precision
      """ 4- Construct an instant for the mesh subclass object by inheriting
      initial parameters from mesh_params() """
      poll.mesh = OrthoMesh()
      """ 5- Assign optional algorithmic parameters to the constructed poll instant  """
      poll.opportunistic = options.opportunistic
      poll.seed = options.seed
      poll.mesh.dtype.precision = options.precision
      poll.mesh.psize = options.psize_init
      poll.eval_budget = options.budget
      poll.store_cache = options.store_cache
      poll.check_cache = options.check_cache
      poll.display = options.display
    else:
      poll = options.extend
    
    n_available_cores = cpu_count()
    if options.parallel_mode and options.np > n_available_cores:
      options.np == n_available_cores
    """ 6- Initialize blackbox handling subclass by copying
     the evaluator 'ev' instance to the poll object"""
    poll.bb_handle = ev
    poll.bb_handle.bb_eval = ev.bb_eval
    """ 7- Evaluate the starting point """
    if options.display:
      print(" Evaluation of the starting points")
    if not is_xs:
      x_start.coordinates = param.baseline
      x_start.sets = param.var_sets
      if param.constraints_type is not None and isinstance(param.constraints_type, list):
        x_start.constraints_type = [xb for xb in param.constraints_type]
      elif param.constraints_type is not None:
        x_start.constraints_type = [param.constraints_type]
    """ 8- Set the variables type """
    if param.var_type is not None:
      c= 0
      x_start.var_type = []
      x_start.var_link = []
      for k in param.var_type:
        c+= 1
        if k.lower()[0] == "r":
          x_start.var_type.append(VAR_TYPE.CONTINUOUS)
          x_start.var_link.append(None)
        elif k.lower()[0] == "i":
          x_start.var_type.append(VAR_TYPE.INTEGER)
          x_start.var_link.append(None)
        elif k.lower()[0] == "d":
          x_start.var_type.append(VAR_TYPE.DISCRETE)
          if x_start.sets is not None and isinstance(x_start.sets, dict):
            if x_start.sets[k.split('_')[1]] is not None:
              x_start.var_link.append(k.split('_')[1])
              if param.ub[c-1] > len(x_start.sets[k.split('_')[1]])-1:
                param.ub[c-1] = len(x_start.sets[k.split('_')[1]])-1
              if param.lb[c-1] < 0:
                param.lb[c-1] = 0
            else:
              x_start.var_link.append(None)
        elif k.lower()[0] == "c":
          x_start.var_type.append(VAR_TYPE.CATEGORICAL)
          if x_start.sets is not None and isinstance(x_start.sets, dict):
            if x_start.sets[k.split('_')[1:][0]] is not None:
              x_start.var_link.append(k.split('_')[1])
            else:
              x_start.var_link.append(None)
        elif k.lower()[0] == "o":
          x_start.var_type.append(VAR_TYPE.ORDINAL)
          x_start.var_link.append(None)
          # TODO: Implementation in progress
        elif k.lower()[0] == "b":
          x_start.var_type.append(VAR_TYPE.BINARY)
        else:
          x_start.var_type.append(VAR_TYPE.CONTINUOUS)
          x_start.var_link.append(None)

    
    x_start.dtype.precision = options.precision
    if x_start.sets is not None and isinstance(x_start.sets,dict):
      p: List[Any] = []
      for i in range(len(x_start.var_type)):
        if (x_start.var_type[i] == VAR_TYPE.DISCRETE or x_start.var_type[i] == VAR_TYPE.CATEGORICAL) and x_start.var_link[i] is not None:
          p.append(x_start.sets[x_start.var_link[i]][int(x_start.coordinates[i])])
        else:
          p.append(x_start.coordinates[i])
      if not is_xs:
        poll.bb_output = poll.bb_handle.eval(p)
    else:
       if not is_xs:
        poll.bb_output = poll.bb_handle.eval(x_start.coordinates)
    x_start.hmax = B._h_max
    x_start.RHO = param.RHO
    if param.LAMBDA is None:
      param.LAMBDA = [0] * len(x_start.c_ineq)
    if not isinstance(param.LAMBDA, list):
      param.LAMBDA = [param.LAMBDA]
    if len(x_start.c_ineq) > len(param.LAMBDA):
      param.LAMBDA += [param.LAMBDA[-1]] * abs(len(param.LAMBDA)-len(x_start.c_ineq))
    if len(x_start.c_ineq) < len(param.LAMBDA):
      del param.LAMBDA[len(x_start.c_ineq):]
    x_start.LAMBDA = param.LAMBDA
    if not is_xs:
      x_start.__eval__(poll.bb_output)
    """ 9- Copy the starting point object to the poll's  minimizer subclass """
    if not extend:
      poll.xmin = copy.deepcopy(x_start)
    """ 10- Hold the starting point in the poll
     directions subclass and define problem parameters"""
    poll.poll_dirs.append(x_start)
    poll.scale(ub=param.ub, lb=param.lb, factor=param.scaling)
    poll.dim = x_start.n_dimensions
    if not extend:
      poll.hashtable = Cache()
    """ 10- Initialize the number of successful points
     found and check if the starting minimizer performs better
    than the worst (f = inf) """
    poll.nb_success = 0
    if not extend and poll.xmin < Point():
      poll.mesh.psize_success = poll.mesh.psize
      poll.mesh.psize_max = maximum(poll.mesh.psize,
                      poll.mesh.psize_max,
                      dtype=poll.dtype.dtype)
      poll.poll_dirs = [poll.xmin]
    elif extend and x_start < poll.xmin:
      poll.xmin = copy.deepcopy(x_start)
      poll.mesh.psize = np.multiply(poll.mesh.psize, 2, dtype=poll.dtype.dtype)
    elif extend and x_start >= poll.xmin:
      poll.mesh.psize = np.divide(poll.mesh.psize, 2, dtype=poll.dtype.dtype)


    """ 11- Construct the results postprocessor class object 'post' """
    post = PostMADS(x_incumbent=[poll.xmin], xmin=poll.xmin, poll_dirs=[poll.xmin])
    post.psize.append(poll.mesh.psize)
    post.bb_eval.append(poll.bb_handle.bb_eval)
    post.iter.append(iteration)

    """ Note: printing the post will print a results row
     within the results table shown in Python console if the
    'display' option is true """
    # if options.display:
    #     print(post)
    """ 12- Add the starting point hash value to the cache memory """
    if options.store_cache:
      poll.hashtable.hash_id = x_start
    """ 13- Initialize the output results file object  """
    out = Output(file_path=param.post_dir, vnames=param.var_names, pname=param.name, runfolder=f'{param.name}_run')
    if options.display:
      print("End of the evaluation of the starting points")
    iteration += 1

    return iteration, x_start, poll, options, param, post, out, B


@dataclass
class Output:
  """ Results output file decorator
  """
  file_path: str
  vnames: List[str]
  file_writer: Any = field(init=False)
  field_names: List[str] = field(default_factory=list)
  pname: str = "MADS0"
  runfolder: str = "undefined"
  replace: bool = True
  stepName: str = "Poll"

  def __post_init__(self):
    if not os.path.exists(self.file_path):
      os.mkdir(self.file_path)
    self.field_names = [f'{"Runtime (Sec)".rjust(25)}', f'{"Iteration".rjust(25)}', f'{"Evaluation #".rjust(25)}', f'{"Step:".rjust(25)}', f'{"Source".rjust(25)}', f'{"Model_name".rjust(25)}', f'{"Delta".rjust(25)}', f'{"Status".rjust(25)}', f'{"phi".rjust(25)}', f'{"fobj".rjust(25)}', f'{"max(c_in)".rjust(25)}', f'{"Penalty_parameter".rjust(25)}', f'{"Multipliers".rjust(25)}', f'{"hmax".rjust(25)}']
    for k in self.vnames:
      self.field_names.append(f'{f"{k}".rjust(25)}')
    sp = os.path.join(self.file_path, self.runfolder)
    if not os.path.exists(sp):
      os.makedirs(sp)
    if self.replace:
      with open(os.path.abspath( sp + f'/{self.pname}.csv'), 'w', newline='') as f:
        self.file_writer = csv.DictWriter(f, fieldnames=self.field_names)
        self.file_writer.writeheader()

  def add_row(self, eval_time: int, iterno: int,
        evalno: int,
        source: str,
        Mname: str,
        poll_size: float,
        status: str,
        fobj: float,
        h: float, f: float, rho: float, L: List[float], hmax: float,
        x: List[float], stepName: str):
    row = {f'{"Runtime (Sec)".rjust(25)}': f'{f"{eval_time}".rjust(25)}', f'{"Iteration".rjust(25)}': f'{f"{iterno}".rjust(25)}', f'{"Evaluation #".rjust(25)}': f'{f"{evalno}".rjust(25)}', f'{"Step:".rjust(25)}': f'{f"{stepName}".rjust(25)}', f'{"Source".rjust(25)}': f'{f"{source}".rjust(25)}', f'{"Model_name".rjust(25)}': f'{f"{Mname}".rjust(25)}', f'{"Delta".rjust(25)}': f'{f"{poll_size}".rjust(25)}', f'{"Status".rjust(25)}': f'{f"{status}".rjust(25)}', f'{"phi".rjust(25)}': f'{f"{f}".rjust(25)}', f'{"fobj".rjust(25)}': f'{f"{fobj}".rjust(25)}', f'{"max(c_in)".rjust(25)}': f'{f"{h}".rjust(25)}', f'{"Penalty_parameter".rjust(25)}': f'{f"{rho}".rjust(25)}', f'{"Multipliers".rjust(25)}': f'{f"{max(L)}".rjust(25)}', f'{"hmax".rjust(25)}': f'{f"{hmax}".rjust(25)}'}
    # row = {'Iter no.': iterno, 'Eval no.': evalno,
    #      'poll_size': poll_size, 'hmin': h, 'fmin': f}
    ss = 0
    for k in range(14, len(self.field_names)):
      row[self.field_names[k]] = f'{f"{x[ss]}".rjust(25)}'
      ss += 1
    with open(os.path.abspath(os.path.join(os.path.join(self.file_path, self.runfolder), f'{self.pname}.csv')),
          'a', newline='') as File:
      self.file_writer = csv.DictWriter(File, fieldnames=self.field_names)
      self.file_writer.writerow(row)


@dataclass
class PostMADS:
  """ Results postprocessor
  """
  x_incumbent: List[Point]
  xmin: Point
  coords: List[List[Point]] = field(default_factory=list)
  poll_dirs: List[Point] = field(default_factory=list)
  iter: List[int] = field(default_factory=list)
  bb_eval: List[int] = field(default_factory=list)
  psize: List[float] = field(default_factory=list)
  step_name: List[str] = None
  def output_results(self, out: Output):
    """ Create a results file from the saved cache"""
    counter = 0
    for p in self.poll_dirs:
      if p.evaluated and counter < len(self.iter):
        out.add_row(eval_time= p.Eval_time,
              iterno=self.iter[counter],
              evalno=self.bb_eval[counter], poll_size=self.psize[counter],
              source=p.source,
              Mname=p.Model,
              f=p.f,
              status=p.status.name,
              h=max(p.c_ineq),
              fobj=p.fobj,
              rho=p.RHO,
              L=p.LAMBDA,
              x=p.coordinates,
              hmax=p.hmax, stepName="Poll-2n" if self.step_name is None else self.step_name[counter])
        counter += 1

  def output_coordinates(self, out: Output):
    """ Save spinners in a json file """
    with open(out.file_path + "/coords.json", "w") as json_file:
      dict_out = {}
      for ii, ob in enumerate(self.coords, start=1):
        entry: Dict[Any, Any] = {}
        p = [ib.coordinates for ib in ob]
        entry['iter'] = ii
        entry['coord'] = p
        entry['x_incumbent'] = self.x_incumbent[ii - 1].coordinates
        dict_out[ii] = entry
        del entry
        json.dump(dict_out, json_file, indent=4, sort_keys=True)

  def __str__(self):
    return f'{"iteration= "} {self.iter[-1]}, {"bbeval= "} ' \
         f'{self.bb_eval[-1]}, {"psize= "} {self.psize[-1]}, ' \
         f'{"hmin = "} 'f'{self.xmin.h}, {"status:"} {self.xmin.status.name} {", fmin = "} {self.xmin.f}'

  def __add_to_cache__(self, x: Point):
    self.x_incumbent.append(x)


def main(*args) -> Dict[str, Any]:
  """ Otho-MADS main algorithm """
  # TODO: add more checks for more defensive code

  """ Parse the parameters files """
  if type(args[0]) is dict:
    data = args[0]
  elif isinstance(args[0], str):
    if os.path.exists(os.path.abspath(args[0])):
      _, file_extension = os.path.splitext(args[0])
      if file_extension == ".json":
        try:
          with open(args[0]) as file:
            data = json.load(file)
        except ValueError:
          raise IOError('invalid json file: ' + args[0])
      else:
        raise IOError(f"The input file {args[0]} is not a JSON dictionary. "
                f"Currently, OMADS supports JSON files solely!")
    else:
      raise IOError(f"Couldn't find {args[0]} file!")
  else:
    raise IOError("The first input argument couldn't be recognized. "
            "It should be either a dictionary object or a JSON file that holds "
            "the required input parameters.")

  """ Run preprocessor for the setup of
   the optimization problem and for the initialization
  of optimization process """
  iteration, xmin, poll, options, param, post, out, B = PreMADS(data).initialize_from_dict()
  out.stepName = "Poll"
  


  """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))

  """ Start the count down for calculating the runtime indicator """
  tic = time.perf_counter()
  peval = 0
  LAMBDA_k = xmin.LAMBDA
  RHO_k = xmin.RHO
  while True:
    poll.mesh.update()
    """ Create the set of poll directions """
    hhm = poll.create_housholder(options.rich_direction, domain=xmin.var_type)
    poll.lb = param.lb
    poll.ub = param.ub
    if B is not None:
      if B._filter is not None:
        B.select_poll_center()
        B.update_and_reset_success()
      else:
        B.insert(xmin)
    poll.hmax = B._h_max
    poll.create_poll_set(hhm=hhm,
               ub=param.ub,
               lb=param.lb, it=iteration, var_type=xmin.var_type, var_sets=xmin.sets, var_link = xmin.var_link, c_types=param.constraints_type, is_prim=True)
    
    if B._sec_poll_center is not None and B._sec_poll_center.evaluated:
      poll.x_sc = B._sec_poll_center
      poll.create_poll_set(hhm=hhm,
               ub=param.ub,
               lb=param.lb, it=iteration, var_type=B._sec_poll_center.var_type, var_sets=B._sec_poll_center.sets, var_link = B._sec_poll_center.var_link, c_types=param.constraints_type, is_prim=False)
    
    poll.LAMBDA = LAMBDA_k
    poll.RHO = RHO_k

    """ Save current poll directions and incumbent solution
     so they can be saved later in the post dir """
    if options.save_coordinates:
      post.coords.append(poll.poll_dirs)
      post.x_incumbent.append(poll.xmin)
    """ Reset success boolean """
    poll.success = False
    """ Reset the BB output """
    poll.bb_output = []
    xt = []
    """ Serial evaluation for points in the poll set """
    if not options.parallel_mode:
      for it in range(len(poll.poll_dirs)):
        peval += 1
        if poll.terminate:
          break
        f = poll.eval_poll_point(it)
        xt.append(f[-1])
        if not f[0]:
          post.bb_eval.append(poll.bb_handle.bb_eval)
          post.iter.append(iteration)
          post.psize.append(poll.mesh.psize)
        else:
          continue

    else:
      poll.point_index = -1
      """ Parallel evaluation for points in the poll set """
      with concurrent.futures.ProcessPoolExecutor(options.np) as executor:
        results = [executor.submit(poll.eval_poll_point,
                       it) for it in range(len(poll.poll_dirs))]
        for f in concurrent.futures.as_completed(results):
          # if f.result()[0]:
          #     executor.shutdown(wait=False)
          # else:
          if options.save_results or options.display:
            peval = peval +1
            poll.bb_eval = peval
            post.bb_eval.append(peval)
            post.iter.append(iteration)
            # post.poll_dirs.append(poll.poll_dirs[f.result()[1]])
            post.psize.append(f.result()[4])
          xt.append(f.result()[-1])

    xpost: List[Point] = poll.master_updates(xt, peval, save_all_best=options.save_all_best, save_all=options.save_results)
    if options.save_results:
      for i in range(len(xpost)):
        post.poll_dirs.append(xpost[i])
    for xv in xt:
      if xv.evaluated:
        B.insert(xv)

    """ Update the xmin in post"""
    post.xmin = copy.deepcopy(poll.xmin)

    """ Updates """
    pev = 0.
    for p in poll.poll_dirs:
      if p.evaluated:
        pev += 1
    # if pev != poll.poll_dirs and not poll.success:
    #   poll.seed += 1
    goToSearch: bool = (pev == 0 and poll.Failure_stop is not None and poll.Failure_stop)
    if poll.success and not goToSearch:
      poll.mesh.psize = np.multiply(poll.mesh.psize, 2, dtype=poll.dtype.dtype)
    else:
      poll.mesh.psize = np.divide(poll.mesh.psize, 2, dtype=poll.dtype.dtype)

    if options.display:
      print(post)
    
    LAMBDA_k = poll.LAMBDA
    RHO_k = poll.RHO
    
    Failure_check = iteration > 0 and poll.Failure_stop is not None and poll.Failure_stop and (not poll.success or goToSearch)
    
    if (Failure_check or poll.bb_eval >= options.budget) or (abs(poll.mesh.psize) < options.tol or poll.bb_eval >= options.budget or poll.terminate):
      break
    iteration += 1
    

  toc = time.perf_counter()

  """ If benchmarking, then populate the results in the benchmarking output report """
  if len(args) > 1 and isinstance(args[1], toy.Run):
    b: toy.Run = args[1]
    if b.test_suite == "uncon":
      ncon = 0
    else:
      ncon = len(poll.bb_output[1])
    if len(poll.bb_output) > 0:
      b.add_row(name=poll.bb_handle.blackbox,
            run_index=int(args[2]),
            nv=len(param.baseline),
            nc=ncon,
            nb_success=poll.nb_success,
            it=iteration,
            BBEVAL=poll.bb_eval,
            runtime=toc - tic,
            feval=poll.bb_handle.bb_eval,
            hmin=poll.xmin.h,
            fmin=poll.xmin.f)
    print(f"{poll.bb_handle.blackbox}: fmin = {poll.xmin.f:.2f} , hmin= {poll.xmin.h:.2f}")

  elif len(args) > 1 and not isinstance(args[1], toy.Run):
    raise IOError("Could not find " + args[1] + " in the internal BM suite.")

  if options.save_results:
    post.output_results(out)

  if options.display:
    print(" end of orthogonal MADS ")
    print(" Final objective value: " + str(poll.xmin.f) + ", hmin= " + str(poll.xmin.h))

  if options.save_coordinates:
    post.output_coordinates(out)
  if options.display:
    print("\n ---Run Summary---")
    print(f" Run completed in {toc - tic:.4f} seconds")
    print(f" Random numbers generator's seed {options.seed}")
    print(" xmin = " + str(poll.xmin))
    print(" hmin = " + str(poll.xmin.h))
    print(" fmin = " + str(poll.xmin.f))
    print(" #bb_eval = " + str(poll.bb_eval))
    print(" #iteration = " + str(iteration))
    print(" nb_success = " + str(poll.nb_success))
    print(" psize = " + str(poll.mesh.psize))
    print(" psize_success = " + str(poll.mesh.psize_success))
    print(" psize_max = " + str(poll.mesh.psize_max))
  xmin = poll.xmin
  """ Evaluation of the blackbox; get output responses """
  if xmin.sets is not None and isinstance(xmin.sets,dict):
    p: List[Any] = []
    for i in range(len(xmin.var_type)):
      if (xmin.var_type[i] == VAR_TYPE.DISCRETE or xmin.var_type[i] == VAR_TYPE.CATEGORICAL) and xmin.var_link[i] is not None:
        p.append(xmin.sets[xmin.var_link[i]][int(xmin.coordinates[i])])
      else:
        p.append(xmin.coordinates[i])
  else:
    p = xmin.coordinates
  output: Dict[str, Any] = {"xmin": p,
                "fmin": poll.xmin.f,
                "hmin": poll.xmin.h,
                "nbb_evals" : poll.bb_eval,
                "niterations" : iteration,
                "nb_success": poll.nb_success,
                "psize": poll.mesh.psize,
                "psuccess": poll.mesh.psize_success,
                "pmax": poll.mesh.psize_max,
                "msize": poll.mesh.msize}

  return output, poll

def rosen(x, p, *argv):
  x = np.asarray(x)
  y = [np.sum(p[0] * (x[1:] - x[:-1] ** p[1]) ** p[1] + (1 - x[:-1]) ** p[1],
        axis=0), [0]]
  return y

def alpine(x):
  y = [abs(x[0]*np.sin(x[0])+0.1*x[0])+abs(x[1]*np.sin(x[1])+0.1*x[1]), [0]]
  return y

def Ackley3(x):
  return [-200*np.exp(-0.2*np.sqrt(x[0]**2+x[1]**2))+5*np.exp(np.cos(3*x[0])+np.sin(3*x[1])), [0]]

def eggHolder(individual):
  x = individual[0]
  y = individual[1]
  f = (-(y + 47.0) * np.sin(np.sqrt(abs(x/2.0 + (y + 47.0)))) - x * np.sin(np.sqrt(abs(x - (y + 47.0)))))
  return [f, [0]]

if __name__ == "__main__":
  freeze_support()
  p_file: str = os.path.abspath("")

  """ Check if an input argument is provided"""
  if len(sys.argv) > 1:
    p_file = os.path.abspath(sys.argv[1])
    main(p_file)

  if (p_file != "" and os.path.exists(p_file)):
    main(p_file)

  if p_file == "":
    raise IOError("Undefined input args."
            " Please specify an appropriate input (parameters) jason file")

