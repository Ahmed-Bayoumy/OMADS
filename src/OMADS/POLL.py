
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
  # TODO: support more variable types
  # TODO: give better control on variabls' resolution (mesh granularity)
  # var_type: List[str] = field(default_factory=["cont", "cont"])
  # resolution: List[int] = field(default_factory=["cont", "cont"])


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
        sig = signature(self.blackbox)
        npar = len(sig.parameters) 
        # Get input arguments defined for the callable 
        inputs = str(sig).replace("(", "").replace(")", "").replace(" ","").split(',')
        # Check if user constants list is defined and if the number of input args of the callable matches what OMADS expects 
        if self.constants is None:
          if (npar == 1 or (npar> 0 and npar <= 2 and ('*argv' in inputs))):
            try:
              f_eval = self.blackbox(values)
            except:
              evalerr = True
              logging.error(f"Callable {str(self.blackbox)} evaluation returned an error at the poll point {values}")
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
  def freal(self):
    return self._freal

  @freal.setter
  def freal(self, other: float):
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
    return (other.h > self._dtype.zero > self.__dh__(other=other)) or \
         ((self._dtype.zero > self.h >= 0.0) and
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
    self.freal = bb_output[0]
    """ Inequality constraints (can be an empty vector) """
    self.c_ineq = bb_output[1]
    """ Aggregate constraints """
    self.h = sum(power(maximum(self.c_ineq, self._dtype.zero,
                   dtype=self._dtype.dtype), 2, dtype=self._dtype.dtype))
    if np.isnan(self.h) or np.any(np.isnan(self.c_ineq)):
      self.h = inf
    """ Penalize the objective """
    if np.isnan(self.f) or self.h > self._dtype.zero:
      self.__penalize__()

  def __penalize__(self):
    self.f = inf

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
  _delta: float = -1.0  # mesh size
  _Delta: float = -1.0  # poll size
  _rho: float = -1.0  # poll size to mesh size ratio
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
    return x.signature in self.hash_id

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
    if not self.is_duplicate(x):
      hash_value: int = hash(tuple(x.coordinates))
      self._cache_dict[hash_value] = x


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

  def create_housholder(self, is_rich: bool) -> np.ndarray:
    """Create householder matrix

    :param is_rich:  A flag that indicates if the rich direction option is enabled
    :type is_rich: bool
    :return: The householder matrix
    :rtype: np.ndarray
    """
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
    hhm = np.dot(np.vstack((hhm, -hhm)), self.scaling)

    return hhm

  def create_poll_set(self, hhm: np.ndarray, ub: List[float], lb: List[float], it: int):
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
    del self.poll_dirs
    temp = np.add(hhm, np.array(self.xmin.coordinates), dtype=self._dtype.dtype)
    # np.random.seed(self._seed)
    temp = np.random.permutation(temp)
    temp = np.minimum(temp, ub, dtype=self._dtype.dtype)
    temp = np.maximum(temp, lb, dtype=self._dtype.dtype)

    for k in range(2 * self.dim):
      tmp = Point()
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

  def eval_poll_point(self, index: int):
    """ Evaluate the point i on the poll set """
    """ Set the dynamic index for this point """
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
    if (
        self.check_cache
        and self.hashtable.size > 0
        and self.hashtable.is_duplicate(xtry)
    ):
      if self.display:
        print("Cache hit ...")
      stop = True
      bb_eval = copy.deepcopy(self.bb_eval)
      psize = copy.deepcopy(self.mesh.psize)
      return [stop, index, self.bb_handle.bb_eval, success, psize, xtry]

    """ Evaluation of the blackbox; get output responses """
    self.bb_output = self.bb_handle.eval(xtry.coordinates)

    """
      Evaluate the poll point:
        - Evaluate objective function
        - Evaluate constraint functions (can be an empty vector)
        - Aggregate constraints
        - Penalize the objective (extreme barrier)
    """
    xtry.__eval__(self.bb_output)

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

  def master_updates(self, x: List[Point], peval):
    if peval >= self.eval_budget:
      self.terminate = True
    for xtry in x:
      """ Check success conditions """
      if xtry < self.xmin:
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
        self.mesh.psize_max = copy.deepcopy(maximum(self.mesh.psize,
                              self.mesh.psize_max,
                              dtype=self._dtype.dtype))




# TODO: More methods and parameters will be added to the
#  'pre_mads' class when OMADS is used for solving MDO
#  subproblems
@dataclass
class PreMADS:
  """ Preprocessor for setting up optimization settings and parameters"""
  data: Dict[Any, Any]

  def initialize_from_dict(self):
    """ MADS initialization """
    """ 1- Construct the following classes by unpacking
     their respective dictionaries from the input JSON file """
    options = Options(**self.data["options"])
    param = Parameters(**self.data["param"])
    ev = Evaluator(**self.data["evaluator"])
    ev.dtype.precision = options.precision
    """ 2- Initialize iteration number and construct a point instant for the starting point """
    iteration: int = 0
    x_start = Point()
    """ 3- Construct an instant for the poll 2n orthogonal directions class object """
    poll = Dirs2n()
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
    n_available_cores = cpu_count()
    if options.parallel_mode and options.np > n_available_cores:
      options.np == n_available_cores
    """ 6- Initialize blackbox handling subclass by copying
     the evaluator 'ev' instance to the poll object"""
    poll.bb_handle = copy.deepcopy(ev)
    """ 7- Evaluate the starting point """
    if options.display:
      print(" Evaluation of the starting points")
    x_start.coordinates = param.baseline
    x_start.dtype.precision = options.precision
    poll.bb_output = poll.bb_handle.eval(x_start.coordinates)
    x_start.__eval__(poll.bb_output)
    """ 8- Copy the starting point object to the poll's  minimizer subclass """
    poll.xmin = copy.deepcopy(x_start)
    """ 9- Hold the starting point in the poll
     directions subclass and define problem parameters"""
    poll.poll_dirs.append(x_start)
    poll.scale(ub=param.ub, lb=param.lb, factor=param.scaling)
    poll.dim = x_start.n_dimensions
    poll.hashtable = Cache()
    """ 10- Initialize the number of successful points
     found and check if the starting minimizer performs better
    than the worst (f = inf) """
    poll.nb_success = 0
    if poll.xmin < Point():
      poll.mesh.psize_success = poll.mesh.psize
      poll.mesh.psize_max = maximum(poll.mesh.psize,
                      poll.mesh.psize_max,
                      dtype=poll.dtype.dtype)
      poll.poll_dirs = [poll.xmin]
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
    out = Output(file_path=param.post_dir, vnames=param.var_names)
    if options.display:
      print("End of the evaluation of the starting points")
    iteration += 1

    return iteration, x_start, poll, options, param, post, out


@dataclass
class Output:
  """ Results output file decorator
  """
  file_path: str
  vnames: List[str]
  file_writer: Any = field(init=False)
  field_names: List[str] = field(default_factory=list)

  def __post_init__(self):
    if not os.path.exists(self.file_path):
      os.mkdir(self.file_path)
    self.field_names = ['Iter no.', 'Eval no.', 'poll_size', 'hmin', 'fmin']
    for k in self.vnames:
      self.field_names.append(k)
    with open(os.path.abspath(self.file_path + '/MADS.csv'), 'w', newline='') as f:
      self.file_writer = csv.DictWriter(f, fieldnames=self.field_names)
      self.file_writer.writeheader()

  def add_row(self, iterno: int,
        evalno: int,
        poll_size: float,
        h: float, f: float,
        x: List[float]):
    row = {'Iter no.': iterno, 'Eval no.': evalno,
         'poll_size': poll_size, 'hmin': h, 'fmin': f}
    for k in range(5, len(self.field_names)):
      row[self.field_names[k]] = x[k - 5]
    with open(os.path.abspath(os.path.join(self.file_path, 'MADS.csv')),
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

  def output_results(self, out: Output):
    """ Create a results file from the saved cache"""
    counter = 0
    for p in self.poll_dirs:
      out.add_row(iterno=self.iter[counter],
            evalno=self.bb_eval[counter], poll_size=self.psize[counter],
            h=p.h,
            f=p.f,
            x=p.coordinates)
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
         f'{"hmin = "} 'f'{self.xmin.h}, {"fmin = "} {self.xmin.f}'

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
  iteration, xmin, poll, options, param, post, out = PreMADS(data).initialize_from_dict()

  """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))

  """ Start the count down for calculating the runtime indicator """
  tic = time.perf_counter()
  peval = 0
  while True:
    poll.mesh.update()
    """ Create the set of poll directions """
    hhm = poll.create_housholder(options.rich_direction)
    poll.create_poll_set(hhm=hhm,
               ub=param.ub,
               lb=param.lb, it=iteration)
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
          if options.save_results:
            if options.save_all_best and not f[3]:
              continue
            post.poll_dirs.append(poll.poll_dirs[it])
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
            if options.save_all_best and not f.result()[3]:
              continue
            peval = peval +1
            poll.bb_eval = peval
            post.bb_eval.append(peval)
            post.iter.append(iteration)
            post.poll_dirs.append(poll.poll_dirs[f.result()[1]])
            post.psize.append(f.result()[4])
          xt.append(f.result()[-1])

    poll.master_updates(xt, peval)

    """ Update the xmin in post"""
    post.xmin = copy.deepcopy(poll.xmin)

    """ Updates """
    if poll.success:
      poll.mesh.psize = np.multiply(poll.mesh.psize, 2, dtype=poll.dtype.dtype)
    else:
      poll.mesh.psize = np.divide(poll.mesh.psize, 2, dtype=poll.dtype.dtype)

    if options.display:
      print(post)

    if abs(poll.mesh.psize) < options.tol or poll.bb_eval >= options.budget or poll.terminate:
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
  output: Dict[str, Any] = {"xmin": poll.xmin.coordinates,
                "fmin": poll.xmin.f,
                "hmin": poll.xmin.h,
                "nbb_evals" : poll.bb_eval,
                "niterations" : iteration,
                "nb_success": poll.nb_success,
                "psize": poll.mesh.psize,
                "psuccess": poll.mesh.psize_success,
                "pmax": poll.mesh.psize_max}

  return output

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