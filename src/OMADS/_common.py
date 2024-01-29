
from dataclasses import dataclass, field
import logging
import operator
import time
import shutil
import os
from typing import List, Dict, Any, Optional
from numpy import sum, subtract, add, maximum, minimum, power, inf
import numpy as np
from .Point import Point
import csv
import json
from ._globals import *
from BMDFO import toy
from inspect import signature
import subprocess

@dataclass
class logger:
  log: None = None
  isVerbose: bool = False

  def initialize(self, file: str, wTime = False, isVerbose = False):
    # Create and configure logger 
    self.isVerbose = isVerbose
    logging.basicConfig(filename=file, 
              format='%(message)s', 
              filemode='w') 

    #Let us Create an object 
    self.log = logging.getLogger() 

    #Now we are going to Set the threshold of logger to DEBUG 
    self.log.setLevel(logging.DEBUG) 
    cur_time = time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime())
    self.log_msg(msg=f"###################################################### \n", msg_type=MSG_TYPE.INFO)
    self.log_msg(msg=f"################# OMADS ver. 2401 #################### \n", msg_type=MSG_TYPE.INFO)
    self.log_msg(msg=f"############### {cur_time} ################# \n", msg_type=MSG_TYPE.INFO)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create and configure logger 
    if wTime:
      logging.basicConfig(filename=file, 
                format='%(asctime)s %(message)s', 
                filemode='a') 
    else:
      logging.basicConfig(filename=file, 
                format='%(message)s', 
                filemode='a') 

    # Let us Create an object 
    self.log = None
    self.log = logging.getLogger() 

    # Now we are going to Set the threshold of logger to DEBUG 
    self.log.setLevel(logging.DEBUG) 
  
  def log_msg(self, msg: str, msg_type: MSG_TYPE):
    if msg_type == MSG_TYPE.DEBUG:
      self.log.debug(msg) 
    elif msg_type == MSG_TYPE.INFO:
      self.log.info(msg) 
    elif msg_type == MSG_TYPE.WARNING:
      self.log.warning(msg) 
    elif msg_type == MSG_TYPE.ERROR:
      self.log.error(msg) 
    elif msg_type == MSG_TYPE.CRITICAL:
      self.log.critical(msg) 
  
  def relocate_logger(self, source_file: str = None, Dest_file: str = None):
    if Dest_file is not None and source_file is not None and os.path.exists(source_file):
      shutil.copy(source_file, Dest_file)
      if os.path.exists("DSMToDMDO.yaml"):
        shutil.copy("DSMToDMDO.yaml", Dest_file)
      # Remove all handlers associated with the root logger object.
      for handler in logging.root.handlers[:]:
          logging.root.removeHandler(handler)
      # Create and configure logger 
      logging.basicConfig(filename=os.path.join(Dest_file, "DMDO.log"), 
                format='%(asctime)s %(message)s', 
                filemode='a')
      #Let us Create an object 
      self.log = logging.getLogger() 

      #Now we are going to Set the threshold of logger to DEBUG 
      self.log.setLevel(logging.DEBUG) 

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
  isVerbose: bool = False

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
    self.field_names = [f'{"Runtime (Sec)".rjust(25)}', f'{"Iteration".rjust(25)}', f'{"Evaluation #".rjust(25)}', f'{"Step".rjust(25)}', f'{"Source".rjust(25)}', f'{"Model_name".rjust(25)}', f'{"Delta".rjust(25)}', f'{"Status".rjust(25)}', f'{"phi".rjust(25)}', f'{"fobj".rjust(25)}', f'{"max(c_in)".rjust(25)}', f'{"Penalty_parameter".rjust(25)}', f'{"Multipliers".rjust(25)}', f'{"hmax".rjust(25)}']
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
    row = {f'{"Runtime (Sec)".rjust(25)}': f'{f"{eval_time}".rjust(25)}', f'{"Iteration".rjust(25)}': f'{f"{iterno}".rjust(25)}', f'{"Evaluation #".rjust(25)}': f'{f"{evalno}".rjust(25)}', f'{"Step".rjust(25)}': f'{f"{stepName}".rjust(25)}', f'{"Source".rjust(25)}': f'{f"{source}".rjust(25)}', f'{"Model_name".rjust(25)}': f'{f"{Mname}".rjust(25)}', f'{"Delta".rjust(25)}': f'{f"{poll_size}".rjust(25)}', f'{"Status".rjust(25)}': f'{f"{status}".rjust(25)}', f'{"phi".rjust(25)}': f'{f"{f}".rjust(25)}', f'{"fobj".rjust(25)}': f'{f"{fobj}".rjust(25)}', f'{"max(c_in)".rjust(25)}': f'{f"{h}".rjust(25)}', f'{"Penalty_parameter".rjust(25)}': f'{f"{rho}".rjust(25)}', f'{"Multipliers".rjust(25)}': f'{f"{max(L) if len(L)>0 else None}".rjust(25)}', f'{"hmax".rjust(25)}': f'{f"{hmax}".rjust(25)}'}
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
    with open(out.file_path + f"/{out.runfolder}/coords.json", "w") as json_file:
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
    """ Initialize the log file """
    return f'{"iteration= "} {self.iter[-1]}, {"bbeval= "} ' \
         f'{self.bb_eval[-1]}, {"psize= "} {self.psize[-1]}, ' \
         f'{"hmin = "} 'f'{self.xmin.h}, {"status: "} {self.xmin.status.name} {", fmin = "} {self.xmin.f}'

  def __add_to_cache__(self, x: Point):
    self.x_incumbent.append(x)

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
            elif (npar == 2 and ('*argv' not in inputs)):
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
  _best_hash_ID: List[int] = field(default_factory=list)
  _cache_dict: Dict[Any, Any] = field(default_factory=lambda: {})
  _n_dim: int = 0

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
    if hash(tuple(other.coordinates)) not in self._hash_ID:
      self._hash_ID.append(hash(tuple(other.coordinates)))
  
  @property
  def best_hash_ID(self)->List[int]:
    """A getter to return the list of hash IDs

    :rtype: List[int]
    """
    return self._best_hash_ID

  @best_hash_ID.setter
  def best_hash_ID(self, id: int):
    self._best_hash_ID.append(id)

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
    
  
  def add_to_best_cache(self, x: Point):
    if not isinstance(x, list):
      if len(self._cache_dict) > 1:
        is_infeas_dom: bool = (x.status == DESIGN_STATUS.INFEASIBLE and (x.h < self._cache_dict[self._best_hash_ID[0]].h) )
        is_feas_dom: bool = (x.status == DESIGN_STATUS.FEASIBLE and x.fobj < self._cache_dict[self._best_hash_ID[0]].fobj)
      else:
        is_infeas_dom: bool = False
        is_feas_dom: bool = False
      if len(self._cache_dict) == 1 or is_infeas_dom or is_feas_dom:
        self._n_dim = len(x.coordinates)
        self._best_hash_ID.append(self._hash_ID[-1])
    else:
      for i in range(len(x)):
        is_infeas_dom: bool = (x[i].status == DESIGN_STATUS.INFEASIBLE and (x[i].h < self._cache_dict[self._best_hash_ID[0]].h) )
        is_feas_dom: bool = (x[i].status == DESIGN_STATUS.FEASIBLE and x[i].fobj < self._cache_dict[self._best_hash_ID[0]].fobj)
        if len(self._cache_dict) == 1 or is_infeas_dom or is_feas_dom:
          self._n_dim = len(x[i].coordinates)
          self._best_hash_ID.append(self._hash_ID[-1])
  
  def get_best_cache_points(self, nsamples):
    """ Get best points """
    temp = np.zeros((nsamples, self._n_dim))
    index = 0
    # for i in range(len(self._best_hash_ID)-1, len(self._best_hash_ID) - nsamples, -1):
    #   temp[index, :] = self._cache_dict[self._best_hash_ID[i]].coordinates
    #   index += 1

    cache_temp = dict(sorted(self._cache_dict.items(), key=operator.itemgetter(1)))

    for k in cache_temp:
      if index < len(temp):
        temp[index, :] = cache_temp[k].coordinates
        index += 1
      else:
        break
    return temp
  
  def get_cache_points(self):
    """ Get best points """
    temp = np.zeros((len(self._hash_ID)-1, self._n_dim))
    for i in range(1, len(self._hash_ID)):
      temp[i-1, :] = self._cache_dict[self._hash_ID[i]].coordinates
    return temp
  
  def get_point(self, key):
    return self._cache_dict[key]
