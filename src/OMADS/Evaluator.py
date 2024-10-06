import copy
import importlib
import platform
import time
from ._globals import DType, VAR_TYPE, DESIGN_STATUS, BB_EVAL_STATUS
import os
from typing import List, Any, Optional, Callable
from numpy import inf
import numpy as np
from inspect import signature
import concurrent.futures
import subprocess
import paramiko
import logging
from .CandidatePoint import CandidatePoint
from .Options import Options
from .PostProcess import PostMADS
from .Point import Point
from dataclasses import dataclass
if importlib.util.find_spec('BMDFO'):
  from BMDFO import toy
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
  command_options: Any = None
  internal: Optional[str] = None
  path: str = "..\\tests\\Rosen"
  input: str = "input.inp"
  output: str = "output.out"
  constants: Optional[List] = None
  bb_eval: int = 0
  _dtype: Optional[DType] = None
  timeout: float = 1000000.
  local_exec_jobs: Optional[List[str]] = None
  candidates: Optional[List[Point]] = None
  directions: Optional[List[Point]] = None
  mesh: List[Any] = None
  constraints_relaxation: Optional[dict] = None
  xmin: Optional[CandidatePoint] = None
  



  def __post_init__(self):
    self._dtype = DType()

  def map_variables(self, eval_set: List[CandidatePoint]):
    self.candidates = []
    self.directions = []
    self.mesh = []
    for xtry in eval_set:
      if xtry.sets is not None and isinstance(xtry.sets,dict):
        p: List[Any] = []
        for i in range(len(xtry.var_type)):
          if (xtry.var_type[i] == VAR_TYPE.DISCRETE or xtry.var_type[i] == VAR_TYPE.CATEGORICAL) and xtry.var_link[i] is not None:
            p.append(xtry.sets[xtry.var_link[i]][int(xtry.coordinates[i])])
          else:
            p.append(xtry.coordinates[i])
        temp_p: Point = Point()
        temp_p.coordinates = p
      else:
        temp_p: Point = Point()
        temp_p.coordinates = copy.deepcopy(xtry.coordinates)
      self.candidates.append(temp_p)
      self.directions.append(xtry.direction)
      self.mesh.append(xtry.mesh)
  
  def run_callable_serial_local(self, iter:int, peval: int, eval_set:List[CandidatePoint], options: Options, post: PostMADS, psize: List[float], step_name: str = None, mesh: Any = None, constraints_relaxation: dict = None, budget:int = 1):
    xc: List[CandidatePoint] = []
    self.map_variables(eval_set)
    self.constraints_relaxation = copy.deepcopy(constraints_relaxation)
    for it in range(len(eval_set)):
      peval += 1
      f = self.evaluate_blackbox(it)
      if f.status != BB_EVAL_STATUS.UNEVALUATED:
        xc.append(f)
        if mesh:
          xc[-1].mesh = copy.deepcopy(mesh)
        
      post.bb_eval.append(peval)
      xc[-1].eval_no = peval
      post.iter.append(iter)
      if step_name:
        post.step_name.append(step_name)
      post.psize.append(psize)
      if options.opportunistic and len(xc) > 0 and xc[-1] < self.xmin:
        break
      if peval == budget:
        break
    return xc, post, peval
  
  def evaluate_blackbox(self, index: int)->List[Any]:
    f, err_status = self.eval(self.candidates[index].coordinates)
    x_cp: CandidatePoint = CandidatePoint()
    x_cp.coordinates = copy.deepcopy(self.candidates[index].coordinates)
    x_cp.lambda_multipliers = copy.deepcopy(self.constraints_relaxation["LAMBDA"])
    x_cp.rho = copy.deepcopy(self.constraints_relaxation["RHO"])
    x_cp.h_max = copy.deepcopy(self.constraints_relaxation["hmax"])
    x_cp.constraints_type = copy.deepcopy(self.constraints_relaxation["constraints_type"])
    x_cp.direction = copy.deepcopy(self.directions[index])
    x_cp.mesh = copy.deepcopy(self.mesh[index])
    x_cp.__eval__(f)
    if err_status:
      x_cp.status = DESIGN_STATUS.ERROR
    # if x_cp.status == DESIGN_STATUS.INFEASIBLE:
      # self.constraintsRelaxation["hmax"] = x_cp.hmax
    if self.constraints_relaxation["LAMBDA"] == None:
      self.constraints_relaxation["LAMBDA"] = copy.deepcopy(self.xmin.lambda_multipliers)
    if len(x_cp.cPB) > len(self.constraints_relaxation["LAMBDA"]):
      self.constraints_relaxation["LAMBDA"] += [self.constraints_relaxation["LAMBDA"][-1]] * abs(len(self.constraints_relaxation["LAMBDA"])-len(x_cp.cPB))
    if len(x_cp.cPB) < len(self.constraints_relaxation["LAMBDA"]):
      del self.constraints_relaxation["LAMBDA"][len(x_cp.cPB):]
    for i in range(len(x_cp.cPB)):
      if np.isclose(self.constraints_relaxation["RHO"], 0., rtol=1e-09, atol=1e-09):
        self.constraints_relaxation["RHO"] = 0.001
      self.constraints_relaxation["LAMBDA"][i] = copy.deepcopy(max(self.dtype.zero, self.constraints_relaxation["LAMBDA"][i] + (1/self.constraints_relaxation["RHO"])*x_cp.cPB[i]))
    
    if x_cp.status == DESIGN_STATUS.FEASIBLE:
      self.constraints_relaxation["RHO"] *= copy.deepcopy(0.5)
    
    return x_cp

  def run_callable_parallel_local(self, iter:int, peval: int, eval_set:List[CandidatePoint], options: Options, post: PostMADS, psize: List[float], mesh: Any = None, step_name: str = None, constraints_relaxation: dict = None, budget:int = 1):
    xc: List[CandidatePoint] = []
    self.map_variables(eval_set)
    self.constraints_relaxation = copy.deepcopy(constraints_relaxation)
    with concurrent.futures.ProcessPoolExecutor(max_workers=options.np) as executor:
      results = [executor.submit(self.evaluate_blackbox, it) for it in range(len(eval_set))]
      for f in concurrent.futures.as_completed(results):
        # if f.result()[0]:
        #     executor.shutdown(wait=False)
        # else:
        peval = peval +1
        if f.result().status != DESIGN_STATUS.UNEVALUATED:
          xc.append(f.result())
          if mesh:
            xc[-1].mesh = copy.deepcopy(mesh)
          
          xc[-1].eval_no = self.bb_eval
          self.bb_eval = peval
          post.bb_eval.append(peval)
          post.iter.append(iter)
          # post.poll_dirs.append(poll.poll_dirs[f.result()[1]])
          if step_name:
            post.step_name.append(step_name)
          post.psize.append(psize)

          if options.opportunistic and len(xc) > 0 and xc[-1] < self.xmin:
            break
          if peval == budget:
            break
        else:
          executor.shutdown(wait=False)
    
    return peval, xc, post, peval

  # Function to execute .exe file locally
  def run_exe(self, exe_path):
    try:
      result = subprocess.run(exe_path, capture_output=True, text=True, shell=True)
      return (exe_path, result.returncode, result.stdout, result.stderr)
    except Exception as e:
      return (exe_path, -1, '', str(e))

  # Function to execute .exe file on a remote node
  def execute_on_remote(self, host, username, password, exe_path):
    try:
      ssh = paramiko.SSHClient()
      ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
      ssh.connect(host, username=username, password=password)
      
      _, stdout, stderr = ssh.exec_command(exe_path)
      
      output = stdout.read().decode()
      error = stderr.read().decode()
      
      return (host, output, error)
    except Exception as e:
      return (host, '', str(e))
    finally:
      ssh.close()
  
  # Function to run .exe files locally
  def run_locally(self, exe_paths):
    with concurrent.futures.ProcessPoolExecutor() as executor:
      futures = [executor.submit(self.run_exe, exe) for exe in exe_paths]
      for future in concurrent.futures.as_completed(futures):
        exe_path, returncode, stdout, stderr = future.result()
        print(f"Local Executable: {exe_path}")
        print(f"Return Code: {returncode}")
        print(f"Output: {stdout}")
        print(f"Error: {stderr}")

  def run_remotely(self, hosts, username, password, exe_path):
    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [executor.submit(self.execute_on_remote, host, username, password, exe_path) for host in hosts]
      for future in concurrent.futures.as_completed(futures):
        host, output, error = future.result()
        print(f"Remote Host: {host}")
        print(f"Output: {output}")
        print(f"Error: {error}")

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
    evalerr = False
    if self.internal is None or self.internal == "None" or self.internal == "none":
      if callable(self.blackbox):
        is_object = False
        try:
          sig = signature(self.blackbox)
        except Warning:
          is_object = True
        if not is_object:
          npar = len(sig.parameters) 
          # Get input arguments defined for the callable 
          inputs = str(sig).replace("(", "").replace(")", "").replace(" ","").split(',')
          # Check if user constants list is defined and if the number of input args of the callable matches what OMADS expects 
          if self.constants is None:
            is_argv = '*argv' in inputs
            if (npar == 1 or (npar> 0 and npar <= 3 and is_argv)) or (npar == 2 and is_argv):
              try:
                f_eval = self.blackbox(values)
              except Warning:
                evalerr = True
                logging.error(f"Callable {str(self.blackbox)} evaluation returned an error at the poll point {values}")
                f_eval = [inf, [inf]]
            else:
              raise IOError(f'The callable {str(self.blackbox)} requires {npar} input args, but only one input can be provided! You can introduce other input parameters to the callable function using the constants list.')
          else:
            if (npar == 2 or (npar> 0 and npar <= 3 and ('*argv' in inputs))):
              try:
                f_eval = self.blackbox(values, self.constants)
              except Warning:
                evalerr = True
                logging.error(f"Callable {str(self.blackbox)} evaluation returned an error at the poll point {values}")
            else:
              raise IOError(f'The callable {str(self.blackbox)} requires {npar} input args, but only two input args can be provided as the constants list is defined!')
        else:
          try:
            f_eval = self.blackbox(values)
          except Warning:
            evalerr = True
            logging.error(f"Callable {str(self.blackbox)} evaluation returned an error at the poll point {values}")
            f_eval = [[inf], [inf]]
        if isinstance(f_eval, list):
          return f_eval, evalerr
        elif isinstance(f_eval, float) or isinstance(f_eval, int):
          return [[f_eval], [0]], evalerr
      else:
        self.write_input(values)
        pwd = os.getcwd()
        os.chdir(self.path)
        is_win = platform.platform().split('-')[0] == 'Windows'
        evalerr = False
        timouterr = False
        #  Check if the file is executable
        executable = os.access(self.blackbox, os.X_OK)
        if not executable:
          raise IOError(f"The blackbox file {str(self.blackbox)} is not an executable! Please provide a valid executable file.")
        # Prepare the execution command based on the running machine's OS
        if is_win and self.command_options is None:
          cmd = self.blackbox
        elif is_win:
          cmd = f'{self.blackbox} {self.command_options}'
        elif self.command_options is None:
          cmd = f'./{self.blackbox}'
        else:
          cmd =  f'./{self.blackbox} {self.command_options}'
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
        return out, evalerr
    elif importlib.util.find_spec('BMDFO') and self.internal == "uncon":
      f_eval = toy.UnconSO(values) # type: ignore
    elif importlib.util.find_spec('BMDFO') and self.internal == "con":
      f_eval = toy.ConSO(values) # type: ignore
    else:
      raise IOError(f"Input dict:: evaluator:: internal:: "
              f"Incorrect internal method :: {self.internal} :: "
              f"it should be a a BM library name, "
              f"or None.")
    f_eval.dtype.dtype = self._dtype.dtype
    f_eval.name = self.blackbox
    f_eval.dtype.dtype = self._dtype.dtype
    return getattr(f_eval, self.blackbox)(), evalerr

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
