import copy
import importlib
from ._globals import *
import os
from typing import List, Dict, Any, Optional, Callable
from numpy import sum, subtract, add, maximum, minimum, power, inf
import numpy as np
from inspect import signature
import concurrent.futures
import subprocess
import paramiko
import logging
from .CandidatePoint import CandidatePoint
from .Options import Options
from .PostProcess import PostMADS
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
  _dtype: DType = None
  timeout: float = 1000000.
  local_exec_jobs: List[str] = None



  def __post_init__(self):
    self._dtype = DType()
  
  def run_callable_serial_local(self, iter:int, peval: int, eval_set:List[CandidatePoint], callFunc: Callable, options: Options, post: PostMADS, psize: List[float], stepName: str = None, mesh: auto = None):
    xt: List[CandidatePoint] = []
    for it in range(len(eval_set)):
      peval += 1
      f = callFunc(it)
      if f[-1].status != DESIGN_STATUS.UNEVALUATED:
        xt.append(f[-1])
        if mesh:
          xt[-1].mesh = copy.deepcopy(mesh)
        if options.opportunistic and it > 0 and xt[-1] < eval_set[it-1]:
          break
      if not f[0]:
        post.bb_eval.append(peval)
        xt[-1].evalNo = peval
        post.iter.append(iter)
        if stepName:
          post.step_name.append(stepName)
        post.psize.append(psize)
      else:
        continue
    return xt, post, peval
  

  def run_callable_parallel_local(self, iter:int, peval: int, njobs:int, eval_set:List[CandidatePoint], callFunc: Callable, options: Options, post: PostMADS, mesh: auto = None, stepName: str = None, eval_call: Callable = None):
    bb_eval = []
    xt: List[CandidatePoint] = []
    with concurrent.futures.ProcessPoolExecutor(options.np) as executor:
      results = [executor.submit(callFunc, it) for it in range(len(eval_set))]
      for f in concurrent.futures.as_completed(results):
        # if f.result()[0]:
        #     executor.shutdown(wait=False)
        # else:
        if options.save_results or options.display:
          peval = peval +1
          if not f.result()[0]:
            if eval_call:
              x, psize = eval_call(f.result()[-1], f.result()[5])
              xt.append(x)
              if mesh:
                xt[-1].mesh = copy.deepcopy(mesh)
              xt[-1].evalNo = self.bb_eval
              
            self.bb_eval = peval
            bb_eval = peval
            post.bb_eval.append(peval)
            post.iter.append(iter)
            # post.poll_dirs.append(poll.poll_dirs[f.result()[1]])
            if stepName:
              post.step_name.append(stepName)
            post.psize.append(f.result()[4] if eval_call is None else psize)
        if f.result()[-1].status != DESIGN_STATUS.UNEVALUATED and eval_call is None:
          xt.append(f.result()[-1])
          if mesh:
            xt[-1].mesh = copy.deepcopy(mesh)
          xt[-1].evalNo = self.bb_eval
    return bb_eval, xt, post, peval

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
      
      stdin, stdout, stderr = ssh.exec_command(exe_path)
      
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
    elif importlib.util.find_spec('BMDFO') and self.internal == "uncon":
      f_eval = toy.UnconSO(values)
    elif importlib.util.find_spec('BMDFO') and self.internal == "con":
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
