"""
# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                               #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the BSD 3-Clause License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the BSD 3-Clause License for more details.   #
#                                                                                     #
#  You should have received a copy of the BSD 3-Clause License along     #
#  with this program. If not, see <https://opensource.org/license/bsd-3-clause/>.     #
#                                                                                     #
#  You can find information on OMADS at                                               #
#  https://github.com/Ahmed-Bayoumy/OMADS                                             #
#  Copyright (C) 2026  Ahmed H. Bayoumy                                               #
# ------------------------------------------------------------------------------------#
"""
from inspect import signature
import concurrent.futures
import subprocess

import logging
import platform
import os
import time
from typing import List, Any, Tuple

import paramiko

from numpy import inf
from .._post._postprocess import PostMADS, Output
from .._setup._options import Options
from .._include import CandidatePoint
from .._include import DType, DESIGN_STATUS, PassException
from .._templates._optimizer import GenericSamplerBase
from .._include import AdaptiveBarrier


class Evaluator:
  """ Define the evaluator attributes and settings
    :param blackbox: The blackbox name (it can be a callable function or an executable file)
    :param commandOptions: Define options that will be added to the execution command of the
    executable file. Command options should be defined as a string in one line
    :param internal: If the blackbox callable function is part of the internal benchmarking library
    :param path: The path of the executable file (if any)
    :param input: The input file name -- should include the file extension
    :param output: The output file name -- should include the file extension
    :param constants: Define constant parameters list, see the documentation in
    Tutorials->Blackbox evaluation->User parameters
    :param _dtype: The precision delegator of the numpy library
    :param timeout: The time out of the evaluation process
  """

  def __init__(
          self, blackbox="BB", command_options=None, internal=None,
          path="..\\tests\\Rosen", input="input.inp", output="output.out",
          constants=None, bb_eval=0, _dtype=None, timeout=1000000,
          local_exec_jobs=None, candidates=None, directions=None, mesh=None,
          incumbent=None):
    self.blackbox = blackbox
    self.command_options = command_options
    self.internal = internal
    self.path = path
    self.input = input
    self.output = output
    self.constants = constants
    self.bb_eval: int = bb_eval
    self._dtype = DType()
    self.timeout = timeout
    self.local_exec_jobs = local_exec_jobs
    self.candidates = candidates
    self.directions = directions
    self.mesh = mesh
    self.incumbent = incumbent

  def run_callable_serial_local(
          self, sampler: GenericSamplerBase, centers: List[int],
          options: Options, stats: Any, active_barrier: AdaptiveBarrier,
          post: PostMADS, step_name: str, parent_indices: List[int],
          out: Output, hashtable=None, internal: bool = False):

    self.candidates = sampler._candidate_points_set
    insertion_flag = [None] * len(sampler._candidate_points_set)
    for index, candidate in enumerate(sampler._candidate_points_set):
      tic = time.perf_counter()
      stats.neval_bb += 1
      self.incumbent = active_barrier.elements[parent_indices[index]]

      if internal:
        self._internal_dummy_Callable(index)
      else:
        self.evaluate_blackbox(index)
      self.candidates[index].fc_index = parent_indices[index]
      toc = time.perf_counter()
      self.candidates[index].eval_time = toc - tic
      self.candidates[index].eval_no = stats.neval_bb
      hashtable.add_to_cache(self.candidates[index])

      if self.candidates[index].is_feasible():
        insertion_flag[index] = active_barrier.add_feasible(
            self.candidates[index],
            sampler.mesh)
        stats.neval_bb_feasible += 1
      elif self.candidates[index].status == DESIGN_STATUS.INFEASIBLE:
        insertion_flag[index] = active_barrier.add_infeasible(
            self.candidates[index],
            sampler.mesh)
        stats.neval_bb_infeasible += 1

      active_barrier.parent_indexes[active_barrier.last_index] = parent_indices[index]

      post.bb_eval.append(stats.neval_bb)

      post.iter.append(sampler._iter)
      if step_name:
        post.step_name.append(step_name)
      else:
        post.step_name = []
        post.step_name.append(step_name)

      post.psize.append(sampler.mesh.get_delta_frame_size().coordinates)
      post.x_incumbent.append(active_barrier.elements[centers[index]])

      if options.opportunistic and self.candidates[index] < self.incumbent:
        stats.nopportunistic_triggers += 1
        break
      if stats.neval_bb == options.budget:
        break

    # COMPLETED: Output results here
      post.iter.append(sampler._iter)
      post.poll_dirs.append(sampler._candidate_points_set[index])
    sampler._candidate_points_set = self.candidates
    if options.save_coordinates:
      post.coords.append(sampler._candidate_points_set)

    if sampler.prob_params.is_pareto:
      post.output_nd_results(out=out)
    else:
      post.output_results(out=out, all_res=False)

    return insertion_flag, post

  def _internal_dummy_Callable(self, index: int):
    return self.candidates[index]

  def neutral_run_callable_serial_local(
          self, candidate_points_set: List[CandidatePoint],
          stats: Any, active_barrier: AdaptiveBarrier, parent_indices: List[int],
          hashtable=None, internal: bool = False):

    self.candidates = candidate_points_set
    insertion_flag = [None] * len(candidate_points_set)
    for index, candidate in enumerate(candidate_points_set):
      tic = time.perf_counter()
      stats.neval_bb += 1
      self.incumbent = active_barrier.elements[parent_indices[index]]

      self.evaluate_blackbox(index)
      self.candidates[index].fc_index = parent_indices[index]
      toc = time.perf_counter()
      self.candidates[index].eval_time = toc - tic
      self.candidates[index].eval_no = stats.neval_bb
      hashtable.add_to_cache(self.candidates[index])

    candidate_points_set = self.candidates

  def evaluate_blackbox_parallel(self, point: CandidatePoint) -> List[Any]:
    f, err_status = self.eval(point.mapped_coords)
    point.__eval__(f)
    if err_status:
      point.status = DESIGN_STATUS.ERROR
    return point

  def evaluate_blackbox(self, index: int) -> List[Any]:
    f, err_status = self.eval(self.candidates[index].mapped_coords)
    self.candidates[index].__eval__(f)
    if err_status:
      self.candidates[index].status = DESIGN_STATUS.ERROR
    return self.candidates[index]

  def run_callable_parallel_local(  # noqa: C901
          self, sampler: GenericSamplerBase, centers: List[int],
          options: Options, stats: Any, active_barrier: AdaptiveBarrier,
          post: PostMADS, step_name: str, parent_indices: List[int],
          out: Output, hashtable=None):
    self.candidates = sampler.candidate_points_set
    insertion_flag = []
    insertion_flag = [None] * len(sampler._candidate_points_set)

    with concurrent.futures.ProcessPoolExecutor(max_workers=options.np) as executor:
      tic = time.perf_counter()
      future_to_index = {
          executor.submit(self.evaluate_blackbox_parallel, point): (i, point)
          for i, point in enumerate(self.candidates)}
      completed_result = None
      for future in concurrent.futures.as_completed(future_to_index):
        time.sleep(0.1)
        toc = time.perf_counter()
        index, point = future_to_index[future]
        stats.neval_bb += 1
        self.candidates[index] = future.result()
        sampler.candidate_points_set[index] = self.candidates[index]
        sampler.candidate_points_set[index].eval_time = toc - tic
        sampler._candidate_points_set[index].eval_no = stats.neval_bb
        hashtable.add_to_cache(self.candidates[index])
        self.incumbent = active_barrier.elements[parent_indices[index]]

        if sampler.candidate_points_set[index].is_feasible():
          insertion_flag[index] = active_barrier.add_feasible(
              sampler.candidate_points_set[index],
              sampler.mesh)
          stats.neval_bb_feasible += 1
        elif sampler.candidate_points_set[index].status == DESIGN_STATUS.INFEASIBLE:
          insertion_flag[index] = active_barrier.add_infeasible(
              sampler.candidate_points_set[index],
              sampler.mesh)
          stats.neval_bb_infeasible += 1

        active_barrier.parent_indexes[active_barrier.last_index] = parent_indices[index]
        post.x_incumbent.append(active_barrier.elements[centers[index]])

        post.bb_eval.append(stats.neval_bb)
        post.iter.append(sampler._iter)
        if step_name:
          post.step_name.append(step_name)
        else:
          post.step_name = []
          post.step_name.append(step_name)
        post.psize.append(sampler.mesh.get_delta_frame_size().coordinates)
        if options.opportunistic and sampler.candidate_points_set[index] < self.incumbent:
          stats.nopportunistic_triggers += 1
          completed_result = future.result()
          break
        if stats.neval_bb == options.budget:
          completed_result = future.result()
          break
      if (completed_result):
        for f in future_to_index:
          if not f.done():
            f.cancel()

    if options.save_coordinates:
      post.coords.append(sampler.candidate_points_set)

    if sampler.prob_params.is_pareto:
      post.output_nd_results(out=out)
    else:
      post.output_results(out=out, all_res=False)

    return insertion_flag, post

  # Function to execute .exe file locally

  def run_exe(self, exe_path):
    try:
      result = subprocess.run(
          exe_path, capture_output=True, text=True, shell=True, check=False)
      return (exe_path, result.returncode, result.stdout, result.stderr)
    except PassException as e:
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
    except PassException as e:
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
      futures = [
          executor.submit(
              self.execute_on_remote, host, username, password, exe_path)
          for host in hosts]
      for future in concurrent.futures.as_completed(futures):
        host, output, error = future.result()
        print(f"Remote Host: {host}")
        print(f"Output: {output}")
        print(f"Error: {error}")

  @property
  def dtype(self):
    return self._dtype

  def eval(self, values: List[float]) -> Tuple[List[float], bool]:
    """Evaluate the poll point.

    :param values: Poll point coordinates (design vector)
    :return: Evaluated optimization functions and error flag
    """
    self.bb_eval += 1

    # Handle internal method
    if self.internal not in (None, "None", "none"):
      raise IOError(f"Invalid internal method: {self.internal}. "
                    "Must be a BM library name or None.")

    # Handle callable blackbox
    if callable(self.blackbox):
      return self._evaluate_callable(values)

    # Handle external executable
    return self._evaluate_executable(values)

  def _evaluate_callable(self, values: List[float]) -> Tuple[List[float], bool]:
    """Evaluate a callable blackbox function."""
    evalerr = False
    try:
      # Determine expected number of args
      sig = signature(self.blackbox)
      npar = len(sig.parameters)
      inputs = str(sig).replace(
          "(", "").replace(
          ")", "").replace(
          " ", "").split(',')

      # Check if constants are used
      if self.constants is None:
        if not self._is_valid_callable_signature(npar, inputs):
          raise IOError(
              f"Callable {self.blackbox} requires {npar} args, "
              "but only one input can be provided. "
              "Use the constants list for additional parameters.")
        f_eval = self.blackbox(values)
      else:
        if not self._is_valid_callable_with_constants(npar, inputs):
          raise IOError(
              f"Callable {self.blackbox} requires {npar} args, "
              "but only two inputs can be provided with constants.")
        f_eval = self.blackbox(values, self.constants)

      return self._format_evaluation_result(f_eval), evalerr

    except PassException:
      evalerr = True
      logging.error(
          "Callable %s evaluation failed at poll point %s", self.blackbox,
          values)
      return [[inf], [inf]], evalerr

  def _evaluate_executable(self, values: List[float]) -> Tuple[List[float], bool]:
    """Evaluate an external executable."""
    evalerr = False
    timouterr = False
    pwd = os.getcwd()
    os.chdir(self.path)

    try:
      # Check if executable
      if not os.access(self.blackbox, os.X_OK):
        raise IOError(f"Blackbox file {self.blackbox} is not executable.")

      # Build command
      cmd = self._build_command()

      # Run with timeout
      p = subprocess.run(cmd, shell=True, timeout=self.timeout, check=False)
      if p.returncode != 0:
        evalerr = True
        logging.error("Evaluation #%d failed at poll point %s",
                      self.bb_eval, values)

    except subprocess.TimeoutExpired:
      timouterr = True
      logging.error("Timeout (%s s) expired for %s at evaluation #%d",
                    self.timeout, self.blackbox, self.bb_eval)

    finally:
      os.chdir(pwd)

    # Read output
    if evalerr or timouterr:
      return [inf, [inf]], evalerr

    output = self.read_output()
    return [output[0], output[1:]], evalerr

  def _build_command(self) -> str:
    """Build execution command based on OS and options."""
    is_win = platform.platform().split('-')[0] == 'Windows'
    if is_win:
      return self.command_options if self.command_options else self.blackbox
    return f'./{self.blackbox} {self.command_options}' if self.command_options else f'./{self.blackbox}'

  def _is_valid_callable_signature(self, npar: int, inputs: List[str]) -> bool:
    """Check if callable signature matches expected input count."""
    return (npar == 1 or (npar > 0 and npar <= 3 and '*argv' in inputs)) or \
           (npar == 2 and '*argv' in inputs)

  def _is_valid_callable_with_constants(
          self, npar: int, inputs: List[str]) -> bool:
    """Check if callable supports constants."""
    return (npar == 2 or (npar > 0 and npar <= 3 and '*argv' in inputs))

  def _format_evaluation_result(self, f_eval) -> List[float]:
    """Format the evaluation result into expected structure."""
    if isinstance(f_eval, list):
      return f_eval
    if isinstance(f_eval, (float, int)):
      return [[f_eval], [0]]
    raise TypeError(f"Unexpected return type from blackbox: {type(f_eval)}")

  def write_input(self, values: List[float]):
    """_summary_

    :param values: Write the variables in the input file
    :type values: List[float]
    """
    inp = os.path.join(self.path, self.input)
    with open(inp, 'w+', encoding='utf-8') as f:
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
    f = open(out, encoding='utf-8')
    for line in f:  # read rest of lines
      f_eval.append(float(line))
    f.close()
    if len(f_eval) == 1:
      f_eval.append(0.0)
    return f_eval
