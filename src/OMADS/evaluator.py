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
from inspect import signature
import concurrent.futures
import subprocess

import logging
import platform
import os
import time
from typing import List, Any

import paramiko

from numpy import inf
import numpy as np
from .postprocess import PostMADS, Output
from .options import Options
from .candidate_point import CandidatePoint
from ._globals import DType, DESIGN_STATUS, PassException
from .optimizer import GenericSamplerBase
from .barriers import AdaptiveBarrier


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
          out: Output, hashtable=None):

    self.candidates = sampler._candidate_points_set
    insertion_flag = [None] * len(sampler._candidate_points_set)
    CS: List[CandidatePoint] = []
    for index, candidate in enumerate(sampler._candidate_points_set):
      tic = time.perf_counter()
      stats.neval_bb += 1
      self.incumbent = active_barrier.elements[parent_indices[index]]

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

  def run_callable_parallel_local(
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
        except PassException:
          is_object = True
        if not is_object:
          npar = len(sig.parameters)
          # Get input arguments defined for the callable
          inputs = str(sig).replace(
              "(", "").replace(
              ")", "").replace(
              " ", "").split(',')
          # Check if user constants list is defined and if the
          # number of input args of the callable matches what OMADS expects
          if self.constants is None:
            is_argv = '*argv' in inputs
            if (npar == 1 or (npar > 0 and npar <= 3 and is_argv)) or (npar == 2 and is_argv):
              try:
                f_eval = self.blackbox(values)
              except PassException:
                evalerr = True
                logging.error(
                    "Callable %s evaluation returned \
                      an error at the poll point %s", str(self.blackbox), values)
                f_eval = [inf, [inf]]
            else:
              raise IOError(
                  f'The callable {str(self.blackbox)} requires {npar} input args, \
                    but only one input can be provided! \
                    You can introduce other input parameters to \
                      the callable function using the constants list.')
          else:
            if (npar == 2 or (npar > 0 and npar <= 3 and ('*argv' in inputs))):
              try:
                f_eval = self.blackbox(values, self.constants)
              except PassException:
                evalerr = True
                logging.error(
                    "Callable %s evaluation returned \
                      an error at the poll point %s", str(self.blackbox), values)
            else:
              raise IOError(
                  f'The callable {str(self.blackbox)} requires {npar} input args, but only two \
                    input args can be provided as the constants list is defined!')
        else:
          try:
            f_eval = self.blackbox(values)
          except PassException:
            evalerr = True
            logging.error(
                "Callable %s evaluation returned \
                  an error at the poll point %s", str(self.blackbox), values)
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
          raise IOError(
              f"The blackbox file {str(self.blackbox)} is not an executable! \
              Please provide a valid executable file.")
        # Prepare the execution command based on the running machine's OS
        if is_win and self.command_options is None:
          cmd = self.blackbox
        elif is_win:
          cmd = f'{self.blackbox} {self.command_options}'
        elif self.command_options is None:
          cmd = f'./{self.blackbox}'
        else:
          cmd = f'./{self.blackbox} {self.command_options}'
        try:
          p = subprocess.run(
              cmd, shell=True, timeout=self.timeout, check=False)
          if p.returncode != 0:
            evalerr = True
            logging.error(
                "Evaluation # {self.bb_eval} is errored at the poll point {values}")
        except subprocess.TimeoutExpired:
          timouterr = True
          logging.error('Timeout for %s(%s s) expired at \
              evaluation  # {%s} at the poll point {values}', cmd, self.timeout, self.bb_eval)

        os.chdir(pwd)

        if evalerr or timouterr:
          out = [np.inf, [np.inf]]
        else:
          out = [self.read_output()[0], [self.read_output()[1:]]]
        return out, evalerr
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
