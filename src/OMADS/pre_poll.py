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

from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import copy

from ._globals import VAR_TYPE, MSG_TYPE, DESIGN_STATUS
from .candidate_point import CandidatePoint
from .barriers import Barrier, BarrierMO
from .omesh import Omesh
from .directions import Dirs2n
from .parameters import Parameters
from .options import Options
from .evaluator import Evaluator
from .postprocess import PostMADS, Output
from ._common import logger
from .gmesh import Gmesh
from .cache import Cache


@dataclass
class PrePoll:
  """ Preprocessor for setting up optimization settings and parameters"""
  data: Dict[Any, Any]
  log: Optional[logger] = None

  def set_variables_type(self, is_xs: bool, xs: List[float],
                         x_start: CandidatePoint,
                         param: Parameters) -> CandidatePoint:
    if not is_xs:
      x_start.coordinates = xs
      x_start.sets = param.var_sets
      if param.constraints_type is not None and isinstance(
              param.constraints_type, list):
        x_start.constraints_type = [xb for xb in param.constraints_type]
      elif param.constraints_type is not None:
        x_start.constraints_type = [param.constraints_type]

    # """ 8- Set the variables type """
    if param.var_type is not None and not is_xs:
      c = 0
      x_start.var_type = []
      x_start.var_link = []
      for k in param.var_type:
        c += 1
        if k.lower()[0] == "r":
          x_start.var_type.append(VAR_TYPE.REAL)
          x_start.var_link.append(None)
        elif k.lower()[0] == "i":
          x_start.var_type.append(VAR_TYPE.INTEGER)
          x_start.var_link.append(None)
        elif k.lower()[0] == "d":
          x_start.var_type.append(VAR_TYPE.DISCRETE)
          if x_start.sets is not None and isinstance(x_start.sets, dict):
            if x_start.sets[k.split('_')[1]] is not None:
              x_start.var_link.append(k.split('_')[1])
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
          # COMPLETED: Implementation in progress
        elif k.lower()[0] == "b":
          x_start.var_type.append(VAR_TYPE.BINARY)
        else:
          raise IOError(
              "Could not recognize the variable of type " + k +
              ". Please use on of the following keywords to identify your variable type:\
                  real, integer, discrete, categorical, ordinal, or binary")
    return x_start

  def poll_preparation_and_initialization(
          self, param: Parameters, x_start: CandidatePoint, is_xs: bool,
          options: Options, poll: Dirs2n, extend: bool, B: Barrier,
          iteration: int):
    """_summary_
    """
    # Get the starting point: best start given a list of coordinates (if any)
    if all(isinstance(item, list) for item in param.baseline):
      bl = param.baseline
    else:
      bl = [param.baseline]
    for xs in bl:
      xs = self.set_variables_type(is_xs=is_xs, xs=xs,
                                   x_start=x_start, param=param)
      if not is_xs:
        x_start.dtype.precision = options.precision
      if x_start.sets is not None and isinstance(x_start.sets, dict):
        p: List[Any] = []
        for i, _ in enumerate((x_start.var_type)):
          if (x_start.var_type[i] == VAR_TYPE.DISCRETE or
                  x_start.var_type[i] == VAR_TYPE.CATEGORICAL) and \
                  x_start.var_link[i] is not None:
            p.append(x_start.sets[x_start.var_link[i]]
                     [int(x_start.coordinates[i])])
          else:
            p.append(x_start.coordinates[i])
        if not is_xs:
          poll.bb_output, _ = poll.bb_handle.eval(p)
      else:
        if not is_xs:
          poll.bb_output, _ = poll.bb_handle.eval(x_start.coordinates)
      x_start.h_max = param.h_max
      x_start.rho = param.rho

      if param.lambda_multipliers is None:
        param.lambda_multipliers = [0] * len(x_start.c_ineq)
      if not isinstance(param.lambda_multipliers, list):
        param.lambda_multipliers = [param.lambda_multipliers]
      if len(x_start.c_ineq) > len(param.lambda_multipliers):
        param.lambda_multipliers += [param.lambda_multipliers[-1]
                                     ] * abs(len(param.lambda_multipliers)-len(x_start.c_ineq))
      if len(x_start.c_ineq) < len(param.lambda_multipliers):
        del param.lambda_multipliers[len(x_start.c_ineq):]
      x_start.lambda_multipliers = param.lambda_multipliers

      if not is_xs:
        x_start.__eval__(poll.bb_output)
        # if isinstance(B, Barrier) or isinstance(B, BarrierMO):
        #   B._h_max = x_start.h_max
      # """ 9- Copy the starting point object to the poll's  minimizer subclass """
      if not extend:
        if x_start.status == DESIGN_STATUS.INFEASIBLE and isinstance(
                B, BarrierMO):
          poll.x_sc = copy.deepcopy(x_start) if (
              poll.x_sc is None or x_start < poll.x_sc) else poll.x_sc
        else:
          poll.xmin = copy.deepcopy(x_start) if (
              poll.xmin is None or not poll.xmin.evaluated or x_start < poll.xmin) else poll.xmin
      # """ 10- Hold the starting point in the poll
      # directions subclass and define problem parameters """
      poll.poll_set.append(x_start)
      poll.scale(ub=param.ub, lb=param.lb, factor=param.scaling)
      poll.dim = x_start.n_dimensions
      if not extend:
        if poll.hashtable is None or poll.hashtable._n_dim == 0:
          poll.hashtable = Cache()
          poll.hashtable._n_dim = len(x_start.coordinates)
          poll.hashtable._is_pareto = param.is_pareto
        if param.is_pareto:
          poll.hashtable.nd_points = []
      # """ 10- Initialize the number of successful points
      # found and check if the starting minimizer performs better
      # than the worst (f = inf) """
      poll.nb_success = 0
      if not extend and poll.xmin.evaluated and poll.xmin < CandidatePoint():
        poll.poll_set = [poll.xmin]
      elif extend and x_start.status == DESIGN_STATUS.FEASIBLE and x_start < poll.xmin:
        poll.xmin = copy.deepcopy(x_start)
        poll.mesh.enlarge_delta_frame_size()
      elif extend and x_start.status == DESIGN_STATUS.INFEASIBLE and x_start < poll.x_sc:
        poll.x_sc = copy.deepcopy(x_start)
      elif extend and x_start.status == DESIGN_STATUS.FEASIBLE and x_start >= poll.xmin:
        poll.mesh.refine_delta_frame_size()
      elif extend and x_start.status == DESIGN_STATUS.INFEASIBLE and x_start >= poll.x_sc:
        poll.mesh.refine_delta_frame_size()

      poll.xmin.mesh = copy.deepcopy(poll.mesh)
      poll.x_sc.mesh = copy.deepcopy(poll.mesh)

      # """ 11- Construct the results postprocessor class object 'post' """
      if poll.xmin.evaluated:
        x_start.eval_no = poll.bb_handle.bb_eval
        poll.xmin.eval_no = poll.bb_handle.bb_eval
        post = PostMADS(
            x_incumbent=[poll.xmin],
            xmin=poll.xmin, poll_dirs=[poll.xmin])
        post.psize.append(poll.mesh.get_delta_frame_size().coordinates)
        post.bb_eval.append(poll.bb_handle.bb_eval)
        x_start.mesh = poll.mesh
        post.iter.append(iteration)
      elif poll.x_sc.evaluated:
        x_start.eval_no = poll.bb_handle.bb_eval
        poll.x_sc.eval_no = poll.bb_handle.bb_eval
        post = PostMADS(
            x_incumbent=[poll.x_sc],
            xmin=poll.x_sc, poll_dirs=[poll.x_sc])
        post.psize.append(poll.mesh.get_delta_frame_size().coordinates)
        post.bb_eval.append(poll.bb_handle.bb_eval)

        post.iter.append(iteration)

      # """ Note: printing the post will print a results row
      # within the results table shown in Python console if the
      # 'display' option is true """
      # """ 12- Add the starting point hash value to the cache memory """
      if options.store_cache:
        poll.hashtable.add_to_cache(x_start)
      # """ 13- Initialize the output results file object  """
      out = Output(
          file_path=param.post_dir, vnames=param.var_names,
          fnames=param.fun_names, pname=param.name,
          runfolder=f'{param.name}_run', suffix="all")
      if param.is_pareto:
        out_p = Output(
            file_path=param.post_dir, vnames=param.var_names,
            fnames=param.fun_names, pname=param.name,
            runfolder=f'{param.name}_ND', suffix="Pareto")
      else:
        out_p = None
      if options.display:
        print("End of the evaluation of the starting points")
        if self.log is not None:
          self.log.log_msg(
              msg="- End of the evaluation of the starting points",
              msg_type=MSG_TYPE.INFO)

      # """ 14- Update the barrier with points """
      B.update_with_points(x_start if isinstance(x_start, list) else [x_start])

      iteration += 1

    return iteration, x_start, poll, options, param, post, out, B, out_p

  def initialize_from_dict(
          self, log: logger = None, xs: CandidatePoint = None):
    # """ MADS initialization """
    # """ 1- Construct the following classes by unpacking
    #  their respective dictionaries from the input JSON file """
    self.log = copy.deepcopy(log)
    if self.log is not None:
      self.log.log_msg(
          msg="---------------- Preprocess the POLL step ----------------",
          msg_type=MSG_TYPE.INFO)
      self.log.log_msg(msg="- Reading the input dictionaries",
                       msg_type=MSG_TYPE.INFO)
    options = Options(**self.data["options"])
    param = Parameters(**self.data["param"])
    log.is_verbose = options.is_verbose
    barrier_defined = BarrierMO(
        param=param, options=options) if param.is_pareto else Barrier(param)
    barrier_defined.h_max = param.h_max
    ev = Evaluator(**self.data["evaluator"])
    if self.log is not None:
      self.log.log_msg(msg="- Set the POLL configurations",
                       msg_type=MSG_TYPE.INFO)
    ev.dtype.precision = options.precision
    if param.constants is not None:
      ev.constants = copy.deepcopy(param.constants)

    iteration: int = 0
    # """ 2- Initialize iteration number and construct a point instant for the starting point """
    extend = options.extend is not None and isinstance(options.extend, Dirs2n)
    is_xs = False
    if xs is None or not isinstance(xs, CandidatePoint) or not xs.evaluated:
      x_start = CandidatePoint()
    else:
      x_start = xs
      is_xs = True

    if not extend:
      # """ 3- Construct an instant for the poll 2n orthogonal directions class object """
      poll = Dirs2n()
      if param.failure_stop is not None and isinstance(
              param.failure_stop, bool):
        poll.failure_stop = param.failure_stop
      poll.dtype.precision = options.precision
      # """ 4- Construct an instant for the mesh subclass object by inheriting
      # initial parameters from mesh_params() """
      # COMPLETED: Add the Gmesh constructor req inputs
      poll.mesh = Gmesh(
          pb_param=param, run_options=options) if (
          param.mesh_type).lower() == "gmesh" else Omesh(
          pb_param=param, run_options=options)
      # """ 5- Assign optional algorithmic parameters to the constructed poll instant  """
      poll.opportunistic = options.opportunistic
      poll.seed = options.seed
      poll.eval_budget = options.budget
      poll.store_cache = options.store_cache
      poll.check_cache = options.check_cache
      poll.display = options.display
      # poll.scaling
    else:
      poll = options.extend

    n_available_cores = cpu_count()
    if options.parallel_mode and options.np > n_available_cores:
      options.np = n_available_cores
    # """ 6- Initialize blackbox handling subclass by copying
    #  the evaluator 'ev' instance to the poll object"""
    poll.bb_handle = ev
    poll.bb_handle.bb_eval = ev.bb_eval
    # """ 7- Evaluate the starting point """
    if options.display:
      print(" Evaluation of the starting points")
      if self.log is not None:
        self.log.log_msg(msg="- Evaluate the starting point",
                         msg_type=MSG_TYPE.INFO)

    return self.poll_preparation_and_initialization(
        param=param, x_start=x_start, is_xs=is_xs, options=options, poll=poll,
        extend=extend, B=barrier_defined, iteration=iteration)
