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

from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional
import copy

from .._include import SAMPLER_TYPE, SUCCESS_TYPES, VAR_TYPE, MSG_TYPE, DType
from .._include import CandidatePoint
from .._include import AdaptiveBarrier
from .._include import Omesh
from .._directions._directions import Dirs2n
from .._setup._parameters import Parameters
from .._setup._options import Options
from .._evaluator._evaluator import Evaluator
from .._post._postprocess import PostMADS, Output
from .._include import logger
from .._include import Gmesh
from .._include import Cache
from .._analytics._metadata import MadsState, MadsStatistics
from .._templates._optimizer import GenericSamplerBase
from .._heuristics._exploration import EfficientExploration, search_sampling


class preprocess:
  """ Preprocessor for setting up optimization settings and parameters"""
  data: Dict[Any, Any] = {}
  log: Optional[logger] = None
  sampler: GenericSamplerBase = None

  def __init__(
          self, data: Dict[Any, Any] = {},
          log: logger = None, sampler_t: SAMPLER_TYPE = None):
    self.data = data
    self.log = log
    if sampler_t == SAMPLER_TYPE.POLL:
      self.sampler = Dirs2n()
    elif sampler_t == SAMPLER_TYPE.SEARCH:
      self.sampler = EfficientExploration()
    else:
      raise IOError(
          "Unknown search type provided in a preprocess instantiation!")

  def set_variables_type(
          self, param: Parameters, x_start: CandidatePoint, is_xs: bool,
          xs: CandidatePoint = None) -> CandidatePoint:
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

  def sampler_preparation_and_initialization(
          self, param: Parameters, x_start: CandidatePoint, is_xs: bool,
          options: Options, B: AdaptiveBarrier, iteration: int, state: MadsState,
          stats: MadsStatistics, ev: Evaluator):
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
          self.sampler.bb_output, _ = ev.eval(p)
      else:
        if not is_xs:
          self.sampler.bb_output, _ = ev.eval(x_start.coordinates)
      x_start.h_max = param.h_max
      x_start.rho = param.rho

      if param.lambda_multipliers is None:
        param.lambda_multipliers = [0] * len(x_start.c_ineq)
      if not isinstance(param.lambda_multipliers, list):
        param.lambda_multipliers = [param.lambda_multipliers]
      if len(x_start.c_ineq) > len(param.lambda_multipliers):
        param.lambda_multipliers += [param.lambda_multipliers[-1]
                                     ] * abs(len(param.lambda_multipliers) - len(x_start.c_ineq))
      if len(x_start.c_ineq) < len(param.lambda_multipliers):
        del param.lambda_multipliers[len(x_start.c_ineq):]
      x_start.lambda_multipliers = param.lambda_multipliers

      if not is_xs:
        x_start.__eval__(self.sampler.bb_output)
        stats.neval_bb += 1
        self.sampler.bb_eval += 1
        x_start.eval_no = stats.neval_bb
      # """ 10- Hold the starting point in the poll
      # directions subclass and define problem parameters """
      self.sampler.candidate_points_set = x_start
      self.sampler.scale(ub=param.ub, lb=param.lb, factor=param.scaling)
      self.sampler.dim = x_start.n_dimensions

      hashtable = Cache(capacity=options.budget)
      hashtable._is_pareto = param.is_pareto

      # """ 10- Initialize the number of successful points
      # found and check if the starting minimizer performs better
      # than the worst (f = inf) """
      self.sampler.nb_success = 0

      post = PostMADS(
          x_incumbent=[x_start],
          xmin=x_start, poll_dirs=[x_start])
      post.psize.append(self.sampler.mesh.get_delta_frame_size().coordinates)
      post.bb_eval.append(self.sampler.bb_eval)
      post.iter.append(iteration)
      post.step_name = []
      post.step_name.append("Poll_2n")

      # """ 11- Construct the results postprocessor class object 'post' """

      # """ Note: printing the post will print a results row
      # within the results table shown in Python console if the
      # 'display' option is true """
      # """ 12- Add the starting point hash value to the cache memory """
      x_start.improving = True
      x_start.was_center = True
      x_start.is_nondominated = True
      if options.store_cache:
        hashtable.add_to_cache(x_start)
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
      stot = x_start if isinstance(x_start, list) else [x_start]
      for xt in stot:
        if xt.is_feasible():
          B.add_feasible(xt, self.sampler.mesh)
        else:
          B.add_infeasible(xt, self.sampler.mesh)

      iteration += 1

      return iteration, x_start, self.sampler, options, param, post, out, B, out_p, state, stats, ev, hashtable

  def initialize_from_dict(self, xs: CandidatePoint = None):
    # """ MADS initialization """
    # """ 1- Construct the following classes by unpacking
    #  their respective dictionaries from the input JSON file """
    self.log = copy.deepcopy(self.log)
    if self.log is not None:
      self.log.log_msg(
          msg="---------------- Preprocess the POLL step ----------------",
          msg_type=MSG_TYPE.INFO)
      self.log.log_msg(msg="- Reading the input dictionaries",
                       msg_type=MSG_TYPE.INFO)
    options = Options(**self.data["options"])
    param = Parameters(**self.data["param"])
    self.log.is_verbose = options.is_verbose
    barrier_defined = AdaptiveBarrier(
        param=param, options=options)  # if param.is_pareto else Barrier(param)
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
      if param.failure_stop is not None and isinstance(
              param.failure_stop, bool):
        self.sampler.failure_stop = param.failure_stop
      self.sampler.dtype = DType()
      self.sampler.dtype.precision = options.precision
      if isinstance(self.sampler, EfficientExploration):
        search_step = search_sampling(**self.data["search"])
        self.sampler.sampling_t = search_step.s_method
        self.sampler.type = search_step.type
        self.sampler.ns = search_step.ns
        self.sampler.sampling_criter = search_step.criterion
        self.sampler.visualize = search_step.visualize
        self.sampler.weights = search_step.weights
      # """ 4- Construct an instant for the mesh subclass object by inheriting
      # initial parameters from mesh_params() """
      # COMPLETED: Add the Gmesh constructor req inputs
      self.sampler.mesh = Gmesh(
          pb_param=param, run_options=options) if (
          param.mesh_type).lower() == "gmesh" else Omesh(
          pb_param=param, run_options=options)
      # """ 5- Assign optional algorithmic parameters to the constructed poll instant  """
      self.sampler.opportunistic = options.opportunistic
      self.sampler.seed = options.seed
      self.sampler.eval_budget = options.budget
      self.sampler.store_cache = options.store_cache
      self.sampler.check_cache = options.check_cache
      self.sampler.display = options.display
      # poll.scaling
    else:
      self.sampler = options.extend

    n_available_cores = cpu_count()
    if options.parallel_mode and options.np > n_available_cores:
      options.np = n_available_cores
    # """ 6- Initialize blackbox handling subclass by copying
    #  the evaluator 'ev' instance to the poll object"""
    self.sampler.bb_eval = ev.bb_eval
    # """ 7- Evaluate the starting point """
    if options.display:
      print(" Evaluation of the starting points")
      if self.log is not None:
        self.log.log_msg(msg="- Evaluate the starting point",
                         msg_type=MSG_TYPE.INFO)
    # x_start.mesh = poll.mesh

    state: MadsState = MadsState()
    state.h_max = param.h_max
    stats: MadsStatistics = MadsStatistics()
    state.last_success = SUCCESS_TYPES.US

    return self.sampler_preparation_and_initialization(
        param=param, x_start=x_start, is_xs=is_xs, options=options,
        B=barrier_defined, iteration=iteration, state=state, stats=stats,
        ev=ev)
