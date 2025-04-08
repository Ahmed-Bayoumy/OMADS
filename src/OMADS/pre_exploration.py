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
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import copy

from ._globals import MSG_TYPE, VAR_TYPE
from .exploration import EfficientExploration, search_sampling
from .parameters import Parameters
from .options import Options
from .omesh import Omesh
from .postprocess import PostMADS, Output
from ._common import logger
from .candidate_point import CandidatePoint
from .barriers import BarrierMO, Barrier
from .evaluator import Evaluator
from .gmesh import Gmesh
from .cache import Cache


@dataclass
class PreExploration:
  """ Preprocessor for setting up optimization settings and parameters"""
  data: Dict[Any, Any]
  log: Optional[logger] = None

  def initialize_from_dict(
          self, log: logger = None, xs: CandidatePoint = None):
    """Initialize and setup from a dictionary

    :param log: _description_, defaults to None
    :type log: logger, optional
    :param xs: _description_, defaults to None
    :type xs: CandidatePoint, optional
    :raises IOError: _description_
    :return: _description_
    :rtype: _type_
    """
    # """ MADS initialization """
    # """ 1- Construct the following classes by unpacking
    #  their respective dictionaries from the input JSON file """
    self.log = copy.deepcopy(log)
    if self.log is not None:
      self.log.log_msg(
          msg="---------------- Preprocess the SEARCH step ----------------",
          msg_type=MSG_TYPE.INFO)
      self.log.log_msg(msg="- Reading the input dictionaries",
                       msg_type=MSG_TYPE.INFO)
    options = Options(**self.data["options"])
    param = Parameters(**self.data["param"])
    log.is_verbose = options.is_verbose
    barrier_defined = BarrierMO(
        param=param, options=options) if param.is_pareto else Barrier(param)
    ev = Evaluator(**self.data["evaluator"])
    barrier_defined.h_max = param.h_max
    if self.log is not None:
      self.log.log_msg(msg="- Set the SEARCH configurations",
                       msg_type=MSG_TYPE.INFO)
    search_step = search_sampling(**self.data["search"])
    ev.dtype.precision = options.precision
    if param.constants is not None:
      ev.constants = copy.deepcopy(param.constants)

    # """ 2- Initialize iteration number and construct a point instant for the starting point """
    iteration: int = 0
    # """ 3- Construct an instant for the poll 2n orthogonal directions class object """
    extend = options.extend is not None and isinstance(
        options.extend, EfficientExploration)
    is_xs = False
    if xs is None or not isinstance(xs, CandidatePoint) or not xs.evaluated:
      x_start = CandidatePoint()
    else:
      x_start = xs
      is_xs = True
    if not extend:
      search = EfficientExploration()
      search.prob_params = copy.deepcopy(param)
      if param.failure_stop is not None and isinstance(
              param.failure_stop, bool):
        search.failure_stop = param.failure_stop
      search._candidate_points_set = []
      search.dtype.precision = options.precision
      search.save_results = options.save_results
      # """ 4- Construct an instant for the mesh subclass object by inheriting
      # initial parameters from mesh_params() """
      search.mesh = Gmesh(
          pb_param=param, run_options=options) if (
          param.mesh_type).lower() == "gmesh" else Omesh(
          pb_param=param, run_options=options)

      search.sampling_t = search_step.s_method
      search.type = search_step.type
      search.ns = search_step.ns
      search.sampling_criter = search_step.criterion
      search.visualize = search_step.visualize
      search.weights = search_step.weights
      # """ 5- Assign optional algorithmic parameters to the constructed poll instant  """
      search.opportunistic = options.opportunistic
      search.seed = options.seed
      search.eval_budget = options.budget
      search.store_cache = options.store_cache
      search.check_cache = options.check_cache
      search.display = options.display
      search.bb_eval = 0
      search.prob_params = copy.deepcopy(param)
    else:
      search = options.extend
      search._candidate_points_set = []
    n_available_cores = cpu_count()
    if options.parallel_mode and options.np > n_available_cores:
      options.np = n_available_cores
    # """ 6- Initialize blackbox handling subclass by copying
    #  the evaluator 'ev' instance to the poll object """
    search.bb_handle = ev
    search.bb_handle.bb_eval = ev.bb_eval
    # """ 7- Evaluate the starting point """
    if options.display:
      print(" Evaluation of the starting points")
      if self.log is not None:
        self.log.log_msg(
            msg="- Evaluation of the starting points...",
            msg_type=MSG_TYPE.INFO)
    x_start.coordinates = param.baseline
    x_start.sets = param.var_sets
    if param.constraints_type is not None and isinstance(
            param.constraints_type, list):
      x_start.constraints_type = [xb for xb in param.constraints_type]
    elif param.constraints_type is not None:
      x_start.constraints_type = [param.constraints_type]
    # """ 8- Set the variables type """
    if param.var_type is not None:
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
          # COMPLETED: Implementation in progress
        elif k.lower()[0] == "b":
          x_start.var_type.append(VAR_TYPE.BINARY)
        else:
          raise IOError(
              "Could not recognize the variable of type " + k +
              ". Please use on of the following keywords to identify your variable type:\
                  real, integer, discrete, categorical, ordinal, or binary")

    x_start.dtype.precision = options.precision
    if x_start.sets is not None and isinstance(x_start.sets, dict):
      p: List[Any] = []
      for i, _ in enumerate((x_start.var_type)):
        if (x_start.var_type[i] == VAR_TYPE.DISCRETE or
            x_start.var_type[i] == VAR_TYPE.CATEGORICAL) and\
                x_start.var_link[i] is not None:
          p.append(x_start.sets[x_start.var_link[i]]
                   [int(x_start.coordinates[i])])
        else:
          p.append(x_start.coordinates[i])
      search.bb_output, _ = search.bb_handle.eval(p)
    else:
      if not is_xs:
        search.bb_output, _ = search.bb_handle.eval(x_start.coordinates)
    x_start.h_max = barrier_defined._h_max
    search.hmax = barrier_defined._h_max
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
    x_start.constraints_type = param.constraints_type
    if not is_xs:
      x_start.__eval__(search.bb_output)
      # or isinstance(B, BarrierMO):
      if isinstance(barrier_defined, BarrierMO):
        barrier_defined._h_max = x_start.h_max
    # """ 9- Copy the starting point object to the poll's minimizer subclass """
    x_start.mesh = copy.deepcopy(search.mesh)
    search.xmin = copy.deepcopy(x_start)
    # """ 10- Hold the starting point in the poll
    #  directions subclass and define problem parameters"""
    search._candidate_points_set.append(x_start)
    search.scale(ub=param.ub, lb=param.lb, factor=param.scaling)
    search.dim = x_start.n_dimensions
    if not extend:
      search.hashtable = Cache()
      search.hashtable._n_dim = len(param.baseline)
      search.hashtable._is_pareto = param.is_pareto
      if param.is_pareto:
        search.hashtable.nd_points = []
    # """ 10- Initialize the number of successful points
    #  found and check if the starting minimizer performs better
    # than the worst (f = inf) """
    search.nb_success = 0
    if search.xmin < CandidatePoint():
      search.mesh.psize_success = search.mesh.get_delta_frame_size().coordinates
      search.mesh.psize_max = copy.deepcopy(
          max(search.mesh.get_delta_frame_size().coordinates))
      search._candidate_points_set = [search.xmin]
    # """ 11- Construct the results postprocessor class object 'post' """
    x_start.eval_no = search.bb_handle.bb_eval
    search.xmin.eval_no = search.bb_handle.bb_eval
    post = PostMADS(
        x_incumbent=[search.xmin],
        xmin=search.xmin, poll_dirs=[search.xmin])
    post.step_name = []
    post.step_name.append(f'Search: {search.type}')
    post.psize.append(search.mesh.get_delta_frame_size().coordinates)
    post.bb_eval.append(search.bb_handle.bb_eval)

    post.iter.append(iteration)

    # """ Note: printing the post will print a results row
    #  within the results table shown in Python console if the
    # 'display' option is true """

    # """ 12- Add the starting point hash value to the cache memory """
    if options.store_cache:
      search.hashtable.hash_id = x_start
    # """ 13- Initialize the output results file object  """
    out = Output(file_path=param.post_dir, vnames=param.var_names,
                 fnames=param.fun_names, pname=param.name,
                 runfolder=f'{param.name}_run', replace=True)
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
            msg="- End of the evaluation of the starting points.",
            msg_type=MSG_TYPE.INFO)
    iteration += 1
    if isinstance(barrier_defined, BarrierMO):
      barrier_defined.update_with_points(
          x_start if isinstance(x_start, list) else [x_start])

    return iteration, x_start, search, options, param, post, out, barrier_defined, out_p
