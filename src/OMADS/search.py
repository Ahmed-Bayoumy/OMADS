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
from multiprocessing import freeze_support

import os
import sys
import time
import copy
from typing import List, Dict, Any, Optional
import numpy as np

from ._globals import INSERTION_FLAG, SEARCH_TYPE, STOP_TYPE, VAR_TYPE, MSG_TYPE, SUCCESS_TYPES
from .candidate_point import CandidatePoint
from ._common import logger, validator
from .exploration import SAMPLING_METHOD, VNS, EfficientExploration
from .pre_exploration import PreExploration
from .barriers import BarrierMO
from .point import Point
from .metrics import Metrics
from .options import Options
from .parameters import Parameters
from .mads import MadsState, MadsStatistics
from .postprocess import PostMADS
from .optimizer import ConstraintsRelaxationParameters


np.set_printoptions(legacy='1.21')


def compute_success(
        active_barrier: BarrierMO, state: MadsState, options: Options,
        insertion_flag):
  """_summary_

  :param active_barrier: Evaluate success criteria and compute the success level
  :type active_barrier: BarrierMO
  :param state: The active algorithm session state and metadata
  :type state: MadsState
  :param options: algorithmic options
  :type options: Options
  :param insertion_flag: The update of non-dominated solutions or prim/sec incumbents
  :type insertion_flag: str
  :return: The success level
  :rtype: SUCCESS_TYPES
  """
  v: CandidatePoint = active_barrier.elements[active_barrier.last_index]

  # Phase one
  if state.is_phase_one:
    if v.h < active_barrier.elements[state.ordered_frame_centers[0]].h:
      return SUCCESS_TYPES.FS
    else:
      return SUCCESS_TYPES.US
  else:
    success_flag = SUCCESS_TYPES.US

    # Feasible case: full success as soon as a new non-dominated point which dominates
    # the current feasible frame center is generated.
    if v.is_feasible():
      # The set of feasible points can be empty before the insertion of the new point.
      if state.fk_frame_center != -1:
        if options.use_dms_success and insertion_flag in [
                'dominates', 'extends', 'improves']:
          success_flag = SUCCESS_TYPES.FS
        if v <= active_barrier.elements[state.fk_frame_center]:
          success_flag = SUCCESS_TYPES.FS
      # If Fk is empty, consider it as a full success, similar to the Nomad software.
      else:
        success_flag = SUCCESS_TYPES.FS
    else:
      # The first iterations are considered as a partial success when the progressive barrier
      # approach is chosen.
      if state.h_max is None:
        if active_barrier.h_max != 0:
          success_flag = SUCCESS_TYPES.PS
      else:
        if state.uk_frame_center != -1:
          # Partial success if h(x) below h(x_inf)
          if v.h < active_barrier.elements[state.uk_frame_center].h:
            success_flag = SUCCESS_TYPES.PS
        # Success if change in Iᵏ with h(x) <= h_max.
        if state.uk_frame_center != -1 and v <= active_barrier.elements[state.uk_frame_center]:
          success_flag = SUCCESS_TYPES.FS
          #  if v.h <= model.barrier.elements[model.state.Uk_frame_center].h and \
          # insertion_flag in ['dominates', 'extends', 'improves']:
          #      success_flag = FULL_SUCCESS
          #  end

    return success_flag


def set_frame_centers_and_hvalues(
        state: MadsState, active_barrier: BarrierMO, param: Parameters,
        options: Options = None):
  """_summary_

  :param state: _description_
  :type state: MadsState
  :param active_barrier: _description_
  :type active_barrier: BarrierMO
  :param param: _description_
  :type param: Parameters
  """
  if options is None:
    options = Options()
  # First case: phase one has been triggered.
  # There is only one primary frame center, the one with minimum h value.
  if state.is_phase_one:
    emin = None
    cmin = 0
    c = 0
    for e in active_barrier.elements:
      if emin is None or (e is not None and e < emin):
        emin = copy.deepcopy(e)
        cmin = copy.deepcopy(c)
      c += 1

    state.ordered_frame_centers = (
        cmin,
        0
    )
  else:
    out = active_barrier.frame_centers(param.w_min, use_dom_selection=True)
    fk_index = out["feasible"]
    uk_index = out["infeasible"]
    # Set primary and secondary frame centers according to trigger conditions.
    if fk_index == -1:
      state.ordered_frame_centers = (uk_index, -1)
    else:
      if uk_index == -1:
        state.ordered_frame_centers = (fk_index, -1)
      else:
        if True:
            # Using extent is slightly more efficient
          dom = min([
              sum(
                  elt.f - np.minimum(
                      active_barrier.elements[uk_index].fs.coordinates,
                      elt.fs.coordinates))
              for elt in active_barrier.get_fk()])
          #  if doM >= params.ρ_trigger * bbproblem.meta.noutputs
          if dom >= param.rho * active_barrier.extent():
            state.ordered_frame_centers = (uk_index, fk_index)
          else:
            state.ordered_frame_centers = (fk_index, uk_index)
        else:  # Classic alternative based on dominance but slightly less efficient.
          if all(active_barrier.elements[fk_index].f - param.rho >=
                 active_barrier.elements[uk_index].f):
            state.ordered_frame_centers = (uk_index, fk_index)
          else:
            state.ordered_frame_centers = (fk_index, uk_index)

    state.fk_frame_center = fk_index if isinstance(
        fk_index, int) else int(fk_index)
    state.uk_frame_center = uk_index if isinstance(
        uk_index, int) else int(uk_index)

    # set h_max
    if len(active_barrier.get_ik()) > 0:
      state.h_max = max(elt.h for elt in active_barrier.get_ik())


def search_step(search: EfficientExploration, options: Options,
                param: Parameters, state: MadsState, stats: MadsStatistics,
                active_barrier: BarrierMO, iteration: int,
                log: logger, post: PostMADS,
                search_vns: VNS = None):  # , xmin: CandidatePoint = None):
  """_summary_

  :param search: _description_
  :type search: efficient_exploration
  :param options: _description_
  :type options: Options
  :param param: _description_
  :type param: Parameters
  :param state: _description_
  :type state: MadsState
  :param stats: _description_
  :type stats: MadsStatistics
  :param active_barrier: _description_
  :type active_barrier: BarrierMO
  :param iteration: _description_
  :type iteration: int
  :param xmin: _description_
  :type xmin: CandidatePoint
  :param log: _description_
  :type log: logger
  :param post: _description_
  :type post: PostMADS
  :param search_vns: _description_, defaults to None
  :type search_vns: VNS, optional
  :return: _description_
  :rtype: _type_
  """
  # Set time
  tic = time.perf_counter()
  search.prob_params = copy.deepcopy(param)
  # Set mesh
  if state.fk_frame_center == -1:
    mk = active_barrier.meshes[state.ordered_frame_centers[0]]
  else:
    mk = active_barrier.meshes[state.fk_frame_center]

  if all(
      [mk.get_delta_frame_size().coordinates[pp] < options.tol
       for pp in range(param.n)]):
    state.stop_reason = STOP_TYPE.MIN_MESH_REACHED
    return

  # Generate candidates
  search.mesh = copy.deepcopy(mk)

  # search.mesh.update()
  # """ Create the candidate points """
  search.constraints_rp.hmax = state.h_max
  search.constraints_handler.hmax = state.h_max
  search.hmax = state.h_max

  parent_index_candidates = []
  generated_during_search_step = []

  search.active_barrier = copy.deepcopy(active_barrier)
  # if HT is not None:
  #   search.hashtable = HT

  for ci, _ in enumerate(state.ordered_frame_centers):
    center = state.ordered_frame_centers[ci]
    if center > -1:
      search.xmin = active_barrier.elements[center]
      search.mesh = copy.deepcopy(mk)

      if search.type == SEARCH_TYPE.VNS.name and search_vns is not None:
        search_vns.active_barrier = active_barrier
        search.candidate_points_set = search_vns.run()
        if search_vns.stop:
          print("Reached maximum number of VNS iterations!")
          # HT = search.hashtable
          # rho_k = search.constraintsHandler.RHO
          # lambda_multipliers_k = search.constraintsHandler.LAMBDA
          return search.xmin
        search.map_samples_from_coords_to_points(
            samples=search.candidate_points_set)

      search.generate_sample_points(
          nsamples=int(
              ((search.dim + 1) / 2) *
              ((search.dim + 2) / 2))
          if search.ns is None else search.ns)

      lambda_multipliers_k = search.xmin.lambda_multipliers
      rho_k = search.xmin.rho

      search.project_on_mesh_and_snap_to_bounds(
          m=mk, x_center=active_barrier.elements[center].coordinates,
          lb=param.lb, ub=param.ub)
      peval = search.bb_handle.bb_eval
      search.omit_duplicates(peval)

      post.x_incumbent.append(search.xmin)

      for _ in search.candidate_points_set:
        parent_index_candidates.append(center)
        generated_during_search_step.append(False)

      #  Evaluation
      # """ Save current search directions and incumbent solution
      #   so they can be saved later in the post dir """
      if options.save_coordinates:
        post.coords.append(search.candidate_points_set)
        post.x_incumbent.append(search.xmin)
      # """ Reset success boolean """
      search.success = SUCCESS_TYPES.US
      # """ Reset the BB output """
      search.bb_output = []
      xt = []
      search.bb_handle.xmin = active_barrier.elements[center]
      # """ Serial evaluation for points in the search candidates set """
      search.constraints_rp.lambda_multipliers = active_barrier.elements[
          center].lambda_multipliers
      search.constraints_rp.rho = active_barrier.elements[center].rho
      search.constraints_rp.constraints_type = active_barrier.elements[center].constraints_type
      search.constraints_rp.hmax = state.h_max

      if not options.parallel_mode:
        xt, post, peval = search.bb_handle.run_callable_serial_local(
            iter=iteration, peval=peval, eval_set=search.candidate_points_set,
            options=options, post=post,
            psize=search.mesh.get_delta_frame_size().coordinates,
            step_name='Search Step', mesh=mk,
            constraints_relaxation=search.constraints_rp.__dict__,
            budget=options.budget)

      else:
        search.point_index = -1
        # """ Parallel evaluation for points in the samples set """
        search.bb_eval, xt, post, peval = search.bb_handle.run_callable_parallel_local(
            iter=iteration, peval=peval, eval_set=search.candidate_points_set,
            options=options, post=post, mesh=mk, step_name='Search Step',
            psize=search.mesh.get_delta_frame_size().coordinates,
            constraints_relaxation=search.constraints_rp.__dict__,
            budget=options.budget)

      if search.bb_handle.constraints_relaxation:
        temp: ConstraintsRelaxationParameters = ConstraintsRelaxationParameters(
            **search.bb_handle.constraints_relaxation)
        for i, _ in enumerate(temp.lambda_multipliers):
          search.constraints_rp.lambda_multipliers[i] = temp.lambda_multipliers[i]
        search.constraints_rp.rho = temp.rho
        search.constraints_rp.constraints_type = temp.constraints_type
        search.constraints_rp.hmax = temp.hmax

      # lambda_multipliers_k = search.bb_handle.constraints_relaxation["LAMBDA"]
      # rho_k = search.bb_handle.constraints_relaxation["RHO"]
      search.postprocess_evaluated_candidates(xt)

      idx = -1
      xpost: List[CandidatePoint] = []
      for i, _ in enumerate(xt):
        xpost.append(xt[i])
        post.poll_dirs.append(xpost[i])
      # assert len(HT.hash_id) - active_barrier.last_index + 1== 0
      for cp in xt:
        idx += 1
        stats.neval_bb_feasible += 1
        if cp.is_feasible():
          _, insertion_flag = active_barrier.update_feas_with_point(cp)
        else:
          stats.neval_bb_infeasible += 1
          _, insertion_flag = active_barrier.update_inf_with_point(cp)
        active_barrier.parent_indexes[active_barrier.last_index] = parent_index_candidates[idx]
        success_flag = SUCCESS_TYPES.US if insertion_flag is None else compute_success(
            active_barrier=active_barrier,
            state=state, options=options,
            insertion_flag=insertion_flag)

        if insertion_flag is not None and insertion_flag in [
                INSERTION_FLAG.DOMINATES, INSERTION_FLAG.EXTENDS]:
          x_parent = np.array(search.hashtable.get_cache_candidate_points()[
                              parent_index_candidates[idx]].coordinates)
          active_barrier.meshes[active_barrier.last_index].enlarge_delta_frame_size(np.array(
              search.hashtable.cache_dict[search.hashtable.hash_id[-1]].coordinates)-x_parent)

        # Update success flag
        if success_flag.value > (
                state.last_success.value
                if
                isinstance(state.last_success, SUCCESS_TYPES) else
                SUCCESS_TYPES[state.last_success].value):
          state.last_success = success_flag

  # lambda_multipliers_k = xmin.lambda_multipliers
  # rho_k = search.xmin.rho
  if active_barrier.current_incumbent_feas is not None:
    search.xmin = active_barrier.current_incumbent_feas
  post.xmin = search.xmin
  search._x_sc = active_barrier.current_incumbent_inf
  if not search.prob_params.is_pareto:
    active_barrier.update_current_incumbents()

  toc = time.perf_counter()

  if log is not None:
    # log.log_msg(msg=" ---Run Summary--- ", msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" Run completed in {toc - tic:.4f} seconds",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" Success status: {search.success}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=post, msg_type=MSG_TYPE.INFO)

  # active_barrier = B
  # lambda_multipliers_k = search.xmin.lambda_multipliers
  search.hashtable.add_to_best_cache(active_barrier.get_all_points())

  # HT = search.hashtable

  if state.last_success == SUCCESS_TYPES.FS and search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
    search.update_local_region(region="expand")
  elif state.last_success == SUCCESS_TYPES.US and search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
    search.update_local_region(region="contract")
  return search.xmin


def update(
        search: EfficientExploration, state: MadsState, options: Options,
        active_barrier: BarrierMO, stats: MadsStatistics):
  """_summary_

  :param search: _description_
  :type search: efficient_exploration
  :param state: _description_
  :type state: MadsState
  :param options: _description_
  :type options: Options
  :param active_barrier: _description_
  :type active_barrier: BarrierMO
  :param stats: _description_
  :type stats: MadsStatistics
  """
  # Phase one is over.
  if state.stop_reason == STOP_TYPE.STOP_IF_FEASIBLE:
    state.is_phase_one = False
    state.stop_reason = STOP_TYPE.UNKNOWN_STOP_REASON

  # No need to update in this case
  if state.stop_reason in [
          STOP_TYPE.DELTA_M_MIN_REACHED, STOP_TYPE.MAX_BB_EVAL_REACHED, STOP_TYPE.
          MAX_BB_EVAL_REACHED]:
    return

  # There can remain some points in the set Uk which have better h-value,
  # To check if the flag is activated
  if not options.use_nomad_partial_success:
    if state.last_success == SUCCESS_TYPES.US and state.uk_frame_center != -1:
      tmp_hx_inf_min = min(elt.h for elt in active_barrier.x_filter_inf)
      if tmp_hx_inf_min < active_barrier.elements[state.uk_frame_center].h:
        state.last_success = SUCCESS_TYPES.PS

  # Update barrier and mesh
  if state.last_success == SUCCESS_TYPES.US:
    # Set to null the last success directions of the current incumbents.
    for frame_center in range(2):
      barrier_index = state.ordered_frame_centers[frame_center]
      if barrier_index != -1:
        active_barrier.parent_indexes[barrier_index] = 0

    # Update the mesh
    if state.fk_frame_center != -1:
      active_barrier.meshes[state.fk_frame_center].refine_delta_frame_size()
    else:
      active_barrier.meshes[state.ordered_frame_centers[0]
                            ].refine_delta_frame_size()

    # Update the barrier threshold
    if state.h_max is not None:
      below_hmax_elements = [
          elt for elt in active_barrier.elements
          if elt is not None and elt.h < state.h_max and elt in
          active_barrier.x_filter_inf]

      if state.uk_frame_center != -1 and len(below_hmax_elements) != 0:
        h_max_tmp = max(elt.h for elt in below_hmax_elements)
        if h_max_tmp > active_barrier.elements[state.uk_frame_center].h:
          active_barrier.update_barrier(h_max_tmp)
        else:
          active_barrier.update_barrier(
              active_barrier.elements[state.uk_frame_center].h)

    stats.nno_successes += 1

  elif state.last_success == SUCCESS_TYPES.PS:
    # Update the barrier threshold. This follows the implementation of Nomad 3.
    if state.h_max is not None:
      below_hxi_elements = [
          elt for elt in active_barrier.elements
          if elt is not None and elt.h < active_barrier.elements
          [state.uk_frame_center].h and active_barrier.within_uk
          [active_barrier.elements.index(elt)]]
      if below_hxi_elements is not None:
        active_barrier.update_barrier(max(elt.h for elt in below_hxi_elements))

    stats.npartial_successes += 1

  else:  # Full success
    # Update the barrier threshold

    if state.h_max is not None:
      below_hmax_elements = []
      for elt in active_barrier.elements:
        if elt is not None and elt.h < state.h_max:
          for elt2 in active_barrier.x_filter_inf:
            if elt.coordinates == elt2.coordinates:
              below_hmax_elements.append(elt)
      if state.uk_frame_center != -1 and below_hmax_elements:
        h_max_tmp = max(elt.h for elt in below_hmax_elements)
        if h_max_tmp > active_barrier.elements[state.uk_frame_center].h:
          active_barrier.update_barrier(h_max_tmp)
        else:
          active_barrier.update_barrier(
              active_barrier.elements[state.uk_frame_center].h)

    # In the case of phase one, update the mesh; otherwise, it was already set before
    if state.is_phase_one:
      emin = None
      cmin = 0
      c = 0
      for e in active_barrier.elements:
        if emin is None or (e is not None and e < emin):
          emin = copy.deepcopy(e)
          cmin = copy.deepcopy(c)
        c += 1
      new_incumbent_index = cmin
      x_parent = np.array((search.hashtable.get_cache_candidate_points()[
                          active_barrier.parent_indexes[new_incumbent_index]]).coordinates)
      active_barrier.meshes[new_incumbent_index].enlarge_delta_frame_size(np.array(
          (search.hashtable.get_cache_candidate_points()
           [new_incumbent_index]).coordinates) - np.array(x_parent))

    stats.nfull_successes += 1
  assert active_barrier.last_index == len(search.hashtable.hash_id)-1

  # Reset state
  state.last_success = SUCCESS_TYPES.US


def main(*args) -> Dict[str, Any]:
  """ MADS: Search step main algorithm """

  # """ Validate and parse the parameters file """
  validate = validator()
  data: dict = validate.check_input_file(args=args)

  # """ Initialize the log file """
  log = logger()
  if not os.path.exists(data["param"]["post_dir"]):
    try:
      os.mkdir(data["param"]["post_dir"])
    except Warning:
      os.makedirs(data["param"]["post_dir"], exist_ok=True)
  log.initialize(data["param"]["post_dir"] + "/OMADS.log")

  # """ Run preprocessor for the setup of
  #  the optimization problem and for the initialization
  # of optimization process """
  iteration, xmin, search, options, param, post, out, active_barrier, out_p = PreExploration(
      data).initialize_from_dict(log=log)

  if out_p:
    out_p.step_name = "Search_ND"

  # """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))

  out.step_name = f"Search: {search.type}"

  # """ Start the count down for calculating the runtime indicator """
  tic = time.perf_counter()

  # peval = 0
  search_vn = None
  if search.type == SEARCH_TYPE.VNS.name:
    search_vn = VNS(active_barrier=active_barrier, params=param)
    search_vn.ns_dist = [
        int(
            ((search.dim + 1) / 2) * ((search.dim + 2) / 2) /
            (len(search_vn.dist))) if search.ns is None else search.ns] * len(
        search_vn.dist)
    search.ns = sum(search_vn.ns_dist)

  search.lb = param.lb
  search.ub = param.ub

  log.log_msg(
      msg=f"---------------- Run the SEARCH step ({search.sampling_t}) ----------------",
      msg_type=MSG_TYPE.INFO)
  num_strat: int = 0
  state = MadsState()
  state.h_max = param.h_max
  stats = MadsStatistics()
  state.last_success = SUCCESS_TYPES.US
  while True:
    bbevalold = search.bb_handle.bb_eval
    search.mesh.update()
    search.iter = iteration
    # 1- Select incumbents (prim. and sec.)
    set_frame_centers_and_hvalues(
        state=state, active_barrier=active_barrier, param=param)
    # 2- Run the search step
    xmin = search_step(
        search=search, options=options, param=param, state=state, stats=stats,
        active_barrier=active_barrier, iteration=iteration, log=log,
        post=post, search_vns=search_vn)
    # 3- Updates
    update(search=search, state=state, active_barrier=active_barrier,
           stats=stats, options=options)
    # 4- Generate output files
    if options.save_results:
      post.nd_points = []

      post.output_results(out=out, all_res=False)
      if param.is_pareto:
        for i in range(len(active_barrier.get_all_points())):
          post.nd_points.append(active_barrier.get_all_points()[i])
        post.output_nd_results(out_p)

    # if log:
    #   log.log_msg(msg=post, msg_type=MSG_TYPE.INFO)
    # if options.display:
    #   print(post)

    failure_check = iteration > 0 and search.failure_stop is not None and \
        search.failure_stop and not (
            search.success != SUCCESS_TYPES.FS or SUCCESS_TYPES.PS)
    if search.bb_handle.bb_eval - bbevalold <= 0:
      num_strat += 1
      search.explore_new = True
    else:
      num_strat = 0
    if (failure_check or search.bb_handle.bb_eval >= options.budget) or \
        (all(abs(search.mesh.get_delta_mesh_size().coordinates[pp]) < options.tol
         for pp in range(search.mesh.n)) or search.bb_handle.bb_eval >= options.budget
         or search.terminate):
      log.log_msg(
          "\n--------------- Termination of the search step  ---------------",
          MSG_TYPE.INFO)
      if (all(abs(search.mesh.get_delta_mesh_size().coordinates[pp]) < options.tol
              for pp in range(search.mesh.n))):
        log.log_msg(
            "Termination criterion hit: the mesh size is below the minimum threshold defined.",
            MSG_TYPE.INFO)
      if (search.bb_handle.bb_eval >= options.budget or search.terminate):
        log.log_msg(
            "Termination criterion hit: evaluation budget is exhausted.",
            MSG_TYPE.INFO)
      if failure_check:
        log.log_msg(
            "Termination criterion hit (optional): failed to find a \
            successful point in iteration # {iteration}.",
            MSG_TYPE.INFO)
      log.log_msg(
          "-----------------------------------------------------------------\n",
          MSG_TYPE.INFO)
      break
    iteration += 1

  toc = time.perf_counter()
  if search.prob_params.is_pareto and isinstance(active_barrier, BarrierMO):
    rp: Optional[CandidatePoint] = None
    if param.ref_point:
      rp = Point()
      rp.coordinates = param.ref_point
    perf_m = Metrics(
        nd_solutions=active_barrier.get_all_feas_points(),
        nobj=active_barrier.nobj, ref_point=rp)
    hv = perf_m.hypervolume()

  # """ If benchmarking, then populate the results in the benchmarking output report """
  # if importlib.util.find_spec('BMDFO') and len(args) > 1 and isinstance(
  #         args[1],
  #         toy.Run):
  #   b: toy.Run = args[1]
  #   if b.test_suite == "uncon":
  #     ncon = 0
  #   else:
  #     ncon = len(search.xmin.c_eq) + len(search.xmin.c_ineq)
  #   if len(search.bb_output) > 0:
  #     b.add_row(name=search.bb_handle.blackbox,
  #               run_index=int(args[2]),
  #               nv=len(param.baseline),
  #               nc=ncon,
  #               nb_success=search.nb_success,
  #               it=iteration,
  #               BBEVAL=search.bb_eval,
  #               runtime=toc - tic,
  #               feval=search.bb_handle.bb_eval,
  #               hmin=search.xmin.h,
  #               fmin=search.xmin.f)
  #   print(f"{search.bb_handle.blackbox}: fmin = {search.xmin.f} , hmin= {search.xmin.h:.2f}")

  # elif importlib.util.find_spec('BMDFO') and len(args) > 1 and not isinstance(args[1], toy.Run):
  #   if log:
  #     log.log_msg(
  #         msg="Could not find " + args[1] + " in the internal BM suite.",
  #         msg_type=MSG_TYPE.ERROR)
  #   raise IOError("Could not find " + args[1] + " in the internal BM suite.")

  if options.display:
    if log:
      log.log_msg(" end of orthogonal MADS ", MSG_TYPE.INFO)
    print(" end of orthogonal MADS ")
    if log:
      log.log_msg(
          " Final objective value: " + str(search.xmin.f) + ", hmin= " +
          str(search.xmin.h),
          MSG_TYPE.INFO)
    print(
        " Final objective value: " + str(search.xmin.f) + ", hmin= " +
        str(search.xmin.h))

  if options.save_coordinates:
    post.output_coordinates(out)

  if log:
    log.log_msg("\n ---Run Summary---", MSG_TYPE.INFO)
    log.log_msg(f" Run completed in {toc - tic:.4f} seconds", MSG_TYPE.INFO)
    log.log_msg(
        msg=f" # of successful search steps = {search.n_successes}",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        f" Random numbers generator's seed {options.seed}", MSG_TYPE.INFO)
    log.log_msg(" xmin = " + str(search.xmin), MSG_TYPE.INFO)
    log.log_msg(" hmin = " + str(search.xmin.h), MSG_TYPE.INFO)
    log.log_msg(" fmin = " + str(search.xmin.f), MSG_TYPE.INFO)
    log.log_msg(" #bb_eval = " + str(search.bb_handle.bb_eval), MSG_TYPE.INFO)
    log.log_msg(" #iteration = " + str(iteration), MSG_TYPE.INFO)

  if options.display:

    print("\n ---Run Summary---")
    print(f" Run completed in {toc - tic:.4f} seconds")
    print(f" Random numbers generator's seed {options.seed}")
    print(" xmin = " + str(search.xmin))
    print(" hmin = " + str(search.xmin.h))
    print(" fmin = " + str(search.xmin.f))
    print(" #bb_eval = " + str(search.bb_eval))
    print(" #iteration = " + str(iteration))
    print(" nb_success = " + str(search.nb_success))
    print(" mesh_size = " + str(search.mesh.get_delta_frame_size().coordinates))
  xmin = search.xmin
  # """ Evaluation of the blackbox; get output responses """
  if xmin.sets is not None and isinstance(xmin.sets, dict):
    p: List[Any] = []
    for i, _ in enumerate(xmin.var_type):
      if (xmin.var_type[i] == VAR_TYPE.DISCRETE or
              xmin.var_type[i] == VAR_TYPE.CATEGORICAL) and xmin.var_link[i] is not None:
        p.append(xmin.sets[xmin.var_link[i]][int(xmin.coordinates[i])])
      else:
        p.append(xmin.coordinates[i])
  else:
    p = xmin.coordinates
  output: Dict[str, Any] = {"xmin": p,
                            "fmin": search.xmin.f,
                            "hmin": search.xmin.h,
                            "nbb_evals": search.bb_eval,
                            "niterations": iteration,
                            "nb_success": search.nb_success,
                            "psize": search.mesh.get_delta_frame_size().coordinates,
                            "psuccess": search.xmin.mesh.get_delta_frame_size().coordinates,
                            # "pmax": search.mesh.psize_max,
                            "msize": search.mesh.get_delta_mesh_size().coordinates,
                            "HV": hv if param.is_pareto else "NA"}

  return output, search


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
    raise IOError(
        "Undefined input args."
        " Please specify an appropriate input (parameters) jason file")
