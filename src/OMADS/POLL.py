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
import copy
import os
import sys
import time
from multiprocessing import freeze_support
from typing import List, Dict, Any, Optional

import numpy as np
from ._globals import INSERTION_FLAG, MSG_TYPE, STOP_TYPE, SUCCESS_TYPES, VAR_TYPE
from .point import Point
from .barriers import BarrierMO
from ._common import validator, logger
from .pre_poll import PrePoll
from .candidate_point import CandidatePoint
from .metrics import Metrics
from .directions import Dirs2n
from .mads import ConstraintsRelaxationParameters, MadsState, MadsStatistics
from .parameters import Parameters
from .options import Options
from .postprocess import PostMADS


np.set_printoptions(legacy='1.21')


def compute_success(
        active_barrier: BarrierMO, state: MadsState, options: Options,
        insertion_flag: str) -> SUCCESS_TYPES:
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
          #  if v.h <= model.barrier.elements[model.state.Uk_frame_center].h and insertion_flag in
          # ['dominates', 'extends', 'improves']:
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
        if options.use_dom_trigger:
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


def poll_step(
        poll: Dirs2n, options: Options, param: Parameters, state: MadsState,
        stats: MadsStatistics, active_barrier: BarrierMO, iteration: int,
        xmin: CandidatePoint, log: logger, post: PostMADS) -> CandidatePoint:
  """_summary_

  :param poll: The poll step algorithm
  :type poll: Dirs2n
  :param options: The directions class object
  :type options: Options
  :param param: Algorithmic options
  :type param: Parameters
  :param state: Algorithmic parameters
  :type state: MadsState
  :param stats: Success status
  :type stats: MadsStatistics
  :param active_barrier: Statistics and metadata class object
  :type active_barrier: BarrierMO
  :param iteration: The active barrier class object
  :type iteration: int
  :param xmin: iteration number
  :type xmin: CandidatePoint
  :param log: current incumbent candidate
  :type log: logger
  :param post: updated log
  :type post: PostMADS
  :return: Updated incumbent candidate
  :rtype: CandidatePoint
  """
  tic = time.perf_counter()
  # Set mesh
  if state.fk_frame_center == -1:
    mk = active_barrier.meshes[state.ordered_frame_centers[0]]
  else:
    mk = active_barrier.meshes[state.fk_frame_center]

  if mk.checkMeshForStopping():
    state.stop_reason = STOP_TYPE.MIN_MESH_REACHED
    return

  poll.mesh.update()
  # """ Create the set of poll directions """
  if state.fk_frame_center >= 0:
    hhm = poll.create_housholder(
        options.rich_direction, domain=active_barrier.elements
        [state.fk_frame_center].var_type)
  elif state.uk_frame_center >= 0:
    hhm = poll.create_housholder(
        options.rich_direction, domain=active_barrier.elements
        [state.uk_frame_center].var_type)
  elif xmin is not None and xmin.evaluated:
    hhm = poll.create_housholder(
        options.rich_direction, domain=xmin.var_type)
  else:
    hhm = poll.create_housholder(
        options.rich_direction, domain=poll.xmin.var_type)

  poll.lb = param.lb
  poll.ub = param.ub
  poll.constraints_handler.hmax = state.h_max
  xmin.h_max = state.h_max
  poll.xmin.h_max = state.h_max
  # xmin.mesh = copy.deepcopy(poll.mesh)
  # B = active_barrier
  # if HT is not None:
  #   poll.hashtable = HT

  parent_index_candidates = []
  generated_during_search_step = []
  for ci, _ in enumerate(state.ordered_frame_centers):
    center = state.ordered_frame_centers[ci]
    if center > -1:
      if active_barrier.elements[center].is_feasible():
        poll.xmin = active_barrier.elements[center]
      else:
        poll.x_sc = active_barrier.elements[center]

      poll.create_poll_set(
          hhm=hhm, ub=param.ub, lb=param.lb, it=iteration,
          var_type=xmin.var_type, var_sets=xmin.sets, var_link=xmin.var_link,
          c_types=param.constraints_type,
          is_prim=active_barrier.elements[center].is_feasible())

      poll.constraints_handler.lambda_multipliers = xmin.lambda_multipliers
      poll.constraints_handler.rho = xmin.rho

      poll.project_on_mesh_and_snap_to_bounds(
          m=mk, x_center=active_barrier.elements[center].coordinates,
          lb=param.lb, ub=param.ub)
      peval = poll.bb_handle.bb_eval
      poll.omit_duplicates(peval)

      for _ in poll.poll_set:
        parent_index_candidates.append(center)
        generated_during_search_step.append(False)

      # Evaluation
      # """ Save current poll directions and incumbent solution
      # so they can be saved later in the post dir """
      if options.save_coordinates:
        post.coords.append(poll.poll_set)
        post.x_incumbent.append(poll.xmin)
      # """ Reset success boolean """
      poll.success = SUCCESS_TYPES.US
      # """ Reset the BB output """
      poll.bb_output = []
      xt = []
      poll.bb_handle.xmin = active_barrier.elements[center]
      # """ Serial evaluation for points in the poll set """
      poll.constraints_rp.lambda_multipliers = active_barrier.elements[center].lambda_multipliers
      poll.constraints_rp.rho = active_barrier.elements[center].rho
      poll.constraints_rp.constraints_type = active_barrier.elements[center].constraints_type
      poll.constraints_rp.hmax = state.h_max

      if not options.parallel_mode:
        xt, post, peval = poll.bb_handle.run_callable_serial_local(
            iter=iteration, peval=peval, eval_set=poll.poll_set,
            options=options, post=post,
            psize=poll.mesh.get_delta_frame_size().coordinates, step_name=None,
            mesh=mk, constraints_relaxation=poll.constraints_rp.__dict__,
            budget=options.budget)

      else:
        poll.point_index = -1
        # """ Parallel evaluation for points in the samples set """
        poll.bb_eval, xt, post, peval = \
            poll.bb_handle.run_callable_parallel_local(
                iter=iteration, peval=peval, eval_set=poll.poll_set,
                options=options, post=post, mesh=mk, step_name=None,
                psize=poll.mesh.get_delta_frame_size().coordinates,
                constraints_relaxation=poll.constraints_rp.__dict__,
                budget=options.budget)

      if poll.bb_handle.constraints_relaxation:
        temp: ConstraintsRelaxationParameters = ConstraintsRelaxationParameters(
            **poll.bb_handle.constraints_relaxation)
        for i, _ in enumerate(temp.lambda_multipliers):
          poll.constraints_rp.lambda_multipliers[i] = temp.lambda_multipliers[i]
        poll.constraints_rp.rho = temp.rho
        poll.constraints_rp.constraints_type = temp.constraints_type
        poll.constraints_rp.hmax = temp.hmax

      # lambda_multipliers_k = poll.bb_handle.constraints_relaxation["LAMBDA"]
      # rho_k = poll.bb_handle.constraints_relaxation["RHO"]
      poll.postprocess_evaluated_candidates(xt)

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
            active_barrier=active_barrier, state=state,
            options=options, insertion_flag=insertion_flag)

        if insertion_flag is not None and insertion_flag in [
                INSERTION_FLAG.DOMINATES, INSERTION_FLAG.EXTENDS]:
          x_parent = np.array(poll.hashtable.get_cache_candidate_points()[
                              parent_index_candidates[idx]].coordinates)
          active_barrier.meshes[active_barrier.last_index].enlarge_delta_frame_size(np.array(
              poll.hashtable.cache_dict[poll.hashtable.hash_id[-1]].coordinates)-x_parent)

        # Update success flag
        if success_flag.value > (
                state.last_success.value
                if
                isinstance(state.last_success, SUCCESS_TYPES) else
                SUCCESS_TYPES[state.last_success].value):
          state.last_success = success_flag

  # lambda_multipliers_k = poll.constraintsHandler.LAMBDA
  # rho_k = poll.constraintsHandler.RHO

  toc = time.perf_counter()

  if log is not None:
    # log.log_msg(msg=" ---Run Summary--- ", msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" Run completed in {toc - tic:.4f} seconds",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" Success status: {poll.success}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=post, msg_type=MSG_TYPE.INFO)

  poll.hashtable.add_to_best_cache(active_barrier.get_all_points())
  # HT = poll.hashtable
  # active_barrier = B
  # lambda_multipliers_k = poll.xmin.lambda_multipliers
  return poll.xmin


def update(poll: Dirs2n, state: MadsState, options: Options,
           active_barrier: BarrierMO, stats: MadsStatistics):
  """Update step

  :param poll: poll directions
  :type poll: Dirs2n
  :param state: Algorithmic current success state
  :type state: MadsState
  :param options: Algorithmic options
  :type options: Options
  :param active_barrier: Active barrier
  :type active_barrier: BarrierMO
  :param stats: Metadata and run statistics
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
      x_parent = np.array((poll.hashtable.get_cache_candidate_points()[
                          active_barrier.parent_indexes[new_incumbent_index]]).coordinates)
      active_barrier.meshes[new_incumbent_index].enlarge_delta_frame_size(
          np.array(
              (poll.hashtable.get_cache_candidate_points()
               [new_incumbent_index]).coordinates) - np.array(x_parent))

    stats.nfull_successes += 1
  assert active_barrier.last_index == len(poll.hashtable.hash_id)-1

  # Reset state
  state.last_success = SUCCESS_TYPES.US


def main(*args) -> Dict[str, Any]:
  """MADS: Poll step main algorithm

  :raises IOError: Exceptions
  :return: output dictionary
  :rtype: Dict[str, Any]
  """
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
  iteration, xmin, poll, options, param, post, out, active_barrier, out_p = PrePoll(
      data).initialize_from_dict(log=log)
  out.step_name = "Poll"
  if out_p:
    out_p.step_name = "Poll_ND"

  # """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))

  # """ Start the count down for calculating the runtime indicator """
  tic = time.perf_counter()
  # peval = poll.bb_handle.bb_eval
  # lambda_k = xmin.lambda_multipliers
  # rho_k = xmin.rho
  state = MadsState()
  state.h_max = param.h_max
  stats = MadsStatistics()
  state.last_success = SUCCESS_TYPES.US
  while True:
    del poll.poll_set
    # 1- Select incumbents (prim. and sec.)
    set_frame_centers_and_hvalues(
        state=state, active_barrier=active_barrier, param=param)
    # 2- Run the poll step
    xmin = poll_step(
        poll=poll, options=options, param=param, state=state, stats=stats,
        active_barrier=active_barrier, iteration=iteration, xmin=xmin, log=log,
        post=post)
    # 3- Updates
    update(poll=poll, state=state, active_barrier=active_barrier,
           stats=stats, options=options)
    # 4- Generate output files
    if options.save_results:
      post.nd_points = []
      post.output_results(out, all_res=False)
      feas_pts: List[CandidatePoint] = active_barrier.get_all_points()
      if param.is_pareto:
        for i, _ in enumerate(feas_pts):
          post.nd_points.append(feas_pts[i])
        post.output_nd_results(out_p)

    failure_check = iteration > 0 and poll.failure_stop is not None and poll.failure_stop and (
        poll.success == SUCCESS_TYPES.US)

    if (failure_check or poll.bb_eval >= options.budget) or \
        (all(abs(poll.mesh.get_delta_frame_size().coordinates[pp]) < options.tol
         for pp in range(poll.dim)) or poll.bb_eval >= options.budget or poll.terminate):
      log.log_msg(
          "\n--------------- Termination of the poll step  ---------------",
          MSG_TYPE.INFO)
      if all(
              abs(poll.mesh.get_delta_frame_size().coordinates[pp]) < options.tol
              for pp in range(poll.dim)):
        log.log_msg(
            "Termination criterion hit: the mesh size is below the minimum threshold defined.",
            MSG_TYPE.INFO)
      if (poll.bb_eval >= options.budget or poll.terminate):
        log.log_msg(
            "Termination criterion hit: evaluation budget is exhausted.",
            MSG_TYPE.INFO)
      if failure_check:
        log.log_msg(
            "Termination criterion hit (optional): failed to find \
            a successful point in iteration # {iteration}.",
            MSG_TYPE.INFO)
      log.log_msg(
          "---------------------------------------------------------------\n",
          MSG_TYPE.INFO)
      break
    iteration += 1

  toc = time.perf_counter()
  if isinstance(active_barrier, BarrierMO):
    rp: Optional[CandidatePoint] = None
    if param.ref_point:
      rp = Point()
      rp.coordinates = param.ref_point
    perf_m = Metrics(
        nd_solutions=active_barrier.get_all_points(),
        nobj=active_barrier.nobj, ref_point=rp)
    hv = perf_m.hypervolume()

  if options.display:
    print(" end of orthogonal MADS ")
    if log:
      log.log_msg(msg=" end of orthogonal MADS ", msg_type=MSG_TYPE.INFO)
    print(
        " Final objective value: " + str(poll.xmin.f) + ", hmin= " +
        str(poll.xmin.h))
    if log:
      log.log_msg(
          msg=" Final objective value: " + str(poll.xmin.f) + ", hmin= " +
          str(poll.xmin.h),
          msg_type=MSG_TYPE.INFO)
    if log and len(args) > 1 and isinstance(args[1], str):
      log.log_msg(
          msg=" end of orthogonal MADS running" + args[1] +
          " in the internal BM suite.", msg_type=MSG_TYPE.INFO)

  if options.save_coordinates:
    post.output_coordinates(out)

  if log:
    log.log_msg(msg="\n---Run Summary---", msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" Run completed in {toc - tic:.4f} seconds",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" Random numbers generator's seed {options.seed}",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" xmin = {poll.xmin}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" hmin = {poll.xmin.h}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" fmin {poll.xmin.fobj}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" #bb_eval =  {poll.bb_eval}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" #iteration =  {iteration}", msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f"  nb_success = {poll.nb_success}", msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" psize = {poll.mesh.get_delta_frame_size().coordinates}",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" psize_success = {poll.xmin.mesh.get_delta_frame_size().coordinates}",
        msg_type=MSG_TYPE.INFO)
    # log.log_msg(msg=f" psize_max = {poll.mesh.psize_max}", msg_type=MSG_TYPE.INFO)

  if options.display:
    print("\n ---Run Summary---")
    print(f" Run completed in {toc - tic:.4f} seconds")
    print(f" Random numbers generator's seed {options.seed}")
    print(" xmin = " + str(poll.xmin))
    print(" hmin = " + str(poll.xmin.h))
    print(" fmin = " + str(poll.xmin.fobj))
    print(" #bb_eval = " + str(poll.bb_eval))
    print(" #iteration = " + str(iteration))
    print(" nb_success = " + str(poll.nb_success))
    print(" psize = " + str(poll.mesh.get_delta_frame_size().coordinates))
    print(
        " psize_success = " +
        str(poll.xmin.mesh.get_delta_frame_size().coordinates))
    # print(" psize_max = " + poll.mesh.psize_max)

  xmin = copy.deepcopy(poll.xmin)
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
                            "fmin": poll.xmin.f,
                            "hmin": poll.xmin.h,
                            "nbb_evals": poll.bb_eval,
                            "niterations": iteration,
                            "nb_success": poll.nb_success,
                            "psize": poll.mesh.get_delta_frame_size().coordinates,
                            "psuccess": poll.xmin.mesh.get_delta_frame_size().coordinates,
                            # "pmax": poll.mesh.psize_max,
                            "msize": poll.mesh.get_delta_mesh_size().coordinates,
                            "HV": hv if param.is_pareto else "NA"}

  return output, poll


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
