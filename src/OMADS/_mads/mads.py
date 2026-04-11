"""_summary_

# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                    #
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
import time
from typing import List, Dict, Any, Optional
import sys
import os
from multiprocessing import freeze_support
import warnings
import numpy as np
from .._include import ProgressBar

from .._include import preprocess

from .._include import AdaptiveBarrier
from .._include import VNS
from .._include import SEARCH_TYPE, SAMPLER_TYPE
from .._include import SUCCESS_TYPES, VAR_TYPE, STOP_TYPE
from .._include import WarningSuppressor
from .._include import Point
from .._include import CandidatePoint
from .._include import logger, MSG_TYPE, validator
from .._include import Parameters
from .._include import Options
from .._include import Output, PostMADS
from .._include import Metrics
from .._include import MadsState, MadsStatistics
from .._include import poll_cycle
from .._include import search_cycle
from .._updates._updates import set_frame_centers_and_hvalues, update

np.set_printoptions(legacy='1.21')


class MADS:
  """
  MADS class object that has both the search and poll steps
  where each considers secondary and primary frame centers
  """

  def __init__(self, data: dict):
    """ Initialize the log file """
    self.param: Optional[Parameters] = None
    self.post: Optional[PostMADS] = None
    self.out: Optional[Output] = None
    self.options: Optional[Options] = None
    self.iteration: int = 0
    self.peval: int = 0
    self.log: Optional[logger] = None
    self.active_barrier: Optional[AdaptiveBarrier] = None
    self.state: MadsState = None
    self.stats: MadsStatistics = MadsStatistics()
    self.log = logger()
    if not os.path.exists(data["param"]["post_dir"]):
      try:
        os.mkdir(data["param"]["post_dir"])
      except ValueError:
        os.makedirs(data["param"]["post_dir"], exist_ok=True)

    self.log.initialize(data["param"]["post_dir"] + "/OMADS.log")
    self.log.log_msg(msg="Preprocess the MADS algorithim...",
                     msg_type=MSG_TYPE.INFO)
    self.log.log_msg(msg="Preprocess the search step...",
                     msg_type=MSG_TYPE.INFO)

    self.log.log_msg(msg="Preprocess the POLL step...",
                     msg_type=MSG_TYPE.INFO)

    self.out.step_name = "Poll"

    self.state = MadsState()

  def check_mads_parameters(self):
    """ Initial exceptions check for parameters dict

    :raises IOError: ρ is a positive parameter
    :raises IOError: w_min is a positive parameter
    :raises IOError: h_init is a positive parameter
    :raises IOError: max_size is a strictly positive parameter
    """
    if self.param.rho <= 0:
      raise IOError("ρ is a positive parameter")
    if self.param.w_min < 0:
      raise IOError("w_min is a positive parameter")
    if self.param.h_init < 0:
      raise IOError("h_init is a positive parameter")
    if self.param.max_size <= 0:
      raise IOError("max_size is a strictly positive parameter")
    # reset rng in case someone changes the seed
    self.param.rng = np.random.Generator(
        np.random.MT19937(seed=self.options.seed))

  def check_mads_options(self):
    """ Initial exceptions check for options dict

    :raises IOError: _description_
    :raises IOError: _description_
    """
    if self.options.budget <= 0:
      raise IOError(
          "The number of blackbox evaluations cannot be negative !")
    if self.options.noutbound_hits_max <= 0:
      raise IOError("The number of blackbox tentative evaluations \
                    outside the bounds cannot be negative !")

  def check_mads_iteration_attributes(self):
    """TODO: Implement and check it
    """
    # if not any([step == self.attributes.first_center_steps[0] \
    # for step in available_poll_symbols]):
    #   raise IOError("A poll step must be performed around the first center")
    return

  def reach_nevals_bb_max(self) -> bool:
    """_summary_
    :return: Check whther exhausted eval budget
    :rtype: bool
    """
    return self.bb_eval >= self.options.budget

  def reach_noutbound_hits_max(self) -> bool:
    """_summary_
    :return: Reached the maximum number of times hitting or exceeding any design variable bounds
    :rtype: bool
    """
    return self.bb_eval >= self.options.noutbound_hits_max

  def init_phase(self):
    """Preliminary checks, validation and possibly search
    """
    # Preliminary checks
    self.check_mads_parameters()
    self.check_mads_options()
    self.check_mads_iteration_attributes()

    # Reset storing structures if change in parameters
    self.param.max_size = max(self.param.max_size, self.options.budget)
    if self.active_barrier is None:
      if self.param.is_pareto:
        self.active_barrier = AdaptiveBarrier(
            self.param, self.options, self.hashtable.get_all_cache_points())
      else:
        self.active_barrier = AdaptiveBarrier(self.param)
    self.state.stop_reason = STOP_TYPE.NO_INIT_CANDIDATES

    self.active_barrier.update_with_points(
        eval_point_list=self.hashtable.get_cache_candidate_points())

    # No init candidates: all evaluated points are above the h_max threshold.
    # Trigger phase one.
    if len(
            self.active_barrier.get_fk()) == 0 and len(
            self.active_barrier.get_uk()) == 0:
      self.state.stop_reason = STOP_TYPE.UNKNOWN_STOP_REASON
      self.state.is_phase_one = True
      return

    # Other cases
    if self.state.stop_reason not in \
            [STOP_TYPE.MAX_BB_EVAL_REACHED, STOP_TYPE.MAX_BB_OUTBOUND_REACHED]:
      self.state.stop_reason = STOP_TYPE.UNKNOWN_STOP_REASON
      return


def main(*args) -> Dict[str, Any]:  # noqa: C901
  """ MADS main algorithm """

  # """ Validate and parse the parameters file """
  validate = validator()
  data: dict = validate.check_input_file(args=args)

  # Install the wrapper
  # 1. Suppress Python warnings
  warnings.filterwarnings("ignore")

  # 2. Wrap stderr for any other warning‑style output
  _original_stderr = sys.stderr
  sys.stderr = WarningSuppressor(sys.stderr)
  del validate
  total_time: int = 0
  # """ Initialize the log file """
  log = logger()
  if not os.path.exists(data["param"]["post_dir"]):
    try:
      os.mkdir(data["param"]["post_dir"])
    except KeyError:
      os.makedirs(data["param"]["post_dir"], exist_ok=True)

  log.initialize(data["param"]["post_dir"] + "/OMADS.log")

  # """ Run preprocessor for the setup of
  #  the optimization problem and for the initialization
  # of optimization process """
  pre = preprocess(
      data=data, log=log, sampler_t=SAMPLER_TYPE.POLL)
  iteration, _, poll_sampler, options, param, post, out, active_barrier, \
      out_p, state, stats, bb_handle, hashtable, ms = pre.initialize_from_dict()
  del pre

  pre = preprocess(
      data=data, log=log, sampler_t=SAMPLER_TYPE.SEARCH)
  _, _, search_sampler, _, _, _, _, _, _, _, _, _, _, _ = pre.initialize_from_dict(
      ignore_ms=True)
  del pre

  # mads_agent: MADS = MADS(data=data)
  # iteration: int

  # """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))

  # """ Start the count down for calculating the runtime indicator """
  if search_sampler.type == SEARCH_TYPE.VNS.name:
    search_vn = VNS(params=param)
    search_vn._ns_dist = [
        int(
            ((search_sampler.dim + 1) / 2) *
            ((search_sampler.dim + 2) / 2) /
            (len(search_vn._dist)))
        if search_sampler.ns is None else
        search_sampler.ns] * len(search_vn._dist)
    search_sampler.ns = sum(search_vn._ns_dist)
  else:
    search_vn = None

  state.h_max = active_barrier.h_max

  search_vn = None
  if search_sampler.type == SEARCH_TYPE.VNS.name:
    search_vn = VNS(params=param)
    search_vn.ns_dist = [
        int(
            ((search_sampler.dim + 1) / 2) * ((search_sampler.dim + 2) / 2) /
            (len(search_vn.dist)))
        if search_sampler.ns is None else search_sampler.ns] * len(
        search_vn.dist)
    search_sampler.ns = sum(search_vn.ns_dist)
  can_search = False if ms is None else True
  while True:
    # """ Run search step (Optional) """
    # COMPLETED: This rule cannot be generalized -- needs further invistigation
    # if poll.dim > 10 and poll.mesh.psize >= 1E-4:
    #   canSearch = False
    # else:
    tic = time.perf_counter()

    # 1- Select incumbents (prim. and sec.)
    set_frame_centers_and_hvalues(
        state=state, active_barrier=active_barrier, param=param,
        options=options)

    # 2- Run the poll step
    if not can_search:
      log.log_msg(
          f"------- Iteration # {iteration}: Run the poll step -------",
          MSG_TYPE.INFO)
      poll_cycle(
          poll=poll_sampler, options=options, param=param, state=state,
          stats=stats, active_barrier=active_barrier, iteration=iteration,
          log=log, post=post, bb_handle=bb_handle, out=out,
          hashtable=hashtable, ms=ms)

      # 3- Updates
      # search_sampler.mesh = copy.deepcopy(poll_sampler.mesh)
      # poll_sampler.frame_size = copy.deepcopy(
      #     poll_sampler.mesh.get_delta_frame_size().coordinates)
      # search_sampler.psize = copy.deepcopy(poll_sampler.frame_size)
      update(sampler=poll_sampler, state=state, active_barrier=active_barrier,
             stats=stats, options=options, hashtable=hashtable)
      post.h_max = active_barrier.h_max

      toc = time.perf_counter()
      total_time += toc - tic
      # 4- Generate output files
      log.log_msg(
          msg=f"Iteration {poll_sampler.iter} completed in {toc - tic:.4f} seconds",
          msg_type=MSG_TYPE.INFO)
      status = (
          'Full Success' if state.last_success == SUCCESS_TYPES.FS else
          'Partial Success' if state.last_success == SUCCESS_TYPES.PS else
          'Unsuccessful'
      )

      log.log_msg(
          msg=f"Iteration {poll_sampler.iter} success status: {status}",
          msg_type=MSG_TYPE.INFO
      )
      log.log_msg(
          msg=post,
          msg_type=MSG_TYPE.INFO)

    # """ Run the search step (optional step) """
    can_search = (iteration == 1 and ms is not None) or (
        state.last_success == SUCCESS_TYPES.US or state.last_success  # noqa: E712
        == False) and iteration > 1
    if can_search:
      log.log_msg(
          f"------- Iteration # {iteration}: Run the search step -------",
          MSG_TYPE.INFO)
      search_sampler.iter = iteration
      stats.niterations = iteration
      search_cycle(
          search=search_sampler, options=options, param=param, state=state,
          stats=stats, active_barrier=active_barrier, iteration=iteration,
          log=log, post=post, search_vns=search_vn, bb_handle=bb_handle,
          hashtable=hashtable, out=out, ms=ms)
      toc = time.perf_counter()
      total_time += toc - tic
      update(sampler=search_sampler, state=state,
             active_barrier=active_barrier, stats=stats, options=options,
             hashtable=hashtable)
      post.h_max = active_barrier.h_max
      log.log_msg(
          msg=f"Iteration {search_sampler.iter} completed in {toc - tic:.4f} seconds",
          msg_type=MSG_TYPE.INFO)
      status = (
          'Full Success' if state.last_success == SUCCESS_TYPES.FS else
          'Partial Success' if state.last_success == SUCCESS_TYPES.PS else
          'Unsuccessful'
      )

      log.log_msg(
          msg=f"Iteration {search_sampler.iter} success status: {status}",
          msg_type=MSG_TYPE.INFO
      )
      log.log_msg(
          msg=post,
          msg_type=MSG_TYPE.INFO)
      if state.last_success == SUCCESS_TYPES.US:
        can_search = False

    # # bt = all(abs(active_barrier.meshes[state.fk_frame_center].get_delta_frame_size(
    # # ).coordinates[pp]) < options.tol for pp in range(poll_sampler.n))
    # if state.fk_frame_center == -1:
    #   mk = active_barrier.meshes[state.ordered_frame_centers[0]]
    # else:
    #   mk = active_barrier.meshes[state.fk_frame_center]
    # search_sampler.mesh = mk
    # poll_sampler.mesh = mk
    # update(sampler=search_sampler, state=state,
    #        active_barrier=active_barrier, stats=stats, options=options,
    #        hashtable=hashtable)
    # """ Check stopping criteria"""
    pt = (
        all(
            abs(poll_sampler.mesh.get_delta_frame_size().coordinates[pp]) <
            options.tol for pp in range(poll_sampler.n)))
    st = (
        all(
            abs(
                search_sampler.mesh.get_delta_mesh_size().coordinates
                [pp]) < options.tol
            for pp in range(search_sampler.mesh.n)))
    if pt or st:
      state.stop_reason = STOP_TYPE.MIN_MESH_REACHED
    # bt = state.stop_reason == STOP_TYPE.MIN_MESH_REACHED
    # last_success = state.last_success
    # if not isinstance(state.last_success, SUCCESS_TYPES)
    # else state.last_success.name
    # """ Updates """

    if options.save_results:
      out.h_max = active_barrier.h_max
      post.output_results(out, False)
      if param.is_pareto:
        post.nd_points = []
        for i in range(len(active_barrier.get_all_points())):
          post.nd_points.append(
              active_barrier.get_all_points()[i])
        post.output_nd_results(out_p)
    state.last_success = SUCCESS_TYPES.US
    pb = ProgressBar(options, active_barrier)
    pb.display(peval=stats.neval_bb)
    if (pt or st or state.stop_reason == STOP_TYPE.MIN_MESH_REACHED or stats.neval_bb >= options.budget):
      log.log_msg(
          "\n--------------- Termination of MADS  ---------------", MSG_TYPE.INFO)
      if pt:
        log.log_msg(
            "Termination criterion hit: the poll size is below the minimum threshold defined.",
            MSG_TYPE.INFO)
      if st:
        log.log_msg(
            "Termination criterion hit: the mesh size is below the minimum threshold defined.",
            MSG_TYPE.INFO)
      if search_sampler.bb_eval + poll_sampler.bb_eval >= options.budget:
        log.log_msg(
            "Termination criterion hit: Evaluation budget is exhausted.",
            MSG_TYPE.INFO)
      log.log_msg(
          "----------------------------------------------------\n", MSG_TYPE.INFO)
      break

    # toc = time.perf_counter()
    if state.stop_reason == STOP_TYPE.STOP_IF_FEASIBLE:
      if options.display:
        log.log_msg(
            "Feasible point found: end of phase one.", MSG_TYPE.INFO)

    # progress_bar_colored(
    #     options=options, active_barrier=active_barrier, peval=stats.neval_bb)
    iteration += 1
  sys.stderr = _original_stderr
  if param.is_pareto:
    rp: Optional[CandidatePoint] = None
    if param.ref_point:
      rp = Point()
      rp.coordinates = param.ref_point
    perf_m = Metrics(
        nd_solutions=[elem
                      for elem in
                      active_barrier.get_fk()],
        nobj=active_barrier.nobj, ref_point=rp)
    hv = perf_m.hypervolume()

  # out_step: Any = None
  # if poll_sampler.xmin < search_sampler.xmin:
  #   out_step = poll_sampler
  # elif search_sampler.xmin < poll_sampler.xmin:
  #   out_step = search_sampler
  # else:
  #   out_step = poll_sampler

  # if out_step is None:
  #   out_step = poll_sampler
  xmin_found = False
  if active_barrier is not None:
    if len(active_barrier.get_fk()) > 0:
      xmin: CandidatePoint = active_barrier.get_fk()[-1]
      xmin_found = True

    if any(active_barrier.within_uk):
      xsc: CandidatePoint = active_barrier.get_uk()[-1]
    else:
      xsc = CandidatePoint()
    if not xmin_found:
        xmin = xsc
  else:
    raise IOError("Internal error: empty active_barrier object!")
  
  if options.save_coordinates:
    post.output_coordinates(out)
  if options.display:
    print(" end of MADS ")
    print(" Final objective value: " + str(xmin.f) +
          ", hmin= " + str(xmin.h))

  if log is not None:
    log.log_msg(
        msg=" --- MADS Run Summary--- ", msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" Run completed in {total_time:.4f} seconds",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" # of successful search steps = {search_sampler.n_successes}",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" # of successful poll steps = {poll_sampler.n_successes}",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" Random numbers generator's seed {options.seed}",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" xmin = {xmin.coordinates} ",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" hmin = {xmin.h} ", msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" fmin {xmin.fobj}", msg_type=MSG_TYPE.INFO)
    if xsc.evaluated:
      log.log_msg(
          msg=f" Secondary xmin = {xsc.coordinates}", msg_type=MSG_TYPE.INFO)
      log.log_msg(msg=f" Secondary hmin = {xsc.h}", msg_type=MSG_TYPE.INFO)
      log.log_msg(msg=f" Secondary fmin = {xsc.fobj}", msg_type=MSG_TYPE.INFO)
      log.log_msg(
          msg=f" Feasibility margin (h_max) = {active_barrier.h_max} ",
          msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f"# BB feasible evals =  {stats.neval_bb_feasible}",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" # BB infeasible evals =  {stats.neval_bb_infeasible}",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" Total # BB evals =  {stats.neval_bb} ",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" # Duplicate (cache) hits =  {stats.ncache_hits}  ",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" # iterations =  {iteration} ", msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" psize = {active_barrier.meshes[state.ordered_frame_centers[0]].get_delta_frame_size().coordinates} ",
        msg_type=MSG_TYPE.INFO)
    if param.is_pareto:
      log.log_msg(
          msg=f" Hypervolume metric = {hv}", msg_type=MSG_TYPE.INFO)
  if options.display:
    print("\n ---MADS Run Summary---")
    print(f" Run completed in {toc - tic:.4f} seconds")
    print(f" Random numbers generator's seed {options.seed}")
    print(" xmin = " + str(xmin))
    print(" hmin = " + str(xmin.h))
    print(" fmin = " + str(xmin.f))
    if xsc.evaluated:
      print(" Secondary xmin =" + str(xsc.coordinates))
      print(" Secondary hmin = " + str(xsc.h))
      print(" Secondary fmin = " + str(xsc.fobj))
      print(" Feasibility margin (h_max) = " + str(active_barrier.h_max))
    print(" #bb_eval = " + str(stats.neval_bb))
    print(f"# BB feasible evals =  {stats.neval_bb_feasible}")
    print(f"# BB infeasible evals =  {stats.neval_bb_infeasible}")
    print(" # iteration = " + str(iteration))
    print(" # Duplicate (cache) hits = " + str(stats.ncache_hits))
    print(" nb_success = " +
          str(poll_sampler.n_successes + search_sampler.n_successes))
    print(" psize = " + str(poll_sampler.mesh.get_delta_frame_size().coordinates))
    print(
        " psize_success = " +
        str(
            active_barrier.meshes[state.ordered_frame_centers[0]].
            get_delta_frame_size().coordinates))

  # """ Evaluation of the blackbox; get output responses """
  if xmin.sets is not None and isinstance(xmin.sets, dict):
    p: List[Any] = []
    for i, _ in enumerate(xmin.var_type):
      if (xmin.var_type[i] == VAR_TYPE.DISCRETE or xmin.var_type[i] == VAR_TYPE.CATEGORICAL) \
              and xmin.var_link[i] is not None:
        p.append(xmin.sets[xmin.var_link[i]][int(xmin.coordinates[i])])
      else:
        p.append(xmin.coordinates[i])
  else:
    p = xmin.coordinates
  output: Dict[str, Any] = {"xmin": p,
                            "fmin": xmin.f,
                            "hmin": xmin.h,
                            "nbb_evals": stats.neval_bb,
                            "niterations": iteration,
                            "nb_success": poll_sampler.n_successes + search_sampler.n_successes,
                            "psize": poll_sampler.mesh.get_delta_frame_size().coordinates,
                            "psuccess":
                            active_barrier.meshes[state.ordered_frame_centers[0]].get_delta_frame_size(
                            ).coordinates,
                            # "pmax": poll.mesh.psize_max,
                            "msize": active_barrier.meshes[state.ordered_frame_centers[0]].get_delta_mesh_size().coordinates,
                            "HV": hv if param.is_pareto else "NA"}
  return output, poll_sampler, search_sampler


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
