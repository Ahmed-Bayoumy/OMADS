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
from multiprocessing import freeze_support

import os
import sys
import time
import copy
from typing import List, Dict, Any, Optional
import warnings
import numpy as np

from .._include import SAMPLER_TYPE, SEARCH_TYPE, VAR_TYPE, MSG_TYPE, SUCCESS_TYPES, STOP_TYPE
from .._include import CandidatePoint
from .._include import logger, validator
from .._include import VNS
from .._include import AdaptiveBarrier
from .._include import Point
from .._include import Metrics
from .._include import MadsState, MadsStatistics
from ._search_step import search_cycle
from .._include import set_frame_centers_and_hvalues, update
from .._include import preprocess
from .._include import ProgressBar
from .._include import WarningSuppressor

np.set_printoptions(legacy='1.21')


def main(*args) -> Dict[str, Any]:  # noqa: C901
  """ MADS: Search step main algorithm """

  # """ Validate and parse the parameters file """
  validate = validator()
  data: dict = validate.check_input_file(args=args)
  total_time = 0

  # Install the wrapper
  # 1. Suppress Python warnings
  warnings.filterwarnings("ignore")

  # 2. Wrap stderr for any other warning‑style output
  _original_stderr = sys.stderr
  sys.stderr = WarningSuppressor(sys.stderr)

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
  state = MadsState()
  stats = MadsStatistics()
  pre = preprocess(
      data=data, log=log, sampler_t=SAMPLER_TYPE.SEARCH)
  iteration, _, search, options, param, post, out, active_barrier, \
      out_p, state, stats, bb_handle, hashtable, ms = pre.initialize_from_dict()
  del pre

  if out_p:
    out_p.step_name = "Search_ND"

  # """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))

  out.step_name = f"Search: {search.type}"

  # peval = 0
  search_vn = None
  if search.type == SEARCH_TYPE.VNS.name:
    search_vn = VNS(params=param)
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

  while True:
    # """ Start the count down for calculating the runtime indicator """
    tic = time.perf_counter()
    bbevalold = search.bb_eval
    search.mesh.update()
    search.iter = iteration
    # 1- Select incumbents (prim. and sec.)
    set_frame_centers_and_hvalues(
        state=state, active_barrier=active_barrier, param=param,
        options=options)
    # 2- Run the search step
    search_cycle(
        search=search, options=options, param=param, state=state, stats=stats,
        active_barrier=active_barrier, iteration=iteration, log=log, post=post,
        search_vns=search_vn, bb_handle=bb_handle, hashtable=hashtable, out=out)
    # 3- Updates
    update(sampler=search, state=state, active_barrier=active_barrier,
           stats=stats, options=options, hashtable=hashtable)
    # 4- Generate output files
    toc = time.perf_counter()

    total_time += toc - tic
    if options.save_results:
      post.nd_points = []
      out.h_max = active_barrier.h_max
      post.output_results(out=out, all_res=False)
      if param.is_pareto:
        for i in range(len(active_barrier.get_filled_elements())):
          post.nd_points.append(active_barrier.get_filled_elements()[i])
        post.output_nd_results(out_p)

    post.h_max = active_barrier.h_max

    log.log_msg(
        msg=f"Iteration {search.iter} completed in {toc - tic:.4f} seconds",
        msg_type=MSG_TYPE.INFO)
    status = (
        'Full Success' if state.last_success == SUCCESS_TYPES.FS else
        'Partial Success' if state.last_success == SUCCESS_TYPES.PS else
        'Unsuccessful'
    )

    if state.last_success == SUCCESS_TYPES.US or status == 'Unsuccessful':
      stats.nno_successes += 1
    else:
      stats.nno_successes = 0

    if stats.nno_successes >= options.budget:
      state.stop_reason = STOP_TYPE.UNKNOWN_STOP_REASON

    log.log_msg(
        msg=f"Iteration {search.iter} success status: {status}",
        msg_type=MSG_TYPE.INFO
    )
    log.log_msg(
        msg=post,
        msg_type=MSG_TYPE.INFO)

    failure_check = iteration > 0 and search.failure_stop is not None and \
        search.failure_stop and not (
            state.last_success != SUCCESS_TYPES.FS or SUCCESS_TYPES.PS)
    if stats.neval_bb - bbevalold <= 0:
      num_strat += 1
      search.explore_new = True
    else:
      num_strat = 0
    state.last_success = SUCCESS_TYPES.US
    pb = ProgressBar(options, active_barrier)
    pb.display(peval=stats.neval_bb)
    if (failure_check or stats.neval_bb >= options.budget or state.stop_reason != STOP_TYPE.NO_STOP) or \
        (all(abs(search.mesh.get_delta_mesh_size().coordinates[pp]) < options.tol
         for pp in range(search.mesh.n)) or stats.neval_bb >= options.budget
         or search.terminate):
      log.log_msg(
          "\n--------------- Termination of the search step  ---------------",
          MSG_TYPE.INFO)
      if (all(abs(search.mesh.get_delta_mesh_size().coordinates[pp]) < options.tol
              for pp in range(search.mesh.n))):
        log.log_msg(
            "Termination criterion hit: the mesh size is below the minimum threshold defined.",
            MSG_TYPE.INFO)
      if (stats.neval_bb >= options.budget or search.terminate):
        log.log_msg(
            "Termination criterion hit: evaluation budget is exhausted.",
            MSG_TYPE.INFO)
      if failure_check:
        log.log_msg(
            "Termination criterion hit (optional): failed to find a \
            successful point in iteration # {iteration}.",
            MSG_TYPE.INFO)
      if (state.stop_reason != STOP_TYPE.NO_STOP):
        log.log_msg(
            f"Termination criterion hit: {state.stop_reason.name}.",
            MSG_TYPE.INFO)

      log.log_msg(
          "-----------------------------------------------------------------\n",
          MSG_TYPE.INFO)
      break
    iteration += 1
  sys.stderr = _original_stderr
  if search.prob_params.is_pareto and isinstance(
          active_barrier, AdaptiveBarrier):
    rp: Optional[CandidatePoint] = None
    if param.ref_point:
      rp = Point()
      rp.coordinates = param.ref_point
    perf_m = Metrics(
        nd_solutions=active_barrier.get_filled_elements(),
        nobj=active_barrier.dims[1], ref_point=rp)
    hv = perf_m.hypervolume()

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
    log.log_msg(msg="\n---Run Summary---", msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" Run completed in {total_time:.4f} seconds",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" Random numbers generator's seed {options.seed}",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" Primary xmin = {xmin.coordinates}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" Primary hmin = {xmin.h}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" Primary fmin = {xmin.fobj}", msg_type=MSG_TYPE.INFO)
    if xsc.evaluated:
      log.log_msg(
          msg=f" Secondary xmin = {xsc.coordinates}", msg_type=MSG_TYPE.INFO)
      log.log_msg(msg=f" Secondary hmin = {xsc.h}", msg_type=MSG_TYPE.INFO)
      log.log_msg(msg=f" Secondary fmin = {xsc.fobj}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" #bb_eval =  {stats.neval_bb}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" #iteration =  {iteration}", msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f"  nb_success = {stats.nfull_successes}", msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" psize = {search.mesh.get_delta_frame_size().coordinates}",
        msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" psize_success = {active_barrier.meshes[state.ordered_frame_centers[0]].get_delta_frame_size().coordinates}",
        msg_type=MSG_TYPE.INFO)

  if options.display:
    print("\n ---Run Summary---")
    print(f" Run completed in {total_time:.4f} seconds")
    print(f" Random numbers generator's seed {options.seed}")
    print(" Primary xmin =" + str(xmin.coordinates))
    print(" Primary hmin = " + str(xmin.h))
    print(" Primary fmin = " + str(xmin.fobj))
    if xsc.evaluated:
      print(" Secondary xmin =" + str(xsc.coordinates))
      print(" Secondary hmin = " + str(xsc.h))
      print(" Secondary fmin = " + str(xsc.fobj))
    print(" #bb_eval = " + str(stats.neval_bb))
    print(" #iteration = " + str(iteration))
    print(" nb_success = " + str(stats.nfull_successes))
    print(" psize = " + str(search.mesh.get_delta_frame_size().coordinates))
    print(
        " psize_success = " +
        str(active_barrier.meshes[state.ordered_frame_centers[0]].get_delta_frame_size().coordinates))

  xmin = copy.deepcopy(xmin)
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
                            "fmin": xmin.f,
                            "hmin": xmin.h,
                            "Secondary fmin": xsc.f,
                            "Secondary hmin": xsc.h,
                            "nbb_evals": stats.neval_bb,
                            "niterations": iteration,
                            "nb_success": stats.nfull_successes,
                            "psize":
                            active_barrier.meshes[state.ordered_frame_centers[0]].get_delta_frame_size(
                            ).coordinates,
                            "psuccess":
                            active_barrier.meshes[state.ordered_frame_centers[0]].get_delta_frame_size(
                            ).coordinates,
                            # "pmax": poll.mesh.psize_max,
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
