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
import copy
import os
import sys
import time
from multiprocessing import freeze_support
from typing import List, Dict, Any, Optional
import warnings

import numpy as np
from .._include import MSG_TYPE, SAMPLER_TYPE, STOP_TYPE, SUCCESS_TYPES, VAR_TYPE
from .._include import Point
from .._include import validator, logger
from .._include import CandidatePoint
from .._include import Metrics
from .._include import MadsState, MadsStatistics

from ._poll_step import poll_cycle
from .._include import set_frame_centers_and_hvalues, update
from .._include import preprocess
from .._include import ProgressBar
from .._include import WarningSuppressor
np.set_printoptions(legacy='1.21')


def main(*args) -> Dict[str, Any]:  # noqa: C901
  """MADS: Poll step main algorithm

  :raises IOError: Exceptions
  :return: output dictionary
  :rtype: Dict[str, Any]
  """
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
  total_time = 0

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
      data=data, log=log, sampler_t=SAMPLER_TYPE.POLL)
  iteration, _, poll, options, param, post, out, active_barrier, \
      out_p, state, stats, bb_handle, hashtable = pre.initialize_from_dict()
  del pre
  out.step_name = "Poll"
  if out_p:
    out_p.step_name = "Poll_ND"

  # """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))

  out.step_name = f"Poll: {poll.__class__.__name__}"

  log.log_msg(
      msg=f"---------------- Run the Poll step ({poll.__class__.__name__}) ----------------",
      msg_type=MSG_TYPE.INFO)

  while True:
    # """ Start the count down for calculating the runtime indicator """
    tic = time.perf_counter()
    del poll.candidate_points_set
    del poll.poll_dirs

    # 1- Select incumbents (prim. and sec.)
    set_frame_centers_and_hvalues(
        state=state, active_barrier=active_barrier, param=param,
        options=options)
    # 2- Run the poll step
    poll_cycle(
        poll=poll, options=options, param=param, state=state, stats=stats,
        active_barrier=active_barrier, iteration=iteration,
        log=log, post=post, bb_handle=bb_handle, out=out, hashtable=hashtable)
    # 3- Updates
    update(sampler=poll, state=state, active_barrier=active_barrier,
           stats=stats, options=options, hashtable=hashtable)

    toc = time.perf_counter()

    total_time += toc - tic

    # 4- Generate output files

    log.log_msg(
        msg=f"Iteration {poll.iter} completed in {toc - tic:.4f} seconds",
        msg_type=MSG_TYPE.INFO)
    status_text = (
        "Full Success" if state.last_success == SUCCESS_TYPES.FS
        else "Partial Success" if state.last_success == SUCCESS_TYPES.PS
        else "Unsuccessful"
    )

    msg = (
        f"Iteration {poll.iter}  success status: "
        f"{status_text}"
    )
    log.log_msg(msg=msg, msg_type=MSG_TYPE.INFO)
    post.h_max = active_barrier.h_max
    log.log_msg(
        msg=post,
        msg_type=MSG_TYPE.INFO)

    failure_check = iteration > 0 and state.stop_reason is not None \
        and state.stop_reason != STOP_TYPE.UNKNOWN_STOP_REASON and (
            state.last_success == SUCCESS_TYPES.US)
    state.last_success = SUCCESS_TYPES.US
    pb = ProgressBar(options, active_barrier)
    pb.display(peval=stats.neval_bb)
    if (failure_check or stats.neval_bb >= options.budget) or \
        (all([abs(poll.mesh.get_delta_frame_size().coordinates[pp]) < options.tol
         for pp in range(poll.dim)]) or stats.neval_bb >= options.budget or poll.terminate):
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
  sys.stderr = _original_stderr
  toc = time.perf_counter()
  if poll.prob_params.is_pareto:
    rp: Optional[CandidatePoint] = None
    if param.ref_point:
      rp = Point()
      rp.coordinates = param.ref_point
    perf_m = Metrics(
        nd_solutions=active_barrier.get_filled_elements(
        ) if not param.is_pareto else active_barrier.get_nd_elements(),
        nobj=active_barrier.dims[1], ref_point=rp)
    hv = perf_m.hypervolume()

  xmin: CandidatePoint = active_barrier.get_fk()[-1]
  if any(active_barrier.within_uk):
    xsc: CandidatePoint = active_barrier.get_uk()[-1]
  else:
    xsc = CandidatePoint()
  if options.display:
    print(" end of orthogonal MADS ")
    if log:
      log.log_msg(msg=" end of orthogonal MADS ", msg_type=MSG_TYPE.INFO)
    print(
        " Final objective value: " + str(xmin.f) + ", hmin= " +
        str(xmin.h))
    if log:
      log.log_msg(
          msg=" Final objective value: " + str(xmin.f) + ", hmin= " +
          str(xmin.h),
          msg_type=MSG_TYPE.INFO)
    if log and len(args) > 1 and isinstance(args[1], str):
      log.log_msg(
          msg=" end of orthogonal MADS running" + args[1] +
          " in the internal BM suite.", msg_type=MSG_TYPE.INFO)

  if options.save_coordinates:
    post.output_coordinates(out)

  if options.save_results:
    post.nd_points = []
    out.h_max = active_barrier.h_max
    post.output_results(out=out, all_res=False)
    if param.is_pareto:
      for i in range(len(active_barrier.get_filled_elements())):
        post.nd_points.append(active_barrier.get_filled_elements()[i])
      post.output_nd_results(out_p)

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
    log.log_msg(msg=f" #bb_eval =  {poll.bb_eval}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" #iteration =  {iteration}", msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f"  nb_success = {poll.nb_success}", msg_type=MSG_TYPE.INFO)
    log.log_msg(
        msg=f" psize = {poll.mesh.get_delta_frame_size().coordinates}",
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
    print(" #bb_eval = " + str(poll.bb_eval))
    print(" #iteration = " + str(iteration))
    print(" nb_success = " + str(poll.nb_success))
    print(" psize = " + str(poll.mesh.get_delta_frame_size().coordinates))
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
                            "nbb_evals": poll.bb_eval,
                            "niterations": iteration,
                            "nb_success": poll.nb_success,
                            "psize":
                            active_barrier.meshes[state.ordered_frame_centers[0]].get_delta_frame_size(
                            ).coordinates,
                            "psuccess":
                            active_barrier.meshes[state.ordered_frame_centers[0]].get_delta_frame_size(
                            ).coordinates,
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
