# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - ORTHO-MADS (MADS)                                    #
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
  This is a python implementation of the orothognal mesh adaptive direct search method (OMADS)
"""
import copy
import json
from multiprocessing import freeze_support
import os
import sys
import numpy as np
import concurrent.futures
import time
from typing import List, Dict, Any
from BMDFO import toy
from .Point import Point
from .Barriers import *
from ._common import *
from .Directions import *
from .PrePoll import *
from .CandidatePoint import CandidatePoint

def main(*args) -> Dict[str, Any]:
  """ MADS: Poll step main algorithm """

  """ Validate and parse the parameters file """
  validate = validator()
  data: dict = validate.checkInputFile(args=args)

  """ Initialize the log file """
  log = logger()
  if not os.path.exists(data["param"]["post_dir"]):
     os.mkdir(data["param"]["post_dir"])
  log.initialize(data["param"]["post_dir"] + "/OMADS.log")

  """ Run preprocessor for the setup of
   the optimization problem and for the initialization
  of optimization process """
  iteration, xmin, poll, options, param, post, out, B = PrePoll(data).initialize_from_dict(log=log)
  out.stepName = "Poll"

  """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))

  """ Start the count down for calculating the runtime indicator """
  tic = time.perf_counter()
  peval = 0
  LAMBDA_k = xmin.LAMBDA
  RHO_k = xmin.RHO
  while True:
    del poll.poll_set
    poll.poll_set = []
    poll.mesh.update()
    poll.LAMBDA = copy.deepcopy(xmin.LAMBDA)
    """ Create the set of poll directions """
    hhm = poll.create_housholder(options.rich_direction, domain=xmin.var_type)
    poll.lb = param.lb
    poll.ub = param.ub
    if B is not None:
      if B._filter is not None:
        B.select_poll_center()
        B.update_and_reset_success()
      else:
        B.insert(xmin)
    poll.hmax = xmin.hmax
    poll.create_poll_set(hhm=hhm,
               ub=param.ub,
               lb=param.lb, it=iteration, var_type=xmin.var_type, var_sets=xmin.sets, var_link = xmin.var_link, c_types=param.constraints_type, is_prim=True)
    
    if B._sec_poll_center is not None and B._sec_poll_center.evaluated:
      del poll.poll_set
      # poll.poll_dirs = []
      poll.x_sc = B._sec_poll_center
      poll.create_poll_set(hhm=hhm,
               ub=param.ub,
               lb=param.lb, it=iteration, var_type=B._sec_poll_center.var_type, var_sets=B._sec_poll_center.sets, var_link = B._sec_poll_center.var_link, c_types=param.constraints_type, is_prim=False)
    
    poll.LAMBDA = LAMBDA_k
    poll.RHO = RHO_k

    """ Save current poll directions and incumbent solution
     so they can be saved later in the post dir """
    if options.save_coordinates:
      post.coords.append(poll.poll_set)
      post.x_incumbent.append(poll.xmin)
    """ Reset success boolean """
    poll.success = SUCCESS_TYPES.US
    """ Reset the BB output """
    poll.bb_output = []
    xt = []
    """ Serial evaluation for points in the poll set """
    if log is not None and log.isVerbose:
      log.log_msg(f"----------- Evaluate poll set # {iteration}-----------", msg_type=MSG_TYPE.INFO)
    poll.log = log
    if not options.parallel_mode:
      for it in range(len(poll.poll_set)):
        peval += 1
        if poll.terminate:
          break
        f = poll.eval_poll_point(it)
        xt.append(f[-1])
        if not f[0]:
          post.bb_eval.append(poll.bb_handle.bb_eval)
          post.iter.append(iteration)
          post.psize.append(poll.mesh.getDeltaFrameSize().coordinates)
        else:
          continue

    else:
      poll.point_index = -1
      """ Parallel evaluation for points in the poll set """
      with concurrent.futures.ProcessPoolExecutor(options.np) as executor:
        results = [executor.submit(poll.eval_poll_point,
                       it) for it in range(len(poll.poll_set))]
        for f in concurrent.futures.as_completed(results):
          # if f.result()[0]:
          #     executor.shutdown(wait=False)
          # else:
          if options.save_results or options.display:
            peval = peval +1
            poll.bb_eval = peval
            post.bb_eval.append(peval)
            post.iter.append(iteration)
            # post.poll_dirs.append(poll.poll_dirs[f.result()[1]])
            post.psize.append(f.result()[4])
          xt.append(f.result()[-1])

    xpost: List[CandidatePoint] = poll.master_updates(xt, peval, save_all_best=options.save_all_best, save_all=options.save_results)
    xmin = copy.deepcopy(poll.xmin)
    if options.save_results:
      for i in range(len(xpost)):
        post.poll_dirs.append(xpost[i])
    for xv in xt:
      if xv.evaluated:
        B.insert(xv)

    """ Update the xmin in post"""
    post.xmin = copy.deepcopy(poll.xmin)

    """ Updates """
    pev = 0.
    for p in poll.poll_set:
      if p.evaluated:
        pev += 1
    # if pev != poll.poll_dirs and not poll.success:
    #   poll.seed += 1
    goToSearch: bool = (pev == 0 and poll.Failure_stop is not None and poll.Failure_stop)
    
    dir: Point = Point(poll._n)
    dir.coordinates = poll.xmin.direction if poll.xmin.direction is not None else [0]*poll._n
    if poll.success == SUCCESS_TYPES.FS and not goToSearch:
      poll.mesh.enlargeDeltaFrameSize(direction=dir) # poll.mesh.psize =  np.multiply(poll.mesh.psize, 2, dtype=poll.dtype.dtype
    elif poll.success == SUCCESS_TYPES.US:
      poll.mesh.refineDeltaFrameSize()
      # poll.mesh.psize = np.divide(poll.mesh.psize, 2, dtype=poll.dtype.dtype)
    poll.mesh.updatedeltaMeshSize()
    if log is not None:
        log.log_msg(msg=post.__str__(), msg_type=MSG_TYPE.INFO)
    if options.display:
      print(post)
    
    LAMBDA_k = poll.LAMBDA
    RHO_k = poll.RHO
    
    Failure_check = iteration > 0 and poll.Failure_stop is not None and poll.Failure_stop and (poll.success == SUCCESS_TYPES.US or goToSearch)
    
    if (Failure_check or poll.bb_eval >= options.budget) or (all(abs(poll.mesh.getDeltaFrameSize().coordinates[pp]) < options.tol for pp in range(poll._n)) or poll.bb_eval >= options.budget or poll.terminate):
      log.log_msg(f"\n--------------- Termination of the poll step  ---------------", MSG_TYPE.INFO)
      if all(abs(poll.mesh.getDeltaFrameSize().coordinates[pp]) < options.tol for pp in range(poll._n)):
        log.log_msg("Termination criterion hit: the mesh size is below the minimum threshold defined.", MSG_TYPE.INFO)
      if (poll.bb_eval >= options.budget or poll.terminate):
        log.log_msg("Termination criterion hit: evaluation budget is exhausted.", MSG_TYPE.INFO)
      if (Failure_check):
        log.log_msg(f"Termination criterion hit (optional): failed to find a successful point in iteration # {iteration}.", MSG_TYPE.INFO)
      log.log_msg(f"---------------------------------------------------------------\n", MSG_TYPE.INFO)
      break
    iteration += 1
    

  toc = time.perf_counter()

  """ If benchmarking, then populate the results in the benchmarking output report """
  if len(args) > 1 and isinstance(args[1], toy.Run):
    b: toy.Run = args[1]
    if b.test_suite == "uncon":
      ncon = 0
    else:
      ncon = len(poll.xmin.c_eq) + len(poll.xmin.c_ineq)
    if len(poll.bb_output) > 0:
      b.add_row(name=poll.bb_handle.blackbox,
            run_index=int(args[2]),
            nv=len(param.baseline),
            nc=ncon,
            nb_success=poll.nb_success,
            it=iteration,
            BBEVAL=poll.bb_eval,
            runtime=toc - tic,
            feval=poll.bb_handle.bb_eval,
            hmin=poll.xmin.h,
            fmin=poll.xmin.f)
    print(f"{poll.bb_handle.blackbox}: fmin = {poll.xmin.f} , hmin= {poll.xmin.h:.2f}")

  elif len(args) > 1 and not isinstance(args[1], toy.Run):
    if log is not None:
      log.log_msg(msg="Could not find " + args[1] + " in the internal BM suite.", msg_type=MSG_TYPE.ERROR)
    raise IOError("Could not find " + args[1] + " in the internal BM suite.")

  if options.save_results:
    post.output_results(out)

  if options.display:
    print(" end of orthogonal MADS ")
    if log is not None:
      log.log_msg(msg=" end of orthogonal MADS ")
    print(" Final objective value: " + str(poll.xmin.f) + ", hmin= " + str(poll.xmin.h))
    if log is not None:
      log.log_msg(msg=" Final objective value: " + str(poll.xmin.f) + ", hmin= " + str(poll.xmin.h), msg_type=MSG_TYPE.INFO)
    if log is not None and isinstance(args[1], str):
      log.log_msg(msg=" end of orthogonal MADS running" + args[1] + " in the internal BM suite.", msg_type=MSG_TYPE.INFO)
    

  if options.save_coordinates:
    post.output_coordinates(out)
  
  if log is not None:
    log.log_msg(msg="\n---Run Summary---", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" Run completed in {toc - tic:.4f} seconds", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" Random numbers generator's seed {options.seed}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" xmin = {poll.xmin.__str__()}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" hmin = {poll.xmin.h}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" fmin {poll.xmin.fobj}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" #bb_eval =  {poll.bb_eval}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" #iteration =  {iteration}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f"  nb_success = {poll.nb_success}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" psize = {poll.mesh.getDeltaFrameSize().coordinates}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" psize_success = {poll.mesh.psize_success}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" psize_max = {poll.mesh.psize_max}", msg_type=MSG_TYPE.INFO)
  
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
    print(" psize = " + str(poll.mesh.getDeltaFrameSize().coordinates))
    print(" psize_success = " + str(poll.mesh.psize_success))
    print(" psize_max = " + poll.mesh.psize_max)
    
  xmin = copy.deepcopy(poll.xmin)
  """ Evaluation of the blackbox; get output responses """
  if xmin.sets is not None and isinstance(xmin.sets,dict):
    p: List[Any] = []
    for i in range(len(xmin.var_type)):
      if (xmin.var_type[i] == VAR_TYPE.DISCRETE or xmin.var_type[i] == VAR_TYPE.CATEGORICAL) and xmin.var_link[i] is not None:
        p.append(xmin.sets[xmin.var_link[i]][int(xmin.coordinates[i])])
      else:
        p.append(xmin.coordinates[i])
  else:
    p = xmin.coordinates
  output: Dict[str, Any] = {"xmin": p,
                "fmin": poll.xmin.f,
                "hmin": poll.xmin.h,
                "nbb_evals" : poll.bb_eval,
                "niterations" : iteration,
                "nb_success": poll.nb_success,
                "psize": poll.mesh.getDeltaFrameSize().coordinates,
                "psuccess": poll.mesh.psize_success,
                "pmax": poll.mesh.psize_max,
                "msize": poll.mesh.getdeltaMeshSize()}

  return output, poll

def rosen(x, p, *argv):
  x = np.asarray(x)
  y = [np.sum(p[0] * (x[1:] - x[:-1] ** p[1]) ** p[1] + (1 - x[:-1]) ** p[1],
        axis=0), [0]]
  return y

def alpine(x):
  y = [abs(x[0]*np.sin(x[0])+0.1*x[0])+abs(x[1]*np.sin(x[1])+0.1*x[1]), [0]]
  return y

def Ackley3(x):
  return [-200*np.exp(-0.2*np.sqrt(x[0]**2+x[1]**2))+5*np.exp(np.cos(3*x[0])+np.sin(3*x[1])), [0]]

def eggHolder(individual):
  x = individual[0]
  y = individual[1]
  f = (-(y + 47.0) * np.sin(np.sqrt(abs(x/2.0 + (y + 47.0)))) - x * np.sin(np.sqrt(abs(x - (y + 47.0)))))
  return [f, [0]]

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
    raise IOError("Undefined input args."
            " Please specify an appropriate input (parameters) jason file")

