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
import importlib
from multiprocessing import freeze_support
import os
import sys
import numpy as np
import time
from typing import List, Dict, Any, Optional
import importlib.util
if importlib.util.find_spec('BMDFO'):
  from BMDFO import toy
from .Point import Point
from .Barriers import Barrier, BarrierMO
from ._common import validator, logger
from .PrePoll import PrePoll
from .CandidatePoint import CandidatePoint
from ._globals import DESIGN_STATUS, MSG_TYPE, SUCCESS_TYPES, VAR_TYPE
from .Metrics import Metrics
np.set_printoptions(legacy='1.21')

def main(*args) -> Dict[str, Any]:
  """ MADS: Poll step main algorithm """

  """ Validate and parse the parameters file """
  validate = validator()
  data: dict = validate.check_input_file(args=args)

  """ Initialize the log file """
  log = logger()
  if not os.path.exists(data["param"]["post_dir"]):
    try:
      os.mkdir(data["param"]["post_dir"])
    except:
      os.makedirs(data["param"]["post_dir"], exist_ok=True)
  log.initialize(data["param"]["post_dir"] + "/OMADS.log")

  """ Run preprocessor for the setup of
   the optimization problem and for the initialization
  of optimization process """
  iteration, xmin, poll, options, param, post, out, B, out_p = PrePoll(data).initialize_from_dict(log=log)
  out.step_name = "Poll"
  if out_p:
    out_p.step_name = "Poll_ND"

  """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))

  """ Start the count down for calculating the runtime indicator """
  tic = time.perf_counter()
  peval = poll.bb_handle.bb_eval
  lambda_k = xmin.lambda_multipliers
  rho_k = xmin.rho
  while True:
    del poll.poll_set
    poll.mesh.update()
    poll.constraints_RP.LAMBDA = copy.deepcopy(xmin.lambda_multipliers)
    poll.constraints_RP.constraints_type = copy.deepcopy(poll.xmin.constraints_type)
    """ Create the set of poll directions """
    hhm = poll.create_housholder(options.rich_direction, domain=xmin.var_type)
    poll.lb = param.lb
    poll.ub = param.ub
    xmin.mesh = copy.deepcopy(poll.mesh)
    if B is not None:
      if isinstance(B, Barrier):
        if B._filter is not None:
          B.select_poll_center()
          B.update_and_reset_success()
        else:
          B.insert(xmin)
      elif isinstance(B, BarrierMO) and iteration == 1:
        B.init(eval_point_list=[xmin])
    
    if isinstance(B, Barrier):
      poll.constraints_RP.hmax = xmin.h_max
      poll.create_poll_set(hhm=hhm,
                ub=param.ub,
                lb=param.lb, it=iteration, var_type=xmin.var_type, var_sets=xmin.sets, var_link = xmin.var_link, c_types=param.constraints_type, is_prim=True)
      if B._sec_poll_center is not None and B._sec_poll_center.evaluated:
        del poll.poll_set
        poll.x_sc = B._sec_poll_center
        poll.create_poll_set(hhm=hhm,
                ub=param.ub,
                lb=param.lb, it=iteration, var_type=B._sec_poll_center.var_type, var_sets=B._sec_poll_center.sets, var_link = B._sec_poll_center.var_link, c_types=param.constraints_type, is_prim=False)
    elif isinstance(B, BarrierMO):
      poll.constraints_RP.hmax = B._h_max
      del poll.poll_set
      del poll.poll_dirs
      if B._currentIncumbentFeas and B._currentIncumbentFeas.evaluated:
        poll.create_poll_set(hhm=hhm,
                ub=param.ub,
                lb=param.lb, it=iteration, var_type=B._currentIncumbentFeas.var_type, var_sets=B._currentIncumbentFeas.sets, var_link = B._currentIncumbentFeas.var_link, c_types=param.constraints_type, is_prim=True)
      elif poll.xmin.status == DESIGN_STATUS.FEASIBLE:
        poll.create_poll_set(hhm=hhm,
                ub=param.ub,
                lb=param.lb, it=iteration, var_type=poll.xmin.var_type, var_sets=poll.xmin.sets, var_link = poll.xmin.var_link, c_types=param.constraints_type, is_prim=True)
      
      if B._currentIncumbentInf and B._currentIncumbentInf.evaluated:
        poll.x_sc = B._currentIncumbentInf
        poll.create_poll_set(hhm=hhm,
                ub=param.ub,
                lb=param.lb, it=iteration, var_type=B._currentIncumbentInf.var_type, var_sets=B._currentIncumbentInf.sets, var_link = B._currentIncumbentInf.var_link, c_types=param.constraints_type, is_prim=False)
      elif poll.xmin.status == DESIGN_STATUS.INFEASIBLE:
        poll.create_poll_set(hhm=hhm,
                ub=param.ub,
                lb=param.lb, it=iteration, var_type=poll.xmin.var_type, var_sets=poll.xmin.sets, var_link = poll.xmin.var_link, c_types=param.constraints_type, is_prim=False)
      
    
    poll.constraints_RP.LAMBDA = lambda_k
    poll.constraints_RP.RHO = rho_k

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
    if log and log.is_verbose:
      log.log_msg(f"----------- Evaluate poll set # {iteration}-----------", msg_type=MSG_TYPE.INFO)
    poll.log = log
    if options.check_cache:
      poll.omit_duplicates()
    poll.bb_handle.xmin = poll.xmin
    if not options.parallel_mode:
      xt, post, peval = poll.bb_handle.run_callable_serial_local(iter=iteration, peval=peval, eval_set=poll.poll_set, options=options, post=post, psize=poll.mesh.getDeltaFrameSize().coordinates, constraints_relaxation=poll.constraints_RP.__dict__, budget=options.budget)
    else:
      poll.point_index = -1
      """ Parallel evaluation for points in the poll set """
      poll.bb_eval, xt, post, peval = poll.bb_handle.run_callable_parallel_local(iter=iteration, peval=peval, eval_set=poll.poll_set, options=options, post=post, psize=poll.mesh.getDeltaFrameSize().coordinates, constraints_relaxation=poll.constraints_RP.__dict__, budget=options.budget)
    poll.postprocess_evaluated_candidates(xt)
    if isinstance(B, Barrier):
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

      go_to_search: bool = (pev == 0 and poll.Failure_stop is not None and poll.Failure_stop)
      
      direction: Point = Point(poll._n)
      direction.coordinates = poll.xmin.direction.coordinates if poll.xmin.direction is not None else [0]*poll._n
      if poll.success == SUCCESS_TYPES.FS and not go_to_search:
        poll.mesh.enlargeDeltaFrameSize(direction=direction) # poll.mesh.psize =  np.multiply(poll.mesh.psize, 2, dtype=poll.dtype.dtype
      elif poll.success == SUCCESS_TYPES.US:
        poll.mesh.refineDeltaFrameSize()
      
    elif isinstance(B, BarrierMO):
      xpost: List[CandidatePoint] = []
      for i in range(len(xt)):
        xpost.append(xt[i])
      updated, _, _ = B.updateWithPoints(eval_point_list=xpost, keep_all_points=False)
      if not updated:
        new_mesh = None
        if B._currentIncumbentInf:
          B._currentIncumbentInf.mesh.refineDeltaFrameSize()
          new_mesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf.mesh) if B._currentIncumbentInf else None
          B.updateCurrentIncumbents()
        if B._currentIncumbentFeas:
          B._currentIncumbentFeas.mesh.refineDeltaFrameSize()
          new_mesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf.mesh) if B._currentIncumbentInf else None
          B.updateCurrentIncumbents()

        
        if new_mesh:
          poll.mesh = new_mesh
        else:
          poll.mesh.refineDeltaFrameSize()
          
      else:
        poll.mesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf.mesh) if B._currentIncumbentInf else poll.mesh
        poll.xmin = copy.deepcopy(B._currentIncumbentFeas) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf) if B._currentIncumbentInf else poll.xmin
      for i in range(len(xpost)):
        post.poll_dirs.append(xpost[i])
      
      post.xmin = B._currentIncumbentFeas if B._currentIncumbentFeas  else B._currentIncumbentInf if  B._currentIncumbentInf else poll.xmin
    poll.mesh.update()
    if log:
        log.log_msg(msg=post.__str__(), msg_type=MSG_TYPE.INFO)
    if options.display:
      print(post)
    
    lambda_k = poll.constraints_RP.LAMBDA
    rho_k = poll.constraints_RP.RHO
    
    if options.save_results:
      post.nd_points = []
      post.output_results(out, all_res=False)
      if param.isPareto:
        for i in range(len(B.getAllPoints())):
          post.nd_points.append(B.getAllPoints()[i])
        post.output_nd_results(out_p)

    failure_check = iteration > 0 and poll.Failure_stop is not None and poll.Failure_stop and (poll.success == SUCCESS_TYPES.US or go_to_search)
    
    if (failure_check or poll.bb_eval >= options.budget) or (all(abs(poll.mesh.getDeltaFrameSize().coordinates[pp]) < options.tol for pp in range(poll._n)) or poll.bb_eval >= options.budget or poll.terminate):
      log.log_msg("\n--------------- Termination of the poll step  ---------------", MSG_TYPE.INFO)
      if all(abs(poll.mesh.getDeltaFrameSize().coordinates[pp]) < options.tol for pp in range(poll._n)):
        log.log_msg("Termination criterion hit: the mesh size is below the minimum threshold defined.", MSG_TYPE.INFO)
      if (poll.bb_eval >= options.budget or poll.terminate):
        log.log_msg("Termination criterion hit: evaluation budget is exhausted.", MSG_TYPE.INFO)
      if (failure_check):
        log.log_msg("Termination criterion hit (optional): failed to find a successful point in iteration # {iteration}.", MSG_TYPE.INFO)
      log.log_msg("---------------------------------------------------------------\n", MSG_TYPE.INFO)
      break
    iteration += 1
    

  toc = time.perf_counter()
  if isinstance(B, BarrierMO):
    rp: Optional[CandidatePoint] = None
    if param.ref_point:
      rp =  Point()
      rp.coordinates = param.ref_point
    perf_m = Metrics(nd_solutions=B.getAllPoints(), nobj=B._nobj, ref_point=rp)
    HV = perf_m.hypervolume()

  """ If benchmarking, then populate the results in the benchmarking output report """
  if importlib.util.find_spec('BMDFO') and len(args) > 1 and isinstance(args[1], toy.Run):
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

  elif importlib.util.find_spec('BMDFO') and len(args) > 1 and not isinstance(args[1], toy.Run):
    temp = " in the internal BM suite."
    if log:
      log.log_msg(msg="Could not find " + args[1] + temp, msg_type=MSG_TYPE.ERROR)
    raise IOError("Could not find " + args[1] + temp)

  

  if options.display:
    print(" end of orthogonal MADS ")
    if log:
      log.log_msg(msg=" end of orthogonal MADS ", msg_type=MSG_TYPE.INFO)
    print(" Final objective value: " + str(poll.xmin.f) + ", hmin= " + str(poll.xmin.h))
    if log:
      log.log_msg(msg=" Final objective value: " + str(poll.xmin.f) + ", hmin= " + str(poll.xmin.h), msg_type=MSG_TYPE.INFO)
    if log and len(args)>1 and isinstance(args[1], str):
      log.log_msg(msg=" end of orthogonal MADS running" + args[1] + " in the internal BM suite.", msg_type=MSG_TYPE.INFO)
    

  if options.save_coordinates:
    post.output_coordinates(out)
  
  if log:
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
    log.log_msg(msg=f" psize_success = {poll.xmin.mesh.getDeltaFrameSize().coordinates}", msg_type=MSG_TYPE.INFO)
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
    print(" psize = " + str(poll.mesh.getDeltaFrameSize().coordinates))
    print(" psize_success = " + str(poll.xmin.mesh.getDeltaFrameSize().coordinates))
    # print(" psize_max = " + poll.mesh.psize_max)
    
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
                "psuccess": poll.xmin.mesh.getDeltaFrameSize().coordinates,
                # "pmax": poll.mesh.psize_max,
                "msize": poll.mesh.getdeltaMeshSize().coordinates,
                "HV": HV if param.isPareto else "NA"}

  return output, poll

def rosen(x, p):
  x = np.asarray(x)
  y = [np.sum(p[0] * (x[1:] - x[:-1] ** p[1]) ** p[1] + (1 - x[:-1]) ** p[1],
        axis=0), [0]]
  return y

def alpine(x):
  y = [abs(x[0]*np.sin(x[0])+0.1*x[0])+abs(x[1]*np.sin(x[1])+0.1*x[1]), [0]]
  return y

def ackley3(x):
  return [-200*np.exp(-0.2*np.sqrt(x[0]**2+x[1]**2))+5*np.exp(np.cos(3*x[0])+np.sin(3*x[1])), [0]]

def egg_holder(individual):
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

