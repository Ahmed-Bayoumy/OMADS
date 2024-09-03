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

import copy
import importlib
import json
from multiprocessing import freeze_support
import os
import sys
import OMADS.POLL as PS
import OMADS.SEARCH as SS
from typing import List, Dict, Any
import numpy as np
if importlib.util.find_spec('BMDFO'):
  from BMDFO import toy
import time
from .Point import Point
from .CandidatePoint import CandidatePoint
from ._common import logger, MSG_TYPE
from ._globals import DESIGN_STATUS, SAMPLING_METHOD, SUCCESS_TYPES
from .Barriers import Barrier, BarrierMO
from .Parameters import Parameters
from .Options import Options
from .PostProcess import Output, PostMADS
from .Metrics import Metrics
from .Optimizer import ConstraintsRelaxationParameters
np.set_printoptions(legacy='1.21')


def search_step(iteration: int, search: SS.efficient_exploration = None, B: SS.auto = None, LAMBDA_k: float=None, RHO_k: float=None, search_VN: SS.VNS = None, post: PS.PostMADS=None, out: PS.Output=None, options: PS.Options=None, xmin: SS.CandidatePoint=None, peval: int=0, HT: Any=None, log:logger = None, outP: PS.Output=None):
  """ Reset success boolean """
  search.success = SUCCESS_TYPES.US
  tic = time.perf_counter()
  search.log = log
  search.xmin = xmin
  search.mesh.update()
  search.LAMBDA = LAMBDA_k
  search.RHO = RHO_k
  if HT is not None:
    search.hashtable = HT
  if B is not None:
    if isinstance(B, Barrier):
      B.insert(search.xmin)
      if B._filter is not None:
        B.select_poll_center()
        B.update_and_reset_success()
    elif isinstance(B, BarrierMO) and iteration == 1:
        B.init(evalPointList=[xmin])
      
  
  # search.hmax = B._h_max
  
  if isinstance(B, Barrier):
    search.hmax = B._h_max
    # TODO: Check whether the commented code below is needed
    # if xmin.status == DESIGN_STATUS.FEASIBLE:
    #   B.insert_feasible(search.xmin)
    # elif xmin.status == DESIGN_STATUS.INFEASIBLE:
    #   B.insert_infeasible(search.xmin)
    # else:
    #   B.insert(search.xmin)
  elif isinstance(B, BarrierMO):
    search.hmax = B._hMax
  """ Create the set of poll directions """
  if search.type == SS.SEARCH_TYPE.VNS.name and search_VN is not None:
    search_VN.active_barrier = B
    search._candidate_points_set = search_VN.run()
    if search_VN.stop:
      print("Reached maximum number of VNS iterations!")
      return search, B, post, out, search.LAMBDA, search.RHO, search.xmin, peval, outP
    search.map_samples_from_coords_to_points(samples=search._candidate_points_set)
  else:
    vvp = vvs = []
    bestFeasible: CandidatePoint = B._currentIncumbentFeas if isinstance(B, BarrierMO) else B._best_feasible
    bestInf: CandidatePoint = B._currentIncumbentInf if isinstance(B, BarrierMO) else B.get_best_infeasible()
    if bestFeasible is not None and bestFeasible.evaluated:
      search.xmin = bestFeasible
      vvp, _ = search.generate_sample_points(int(((search.dim+1)/2)*((search.dim+2)/2)) if search.ns is None else search.ns)
    if bestInf is not None and bestInf.evaluated:
    # if B._filter is not None and B.get_best_infeasible().evaluated:
      xmin_bup = search.xmin
      Prim_samples = search._candidate_points_set
      search.xmin = bestInf#B.get_best_infeasible()
      vvs, _ = search.generate_sample_points(int(((search.dim+1)/2)*((search.dim+2)/2)) if search.ns is None else search.ns)
      search._candidate_points_set += Prim_samples
      search.xmin = xmin_bup
    
    if isinstance(vvs, list) and len(vvs) > 0:
      vv = vvp + vvs
    else:
      vv = vvp


  """ Save current poll directions and incumbent solution
    so they can be saved later in the post dir """
  search.omit_duplicates()
  if options.save_coordinates:
    post.coords.append(search._candidate_points_set)
    post.x_incumbent.append(search.xmin)
  """ Reset success boolean """
  search.success = SUCCESS_TYPES.US
  """ Reset the BB output """
  search.bb_output = []
  xt = []
  """ Serial evaluation for points in the poll set """
  if search_VN is not None:
    search.lb = search_VN.params.lb
    search.ub = search_VN.params.ub
  search.bb_handle.xmin = xmin
  search.constraints_RP.LAMBDA = xmin.LAMBDA
  search.constraints_RP.RHO = xmin.RHO
  search.constraints_RP.constraints_type = xmin.constraints_type
  search.constraints_RP.hmax = xmin.hmax
  if not options.parallel_mode:
    xt, post, peval = search.bb_handle.run_callable_serial_local(iter=iteration, peval=peval, eval_set=search._candidate_points_set, options=options, post=post, psize=search.mesh.getDeltaFrameSize().coordinates, stepName=f'Search: {search.type}', mesh=search.mesh, constraintsRelaxation=search.constraints_RP.__dict__, budget=options.budget)

  else:
    search._point_index = -1
    """ Parallel evaluation for points in the samples set """
    search.bb_eval, xt, post, peval = search.bb_handle.run_callable_parallel_local(iter=iteration, peval=peval, njobs=options.np, eval_set=search._candidate_points_set, options=options, post=post, mesh=search.mesh, stepName=f'Search: {search.type}', psize=search.mesh.getDeltaFrameSize().coordinates, constraintsRelaxation=search.constraints_RP.__dict__, budget=options.budget)
    
  if search.bb_handle.constraintsRelaxation:
    temp:ConstraintsRelaxationParameters = ConstraintsRelaxationParameters(**search.bb_handle.constraintsRelaxation)
    for i in range(len(temp.LAMBDA)):
      search.constraints_RP.LAMBDA[i] = temp.LAMBDA[i]
    search.constraints_RP.RHO = temp.RHO
    search.constraints_RP.constraints_type = temp.constraints_type
    search.constraints_RP.hmax = temp.hmax
    # if options.store_cache:
    #   for xi in xt:
    #     search.hashtable.hash_id = xi
    #     if not search.hashtable._isPareto:
    #       search.hashtable.add_to_best_cache(xi)
  LAMBDA_k = search.bb_handle.constraintsRelaxation["LAMBDA"]
  RHO_k = search.bb_handle.constraintsRelaxation["RHO"]
  search.postprocess_evaluated_candidates(xt)

  
    
  if isinstance(B, Barrier):
    xpost: List[CandidatePoint] = search.master_updates(xt, peval, save_all_best=options.save_all_best, save_all=options.save_results)
    if options.save_results:
      for i in range(len(xpost)):
        post.poll_dirs.append(xpost[i])
    for xv in xt:
      if xv.evaluated:
        B.insert(xv)

    """ Update the xmin in post"""
    post.xmin = copy.deepcopy(search.xmin)


    if iteration == 1:
      search.vicinity_ratio = np.ones((len(search.xmin.coordinates),1))
    
    """ Updates """
    
    if search.success == SUCCESS_TYPES.FS:
      dir: Point = Point(search.mesh._n)
      dir.coordinates = search.xmin.direction.coordinates
      # search.mesh.psize = np.multiply(search.mesh.get, 2, dtype=search.dtype.dtype)
      search.mesh.enlargeDeltaFrameSize(direction=dir)
      if search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
        search.update_local_region(region="expand")
    elif search.success == SUCCESS_TYPES.US:
      # search.mesh.psize = np.divide(search.mesh.psize, 2, dtype=search.dtype.dtype)
      search.mesh.refineDeltaFrameSize()
      if search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
        search.update_local_region(region="contract")
  elif isinstance(B, BarrierMO):
    xpost: List[CandidatePoint] = []
    for i in range(len(xt)):
      xpost.append(xt[i])
    updated, updatedF, updatedInf = B.updateWithPoints(evalPointList=xpost, evalType=None, keepAllPoints=False, updateInfeasibleIncumbentAndHmax=True)
    if not updated:
      newMesh = None
      if B._currentIncumbentInf:
        B._currentIncumbentInf.mesh.refineDeltaFrameSize()
        newMesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf.mesh) if B._currentIncumbentInf else None
        B.updateCurrentIncumbents()
        if search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
          search.update_local_region(region="contract")
      if B._currentIncumbentFeas:
        B._currentIncumbentFeas.mesh.refineDeltaFrameSize()
        newMesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf.mesh) if B._currentIncumbentInf else None
        B.updateCurrentIncumbents()
        if search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
          search.update_local_region(region="contract")
      
      if iteration == 1:
        search.vicinity_ratio = np.ones((len(search.xmin.coordinates),1))
      if newMesh:
        search.mesh = newMesh
      else:
        search.mesh.refineDeltaFrameSize()
        if search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
          search.update_local_region(region="contract")
    else:
      search.mesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if updatedF else copy.deepcopy(B._currentIncumbentInf.mesh) if updatedInf else search.mesh
      search.xmin = copy.deepcopy(B._currentIncumbentFeas) if updatedF else copy.deepcopy(B._currentIncumbentInf) if updatedInf else search.xmin
      if search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
        search.update_local_region(region="expand")
    
    for i in range(len(xpost)):
      post.poll_dirs.append(xpost[i])
    search.hashtable.best_hash_ID = []
    search.hashtable.add_to_best_cache(B.getAllPoints())
    post.xmin = B._currentIncumbentFeas if updatedF  else B._currentIncumbentInf if  updatedInf else search.xmin
    
  search.mesh.update()
  
  
  if options.display:
    if log is not None:
      log.log_msg(msg=post.__str__(), msg_type=MSG_TYPE.INFO)
    print(post)

  toc = time.perf_counter()
  if log is not None:
    log.log_msg(f" Run completed in {toc - tic:.4f} seconds", MSG_TYPE.INFO)
    log.log_msg(msg=f" Success status: {search.success}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=post.__str__(), msg_type=MSG_TYPE.INFO)
    # log.log_msg(f" Random numbers generator's seed {options.seed}", MSG_TYPE.INFO)
    # log.log_msg(" xmin = " + str(search.xmin), MSG_TYPE.INFO)
    # log.log_msg(" hmin = " + str(search.xmin.h), MSG_TYPE.INFO)
    # log.log_msg(" fmin = " + str(search.xmin.f), MSG_TYPE.INFO)
    # log.log_msg(" #bb_eval = " + str(search.bb_handle.bb_eval), MSG_TYPE.INFO)
    # log.log_msg(" nb_success = " + str(search.nb_success), MSG_TYPE.INFO)

  # Failure_check = iteration > 0 and search.Failure_stop is not None and search.Failure_stop and not search.success
  # if (Failure_check) or (abs(search.mesh.msize) < options.tol or search.bb_eval >= options.budget or search.terminate):
  #   break
  # iteration += 1
  return search, B, post, out, search.LAMBDA, search.RHO, search.xmin, peval, outP

def poll_step(iteration: int, poll: PS.Dirs2n = None, B: SS.auto = None, LAMBDA_k: float=None, RHO_k: float=None, param: PS.Parameters=None, post: PS.PostMADS=None, xmin: PS.CandidatePoint=None, out: PS.Output=None, options: PS.Options=None, peval: int = 0, HT: Any = None, log:logger = None, outP: PS.Output=None):
  tic = time.perf_counter()
  poll.xmin = xmin
  poll.mesh.update()
  """ Create the set of poll directions """
  hhm = poll.create_housholder(options.rich_direction, domain=xmin.var_type)
  poll.lb = param.lb
  poll.ub = param.ub
  poll.xmin = copy.deepcopy(xmin)
  xmin.mesh = copy.deepcopy(poll.mesh)
  if HT is not None:
    poll.hashtable = HT
  if B is not None:
    if isinstance(B, Barrier):
      B.insert(xmin)
      if B._filter is not None:
        B.select_poll_center()
        B.update_and_reset_success()
        
    elif isinstance(B, BarrierMO) and iteration == 1:
      B.init(evalPointList=[xmin])
      
  if isinstance(B, Barrier):
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
  elif isinstance(B, BarrierMO):
    poll.hmax = B._hMax
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
      # del poll.poll_set
      poll.x_sc = B._currentIncumbentInf
      poll.create_poll_set(hhm=hhm,
              ub=param.ub,
              lb=param.lb, it=iteration, var_type=B._currentIncumbentInf.var_type, var_sets=B._currentIncumbentInf.sets, var_link = B._currentIncumbentInf.var_link, c_types=param.constraints_type, is_prim=False)
    elif poll.xmin.status == DESIGN_STATUS.INFEASIBLE:
      poll.create_poll_set(hhm=hhm,
              ub=param.ub,
              lb=param.lb, it=iteration, var_type=poll.xmin.var_type, var_sets=poll.xmin.sets, var_link = poll.xmin.var_link, c_types=param.constraints_type, is_prim=False)
    
  
  poll.LAMBDA = LAMBDA_k
  poll.RHO = RHO_k

  """ Save current poll directions and incumbent solution
    so they can be saved later in the post dir """
  poll.omit_duplicates()
  if options.save_coordinates:
    post.coords.append(poll.poll_set)
    post.x_incumbent.append(poll.xmin)
  """ Reset success boolean """
  poll.success = SUCCESS_TYPES.US
  """ Reset the BB output """
  poll.bb_output = []
  xt = []
  poll.bb_handle.xmin = xmin
  """ Serial evaluation for points in the poll set """
  poll.constraints_RP.LAMBDA = xmin.LAMBDA
  poll.constraints_RP.RHO = xmin.RHO
  poll.constraints_RP.constraints_type = xmin.constraints_type
  poll.constraints_RP.hmax = xmin.hmax
  if not options.parallel_mode:
    xt, post, peval = poll.bb_handle.run_callable_serial_local(iter=iteration, peval=peval, eval_set=poll._candidate_points_set, options=options, post=post, psize=poll.mesh.getDeltaFrameSize().coordinates, stepName=f'Poll Step', mesh=poll.mesh, constraintsRelaxation=poll.constraints_RP.__dict__, budget=options.budget)

  else:
    poll.point_index = -1
    """ Parallel evaluation for points in the samples set """
    poll.bb_eval, xt, post, peval = poll.bb_handle.run_callable_parallel_local(iter=iteration, peval=peval, njobs=options.np, eval_set=poll._candidate_points_set, options=options, post=post, mesh=poll.mesh, stepName=f'Poll Step', psize=poll.mesh.getDeltaFrameSize().coordinates, constraintsRelaxation=poll.constraints_RP.__dict__, budget=options.budget)
    
  if poll.bb_handle.constraintsRelaxation:
    temp:ConstraintsRelaxationParameters = ConstraintsRelaxationParameters(**poll.bb_handle.constraintsRelaxation)
    for i in range(len(temp.LAMBDA)):
      poll.constraints_RP.LAMBDA[i] = temp.LAMBDA[i]
    poll.constraints_RP.RHO = temp.RHO
    poll.constraints_RP.constraints_type = temp.constraints_type
    poll.constraints_RP.hmax = temp.hmax
    # if options.store_cache:
    #   for xi in xt:
    #     poll.hashtable.hash_id = xi
    #     if not poll.hashtable._isPareto:
    #       poll.hashtable.add_to_best_cache(xi)
  LAMBDA_k = poll.bb_handle.constraintsRelaxation["LAMBDA"]
  RHO_k = poll.bb_handle.constraintsRelaxation["RHO"]
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
      # if pev != poll.poll_dirs and not poll.success:
      #   poll.seed += 1
      goToSearch: bool = (pev == 0 and poll.Failure_stop is not None and poll.Failure_stop)
      
      dir: Point = Point(poll._n)
      dir.coordinates = poll.xmin.direction.coordinates if poll.xmin.direction is not None else [0]*poll._n
      if poll.success == SUCCESS_TYPES.FS and not goToSearch:
        poll.mesh.enlargeDeltaFrameSize(direction=dir) # poll.mesh.psize =  np.multiply(poll.mesh.psize, 2, dtype=poll.dtype.dtype
      elif poll.success == SUCCESS_TYPES.US:
        poll.mesh.refineDeltaFrameSize()
        # poll.mesh.psize = np.divide(poll.mesh.psize, 2, dtype=poll.dtype.dtype)
      
  elif isinstance(B, BarrierMO):
    xpost: List[CandidatePoint] = []
    for i in range(len(xt)):
      xpost.append(xt[i])
    updated, _, _ = B.updateWithPoints(evalPointList=xpost, evalType=None, keepAllPoints=False, updateInfeasibleIncumbentAndHmax=True)
    if not updated:
      newMesh = None
      if B._currentIncumbentInf:
        B._currentIncumbentInf.mesh.refineDeltaFrameSize()
        newMesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf.mesh) if B._currentIncumbentInf else None
        B.updateCurrentIncumbents()
      if B._currentIncumbentFeas:
        B._currentIncumbentFeas.mesh.refineDeltaFrameSize()
        newMesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf.mesh) if B._currentIncumbentInf else None
        B.updateCurrentIncumbents()

      
      if newMesh:
        poll.mesh = newMesh
      else:
        poll.mesh.refineDeltaFrameSize()
        
    else:
      poll.mesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf.mesh) if B._currentIncumbentInf else poll.mesh
      poll.xmin = copy.deepcopy(B._currentIncumbentFeas) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf) if B._currentIncumbentInf else poll.xmin
    for i in range(len(xpost)):
      post.poll_dirs.append(xpost[i])
    
    post.xmin = B._currentIncumbentFeas if B._currentIncumbentFeas  else B._currentIncumbentInf if  B._currentIncumbentInf else poll.xmin
  poll.mesh.update()
  if log is not None:
      log.log_msg(msg=post.__str__(), msg_type=MSG_TYPE.INFO)
  if options.display:
    print(post)
  
  LAMBDA_k = poll.LAMBDA
  RHO_k = poll.RHO

  toc = time.perf_counter()

  if log is not None:
    # log.log_msg(msg=" ---Run Summary--- ", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" Run completed in {toc - tic:.4f} seconds", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" Success status: {poll.success}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=post.__str__(), msg_type=MSG_TYPE.INFO)
    # log.log_msg(msg=f" Random numbers generator's seed {options.seed}", msg_type=MSG_TYPE.INFO)
    # log.log_msg(msg=f" xmin = {poll.xmin.__str__()} ", msg_type=MSG_TYPE.INFO)
    # log.log_msg(msg=f" hmin = {poll.xmin.h} ", msg_type=MSG_TYPE.INFO)
    # log.log_msg(msg=f" fmin {poll.xmin.fobj}", msg_type=MSG_TYPE.INFO)
    # log.log_msg(msg=f" #bb_eval =  {poll.bb_eval} ", msg_type=MSG_TYPE.INFO)
    # log.log_msg(msg=f" #iteration =  {iteration} ", msg_type=MSG_TYPE.INFO)
    # log.log_msg(msg=f"  nb_success = {poll.nb_success} ", msg_type=MSG_TYPE.INFO)
    # log.log_msg(msg=f" psize = {poll.mesh.psize} ", msg_type=MSG_TYPE.INFO)
    # log.log_msg(msg=f" psize_success = {poll.mesh.psize_success} ", msg_type=MSG_TYPE.INFO)
    # log.log_msg(msg=f" psize_max = {poll.mesh.psize_max} ", msg_type=MSG_TYPE.INFO)
  
  return poll, B, post, out, poll.xmin.LAMBDA, poll.xmin.RHO, poll.xmin, peval, outP

def main(*args) -> Dict[str, Any]:
  """ Otho-MADS main algorithm """
  # COMPLETED: add more checks for more defensive code

  """ Parse the parameters files """
  if type(args[0]) is dict:
    data = args[0]
  elif isinstance(args[0], str):
    if os.path.exists(os.path.abspath(args[0])):
      _, file_extension = os.path.splitext(args[0])
      if file_extension == ".json":
        try:
          with open(args[0]) as file:
            data = json.load(file)
        except ValueError:
          raise IOError('invalid json file: ' + args[0])
      else:
        raise IOError(f"The input file {args[0]} is not a JSON dictionary. "
                f"Currently, OMADS supports JSON files solely!")
    else:
      raise IOError(f"Couldn't find {args[0]} file!")
  else:
    raise IOError("The first input argument couldn't be recognized. "
            "It should be either a dictionary object or a JSON file that holds "
            "the required input parameters.")
  
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
  iteration: int
  xmin: CandidatePoint
  options: Options
  param: Parameters 
  post: PostMADS 
  out: Output 
  B: Barrier
  poll: PS.Dirs2n
  search: SS.efficient_exploration
  log.log_msg(msg="Preprocess the search step...", msg_type=PS.MSG_TYPE.INFO)
  _, _, search, _, _, _, _, _, _ = SS.PreExploration(data).initialize_from_dict(log=log)
  log.log_msg(msg="Preprocess the MADS algorithim...", msg_type=PS.MSG_TYPE.INFO)
  iteration, xmin, poll, options, param, post, out, B, outP = PS.PrePoll(data).initialize_from_dict(log=log, xs=search.xmin)
  out.stepName = "Poll"
  post.step_name = [f'Search: {search.type}']

  HT = poll.hashtable
  
  # if MADS_LINK.REPLACE is not None and not MADS_LINK.REPLACE:
  #   out.replace = False

  """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))

  """ Start the count down for calculating the runtime indicator """
  tic = PS.time.perf_counter()
  peval = 0
  LAMBDA_k = xmin.LAMBDA
  RHO_k = xmin.RHO

  if search.type == SS.SEARCH_TYPE.VNS.name:
    search_VN = SS.VNS(active_barrier=B, params=param)
    search_VN._ns_dist = [int(((search.dim+1)/2)*((search.dim+2)/2)/(len(search_VN._dist))) if search.ns is None else search.ns] * len(search_VN._dist)
    search.ns = sum(search_VN._ns_dist)
  else:
    search_VN = None
  
  search.lb = param.lb
  search.ub = param.ub

  while True:
    """ Run search step (Optional) """
    # TODO: This rule cannot be generalized -- needs further invistigation
    # if poll.dim > 10 and poll.mesh.psize >= 1E-4:
    #   canSearch = False
    # else:
    canSearch = True

    if canSearch and (poll.success == SUCCESS_TYPES.US or iteration == 1):
      log.log_msg(f"------- Iteration # {iteration}: Run the search step -------", MSG_TYPE.INFO)
      search.iter = iteration
      search, B, post, out, LAMBDA_k, RHO_k, xmin, peval, outP = search_step(search=search, B=B, LAMBDA_k=LAMBDA_k, RHO_k=RHO_k, iteration=iteration , search_VN=search_VN, post=post, out=out, options=options, xmin=xmin, peval=peval, HT=HT, log=log, outP=outP)
      HT = search.hashtable
    """ Run the poll step (Mandatory step) """
    log.log_msg(f"------- Iteration # {iteration}: Run the poll step -------", MSG_TYPE.INFO)
    poll, B, post, out, LAMBDA_k, RHO_k, xmin, peval, outP = poll_step(iteration=iteration, poll=poll, B=B, LAMBDA_k=LAMBDA_k, RHO_k=RHO_k, param=param, post=post, xmin=xmin, out=out, options=options, peval=peval, HT=HT, log=log, outP=outP)
    HT = poll.hashtable
    xmin = poll.xmin
    search.mesh = copy.deepcopy(poll.mesh)
    search.psize = copy.deepcopy(poll.psize)
    """ Check stopping criteria"""
    pt = (all(abs(poll.mesh.getDeltaFrameSize().coordinates[pp]) < options.tol for pp in range(poll._n)))
    st = (all(abs(search.mesh.getdeltaMeshSize().coordinates[pp]) < options.tol  for pp in range(search.mesh._n)))
    if options.save_results:
      post.output_results(out, False)
      if param.isPareto:
        post.nd_points = []
        for i in range(len(B.getAllPoints())):
          post.nd_points.append(B.getAllPoints()[i])
        post.output_nd_results(outP)
    if (pt or st or search.bb_eval + poll.bb_eval >= options.budget):
      log.log_msg(f"\n--------------- Termination of MADS  ---------------", MSG_TYPE.INFO)
      if pt:
        log.log_msg(f"Termination criterion hit: the poll size is below the minimum threshold defined.", MSG_TYPE.INFO)
      if st:
        log.log_msg(f"Termination criterion hit: the mesh size is below the minimum threshold defined.", MSG_TYPE.INFO)
      if (search.bb_eval + poll.bb_eval >= options.budget):
        log.log_msg(f"Termination criterion hit: Evaluation budget is exhausted.", MSG_TYPE.INFO)
      log.log_msg(f"----------------------------------------------------\n", MSG_TYPE.INFO)
      break
    iteration += 1
    

  toc = PS.time.perf_counter()
  if isinstance(B, BarrierMO):
    perfM = Metrics(ND_solutions=B.getAllPoints(), nobj=B._nobj)
    HV = perfM.hypervolume()

  """ If benchmarking, then populate the results in the benchmarking output report """
  if importlib.util.find_spec('BMDFO') and len(args) > 1 and isinstance(args[1], PS.toy.Run):
    b: PS.toy.Run = args[1]
    if b.test_suite == "uncon":
      ncon = 0
    else:
      ncon = len(xmin.c_ineq)
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
    raise IOError("Could not find " + args[1] + " in the internal BM suite.")

  # if options.save_results:
  #   post.output_results(out)
  
  out_step: Any = None
  if poll.xmin < search.xmin:
    out_step = poll
  elif search.xmin < poll.xmin:
    out_step = search
  else:
    out_step = poll
  
  if out_step is None:
    out_step = poll
  

  if options.display:
    print(" end of orthogonal MADS ")
    print(" Final objective value: " + str(out_step.xmin.f) + ", hmin= " + str(out_step.xmin.h))

  if options.save_coordinates:
    post.output_coordinates(out)
  
  if log is not None:
    log.log_msg(msg=" --- MADS Run Summary--- ", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" Run completed in {toc - tic:.4f} seconds", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" # of successful search steps = {search.n_successes}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" # of successful poll steps = {poll.n_successes}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" Run completed in {toc - tic:.4f} seconds", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" Random numbers generator's seed {options.seed}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" xmin = {poll.xmin.__str__()} ", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" hmin = {poll.xmin.h} ", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" fmin {poll.xmin.fobj}", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" Search step # BB evals =  {search.bb_eval} ", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" Poll step # BB evals =  {poll.bb_eval} ", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" Total # BB evals =  {poll.bb_eval + search.bb_eval} ", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" #iterations =  {iteration} ", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" psize = {poll.mesh.getDeltaFrameSize().coordinates} ", msg_type=MSG_TYPE.INFO)
    log.log_msg(msg=f" psize_success = {poll.xmin.mesh.getDeltaFrameSize().coordinates}", msg_type=MSG_TYPE.INFO)
    if isinstance(B, BarrierMO):
      log.log_msg(msg=f" Hypervolume metric = {HV}", msg_type=MSG_TYPE.INFO)
    # log.log_msg(msg=f" psize_max = {poll.mesh.psize_max} ", msg_type=MSG_TYPE.INFO)
  if options.display:
    print("\n ---MADS Run Summary---")
    print(f" Run completed in {toc - tic:.4f} seconds")
    print(f" Random numbers generator's seed {options.seed}")
    print(" xmin = " + str(out_step.xmin))
    print(" hmin = " + str(out_step.xmin.h))
    print(" fmin = " + str(out_step.xmin.f))
    print(" #bb_eval = " + str(out_step.bb_eval))
    print(" #iteration = " + str(iteration))
    print(" nb_success = " + str(poll.nb_success + search.nb_success))
    print(" psize = " + str(poll.mesh.getDeltaFrameSize().coordinates))
    print(" psize_success = " + str(poll.xmin.mesh.getDeltaFrameSize().coordinates))
    # print(" psize_max = " + str(poll.mesh.psize_max))
    
  xmin = out_step.xmin
  """ Evaluation of the blackbox; get output responses """
  if xmin.sets is not None and isinstance(xmin.sets,dict):
    p: List[Any] = []
    for i in range(len(xmin.var_type)):
      if (xmin.var_type[i] == PS.VAR_TYPE.DISCRETE or xmin.var_type[i] == PS.VAR_TYPE.CATEGORICAL) and xmin.var_link[i] is not None:
        p.append(xmin.sets[xmin.var_link[i]][int(xmin.coordinates[i])])
      else:
        p.append(xmin.coordinates[i])
  else:
    p = xmin.coordinates
  output: Dict[str, Any] = {"xmin": p,
                "fmin": out_step.xmin.f,
                "hmin": out_step.xmin.h,
                "nbb_evals" : out_step.bb_eval,
                "niterations" : iteration,
                "nb_success": poll.nb_success + search.nb_success,
                "psize": poll.mesh.getDeltaFrameSize().coordinates,
                "psuccess": poll.xmin.mesh.getDeltaFrameSize().coordinates,
                # "pmax": poll.mesh.psize_max,
                "msize": out_step.mesh.getdeltaMeshSize().coordinates}

  return output, out_step


def rosen(x, *argv):
  x = np.asarray(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
        axis=0), [0]]
  return y


def test_omads_callable_quick():
  eval = {"blackbox": rosen}
  param = {"baseline": [-2.0, -2.0],
       "lb": [-5, -5],
       "ub": [10, 10],
       "var_names": ["x1", "x2"],
       "scaling": 15.0,
       "post_dir": "./post",
       "Failure_stop": True}
  sampling = {
    "method": 2,
    "ns": 5,
    "visualize": False
  }
  options = {"seed": 0, "budget": 100000, "tol": 1e-12, "display": True, "check_cache": True, "store_cache": True, "rich_direction": True, "psize_init": 1., "precision": "high"}

  data = {"evaluator": eval, "param": param, "options": options, "sampling": sampling}

  out: Dict = main(data)
  print(out)


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