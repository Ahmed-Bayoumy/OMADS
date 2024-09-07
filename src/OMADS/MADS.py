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
from dataclasses import dataclass
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
from .Evaluator import Evaluator
np.set_printoptions(legacy='1.21')

@dataclass
class MADS:
  search: SS.efficient_exploration = None
  search_VN: SS.VNS = None
  poll: PS.Dirs2n = None
  param: Parameters = None
  evaluator: Evaluator = None
  post: PostMADS = None
  out: Output = None
  outP: Output = None
  options: Options = None
  data: dict = None
  xmin: CandidatePoint = None
  iteration: int = 0
  peval: int = 0
  HT: Any = None
  log: logger = None
  B: Any = None
  LAMBDA_k: float = 0.
  RHO_k: float = 0.
  tic: Any = None
  toc: Any = None
  active_barrier: Barrier = None

  def __init__(self, data: dict):
    """ Initialize the log file """
    self.log = logger()
    if not os.path.exists(data["param"]["post_dir"]):
      try:
        os.mkdir(data["param"]["post_dir"])
      except:
        os.makedirs(data["param"]["post_dir"], exist_ok=True)

    self.log.initialize(data["param"]["post_dir"] + "/OMADS.log")
    self.log.log_msg(msg="Preprocess the MADS algorithim...", msg_type=PS.MSG_TYPE.INFO)
    self.log.log_msg(msg="Preprocess the search step...", msg_type=PS.MSG_TYPE.INFO)
    _, _, self.search, _, _, _, _, _, _ = SS.PreExploration(data).initialize_from_dict(log=self.log)
    self.log.log_msg(msg="Preprocess the POLL step...", msg_type=PS.MSG_TYPE.INFO)
    self.iteration, self.xmin, self.poll, self.options, self.param, self.post, self.out, self.B, self.outP = PS.PrePoll(data).initialize_from_dict(log=self.log, xs=self.search.xmin)
    self.out.stepName = "Poll"
    self.post.step_name = [f'Search: {self.search.type}']

    self.HT = copy.deepcopy(self.poll.hashtable)

  def search_step(self, xmin: SS.CandidatePoint=None):
    """ Reset success boolean """
    self.search.success = SUCCESS_TYPES.US
    tic = time.perf_counter()
    self.search.log = self.log
    self.search.xmin = xmin
    self.search.mesh.update()
    self.search.LAMBDA = self.LAMBDA_k
    self.search.RHO = self.RHO_k
    B = self.active_barrier
    if self.HT is not None:
      self.search.hashtable = self.HT
    if B is not None:
      if isinstance(B, Barrier):
        B.insert(self.search.xmin)
        if B._filter is not None:
          B.select_poll_center()
          B.update_and_reset_success()
      elif isinstance(B, BarrierMO) and self.iteration == 1:
          B.init(evalPointList=[xmin])
        
    
    # search.hmax = B._h_max
    
    if isinstance(B, Barrier):
      self.search.hmax = B._h_max
      # TODO: Check whether the commented code below is needed
      # if xmin.status == DESIGN_STATUS.FEASIBLE:
      #   B.insert_feasible(search.xmin)
      # elif xmin.status == DESIGN_STATUS.INFEASIBLE:
      #   B.insert_infeasible(search.xmin)
      # else:
      #   B.insert(search.xmin)
    elif isinstance(B, BarrierMO):
      self.search.hmax = B._hMax
    """ Create the set of poll directions """
    if self.search.type == SS.SEARCH_TYPE.VNS.name and self.search_VN is not None:
      self.search_VN.active_barrier = B
      self.search._candidate_points_set = self.search_VN.run()
      if self.search_VN.stop:
        print("Reached maximum number of VNS iterations!")
        self.HT = self.search.hashtable
        self.RHO_k = self.search.RHO
        self.active_barrier = B
        self.LAMBDA_k = self.search.LAMBDA
        return self.search.xmin
      self.search.map_samples_from_coords_to_points(samples=self.search._candidate_points_set)
    else:
      vvp = vvs = []
      bestFeasible: CandidatePoint = B._currentIncumbentFeas if isinstance(B, BarrierMO) else B._best_feasible
      bestInf: CandidatePoint = B._currentIncumbentInf if isinstance(B, BarrierMO) else B.get_best_infeasible()
      if bestFeasible is not None and bestFeasible.evaluated:
        self.search.xmin = bestFeasible
        vvp, _ = self.search.generate_sample_points(int(((self.search.dim+1)/2)*((self.search.dim+2)/2)) if self.search.ns is None else self.search.ns)
      if bestInf is not None and bestInf.evaluated:
      # if B._filter is not None and B.get_best_infeasible().evaluated:
        xmin_bup = self.search.xmin
        Prim_samples = self.search._candidate_points_set
        self.search.xmin = bestInf#B.get_best_infeasible()
        vvs, _ = self.search.generate_sample_points(int(((self.search.dim+1)/2)*((self.search.dim+2)/2)) if self.search.ns is None else self.search.ns)
        self.search._candidate_points_set += Prim_samples
        self.search.xmin = xmin_bup
      
      if isinstance(vvs, list) and len(vvs) > 0:
        vv = vvp + vvs
      else:
        vv = vvp


    """ Save current poll directions and incumbent solution
      so they can be saved later in the post dir """
    self.search.omit_duplicates()
    if self.options.save_coordinates:
      self.post.coords.append(self.search._candidate_points_set)
      self.post.x_incumbent.append(self.search.xmin)
    """ Reset success boolean """
    self.search.success = SUCCESS_TYPES.US
    """ Reset the BB output """
    self.search.bb_output = []
    xt = []
    """ Serial evaluation for points in the poll set """
    if self.search_VN is not None:
      self.search.lb = self.search_VN.params.lb
      self.search.ub = self.search_VN.params.ub
    self.search.bb_handle.xmin = xmin
    self.search.constraints_RP.LAMBDA = xmin.LAMBDA
    self.search.constraints_RP.RHO = xmin.RHO
    self.search.constraints_RP.constraints_type = xmin.constraints_type
    self.search.constraints_RP.hmax = xmin.hmax
    if not self.options.parallel_mode:
      xt, self.post, self.peval = self.search.bb_handle.run_callable_serial_local(iter=self.iteration, peval=self.peval, eval_set=self.search._candidate_points_set, options=self.options, post=self.post, psize=self.search.mesh.getDeltaFrameSize().coordinates, stepName=f'Search: {self.search.type}', mesh=self.search.mesh, constraintsRelaxation=self.search.constraints_RP.__dict__, budget=self.options.budget)

    else:
      self.search._point_index = -1
      """ Parallel evaluation for points in the samples set """
      self.search.bb_eval, xt, self.post, self.peval = self.search.bb_handle.run_callable_parallel_local(iter=self.iteration, peval=self.peval, njobs=self.options.np, eval_set=self.search._candidate_points_set, options=self.options, post=self.post, mesh=self.search.mesh, stepName=f'Search: {self.search.type}', psize=self.search.mesh.getDeltaFrameSize().coordinates, constraintsRelaxation=self.search.constraints_RP.__dict__, budget=self.options.budget)
      
    if self.search.bb_handle.constraintsRelaxation:
      temp:ConstraintsRelaxationParameters = ConstraintsRelaxationParameters(**self.search.bb_handle.constraintsRelaxation)
      for i in range(len(temp.LAMBDA)):
        self.search.constraints_RP.LAMBDA[i] = temp.LAMBDA[i]
      self.search.constraints_RP.RHO = temp.RHO
      self.search.constraints_RP.constraints_type = temp.constraints_type
      self.search.constraints_RP.hmax = temp.hmax
      # if options.store_cache:
      #   for xi in xt:
      #     search.hashtable.hash_id = xi
      #     if not search.hashtable._isPareto:
      #       search.hashtable.add_to_best_cache(xi)
    self.LAMBDA_k = self.search.bb_handle.constraintsRelaxation["LAMBDA"]
    self.RHO_k = self.search.bb_handle.constraintsRelaxation["RHO"]
    self.search.postprocess_evaluated_candidates(xt)

    
      
    if isinstance(B, Barrier):
      xpost: List[CandidatePoint] = self.search.master_updates(xt, self.peval, save_all_best=self.options.save_all_best, save_all=self.options.save_results)
      if self.options.save_results:
        for i in range(len(xpost)):
          self.post.poll_dirs.append(xpost[i])
      for xv in xt:
        if xv.evaluated:
          B.insert(xv)

      """ Update the xmin in post"""
      self.post.xmin = copy.deepcopy(self.search.xmin)


      if self.iteration == 1:
        self.search.vicinity_ratio = np.ones((len(self.search.xmin.coordinates),1))
      
      """ Updates """
      
      if self.search.success == SUCCESS_TYPES.FS:
        dir: Point = Point(self.search.mesh._n)
        dir.coordinates = self.search.xmin.direction.coordinates
        # search.mesh.psize = np.multiply(search.mesh.get, 2, dtype=search.dtype.dtype)
        self.search.mesh.enlargeDeltaFrameSize(direction=dir)
        if self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
          self.search.update_local_region(region="expand")
      elif self.search.success == SUCCESS_TYPES.US:
        # search.mesh.psize = np.divide(search.mesh.psize, 2, dtype=search.dtype.dtype)
        self.search.mesh.refineDeltaFrameSize()
        if self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
          self.search.update_local_region(region="contract")
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
          if self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
            self.search.update_local_region(region="contract")
        if B._currentIncumbentFeas:
          B._currentIncumbentFeas.mesh.refineDeltaFrameSize()
          newMesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf.mesh) if B._currentIncumbentInf else None
          B.updateCurrentIncumbents()
          if self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
            self.search.update_local_region(region="contract")
        
        if self.iteration == 1:
          self.search.vicinity_ratio = np.ones((len(self.search.xmin.coordinates),1))
        if newMesh:
          self.search.mesh = newMesh
        else:
          self.search.mesh.refineDeltaFrameSize()
          if self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
            self.search.update_local_region(region="contract")
      else:
        self.search.mesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if updatedF else copy.deepcopy(B._currentIncumbentInf.mesh) if updatedInf else self.search.mesh
        self.search.xmin = copy.deepcopy(B._currentIncumbentFeas) if updatedF else copy.deepcopy(B._currentIncumbentInf) if updatedInf else self.search.xmin
        if self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
          self.search.update_local_region(region="expand")
      
      for i in range(len(xpost)):
        self.post.poll_dirs.append(xpost[i])
      self.search.hashtable.best_hash_ID = []
      self.search.hashtable.add_to_best_cache(B.getAllPoints())
      self.post.xmin = B._currentIncumbentFeas if updatedF  else B._currentIncumbentInf if  updatedInf else self.search.xmin
      
    self.search.mesh.update()
    
    
    if self.options.display:
      if self.log is not None:
        self.log.log_msg(msg=self.post.__str__(), msg_type=MSG_TYPE.INFO)
      print(self.post)

    toc = time.perf_counter()
    if self.log is not None:
      self.log.log_msg(f" Run completed in {toc - tic:.4f} seconds", MSG_TYPE.INFO)
      self.log.log_msg(msg=f" Success status: {self.search.success}", msg_type=MSG_TYPE.INFO)
      self.log.log_msg(msg=self.post.__str__(), msg_type=MSG_TYPE.INFO)
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
    self.RHO_k = self.search.RHO
    self.HT = self.search.hashtable
    self.active_barrier = B
    self.LAMBDA_k = self.search.LAMBDA
    return self.search.xmin

  def poll_step(self, xmin: PS.CandidatePoint=None):
    tic = time.perf_counter()
    self.poll.xmin = xmin
    self.poll.mesh.update()
    """ Create the set of poll directions """
    hhm = self.poll.create_housholder(self.options.rich_direction, domain=xmin.var_type)
    self.poll.lb = self.param.lb
    self.poll.ub = self.param.ub
    self.poll.xmin = copy.deepcopy(xmin)
    xmin.mesh = copy.deepcopy(self.poll.mesh)
    B = self.active_barrier
    if self.HT is not None:
      self.poll.hashtable = self.HT
    if B is not None:
      if isinstance(B, Barrier):
        B.insert(xmin)
        if B._filter is not None:
          B.select_poll_center()
          B.update_and_reset_success()
          
      elif isinstance(B, BarrierMO) and self.iteration == 1:
        B.init(evalPointList=[xmin])
        
    if isinstance(B, Barrier):
        self.poll.hmax = xmin.hmax
        self.poll.create_poll_set(hhm=hhm,
                  ub=self.param.ub,
                  lb=self.param.lb, it=self.iteration, var_type=xmin.var_type, var_sets=xmin.sets, var_link = xmin.var_link, c_types=self.param.constraints_type, is_prim=True)
        if B._sec_poll_center is not None and B._sec_poll_center.evaluated:
          del self.poll.poll_set
          # poll.poll_dirs = []
          self.poll.x_sc = B._sec_poll_center
          self.poll.create_poll_set(hhm=hhm,
                  ub=self.param.ub,
                  lb=self.param.lb, it=self.iteration, var_type=B._sec_poll_center.var_type, var_sets=B._sec_poll_center.sets, var_link = B._sec_poll_center.var_link, c_types=self.param.constraints_type, is_prim=False)
    elif isinstance(B, BarrierMO):
      self.poll.hmax = B._hMax
      del self.poll.poll_set
      del self.poll.poll_dirs
      if B._currentIncumbentFeas and B._currentIncumbentFeas.evaluated:
        self.poll.create_poll_set(hhm=hhm,
                ub=self.param.ub,
                lb=self.param.lb, it=self.iteration, var_type=B._currentIncumbentFeas.var_type, var_sets=B._currentIncumbentFeas.sets, var_link = B._currentIncumbentFeas.var_link, c_types=self.param.constraints_type, is_prim=True)
      elif self.poll.xmin.status == DESIGN_STATUS.FEASIBLE:
        self.poll.create_poll_set(hhm=hhm,
                ub=self.param.ub,
                lb=self.param.lb, it=self.iteration, var_type=self.poll.xmin.var_type, var_sets=self.poll.xmin.sets, var_link = self.poll.xmin.var_link, c_types=self.param.constraints_type, is_prim=True)
      
      if B._currentIncumbentInf and B._currentIncumbentInf.evaluated:
        # del poll.poll_set
        self.poll.x_sc = B._currentIncumbentInf
        self.poll.create_poll_set(hhm=hhm,
                ub=self.param.ub,
                lb=self.param.lb, it=self.iteration, var_type=B._currentIncumbentInf.var_type, var_sets=B._currentIncumbentInf.sets, var_link = B._currentIncumbentInf.var_link, c_types=self.param.constraints_type, is_prim=False)
      elif self.poll.xmin.status == DESIGN_STATUS.INFEASIBLE:
        self.poll.create_poll_set(hhm=hhm,
                ub=self.param.ub,
                lb=self.param.lb, it=self.iteration, var_type=self.poll.xmin.var_type, var_sets=self.poll.xmin.sets, var_link = self.poll.xmin.var_link, c_types=self.param.constraints_type, is_prim=False)
      
    
    self.poll.LAMBDA = self.LAMBDA_k
    self.poll.RHO = self.RHO_k

    """ Save current poll directions and incumbent solution
      so they can be saved later in the post dir """
    self.poll.omit_duplicates()
    if self.options.save_coordinates:
      self.post.coords.append(self.poll.poll_set)
      self.post.x_incumbent.append(self.poll.xmin)
    """ Reset success boolean """
    self.poll.success = SUCCESS_TYPES.US
    """ Reset the BB output """
    self.poll.bb_output = []
    xt = []
    self.poll.bb_handle.xmin = xmin
    """ Serial evaluation for points in the poll set """
    self.poll.constraints_RP.LAMBDA = xmin.LAMBDA
    self.poll.constraints_RP.RHO = xmin.RHO
    self.poll.constraints_RP.constraints_type = xmin.constraints_type
    self.poll.constraints_RP.hmax = xmin.hmax
    if not self.options.parallel_mode:
      xt, self.post, self.peval = self.poll.bb_handle.run_callable_serial_local(iter=self.iteration, peval=self.peval, eval_set=self.poll._candidate_points_set, options=self.options, post=self.post, psize=self.poll.mesh.getDeltaFrameSize().coordinates, stepName=f'Poll Step', mesh=self.poll.mesh, constraintsRelaxation=self.poll.constraints_RP.__dict__, budget=self.options.budget)

    else:
      self.poll.point_index = -1
      """ Parallel evaluation for points in the samples set """
      self.poll.bb_eval, xt, self.post, self.peval = self.poll.bb_handle.run_callable_parallel_local(iter=self.iteration, peval=self.peval, njobs=self.options.np, eval_set=self.poll._candidate_points_set, options=self.options, post=self.post, mesh=self.poll.mesh, stepName=f'Poll Step', psize=self.poll.mesh.getDeltaFrameSize().coordinates, constraintsRelaxation=self.poll.constraints_RP.__dict__, budget=self.options.budget)
      
    if self.poll.bb_handle.constraintsRelaxation:
      temp:ConstraintsRelaxationParameters = ConstraintsRelaxationParameters(**self.poll.bb_handle.constraintsRelaxation)
      for i in range(len(temp.LAMBDA)):
        self.poll.constraints_RP.LAMBDA[i] = temp.LAMBDA[i]
      self.poll.constraints_RP.RHO = temp.RHO
      self.poll.constraints_RP.constraints_type = temp.constraints_type
      self.poll.constraints_RP.hmax = temp.hmax
      # if options.store_cache:
      #   for xi in xt:
      #     poll.hashtable.hash_id = xi
      #     if not poll.hashtable._isPareto:
      #       poll.hashtable.add_to_best_cache(xi)
    self.LAMBDA_k = self.poll.bb_handle.constraintsRelaxation["LAMBDA"]
    self.RHO_k = self.poll.bb_handle.constraintsRelaxation["RHO"]
    self.poll.postprocess_evaluated_candidates(xt)

    if isinstance(B, Barrier):
        xpost: List[CandidatePoint] = self.poll.master_updates(xt, self.peval, save_all_best=self.options.save_all_best, save_all=self.options.save_results)
        xmin = copy.deepcopy(self.poll.xmin)
        if self.options.save_results:
          for i in range(len(xpost)):
            self.post.poll_dirs.append(xpost[i])
        for xv in xt:
          if xv.evaluated:
            B.insert(xv)

        """ Update the xmin in post"""
        self.post.xmin = copy.deepcopy(self.poll.xmin)

        """ Updates """
        pev = 0.
        for p in self.poll.poll_set:
          if p.evaluated:
            pev += 1
        # if pev != poll.poll_dirs and not poll.success:
        #   poll.seed += 1
        goToSearch: bool = (pev == 0 and self.poll.Failure_stop is not None and self.poll.Failure_stop)
        
        dir: Point = Point(self.poll._n)
        dir.coordinates = self.poll.xmin.direction.coordinates if self.poll.xmin.direction is not None else [0]*self.poll._n
        if self.poll.success == SUCCESS_TYPES.FS and not goToSearch:
          self.poll.mesh.enlargeDeltaFrameSize(direction=dir) # poll.mesh.psize =  np.multiply(poll.mesh.psize, 2, dtype=poll.dtype.dtype
        elif self.poll.success == SUCCESS_TYPES.US:
          self.poll.mesh.refineDeltaFrameSize()
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
          self.poll.mesh = newMesh
        else:
          self.poll.mesh.refineDeltaFrameSize()
          
      else:
        self.poll.mesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf.mesh) if B._currentIncumbentInf else self.poll.mesh
        self.poll.xmin = copy.deepcopy(B._currentIncumbentFeas) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf) if B._currentIncumbentInf else self.poll.xmin
      for i in range(len(xpost)):
        self.post.poll_dirs.append(xpost[i])
      
      self.post.xmin = B._currentIncumbentFeas if B._currentIncumbentFeas  else B._currentIncumbentInf if  B._currentIncumbentInf else self.poll.xmin
    self.poll.mesh.update()
    if self.log is not None:
        self.log.log_msg(msg=self.post.__str__(), msg_type=MSG_TYPE.INFO)
    if self.options.display:
      print(self.post)
    
    self.LAMBDA_k = self.poll.LAMBDA
    self.RHO_k = self.poll.xmin.RHO

    toc = time.perf_counter()

    if self.log is not None:
      # log.log_msg(msg=" ---Run Summary--- ", msg_type=MSG_TYPE.INFO)
      self.log.log_msg(msg=f" Run completed in {toc - tic:.4f} seconds", msg_type=MSG_TYPE.INFO)
      self.log.log_msg(msg=f" Success status: {self.poll.success}", msg_type=MSG_TYPE.INFO)
      self.log.log_msg(msg=self.post.__str__(), msg_type=MSG_TYPE.INFO)
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
    self.HT = self.poll.hashtable
    self.active_barrier = B
    self.LAMBDA_k = self.poll.xmin.LAMBDA
    return self.poll.xmin

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
  MADS_agent: MADS = MADS(data=data)
  iteration: int
  xmin: CandidatePoint
  options: Options
  param: Parameters 
  post: PostMADS 
  out: Output 
  B: Barrier
  # poll: PS.Dirs2n
  search: SS.efficient_exploration
  log.log_msg(msg="Preprocess the search step...", msg_type=PS.MSG_TYPE.INFO)
  _, _, MADS_agent.search, _, _, _, _, _, _ = SS.PreExploration(data).initialize_from_dict(log=log)
  log.log_msg(msg="Preprocess the MADS algorithim...", msg_type=PS.MSG_TYPE.INFO)
  iteration, xmin, MADS_agent.poll, options, param, post, out, B, outP = PS.PrePoll(data).initialize_from_dict(log=log, xs=MADS_agent.search.xmin)
  out.stepName = "Poll"
  post.step_name = [f'Search: {MADS_agent.search.type}']

  HT = MADS_agent.poll.hashtable
  
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

  if MADS_agent.search.type == SS.SEARCH_TYPE.VNS.name:
    search_VN = SS.VNS(active_barrier=B, params=param)
    search_VN._ns_dist = [int(((MADS_agent.search.dim+1)/2)*((MADS_agent.search.dim+2)/2)/(len(search_VN._dist))) if MADS_agent.search.ns is None else MADS_agent.search.ns] * len(search_VN._dist)
    MADS_agent.search.ns = sum(search_VN._ns_dist)
  else:
    search_VN = None
  
  MADS_agent.search.lb = param.lb
  MADS_agent.search.ub = param.ub
  MADS_agent.options = options
  MADS_agent.param = param
  MADS_agent.log = log
  MADS_agent.outP = outP
  MADS_agent.out = out
  MADS_agent.post = post
  MADS_agent.HT = HT
  MADS_agent.peval = peval
  MADS_agent.RHO_k = RHO_k
  MADS_agent.active_barrier = B
  MADS_agent.LAMBDA_k = LAMBDA_k

  while True:
    """ Run search step (Optional) """
    # TODO: This rule cannot be generalized -- needs further invistigation
    # if poll.dim > 10 and poll.mesh.psize >= 1E-4:
    #   canSearch = False
    # else:
    canSearch = True
    MADS_agent.iteration = iteration

    if canSearch and (MADS_agent.poll.success == SUCCESS_TYPES.US or iteration == 1):
      MADS_agent.log.log_msg(f"------- Iteration # {iteration}: Run the search step -------", MSG_TYPE.INFO)
      MADS_agent.search.iter = iteration
      xmin = MADS_agent.search_step(xmin=xmin)
    """ Run the poll step (Mandatory step) """
    MADS_agent.log.log_msg(f"------- Iteration # {iteration}: Run the poll step -------", MSG_TYPE.INFO)
    xmin = MADS_agent.poll_step(xmin=xmin)
    xmin = MADS_agent.poll.xmin
    MADS_agent.search.mesh = copy.deepcopy(MADS_agent.poll.mesh)
    MADS_agent.search.psize = copy.deepcopy(MADS_agent.poll.psize)
    """ Check stopping criteria"""
    pt = (all(abs(MADS_agent.poll.mesh.getDeltaFrameSize().coordinates[pp]) < options.tol for pp in range(MADS_agent.poll._n)))
    st = (all(abs(MADS_agent.search.mesh.getdeltaMeshSize().coordinates[pp]) < options.tol  for pp in range(MADS_agent.search.mesh._n)))
    if options.save_results:
      MADS_agent.post.output_results(out, False)
      if param.isPareto:
        MADS_agent.post.nd_points = []
        for i in range(len(B.getAllPoints())):
          MADS_agent.post.nd_points.append(B.getAllPoints()[i])
        MADS_agent.post.output_nd_results(outP)
    if (pt or st or MADS_agent.search.bb_eval + MADS_agent.poll.bb_eval >= options.budget):
      MADS_agent.log.log_msg(f"\n--------------- Termination of MADS  ---------------", MSG_TYPE.INFO)
      if pt:
        MADS_agent.log.log_msg(f"Termination criterion hit: the poll size is below the minimum threshold defined.", MSG_TYPE.INFO)
      if st:
        MADS_agent.log.log_msg(f"Termination criterion hit: the mesh size is below the minimum threshold defined.", MSG_TYPE.INFO)
      if (MADS_agent.search.bb_eval + MADS_agent.poll.bb_eval >= options.budget):
        MADS_agent.log.log_msg(f"Termination criterion hit: Evaluation budget is exhausted.", MSG_TYPE.INFO)
      MADS_agent.log.log_msg(f"----------------------------------------------------\n", MSG_TYPE.INFO)
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
    if len(MADS_agent.poll.bb_output) > 0:
      b.add_row(name=MADS_agent.poll.bb_handle.blackbox,
            run_index=int(args[2]),
            nv=len(param.baseline),
            nc=ncon,
            nb_success=MADS_agent.poll.nb_success,
            it=iteration,
            BBEVAL=MADS_agent.poll.bb_eval,
            runtime=toc - tic,
            feval=MADS_agent.poll.bb_handle.bb_eval,
            hmin=MADS_agent.poll.xmin.h,
            fmin=MADS_agent.poll.xmin.f)
    print(f"{MADS_agent.poll.bb_handle.blackbox}: fmin = {MADS_agent.poll.xmin.f} , hmin= {MADS_agent.poll.xmin.h:.2f}")

  elif importlib.util.find_spec('BMDFO') and len(args) > 1 and not isinstance(args[1], toy.Run):
    raise IOError("Could not find " + args[1] + " in the internal BM suite.")

  # if options.save_results:
  #   post.output_results(out)
  
  out_step: Any = None
  if MADS_agent.poll.xmin < MADS_agent.search.xmin:
    out_step = MADS_agent.poll
  elif MADS_agent.search.xmin < MADS_agent.poll.xmin:
    out_step = MADS_agent.search
  else:
    out_step = MADS_agent.poll
  
  if out_step is None:
    out_step = MADS_agent.poll
  

  if options.display:
    print(" end of orthogonal MADS ")
    print(" Final objective value: " + str(out_step.xmin.f) + ", hmin= " + str(out_step.xmin.h))

  if options.save_coordinates:
    MADS_agent.post.output_coordinates(out)
  
  if MADS_agent.log is not None:
    MADS_agent.log.log_msg(msg=" --- MADS Run Summary--- ", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" Run completed in {toc - tic:.4f} seconds", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" # of successful search steps = {MADS_agent.search.n_successes}", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" # of successful poll steps = {MADS_agent.poll.n_successes}", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" Run completed in {toc - tic:.4f} seconds", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" Random numbers generator's seed {options.seed}", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" xmin = {MADS_agent.poll.xmin.__str__()} ", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" hmin = {MADS_agent.poll.xmin.h} ", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" fmin {MADS_agent.poll.xmin.fobj}", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" Search step # BB evals =  {MADS_agent.search.bb_eval} ", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" Poll step # BB evals =  {MADS_agent.poll.bb_eval} ", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" Total # BB evals =  {MADS_agent.poll.bb_eval + MADS_agent.search.bb_eval} ", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" #iterations =  {iteration} ", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" psize = {MADS_agent.poll.mesh.getDeltaFrameSize().coordinates} ", msg_type=MSG_TYPE.INFO)
    MADS_agent.log.log_msg(msg=f" psize_success = {MADS_agent.poll.xmin.mesh.getDeltaFrameSize().coordinates}", msg_type=MSG_TYPE.INFO)
    if isinstance(B, BarrierMO):
      MADS_agent.log.log_msg(msg=f" Hypervolume metric = {HV}", msg_type=MSG_TYPE.INFO)
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
    print(" nb_success = " + str(MADS_agent.poll.nb_success + MADS_agent.search.nb_success))
    print(" psize = " + str(MADS_agent.poll.mesh.getDeltaFrameSize().coordinates))
    print(" psize_success = " + str(MADS_agent.poll.xmin.mesh.getDeltaFrameSize().coordinates))
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
                "nb_success": MADS_agent.poll.nb_success + MADS_agent.search.nb_success,
                "psize": MADS_agent.poll.mesh.getDeltaFrameSize().coordinates,
                "psuccess": MADS_agent.poll.xmin.mesh.getDeltaFrameSize().coordinates,
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