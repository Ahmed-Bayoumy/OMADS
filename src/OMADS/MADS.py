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
from .Exploration import efficient_exploration
from .Directions import Dirs2n
from typing import List, Dict, Any, Optional
import numpy as np
if importlib.util.find_spec('BMDFO'):
  from BMDFO import toy # type: ignore
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
  search: Optional[efficient_exploration] = None
  search_vns: SS.VNS = None
  poll: Optional[Dirs2n] = None
  param: Optional[Parameters] = None
  evaluator: Optional[Evaluator] = None
  post: Optional[PostMADS] = None
  out: Optional[Output] = None
  out_p: Optional[Output] = None
  options: Optional[Options] = None
  data: Optional[dict] = None
  xmin: Optional[CandidatePoint] = None
  iteration: int = 0
  peval: int = 0
  HT: Any = None
  log: Optional[logger] = None
  B: Any = None
  lambda_multipliers_k: float = 0.
  rho_k: float = 0.
  tic: Any = None
  toc: Any = None
  active_barrier: Optional[Barrier] = None

  def __init__(self, data: dict):
    """ Initialize the log file """
    self.log = logger()
    if not os.path.exists(data["param"]["post_dir"]):
      try:
        os.mkdir(data["param"]["post_dir"])
      except Warning:
        os.makedirs(data["param"]["post_dir"], exist_ok=True)

    self.log.initialize(data["param"]["post_dir"] + "/OMADS.log")
    self.log.log_msg(msg="Preprocess the MADS algorithim...", msg_type=MSG_TYPE.INFO)
    self.log.log_msg(msg="Preprocess the search step...", msg_type=MSG_TYPE.INFO)
    _, _, self.search, _, _, _, _, _, _ = SS.PreExploration(data).initialize_from_dict(log=self.log)
    self.log.log_msg(msg="Preprocess the POLL step...", msg_type=MSG_TYPE.INFO)
    self.iteration, self.xmin, self.poll, self.options, self.param, self.post, self.out, self.B, self.out_p = PS.PrePoll(data).initialize_from_dict(log=self.log, xs=self.search.xmin)
    self.out.step_name = "Poll"
    self.post.step_name = [f'Search: {self.search.type}']

    self.HT = copy.deepcopy(self.poll.hashtable)

  def search_step(self, xmin: SS.CandidatePoint=None):
    """ Reset success boolean """
    self.search.success = SUCCESS_TYPES.US
    tic = time.perf_counter()
    self.search.log = self.log
    self.search.xmin = xmin
    self.search.mesh.update()
    self.search.LAMBDA = self.lambda_multipliers_k
    self.search.RHO = self.rho_k
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
          B.init(eval_point_list=[xmin])
        
        
    if isinstance(B, Barrier) or isinstance(B, BarrierMO):
      self.search.hmax = B._h_max
      # COMPLETED: Check whether the commented code below is needed
    """ Create the set of poll directions """
    if self.search.type == SS.SEARCH_TYPE.VNS.name and self.search_vns is not None:
      self.search_vns.active_barrier = B
      self.search._candidate_points_set = self.search_vns.run()
      if self.search_vns.stop:
        print("Reached maximum number of VNS iterations!")
        self.HT = self.search.hashtable
        self.rho_k = self.search.RHO
        self.active_barrier = B
        self.lambda_multipliers_k = self.search.LAMBDA
        return self.search.xmin
      self.search.map_samples_from_coords_to_points(samples=self.search._candidate_points_set)
    else:
      best_feasible: CandidatePoint = B._currentIncumbentFeas if isinstance(B, BarrierMO) else B._best_feasible
      best_inf: CandidatePoint = B._currentIncumbentInf if isinstance(B, BarrierMO) else B.get_best_infeasible()
      if best_feasible is not None and best_feasible.evaluated:
        self.search.xmin = best_feasible
        self.search.generate_sample_points(int(((self.search.dim+1)/2)*((self.search.dim+2)/2)) if self.search.ns is None else self.search.ns)
      if best_inf is not None and best_inf.evaluated:
      # if B._filter is not None and B.get_best_infeasible().evaluated:
        xmin_bup = self.search.xmin
        prim_samples = self.search._candidate_points_set
        self.search.xmin = best_inf#B.get_best_infeasible()
        self.search.generate_sample_points(int(((self.search.dim+1)/2)*((self.search.dim+2)/2)) if self.search.ns is None else self.search.ns)
        self.search._candidate_points_set += prim_samples
        self.search.xmin = xmin_bup


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
    if self.search_vns is not None:
      self.search.lb = self.search_vns.params.lb
      self.search.ub = self.search_vns.params.ub
    self.search.bb_handle.xmin = xmin
    self.search.constraints_RP.LAMBDA = xmin.lambda_multipliers
    self.search.constraints_RP.RHO = xmin.rho
    self.search.constraints_RP.constraints_type = xmin.constraints_type
    self.search.constraints_RP.hmax = xmin.h_max
    if not self.options.parallel_mode:
      xt, self.post, self.peval = self.search.bb_handle.run_callable_serial_local(iter=self.iteration, peval=self.peval, eval_set=self.search._candidate_points_set, options=self.options, post=self.post, psize=self.search.mesh.getDeltaFrameSize().coordinates, step_name=f'Search: {self.search.type}', mesh=self.search.mesh, constraints_relaxation=self.search.constraints_RP.__dict__, budget=self.options.budget)

    else:
      self.search._point_index = -1
      """ Parallel evaluation for points in the samples set """
      self.search.bb_eval, xt, self.post, self.peval = self.search.bb_handle.run_callable_parallel_local(iter=self.iteration, peval=self.peval, eval_set=self.search._candidate_points_set, options=self.options, post=self.post, mesh=self.search.mesh, step_name=f'Search: {self.search.type}', psize=self.search.mesh.getDeltaFrameSize().coordinates, constraints_relaxation=self.search.constraints_RP.__dict__, budget=self.options.budget)
      
    if self.search.bb_handle.constraints_relaxation:
      temp:ConstraintsRelaxationParameters = ConstraintsRelaxationParameters(**self.search.bb_handle.constraints_relaxation)
      for i in range(len(temp.LAMBDA)):
        self.search.constraints_RP.LAMBDA[i] = temp.LAMBDA[i]
      self.search.constraints_RP.RHO = temp.RHO
      self.search.constraints_RP.constraints_type = temp.constraints_type
      self.search.constraints_RP.hmax = temp.hmax

    self.lambda_multipliers_k = self.search.bb_handle.constraints_relaxation["LAMBDA"]
    self.rho_k = self.search.bb_handle.constraints_relaxation["RHO"]
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
        direction: Point = Point(self.search.mesh._n)
        direction.coordinates = self.search.xmin.direction.coordinates
        self.search.mesh.enlargeDeltaFrameSize(direction=direction)
        if self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
          self.search.update_local_region(region="expand")
      elif self.search.success == SUCCESS_TYPES.US:
        self.search.mesh.refineDeltaFrameSize()
        if self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
          self.search.update_local_region(region="contract")
    elif isinstance(B, BarrierMO):
      xpost: List[CandidatePoint] = []
      for i in range(len(xt)):
        xpost.append(xt[i])
      updated, updated_f, updated_inf = B.updateWithPoints(eval_point_list=xpost, keep_all_points=False)
      if not updated:
        new_mesh = None
        if B._currentIncumbentInf:
          B._currentIncumbentInf.mesh.refineDeltaFrameSize()
          new_mesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf.mesh) if B._currentIncumbentInf else None
          B.updateCurrentIncumbents()
          if self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
            self.search.update_local_region(region="contract")
        if B._currentIncumbentFeas:
          B._currentIncumbentFeas.mesh.refineDeltaFrameSize()
          new_mesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf.mesh) if B._currentIncumbentInf else None
          B.updateCurrentIncumbents()
          if self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
            self.search.update_local_region(region="contract")
        
        if self.iteration == 1:
          self.search.vicinity_ratio = np.ones((len(self.search.xmin.coordinates),1))
        if new_mesh:
          self.search.mesh = new_mesh
        else:
          self.search.mesh.refineDeltaFrameSize()
          if self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
            self.search.update_local_region(region="contract")
      else:
        self.search.mesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if updated_f else copy.deepcopy(B._currentIncumbentInf.mesh) if updated_inf else self.search.mesh
        self.search.xmin = copy.deepcopy(B._currentIncumbentFeas) if updated_f else copy.deepcopy(B._currentIncumbentInf) if updated_inf else self.search.xmin
        if self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
          self.search.update_local_region(region="expand")
      
      for i in range(len(xpost)):
        self.post.poll_dirs.append(xpost[i])
      self.search.hashtable.best_hash_ID = []
      self.search.hashtable.add_to_best_cache(B.getAllPoints())
      self.post.xmin = B._currentIncumbentFeas if updated_f  else B._currentIncumbentInf if  updated_inf else self.search.xmin
      
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
      
    self.rho_k = self.search.RHO
    self.HT = self.search.hashtable
    self.active_barrier = B
    self.lambda_multipliers_k = self.search.LAMBDA
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
        B.init(eval_point_list=[xmin])
        
    if isinstance(B, Barrier):
        self.poll.hmax = xmin.h_max
        self.poll.create_poll_set(hhm=hhm,
                  ub=self.param.ub,
                  lb=self.param.lb, it=self.iteration, var_type=xmin.var_type, var_sets=xmin.sets, var_link = xmin.var_link, c_types=self.param.constraints_type, is_prim=True)
        if B._sec_poll_center is not None and B._sec_poll_center.evaluated:
          del self.poll.poll_set
          self.poll.x_sc = B._sec_poll_center
          self.poll.create_poll_set(hhm=hhm,
                  ub=self.param.ub,
                  lb=self.param.lb, it=self.iteration, var_type=B._sec_poll_center.var_type, var_sets=B._sec_poll_center.sets, var_link = B._sec_poll_center.var_link, c_types=self.param.constraints_type, is_prim=False)
    elif isinstance(B, BarrierMO):
      self.poll.hmax = B._h_max
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
        self.poll.x_sc = B._currentIncumbentInf
        self.poll.create_poll_set(hhm=hhm,
                ub=self.param.ub,
                lb=self.param.lb, it=self.iteration, var_type=B._currentIncumbentInf.var_type, var_sets=B._currentIncumbentInf.sets, var_link = B._currentIncumbentInf.var_link, c_types=self.param.constraints_type, is_prim=False)
      elif self.poll.xmin.status == DESIGN_STATUS.INFEASIBLE:
        self.poll.create_poll_set(hhm=hhm,
                ub=self.param.ub,
                lb=self.param.lb, it=self.iteration, var_type=self.poll.xmin.var_type, var_sets=self.poll.xmin.sets, var_link = self.poll.xmin.var_link, c_types=self.param.constraints_type, is_prim=False)
      
    
    self.poll.LAMBDA = self.lambda_multipliers_k
    self.poll.RHO = self.rho_k

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
    self.poll.constraints_RP.LAMBDA = xmin.lambda_multipliers
    self.poll.constraints_RP.RHO = xmin.rho
    self.poll.constraints_RP.constraints_type = xmin.constraints_type
    self.poll.constraints_RP.hmax = xmin.h_max
    if not self.options.parallel_mode:
      xt, self.post, self.peval = self.poll.bb_handle.run_callable_serial_local(iter=self.iteration, peval=self.peval, eval_set=self.poll._candidate_points_set, options=self.options, post=self.post, psize=self.poll.mesh.getDeltaFrameSize().coordinates, step_name='Poll Step', mesh=self.poll.mesh, constraints_relaxation=self.poll.constraints_RP.__dict__, budget=self.options.budget)

    else:
      self.poll.point_index = -1
      """ Parallel evaluation for points in the samples set """
      self.poll.bb_eval, xt, self.post, self.peval = self.poll.bb_handle.run_callable_parallel_local(iter=self.iteration, peval=self.peval, eval_set=self.poll._candidate_points_set, options=self.options, post=self.post, mesh=self.poll.mesh, step_name='Poll Step', psize=self.poll.mesh.getDeltaFrameSize().coordinates, constraints_relaxation=self.poll.constraints_RP.__dict__, budget=self.options.budget)
      
    if self.poll.bb_handle.constraints_relaxation:
      temp:ConstraintsRelaxationParameters = ConstraintsRelaxationParameters(**self.poll.bb_handle.constraints_relaxation)
      for i in range(len(temp.LAMBDA)):
        self.poll.constraints_RP.LAMBDA[i] = temp.LAMBDA[i]
      self.poll.constraints_RP.RHO = temp.RHO
      self.poll.constraints_RP.constraints_type = temp.constraints_type
      self.poll.constraints_RP.hmax = temp.hmax

    self.lambda_multipliers_k = self.poll.bb_handle.constraints_relaxation["LAMBDA"]
    self.rho_k = self.poll.bb_handle.constraints_relaxation["RHO"]
    self.poll.postprocess_evaluated_candidates(xt)

    if isinstance(B, Barrier):
        xpost: List[CandidatePoint] = self.poll.master_updates(xt, self.peval, save_all_best=self.options.save_all_best, save_all=self.options.save_results)
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

        go_to_search: bool = (pev == 0 and self.poll.Failure_stop is not None and self.poll.Failure_stop)
        
        direction: Point = Point(self.poll._n)
        direction.coordinates = self.poll.xmin.direction.coordinates if self.poll.xmin.direction is not None else [0]*self.poll._n
        if self.poll.success == SUCCESS_TYPES.FS and not go_to_search:
          self.poll.mesh.enlargeDeltaFrameSize(direction=direction) # poll.mesh.psize =  np.multiply(poll.mesh.psize, 2, dtype=poll.dtype.dtype
        elif self.poll.success == SUCCESS_TYPES.US:
          self.poll.mesh.refineDeltaFrameSize()
        
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
          self.poll.mesh = new_mesh
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
    
    self.lambda_multipliers_k = self.poll.LAMBDA
    self.rho_k = self.poll.xmin.rho

    toc = time.perf_counter()

    if self.log is not None:
      # log.log_msg(msg=" ---Run Summary--- ", msg_type=MSG_TYPE.INFO)
      self.log.log_msg(msg=f" Run completed in {toc - tic:.4f} seconds", msg_type=MSG_TYPE.INFO)
      self.log.log_msg(msg=f" Success status: {self.poll.success}", msg_type=MSG_TYPE.INFO)
      self.log.log_msg(msg=self.post.__str__(), msg_type=MSG_TYPE.INFO)

    self.HT = self.poll.hashtable
    self.active_barrier = B
    self.lambda_multipliers_k = self.poll.xmin.lambda_multipliers
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
     except Warning:
      os.makedirs(data["param"]["post_dir"], exist_ok=True)

  log.initialize(data["param"]["post_dir"] + "/OMADS.log")

  """ Run preprocessor for the setup of
   the optimization problem and for the initialization
  of optimization process """
  mads_agent: MADS = MADS(data=data)
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
  _, _, mads_agent.search, _, _, _, _, _, _ = SS.PreExploration(data).initialize_from_dict(log=log)
  log.log_msg(msg="Preprocess the MADS algorithim...", msg_type=PS.MSG_TYPE.INFO)
  iteration, xmin, mads_agent.poll, options, param, post, out, B, out_p = PS.PrePoll(data).initialize_from_dict(log=log, xs=mads_agent.search.xmin)
  out.step_name = "Poll"
  post.step_name = [f'Search: {mads_agent.search.type}']

  HT = mads_agent.poll.hashtable
  

  """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))

  """ Start the count down for calculating the runtime indicator """
  tic = PS.time.perf_counter()
  peval = 0
  lambda_multipliers = xmin.lambda_multipliers
  rho_k = xmin.rho

  if mads_agent.search.type == SS.SEARCH_TYPE.VNS.name:
    search_vn = SS.VNS(active_barrier=B, params=param)
    search_vn._ns_dist = [int(((mads_agent.search.dim+1)/2)*((mads_agent.search.dim+2)/2)/(len(search_vn._dist))) if mads_agent.search.ns is None else mads_agent.search.ns] * len(search_vn._dist)
    mads_agent.search.ns = sum(search_vn._ns_dist)
  else:
    search_vn = None
  
  mads_agent.search.lb = param.lb
  mads_agent.search.ub = param.ub
  mads_agent.options = options
  mads_agent.param = param
  mads_agent.log = log
  mads_agent.out_p = out_p
  mads_agent.out = out
  mads_agent.post = post
  mads_agent.HT = HT
  mads_agent.peval = peval
  mads_agent.rho_k = rho_k
  mads_agent.active_barrier = B
  mads_agent.lambda_multipliers_k = lambda_multipliers
  original_st = copy.deepcopy(mads_agent.search.sampling_t)
  while True:
    """ Run search step (Optional) """
    # COMPLETED: This rule cannot be generalized -- needs further invistigation
    # if poll.dim > 10 and poll.mesh.psize >= 1E-4:
    #   canSearch = False
    # else:
    can_search = True
    mads_agent.iteration = iteration

    if can_search and (mads_agent.poll.success == SUCCESS_TYPES.US or iteration == 1):
      mads_agent.log.log_msg(f"------- Iteration # {iteration}: Run the search step -------", MSG_TYPE.INFO)
      mads_agent.search.iter = iteration
      xmin = mads_agent.search_step(xmin=xmin)
    """ Run the poll step (Mandatory step) """
    mads_agent.log.log_msg(f"------- Iteration # {iteration}: Run the poll step -------", MSG_TYPE.INFO)
    xmin = mads_agent.poll_step(xmin=xmin)
    xmin = mads_agent.poll.xmin
    mads_agent.search.mesh = copy.deepcopy(mads_agent.poll.mesh)
    mads_agent.search.psize = copy.deepcopy(mads_agent.poll.psize)
    """ Check stopping criteria"""
    pt = (all(abs(mads_agent.poll.mesh.getDeltaFrameSize().coordinates[pp]) < options.tol for pp in range(mads_agent.poll._n)))
    st = (all(abs(mads_agent.search.mesh.getdeltaMeshSize().coordinates[pp]) < options.tol  for pp in range(mads_agent.search.mesh._n)))
    if options.save_results:
      mads_agent.post.output_results(out, False)
      if param.isPareto:
        mads_agent.post.nd_points = []
        for i in range(len(B.getAllPoints())):
          mads_agent.post.nd_points.append(B.getAllPoints()[i])
        mads_agent.post.output_nd_results(out_p)
    if (pt or st or mads_agent.search.bb_eval + mads_agent.poll.bb_eval >= options.budget):
      mads_agent.log.log_msg("\n--------------- Termination of MADS  ---------------", MSG_TYPE.INFO)
      if pt:
        mads_agent.log.log_msg("Termination criterion hit: the poll size is below the minimum threshold defined.", MSG_TYPE.INFO)
      if st:
        mads_agent.log.log_msg("Termination criterion hit: the mesh size is below the minimum threshold defined.", MSG_TYPE.INFO)
      if (mads_agent.search.bb_eval + mads_agent.poll.bb_eval >= options.budget):
        mads_agent.log.log_msg("Termination criterion hit: Evaluation budget is exhausted.", MSG_TYPE.INFO)
      mads_agent.log.log_msg("----------------------------------------------------\n", MSG_TYPE.INFO)
      break
    iteration += 1
    

  toc = PS.time.perf_counter()
  if isinstance(B, BarrierMO):
    rp: Optional[CandidatePoint] = None
    if param.ref_point:
      rp =  Point()
      rp.coordinates = param.ref_point
    perf_m = Metrics(nd_solutions=B.getAllPoints(), nobj=B._nobj, ref_point=rp)
    HV = perf_m.hypervolume()

  """ If benchmarking, then populate the results in the benchmarking output report """
  if importlib.util.find_spec('BMDFO') and len(args) > 1 and isinstance(args[1], PS.toy.Run):
    b: PS.toy.Run = args[1]
    if b.test_suite == "uncon":
      ncon = 0
    else:
      ncon = len(xmin.c_ineq)
    if len(mads_agent.poll.bb_output) > 0:
      b.add_row(name=mads_agent.poll.bb_handle.blackbox,
            run_index=int(args[2]),
            nv=len(param.baseline),
            nc=ncon,
            nb_success=mads_agent.poll.nb_success,
            it=iteration,
            BBEVAL=mads_agent.poll.bb_eval,
            runtime=toc - tic,
            feval=mads_agent.poll.bb_handle.bb_eval,
            hmin=mads_agent.poll.xmin.h,
            fmin=mads_agent.poll.xmin.f)
    print(f"{mads_agent.poll.bb_handle.blackbox}: fmin = {mads_agent.poll.xmin.f} , hmin= {mads_agent.poll.xmin.h:.2f}")

  elif importlib.util.find_spec('BMDFO') and len(args) > 1 and not isinstance(args[1], toy.Run):
    raise IOError("Could not find " + args[1] + " in the internal BM suite.")

  
  out_step: Any = None
  if mads_agent.poll.xmin < mads_agent.search.xmin:
    out_step = mads_agent.poll
  elif mads_agent.search.xmin < mads_agent.poll.xmin:
    out_step = mads_agent.search
  else:
    out_step = mads_agent.poll
  
  if out_step is None:
    out_step = mads_agent.poll
  

  if options.display:
    print(" end of orthogonal MADS ")
    print(" Final objective value: " + str(out_step.xmin.f) + ", hmin= " + str(out_step.xmin.h))

  if options.save_coordinates:
    mads_agent.post.output_coordinates(out)
  
  if mads_agent.log is not None:
    mads_agent.log.log_msg(msg=" --- MADS Run Summary--- ", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" Run completed in {toc - tic:.4f} seconds", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" # of successful search steps = {mads_agent.search.n_successes}", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" # of successful poll steps = {mads_agent.poll.n_successes}", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" Run completed in {toc - tic:.4f} seconds", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" Random numbers generator's seed {options.seed}", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" xmin = {mads_agent.poll.xmin.__str__()} ", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" hmin = {mads_agent.poll.xmin.h} ", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" fmin {mads_agent.poll.xmin.fobj}", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" Search step # BB evals =  {mads_agent.search.bb_eval} ", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" Poll step # BB evals =  {mads_agent.poll.bb_eval} ", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" Total # BB evals =  {mads_agent.poll.bb_eval + mads_agent.search.bb_eval} ", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" #iterations =  {iteration} ", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" psize = {mads_agent.poll.mesh.getDeltaFrameSize().coordinates} ", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(msg=f" psize_success = {mads_agent.poll.xmin.mesh.getDeltaFrameSize().coordinates}", msg_type=MSG_TYPE.INFO)
    if isinstance(B, BarrierMO):
      mads_agent.log.log_msg(msg=f" Hypervolume metric = {HV}", msg_type=MSG_TYPE.INFO)
  if options.display:
    print("\n ---MADS Run Summary---")
    print(f" Run completed in {toc - tic:.4f} seconds")
    print(f" Random numbers generator's seed {options.seed}")
    print(" xmin = " + str(out_step.xmin))
    print(" hmin = " + str(out_step.xmin.h))
    print(" fmin = " + str(out_step.xmin.f))
    print(" #bb_eval = " + str(out_step.bb_eval))
    print(" #iteration = " + str(iteration))
    print(" nb_success = " + str(mads_agent.poll.nb_success + mads_agent.search.nb_success))
    print(" psize = " + str(mads_agent.poll.mesh.getDeltaFrameSize().coordinates))
    print(" psize_success = " + str(mads_agent.poll.xmin.mesh.getDeltaFrameSize().coordinates))
    
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
                "nb_success": mads_agent.poll.nb_success + mads_agent.search.nb_success,
                "psize": mads_agent.poll.mesh.getDeltaFrameSize().coordinates,
                "psuccess": mads_agent.poll.xmin.mesh.getDeltaFrameSize().coordinates,
                # "pmax": poll.mesh.psize_max,
                "msize": out_step.mesh.getdeltaMeshSize().coordinates,
                "HV": HV if param.isPareto else "NA"}

  return output, out_step


def rosen(x, *argv):
  x = np.asarray(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
        axis=0), [0]]
  return y


def test_omads_callable_quick():
  eval_bb = {"blackbox": rosen}
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

  data = {"evaluator": eval_bb, "param": param, "options": options, "sampling": sampling}

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