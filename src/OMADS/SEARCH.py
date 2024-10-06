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

import importlib
from multiprocessing import freeze_support
import os
import sys
import time
import numpy as np
import copy
from typing import List, Dict, Any, Optional
from matplotlib import pyplot as plt

if importlib.util.find_spec('BMDFO'):
  from BMDFO import toy
from .CandidatePoint import CandidatePoint
from ._common import logger, validator
from .Exploration import SAMPLING_METHOD, VNS
from .PreExploration import PreExploration
from ._globals import SEARCH_TYPE, VAR_TYPE, DESIGN_STATUS, MSG_TYPE, SUCCESS_TYPES
from .Barriers import Barrier, BarrierMO
from .Point import Point
from .Metrics import Metrics

np.set_printoptions(legacy='1.21')

def main(*args) -> Dict[str, Any]:
  """ MADS: Search step main algorithm """

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
  iteration, xmin, search, options, param, post, out, B, out_p = PreExploration(data).initialize_from_dict(log=log)

  if out_p:
    out_p.step_name = "Search_ND"

  """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))
  
  out.step_name = f"Search: {search.type}"

  """ Start the count down for calculating the runtime indicator """
  tic = time.perf_counter()

  peval = 0

  if search.type == SEARCH_TYPE.VNS.name:
    search_vn = VNS(active_barrier=B, params=param)
    search_vn._ns_dist = [int(((search.dim+1)/2)*((search.dim+2)/2)/(len(search_vn._dist))) if search.ns is None else search.ns] * len(search_vn._dist)
    search.ns = sum(search_vn._ns_dist)


  search.lb = param.lb
  search.ub = param.ub

  log.log_msg(msg=f"---------------- Run the SEARCH step ({search.sampling_t}) ----------------", msg_type=MSG_TYPE.INFO)
  num_strat: int = 0
  while True:
    bbevalold =  search.bb_handle.bb_eval
    search.mesh.update()
    search.iter = iteration
    if B is not None:
      if isinstance(B, Barrier):
        if B._filter is not None:
          B.select_poll_center()
          B.update_and_reset_success()
        else:
          B.insert(search.xmin)
      elif isinstance(B, BarrierMO) and iteration == 1:
          B.init(eval_point_list=[xmin])
    
    if isinstance(B, Barrier):
      search.hmax = B._h_max
      if xmin.status == DESIGN_STATUS.FEASIBLE:
        B.insert_feasible(search.xmin)
      elif xmin.status == DESIGN_STATUS.INFEASIBLE:
        B.insert_infeasible(search.xmin)
      else:
        B.insert(search.xmin)


    
    """ Create the set of poll directions """
    if search.type == SEARCH_TYPE.VNS.name:
      search_vn.active_barrier = B
      search._candidate_points_set = search_vn.run()
      if search_vn.stop:
        print("Reached maximum number of VNS iterations!")
        break
      search.map_samples_from_coords_to_points(samples=search._candidate_points_set)
    else:
      best_feasible: CandidatePoint = B._currentIncumbentFeas if isinstance(B, BarrierMO) else B._best_feasible
      best_inf: CandidatePoint = B._currentIncumbentInf if isinstance(B, BarrierMO) else B.get_best_infeasible()
      if best_feasible is not None and best_feasible.evaluated:
        search.xmin = best_feasible
        search.generate_sample_points(int(((search.dim+1)/2)*((search.dim+2)/2)) if search.ns is None else search.ns)
      if best_inf is not None and best_inf.evaluated:
      # if B._filter is not None and B.get_best_infeasible().evaluated:
        xmin_bup = search.xmin
        prim_samples = search._candidate_points_set
        search.xmin = best_inf
        search.generate_sample_points(int(((search.dim+1)/2)*((search.dim+2)/2)) if search.ns is None else search.ns)
        search._candidate_points_set += prim_samples
        search.xmin = xmin_bup


    """ Save current poll directions and incumbent solution
     so they can be saved later in the post dir """
    if options.save_coordinates:
      post.coords.append(search._candidate_points_set)
      post.x_incumbent.append(search.xmin)
    """ Reset success boolean """
    search.success = SUCCESS_TYPES.US
    """ Reset the BB output """
    search.bb_output = []
    xt = []
    """ Serial evaluation for points in the poll set """
    if log and log.is_verbose:
      log.log_msg(f"----------- Evaluate Search iteration # {iteration}-----------", msg_type=MSG_TYPE.INFO)
    search.log = log
    if options.check_cache:
      search.omit_duplicates()
    search.bb_handle.xmin = xmin
    if not options.parallel_mode:
      xt, post, peval = search.bb_handle.run_callable_serial_local(iter=iteration, peval=peval, eval_set=search._candidate_points_set, options=options, post=post, psize=search.mesh.getDeltaFrameSize().coordinates, step_name=f'Search: {search.type}', mesh=search.mesh, constraints_relaxation=search.constraints_RP.__dict__, budget=options.budget)
    else:
      """ Parallel evaluation for points in the samples set """
      search.point_index = -1
      search.bb_eval, xt, post, peval = search.bb_handle.run_callable_parallel_local(iter=iteration, peval=peval, eval_set=search._candidate_points_set, options=options, post=post, mesh=search.mesh, step_name=f'Search: {search.type}', psize=search.mesh.getDeltaFrameSize().coordinates, constraints_relaxation=search.constraints_RP.__dict__, budget=options.budget)
    
    search.postprocess_evaluated_candidates(xt)
    
    if iteration == 1:
      search.vicinity_ratio = np.ones((len(search.xmin.coordinates),1))
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

      

      """ Updates """
      
      if search.success == SUCCESS_TYPES.FS:
        direction: Point = Point(search.mesh._n)
        direction.coordinates = search.xmin.direction.coordinates
        search.mesh.enlargeDeltaFrameSize(direction=direction)
        if search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
          search.update_local_region(region="expand")
      elif search.success == SUCCESS_TYPES.US:
        search.mesh.refineDeltaFrameSize()
        if search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
          search.update_local_region(region="contract")
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
          if search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
            search.update_local_region(region="contract")
        if B._currentIncumbentFeas:
          B._currentIncumbentFeas.mesh.refineDeltaFrameSize()
          new_mesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if B._currentIncumbentFeas else copy.deepcopy(B._currentIncumbentInf.mesh) if B._currentIncumbentInf else None
          B.updateCurrentIncumbents()
          if search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
            search.update_local_region(region="contract")
        
        
        if new_mesh:
          search.mesh = new_mesh
        else:
          search.mesh.refineDeltaFrameSize()
          if search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
            search.update_local_region(region="contract")
      else:
        search.mesh = copy.deepcopy(B._currentIncumbentFeas.mesh) if updated_f else copy.deepcopy(B._currentIncumbentInf.mesh) if updated_inf else search.mesh
        search.xmin = copy.deepcopy(B._currentIncumbentFeas) if updated_f else copy.deepcopy(B._currentIncumbentInf) if updated_inf else search.xmin
        if search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
          search.update_local_region(region="expand")
      
      for i in range(len(xpost)):
        post.poll_dirs.append(xpost[i])
      search.hashtable.best_hash_id = []
      search.hashtable.add_to_best_cache(B.getAllPoints())
      post.xmin = B._currentIncumbentFeas if updated_f  else B._currentIncumbentInf if  updated_inf else search.xmin
      
    search.mesh.update()
    if iteration == 1:
        search.vicinity_ratio = np.ones((len(search.xmin.coordinates),1))

    if options.save_results:
      post.nd_points = []
      
      post.output_results(out=out, all_res=False)
      if param.isPareto:
        for i in range(len(B.getAllPoints())):
          post.nd_points.append(B.getAllPoints()[i])
        post.output_nd_results(out_p)
      
    if log:
      log.log_msg(msg=post.__str__(), msg_type=MSG_TYPE.INFO)
    if options.display:
      print(post)

    failure_check = iteration > 0 and search.Failure_stop is not None and search.Failure_stop and not (search.success != SUCCESS_TYPES.FS or SUCCESS_TYPES.PS)
    if search.bb_handle.bb_eval - bbevalold <= 0:
      num_strat += 1
      if num_strat > 5:
        search.exploreNew = True
        num_strat = 0
    else:
      num_strat = 0
    if (failure_check or search.bb_handle.bb_eval >= options.budget) or (all(abs(search.mesh.getdeltaMeshSize().coordinates[pp]) < options.tol  for pp in range(search.mesh._n)) or search.bb_handle.bb_eval >= options.budget or search.terminate):
      log.log_msg("\n--------------- Termination of the search step  ---------------", MSG_TYPE.INFO)
      if (all(abs(search.mesh.getdeltaMeshSize().coordinates[pp]) < options.tol  for pp in range(search.mesh._n))):
        log.log_msg("Termination criterion hit: the mesh size is below the minimum threshold defined.", MSG_TYPE.INFO)
      if (search.bb_handle.bb_eval >= options.budget or search.terminate):
        log.log_msg("Termination criterion hit: evaluation budget is exhausted.", MSG_TYPE.INFO)
      if (failure_check):
        log.log_msg("Termination criterion hit (optional): failed to find a successful point in iteration # {iteration}.", MSG_TYPE.INFO)
      log.log_msg("-----------------------------------------------------------------\n", MSG_TYPE.INFO)
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
      ncon = len(search.xmin.c_eq) + len(search.xmin.c_ineq)
    if len(search.bb_output) > 0:
      b.add_row(name=search.bb_handle.blackbox,
            run_index=int(args[2]),
            nv=len(param.baseline),
            nc=ncon,
            nb_success=search.nb_success,
            it=iteration,
            BBEVAL=search.bb_eval,
            runtime=toc - tic,
            feval=search.bb_handle.bb_eval,
            hmin=search.xmin.h,
            fmin=search.xmin.f)
    print(f"{search.bb_handle.blackbox}: fmin = {search.xmin.f} , hmin= {search.xmin.h:.2f}")

  elif importlib.util.find_spec('BMDFO') and len(args) > 1 and not isinstance(args[1], toy.Run):
    if log:
      log.log_msg(msg="Could not find " + args[1] + " in the internal BM suite.", msg_type=MSG_TYPE.ERROR)
    raise IOError("Could not find " + args[1] + " in the internal BM suite.")

  

  if options.display:
    if log:
      log.log_msg(" end of orthogonal MADS ", MSG_TYPE.INFO)
    print(" end of orthogonal MADS ")
    if log:
      log.log_msg(" Final objective value: " + str(search.xmin.f) + ", hmin= " + str(search.xmin.h), MSG_TYPE.INFO)
    print(" Final objective value: " + str(search.xmin.f) + ", hmin= " + str(search.xmin.h))

  if options.save_coordinates:
    post.output_coordinates(out)
  
  if log:
    log.log_msg("\n ---Run Summary---", MSG_TYPE.INFO)
    log.log_msg(f" Run completed in {toc - tic:.4f} seconds", MSG_TYPE.INFO)
    log.log_msg(msg=f" # of successful search steps = {search.n_successes}", msg_type=MSG_TYPE.INFO)
    log.log_msg(f" Random numbers generator's seed {options.seed}", MSG_TYPE.INFO)
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
    print(" mesh_size = " + str(search.mesh.getDeltaFrameSize().coordinates))
  xmin = search.xmin
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
                "fmin": search.xmin.f,
                "hmin": search.xmin.h,
                "nbb_evals" : search.bb_eval,
                "niterations" : iteration,
                "nb_success": search.nb_success,
                "psize": search.mesh.getDeltaFrameSize().coordinates,
                "psuccess": search.xmin.mesh.getDeltaFrameSize().coordinates,
                # "pmax": search.mesh.psize_max,
                "msize": search.mesh.getdeltaMeshSize().coordinates,
                "HV": HV if param.isPareto else "NA"}

  return output, search

def rosen(x, *argv):
  x = np.asarray(x)
  y = [np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0,
        axis=0), [0]]
  return y




def test_omads_callable_quick():
  eval_func = {"blackbox": rosen}
  param = {"baseline": [-2.0, -2.0],
       "lb": [-5, -5],
       "ub": [10, 10],
       "var_names": ["x1", "x2"],
       "scaling": 15.0,
       "post_dir": "./post",
       "Failure_stop": False}
  sampling = {
    "method": SAMPLING_METHOD.LH.value,
    "ns": 10,
    "visualize": True
  }
  options = {"seed": 0, "budget": 100000, "tol": 1e-12, "display": True}

  data = {"evaluator": eval_func, "param": param, "options": options, "sampling": sampling}

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