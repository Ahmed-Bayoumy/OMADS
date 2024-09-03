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
import json
from multiprocessing import freeze_support
import os
import sys
import time
import numpy as np
import copy
from typing import List, Dict, Any
import concurrent.futures
from matplotlib import pyplot as plt

if importlib.util.find_spec('BMDFO'):
  from BMDFO import toy
from .CandidatePoint import CandidatePoint
from ._common import *
from .Directions import *
from .Exploration import *
from .PreExploration import *
np.set_printoptions(legacy='1.21')

def main(*args) -> Dict[str, Any]:
  """ MADS: Search step main algorithm """

  """ Validate and parse the parameters file """
  validate = validator()
  data: dict = validate.checkInputFile(args=args)

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
  iteration, xmin, search, options, param, post, out, B, outP = PreExploration(data).initialize_from_dict(log=log)

  if outP:
    outP.stepName = "Search_ND"

  """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))
  
  out.stepName = f"Search: {search.type}"
  

  """ Initialize the visualization figure"""
  if search.visualize:
    plt.ion()
    fig = plt.figure()
    ax=[]
    nplots = len(param.var_names)-1
    ps = [None]*nplots**2
    for ii in range(nplots**2):
      ax.append(fig.add_subplot(nplots, nplots, ii+1))

  """ Start the count down for calculating the runtime indicator """
  tic = time.perf_counter()

  peval = 0

  if search.type == SEARCH_TYPE.VNS.name:
    search_VN = VNS(active_barrier=B, params=param)
    search_VN._ns_dist = [int(((search.dim+1)/2)*((search.dim+2)/2)/(len(search_VN._dist))) if search.ns is None else search.ns] * len(search_VN._dist)
    search.ns = sum(search_VN._ns_dist)


  search.lb = param.lb
  search.ub = param.ub

  LAMBDA_k = xmin.LAMBDA
  RHO_k = xmin.RHO

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
          B.init(evalPointList=[xmin])
    
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
      search_VN.active_barrier = B
      search._candidate_points_set = search_VN.run()
      if search_VN.stop:
        print("Reached maximum number of VNS iterations!")
        break
      vv = search.map_samples_from_coords_to_points(samples=search._candidate_points_set)
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
        search.xmin = bestInf
        vvs, _ = search.generate_sample_points(int(((search.dim+1)/2)*((search.dim+2)/2)) if search.ns is None else search.ns)
        search._candidate_points_set += Prim_samples
        search.xmin = xmin_bup
      
      if isinstance(vvs, list) and len(vvs) > 0:
        vv = vvp + vvs
      else:
        vv = vvp

    if search.visualize:
      sc_old = search.store_cache
      cc_old = search.check_cache
      search.check_cache = False
      search.store_cache = False
      for iii in range(len(ax)):
        for jjj in range(len(xmin.coordinates)):
          for kkk in range(jjj, len(xmin.coordinates)):
            if kkk != jjj:
              if all([psi is None for psi in ps]):
                xinput = [search.xmin]
              else:
                xinput = search._candidate_points_set
              ps = visualize(xinput, jjj, kkk, search.mesh.getdeltaMeshSize().coordinates, vv, fig, ax, search.xmin, ps, bbeval=search.bb_handle, lb=search.prob_params.lb, ub=search.prob_params.ub, spindex=iii, bestKnown=search.prob_params.best_known, blk=False)
      search.store_cache = sc_old
      search.check_cache = cc_old


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
    if log is not None and log.isVerbose:
      log.log_msg(f"----------- Evaluate Search iteration # {iteration}-----------", msg_type=MSG_TYPE.INFO)
    search.log = log
    if options.check_cache:
      search.omit_duplicates()
    search.bb_handle.xmin = xmin
    if not options.parallel_mode:
      xt, post, peval = search.bb_handle.run_callable_serial_local(iter=iteration, peval=peval, eval_set=search._candidate_points_set, options=options, post=post, psize=search.mesh.getDeltaFrameSize().coordinates, stepName=f'Search: {search.type}', mesh=search.mesh, constraintsRelaxation=search.constraints_RP.__dict__, budget=options.budget)
    else:
      """ Parallel evaluation for points in the samples set """
      search.point_index = -1
      search.bb_eval, xt, post, peval = search.bb_handle.run_callable_parallel_local(iter=iteration, peval=peval, njobs=options.np, eval_set=search._candidate_points_set, options=options, post=post, mesh=search.mesh, stepName=f'Search: {search.type}', psize=search.mesh.getDeltaFrameSize().coordinates, constraintsRelaxation=search.constraints_RP.__dict__, budget=options.budget)
    
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
    if iteration == 1:
        search.vicinity_ratio = np.ones((len(search.xmin.coordinates),1))

    if options.save_results:
      post.nd_points = []
      
      post.output_results(out=out, allRes=False)
      if param.isPareto:
        for i in range(len(B.getAllPoints())):
          post.nd_points.append(B.getAllPoints()[i])
        post.output_nd_results(outP)
      
    if log is not None:
      log.log_msg(msg=post.__str__(), msg_type=MSG_TYPE.INFO)
    if options.display:
      print(post)

    Failure_check = iteration > 0 and search.Failure_stop is not None and search.Failure_stop and not (search.success != SUCCESS_TYPES.FS or SUCCESS_TYPES.PS)
    if search.bb_handle.bb_eval - bbevalold <= 0:
      num_strat += 1
      if num_strat > 5:
        search.terminate = True
    if (Failure_check or search.bb_handle.bb_eval >= options.budget) or (all(abs(search.mesh.getdeltaMeshSize().coordinates[pp]) < options.tol  for pp in range(search.mesh._n)) or search.bb_handle.bb_eval >= options.budget or search.terminate):
      log.log_msg(f"\n--------------- Termination of the search step  ---------------", MSG_TYPE.INFO)
      if (all(abs(search.mesh.getdeltaMeshSize().coordinates[pp]) < options.tol  for pp in range(search.mesh._n))):
        log.log_msg("Termination criterion hit: the mesh size is below the minimum threshold defined.", MSG_TYPE.INFO)
      if (search.bb_handle.bb_eval >= options.budget or search.terminate):
        log.log_msg("Termination criterion hit: evaluation budget is exhausted.", MSG_TYPE.INFO)
      if (Failure_check):
        log.log_msg(f"Termination criterion hit (optional): failed to find a successful point in iteration # {iteration}.", MSG_TYPE.INFO)
      log.log_msg(f"-----------------------------------------------------------------\n", MSG_TYPE.INFO)
      break
    iteration += 1

  toc = time.perf_counter()

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
    if log is not None:
      log.log_msg(msg="Could not find " + args[1] + " in the internal BM suite.", msg_type=MSG_TYPE.ERROR)
    raise IOError("Could not find " + args[1] + " in the internal BM suite.")

  

  if options.display:
    if log is not None:
      log.log_msg(" end of orthogonal MADS ", MSG_TYPE.INFO)
    print(" end of orthogonal MADS ")
    if log is not None:
      log.log_msg(" Final objective value: " + str(search.xmin.f) + ", hmin= " + str(search.xmin.h), MSG_TYPE.INFO)
    print(" Final objective value: " + str(search.xmin.f) + ", hmin= " + str(search.xmin.h))

  if options.save_coordinates:
    post.output_coordinates(out)
  
  if log is not None:
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
                "msize": search.mesh.getdeltaMeshSize().coordinates}
  
  if search.visualize:
    sc_old = search.store_cache
    cc_old = search.check_cache
    search.check_cache = False
    search.store_cache = False
    temp = CandidatePoint()
    temp.coordinates = output["xmin"]
    for ii in range(len(ax)):
      for jj in range(len(xmin.coordinates)):
          for kk in range(jj+1, len(xmin.coordinates)):
            if kk != jj:
              ps = visualize(xinput, jj, kk, search.mesh.getdeltaMeshSize().coordinates, vv, fig, ax, temp, ps, bbeval=search.bb_handle, lb=search.prob_params.lb, ub=search.prob_params.ub, title=search.prob_params.problem_name, blk=True,vnames=search.prob_params.var_names, spindex=ii, bestKnown=search.prob_params.best_known)
    search.check_cache = sc_old
    search.store_cache = cc_old

  return output, search



def visualize(points: List[CandidatePoint], hc_index, vc_index, msize, vlim, fig, axes, pmin, ps = None, title="unknown", blk=False, vnames=None, bbeval=None, lb = None, ub=None, spindex=0, bestKnown=None):

  x: np.ndarray = np.zeros(len(points))
  y: np.ndarray = np.zeros(len(points))

  for i in range(len(points)):
    x[i] = points[i].coordinates[hc_index]
    y[i] = points[i].coordinates[vc_index]
  xmin = pmin.coordinates[hc_index]
  ymin = pmin.coordinates[vc_index]
  
  # Plot grid's dynamic updates
  # nrx = int((vlim[hc_index, 1] - vlim[hc_index, 0])/msize)
  # nry = int((vlim[vc_index, 1] - vlim[vc_index, 0])/msize)
  
  # minor_ticksx=np.linspace(vlim[hc_index, 0],vlim[hc_index, 1],nrx+1)
  # minor_ticksy=np.linspace(vlim[vc_index, 0],vlim[vc_index, 1],nry+1)
  isFirst = False

  if ps[spindex] == None:
    isFirst = True
    ps[spindex] =[]
    if bbeval is not None and lb is not None and ub is not None:
      xx = np.arange(lb[hc_index], ub[hc_index], 0.1)
      yy = np.arange(lb[vc_index], ub[vc_index], 0.1)
      X, Y = np.meshgrid(xx, yy)
      Z = np.zeros_like(X)
      for i in range(X.shape[0]):
        for j in range(X.shape[1]):
          Z[i,j] = bbeval.eval([X[i,j], Y[i,j]])[0]
          bbeval.bb_eval -= 1
    if bestKnown is not None:
      best_index = np.argwhere(Z <= bestKnown+0.005)
      if best_index.size == 0:
        best_index = np.argwhere(Z == np.min(Z))
      xbk = X[best_index[0][0], best_index[0][1]]
      ybk = Y[best_index[0][0], best_index[0][1]]
    temp1 = axes[spindex].contourf(X, Y, Z, 100)
    axes[spindex].set_aspect('equal')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.01, 0.85])
    fig.colorbar(temp1, cbar_ax)
    fig.suptitle(title)

    ps[spindex].append(temp1)

    

    temp2, = axes[spindex].plot(xmin, ymin, 'ok', alpha=0.08, markersize=2)

    ps[spindex].append(temp2)

    if bestKnown is not None:
      temp3, = axes[spindex].plot(xbk, ybk, 'dr', markersize=2)
      ps[spindex].append(temp3)
    

    
  else:
    ps[spindex][1].set_xdata(x)
    ps[spindex][1].set_ydata(y)
  
  

  fig.canvas.draw()
  fig.canvas.flush_events()
  # axes.set_xticks(minor_ticksx,major=True)
  # axes.set_yticks(minor_ticksy,major=True)

  # axes.grid(which="major",alpha=0.3)
  # ps[1].set_xdata(x)
  # ps[1].set_ydata(y)
  if blk:
    if bestKnown is not None:
      t1 = ps[spindex][2]
    t2, =axes[spindex].plot(x, y, 'ok', alpha=0.08, markersize=2)
    

    t3, = axes[spindex].plot(xmin, ymin, '*b', markersize=4)
    if bestKnown is not None:
      fig.legend((t1, t2, t3), ("best_known", "sample_points", "best_found"))
    else:
      fig.legend((t2, t3), ("sample_points", "best_found"))
  else:
    axes[spindex].plot(x, y, 'ok', alpha=0.08, markersize=2)
    # axes[spindex].plot(xmin, ymin, '*b', markersize=4)
  if vnames is not None:
    axes[spindex].set_xlabel(vnames[hc_index])
    axes[spindex].set_ylabel(vnames[vc_index])
  if lb is not None and ub is not None:
    axes[spindex].set_xlim([lb[hc_index], ub[hc_index]])
    axes[spindex].set_ylim([lb[vc_index], ub[vc_index]])
  plt.show(block=blk)

  if blk:
    fig.savefig(f"{title}.png", bbox_inches='tight')
    plt.close(fig)
  
  return ps
  


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
       "Failure_stop": False}
  sampling = {
    "method": SAMPLING_METHOD.LH.value,
    "ns": 10,
    "visualize": True
  }
  options = {"seed": 0, "budget": 100000, "tol": 1e-12, "display": True}

  data = {"evaluator": eval, "param": param, "options": options, "sampling": sampling}

  out: Dict = main(data)
  print(out)

def test_omads_file_quick():
  file = "tests\\bm\\constrained\\sphere.json"

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