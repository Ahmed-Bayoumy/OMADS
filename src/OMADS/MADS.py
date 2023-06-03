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
import json
from multiprocessing import freeze_support
import os
import sys
import OMADS.POLL as PS
import OMADS.SEARCH as SS
from typing import List, Dict, Any, Callable, Protocol, Optional
import numpy as np
from BMDFO import toy

def search_step(iteration: int, search: SS.efficient_exploration = None, B: SS.Barrier = None, LAMBDA_k: float=None, RHO_k: float=None, search_VN: SS.VNS = None, post: PS.PostMADS=None, out: PS.Output=None, options: PS.Options=None, xmin: SS.Point=None, peval: int=0, HT: Any=None):
  search.xmin = xmin
  search.mesh.update()
  search.LAMBDA = LAMBDA_k
  search.RHO = RHO_k
  if HT is not None:
    search.hashtable = HT
  if B is not None:
    B.insert(search.xmin)
    if B._filter is not None:
      B.select_poll_center()
      B.update_and_reset_success()
      
  
  search.hmax = B._h_max
  
  """ Create the set of poll directions """
  if search.type == SS.SEARCH_TYPE.VNS.name and search_VN is not None:
    search_VN.active_barrier = B
    search.samples = search_VN.run()
    if search_VN.stop:
      print("Reached maximum number of VNS iterations!")
      return search, B, post, out, search.LAMBDA, search.RHO, search.xmin, peval
    search.map_samples_from_coords_to_points(samples=search.samples)
  else:
    if B._best_feasible is not None and B._best_feasible.evaluated:
      vvp, _ = search.generate_sample_points(int(((search.dim+1)/2)*((search.dim+2)/2)) if search.ns is None else search.ns)
    if B._filter is not None and B.get_best_infeasible().evaluated:
      xmin_bup = search.xmin
      Prim_samples = search.samples
      search.xmin = B.get_best_infeasible()
      search.generate_sample_points(int(((search.dim+1)/2)*((search.dim+2)/2)) if search.ns is None else search.ns)
      search.samples += Prim_samples
      search.xmin = xmin_bup


  """ Save current poll directions and incumbent solution
    so they can be saved later in the post dir """
  if options.save_coordinates:
    post.coords.append(search.samples)
    post.x_incumbent.append(search.xmin)
  """ Reset success boolean """
  search.success = False
  """ Reset the BB output """
  search.bb_output = []
  xt = []
  """ Serial evaluation for points in the poll set """
  if search_VN is not None:
    search.lb = search_VN.params.lb
    search.ub = search_VN.params.ub
  
  if not options.parallel_mode:
    for it in range(len(search.samples)):
      if search.terminate:
        break
      f = search.evaluate_sample_point(it)
      xt.append(f[-1])
      if not f[0]:
        post.bb_eval.append(search.bb_handle.bb_eval)
        post.step_name.append(f'Search: {search.type}')
        peval += 1
        post.iter.append(iteration)
        post.psize.append(search.mesh.psize)
      else:
        continue

  else:
    search.point_index = -1
    """ Parallel evaluation for points in the poll set """
    with SS.concurrent.futures.ProcessPoolExecutor(options.np) as executor:
      results = [executor.submit(search.evaluate_sample_point,
                      it) for it in range(len(search.samples))]
      for f in SS.concurrent.futures.as_completed(results):
        if options.save_results or options.display:
          peval = peval +1
          search.bb_eval = peval
          post.bb_eval.append(peval)
          post.step_name.append(f'Search: {search.type}')
          post.iter.append(iteration)
          post.psize.append(f.result()[4])
        xt.append(f.result()[-1])

  xpost: List[SS.Point] = search.master_updates(xt, peval, save_all_best=options.save_all_best, save_all=options.save_results)
  if options.save_results:
    for i in range(len(xpost)):
      post.poll_dirs.append(xpost[i])
  xv: SS.Point = None
  for xv in xt:
    if xv.evaluated:
      B.insert(xv)

  """ Update the xmin in post"""
  post.xmin = copy.deepcopy(search.xmin)

  if iteration == 1:
    search.vicinity_ratio = np.ones((len(search.xmin.coordinates),1))

  """ Updates """
  if search.success:
    search.mesh.psize = search.mesh.msize = np.multiply(search.mesh.msize, 2, dtype=search.dtype.dtype)
    search.update_local_region(region="expand")
  else:
    search.mesh.psize = search.mesh.msize = np.divide(search.mesh.msize, 2, dtype=search.dtype.dtype)
    search.update_local_region(region="contract")

  if options.display:
    print(post)

  # Failure_check = iteration > 0 and search.Failure_stop is not None and search.Failure_stop and not search.success
  # if (Failure_check) or (abs(search.mesh.msize) < options.tol or search.bb_eval >= options.budget or search.terminate):
  #   break
  # iteration += 1
  return search, B, post, out, search.LAMBDA, search.RHO, search.xmin, peval

def poll_step(iteration: int, poll: PS.Dirs2n = None, B: SS.Barrier = None, LAMBDA_k: float=None, RHO_k: float=None, param: PS.Parameters=None, post: PS.PostMADS=None, xmin: PS.Point=None, out: PS.Output=None, options: PS.Options=None, peval: int = 0, HT: Any = None):
  poll.xmin = xmin
  poll.mesh.update()
  """ Create the set of poll directions """
  hhm = poll.create_housholder(options.rich_direction, domain=xmin.var_type)
  poll.lb = param.lb
  poll.ub = param.ub
  poll.xmin = copy.deepcopy(xmin)
  if HT is not None:
    poll.hashtable = HT
  B.insert(xmin)
  if B is not None:
    if B._filter is not None:
      B.select_poll_center()
      B.update_and_reset_success()
      
  poll.hmax = B._h_max
  poll.create_poll_set(hhm=hhm,
              ub=param.ub,
              lb=param.lb, it=iteration, var_type=xmin.var_type, var_sets=xmin.sets, var_link = xmin.var_link, c_types=param.constraints_type, is_prim=True)
  
  if B._sec_poll_center is not None and B._sec_poll_center.evaluated:
    poll.x_sc = B._sec_poll_center
    poll.create_poll_set(hhm=hhm,
              ub=param.ub,
              lb=param.lb, it=iteration, var_type=B._sec_poll_center.var_type, var_sets=B._sec_poll_center.sets, var_link = B._sec_poll_center.var_link, c_types=param.constraints_type, is_prim=False)
  
  poll.LAMBDA = LAMBDA_k
  poll.RHO = RHO_k

  """ Save current poll directions and incumbent solution
    so they can be saved later in the post dir """
  if options.save_coordinates:
    post.coords.append(poll.poll_dirs)
    post.x_incumbent.append(poll.xmin)
  """ Reset success boolean """
  poll.success = False
  """ Reset the BB output """
  poll.bb_output = []
  xt = []
  """ Serial evaluation for points in the poll set """
  if not options.parallel_mode:
    for it in range(len(poll.poll_dirs)):
      peval += 1
      if poll.terminate:
        break
      f = poll.eval_poll_point(it)
      xt.append(f[-1])
      if not f[0]:
        post.bb_eval.append(poll.bb_handle.bb_eval)
        post.step_name.append(f'Poll-2n')
        post.iter.append(iteration)
        post.psize.append(poll.mesh.psize)
      else:
        continue

  else:
    poll.point_index = -1
    """ Parallel evaluation for points in the poll set """
    with PS.concurrent.futures.ProcessPoolExecutor(options.np) as executor:
      results = [executor.submit(poll.eval_poll_point,
                      it) for it in range(len(poll.poll_dirs))]
      for f in PS.concurrent.futures.as_completed(results):
        # if f.result()[0]:
        #     executor.shutdown(wait=False)
        # else:
        if options.save_results or options.display:
          peval = peval +1
          poll.bb_eval = peval
          post.bb_eval.append(peval)
          post.step_name.append(f'Poll-2n')
          post.iter.append(iteration)
          # post.poll_dirs.append(poll.poll_dirs[f.result()[1]])
          post.psize.append(f.result()[4])
        xt.append(f.result()[-1])

  xpost: List[PS.Point] = poll.master_updates(xt, peval, save_all_best=options.save_all_best, save_all=options.save_results)
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
  for p in poll.poll_dirs:
    if p.evaluated:
      pev += 1
  # if pev != poll.poll_dirs and not poll.success:
  #   poll.seed += 1
  goToSearch: bool = (pev == 0 and poll.Failure_stop is not None and poll.Failure_stop)
  if poll.success and not goToSearch:
    poll.mesh.psize = np.multiply(poll.mesh.psize, 2, dtype=poll.dtype.dtype)
  else:
    poll.mesh.psize = np.divide(poll.mesh.psize, 2, dtype=poll.dtype.dtype)

  if options.display:
    print(post)
  
  LAMBDA_k = poll.LAMBDA
  RHO_k = poll.RHO
  
  return poll, B, post, out, poll.xmin.LAMBDA, poll.xmin.RHO, poll.xmin, peval

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

  """ Run preprocessor for the setup of
   the optimization problem and for the initialization
  of optimization process """
  iteration: int
  xmin: PS.Point
  options: PS.Options
  param: PS.Parameters 
  post: PS.PostMADS 
  out: PS.Output 
  B: PS.Barrier
  poll: PS.Dirs2n
  search: SS.efficient_exploration
  _, _, search, _, _, _, _, _ = SS.PreExploration(data).initialize_from_dict()
  iteration, xmin, poll, options, param, post, out, B = PS.PreMADS(data).initialize_from_dict(xs=search.xmin)
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
    if not poll.success or iteration == 1:
      search, B, post, out, LAMBDA_k, RHO_k, xmin, peval = search_step(search=search, B=B, LAMBDA_k=LAMBDA_k, RHO_k=RHO_k, iteration=iteration , search_VN=search_VN, post=post, out=out, options=options, xmin=xmin, peval=peval, HT=HT)
    """ Run the poll step (Mandatory step) """
    HT = search.hashtable
    poll, B, post, out, LAMBDA_k, RHO_k, xmin, peval = poll_step(iteration=iteration, poll=poll, B=B, LAMBDA_k=LAMBDA_k, RHO_k=RHO_k, param=param, post=post, xmin=xmin, out=out, options=options, peval=peval, HT=HT)
    HT = poll.hashtable
    xmin = poll.xmin
    search.mesh = copy.deepcopy(poll.mesh)
    search.psize = copy.deepcopy(poll.psize)
    """ Check stopping criteria"""
    pt = (abs(poll.mesh.psize) < options.tol)
    st = (abs(search.mesh.psize) < options.tol)
    if (pt or st or search.bb_eval+poll.bb_eval >= options.budget):
      break
    iteration += 1
    

  toc = PS.time.perf_counter()

  """ If benchmarking, then populate the results in the benchmarking output report """
  if len(args) > 1 and isinstance(args[1], PS.toy.Run):
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
    print(f"{poll.bb_handle.blackbox}: fmin = {poll.xmin.f:.2f} , hmin= {poll.xmin.h:.2f}")

  elif len(args) > 1 and not isinstance(args[1], toy.Run):
    raise IOError("Could not find " + args[1] + " in the internal BM suite.")

  if options.save_results:
    post.output_results(out)
  
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
  if options.display:
    print("\n ---Run Summary---")
    print(f" Run completed in {toc - tic:.4f} seconds")
    print(f" Random numbers generator's seed {options.seed}")
    print(" xmin = " + str(out_step.xmin))
    print(" hmin = " + str(out_step.xmin.h))
    print(" fmin = " + str(out_step.xmin.f))
    print(" #bb_eval = " + str(out_step.bb_eval))
    print(" #iteration = " + str(iteration))
    print(" nb_success = " + str(poll.nb_success + search.nb_success))
    print(" psize = " + str(poll.mesh.psize))
    print(" psize_success = " + str(poll.mesh.psize_success))
    print(" psize_max = " + str(poll.mesh.psize_max))
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
                "psize": poll.mesh.psize,
                "psuccess": poll.mesh.psize_success,
                "pmax": poll.mesh.psize_max,
                "msize": out_step.mesh.msize}

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