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
from multiprocessing import freeze_support, cpu_count
import os
import sys
import numpy as np
import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
from BMDFO import toy
from numpy import subtract, maximum
from .Point import Point
from .Barriers import *
from ._common import *
from .Directions import *

@dataclass
class PreMADS:
  """ Preprocessor for setting up optimization settings and parameters"""
  data: Dict[Any, Any]
  log: logger = None

  def initialize_from_dict(self, log: logger = None, xs: Point=None):
    """ MADS initialization """
    """ 1- Construct the following classes by unpacking
     their respective dictionaries from the input JSON file """
    self.log = copy.deepcopy(log)
    if self.log is not None:
      self.log.log_msg(msg="---------------- Preprocess the POLL step ----------------", msg_type=MSG_TYPE.INFO)
      self.log.log_msg(msg="- Reading the input dictionaries", msg_type=MSG_TYPE.INFO)
    options = Options(**self.data["options"])
    param = Parameters(**self.data["param"])
    log.isVerbose = options.isVerbose
    B = Barrier(param)
    ev = Evaluator(**self.data["evaluator"])
    if self.log is not None:
      self.log.log_msg(msg="- Set the POLL configurations", msg_type=MSG_TYPE.INFO)
    ev.dtype.precision = options.precision
    if param.constants != None:
      ev.constants = copy.deepcopy(param.constants)
    
    if param.constraints_type is not None and isinstance(param.constraints_type, list):
      for i in range(len(param.constraints_type)):
        if param.constraints_type[i] == BARRIER_TYPES.PB.name or param.constraints_type[i] == BARRIER_TYPES.PB:
          param.constraints_type[i] = BARRIER_TYPES.PB
        elif param.constraints_type[i] == BARRIER_TYPES.RB.name or param.constraints_type[i] == BARRIER_TYPES.RB:
          param.constraints_type[i] = BARRIER_TYPES.RB
        elif param.constraints_type[i] == BARRIER_TYPES.PEB.name or param.constraints_type[i] == BARRIER_TYPES.PEB:
          param.constraints_type[i] = BARRIER_TYPES.PEB
        else:
          param.constraints_type[i] = BARRIER_TYPES.EB
    elif param.constraints_type is not None:
      param.constraints_type = BARRIER_TYPES(param.constraints_type)
  
    iteration: int =  0
    """ 2- Initialize iteration number and construct a point instant for the starting point """
    extend = options.extend is not None and isinstance(options.extend, Dirs2n)
    is_xs = False
    if xs is None or not isinstance(xs, Point) or not xs.evaluated:
      x_start = Point()
    else:
      x_start = xs
      is_xs = True

    if not extend:
      """ 3- Construct an instant for the poll 2n orthogonal directions class object """
      poll = Dirs2n()
      if param.Failure_stop != None and isinstance(param.Failure_stop, bool):
        poll.Failure_stop = param.Failure_stop
      poll.dtype.precision = options.precision
      """ 4- Construct an instant for the mesh subclass object by inheriting
      initial parameters from mesh_params() """
      poll.mesh = OrthoMesh()
      """ 5- Assign optional algorithmic parameters to the constructed poll instant  """
      poll.opportunistic = options.opportunistic
      poll.seed = options.seed
      poll.mesh.dtype.precision = options.precision
      poll.mesh.psize = options.psize_init
      poll.eval_budget = options.budget
      poll.store_cache = options.store_cache
      poll.check_cache = options.check_cache
      poll.display = options.display
      poll.scaling
    else:
      poll = options.extend
    
    n_available_cores = cpu_count()
    if options.parallel_mode and options.np > n_available_cores:
      options.np == n_available_cores
    """ 6- Initialize blackbox handling subclass by copying
     the evaluator 'ev' instance to the poll object"""
    poll.bb_handle = ev
    poll.bb_handle.bb_eval = ev.bb_eval
    """ 7- Evaluate the starting point """
    if options.display:
      print(" Evaluation of the starting points")
      if self.log is not None:
        self.log.log_msg(msg="- Evaluate the starting point", msg_type=MSG_TYPE.INFO)
    if not is_xs:
      x_start.coordinates = param.baseline
      x_start.sets = param.var_sets
      if param.constraints_type is not None and isinstance(param.constraints_type, list):
        x_start.constraints_type = [xb for xb in param.constraints_type]
      elif param.constraints_type is not None:
        x_start.constraints_type = [param.constraints_type]
    
    """ 8- Set the variables type """
    if param.var_type is not None:
      c= 0
      x_start.var_type = []
      x_start.var_link = []
      for k in param.var_type:
        c+= 1
        if k.lower()[0] == "r":
          x_start.var_type.append(VAR_TYPE.CONTINUOUS)
          x_start.var_link.append(None)
        elif k.lower()[0] == "i":
          x_start.var_type.append(VAR_TYPE.INTEGER)
          x_start.var_link.append(None)
        elif k.lower()[0] == "d":
          x_start.var_type.append(VAR_TYPE.DISCRETE)
          if x_start.sets is not None and isinstance(x_start.sets, dict):
            if x_start.sets[k.split('_')[1]] is not None:
              x_start.var_link.append(k.split('_')[1])
              if param.ub[c-1] > len(x_start.sets[k.split('_')[1]])-1:
                param.ub[c-1] = len(x_start.sets[k.split('_')[1]])-1
              if param.lb[c-1] < 0:
                param.lb[c-1] = 0
            else:
              x_start.var_link.append(None)
        elif k.lower()[0] == "c":
          x_start.var_type.append(VAR_TYPE.CATEGORICAL)
          if x_start.sets is not None and isinstance(x_start.sets, dict):
            if x_start.sets[k.split('_')[1:][0]] is not None:
              x_start.var_link.append(k.split('_')[1])
            else:
              x_start.var_link.append(None)
        elif k.lower()[0] == "o":
          x_start.var_type.append(VAR_TYPE.ORDINAL)
          x_start.var_link.append(None)
          # TODO: Implementation in progress
        elif k.lower()[0] == "b":
          x_start.var_type.append(VAR_TYPE.BINARY)
        else:
          x_start.var_type.append(VAR_TYPE.CONTINUOUS)
          x_start.var_link.append(None)

    
    x_start.dtype.precision = options.precision
    if x_start.sets is not None and isinstance(x_start.sets,dict):
      p: List[Any] = []
      for i in range(len(x_start.var_type)):
        if (x_start.var_type[i] == VAR_TYPE.DISCRETE or x_start.var_type[i] == VAR_TYPE.CATEGORICAL) and x_start.var_link[i] is not None:
          p.append(x_start.sets[x_start.var_link[i]][int(x_start.coordinates[i])])
        else:
          p.append(x_start.coordinates[i])
      if not is_xs:
        poll.bb_output = poll.bb_handle.eval(p)
    else:
       if not is_xs:
        poll.bb_output = poll.bb_handle.eval(x_start.coordinates)
    x_start.hmax = B._h_max
    x_start.RHO = param.RHO
    if param.LAMBDA is None:
      param.LAMBDA = [0]
    if not isinstance(param.LAMBDA, list):
      param.LAMBDA = [param.LAMBDA]
    x_start.LAMBDA = param.LAMBDA
    
    x_start.LAMBDA = param.LAMBDA
    if not is_xs:
      x_start.__eval__(poll.bb_output)
      B._h_max = x_start.hmax
    """ 9- Copy the starting point object to the poll's  minimizer subclass """
    if not extend:
      poll.xmin = copy.deepcopy(x_start)
    """ 10- Hold the starting point in the poll
     directions subclass and define problem parameters"""
    poll.poll_dirs.append(x_start)
    poll.scale(ub=param.ub, lb=param.lb, factor=param.scaling)
    poll.dim = x_start.n_dimensions
    if not extend:
      poll.hashtable = Cache()
    """ 10- Initialize the number of successful points
     found and check if the starting minimizer performs better
    than the worst (f = inf) """
    poll.nb_success = 0
    if not extend and poll.xmin < Point():
      poll.mesh.psize_success = poll.mesh.psize
      poll.mesh.psize_max = maximum(poll.mesh.psize,
                      poll.mesh.psize_max,
                      dtype=poll.dtype.dtype)
      poll.poll_dirs = [poll.xmin]
    elif extend and x_start < poll.xmin:
      poll.xmin = copy.deepcopy(x_start)
      poll.mesh.psize = np.multiply(poll.mesh.psize, 2, dtype=poll.dtype.dtype)
    elif extend and x_start >= poll.xmin:
      poll.mesh.psize = np.divide(poll.mesh.psize, 2, dtype=poll.dtype.dtype)


    """ 11- Construct the results postprocessor class object 'post' """
    post = PostMADS(x_incumbent=[poll.xmin], xmin=poll.xmin, poll_dirs=[poll.xmin])
    post.psize.append(poll.mesh.psize)
    post.bb_eval.append(poll.bb_handle.bb_eval)
    post.iter.append(iteration)

    """ Note: printing the post will print a results row
     within the results table shown in Python console if the
    'display' option is true """
    # if options.display:
    #     print(post)
    """ 12- Add the starting point hash value to the cache memory """
    if options.store_cache:
      poll.hashtable.hash_id = x_start
    """ 13- Initialize the output results file object  """
    out = Output(file_path=param.post_dir, vnames=param.var_names, pname=param.name, runfolder=f'{param.name}_run')
    if options.display:
      print("End of the evaluation of the starting points")
      if self.log is not None:
        self.log.log_msg(msg="- End of the evaluation of the starting points", msg_type=MSG_TYPE.INFO)

    iteration += 1

    return iteration, x_start, poll, options, param, post, out, B

def main(*args) -> Dict[str, Any]:
  """ Otho-MADS main algorithm """
  # TODO: add more checks for more defensive code

  

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
     os.mkdir(data["param"]["post_dir"])
  log.initialize(data["param"]["post_dir"] + "/OMADS.log")

  """ Run preprocessor for the setup of
   the optimization problem and for the initialization
  of optimization process """
  iteration, xmin, poll, options, param, post, out, B = PreMADS(data).initialize_from_dict(log=log)
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
    del poll.poll_dirs
    poll.poll_dirs = []
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
      del poll.poll_dirs
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
      post.coords.append(poll.poll_dirs)
      post.x_incumbent.append(poll.xmin)
    """ Reset success boolean """
    poll.success = False
    """ Reset the BB output """
    poll.bb_output = []
    xt = []
    """ Serial evaluation for points in the poll set """
    if log is not None and log.isVerbose:
      log.log_msg(f"----------- Evaluate poll set # {iteration}-----------", msg_type=MSG_TYPE.INFO)
    poll.log = log
    if not options.parallel_mode:
      for it in range(len(poll.poll_dirs)):
        peval += 1
        if poll.terminate:
          break
        f = poll.eval_poll_point(it)
        xt.append(f[-1])
        if not f[0]:
          post.bb_eval.append(poll.bb_handle.bb_eval)
          post.iter.append(iteration)
          post.psize.append(poll.mesh.psize)
        else:
          continue

    else:
      poll.point_index = -1
      """ Parallel evaluation for points in the poll set """
      with concurrent.futures.ProcessPoolExecutor(options.np) as executor:
        results = [executor.submit(poll.eval_poll_point,
                       it) for it in range(len(poll.poll_dirs))]
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

    xpost: List[Point] = poll.master_updates(xt, peval, save_all_best=options.save_all_best, save_all=options.save_results)
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

    if log is not None:
        log.log_msg(msg=post.__str__(), msg_type=MSG_TYPE.INFO)
    if options.display:
      print(post)
    
    LAMBDA_k = poll.LAMBDA
    RHO_k = poll.RHO
    
    Failure_check = iteration > 0 and poll.Failure_stop is not None and poll.Failure_stop and (not poll.success or goToSearch)
    
    if (Failure_check or poll.bb_eval >= options.budget) or (abs(poll.mesh.psize) < options.tol or poll.bb_eval >= options.budget or poll.terminate):
      log.log_msg(f"\n--------------- Termination of the poll step  ---------------", MSG_TYPE.INFO)
      if (abs(poll.mesh.psize) < options.tol):
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
    print(f"{poll.bb_handle.blackbox}: fmin = {poll.xmin.f:.2f} , hmin= {poll.xmin.h:.2f}")

  elif len(args) > 1 and not isinstance(args[1], toy.Run):
    if log is not None:
      log.log_msg(msg="Could not find " + args[1] + " in the internal BM suite.", msg_type=MSG_TYPE.ERROR)
    raise IOError("Could not find " + args[1] + " in the internal BM suite.")

  if options.save_results:
    post.output_results(out)

  if options.display:
    print(" end of orthogonal MADS ")
    if log is not None:
      log.log_msg(msg=" end of orthogonal MADS " + args[1] + " in the internal BM suite.", msg_type=MSG_TYPE.INFO)
    print(" Final objective value: " + str(poll.xmin.f) + ", hmin= " + str(poll.xmin.h))
    if log is not None:
      log.log_msg(msg=" Final objective value: " + args[1] + " in the internal BM suite.", msg_type=MSG_TYPE.INFO)

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
    log.log_msg(msg=f" psize = {poll.mesh.psize}", msg_type=MSG_TYPE.INFO)
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
    print(" psize = " + str(poll.mesh.psize))
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
                "psize": poll.mesh.psize,
                "psuccess": poll.mesh.psize_success,
                "pmax": poll.mesh.psize_max,
                "msize": poll.mesh.msize}

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

