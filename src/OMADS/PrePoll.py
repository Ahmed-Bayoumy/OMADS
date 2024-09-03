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

from .CandidatePoint import CandidatePoint
from .Barriers import Barrier, BarrierMO
# from ._common import *
from .Omesh import Omesh
from .Directions import *
from .Parameters import Parameters
from .Options import Options
from .Evaluator import Evaluator
from multiprocessing import cpu_count
from .PostProcess import PostMADS, Output

@dataclass
class PrePoll:
  """ Preprocessor for setting up optimization settings and parameters"""
  data: Dict[Any, Any]
  log: logger = None

  def initialize_from_dict(self, log: logger = None, xs: CandidatePoint=None):
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
    B = BarrierMO(param=param, options=options) if param.isPareto else Barrier(param)
    ev = Evaluator(**self.data["evaluator"])
    if self.log is not None:
      self.log.log_msg(msg="- Set the POLL configurations", msg_type=MSG_TYPE.INFO)
    ev.dtype.precision = options.precision
    if param.constants != None:
      ev.constants = copy.deepcopy(param.constants)
    
    
  
    iteration: int =  0
    """ 2- Initialize iteration number and construct a point instant for the starting point """
    extend = options.extend is not None and isinstance(options.extend, Dirs2n)
    is_xs = False
    if xs is None or not isinstance(xs, CandidatePoint) or not xs.evaluated:
      x_start = CandidatePoint()
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
      # COMPLETED: Add the Gmesh constructor req inputs
      poll.mesh = Gmesh(pbParam=param, runOptions=options) if (param.meshType).lower() == "gmesh" else Omesh(pbParam=param, runOptions=options)
      """ 5- Assign optional algorithmic parameters to the constructed poll instant  """
      poll.opportunistic = options.opportunistic
      poll.seed = options.seed
      # poll.mesh.dtype.precision = options.precision
      # poll.mesh.psize = options.psize_init
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
          x_start.var_type.append(VAR_TYPE.REAL)
          x_start.var_link.append(None)
        elif k.lower()[0] == "i":
          x_start.var_type.append(VAR_TYPE.INTEGER)
          x_start.var_link.append(None)
        elif k.lower()[0] == "d":
          x_start.var_type.append(VAR_TYPE.DISCRETE)
          if x_start.sets is not None and isinstance(x_start.sets, dict):
            if x_start.sets[k.split('_')[1]] is not None:
              x_start.var_link.append(k.split('_')[1])
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
          x_start.var_type.append(VAR_TYPE.REAL)
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
        poll.bb_output, _ = poll.bb_handle.eval(p)
    else:
       if not is_xs:
        poll.bb_output, _ = poll.bb_handle.eval(x_start.coordinates)
    x_start.hmax = B._h_max if isinstance(B, Barrier) else B._hMax
    x_start.RHO = param.RHO
    
    if param.LAMBDA is None:
      param.LAMBDA = [0] * len(x_start.c_ineq)
    if not isinstance(param.LAMBDA, list):
      param.LAMBDA = [param.LAMBDA]
    if len(x_start.c_ineq) > len(param.LAMBDA):
      param.LAMBDA += [param.LAMBDA[-1]] * abs(len(param.LAMBDA)-len(x_start.c_ineq))
    if len(x_start.c_ineq) < len(param.LAMBDA):
      del param.LAMBDA[len(x_start.c_ineq):]
    x_start.LAMBDA = param.LAMBDA
    
    if not is_xs:
      x_start.__eval__(poll.bb_output)
      if isinstance(B, Barrier):
        B._h_max = x_start.hmax
      elif isinstance(B, BarrierMO):
        B._hMax = x_start.hmax
    """ 9- Copy the starting point object to the poll's  minimizer subclass """
    if not extend:
      if x_start.status == DESIGN_STATUS.INFEASIBLE and isinstance(B, BarrierMO):
        poll.x_sc = copy.deepcopy(x_start)
      else:
        poll.xmin = copy.deepcopy(x_start)
    """ 10- Hold the starting point in the poll
     directions subclass and define problem parameters """
    poll.poll_set.append(x_start)
    poll.scale(ub=param.ub, lb=param.lb, factor=param.scaling)
    poll.dim = x_start.n_dimensions
    if not extend:
      poll.hashtable = Cache()
      poll.hashtable._n_dim = len(param.baseline)
      poll.hashtable._isPareto = param.isPareto
      if param.isPareto:
        poll.hashtable.ND_points = []
    """ 10- Initialize the number of successful points
     found and check if the starting minimizer performs better
    than the worst (f = inf) """
    poll.nb_success = 0
    if not extend and poll.xmin.evaluated and poll.xmin < CandidatePoint():
      poll.poll_set = [poll.xmin]
    elif extend and x_start.status == DESIGN_STATUS.FEASIBLE and x_start < poll.xmin:
      poll.xmin = copy.deepcopy(x_start)
      poll.mesh.enlargeDeltaFrameSize()
    elif extend and x_start.status == DESIGN_STATUS.INFEASIBLE and x_start < poll.x_sc:
      poll.x_sc = copy.deepcopy(x_start)
    elif extend and x_start.status == DESIGN_STATUS.FEASIBLE and x_start >= poll.xmin:
      poll.mesh.refineDeltaFrameSize()
    elif extend and x_start.status == DESIGN_STATUS.INFEASIBLE and x_start >= poll.x_sc:
      poll.mesh.refineDeltaFrameSize()
    
    poll.xmin.mesh = copy.deepcopy(poll.mesh)
    poll.x_sc.mesh = copy.deepcopy(poll.mesh)


    """ 11- Construct the results postprocessor class object 'post' """
    if poll.xmin.evaluated:
      x_start.evalNo = poll.bb_handle.bb_eval
      poll.xmin.evalNo = poll.bb_handle.bb_eval
      post = PostMADS(x_incumbent=[poll.xmin], xmin=poll.xmin, poll_dirs=[poll.xmin])
      post.psize.append(poll.mesh.getDeltaFrameSize().coordinates)
      post.bb_eval.append(poll.bb_handle.bb_eval)
      
      post.iter.append(iteration)
    elif poll.x_sc.evaluated:
      x_start.evalNo = poll.bb_handle.bb_eval
      poll.x_sc.evalNo = poll.bb_handle.bb_eval
      post = PostMADS(x_incumbent=[poll.x_sc], xmin=poll.x_sc, poll_dirs=[poll.x_sc])
      post.psize.append(poll.mesh.getDeltaFrameSize().coordinates)
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
    out = Output(file_path=param.post_dir, vnames=param.var_names, fnames=param.fun_names, pname=param.name, runfolder=f'{param.name}_run', suffix="all")
    if param.isPareto:
      outP = Output(file_path=param.post_dir, vnames=param.var_names, fnames=param.fun_names, pname=param.name, runfolder=f'{param.name}_ND', suffix="Pareto")
    else:
      outP = None
    if options.display:
      print("End of the evaluation of the starting points")
      if self.log is not None:
        self.log.log_msg(msg="- End of the evaluation of the starting points", msg_type=MSG_TYPE.INFO)

    iteration += 1

    return iteration, x_start, poll, options, param, post, out, B, outP
