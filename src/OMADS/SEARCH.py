from enum import auto, Enum
import json
from multiprocessing import freeze_support, cpu_count
import os
import sys
import time
import SLML as explore
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from OMADS.POLL import Options, VAR_TYPE, Parameters, Point, Cache, Evaluator, OrthoMesh, PostMADS, Output, DType, SUCCESS_TYPES, Barrier, BARRIER_TYPES, Dirs2n, DESIGN_STATUS
import copy
from typing import List, Dict, Any, Callable, Protocol, Optional
import concurrent.futures
from matplotlib import pyplot as plt

class SAMPLING_METHOD(Enum):
  FULLFACTORIAL: int = auto()
  LH: int = auto()
  RS: int = auto()
  HALTON: int = auto()

class SEARCH_TYPE(Enum):
  SAMPLING: int = auto()
  SURROGATE: int = auto()
  VNS: int = auto()
  BAYESIAN: int = auto()
  NM: int = auto()
  PSO: int = auto()

class DIST_TYPE(Enum):
  GAUSS: int = auto()
  GAMMA: int = auto()
  EXPONENTIAL: int = auto()
  BIONOMIAL: int = auto()
  POISSON: int = auto()

class STOP_TYPE(Enum):
  NO_STOP: int = auto()
  ERROR: int = auto()
  UNKNOWN_STOP_REASON: int = auto()
  CTRL_C: int = auto()
  USER_STOPPED: int = auto()
  MESH_PREC_REACHED: int = auto()
  X0_FAIL: int = auto()
  P1_FAIL: int = auto()
  DELTA_M_MIN_REACHED: int = auto()
  DELTA_P_MIN_REACHED: int = auto()
  MAX_TIME_REACHED: int = auto()
  MAX_BB_EVAL_REACHED: int = auto()
  MAX_SGTE_EVAL_REACHED: int = auto()
  F_TARGET_REACHED: int = auto()
  MAX_CACHE_MEMORY_REACHED: int = auto()

@dataclass
class VNS_data:
  fixed_vars: List[Point] = None
  nb_search_pts: int = 0
  stop: bool = False
  stop_reason: STOP_TYPE = STOP_TYPE.NO_STOP
  success: SUCCESS_TYPES = SUCCESS_TYPES.US
  count_search: bool = False
  new_feas_inc: Point = None
  new_infeas_inc: Point = None
  params: Parameters = None
  # true_barrier: Barrier = None
  # sgte_barrier: Barrier = None
  active_barrier: Barrier = None


@dataclass
class VNS(VNS_data):
  """ 
  """
  _k: int = 1
  _k_max: int = 100
  _old_x: Point = None
  _dist: List[DIST_TYPE] = None
  _ns_dist: List[int] = None
  _rho: float = 0.1
  _seed: int = 0
  _rho0: float = 0.1

  def __init__(self, active_barrier: Barrier, stop: bool=False, true_barrier: Barrier=None, sgte_barrier: Barrier=None, params=None):
    self.stop = stop
    self.count_search = not self.stop
    # self.params._opt_only_sgte = False
    self._dist = [DIST_TYPE.GAUSS, DIST_TYPE.GAMMA, DIST_TYPE.EXPONENTIAL, DIST_TYPE.POISSON]
    # self.true_barrier = true_barrier
    # self.sgte_barrier = sgte_barrier
    self.active_barrier = active_barrier
    self.params = params
  
  def draw_from_gauss(self, mean: Point) -> List[Point]:
    """_summary_
    """
    np.random.seed(self._seed)
    cs = np.zeros((self._ns_dist[0], mean.n_dimensions))
    # pts: List[Point] = [Point()] * self._ns_dist[0]
    for i in range(mean.n_dimensions):
      cs[:, i] = np.random.normal(loc=mean.coordinates[i], scale=self._rho, size=(self._ns_dist[0],))
    
    return cs

  
  def draw_from_gamma(self, mean: Point) -> List[Point]:
    """_summary_
    """
    np.random.seed(self._seed)
    cs = np.zeros((self._ns_dist[1], mean.n_dimensions))
    # pts: List[Point] = [Point()] * self._ns_dist[1]
    for i in range(mean.n_dimensions):
      val = mean.coordinates
      delta = 0.
      if val[i]<=0. or 0 < val[i] <= 0.5:
        delta = 5- val[i]
      cs[:, i] = np.random.gamma(shape=(mean.coordinates[i]+delta)/self._rho, scale=self._rho, size=(self._ns_dist[1],))-delta
    
    return cs

  def draw_from_exp(self, mean: Point) -> List[Point]:
    """_summary_
    """
    np.random.seed(self._seed)
    cs = np.zeros((self._ns_dist[2], mean.n_dimensions))
    # pts: List[Point] = [Point()] * self._ns_dist[2]
    for i in range(mean.n_dimensions):
      cs[:, i] = (np.random.exponential(scale=self._rho, size=self._ns_dist[2]))+mean.coordinates[i]
    
    # for i in range(self._ns_dist[2]):
    #   pts[i].coordinates = copy.deepcopy(cs[i, :])

    return cs
  
  def draw_from_poisson(self, mean: Point) -> List[Point]:
    """_summary_
    """
    np.random.seed(self._seed)
    cs = np.zeros((self._ns_dist[3], mean.n_dimensions))
    # pts: List[Point] = [Point()] * self._ns_dist[2]
    for i in range(mean.n_dimensions):
      val = mean.coordinates
      delta = 0.
      if val[i]<=0. or 0 < val[i] <= 0.5:
        delta = 5- val[i]
      cs[:, i] = (np.random.poisson(lam=(mean.coordinates[i]+delta), size=(self._ns_dist[3],))-delta)*self._rho
    
    return cs

  def draw_from_binomial(self, mean: Point) -> List[Point]:
    """_summary_
    """
    np.random.seed(self._seed)
    cs = np.zeros((self._ns_dist[4], mean.n_dimensions))
    # pts: List[Point] = [Point()] * self._ns_dist[2]
    for i in range(mean.n_dimensions):
      val = mean.coordinates
      delta = 0.
      if val[i]<=0. or 0 < val[i] <= 0.5:
        delta = 5- val[i]
      cs[:, i] = (np.random.binomial(n=(mean.coordinates[i]+delta)/((1/self._rho) if self._rho > 1. else self._rho), p=(1/self._rho) if self._rho > 1. else self._rho, size=(self._ns_dist[4],))-delta)
    
    # for i in range(self._ns_dist[2]):
    #   pts[i].coordinates = copy.deepcopy(cs[i, :])

    return cs

  def generate_samples(self, x_inc: Point, dist: DIST_TYPE)->List[float]:
    """_summary_
    """
    if not x_inc.evaluated:
      return None
    else:
      if dist == DIST_TYPE.GAUSS:
        return self.draw_from_gauss(x_inc)
      
      if dist == DIST_TYPE.GAMMA:
        return self.draw_from_gamma(x_inc)
      
      if dist == DIST_TYPE.EXPONENTIAL:
        return self.draw_from_exp(x_inc)
      
      if dist == DIST_TYPE.POISSON:
        return self.draw_from_poisson(x_inc)
      
      if dist == DIST_TYPE.BIONOMIAL:
        return self.draw_from_binomial(x_inc)
    
    return None


  def run(self):
    if self.stop:
      return
    # Initial 
    # opt_only_sgte = self.params._opt_only_sgte

    # point x
    x: Point = self.active_barrier._best_feasible
    if (x is None or not x.evaluated) and self.active_barrier._filter is not None:
      x = self.active_barrier.get_best_infeasible()
    
    if (x is None or not x.evaluated) and self.active_barrier._all_inserted is not None:
      x = self.active_barrier._all_inserted[0]
    # // update _k and _old_x:
    
    if self._old_x is not None and x != self._old_x:
      self._rho = np.sqrt(np.sum([abs(self._old_x.coordinates[i]-x.coordinates[i])**2 for i in range(len(self._old_x.coordinates))]))
      # self._rho *= 2
      self._k += 1
    if self._k > self._k_max:
      self.stop = True
    
    self._old_x = x

    samples = np.zeros((sum(self._ns_dist), len(x.coordinates)))
    c = 0
    self._seed += np.random.randint(0, 10000)
    np.random.seed(self._seed)
    if x.status is DESIGN_STATUS.FEASIBLE:
      for i in range(len(self._dist)):
        temp = self.generate_samples(x_inc=x, dist= self._dist[i])
        for p in temp:
          samples[c, :] = p
          c += 1
    
    ns_dist_old = self._ns_dist
    self._ns_dist = [int(0.1*xds) for xds in self._ns_dist]

    if self.active_barrier._sec_poll_center is not None and self.active_barrier.get_best_infeasible().evaluated:
      for i in range(len(self._dist)):
        temp = self.generate_samples(x_inc= self.active_barrier.get_best_infeasible(), dist= self._dist[i])
        for p in temp:
          samples = np.vstack((samples, p))
          c += 1
    self._ns_dist = ns_dist_old
    return samples
      



@dataclass
class efficient_exploration:
  mesh: OrthoMesh  = field(default_factory=OrthoMesh)
  _success: bool = False
  _xmin: Point = Point()
  prob_params: Parameters = Parameters()
  sampling_t: int = 3
  _seed: int = 0
  _dtype: DType = DType()
  iter: int = 1
  vicinity_ratio: np.ndarray = None
  vicinity_min: float = 0.001
  opportunistic: bool = False
  eval_budget: int = 10
  store_cache: bool = True
  check_cache: bool = True
  display: bool = False
  bb_handle: Evaluator = Evaluator()
  bb_output: List = None
  samples: List[Point] = None
  hashtable: Cache = None
  _dim: int = 0
  nb_success: int = 0
  terminate: bool =False
  _save_results: bool = False
  visualize: bool = False
  Failure_stop: bool = None
  sampling_criter: str = None
  weights: List[float] = None
  _type: str = "sampling"
  LAMBDA: List[float] = None
  RHO: float = 0.0005
  hmax: float = 0.

  @property
  def type(self):
    return self._type
  
  @type.setter
  def type(self, value: Any) -> Any:
    self._type = value
  
  @property
  def save_results(self):
    return self._save_results
  
  @save_results.setter
  def save_results(self, value: bool) -> bool:
    self._save_results = value
  
  @property
  def dim(self):
    return self._dim
  
  @dim.setter
  def dim(self, value: Any) -> Any:
    self._dim = value
  
  @property
  def xmin(self):
    return self._xmin
  
  @xmin.setter
  def xmin(self, value: Any) -> Any:
    self._xmin = value
  
  @property
  def success(self):
    return self._success
  
  @success.setter
  def success(self, value: Any) -> Any:
    self._success = value
  
  @property
  def seed(self):
    return self._seed
  
  @seed.setter
  def seed(self, value: Any) -> Any:
    self._seed = value
  
  @property
  def dtype(self):
    return self._dtype
  
  @dtype.setter
  def dtype(self, value: Any) -> Any:
    self._dtype = value
  
  def scale(self, ub: List[float], lb: List[float], factor: float = 10.0):
    self.scaling = np.divide(np.subtract(ub, lb, dtype=self._dtype.dtype),
                 factor, dtype=self._dtype.dtype)
    if any(np.isinf(self.scaling)):
      for k, x in enumerate(np.isinf(self.scaling)):
        if x:
          self.scaling[k][0] = 1.0
    s_array = np.diag(self.scaling)
  
  def generate_sample_points(self, nsamples: int = None) -> List[Point]:
    """ Generate the sample points """
    xlim = []
    self.nvars = len(self.prob_params.baseline)
    v = np.empty((self.nvars, 2))
    if self.xmin and self.iter > 1:
      for i in range(len(self.prob_params.lb)):
        D = abs(self.prob_params.ub[i] - self.prob_params.lb[i])
        lb = copy.deepcopy(self.xmin.coordinates[i]-(D * self.vicinity_ratio[i]))
        ub = copy.deepcopy(self.xmin.coordinates[i]+(D * self.vicinity_ratio[i]))
        if lb <= self.prob_params.lb[i]:
          lb = copy.deepcopy(self.prob_params.lb[i])
        elif  lb >= self.prob_params.ub[i]:
          lb = self.xmin.coordinates[i]
        if ub >= self.prob_params.ub[i]:
          ub = copy.deepcopy(self.prob_params.ub[i])
        elif ub <= self.prob_params.lb[i]:
          ub = self.xmin.coordinates[i]
        v[i] = [lb, ub]
    else:
      for i in range(len(self.prob_params.lb)):
        lb = copy.deepcopy(self.prob_params.lb[i])
        ub = copy.deepcopy(self.prob_params.ub[i])
        v[i] = [lb, ub]
    if nsamples is None:
      nsamples = int((self.nvars+1)*(self.nvars+2)/2)
    
    self.ns = nsamples
    resize = False
    clipping = True
    self.seed += np.random.randint(0, 10000)

    if self.sampling_t == SAMPLING_METHOD.FULLFACTORIAL.name:
      sampling = explore.FullFactorial(ns=nsamples, vlim=v, w=self.weights, c=clipping)
      if clipping:
        resize = True
    elif self.sampling_t == SAMPLING_METHOD.RS.name:
      sampling = explore.RS(ns=nsamples, vlim=v)
      sampling.options["randomness"] = self.seed
    elif self.sampling_t == SAMPLING_METHOD.HALTON.name:
      sampling = explore.halton(ns=nsamples, vlim=v, is_ham=True)
    else:
      sampling = explore.LHS(ns=nsamples, vlim=v)
      sampling.options["randomness"] = self.seed
      sampling.options["criterion"] = self.sampling_criter
      sampling.options["msize"] = self.mesh.msize

    
    Ps= copy.deepcopy(sampling.generate_samples())
    if resize:
      self.ns = len(Ps)
      nsamples = len(Ps)

    # if self.xmin is not None:
    #   self.visualize_samples(self.xmin.coordinates[0], self.xmin.coordinates[1])
    self.map_samples_from_coords_to_points(Ps)
    return v, Ps
  
  def project_coords_to_mesh(self, x:List[float], ref: List[float] = None):
    if ref == None:
      ref = [0.]*len(x)
    for i in range(len(x)):
      x[i] = ref[i] + (np.round((x[i]-ref[i])/self.mesh.msize) * self.mesh.msize)
      if x[i] < self.prob_params.lb[i]:
        x[i] = self.prob_params.lb[i] + (self.prob_params.lb[i] - x[i])
        if x[i] > self.prob_params.ub[i]:
          x[i] = self.prob_params.ub[i]
      if x[i] > self.prob_params.ub[i]:
        x[i] = self.prob_params.ub[i] - (x[i] - self.prob_params.ub[i])
        if x[i] < self.prob_params.lb[i]:
          x[i] = self.prob_params.lb[i]

    return x

  def map_samples_from_coords_to_points(self, samples: np.ndarray):
    
    for i in range(len(samples)):
      samples[i, :] = self.project_coords_to_mesh(samples[i, :], ref=np.subtract(self.prob_params.ub , self.prob_params.lb).tolist())
    samples = np.unique(samples, axis=0)
    self.samples: List[Point] = [0] *len(samples)
    for i in range(len(samples)):
      self.samples[i] = Point()
      if self.xmin.var_type is not None:
        self.samples[i].var_type = self.xmin.var_type
      else:
        self.samples[i].var_type = None
      self.samples[i].sets = self.xmin.sets
      self.samples[i].var_link = self.xmin.var_link
      self.samples[i].n_dimensions = len(samples[i, :])
      self.samples[i]._coords = copy.deepcopy(samples[i, :])


  def gauss_perturbation(self, p: Point, npts: int = 5) -> List[Point]:
    lb = self.lb
    ub = self.ub
    # np.random.seed(self.seed)
    cs = np.zeros((npts, p.n_dimensions))
    pts: List[Point] = [0] * npts
    mp = 1.
    for k in range(p.n_dimensions):
      cs[:, k] = np.random.normal(loc=p.coordinates[k], scale=self.mesh.msize, size=(npts,))
      for i in range(npts):
        if cs[i, k] < lb[k]:
          cs[i, k] = lb[k]
        if cs[i, k] > ub[k]:
          cs[i, k] = ub[k]
    
    for i in range(npts):
      pts[i] = Point()
      pts[i].coordinates = copy.deepcopy(cs[i, :])
    
    return pts
  
  def evaluate_sample_point(self, index: int):
    """ Evaluate the sample point i on the points set """
    """ Set the dynamic index for this point """
    tic = time.perf_counter()
    self.point_index = index
    """ Initialize stopping and success conditions"""
    stop: bool = False
    """ Copy the point i to a trial one """
    xtry: Point = self.samples[index]
    """ This is a success bool parameter used for
     filtering out successful designs to be printed
    in the output results file"""
    success = False

    """ Check the cache memory; check if the trial point
     is a duplicate (it has already been evaluated) """
    unique_p_trials: int = 0
    is_duplicate: bool = (self.check_cache and self.hashtable.size > 0 and self.hashtable.is_duplicate(xtry))
    while is_duplicate and unique_p_trials < 5:
      if self.display:
        print(f'Cache hit. Trial# {unique_p_trials}: Looking for a non-duplicate in the vicinity of the duplicate point ...')
      xtries: List[Point] = self.gauss_perturbation(p=xtry, npts=len(self.poll_dirs)*2)
      for tr in range(len(xtries)):
        is_duplicate = self.hashtable.is_duplicate(xtries[tr])
        if is_duplicate:
           continue 
        else:
          xtry = copy.deepcopy(xtries[tr])
          break
      unique_p_trials += 1

    if (is_duplicate):
      if self.display:
        print("Cache hit ... Failed to find a non-duplicate alternative.")
      stop = True
      bb_eval = copy.deepcopy(self.bb_eval)
      psize = copy.deepcopy(self.mesh.psize)
      return [stop, index, self.bb_handle.bb_eval, success, psize, xtry]


    """ Evaluation of the blackbox; get output responses """
    if xtry.sets is not None and isinstance(xtry.sets,dict):
      p: List[Any] = []
      for i in range(len(xtry.var_type)):
        if (xtry.var_type[i] == VAR_TYPE.DISCRETE or xtry.var_type[i] == VAR_TYPE.CATEGORICAL) and xtry.var_link[i] is not None:
          p.append(xtry.sets[xtry.var_link[i]][int(xtry.coordinates[i])])
        else:
          p.append(xtry.coordinates[i])
      self.bb_output = self.bb_handle.eval(p)
    else:
      self.bb_output = self.bb_handle.eval(xtry.coordinates)

    """
      Evaluate the poll point:
        - Set multipliers and penalty
        - Evaluate objective function
        - Evaluate constraint functions (can be an empty vector)
        - Aggregate constraints
        - Penalize the objective (extreme barrier)
    """
    xtry.LAMBDA = copy.deepcopy(self.LAMBDA)
    xtry.RHO = copy.deepcopy(self.RHO)
    xtry.hmax = copy.deepcopy(self.hmax)
    xtry.constraints_type = copy.deepcopy(self.prob_params.constraints_type)
    xtry.__eval__(self.bb_output)
    toc = time.perf_counter()
    xtry.Eval_time = (toc - tic)

    """ Update multipliers and penalty """
    if self.LAMBDA == None:
      self.LAMBDA = self.xmin.LAMBDA
    if len(xtry.c_ineq) > len(self.LAMBDA):
      self.LAMBDA += [self.LAMBDA[-1]] * abs(len(self.LAMBDA)-len(xtry.c_ineq))
    if len(xtry.c_ineq) < len(self.LAMBDA):
      del self.LAMBDA[len(xtry.c_ineq):]
    for i in range(len(xtry.c_ineq)):
      if self.RHO == 0.:
        self.RHO = 0.001
      if self.LAMBDA is None:
        self.LAMBDA = xtry.LAMBDA
      self.LAMBDA[i] = copy.deepcopy(max(self.dtype.zero, self.LAMBDA[i] + (1/self.RHO)*xtry.c_ineq[i]))
    
    if xtry.status == DESIGN_STATUS.FEASIBLE:
      self.RHO *= copy.deepcopy(0.5)

    """ Add to the cache memory """
    if self.store_cache:
      self.hashtable.hash_id = xtry

    if self.save_results or self.display:
      self.bb_eval = self.bb_handle.bb_eval
      self.psize = copy.deepcopy(self.mesh.psize)
    psize = copy.deepcopy(self.mesh.psize)



    if self.success and self.opportunistic and self.iter > 1:
      stop = True

    """ Check stopping criteria """
    if self.bb_eval >= self.eval_budget:
      self.terminate = True
      stop = True
      return [stop, index, self.bb_handle.bb_eval, success, psize, xtry]

    return [stop, index, self.bb_handle.bb_eval, success, psize, xtry]

  def master_updates(self, x: List[Point], peval, save_all_best: bool = False, save_all:bool = False):
    if peval >= self.eval_budget:
      self.terminate = True
    x_post: List[Point] = []
    c = 0
    for xtry in x:
      c += 1
      """ Check success conditions """
      is_infeas_dom: bool = (self.xmin.status == xtry.status == DESIGN_STATUS.INFEASIBLE and (xtry.h < self.xmin.h and xtry.h <= xtry.hmax and xtry.fobj <= self.xmin.fobj) )
      is_feas_dom: bool = (self.xmin.status == xtry.status == DESIGN_STATUS.FEASIBLE and xtry < self.xmin)
      is_infea_improving: bool = (self.xmin.status == DESIGN_STATUS.FEASIBLE and xtry.status == DESIGN_STATUS.INFEASIBLE and (xtry.fobj < self.xmin.fobj and xtry.h <= xtry.hmax))
      is_feas_improving: bool = (self.xmin.status == DESIGN_STATUS.INFEASIBLE and xtry.status == DESIGN_STATUS.FEASIBLE and (xtry.fobj < self.xmin.fobj))
      success = False
      if (xtry.is_EB_passed and (is_infeas_dom or is_feas_dom or is_infea_improving or is_feas_improving)):
        self.success = True
        success = True  # <- This redundant variable is important
        # for managing concurrent parallel execution
        self.nb_success += 1
        """ Update the post instant """
        del self._xmin
        self._xmin = Point()
        self._xmin = copy.deepcopy(xtry)
        if self.display:
          if self._dtype.dtype == np.float64:
            print(f"Success: fmin = {self.xmin.f:.15f} (hmin = {self.xmin.h:.15})")
          elif self._dtype.dtype == np.float32:
            print(f"Success: fmin = {self.xmin.f:.6f} (hmin = {self.xmin.h:.6})")
          else:
            print(f"Success: fmin = {self.xmin.f:.18f} (hmin = {self.xmin.h:.18})")

        self.mesh.psize_success = copy.deepcopy(self.mesh.psize)
        self.mesh.psize_max = copy.deepcopy(np.maximum(self.mesh.psize,
                              self.mesh.psize_max,
                              dtype=self._dtype.dtype))
      if (save_all_best and success) or (save_all):
        x_post.append(xtry)

    return x_post
        
  def update_local_region(self, region="expand"):
    if region =="expand":
      for i in range(len(self.vicinity_ratio)):
        if self.vicinity_ratio[i] * 2 < self.prob_params.ub[i]:
          self.vicinity_ratio[i] *= 2
    elif region == "contract":
      for i in range(len(self.vicinity_ratio)):
        if self.vicinity_ratio[i] / 2 > self.prob_params.lb[i] and self.vicinity_ratio[i] > self.vicinity_min:
          self.vicinity_ratio[i] /= 2
    else:
      raise IOError(f"Unrecognized {region} local region operation")
@dataclass
class search_sampling:
  s_method: str = SAMPLING_METHOD.LH.name
  ns: int = 3        
  visualize: bool = False
  criterion: str = None
  weights: List[float] = None
  type: str = SEARCH_TYPE.SAMPLING.name
@dataclass
class PreExploration:
  """ Preprocessor for setting up optimization settings and parameters"""
  data: Dict[Any, Any]

  def initialize_from_dict(self):
    """ MADS initialization """
    """ 1- Construct the following classes by unpacking
     their respective dictionaries from the input JSON file """
    options = Options(**self.data["options"])
    param = Parameters(**self.data["param"])
    B = Barrier(param)
    ev = Evaluator(**self.data["evaluator"])
    search_step = search_sampling(**self.data["search"])
    ev.dtype.precision = options.precision
    if param.constants != None:
      ev.constants = copy.deepcopy(param.constants)

    if param.constraints_type is not None and isinstance(param.constraints_type, list):
      for i in range(len(param.constraints_type)):
        if param.constraints_type[i] == BARRIER_TYPES.PB.name:
          param.constraints_type[i] = BARRIER_TYPES.PB
        elif param.constraints_type[i] == BARRIER_TYPES.RB.name:
          param.constraints_type[i] = BARRIER_TYPES.RB
        elif param.constraints_type[i] == BARRIER_TYPES.PEB.name:
          param.constraints_type[i] = BARRIER_TYPES.PEB
        else:
          param.constraints_type[i] = BARRIER_TYPES.EB
    elif param.constraints_type is not None:
      param.constraints_type = BARRIER_TYPES(param.constraints_type)
    
    """ 2- Initialize iteration number and construct a point instant for the starting point """
    iteration: int =  0
    x_start = Point()
    """ 3- Construct an instant for the poll 2n orthogonal directions class object """
    extend = options.extend is not None and isinstance(options.extend, efficient_exploration)
    if not extend:
      search = efficient_exploration()
      if param.Failure_stop != None and isinstance(param.Failure_stop, bool):
        search.Failure_stop = param.Failure_stop
      search.samples = []
      search.dtype.precision = options.precision
      search.save_results = options.save_results
      """ 4- Construct an instant for the mesh subclass object by inheriting
      initial parameters from mesh_params() """
      search.mesh = OrthoMesh()
      search.sampling_t = search_step.s_method
      search.type = search_step.type
      search.ns = search_step.ns
      search.sampling_criter = search_step.criterion
      search.visualize = search_step.visualize
      search.weights = search_step.weights
      """ 5- Assign optional algorithmic parameters to the constructed poll instant  """
      search.opportunistic = options.opportunistic
      search.seed = options.seed
      search.mesh.dtype.precision = options.precision
      search.mesh.psize = options.psize_init
      search.eval_budget = options.budget
      search.store_cache = options.store_cache
      search.check_cache = options.check_cache
      search.display = options.display
      search.bb_eval = 0
      search.prob_params = Parameters(**self.data["param"])
    else:
      search = options.extend
      search.samples = []
    n_available_cores = cpu_count()
    if options.parallel_mode and options.np > n_available_cores:
      options.np == n_available_cores
    """ 6- Initialize blackbox handling subclass by copying
     the evaluator 'ev' instance to the poll object"""
    search.bb_handle = ev
    search.bb_handle.bb_eval = ev.bb_eval
    """ 7- Evaluate the starting point """
    if options.display:
      print(" Evaluation of the starting points")
    x_start.coordinates = param.baseline
    x_start.sets = param.var_sets
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
      search.bb_output = search.bb_handle.eval(p)
    else:
      search.bb_output = search.bb_handle.eval(x_start.coordinates)
    x_start.hmax = B._h_max
    search.hmax = B._h_max
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
    x_start.constraints_type = param.constraints_type
    x_start.__eval__(search.bb_output)
    """ 9- Copy the starting point object to the poll's minimizer subclass """
    search.xmin = copy.deepcopy(x_start)
    """ 10- Hold the starting point in the poll
     directions subclass and define problem parameters"""
    search.samples.append(x_start)
    search.scale(ub=param.ub, lb=param.lb, factor=param.scaling)
    search.dim = x_start.n_dimensions
    if not extend:
      search.hashtable = Cache()
    """ 10- Initialize the number of successful points
     found and check if the starting minimizer performs better
    than the worst (f = inf) """
    search.nb_success = 0
    if search.xmin < Point():
      search.mesh.psize_success = search.mesh.psize
      search.mesh.psize_max = np.maximum(search.mesh.psize,
                      search.mesh.psize_max,
                      dtype=search.dtype.dtype)
      search.samples = [search.xmin]
    """ 11- Construct the results postprocessor class object 'post' """
    post = PostMADS(x_incumbent=[search.xmin], xmin=search.xmin, poll_dirs=[search.xmin])
    post.psize.append(search.mesh.psize)
    post.bb_eval.append(search.bb_handle.bb_eval)
    post.iter.append(iteration)

    """ Note: printing the post will print a results row
     within the results table shown in Python console if the
    'display' option is true """
    # if options.display:
    #     print(post)
    """ 12- Add the starting point hash value to the cache memory """
    if options.store_cache:
      search.hashtable.hash_id = x_start
    """ 13- Initialize the output results file object  """
    out = Output(file_path=param.post_dir, vnames=param.var_names, pname=param.name, runfolder=f'{param.name}_run', replace=True)
    if options.display:
      print("End of the evaluation of the starting points")
    iteration += 1

    return iteration, x_start, search, options, param, post, out, B
  

  

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

  """ Run preprocessor for the setup of
   the optimization problem and for the initialization
  of optimization process """
  iteration, xmin, search, options, param, post, out, B = PreExploration(data).initialize_from_dict()

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


  

  LAMBDA_k = xmin.LAMBDA
  RHO_k = xmin.RHO

  while True:
    search.mesh.update()
    if B is not None:
      if B._filter is not None:
        B.select_poll_center()
        B.update_and_reset_success()
      else:
        B.insert(search.xmin)
    
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
      search.samples = search_VN.run()
      if search_VN.stop:
        print("Reached maximum number of VNS iterations!")
        break
      vv = search.map_samples_from_coords_to_points(samples=search.samples)
    else:
      vvp = vvs = []
      if B._best_feasible is not None and B._best_feasible.evaluated:
        search.xmin = B._best_feasible
        vvp, _ = search.generate_sample_points(int(((search.dim+1)/2)*((search.dim+2)/2)) if search.ns is None else search.ns)
      if B._filter is not None and B.get_best_infeasible().evaluated:
        xmin_bup = search.xmin
        Prim_samples = search.samples
        search.xmin = B.get_best_infeasible()
        vvs, _ = search.generate_sample_points(int(((search.dim+1)/2)*((search.dim+2)/2)) if search.ns is None else search.ns)
        search.samples += Prim_samples
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
                xinput = search.samples
              ps = visualize(xinput, jjj, kkk, search.mesh.msize, vv, fig, ax, search.xmin, ps, bbeval=search.bb_handle, lb=search.prob_params.lb, ub=search.prob_params.ub, spindex=iii, bestKnown=search.prob_params.best_known, blk=False)
      search.store_cache = sc_old
      search.check_cache = cc_old


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
    if not options.parallel_mode:
      for it in range(len(search.samples)):
        if search.terminate:
          break
        f = search.evaluate_sample_point(it)
        xt.append(f[-1])
        if not f[0]:
          post.bb_eval.append(search.bb_handle.bb_eval)
          peval += 1
          post.iter.append(iteration)
          post.psize.append(search.mesh.psize)
        else:
          continue

    else:
      search.point_index = -1
      """ Parallel evaluation for points in the poll set """
      with concurrent.futures.ProcessPoolExecutor(options.np) as executor:
        results = [executor.submit(search.evaluate_sample_point,
                       it) for it in range(len(search.samples))]
        for f in concurrent.futures.as_completed(results):
          # if f.result()[0]:
          #     executor.shutdown(wait=False)
          # else:
          if options.save_results or options.display:
            peval = peval +1
            search.bb_eval = peval
            post.bb_eval.append(peval)
            post.iter.append(iteration)
            # post.poll_dirs.append(poll.poll_dirs[f.result()[1]])
            post.psize.append(f.result()[4])
          xt.append(f.result()[-1])
  
    xpost: List[Point] = search.master_updates(xt, peval, save_all_best=options.save_all_best, save_all=options.save_results)
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
    if search.success:
      search.mesh.psize = search.mesh.msize = np.multiply(search.mesh.msize, 2, dtype=search.dtype.dtype)
      search.update_local_region(region="expand")
    else:
      search.mesh.psize = search.mesh.msize = np.divide(search.mesh.msize, 2, dtype=search.dtype.dtype)
      search.update_local_region(region="contract")

    if options.display:
      print(post)

    Failure_check = iteration > 0 and search.Failure_stop is not None and search.Failure_stop and not search.success
    if (Failure_check) or (abs(search.mesh.msize) < options.tol or search.bb_eval >= options.budget or search.terminate):
      break
    iteration += 1

  toc = time.perf_counter()

  

  if options.save_results:
    post.output_results(out)

  if options.display:
    print(" end of orthogonal MADS ")
    print(" Final objective value: " + str(search.xmin.f) + ", hmin= " + str(search.xmin.h))

  if options.save_coordinates:
    post.output_coordinates(out)
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
    print(" psize = " + str(search.mesh.psize))
    print(" psize_success = " + str(search.mesh.psize_success))
    print(" psize_max = " + str(search.mesh.psize_max))
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
                "psize": search.mesh.psize,
                "psuccess": search.mesh.psize_success,
                "pmax": search.mesh.psize_max,
                "msize": search.mesh.msize}
  
  if search.visualize:
    sc_old = search.store_cache
    cc_old = search.check_cache
    search.check_cache = False
    search.store_cache = False
    temp = Point()
    temp.coordinates = output["xmin"]
    for ii in range(len(ax)):
      for jj in range(len(xmin.coordinates)):
          for kk in range(jj+1, len(xmin.coordinates)):
            if kk != jj:
              ps = visualize(xinput, jj, kk, search.mesh.msize, vv, fig, ax, temp, ps, bbeval=search.bb_handle, lb=search.prob_params.lb, ub=search.prob_params.ub, title=search.prob_params.problem_name, blk=True,vnames=search.prob_params.var_names, spindex=ii, bestKnown=search.prob_params.best_known)
    search.check_cache = sc_old
    search.store_cache = cc_old

  return output, search



def visualize(points: List[Point], hc_index, vc_index, msize, vlim, fig, axes, pmin, ps = None, title="unknown", blk=False, vnames=None, bbeval=None, lb = None, ub=None, spindex=0, bestKnown=None):

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