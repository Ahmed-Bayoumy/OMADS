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
from .Point import Point
from .Barriers import Barrier
# from ._common import *
from .Directions import *
import samplersLib as explore
import random
from matplotlib import pyplot as plt
from ._globals import *
from .Parameters import Parameters
from .Evaluator import Evaluator

@dataclass
class VNS_data:
  fixed_vars: List[CandidatePoint] = None
  nb_search_pts: int = 0
  stop: bool = False
  stop_reason: STOP_TYPE = STOP_TYPE.NO_STOP
  success: SUCCESS_TYPES = SUCCESS_TYPES.US
  count_search: bool = False
  new_feas_inc: CandidatePoint = None
  new_infeas_inc: CandidatePoint = None
  params: Parameters = None
  # true_barrier: Barrier = None
  # sgte_barrier: Barrier = None
  active_barrier: auto = None


@dataclass
class VNS(VNS_data):
  """ 
  """
  _k: int = 1
  _k_max: int = 100
  _old_x: CandidatePoint = None
  _dist: List[DIST_TYPE] = None
  _ns_dist: List[int] = None
  _rho: float = 0.1
  _seed: int = 0
  _rho0: float = 0.1

  def __init__(self, active_barrier: auto, stop: bool=False, true_barrier: Barrier=None, sgte_barrier: Barrier=None, params=None):
    self.stop = stop
    self.count_search = not self.stop
    # self.params._opt_only_sgte = False
    self._dist = [DIST_TYPE.GAUSS, DIST_TYPE.GAMMA, DIST_TYPE.EXPONENTIAL, DIST_TYPE.POISSON]
    # self.true_barrier = true_barrier
    # self.sgte_barrier = sgte_barrier
    self.active_barrier = active_barrier
    self.params = params
  
  def draw_from_gauss(self, mean: CandidatePoint) -> List[CandidatePoint]:
    """_summary_
    """
    np.random.seed(self._seed)
    cs = np.zeros((self._ns_dist[0], mean.n_dimensions))
    # pts: List[Point] = [Point()] * self._ns_dist[0]
    for i in range(mean.n_dimensions):
      if mean.var_type is not None:
        if mean.var_type[i] == VAR_TYPE.REAL:
          cs[:, i] = np.random.normal(loc=mean.coordinates[i], scale=self._rho, size=(self._ns_dist[0],))
        elif mean.var_type[i] == VAR_TYPE.INTEGER or mean.var_type[i] == VAR_TYPE.DISCRETE:
          cs[:, i] = np.random.randint(low=int(mean.coordinates[i]-self._rho), high=int(np.ceil(mean.coordinates[i]+(self._rho if self._rho>0 else 0.001))), size=(self._ns_dist[0],))
        elif mean.var_type[i] == VAR_TYPE.CATEGORICAL:
          stemp = np.linspace(self.params.lb[i], self.params.ub[i], int(self.params.ub[i]-self.params.lb[i])+1).tolist()
          cs[:len(stemp), i] = random.sample(stemp, len(stemp))
        else:
          cs[:, i] = [mean.coordinates[i]]*self._ns_dist[0]
      else:
        cs[:, i] = np.random.normal(loc=mean.coordinates[i], scale=self._rho, size=(self._ns_dist[0],))
    
    return cs

  
  def draw_from_gamma(self, mean: CandidatePoint) -> List[CandidatePoint]:
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
      if mean.var_type is not None:
        if mean.var_type[i] == VAR_TYPE.REAL:
          cs[:, i] = np.random.gamma(shape=(mean.coordinates[i]+delta)/self._rho, scale=self._rho, size=(self._ns_dist[1],))-delta
        elif mean.var_type[i] == VAR_TYPE.INTEGER or mean.var_type[i] == VAR_TYPE.CATEGORICAL or mean.var_type[i] == VAR_TYPE.DISCRETE:
          cs[:, i] = np.random.randint(low=int(mean.coordinates[i]-self._rho), high=int(np.ceil(mean.coordinates[i]+(self._rho if self._rho>0 else 0.001))), size=(self._ns_dist[1],))
        else:
          cs[:, i] = [mean.coordinates[i]]*self._ns_dist[1]
      else:
        cs[:, i] = np.random.gamma(shape=(mean.coordinates[i]+delta)/self._rho, scale=self._rho, size=(self._ns_dist[1],))-delta
    
    return cs

  def draw_from_exp(self, mean: CandidatePoint) -> List[CandidatePoint]:
    """_summary_
    """
    np.random.seed(self._seed)
    cs = np.zeros((self._ns_dist[2], mean.n_dimensions))
    # pts: List[Point] = [Point()] * self._ns_dist[2]
    for i in range(mean.n_dimensions):
      if mean.var_type is not None:
        if mean.var_type[i] == VAR_TYPE.REAL:
          cs[:, i] = (np.random.exponential(scale=self._rho, size=self._ns_dist[2]))+mean.coordinates[i]
        elif mean.var_type[i] == VAR_TYPE.INTEGER or mean.var_type[i] == VAR_TYPE.CATEGORICAL or mean.var_type[i] == VAR_TYPE.DISCRETE:
          cs[:, i] = np.random.randint(low=int(mean.coordinates[i]-self._rho), high=int(np.ceil(mean.coordinates[i]+(self._rho if self._rho>0 else 0.001))), size=(self._ns_dist[2],))
        else:
          cs[:, i] = [mean.coordinates[i]]*self._ns_dist[2]
      else:
        cs[:, i] = (np.random.exponential(scale=self._rho, size=self._ns_dist[2]))+mean.coordinates[i]

    
    # for i in range(self._ns_dist[2]):
    #   pts[i].coordinates = copy.deepcopy(cs[i, :])

    return cs
  
  def draw_from_poisson(self, mean: CandidatePoint) -> List[CandidatePoint]:
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
      if mean.var_type is not None:
        if mean.var_type[i] == VAR_TYPE.REAL:
          cs[:, i] = (np.random.poisson(lam=(mean.coordinates[i]+delta), size=(self._ns_dist[3],))-delta)*self._rho
        elif mean.var_type[i] == VAR_TYPE.INTEGER or mean.var_type[i] == VAR_TYPE.CATEGORICAL or mean.var_type[i] == VAR_TYPE.DISCRETE:
          cs[:, i] = np.random.randint(low=int(mean.coordinates[i]-self._rho), high=int(np.ceil(mean.coordinates[i]+(self._rho if self._rho>0 else 0.001))), size=(self._ns_dist[3],))
        else:
          cs[:, i] = [mean.coordinates[i]]*self._ns_dist[3]
      else:
        cs[:, i] = (np.random.poisson(lam=(mean.coordinates[i]+delta), size=(self._ns_dist[3],))-delta)*self._rho
    
    return cs

  def draw_from_binomial(self, mean: CandidatePoint) -> List[CandidatePoint]:
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
      if mean.var_type is not None:
        if mean.var_type[i] == VAR_TYPE.REAL:
          cs[:, i] = (np.random.binomial(n=(mean.coordinates[i]+delta)/((1/self._rho) if self._rho > 1. else self._rho), p=(1/self._rho) if self._rho > 1. else self._rho, size=(self._ns_dist[4],))-delta)
        elif mean.var_type[i] == VAR_TYPE.INTEGER or mean.var_type[i] == VAR_TYPE.CATEGORICAL or mean.var_type[i] == VAR_TYPE.DISCRETE:
          cs[:, i] = np.random.randint(low=int(mean.coordinates[i]-self._rho), high=int(np.ceil(mean.coordinates[i]+(self._rho if self._rho>0 else 0.001))), size=(self._ns_dist[4],))
        else:
          cs[:, i] = [mean.coordinates[i]]*self._ns_dist[4]
      else:
        cs[:, i] = (np.random.binomial(n=(mean.coordinates[i]+delta)/((1/self._rho) if self._rho > 1. else self._rho), p=(1/self._rho) if self._rho > 1. else self._rho, size=(self._ns_dist[4],))-delta)
    
    # for i in range(self._ns_dist[2]):
    #   pts[i].coordinates = copy.deepcopy(cs[i, :])

    return cs

  def generate_samples(self, x_inc: CandidatePoint=None, dist: DIST_TYPE = None)->List[float]:
    """_summary_
    """
    if isinstance(self.active_barrier, BarrierMO):
      if x_inc is None:
        x_inc = self.active_barrier.getAllPoints()[0]
    elif isinstance(self.active_barrier, Barrier):
      if x_inc is None:
        x_inc = self.active_barrier.select_poll_center()
    if x_inc or not x_inc.evaluated:
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
    if isinstance(self.active_barrier, Barrier):
      x: CandidatePoint = self.active_barrier._best_feasible
      if (x is None or not x.evaluated) and self.active_barrier._filter is not None:
        x = self.active_barrier.get_best_infeasible()
      
      if (x is None or not x.evaluated) and self.active_barrier._all_inserted is not None:
        x = self.active_barrier._all_inserted[0]
    elif isinstance(self.active_barrier, BarrierMO):
      if self._old_x:
        x: CandidatePoint = self.active_barrier._currentIncumbentFeas if self._old_x.status == DESIGN_STATUS.FEASIBLE else self.active_barrier._currentIncumbentInf
      else:
        x: CandidatePoint = self.active_barrier._currentIncumbentFeas if self.active_barrier._currentIncumbentFeas else self.active_barrier._currentIncumbentInf if self.active_barrier._currentIncumbentInf else CandidatePoint(n=len(self.params.baseline),_coords=self.params.baseline)
      if (x is None or not x.evaluated) and self.active_barrier._xFilterInf is not None:
        x = self.active_barrier._currentIncumbentInf

      if (x is None or not x.evaluated):
        x = self._old_x
    # // update _k and _old_x:
    
    if self._old_x is not None and x != self._old_x:
      self._rho = np.sqrt(np.sum([abs(self._old_x.coordinates[i]-x.coordinates[i])**2 for i in range(len(self._old_x.coordinates))]))
      # self._rho *= 2
      self._k += 1
    if self._k > self._k_max:
      self.stop = True
    
    self._old_x = x

    samples = np.zeros((sum(self._ns_dist), len(self.params.baseline)))
    c = 0
    self._seed += np.random.randint(0, 10000)
    np.random.seed(self._seed)
    if x.status is DESIGN_STATUS.FEASIBLE:
      for i in range(len(self._dist)):
        temp = self.generate_samples(x_inc=x, dist= self._dist[i])
        if temp is None:
          continue
        temp = np.unique(temp, axis=0)
        for p in temp:
          if p not in samples:
            samples[c, :] = p
            c += 1
    
    ns_dist_old = self._ns_dist
    self._ns_dist = [int(0.1*xds) for xds in self._ns_dist]

    if isinstance(self.active_barrier, Barrier):
      if self.active_barrier._sec_poll_center is not None and self.active_barrier.get_best_infeasible().evaluated:
        for i in range(len(self._dist)):
          temp = self.generate_samples(x_inc= self.active_barrier.get_best_infeasible(), dist= self._dist[i])
          temp = np.unique(temp, axis=0)
          for p in temp:
            samples = np.vstack((samples, p))
            c += 1
    elif isinstance(self.active_barrier, BarrierMO):
      if self.active_barrier._currentIncumbentFeas is not None and self.active_barrier._currentIncumbentFeas.evaluated:
        for i in range(len(self._dist)):
          temp = self.generate_samples(x_inc= self.active_barrier._currentIncumbentInf, dist= self._dist[i])
          temp = np.unique(temp, axis=0)
          for p in temp:
            samples = np.vstack((samples, p))
            c += 1
    self._ns_dist = ns_dist_old
    samples = np.unique(samples, axis=0)
    return samples
      

@dataclass
class efficient_exploration:
  mesh: Gmesh  = None
  _success: bool = False
  _xmin: CandidatePoint = None
  prob_params: Parameters = None
  sampling_t: int = 3
  _seed: int = 0
  _dtype: DType = None
  iter: int = 1
  vicinity_ratio: np.ndarray = None
  vicinity_min: float = 0.001
  opportunistic: bool = False
  eval_budget: int = 10
  store_cache: bool = True
  check_cache: bool = True
  display: bool = False
  bb_handle: Evaluator = None
  bb_output: List = None
  samples: List[CandidatePoint] = None
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
  log: logger = None
  AS: explore.samplers.activeSampling = None
  best_samples: int = 0
  estGrid: explore.samplers.sampling = None
  n_successes: int = 0 
  bb_eval: Any = None
  activeBarrier: BarrierMO = None

  def __post_init__(self):
    self._xmin = CandidatePoint()
    self.bb_handle = Evaluator()
    self._dtype = DType()

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

  def get_list_of_coords_from_list_of_points(self, xps: List[CandidatePoint] = None) -> np.ndarray:
    coords_array = np.zeros((len(xps), self.dim))
    for i in range(len(xps)):
      coords_array[i, :] = xps[i].coordinates
    
    return coords_array


  def generate_2ngrid(self, vlim: np.ndarray = None, x_incumbent: CandidatePoint = None, p_in: List[float] = [0.01]) -> np.ndarray:
    grid = Dirs2n()
    grid.mesh = copy.deepcopy(self.mesh)
    """ 5- Assign optional algorithmic parameters to the constructed poll instant  """
    grid.seed = int(self.seed + self.iter)
    grid.mesh.dtype.precision = "medium"
    grid.mesh.psize = p_in
    grid.scaling = self.scaling
    grid.dim = self.dim
    grid._n = self.dim
    grid.xmin = x_incumbent
    grid.scale(ub=vlim[:, 0], lb=vlim[:, 1], factor=self.scaling)
    hhm = grid.create_housholder(False, domain=self.xmin.var_type)
    grid.lb = vlim[:, 0]
    grid.ub = vlim[:, 1]
    grid.hmax = self.xmin.hmax
    
    grid.create_poll_set(hhm=hhm,
              ub=grid.ub,
              lb=grid.lb, it=self.iter, var_type=self.xmin.var_type, var_sets=self.xmin.sets, var_link = self.xmin.var_link, c_types=None, is_prim=True)
    
    return self.get_list_of_coords_from_list_of_points(grid.poll_set)


  def HD_grid(self, n: int =3, vlim: np.ndarray = None) -> np.ndarray:
    grid_points = None
    
    if n <= 2* self.dim:
      x_inc = CandidatePoint()
      x_inc.coordinates = self.hashtable.get_best_cache_points(nsamples=n)[0]
      grid_points = self.generate_2ngrid(vlim=vlim, x_incumbent=x_inc, p_in=self.psize)[:n]
    else:
      grid_points: np.ndarray
      for i in range(int(n/(2*self.dim))+1):
        x_inc = CandidatePoint()
        x_inc.coordinates = self.hashtable.get_best_cache_points(nsamples=n)[i]
        temp = self.generate_2ngrid(vlim=vlim, x_incumbent=x_inc, p_in=self.mesh.getDeltaFrameSize())#p_in=1/(self.iter+i)) #add different incumbents from ordered cache matrix
        if i == 0:
          grid_points = temp
        else:
          grid_points = np.vstack((grid_points, temp))
    
    return grid_points[:n, :]

  
  def generate_sample_points(self, nsamples: int = None, samples_in: np.ndarray = None) -> List[CandidatePoint]:
    """ Generate the sample points """
    xlim = []
    self.nvars = len(self.prob_params.baseline)
    is_AS = False
    v = np.empty((self.nvars, 2))
    if self.bb_handle.bb_eval + nsamples > self.eval_budget:
      nsamples = self.eval_budget + self.bb_handle.bb_eval
    if self.xmin and self.iter > 1 and self.sampling_t != SAMPLING_METHOD.ACTIVE.name:
      for i in range(len(self.prob_params.lb)):
        D = abs(self.prob_params.ub[i] - self.prob_params.lb[i])
        lb = copy.deepcopy(self.xmin.coordinates[i]-(D * self.vicinity_ratio[i][0]))
        ub = copy.deepcopy(self.xmin.coordinates[i]+(D * self.vicinity_ratio[i][0]))
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
    is_lhs = False
    self.ns = nsamples
    resize = False
    clipping = True
    # self.seed += np.random.randint(0, 10000)
    if self.sampling_t == SAMPLING_METHOD.FULLFACTORIAL.name:
      sampling = explore.samplers.FullFactorial(ns=nsamples, vlim=v, w=self.weights, c=clipping)
      if clipping:
        resize = True
    elif self.sampling_t == SAMPLING_METHOD.RS.name:
      sampling = explore.samplers.RS(ns=nsamples, vlim=v)
      sampling.options["randomness"] = self.seed
    elif self.sampling_t == SAMPLING_METHOD.HALTON.name:
      sampling = explore.samplers.halton(ns=nsamples, vlim=v, is_ham=True)
    elif self.sampling_t == SAMPLING_METHOD.LH.name:
      sampling = explore.samplers.LHS(ns=nsamples, vlim=v)
      sampling.options["randomness"] = self.seed
      sampling.options["criterion"] = self.sampling_criter
      sampling.options["msize"] = self.mesh.getdeltaMeshSize().coordinates
      is_lhs = True
    else:
      if self.iter == 1 or (len(self.hashtable._cache_dict) if isinstance(self.activeBarrier, Barrier) or self.activeBarrier is None else len(self.hashtable._best_hash_ID)) < nsamples:# or self.n_successes / (self.iter) <= 0.25:
        sampling = explore.samplers.halton(ns=nsamples, vlim=v) if isinstance(self.activeBarrier, Barrier) or self.activeBarrier is None else explore.samplers.LHS(ns=nsamples, vlim=v)
        sampling.options["randomness"] = self.seed + self.iter
        sampling.options["criterion"] = self.sampling_criter
        sampling.options["msize"] = self.mesh.getdeltaMeshSize().coordinates
        sampling.options["varLimits"] = v
      else:
        # if len(self.hashtable._best_hash_ID) > self.best_samples:
        # if len(self.hashtable._best_hash_ID) > self.best_samples:
        self.best_samples = len(self.hashtable._best_hash_ID)
        self.AS = explore.samplers.activeSampling(data=self.hashtable.get_best_cache_points(nsamples=nsamples), 
                                                  n_r=nsamples, 
                                                  vlim=v, 
                                                  kernel_type="Gaussian" if self.dim <= 30 else "Silverman", 
                                                  bw_method="SILVERMAN", 
                                                  seed=int(self.seed + self.iter))
        # estGrid = explore.FullFactorial(ns=nsamples, vlim=v, w=self.weights, c=clipping)
        if self.estGrid is None:
          if self.dim <= 30:
            self.estGrid = explore.samplers.FullFactorial(ns=nsamples, vlim=v, w=self.weights, c=clipping)
          # else:
          #   if (self.iter % 2) == 0:
          #     self.estGrid = explore.samplers.halton(ns=nsamples, vlim=v)
          #   else:
          #     self.estGrid = explore.samplers.RS(ns=nsamples, vlim=v)
          #     self.estGrid.set_options(c=self.sampling_criter, r= self.seed + self.iter)
          #   self.estGrid.options["msize"] = self.mesh.msize

        # self.estGrid.set_options(c=self.sampling_criter, r=int(self.seed + self.iter))
        self.AS.kernel.bw_method = "SILVERMAN"
        if self.dim <=30:
          S = self.estGrid.generate_samples()
        else:
          if True: #(self.iter % 2) == 0:
            if self.estGrid == None:
              self.estGrid = explore.samplers.LHS(ns=nsamples, vlim=v)
              S = self.estGrid.generate_samples()
            else:
              S = self.estGrid.expand_lhs(x=self.hashtable.get_best_cache_points(nsamples=nsamples), n_points=nsamples, method="ExactSE")
          else:
            S = self.HD_grid(n=nsamples, vlim=v)
        if nsamples < len(S):
          self.AS.kernel.estimate_pdf(S[:nsamples, :])
        else:
          self.AS.kernel.estimate_pdf(S)
        is_AS = True

    if self.iter > 1 and is_lhs:
      Ps = copy.deepcopy(sampling.expand_lhs(x=self.map_samples_from_points_to_coords(), n_points=nsamples, method= "basic"))
    else:
      if is_AS:
        Ps = copy.deepcopy(self.AS.resample(size=nsamples, seed=int(self.seed + self.iter)))
      else:
        Ps= copy.deepcopy(sampling.generate_samples())

    if False:
      self.df =  pd.DataFrame(Ps, columns=[f'x{i}' for i in range(self.dim)])
      pd.plotting.scatter_matrix(self.df, alpha=0.2)
      plt.show()
    if resize:
      self.ns = len(Ps)
      nsamples = len(Ps)

    # if self.xmin is not None:
    #   self.visualize_samples(self.xmin.coordinates[0], self.xmin.coordinates[1])
    if self.iter > 1 and is_lhs:
      self.map_samples_from_coords_to_points(Ps[len(Ps)-nsamples:])
    else:
      self.map_samples_from_coords_to_points(Ps)
    return v, Ps
  
  def project_coords_to_mesh(self, x:List[float], ref: List[float] = None):
    pref = Point(self.mesh._n)
    pref.coordinates = ref
    px = Point(self.mesh._n)
    px.coordinates = x
    xProjected: Point = self.mesh.projectOnMesh(px, pref)
    # if ref == None:
    #   ref = [0.]*len(x)
    # if self.xmin.var_type is None:
    #   self.xmin.var_type = [VAR_TYPE.REAL] * len(self.xmin.coordinates)
    # for i in range(len(x)):
    #   if self.xmin.var_type[i] != VAR_TYPE.CATEGORICAL:
    #     if self.xmin.var_type[i] == VAR_TYPE.REAL:
    #       x[i] = ref[i] + (np.round((x[i]-ref[i])/self.mesh.msize) * self.mesh.msize)
    #     else:
    #       x[i] = int(ref[i] + int(int((x[i]-ref[i])/self.mesh.msize) * self.mesh.msize))
    #   else:
    #     x[i] = int(x[i])
    #   if x[i] < self.prob_params.lb[i]:
    #     x[i] = self.prob_params.lb[i] + (self.prob_params.lb[i] - x[i])
    #     if x[i] > self.prob_params.ub[i]:
    #       x[i] = self.prob_params.ub[i]
    #   if x[i] > self.prob_params.ub[i]:
    #     x[i] = self.prob_params.ub[i] - (x[i] - self.prob_params.ub[i])
    #     if x[i] < self.prob_params.lb[i]:
    #       x[i] = self.prob_params.lb[i]

    return xProjected.coordinates

  def map_samples_from_coords_to_points(self, samples: np.ndarray):
    
    for i in range(len(samples)):
      samples[i, :] = self.project_coords_to_mesh(samples[i, :], ref=np.subtract(self.prob_params.ub , self.prob_params.lb).tolist())
    samples = np.unique(samples, axis=0)
    self.samples: List[CandidatePoint] = [0] *len(samples)
    for i in range(len(samples)):
      self.samples[i] = CandidatePoint()
      if self.xmin.var_type is not None:
        self.samples[i].var_type = self.xmin.var_type
      else:
        self.samples[i].var_type = None
      self.samples[i].sets = self.xmin.sets
      self.samples[i].var_link = self.xmin.var_link
      self.samples[i].n_dimensions = len(samples[i, :])
      self.samples[i].coordinates = copy.deepcopy(samples[i, :])
      self.samples[i].direction = Point(self.mesh._n)
      self.samples[i].direction.coordinates = np.subtract(self.xmin.coordinates, self.samples[i].coordinates)
  
  def map_samples_from_points_to_coords(self):
    return np.array([x.coordinates for x in self.samples])


  def gauss_perturbation(self, p: CandidatePoint, npts: int = 5) -> List[CandidatePoint]:
    lb = self.lb
    ub = self.ub
    # np.random.seed(self.seed)
    cs = np.zeros((npts, p.n_dimensions))
    pts: List[CandidatePoint] = [0] * npts
    mp = 1.
    for k in range(p.n_dimensions):
      if p.var_type[k] == VAR_TYPE.REAL:
        cs[:, k] = np.random.normal(loc=p.coordinates[k], scale=self.mesh.getdeltaMeshSize().coordinates[k], size=(npts,))
      elif p.var_type[k] == VAR_TYPE.INTEGER or p.var_type[k] == VAR_TYPE.DISCRETE or p.var_type[k] == VAR_TYPE.CATEGORICAL:
        cs[:, k] = np.random.randint(low=lb[k], high=ub[k], size=(npts,))
      else:
        cs[:, k] = [p.coordinates[k]]*npts
      for i in range(npts):
        if cs[i, k] < lb[k]:
          cs[i, k] = lb[k]
        if cs[i, k] > ub[k]:
          cs[i, k] = ub[k]
    
    for i in range(npts):
      pts[i] = p
      pts[i].coordinates = copy.deepcopy(cs[i, :])
    
    return pts
  
  def evaluate_sample_point(self, index: int):
    """ Evaluate the sample point i on the points set """
    """ Set the dynamic index for this point """
    tic = time.perf_counter()
    self.point_index = index
    if self.log is not None and self.log.isVerbose:
      self.log.log_msg(msg=f"Evaluate sample point # {index}...", msg_type=MSG_TYPE.INFO)
    """ Initialize stopping and success conditions"""
    stop: bool = False
    """ Copy the point i to a trial one """
    xtry: CandidatePoint = self.samples[index]
    """ This is a success bool parameter used for
     filtering out successful designs to be printed
    in the output results file"""
    success = SUCCESS_TYPES.US

    """ Check the cache memory; check if the trial point
     is a duplicate (it has already been evaluated) """
    unique_p_trials: int = 0
    is_duplicate: bool = (self.check_cache and self.hashtable.size > 0 and self.hashtable.is_duplicate(xtry))
    # while is_duplicate and unique_p_trials < 5:
    #   self.log.log_msg(f'Cache hit. Trial# {unique_p_trials}: Looking for a non-duplicate in the vicinity of the duplicate point ...', MSG_TYPE.INFO)
    #   if self.display:
    #     print(f'Cache hit. Trial# {unique_p_trials}: Looking for a non-duplicate in the vicinity of the duplicate point ...')
    #   if xtry.var_type is None:
    #     if self.xmin.var_type is not None:
    #       xtry.var_type = self.xmin.var_type
    #     else:
    #       xtry.var_type = [VAR_TYPE.CONTINUOUS] * len(self.xmin.coordinates)
      
      # xtries: List[Point] = self.gauss_perturbation(p=xtry, npts=len(self.samples)*2)
      # for tr in range(len(xtries)):
      #   is_duplicate = self.hashtable.is_duplicate(xtries[tr])
      #   if is_duplicate:
      #      continue 
      #   else:
      #     xtry = copy.deepcopy(xtries[tr])
      #     break
      # unique_p_trials += 1

    if (is_duplicate):
      if self.log is not None and self.log.isVerbose:
        self.log.log_msg(msg="Cache hit ... Failed to find a non-duplicate alternative.", msg_type=MSG_TYPE.INFO)
      if self.display:
        print("Cache hit ... Failed to find a non-duplicate alternative.")
      stop = True
      bb_eval = copy.deepcopy(self.bb_eval)
      psize = copy.deepcopy(self.mesh.getDeltaFrameSize().coordinates)
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
    if not self.hashtable._isPareto:
      self.hashtable.add_to_best_cache(xtry)
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
    
    if self.log is not None and self.log.isVerbose:
      self.log.log_msg(msg=f"Completed evaluation of point # {index} in {xtry.Eval_time} seconds, ftry={xtry.f}, status={xtry.status.name} and htry={xtry.h}. \n", msg_type=MSG_TYPE.INFO)

    """ Add to the cache memory """
    if self.store_cache:
      self.hashtable.hash_id = xtry

    # if self.save_results or self.display:
    self.bb_eval = self.bb_handle.bb_eval
    self.psize = copy.deepcopy(self.mesh.getDeltaFrameSize().coordinates)
    psize = copy.deepcopy(self.mesh.getDeltaFrameSize().coordinates)

    if xtry < self.xmin:
      success = SUCCESS_TYPES.FS

    if success == SUCCESS_TYPES.FS and self.opportunistic and self.iter > 1:
      stop = True

    """ Check stopping criteria """
    if self.bb_eval >= self.eval_budget:
      self.terminate = True
      stop = True
    return [stop, index, self.bb_handle.bb_eval, success, psize, xtry]

  def master_updates(self, x: List[CandidatePoint], peval, save_all_best: bool = False, save_all:bool = False):
    if peval >= self.eval_budget:
      self.terminate = True
    x_post: List[CandidatePoint] = []
    c = 0
    for xtry in x:
      c += 1
      """ Check success conditions """
      is_infeas_dom: bool = (xtry.status == DESIGN_STATUS.INFEASIBLE and (xtry.h < self.xmin.h) )
      is_feas_dom: bool = (xtry.status == DESIGN_STATUS.FEASIBLE and xtry.fobj < self.xmin.fobj)
      is_infea_improving: bool = (self.xmin.status == DESIGN_STATUS.FEASIBLE and xtry.status == DESIGN_STATUS.INFEASIBLE and (xtry.fobj < self.xmin.fobj and xtry.h <= self.xmin.hmax))
      is_feas_improving: bool = (self.xmin.status == DESIGN_STATUS.INFEASIBLE and xtry.status == DESIGN_STATUS.FEASIBLE and xtry.fobj < self.xmin.fobj)

      success = SUCCESS_TYPES.US
      if ((is_infeas_dom or is_feas_dom)):
        self.success = SUCCESS_TYPES.FS
        success = SUCCESS_TYPES.US  # <- This redundant variable is important
        # for managing concurrent parallel execution
        self.nb_success += 1
        """ Update the post instant """
        del self._xmin
        self._xmin = CandidatePoint()
        self._xmin = copy.deepcopy(xtry)
        if self.display:
          if self._dtype.dtype == np.float64:
            print(f"Success: fmin = {self.xmin.f} (hmin = {self.xmin.h:.15})")
          elif self._dtype.dtype == np.float32:
            print(f"Success: fmin = {self.xmin.f} (hmin = {self.xmin.h:.6})")
          else:
            print(f"Success: fmin = {self.xmin.f} (hmin = {self.xmin.h:.18})")

        self.mesh.psize_success = copy.deepcopy(self.mesh.getDeltaFrameSize().coordinates)
        self.mesh.psize_max = copy.deepcopy(max(self.mesh.getDeltaFrameSize().coordinates))

      if (save_all_best and success == SUCCESS_TYPES.FS) or (save_all):
        x_post.append(xtry)
    
    if self.success == SUCCESS_TYPES.FS:
      self.n_successes += 1
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