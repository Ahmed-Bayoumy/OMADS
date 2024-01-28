from .Point import Point
from .Barriers import *
from ._common import *
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Dirs2n:
  """This is the orthognal 2n-directions class used for the poll step

    :param _poll_dirs: Poll set (list of points)
    :param _point_index: List of point indices
    :param _n: Number of directions
    :param _defined: A boolean that indicate if the poll points are defined
  """
  _poll_dirs: List[Point] = field(default_factory=list)
  _point_index: List[int] = field(default_factory=list)
  _n: int = 0
  _defined: List[bool] = field(default_factory=lambda: [False])
  scaling: List[List[float]] = field(default_factory=list)
  _xmin: Point = Point()
  _x_sc: Point = Point()
  _nb_success: int = 0
  _bb_eval: int = field(default_factory=int)
  _psize: float = field(default_factory=float)
  _iter: int = field(default_factory=int)
  hashtable: Cache = field(default_factory=Cache)
  _check_cache: bool = True
  _display: bool = True
  _store_cache: bool = True
  _save_results = True
  mesh: OrthoMesh = field(default_factory=OrthoMesh)
  _opportunistic: bool = False
  _eval_budget: int = 100
  _dtype: DType = DType()
  bb_handle: Evaluator = field(default_factory=Evaluator)
  _success: bool = False
  _seed: int = 0
  _terminate: bool = False
  _bb_output: List[float] = field(default_factory=list)
  Failure_stop: bool = None
  RHO: float = MPP.RHO
  LAMBDA: List[float] = None
  hmax: float = 1.
  log: logger = None
  n_successes: int = 0

  @property
  def x_sc(self) -> Point:
    return self._x_sc
  
  @x_sc.setter
  def x_sc(self, value: Point):
    self._x_sc = value
  

  @property
  def bb_output(self) -> List[float]:
    return self._bb_output

  @bb_output.setter
  def bb_output(self, other: List[float]):
    self._bb_output = other

  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, other: DType):
    self._dtype = other

  @property
  def point_index(self):
    return self._point_index

  @point_index.setter
  def point_index(self, other: int):
    if other == -1:
      self._point_index = []
    else:
      self._point_index.append(other)

  @property
  def terminate(self):
    return self._terminate

  @terminate.setter
  def terminate(self, other: bool):
    self._terminate = other

  @property
  def seed(self):
    return self._seed

  @seed.setter
  def seed(self, other: int):
    self._seed = other

  @property
  def success(self):
    return self._success

  @success.setter
  def success(self, other: bool):
    self._success = other

  @property
  def eval_budget(self):
    return self._eval_budget

  @eval_budget.setter
  def eval_budget(self, other: int):
    self._eval_budget = other

  @property
  def opportunistic(self):
    return self._opportunistic

  @opportunistic.setter
  def opportunistic(self, other: bool):
    self._opportunistic = other

  @property
  def save_results(self):
    return self._save_results

  @save_results.setter
  def save_results(self, other: bool):
    self._save_results = other

  @property
  def store_cache(self):
    return self._store_cache

  @store_cache.setter
  def store_cache(self, other: bool):
    self._store_cache = other

  @property
  def check_cache(self):
    return self._check_cache

  @check_cache.setter
  def check_cache(self, other: bool):
    self._check_cache = other

  @property
  def display(self):
    return self._display

  @display.setter
  def display(self, other: bool):
    self._display = other

  @property
  def bb_eval(self):
    return self._bb_eval

  @bb_eval.setter
  def bb_eval(self, other: int):
    self._bb_eval = other

  @property
  def psize(self):
    return self._psize

  @psize.setter
  def psize(self, other: float):
    self._psize = other

  @property
  def iter(self):
    return self._iter

  @iter.setter
  def iter(self, other: int):
    self._iter = other

  @property
  def poll_dirs(self):
    return self._poll_dirs

  @poll_dirs.setter
  def poll_dirs(self, p: Point):
    self._poll_dirs.append(p)

  @poll_dirs.deleter
  def poll_dirs(self):
    del self._poll_dirs
    self._poll_dirs = []

  @property
  def dim(self):
    return self._n

  @dim.setter
  def dim(self, n: int):
    self._n = n

  @dim.deleter
  def dim(self):
    self._n = 0

  @property
  def defined(self):
    return any(self._defined)

  @defined.setter
  def defined(self, defined):
    self._defined = defined

  @defined.deleter
  def defined(self):
    del self._defined

  @property
  def xmin(self):
    return self._xmin

  @xmin.setter
  def xmin(self, other: Point):
    self._xmin = other

  @property
  def nb_success(self):
    return self._nb_success

  @nb_success.setter
  def nb_success(self, other: int):
    self._nb_success = other

  def generate_dir(self):
    return np.random.rand(self._n).tolist()

  def ran(self):
    return np.random.random(self._n).astype(dtype=self._dtype.dtype)

  def create_housholder(self, is_rich: bool, domain: List[int] = None, is_oneDir: bool=False) -> np.ndarray:
    """Create householder matrix

    :param is_rich:  A flag that indicates if the rich direction option is enabled
    :type is_rich: bool
    :return: The householder matrix
    :rtype: np.ndarray
    """
    if domain is None:
      domain = [VAR_TYPE.CONTINUOUS] * self._n
    elif len(domain) != self._n:
      raise IOError("Number of dimensions doesn't match the size of the variables type list invoked to Dirs2n::create_householder.")
    elif not isinstance(domain, list):
      raise IOError("The variables domain type input invoked to Dirs2n::create_householder should be of type list.")
    
    hhm: np.ndarray
    if is_rich:
      v_dir = copy.deepcopy(self.ran())
      v_dir_array = np.array(v_dir, dtype=self._dtype.dtype)
      v_dir_array = np.divide(v_dir_array,
                  (np.linalg.norm(v_dir_array,
                          2).astype(dtype=self._dtype.dtype)),
                  dtype=self._dtype.dtype)
      hhm = np.subtract(np.eye(self.dim, dtype=self._dtype.dtype),
                np.multiply(2.0, np.outer(v_dir_array, v_dir_array.T),
                      dtype=self._dtype.dtype),
                dtype=self._dtype.dtype)
    else:
      hhm = np.eye(self.dim, dtype=self._dtype.dtype)
    hhm = np.dot(hhm, np.diag((np.abs(hhm, dtype=self._dtype.dtype)).max(1) ** (-1)))
    # Rounding( and transpose)
    tmp = np.multiply(self.mesh.rho, hhm, dtype=self._dtype.dtype)
    hhm = np.transpose(np.multiply(self.mesh.msize, np.ceil(tmp), dtype=self._dtype.dtype))
    hhm = np.dot(hhm, self.scaling)

    for i in range(len(domain)):
      if domain[i] == VAR_TYPE.DISCRETE or domain[i] == VAR_TYPE.BINARY or domain[i] == VAR_TYPE.INTEGER:
        hhm[i][i] = int(np.floor((-1 if i%2 else 1) - 2**self.mesh.msize))
      elif domain[i] == VAR_TYPE.CATEGORICAL:
        hhm[i][i] = np.ceil(np.random.random(1).astype(dtype=self._dtype.dtype))
      else:
        for j in range(len(domain)):
          if domain[j] != VAR_TYPE.CONTINUOUS:
            hhm[i][j] = int(np.floor(-1 + 2**self.mesh.msize))
    
    if is_oneDir:
      return hhm
    else:
      hhm = np.vstack((hhm, -hhm))


    return hhm

  def create_poll_set(self, hhm: np.ndarray, ub: List[float], lb: List[float], it: int, var_type: List, var_sets: Dict, var_link: List[str], c_types: List[BARRIER_TYPES]=None, is_prim: bool = True):
    """Create the poll directions

    :param hhm: Householder matrix
    :type hhm: np.ndarray
    :param ub: Variables upper bound
    :type ub: List[float]
    :param lb: Variables lower bound
    :type lb: List[float]
    :param it: iteration
    :type it: int
    """
    if is_prim:
      del self.poll_dirs
      temp = np.add(hhm, np.array(self.xmin.coordinates), dtype=self._dtype.dtype)
    else:
      temp = np.add(hhm, np.array(self.x_sc.coordinates), dtype=self._dtype.dtype)
    # np.random.seed(self._seed)
    np.random.seed(seed=self.seed+self.iter*123)
    temp = np.random.permutation(temp)
    temp = np.minimum(temp, ub, dtype=self._dtype.dtype)
    temp = np.maximum(temp, lb, dtype=self._dtype.dtype)
    temp = np.unique(temp, axis=0)
    if isinstance(temp, list) or isinstance(temp, np.ndarray):
      ndirs = len(temp) if isinstance(temp[0], list) or isinstance(temp[0], np.ndarray) else 1
    else:
      ndirs = 0
    for k in range(ndirs):
      tmp = Point()
      tmp.constraints_type = copy.deepcopy([xb for xb in c_types] if isinstance(c_types, list) else [c_types])
      tmp.sets = copy.deepcopy(var_sets)
      tmp.var_type = copy.deepcopy(var_type)
      tmp.var_link = copy.deepcopy(var_link)
      tmp.coordinates = temp[k]
      tmp.dtype.precision = self.dtype.precision
      self.poll_dirs = tmp
      del tmp
    del temp

    self.iter = it

  def scale(self, ub: List[float], lb: List[float], factor: float = 10.0):
    self.scaling = np.divide(subtract(ub, lb, dtype=self._dtype.dtype),
                 factor, dtype=self._dtype.dtype)
    if any(np.isinf(self.scaling)):
      for k, x in enumerate(np.isinf(self.scaling)):
        if x:
          self.scaling[k][0] = 1.0
    s_array = np.diag(self.scaling)

    self.scaling = []
    for k in range(len(s_array)):
      temp: List[float] = []
      for j in range(len(s_array[k])):
        temp.append(s_array[k][j])
      self.scaling.append(temp)
      del temp
  
  def directional_scaling(self, p: Point, npts: int = 5) -> List[Point]:
    lb = self.lb
    ub = self.ub
    # np.random.seed(self.seed)
    scaling = [self.mesh.msize, 2*self.mesh.msize]
    p_trials: List[Point] = [0]*len(scaling)
    for k in range(len(scaling)):
      p_trials[k] = copy.deepcopy(p)
      p_trials[k].coordinates = copy.deepcopy(np.subtract(p_trials[k].coordinates, scaling[k]))
      for i in range(p_trials[k].n_dimensions):
        if p_trials[k].coordinates[i] < lb[k]:
          p_trials[k].coordinates[i] = copy.deepcopy(lb[k])
        if p_trials[k].coordinates[i] > ub[k]:
          p_trials[k].coordinates[i] = copy.deepcopy(ub[k])
    
    return p_trials
  
  def gauss_perturbation(self, p: Point, npts: int = 5) -> List[Point]:
    lb = self.lb
    ub = self.ub
    # np.random.seed(self.seed)
    cs = np.zeros((npts, p.n_dimensions))
    pts: List[Point] = [0] * npts
    mp = 1.
    for k in range(p.n_dimensions):
      if p.var_type[k] == VAR_TYPE.CONTINUOUS:
        cs[:, k] = np.random.normal(loc=p.coordinates[k], scale=self.mesh.msize, size=(npts,))
      elif p.var_type[k] == VAR_TYPE.INTEGER or p.var_type[k] == VAR_TYPE.CATEGORICAL or p.var_type[k] == VAR_TYPE.DISCRETE:
        cs[:, k] = np.random.randint(low=lb[k], high=ub[k], size=(npts,))
        for i in range(npts):
          cs[i, k] = int(cs[i, k])
      for i in range(npts):
        if cs[i, k] < lb[k]:
          cs[i, k] = lb[k]
        if cs[i, k] > ub[k]:
          cs[i, k] = ub[k]
    
    for i in range(npts):
      pts[i] = p
      pts[i].coordinates = copy.deepcopy(cs[i, :])
    
    return pts


  def eval_poll_point(self, index: int):
    """ Evaluate the point i on the poll set """
    """ Set the dynamic index for this point """
    tic = time.perf_counter()
    self.point_index = index
    if self.log is not None and self.log.isVerbose:
      self.log.log_msg(msg=f"Evaluate poll point # {index}...", msg_type=MSG_TYPE.INFO)
    """ Initialize stopping and success conditions"""
    stop: bool = False
    """ Copy the point i to a trial one """
    xtry: Point = self.poll_dirs[index]
    """ This is a success bool parameter used for
     filtering out successful designs to be printed
    in the output results file"""
    success = False

    """ Check the cache memory; check if the trial point
     is a duplicate (it has already been evaluated) """
    unique_p_trials: int = 0
    is_duplicate: bool = (self.check_cache and self.hashtable.size > 0 and self.hashtable.is_duplicate(xtry))
    # TODO: The commented logic below needs more investigation to make sure that it doesn't hurt.
    # while is_duplicate and unique_p_trials < 5:
    #   if self.display:
    #     print(f'Cache hit. Trial# {unique_p_trials}: Looking for a non-duplicate along the poll direction where the duplicate point is located...')
    #   if xtry.var_type is None:
    #     if self.xmin.var_type is not None:
    #       xtry.var_type = self.xmin.var_type
    #     else:
    #       xtry.var_type = [VAR_TYPE.CONTINUOUS] * len(self.xmin.coordinates)
    #   xtries: List[Point] = self.directional_scaling(p=xtry, npts=len(self.poll_dirs)*2)
    #   for tr in range(len(xtries)):
    #     is_duplicate = self.hashtable.is_duplicate(xtries[tr])
    #     if is_duplicate:
    #        continue 
    #     else:
    #       xtry = copy.deepcopy(xtries[tr])
    #       break
    #   unique_p_trials += 1

    if (is_duplicate):
      if self.log is not None and self.log.isVerbose:
        self.log.log_msg(msg="Cache hit ... Failed to find a non-duplicate alternative.", msg_type=MSG_TYPE.INFO)
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
    xtry.__eval__(self.bb_output)
    self.hashtable.add_to_best_cache(xtry)
    self.hmax = copy.deepcopy(xtry.hmax)
    toc = time.perf_counter()
    xtry.Eval_time = (toc - tic)
    

    """ Update multipliers and penalty """
    if self.LAMBDA == None:
      self.LAMBDA = self.xmin.LAMBDA
    if len(xtry.cPB) > len(self.LAMBDA):
      self.LAMBDA += [self.LAMBDA[-1]] * abs(len(self.LAMBDA)-len(xtry.cPB))
    if len(xtry.cPB) < len(self.LAMBDA):
      del self.LAMBDA[len(xtry.cPB):]
    for i in range(len(xtry.cPB)):
      if self.RHO == 0.:
        self.RHO = 0.001
      self.LAMBDA[i] = copy.deepcopy(max(self.dtype.zero, self.LAMBDA[i] + (1/self.RHO)*xtry.cPB[i]))
    
    if xtry.status == DESIGN_STATUS.FEASIBLE:
      self.RHO *= copy.deepcopy(0.5)

    if self.log is not None and self.log.isVerbose:
      self.log.log_msg(msg=f"Completed evaluation of point # {index} in {xtry.Eval_time} seconds, ftry={xtry.f}, status={xtry.status.name} and htry={xtry.h}. \n", msg_type=MSG_TYPE.INFO)

    # if xtry < self.xmin:
    #   self.success = True
    #   success = True

    """ Add to the cache memory """
    if self.store_cache:
      self.hashtable.hash_id = xtry

    # if self.save_results or self.display:
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
    for xtry in x:
      """ Check success conditions """
      is_infeas_dom: bool = (xtry.status == DESIGN_STATUS.INFEASIBLE and (xtry.h < self.xmin.h) )
      is_feas_dom: bool = (xtry.status == DESIGN_STATUS.FEASIBLE and xtry.fobj < self.xmin.fobj)
      is_infea_improving: bool = (self.xmin.status == DESIGN_STATUS.FEASIBLE and xtry.status == DESIGN_STATUS.INFEASIBLE and (xtry.fobj < self.xmin.fobj and xtry.h <= self.xmin.hmax))
      is_feas_improving: bool = (self.xmin.status == DESIGN_STATUS.INFEASIBLE and xtry.status == DESIGN_STATUS.FEASIBLE and xtry.fobj < self.xmin.fobj)
      
      success = False
      if ((is_infeas_dom or is_feas_dom)):
        self.success = True
        self.n_successes += 1
        success = True  # <- This redundant variable is important
        # for managing concurrent parallel execution
        self.nb_success += 1
        """ Update the post instant """
        del self._xmin
        self._xmin = Point()
        self._xmin = copy.deepcopy(xtry)
        self.hmax = copy.deepcopy(xtry.hmax)
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
