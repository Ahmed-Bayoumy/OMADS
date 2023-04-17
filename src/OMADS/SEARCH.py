import json
from multiprocessing import freeze_support, cpu_count
import os
import sys
import time
import SLML as explore
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from OMADS.POLL import Options, VAR_TYPE, Parameters, Point, Cache, Evaluator, OrthoMesh, PostMADS, Output, DType
import copy
from typing import List, Dict, Any, Callable, Protocol, Optional
import concurrent.futures
from matplotlib import pyplot as plt


@dataclass
class global_exploration:
  mesh: OrthoMesh  = field(default_factory=OrthoMesh)
  _success: bool = False
  _xmin: Point = Point()
  prob_params: Parameters = Parameters()
  sampling_t: int = 3
  _seed: int = 0
  _dtype: DType = DType()
  iter: int = 1
  vicinity_ratio: np.ndarray = field(default=np.ones((1)))
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
        lb = copy.deepcopy(self.xmin.coordinates[i]-abs(self.xmin.coordinates[i] * self.vicinity_ratio[i]))
        ub = copy.deepcopy(self.xmin.coordinates[i]+abs(self.xmin.coordinates[i] * self.vicinity_ratio[i]))
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

    if self.sampling_t == explore.SAMPLING_METHOD.FULLFACTORIAL.value:
      sampling = explore.FullFactorial(ns=nsamples, vlim=v, w=np.array([0.1, 0.05]), c=True)
    elif self.sampling_t == explore.SAMPLING_METHOD.LH.value: 
      sampling = explore.LHS(ns=nsamples, vlim=v)
      self.seed += np.random.randint(0, 10000)
      sampling.options["randomness"] = self.seed
      sampling.options["criterion"] = self.sampling_criter
      sampling.options["msize"] = self.mesh.msize
    elif self.sampling_t == explore.SAMPLING_METHOD.RS.value:
      sampling = explore.RS(ns=nsamples, vlim=v)
    elif self.sampling_t == explore.SAMPLING_METHOD.HALTON.value:
      sampling = explore.halton(ns=nsamples, vlim=v, is_ham=True)
    
    Ps= copy.deepcopy(sampling.generate_samples())
    
    # if self.xmin is not None:
    #   self.visualize_samples(self.xmin.coordinates[0], self.xmin.coordinates[1])
    self.map_samples_from_coords_to_points(Ps)
    return v, Ps
    

  def map_samples_from_coords_to_points(self, samples: np.ndarray):
    self.samples = [0] *self.ns
    for i in range(len(samples)):
      self.samples[i] = Point()
      self.samples[i].var_type = self.xmin.var_type
      self.samples[i].sets = self.xmin.sets
      self.samples[i].var_link = self.xmin.var_link
      self.samples[i]._coords = copy.deepcopy(samples[i])
  
  def evaluate_point(self, index: int):
    """ Evaluate the sample point i on the poll set """
    """ Set the dynamic index for this point """
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
    if (
        self.check_cache
        and self.hashtable.size > 0
        and self.hashtable.is_duplicate(xtry)
    ):
      if self.display:
        print("Cache hit ...")
      stop = True
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
      Evaluate the sample point:
        - Evaluate objective function
        - Evaluate constraint functions (can be an empty vector)
        - Aggregate constraints
        - Penalize the objective (extreme barrier)
    """
    xtry.__eval__(self.bb_output)

    """ Add to the cache memory """
    if self.store_cache:
      self.hashtable.hash_id = xtry

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

  def master_updates(self, x: List[Point], peval):
    if peval >= self.eval_budget:
      self.terminate = True
    for xtry in x:
      """ Check success conditions """
      if xtry < self.xmin:
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
@dataclass
class search_sampling:
  method: int = explore.SAMPLING_METHOD.LH.value
  ns: int = 3        
  visualize: bool = False
  criterion: str = None
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
    ev = Evaluator(**self.data["evaluator"])
    sampling = search_sampling(**self.data["sampling"])
    ev.dtype.precision = options.precision
    if param.constants != None:
      ev.constants = copy.deepcopy(param.constants)
    
    """ 2- Initialize iteration number and construct a point instant for the starting point """
    iteration: int = 0
    x_start = Point()
    """ 3- Construct an instant for the poll 2n orthogonal directions class object """
    extend = options.extend is not None and isinstance(options.extend, global_exploration)
    if not extend:
      search = global_exploration()
      if param.Failure_stop != None and isinstance(param.Failure_stop, bool):
        search.Failure_stop = param.Failure_stop
      search.samples = []
      search.dtype.precision = options.precision
      """ 4- Construct an instant for the mesh subclass object by inheriting
      initial parameters from mesh_params() """
      search.mesh = OrthoMesh()
      search.sampling_t = sampling.method
      search.ns = sampling.ns
      search.sampling_criter = sampling.criterion
      search.visualize = sampling.visualize
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
    out = Output(file_path=param.post_dir, vnames=param.var_names)
    if options.display:
      print("End of the evaluation of the starting points")
    iteration += 1

    return iteration, x_start, search, options, param, post, out
  

  

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
  iteration, xmin, search, options, param, post, out = PreExploration(data).initialize_from_dict()

  """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(options.seed)
  else:
    np.random.seed(int(args[3]))

  """ Start the count down for calculating the runtime indicator """
  tic = time.perf_counter()
  peval = 0
  if search.visualize:
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ps = None



  while True:
    search.mesh.update()
    """ Create the set of poll directions """
    vv, _ = search.generate_sample_points(int(((search.dim+1)/2)*((search.dim+2)/2)) if search.ns is None else search.ns)
    if search.visualize:
      ps = visualize(search.samples, 0, 1, search.mesh.msize, vv, fig, ax, search.xmin, ps)

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
        peval += 1
        if search.terminate:
          break
        f = search.evaluate_point(it)
        xt.append(f[-1])
        if not f[0]:
          post.bb_eval.append(search.bb_handle.bb_eval)
          post.iter.append(iteration)
          post.psize.append(search.mesh.psize)
          if options.save_results:
            if options.save_all_best and not f[3]:
              continue
            post.poll_dirs.append(search.samples[it])
        else:
          continue

    else:
      search.point_index = -1
      """ Parallel evaluation for points in the poll set """
      with concurrent.futures.ProcessPoolExecutor(options.np) as executor:
        results = [executor.submit(search.evaluate_point,
                       it) for it in range(len(search.samples))]
        for f in concurrent.futures.as_completed(results):
          # if f.result()[0]:
          #     executor.shutdown(wait=False)
          # else:
          if options.save_results or options.display:
            if options.save_all_best and not f.result()[3]:
              continue
            peval = peval +1
            search.bb_eval = peval
            post.bb_eval.append(peval)
            post.iter.append(iteration)
            post.poll_dirs.append(search.samples[f.result()[1]])
            post.psize.append(f.result()[4])
          xt.append(f.result()[-1])
  
    search.master_updates(xt, peval)

    """ Update the xmin in post"""
    post.xmin = copy.deepcopy(search.xmin)

    """ Updates """
    if search.success:
      search.mesh.psize = np.multiply(search.mesh.psize, 2, dtype=search.dtype.dtype)
    else:
      search.mesh.psize = np.divide(search.mesh.psize, 2, dtype=search.dtype.dtype)

    if options.display:
      print(post)

    Failure_check = search.Failure_stop is not None and search.Failure_stop and not search.success
    if (Failure_check) or (abs(search.mesh.psize) < options.tol or search.bb_eval >= options.budget or search.terminate):
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

  return output, search



def visualize(points: List[Point], hc_index, vc_index, msize, vlim, fig, axes, pmin, ps = None):
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

  if ps == None:
    ps, = axes.plot(x, y, 'ob')
  else:
    ps.set_xdata(x)
    ps.set_ydata(y)
  fig.canvas.draw()
  fig.canvas.flush_events()


  # axes.set_xticks(minor_ticksx,major=True)
  # axes.set_yticks(minor_ticksy,major=True)
  axes.set_title("Sample points")
  # axes.grid(which="major",alpha=0.3)
  ps.set_xdata(x)
  ps.set_ydata(y)
  axes.plot(x, y, 'ok', alpha=0.08)
  axes.plot(xmin, ymin, 'og')


  plt.show()
  plt.pause(0.05)
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
       "failure_stop": False}
  sampling = {
    "method": explore.SAMPLING_METHOD.LH.value,
    "ns": 10,
    "visualize": False
  }
  options = {"seed": 0, "budget": 100000, "tol": 1e-12, "display": True}

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