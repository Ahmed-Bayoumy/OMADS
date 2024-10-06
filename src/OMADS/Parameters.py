from dataclasses import dataclass
import os
from typing import List, Dict, Optional
import warnings
import numpy as np
from .Point import Point
from ._globals import DType, VAR_TYPE, BARRIER_TYPES, MESH_TYPE
import copy

@dataclass
class Parameters:
  """ Variables and algorithmic parameters 
  
    :param baseline: Baseline design point (initial point ``x0``)
    :param lb: The variables lower bound
    :param ub: The variables upper bound
    :param var_names: The variables name
    :param scaling: Scaling factor (can be defined as a list (assigning a factor for each variable) or a scalar value that will be applied on all variables)
    :param post_dir: The location and name of the post directory where the output results file will live in (if any)
  """
  _n: Optional[int] = None
  baseline: Optional[List[float]] = None
  lb: Optional[List[float]] = None
  ub: Optional[List[float]] = None
  var_names: Optional[List[str]] = None
  fun_names: Optional[List[str]] = None
  scaling: Optional[List[float]] = None
  post_dir: Optional[str] = os.path.abspath("./")
  var_type: Optional[List[str]] = None
  var_sets: Optional[Dict] = None
  constants: Optional[List] = None
  constants_name: Optional[List] = None
  failure_stop: Optional[bool] = None
  problem_name: str = "unknown"
  best_known: Optional[List[float]] = None
  constraints_type: Optional[List[BARRIER_TYPES]] = None
  function_weights: Optional[List[float]] = None 
  h_max: float = 0
  RHO: float = 0.00005
  LAMBDA: Optional[List[float]] = None
  name: str = "undefined"
  nobj: int = 1
  ref_point: Optional[List[float]] = None
  lhs_search_initialization: Optional[bool] = False


  # Mesh options
  meshType: str = MESH_TYPE.ORTHO.name
  fixed_variables: Optional[Point] = None
  granularity: Optional[Point] = None
  minMeshSize: Optional[Point] = None
  minFrameSize: Optional[Point] = None
  initialMeshSize: Optional[Point] = None
  initialFrameSize: Optional[Point] = None
  warningInitialFrameSizeReset: bool = True
  x0: Optional[Point] = None
  _initialized_and_checked: bool = False
  isPareto: bool = False
  incumbentincumbentSelectionParam: int = 1
  barrierInitializedFromCache: bool = True

  def __init__(
      self,
      baseline: List[float] = None,
      lb: List[float] = None,
      ub: List[float] = None,
      var_names: List[str] = None,
      fun_names: List[str] = None,
      function_weights: List[float] = None,
      scaling: float = 10.0,
      post_dir: str = os.path.abspath("./"),
      var_type: List[str] = None,
      var_sets: Dict = None,
      constants: List = None,
      constants_name: List = None,
      Failure_stop: bool = None,
      problem_name: str = "unknown",
      best_known: List[float] = None,
      constraints_type: List[BARRIER_TYPES] = None,
      h_max: float = 0,
      RHO: float = 0.00005,
      LAMBDA: List[float] = None,
      name: str = "undefined",
      meshType: str = MESH_TYPE.ORTHO.name,
      fixed_variables: List[float] = None,
      granularity: List[float] = None,
      minMeshSize: List[float] = None,
      minFrameSize: List[float] = None,
      initialMeshSize: List[float] = None,
      initialFrameSize: List[float] = None,
      isPareto: bool = False,
      nobj: int=1,
      incumbentincumbentSelectionParam: int=1,
      barrierInitializedFromCache:bool =True,
      ref_point: List[float]=None,
      lhs_search_initialization: bool = False):
    self.incumbentincumbentSelectionParam = incumbentincumbentSelectionParam
    self.barrierInitializedFromCache = barrierInitializedFromCache
    self.nobj = nobj
    self.baseline = baseline
    self._n = len(baseline)
    self.x0 = Point(self._n)
    self.x0.coordinates = self.baseline
    self.lb = lb
    self.ub = ub
    self.var_names = var_names if var_names else [f'x_{i}' for i in range(self._n)]
    self.fun_names = fun_names if fun_names else ["fobj"]
    self.function_weights = (np.divide(function_weights, np.sum(function_weights))).tolist() if function_weights else [1/(self.nobj)] * self.nobj
    self.scaling = scaling
    self.post_dir = post_dir
    self.var_type = var_type
    self.constants = constants
    self.constants_name = constants_name
    self.failure_stop: bool = Failure_stop
    self.problem_name = problem_name
    self.best_known = best_known
    self.constraints_type = constraints_type
    self.h_max = h_max
    self.RHO = RHO
    self.LAMBDA = LAMBDA
    self.name = name
    self.var_sets = var_sets
    self.isPareto = isPareto
    self.lhs_search_initialization = lhs_search_initialization
    # Mesh options
    self.meshType = meshType
    point_init = Point()
    point_init.reset(self._n, d=0)
    if fixed_variables:
      self.fixed_variables = point_init
      self.fixed_variables
      self.fixed_variables.coordinates = fixed_variables
    else:
      self.fixed_variables = None

    self.granularity = copy.deepcopy(point_init)
    self.minMeshSize = copy.deepcopy(point_init)
    self.minFrameSize = copy.deepcopy(point_init)
    self.initialMeshSize = copy.deepcopy(point_init)
    self.initialFrameSize = copy.deepcopy(point_init)
    if granularity:
      self.granularity.coordinates = granularity
    if minMeshSize:
      self.minMeshSize.coordinates = minMeshSize
    else:
      self.minMeshSize.coordinates = [1E-9] * self._n
    if minFrameSize:
      self.minFrameSize.coordinates = minFrameSize
    else:
      self.minFrameSize.coordinates = [1E-9] * self._n
    if initialMeshSize:
      self.initialMeshSize.coordinates = initialMeshSize
    if initialFrameSize:
      self.initialFrameSize.coordinates = initialFrameSize
    self.warningInitialFrameSizeReset: bool = True

    if self.constraints_type is not None and isinstance(self.constraints_type, list):
      for i in range(len(self.constraints_type)):
        if self.constraints_type[i] == BARRIER_TYPES.PB.name or self.constraints_type[i] == BARRIER_TYPES.PB:
          self.constraints_type[i] = BARRIER_TYPES.PB
        elif self.constraints_type[i] == BARRIER_TYPES.RB.name or self.constraints_type[i] == BARRIER_TYPES.RB:
          self.constraints_type[i] = BARRIER_TYPES.RB
        elif self.constraints_type[i] == BARRIER_TYPES.PEB.name or self.constraints_type[i] == BARRIER_TYPES.PEB:
          self.constraints_type[i] = BARRIER_TYPES.PEB
        else:
          self.constraints_type[i] = BARRIER_TYPES.EB
    elif self.constraints_type is not None:
      self.constraints_type = BARRIER_TYPES(self.constraints_type)

    if self.LAMBDA is None:
      self.LAMBDA = [0]
    if not isinstance(self.LAMBDA, list):
      self.LAMBDA = [self.LAMBDA]

    if self.var_type is not None:
      c = 0
      for k in self.var_type:
        c+= 1
        if k.lower()[0] == "d":
          if self.var_sets is not None and isinstance(self.var_sets, dict):
            if self.var_sets[k.split('_')[1]] is not None:
              if self.ub[c-1] > len(self.var_sets[k.split('_')[1]])-1:
                self.ub[c-1] = len(self.var_sets[k.split('_')[1]])-1
              if self.lb[c-1] < 0:
                self.lb[c-1] = 0
    if constants:
      self.fixed_variables = point_init
      self.fixed_variables.coordinates = constants

    if self.granularity and self.granularity.size != self._n:
      if self.granularity.size > 0 and self.granularity.is_all_defined():
        raise IOError(f'Parameter granularity has dimension {self.granularity.size}  which is different from problem dimension {self._n}')
      self.granularity.reset(self._n, 0.0)
    
    for i in range(self.n):
      if not self.granularity.defined[i]:
        self.granularity[i] = 0.
      elif self.granularity[i] < 0.:
        raise IOError("Check: invalid granular variables (negative values)")
    
    if self.var_type is None or len(self.var_type) <= 0:
      self.var_type = [VAR_TYPE.REAL.name] * self.n
    self.set_min_mesh_parameters()
    self.set_min_frame_parameters()
    self.set_initial_mesh_parameters()
    self.x0.check_for_granularity(g=self.granularity, name="baseline")
    self.minMeshSize.check_for_granularity(g=self.granularity, name="minMeshSize")
    self.minFrameSize.check_for_granularity(g=self.granularity, name="minFrameSize")
    self.initialMeshSize.check_for_granularity(g=self.granularity, name="initialMeshSize")
    self.initialFrameSize.check_for_granularity(g=self.granularity, name="initialFrameSize")

    self._initialized_and_checked = True
    self.ref_point = ref_point


  def set_initial_mesh_parameters(self):
    if self.initialMeshSize.is_all_defined() and self.initialMeshSize.size != self.n:
      raise IOError(f"INITIAL_MESH_SIZE has dimension {self.initialMeshSize.size} which is different from problem dimension {self.n}")
    
    if self.initialFrameSize.is_all_defined() and self.initialFrameSize.size != self.n:
      raise IOError(f"INITIAL_FRAME_SIZE has dimension {self.initialFrameSize.size} which is different from problem dimension {self.n}")
    
    if self.initialMeshSize.is_all_defined() and self.initialFrameSize.is_all_defined():
      self.initialMeshSize.reset(self._n)

    if not self.initialMeshSize.is_all_defined():
      self.initialMeshSize.reset(self.n, 0.)
    
    if not self.initialFrameSize.is_all_defined():
      self.initialFrameSize.reset(self.n, 0.)
    lb = self.lb
    ub = self.ub
    for j in range(self.n):
      if self.lb[j] == None or self.baseline[j] < self.lb[j]:
        lb[j] = self.baseline[j]
      if self.ub[j] == None or self.baseline[j] > self.ub[j]:
        ub[j] = self.baseline[j]

    for i in range(self.n):
      if self.initialMeshSize.defined[i]:
        if self.initialFrameSize.defined[i] and self.warningInitialFrameSizeReset:
          self.warningInitialFrameSizeReset = False
          warnings.warn("Initial frame size reset from initial mesh")
        self.minFrameSize[i] = self.initialMeshSize[i] * np.power(self.n, 0.5)
        self.initialFrameSize[i] = self.initialFrameSize.next_mult(g=self.granularity[i], i=i)
        if self.initialFrameSize[i] < self.minFrameSize[i]:
          self.initialFrameSize[i] = self.minFrameSize[i]

      if not self.initialFrameSize.defined[i]:
        if lb[i] is not None and ub[i] is not None:
          self.initialFrameSize[i] = (ub[i] - lb[i]) / 10
        elif lb[i] is not None and self.lb[i] is not None and lb[i] != self.lb[i]:
          self.initialFrameSize[i] = (lb[i] - self.lb[i]) / 10.0
        elif (ub[i] is not None and self.ub[i] is not None and ub[i] != self.ub[i]):
          self.initialFrameSize[i] = (self.ub[i] - ub[i]) / 10.0
        else:
          if lb[i] is not None and abs(lb[i]) > DType("high")._zero * 10.0:
            self.initialFrameSize[i] = abs(lb[i]) / 10.0
          else:
            self.initialFrameSize[i] = 1.0
        # Adjust value with granularity
        self.initialFrameSize[i] = self.initialFrameSize.next_mult(g=self.granularity[i], i=i)
        # Adjust value with minFrameSize
        if self.initialFrameSize[i] < self.minFrameSize[i]:
          self.initialFrameSize[i] = self.minFrameSize[i]
      # Determine initial mesh size from initial frame size
      if not self.initialMeshSize.defined[i]:
        self.initialMeshSize[i] = self.initialFrameSize[i] * self.n**-0.5
        # Adjust value with granularity
        self.initialMeshSize[i] = self.initialMeshSize.next_mult(g=self.granularity[i], i=i)
        # Adjust value with minMeshSize
        if (self.initialMeshSize[i] < self.minMeshSize[i]):
          self.initialMeshSize[i] = self.minMeshSize[i]
      
      if (self.minMeshSize[i] > self.initialMeshSize[i]):
        raise IOError("Check: initial mesh size is lower than min mesh size.\n"
                      + f"INITIAL_MESH_SIZE  + {self.initialMeshSize[i]} \n"
                      + f"MIN_MESH_SIZE {self.minMeshSize[i]}")
      if (self.minFrameSize[i] > self.minFrameSize[i]):
        raise IOError("Check: initial frame size is lower than min frame size.\n"
                      + f"INITIAL_FRAME_SIZE  + {self.minFrameSize[i]} \n"
                      + f"MIN_FRAME_SIZE {self.minFrameSize[i]}")

  def set_min_mesh_parameters(self):
    if not self.minMeshSize.is_all_defined():
      for i in range(self.n):
        if self.granularity[i] > 0.0:
          self.minMeshSize[i] = self.granularity[i]
    else:
      if self.minMeshSize.size != self.n:
        raise IOError(f"minMeshSize parameter of size {self.minFrameSize.size} doesn't match the parameters dimensionality of {self.n}")
      for i in range(self.n):
        if self.minMeshSize.defined[i] and self.minMeshSize[i] < 0.0:
          raise IOError(f"Invalid minMeshSize defined of value {self.minMeshSize[i]}")
        elif (not self.minMeshSize.defined[i]) or (0.0 < self.granularity[i] and self.minMeshSize[i] < self.granularity[i]):
          if self.granularity[i] > 0.0:
            self.minMeshSize[i] = self.granularity[i]
          else:
            raise IOError("Error: granularity is defined with a negative value.")
  
  def set_min_frame_parameters(self):
    if not self.minFrameSize.is_all_defined():
      for i in range(self.n):
        if self.granularity[i] > 0.0:
          self.minFrameSize[i] = self.granularity[i]
    else:
      if self.minFrameSize.size != self.n:
        raise IOError(f"minFrameSize parameter of size {self.minFrameSize.size} doesn't match the parameters dimensionality of {self.n}")
      for i in range(self.n):
        if self.minFrameSize.defined[i] and self.minFrameSize[i] < 0.0:
          raise IOError(f"Invalid minFrameSize defined of value {self.minFrameSize[i]}")
        elif (not self.minFrameSize.defined[i]) or (0.0 < self.granularity[i] and self.minFrameSize[i] < self.granularity[i]):
          if self.granularity[i] > 0.0:
            self.minFrameSize[i] = self.granularity[i]
          else:
            raise IOError("Error: granularity is defined with a negative value.")

  def to_be_checked(self)-> bool:
    return self._initialized_and_checked

  # TODO: give better control on variabls' resolution (mesh granularity)
  # var_type: List[str] = field(default_factory=["cont", "cont"])
  # resolution: List[int] = field(default_factory=["cont", "cont"])
      



  def get_barrier_type(self):
    if self.constraints_type is not None:
      if isinstance(self.constraints_type, list):
        for i in range(len(self.constraints_type)):
          if self.constraints_type[i] == BARRIER_TYPES.PB:
            return BARRIER_TYPES.PB
      else:
        if self.constraints_type == BARRIER_TYPES.PB:
            return BARRIER_TYPES.PB

    
    return BARRIER_TYPES.EB
  
  def get_h_max_0 (self):
    return self.h_max
  
  @property
  def n(self)->int:
    return self._n
  
  @n.setter
  def n(self, value: int) -> int:
    self._n = value
  
