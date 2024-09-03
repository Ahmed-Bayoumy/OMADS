from .CandidatePoint import CandidatePoint
from .Point import Point
from .Barriers import *
from ._common import logger
from dataclasses import dataclass, field
from typing import List, Dict, Any, Protocol
from .Gmesh import Gmesh
from .Cache import Cache
from .Evaluator import Evaluator
import samplersLib as explore

@dataclass
class ConstraintsRelaxationParameters:
  RHO: float = MPP.RHO
  LAMBDA: List[float] = field(default_factory=lambda: [MPP.LAMBDA])
  hmax: float = 1.
  constraints_type: List[int] = None

@dataclass
class GenericSamplerBaseData(Protocol):
  
  scaling: List[List[float]] = field(default_factory=list)
  hashtable: Cache = field(default_factory=Cache)
  mesh: Gmesh = None
  bb_handle: Evaluator = field(default_factory=Evaluator)
  Failure_stop: bool = None
  constraintsHandler: ConstraintsRelaxationParameters = field(default_factory=lambda: ConstraintsRelaxationParameters)
  log: logger = None
  n_successes: int = 0
  prob_params: Parameters = None
  sampling_t: int = 3
  vicinity_ratio: np.ndarray = None
  vicinity_min: float = 0.001
  terminate: bool =False
  visualize: bool = False
  sampling_criter: str = None
  weights: List[float] = None
  AS: explore.samplers.activeSampling = None
  best_samples: int = 0
  estGrid: explore.samplers.sampling = None
  activeBarrier: BarrierMO = None
  constraints_RP: ConstraintsRelaxationParameters = field(default_factory=lambda: ConstraintsRelaxationParameters(RHO=MPP.RHO.value, LAMBDA=None, hmax=1., constraints_type=None))
  _evalSet: List[CandidatePoint] = None
  _points: List[Point] = None
  _pointsIndex: List[int] = None
  _n: int = 0 
  _candidate_points_set : List[CandidatePoint] = field(default_factory=list)
  _point_index: List[int] = field(default_factory=list)
  _directions_set: List[Point] = field(default_factory=list)
  _defined: List[bool] = field(default_factory=lambda: [False])
  _xmin: CandidatePoint = None
  _x_sc: CandidatePoint = None
  _nb_success: int = 0
  _bb_eval: int = field(default_factory=int)
  _psize: float = field(default_factory=float)
  _iter: int = field(default_factory=int)
  _check_cache: bool = True
  _display: bool = True
  _store_cache: bool = True
  _save_results = True
  _opportunistic: bool = False
  _eval_budget: int = 100
  _dtype: DType = None
  _success: bool = False
  _seed: int = 0
  _terminate: bool = False
  _bb_output: List[float] = field(default_factory=list)
  _dim: int = 0
  _save_results: bool = False
  _type: str = "sampling"

@dataclass
class GenericSamplerBase(GenericSamplerBaseData, Protocol):
  
  def scale(self,  ub: List[float], lb: List[float], factor: float = 10.0):
    ...
  
  def gauss_perturbation(self, p: CandidatePoint, npts: int = 5) -> List[CandidatePoint]:
    ...
  
  def generate_candidate_points(self)->List[CandidatePoint]:
    ...
  
  def omit_duplicates(self):
    ...
  
  def postprocess_evaluated_candidates(self):
    ...
  
  def update(self):
    ...

@dataclass
class genericGlobalLocalSamplerBaseData(Protocol):
  localSearch: GenericSamplerBase = None
  globalSearch: GenericSamplerBase = None
  param: Parameters = None
  options: Options = None
  

@dataclass
class genericGlobalLocalSamplerBase(genericGlobalLocalSamplerBaseData, Protocol):

  def generate_candidate_points(self)->List[CandidatePoint]:
    ...
  
  def map_LS_to_GS_attrs(self):
    ...
    # if not isinstance(self.localSearch, GenericSampler) or not isinstance(self.globalSearch, GenericSampler):
    #     raise TypeError("Both instances must implement 'GenericSampler' class.")
    
    # attributes = ['B', 'log', 'iter', 'xmin', 'LAMBDA_k', 'RHO_k', 'search_VN', ]
    # mapping = {}
    
    # for attr in attributes:
    #     setattr(self.globalSearch, attr, getattr(self.localSearch, attr))
  
  def map_GS_to_LS_attrs(self):
    ...
  
  def update(self):
    ...