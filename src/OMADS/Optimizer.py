import numpy as np
from .CandidatePoint import CandidatePoint
from .Point import Point
from .Barriers import BarrierMO
from ._common import logger
from dataclasses import dataclass, field
from typing import List, Protocol, Optional
from .Gmesh import Gmesh
from .Cache import Cache
from .Evaluator import Evaluator
import samplersLib as explore
from ._globals import MPP, DType
from .Parameters import Parameters
from .Options import Options
@dataclass
class ConstraintsRelaxationParameters:
  RHO: float = MPP.RHO
  LAMBDA: List[float] = field(default_factory=lambda: [MPP.LAMBDA])
  hmax: float = 1.
  constraints_type: Optional[List[int]] = None

@dataclass
class GenericSamplerBaseData(Protocol):
  
  scaling: List[List[float]] = field(default_factory=list)
  hashtable: Cache = field(default_factory=Cache)
  mesh: Optional[Gmesh] = None
  bb_handle: Evaluator = field(default_factory=Evaluator)
  Failure_stop: Optional[bool] = None
  constraintsHandler: ConstraintsRelaxationParameters = field(default_factory=lambda: ConstraintsRelaxationParameters)
  log: Optional[logger] = None
  n_successes: int = 0
  prob_params: Optional[Parameters] = None
  sampling_t: int = 3
  vicinity_ratio: np.ndarray = None
  vicinity_min: float = 0.001
  terminate: bool =False
  visualize: bool = False
  sampling_criter: Optional[str] = None
  weights: Optional[List[float]] = None
  AS: explore.samplers.activeSampling = None
  best_samples: int = 0
  estGrid: explore.samplers.sampling = None
  activeBarrier: Optional[BarrierMO] = None
  constraints_RP: ConstraintsRelaxationParameters = field(default_factory=lambda: ConstraintsRelaxationParameters(RHO=MPP.RHO.value, LAMBDA=None, hmax=1., constraints_type=None))
  _evalSet: Optional[List[CandidatePoint]] = None
  _points: Optional[List[Point]] = None
  _pointsIndex: Optional[List[int]] = None
  _n: int = 0 
  _candidate_points_set : List[CandidatePoint] = field(default_factory=list)
  _point_index: List[int] = field(default_factory=list)
  _directions_set: List[Point] = field(default_factory=list)
  _defined: List[bool] = field(default_factory=lambda: [False])
  _xmin: Optional[CandidatePoint] = None
  _x_sc: Optional[CandidatePoint] = None
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
  _dtype: Optional[DType] = None
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
class GenericGlobalLocalSamplerBaseData(Protocol):
  localSearch: Optional[GenericSamplerBase] = None
  globalSearch: Optional[GenericSamplerBase] = None
  param: Optional[Parameters] = None
  options: Optional[Options] = None
  

@dataclass
class GenericGlobalLocalSamplerBase(GenericGlobalLocalSamplerBaseData, Protocol):

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