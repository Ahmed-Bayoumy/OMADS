"""
# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                               #
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
from dataclasses import dataclass, field
from typing import List, Protocol, Optional
import numpy as np
import samplersLib as explore


from ._globals import MPP, DType
from .candidate_point import CandidatePoint
from .point import Point
from .barriers import BarrierMO
from ._common import logger
from .gmesh import Gmesh
from .cache import Cache
from .evaluator import Evaluator
from .parameters import Parameters
from .options import Options


@dataclass
class ConstraintsRelaxationParameters:
  """Constraints relaxation parameters data class
  """
  rho: float = MPP.RHO
  lambda_multipliers: List[float] = field(default_factory=lambda: [MPP.LAMBDA])
  hmax: float = 1.
  constraints_type: Optional[List[int]] = None


@dataclass
class GenericSamplerBaseData(Protocol):
  """Generic sampler base class

  :param Protocol: _description_
  :type Protocol: _type_
  """
  scaling: List[List[float]] = field(default_factory=list)
  hashtable: Cache = field(default_factory=Cache)
  mesh: Optional[Gmesh] = None
  bb_handle: Evaluator = field(default_factory=Evaluator)
  failure_stop: Optional[bool] = None
  constraints_handler: ConstraintsRelaxationParameters = field(
      default_factory=lambda: ConstraintsRelaxationParameters)
  log: Optional[logger] = None
  n_successes: int = 0
  prob_params: Optional[Parameters] = None
  sampling_t: int = 3
  vicinity_ratio: np.ndarray = None
  vicinity_min: float = 0.001
  terminate: bool = False
  visualize: bool = False
  sampling_criter: Optional[str] = None
  weights: Optional[List[float]] = None
  active_sampling: explore.samplers.activeSampling = None
  best_samples: int = 0
  estimation_grid: explore.samplers.sampling = None
  active_barrier: Optional[BarrierMO] = None
  constraints_rp: ConstraintsRelaxationParameters = field(
      default_factory=lambda:
      ConstraintsRelaxationParameters(
          rho=MPP.RHO.value, lambda_multipliers=None, hmax=1.,
          constraints_type=None))
  _eval_set: Optional[List[CandidatePoint]] = None
  _points: Optional[List[Point]] = None
  _points_index: Optional[List[int]] = None
  _n: int = 0
  _candidate_points_set: List[CandidatePoint] = field(default_factory=list)
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
  """Generic sampler class template

  :param GenericSamplerBaseData: _description_
  :type GenericSamplerBaseData: _type_
  :param Protocol: _description_
  :type Protocol: _type_
  """

  def scale(self,  ub: List[float], lb: List[float], factor: float = 10.0):
    """Scaling abstract method

    :param ub: _description_
    :type ub: List[float]
    :param lb: _description_
    :type lb: List[float]
    :param factor: _description_, defaults to 10.0
    :type factor: float, optional
    """
    ...

  def gauss_perturbation(
          self, p: CandidatePoint, npts: int = 5) -> List[CandidatePoint]:
    """Gaussian perturbation abstract method

    :param p: _description_
    :type p: CandidatePoint
    :param npts: _description_, defaults to 5
    :type npts: int, optional
    :return: _description_
    :rtype: List[CandidatePoint]
    """
    ...

  def generate_candidate_points(self) -> List[CandidatePoint]:
    """Candidates generation abstract method

    :return: _description_
    :rtype: List[CandidatePoint]
    """
    ...

  def omit_duplicates(self):
    """Omitting duplicate designs generated
    """
    ...

  def postprocess_evaluated_candidates(self):
    """Postprocess and evaluate design criteria abstract method
    """
    ...

  def update(self):
    """Update step abstract method
    """
    ...


@dataclass
class GenericGlobalLocalSamplerBaseData(Protocol):
  """Generic global-local sampler class template

  :param Protocol: _description_
  :type Protocol: _type_
  """
  local_search: Optional[GenericSamplerBase] = None
  global_search: Optional[GenericSamplerBase] = None
  param: Optional[Parameters] = None
  options: Optional[Options] = None


@dataclass
class GenericGlobalLocalSamplerBase(
        GenericGlobalLocalSamplerBaseData, Protocol):
  """Generic global-local sampler class template

  :param GenericGlobalLocalSamplerBaseData: _description_
  :type GenericGlobalLocalSamplerBaseData: _type_
  :param Protocol: _description_
  :type Protocol: _type_
  """

  def generate_candidate_points(self) -> List[CandidatePoint]:
    """Candidates generation abstract method

    :return: _description_
    :rtype: List[CandidatePoint]
    """
    ...

  def map_ls_to_gs_attrs(self):
    """Mapping local optimizer to global optimizer attributes and metadata
    """
    ...
    # if not isinstance(self.localSearch, GenericSampler) or
    # not isinstance(self.globalSearch, GenericSampler):
    #     raise TypeError("Both instances must implement 'GenericSampler' class.")

    # attributes = ['B', 'log', 'iter', 'xmin', 'LAMBDA_k', 'RHO_k', 'search_VN', ]
    # mapping = {}

    # for attr in attributes:
    #     setattr(self.globalSearch, attr, getattr(self.localSearch, attr))

  def map_gs_to_ls_attrs(self):
    """Mapping global optimizer to local optimizer attributes and metadata
    """
    ...

  def update(self):
    """Update step abstract method
    """
    ...
