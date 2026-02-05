"""
# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                               #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the BSD 3-Clause License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the BSD 3-Clause License for more details.   #
#                                                                                     #
#  You should have received a copy of the BSD 3-Clause License along     #
#  with this program. If not, see <https://opensource.org/license/bsd-3-clause/>.     #
#                                                                                     #
#  You can find information on OMADS at                                               #
#  https://github.com/Ahmed-Bayoumy/OMADS                                             #
#  Copyright (C) 2026  Ahmed H. Bayoumy                                               #
# ------------------------------------------------------------------------------------#
"""
from typing import List, Optional
import numpy as np
import samplersLib as explore


from .._include import MPP, DType
from .._include import CandidatePoint
from .._include import Point
from .._include import AdaptiveBarrier
from .._include import logger
from .._include import Gmesh
from .._setup._parameters import Parameters
from .._setup._options import Options
from abc import ABC, abstractmethod


class ConstraintsRelaxationParameters:
  """Constraints relaxation parameters data class
  """

  def __init__(self):
    self.rho: float = MPP.RHO
    self.lambda_multipliers: List[float] = [MPP.LAMBDA]
    self.hmax: float = 1.
    self.constraints_type: Optional[List[int]] = None


class GenericSamplerBaseData:
  """Generic sampler base class

  :param Protocol: _description_
  :type Protocol: _type_
  """

  def __init__(self):
    self.scaling: List[List[float]] = []
    self.mesh: Optional[Gmesh] = None
    self.failure_stop: Optional[bool] = None
    self.constraints_handler: ConstraintsRelaxationParameters = ConstraintsRelaxationParameters()
    self.log: Optional[logger] = None
    self.n_successes: int = 0
    self.prob_params: Optional[Parameters] = None
    self.sampling_t: int = 3
    self.vicinity_ratio: np.ndarray = None
    self.vicinity_min: float = 0.001
    self.terminate: bool = False
    self.visualize: bool = False
    self.sampling_criter: Optional[str] = None
    self.weights: Optional[List[float]] = None
    self.active_sampling: explore.samplers.activeSamplingNew = None
    self.best_samples: int = 0
    self.estimation_grid: explore.samplers.sampling = None
    self.active_barrier: Optional[AdaptiveBarrier] = None
    self.constraints_rp: ConstraintsRelaxationParameters = ConstraintsRelaxationParameters()
    self._eval_set: Optional[List[CandidatePoint]] = None
    self._points: Optional[List[Point]] = None
    self._points_index: Optional[List[int]] = None
    self._n: int = 0
    self._candidate_points_set: List[CandidatePoint] = []
    self._point_index: List[int] = []
    self._directions_set: List[Point] = []
    self._defined: List[bool] = []
    self._xmin: Optional[CandidatePoint] = None
    self._x_sc: Optional[CandidatePoint] = None
    self._nb_success: int = 0
    self._bb_eval: int = 0
    self._frame_size: float = 0
    self._iter: int = 0
    self._check_cache: bool = True
    self._display: bool = True
    self._store_cache: bool = True
    self._opportunistic: bool = False
    self._eval_budget: int = 100
    self._dtype: Optional[DType] = None
    self._success: bool = False
    self._seed: int = 0
    self._terminate: bool = False
    self._bb_output: List[float] = []
    self._dim: int = 0
    self._save_results: bool = False
    self._type: str = "sampling"
    self._x_primary_fc: CandidatePoint = None
    self._x_secondary_fc: CandidatePoint = None
    self._center: CandidatePoint = None
    self._step_name: str = None
    self._rng: np.random = None


class GenericSamplerBase(ABC):
  """Generic sampler class template

  :param GenericSamplerBaseData: _description_
  :type GenericSamplerBaseData: _type_
  :param Protocol: _description_
  :type Protocol: _type_
  """
  @abstractmethod
  def scale(self, ub: List[float], lb: List[float], factor: float = 10.0):
    """Scaling abstract method

    :param ub: _description_
    :type ub: List[float]
    :param lb: _description_
    :type lb: List[float]
    :param factor: _description_, defaults to 10.0
    :type factor: float, optional
    """
    pass

  @abstractmethod
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
    pass

  @abstractmethod
  def generate_candidate_points(self) -> List[CandidatePoint]:
    """Candidates generation abstract method

    :return: _description_
    :rtype: List[CandidatePoint]
    """
    pass

  @abstractmethod
  def omit_duplicates(self):
    """Omitting duplicate designs generated
    """
    pass

  @abstractmethod
  def postprocess_evaluated_candidates(self):
    """Postprocess and evaluate design criteria abstract method
    """
    pass

  @abstractmethod
  def update(self):
    """Update step abstract method
    """
    pass

  @property
  @abstractmethod
  def candidate_points_set(self) -> List[CandidatePoint]:
    """Candidates getter
    """
    pass

  @candidate_points_set.setter
  @abstractmethod
  def candidate_points_set(self, p: CandidatePoint):
    """Candidates setter

    :param p: _description_
    :type p: CandidatePoint
    """
    pass

  @candidate_points_set.deleter
  @abstractmethod
  def candidate_points_set(self):
    """Candidates deleter
    """
    pass

  @property
  @abstractmethod
  def frame_size(self):
    """Frame size getter
    """
    pass

  @frame_size.setter
  @abstractmethod
  def frame_size(self, value):
    """Frame size setter

    :param value: _description_
    :type value: _type_
    """
    pass


class GenericGlobalLocalSamplerBaseData:
  """Generic global-local sampler class template

  :param Protocol: _description_
  :type Protocol: _type_
  """

  def __init__(self):
    self.local_search: Optional[GenericSamplerBase] = None
    self.global_search: Optional[GenericSamplerBase] = None
    self.param: Optional[Parameters] = None
    self.options: Optional[Options] = None


class GenericGlobalLocalSamplerBase(
        ABC, GenericGlobalLocalSamplerBaseData):
  """Generic global-local sampler class template

  :param GenericGlobalLocalSamplerBaseData: _description_
  :type GenericGlobalLocalSamplerBaseData: _type_
  :param Protocol: _description_
  :type Protocol: _type_
  """
  @abstractmethod
  def generate_candidate_points(self) -> List[CandidatePoint]:
    """Candidates generation abstract method

    :return: _description_
    :rtype: List[CandidatePoint]
    """
    pass

  @abstractmethod
  def map_ls_to_gs_attrs(self):
    """Mapping local optimizer to global optimizer attributes and metadata
    """
    pass
    # if not isinstance(self.localSearch, GenericSampler) or
    # not isinstance(self.globalSearch, GenericSampler):
    #     raise TypeError("Both instances must implement 'GenericSampler' class.")

    # attributes = ['B', 'log', 'iter', 'xmin', 'LAMBDA_k', 'RHO_k', 'search_VN', ]
    # mapping = {}

    # for attr in attributes:
    #     setattr(self.globalSearch, attr, getattr(self.localSearch, attr))

  @abstractmethod
  def map_gs_to_ls_attrs(self):
    """Mapping global optimizer to local optimizer attributes and metadata
    """
    pass

  @abstractmethod
  def update(self):
    """Update step abstract method
    """
    pass
