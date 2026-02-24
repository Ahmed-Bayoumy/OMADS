import copy
import time
from typing import Dict, List, Tuple, Any
from UARAF.multisource_dataset import MultisourceDataset
from UARAF.uaraf import UARAF

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, ConfigDict, field_validator, model_validator
from scipy.stats import kendalltau

from .._include import Cache, validator, logger, MSG_TYPE
from .._include import CandidatePoint
from .._setup._options import Options
from .._include import AdaptiveBarrier
from .._evaluator._evaluator import Evaluator
from .._templates._optimizer import GenericSamplerBase
from .._post._postprocess import PostMADS, Output


class MultiSource(BaseModel):
  model_config = {
      # 👈 This tells Pydantic to skip schema generation for unknown types
      "arbitrary_types_allowed": True
  }
  sources: Dict[str, Evaluator]
  conf: Dict
  _allowed_conf: Tuple = {
      "abs_ref", "rel_ref", "acceleration", "adequacy", "relevance",
      "confidence", "data_split", "save_dir"}
  _uaraf: UARAF | None = None
  _initialized: bool = False
  _train: bool = False
  _training_freq: int = 150

  @classmethod
  @field_validator("conf", mode="before")
  def validate_keys(cls, v):
    for key in v:
      if key not in cls._allowed_options:
        raise ValueError(
            f"Invalid MultiSource::conf key '{key}'. Allowed: {cls._allowed_options}")
    return v

  @model_validator(mode='after')
  def validate_sources(self) -> 'MultiSource':
    for key, evaluator in self.sources.items():
      if not evaluator.__class__.__name__.strip():
        raise ValueError(f"Evaluator '{key}' must have a non-empty name.")
      if not evaluator.blackbox:
        raise ValueError(
            f"Evaluator blackbox key must have a non-empty executable/callable name.")
      # Add more custom validations as needed
    return self

  def collect_training_data(
          self, sampler: GenericSamplerBase, centers: List[int],
          options: Options, stats: Any, active_barrier: AdaptiveBarrier,
          post: PostMADS, step_name: str, parent_indices: List[int],
          out: Output, hashtable=None, data_dict: Dict = None,
          output_crit=None):
    return

  def initialize_multisource_manager(
          self, sampler: GenericSamplerBase, centers: List[int],
          options: Options, stats: Any, active_barrier: AdaptiveBarrier,
          post: PostMADS, step_name: str, parent_indices: List[int],
          out: Output, hashtable: Cache = None, log: logger = None):

    return

  @property
  def train(self):
    return False

  def query_adequate_evaluators(
          self, x, output_name) -> List[Evaluator]:

    return

  def ms_serial_evaluation(
          self, sampler: GenericSamplerBase, centers: List[int],
          options: Options, stats: Any, active_barrier: AdaptiveBarrier,
          post: PostMADS, step_name: str, parent_indices: List[int],
          out: Output, hashtable=None):

    return
