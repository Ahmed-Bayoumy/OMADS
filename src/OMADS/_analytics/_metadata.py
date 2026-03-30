"""_summary_

# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                    #
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
from typing import List, Tuple, Union
import numpy as np

from .._include import SUCCESS_TYPES, STOP_TYPE
from .._templates._optimizer import GenericSamplerBase


class MadsState:
  """
  A data class for the MADS session state and meta data
  """
  fk_frame_center: int = 0
  uk_frame_center: int = 0
  ncache_hits: int = 0

  ordered_frame_centers: Tuple[int, int] = (-1, -1)
  h_max: Union[float, None] = np.inf

  is_phase_one: bool = False

  last_success: SUCCESS_TYPES = SUCCESS_TYPES.US
  stop_reason: STOP_TYPE = STOP_TYPE.UNKNOWN_STOP_REASON

  def __init__(self):
    self.fk_frame_center = 0
    self.uk_frame_center = 0
    self.ncache_hits = 0

    self.ordered_frame_centers = (-1, -1)
    self.h_max = np.inf

    self.is_phase_one = False

    self.last_success = SUCCESS_TYPES.US
    self.stop_reason = STOP_TYPE.UNKNOWN_STOP_REASON


class MadsStatistics:
  """
  A class for the MADS statistical metrics
  """

  def __init__(self):
    self.neval_bb = 0
    self.noutbound_hits = 0
    self.ncache_hits = 0
    self.niterations = 0
    self.neval_bb_feasible = 0
    self.neval_bb_infeasible = 0
    self.nfull_successes = 0
    self.npartial_successes = 0
    self.nno_successes = 0
    self.nopportunistic_triggers = 0


class MadsIterationAttributes:
  """
  Steps that must be executed around each iteration center
  """
  first_center_steps: List[GenericSamplerBase] = None
  second_center_steps: List[GenericSamplerBase] = None

  def __init__(self):
    self.first_center_steps = None
    self.second_center_steps = None
