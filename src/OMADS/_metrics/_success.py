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

from .._include import SUCCESS_TYPES
from .._include import AdaptiveBarrier
from .._include import CandidatePoint
from .._analytics._metadata import MadsState
from .._include import Options


def compute_success(  # noqa: C901
        active_barrier: AdaptiveBarrier, state: MadsState, options: Options,
        insertion_flag: str, v: CandidatePoint) -> SUCCESS_TYPES:
  """_summary_

  :param active_barrier: Evaluate success criteria and compute the success level
  :type active_barrier: BarrierMO
  :param state: The active algorithm session state and metadata
  :type state: MadsState
  :param options: algorithmic options
  :type options: Options
  :param insertion_flag: The update of non-dominated solutions or prim/sec incumbents
  :type insertion_flag: str
  :return: The success level
  :rtype: SUCCESS_TYPES
  """

  # Phase one
  if state.is_phase_one:
    if v.h < active_barrier.elements[state.ordered_frame_centers[0]].h:
      return SUCCESS_TYPES.FS
    else:
      return SUCCESS_TYPES.US
  else:
    success_flag = SUCCESS_TYPES.US

    # Feasible case: full success as soon as a new non-dominated point which dominates
    # the current feasible frame center is generated.
    if v.is_feasible():
      # The set of feasible points can be empty before the insertion of the new point.
      if state.fk_frame_center != -1:
        if options.use_dms_success and insertion_flag in [
                'dominates', 'extends', 'improves']:
          success_flag = SUCCESS_TYPES.FS
        if v <= active_barrier.elements[state.fk_frame_center]:
          success_flag = SUCCESS_TYPES.FS
      # If Fk is empty, consider it as a full success, similar to the Nomad software.
      else:
        success_flag = SUCCESS_TYPES.FS
    else:
      # The first iterations are considered as a partial success when the progressive barrier
      # approach is chosen.
      if state.h_max is None:
        if active_barrier.h_max != 0:
          success_flag = SUCCESS_TYPES.PS
      else:
        # The second (and) check in the following if statement shows better results for the GP problem
        if state.uk_frame_center != -1:  # and state.fk_frame_center == -1:
          # Partial success if h(x) below h(x_inf)
          if v.h < active_barrier.elements[state.uk_frame_center].h:
            success_flag = SUCCESS_TYPES.PS
        # Success if change in Iᵏ with h(x) <= h_max.
        if state.uk_frame_center != -1 and v <= active_barrier.elements[state.uk_frame_center]:
          success_flag = SUCCESS_TYPES.FS
          #  if v.h <= model.barrier.elements[model.state.Uk_frame_center].h and insertion_flag in
          # ['dominates', 'extends', 'improves']:
          #      success_flag = FULL_SUCCESS
          #  end

    return success_flag
