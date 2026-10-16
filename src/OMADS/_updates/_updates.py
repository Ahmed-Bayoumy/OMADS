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
import copy


import numpy as np
from .._include import STOP_TYPE, SUCCESS_TYPES
from .._include import AdaptiveBarrier
from .._analytics._metadata import MadsState, MadsStatistics
from .._setup._parameters import Parameters
from .._setup._options import Options
from .._templates._optimizer import GenericSamplerBase
from .._include import Cache

np.set_printoptions(legacy='1.21')


def set_frame_centers_and_hvalues(
        state: MadsState, active_barrier: AdaptiveBarrier, param: Parameters,
        options: Options = None):
  """_summary_

  :param state: _description_
  :type state: MadsState
  :param active_barrier: _description_
  :type active_barrier: BarrierMO
  :param param: _description_
  :type param: Parameters
  """
  # First case: phase one has been triggered.
  # There is only one primary frame center, the one with minimum h value.
  if state.is_phase_one:
    emin = None
    cmin = 0
    c = 0
    for e in active_barrier.elements[0:
                                     sum(
                                         x is not None
                                         for x in active_barrier.elements)]:
      if emin is None or (e is not None and e < emin):
        emin = copy.deepcopy(e)
        cmin = copy.deepcopy(c)
      c += 1

    state.ordered_frame_centers = (
        cmin,
        -1
    )
  else:
    out = active_barrier.frame_centers(
        param.w_min, use_dom_selection=options.use_dom_trigger)
    fk_index = out["feasible"]
    uk_index = out["infeasible"]
    # Set primary and secondary frame centers according to trigger conditions.
    # NOTE: the secondary slot used to fall back to hardcoded index 0 (the
    # very first ever-evaluated candidate) whenever there was no real
    # secondary/infeasible center -- always true for unconstrained problems,
    # since no point is ever infeasible. That silently wasted roughly half
    # of every iteration's sampling budget re-exploring around the original
    # baseline point forever, since it never gets updated as better points
    # are found. -1 is the actual "no center" sentinel used everywhere else
    # in this codebase (MadsState's own default is (-1, -1), and
    # search_step/poll_step already guard on `frame_center != -1`), so use
    # it here too instead of the 0 typo.
    if fk_index == -1:
      state.ordered_frame_centers = (uk_index, -1)
    else:
      if uk_index == -1:
        state.ordered_frame_centers = (fk_index, -1)
      else:
        if options.use_dom_trigger:
            # Using extent is slightly more efficient
          dom = min([
              sum(
                  elt.f - np.minimum(
                      active_barrier.elements[uk_index].fs,
                      elt.fs))
              for elt in active_barrier.get_fk()])
          #  if doM >= params.ρ_trigger * bbproblem.meta.noutputs
          if dom >= param.rho * active_barrier.extent():
            state.ordered_frame_centers = (uk_index, fk_index)
          else:
            state.ordered_frame_centers = (fk_index, uk_index)
        else:  # Classic alternative based on dominance but slightly less efficient.
          if all(active_barrier.elements[fk_index].f - param.rho >=
                 active_barrier.elements[uk_index].f):
            state.ordered_frame_centers = (uk_index, fk_index)
          else:
            state.ordered_frame_centers = (fk_index, uk_index)

    state.fk_frame_center = fk_index if isinstance(
        fk_index, int) else int(fk_index)
    state.uk_frame_center = uk_index if isinstance(
        uk_index, int) else int(uk_index)

    # set h_max
    if len(active_barrier.get_ik()) > 0:
      state.h_max = max(elt.h for elt in active_barrier.get_ik())


def update(sampler: GenericSamplerBase, state: MadsState, options: Options,  # noqa: C901
           active_barrier: AdaptiveBarrier, stats: MadsStatistics,
           hashtable: Cache = None):
  """Update step

  :param poll: poll directions
  :type poll: Dirs2n
  :param state: Algorithmic current success state
  :type state: MadsState
  :param options: Algorithmic options
  :type options: Options
  :param active_barrier: Active barrier
  :type active_barrier: BarrierMO
  :param stats: Metadata and run statistics
  :type stats: MadsStatistics
  """
  # Phase one is over.
  if state.stop_reason == STOP_TYPE.STOP_IF_FEASIBLE:
    state.is_phase_one = False
    state.stop_reason = STOP_TYPE.UNKNOWN_STOP_REASON

  # No need to update in this case
  if state.stop_reason in [
          STOP_TYPE.MIN_MESH_REACHED, STOP_TYPE.DELTA_M_MIN_REACHED, STOP_TYPE.
          MAX_BB_EVAL_REACHED, STOP_TYPE.MAX_BB_EVAL_REACHED]:
    sampler.terminate = True
    return

  # There can remain some points in the set Uk which have better h-value,
  # To check if the flag is activated
  if not options.use_omads_partial_success:
    if state.last_success == SUCCESS_TYPES.US and state.uk_frame_center != -1:
      tmp_hx_inf_min = min(elt.h for elt in active_barrier.get_uk())
      if tmp_hx_inf_min < active_barrier.elements[state.uk_frame_center].h:
        state.last_success = SUCCESS_TYPES.PS

  # Update barrier and mesh
  if state.last_success == SUCCESS_TYPES.US:
    # Set to null the last success directions of the current incumbents.
    for frame_center in range(2):
      barrier_index = state.ordered_frame_centers[frame_center]
      if barrier_index != -1:
        active_barrier.parent_indexes[barrier_index] = 0

    # Update the mesh
    if state.fk_frame_center != -1:
      active_barrier.meshes[state.fk_frame_center].refine_delta_frame_size()
    else:
      active_barrier.meshes[state.ordered_frame_centers[0]
                            ].refine_delta_frame_size()

    # Update the barrier threshold
    if state.h_max is not None:
      below_hmax_elements = [
          elt
          for elt in active_barrier.elements
          if elt is not None and elt.h < state.h_max and active_barrier.within_uk
          [active_barrier.elements.index(elt)]]

      if state.uk_frame_center != -1 and len(below_hmax_elements) != 0:
        h_max_tmp = max(elt.h for elt in below_hmax_elements)
        if h_max_tmp > active_barrier.elements[state.uk_frame_center].h:
          active_barrier.update_barrier(h_max_tmp)
        else:
          active_barrier.update_barrier(
              active_barrier.elements[state.uk_frame_center].h)

    stats.nno_successes += 1

  elif state.last_success == SUCCESS_TYPES.PS:
    if state.h_max is not None:
      below_hxi_elements = [
          elt
          for elt in active_barrier.elements
          if elt is not None and elt.h < active_barrier.elements
          [state.uk_frame_center].h and active_barrier.within_uk
          [active_barrier.elements.index(elt)]]
      if below_hxi_elements is not None and len(below_hxi_elements) > 0:
        active_barrier.update_barrier(max(elt.h for elt in below_hxi_elements))

    stats.npartial_successes += 1

  else:  # Full success
    # Update the barrier threshold

    if state.h_max is not None:
      below_hmax_elements = [
          elt for elt in active_barrier.elements
          if elt is not None and elt.h < state.h_max and active_barrier.within_uk
          [active_barrier.elements.index(elt)]]
      if state.uk_frame_center != -1 and below_hmax_elements and len(
              below_hmax_elements) > 0:
        h_max_tmp = max(elt.h for elt in below_hmax_elements)
        if h_max_tmp > active_barrier.elements[state.uk_frame_center].h:
          active_barrier.update_barrier(h_max_tmp)
        else:
          active_barrier.update_barrier(
              active_barrier.elements[state.uk_frame_center].h)

    # In the case of phase one, update the mesh; otherwise, it was already set before
    if state.is_phase_one:
      emin = None
      cmin = 0
      c = 0
      filled_elts_size = sum([x is not None
                              for x in active_barrier.elements])
      for e in active_barrier.elements[0:filled_elts_size]:
        if emin is None or (e is not None and e < emin):
          emin = copy.deepcopy(e)
          cmin = copy.deepcopy(c)
        c += 1
      new_incumbent_index = cmin
      x_parent = np.array((hashtable.get_candidate_from_cache_by_index(
          active_barrier.parent_indexes[new_incumbent_index])).coordinates)
      active_barrier.meshes[new_incumbent_index].enlarge_delta_frame_size(np.array(
          (hashtable.get_candidate_from_cache_by_index(new_incumbent_index)).coordinates) - np.array(x_parent))

    stats.nfull_successes += 1
  assert active_barrier.last_index == hashtable._last_index
