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
from typing import List

import numpy as np

from .._include import Evaluator
from .._multisource._multisource import MultiSource

from .._include import Point
from .._include import INSERTION_FLAG, STOP_TYPE, SUCCESS_TYPES
from .._include import AdaptiveBarrier
from .._include import CandidatePoint
from .._include import logger
from .._include import Dirs2n
from .._include import Parameters
from .._include import Options
from .._include import PostMADS, Output
from .._metrics._success import compute_success
from .._include import MadsState, MadsStatistics
from .._include import Cache


def poll_cycle(poll: Dirs2n, options: Options, param: Parameters,  # noqa: C901
               state: MadsState, stats: MadsStatistics,
               active_barrier: AdaptiveBarrier, iteration: int, log: logger,
               post: PostMADS, bb_handle: Evaluator, out: Output,
               hashtable: Cache, ms: MultiSource = None):
  # Forget any sets and directions been previously generated
  del poll.candidate_points_set
  del poll.poll_dirs

  if post.step_name is None:
    post.step_name = []
  # tic = time.perf_counter()
  candidates: List[CandidatePoint] = []
  generated_candidates_during_step = []
  parent_index_candidates = []
  generated_during_search_step = []
  # Generate candidate points given each center
  for center in state.ordered_frame_centers:
    if center == -1:
      # No real secondary/infeasible center this iteration (see
      # set_frame_centers_and_hvalues) -- nothing to do here.
      continue
    generated_candidates_during_step = poll_step(
        poll=poll, options=options, param=param, state=state, stats=stats,
        active_barrier=active_barrier, iteration=iteration,
        frame_center=center, log=log, post=post, hashtable=hashtable)

    # Add generated points to the list of candidates
    if generated_candidates_during_step is not None:
      candidates.extend(generated_candidates_during_step)
      for _ in range(len(generated_candidates_during_step)):
        parent_index_candidates.append(center)
        generated_during_search_step.append(False)

  # Evaluate the generated candidate points
  poll._candidate_points_set = []
  for ci, c in enumerate(candidates):
    if hashtable.last_index + 1 < options.budget and not hashtable.is_duplicate(
            c, add=False) and not hashtable.is_duplicate_in_set(
            c, poll._candidate_points_set):
      poll.candidate_points_set = c
  del candidates
  candidates = poll._candidate_points_set
  if ms is None:
    if not options.parallel_mode:
      insertion_flag, post = bb_handle.run_callable_serial_local(
          sampler=poll, centers=parent_index_candidates, options=options,
          stats=stats, active_barrier=active_barrier, post=post,
          step_name='Poll_2n', parent_indices=parent_index_candidates, out=out,
          hashtable=hashtable)

    else:
      # COMPLETED: Review and make it consistent with the serial evaluator
      poll.point_index = -1
      # """ Parallel evaluation for points in the samples set """
      insertion_flag, post = bb_handle.run_callable_parallel_local(
          sampler=poll, options=options, stats=stats,
          active_barrier=active_barrier, post=post, step_name='Poll_2n',
          parent_indices=parent_index_candidates, hashtable=hashtable,
          centers=parent_index_candidates, out=out)
  else:
    if iteration == 1:
      ms.initialize_multisource_manager(
          sampler=poll, centers=parent_index_candidates, options=options,
          stats=stats, active_barrier=active_barrier, post=post,
          step_name='Poll_2n', parent_indices=parent_index_candidates, out=out,
          hashtable=hashtable, log=log)
    insertion_flag, post = ms.ms_serial_evaluation(
        sampler=poll, centers=parent_index_candidates, options=options,
        stats=stats, active_barrier=active_barrier, post=post,
        step_name='Poll', parent_indices=parent_index_candidates, out=out,
        hashtable=hashtable)

  # COMPLETED: Add a new logic to evaluate suceess criteria and update insertion flags accordingly

  for index, candidate in enumerate(candidates):
    success_flag = SUCCESS_TYPES.US if insertion_flag is None or insertion_flag[index] is None else compute_success(
        active_barrier=active_barrier, state=state, options=options, insertion_flag=insertion_flag[index], v=candidate)
    # Update mesh; slightly different from the article for better performance
    if insertion_flag[index] is not None and insertion_flag[index] in [
            INSERTION_FLAG.DOMINATES, INSERTION_FLAG.EXTENDS]:
      x_parent: Point = Point()
      x_parent.coordinates = hashtable.get_candidate_from_cache_by_index(
          parent_index_candidates[index]).coordinates
      direction: Point = Point()
      direction.coordinates = np.array(hashtable.get_candidate_from_cache_by_index(
          hashtable.last_index).coordinates) - np.array(x_parent.coordinates)
      m_index: int = active_barrier.elements.get_index_loc_by_signature(
          candidate.signature)
      active_barrier.meshes[m_index].enlarge_delta_frame_size(
          direction=direction)
      post.xmin = active_barrier.elements.get_candidate_from_elements_by_signature(
          candidate.signature)
      hashtable.set_improving_candidate(x=candidate)

    # Update success flag
    if success_flag.value > (
            state.last_success.value
            if
            isinstance(state.last_success, SUCCESS_TYPES) else
            SUCCESS_TYPES[state.last_success].value):

      state.last_success = success_flag
      poll.n_successes += 1

    # Detect potential stopping reasons
    if (poll.bb_eval >= options.budget):
      state.stop_reason = STOP_TYPE.MAX_BB_EVAL_REACHED
      return

    if (poll.bb_eval >= options.noutbound_hits_max):
      state.stop_reason = STOP_TYPE.MAX_BB_OUTBOUND_REACHED
      return

    # Detect feasibility in case we are in phase one
    if state.is_phase_one and insertion_flag is not None:
      if active_barrier.elements[active_barrier.last_index].h == 0:
        state.stop_reason = STOP_TYPE.STOP_IF_FEASIBLE
        return

    # Trigger opportunistic strategy only when an iteration is considered as a full success
    if success_flag.value > SUCCESS_TYPES.PS.value and options.opportunistic:
      stats.nopportunistic_triggers += 1
      break

  # toc = time.perf_counter()


def poll_step(
        poll: Dirs2n, options: Options, param: Parameters, state: MadsState,
        stats: MadsStatistics, active_barrier: AdaptiveBarrier, iteration: int,
        frame_center: int, log: logger, post: PostMADS, hashtable: Cache) -> List[CandidatePoint]:
  """_summary_

  :param poll: The poll step algorithm
  :type poll: Dirs2n
  :param options: The directions class object
  :type options: Options
  :param param: Algorithmic options
  :type param: Parameters
  :param state: Algorithmic parameters
  :type state: MadsState
  :param stats: Success status
  :type stats: MadsStatistics
  :param active_barrier: Statistics and metadata class object
  :type active_barrier: BarrierMO
  :param iteration: The active barrier class object
  :type iteration: int
  :param xmin: iteration number
  :type xmin: CandidatePoint
  :param log: current incumbent candidate
  :type log: logger
  :param post: updated log
  :type post: PostMADS
  :return: Updated incumbent candidate
  :rtype: CandidatePoint
  """

  poll.prob_params = copy.deepcopy(param)

  # Set mesh
  if state.fk_frame_center == -1:
    mk = active_barrier.meshes[state.ordered_frame_centers[0]]
  else:
    mk = active_barrier.meshes[state.fk_frame_center]

  post.mesh = mk

  # Check the mesh size with the min mesh threshold
  if all(
      [mk.get_delta_frame_size().coordinates[pp] < options.tol
       for pp in range(poll.n)]):
    state.stop_reason = STOP_TYPE.MIN_MESH_REACHED
    return

  # Assign the mesh to the poll instant and constraints relaxation parameters from the current state instant
  poll.mesh = copy.deepcopy(mk)
  poll.constraints_rp.hmax = state.h_max
  poll.constraints_handler.hmax = state.h_max

  # poll.active_barrier = copy.deepcopy(active_barrier)
  poll.success = SUCCESS_TYPES.US

  if frame_center != -1:
    xcp_frame: CandidatePoint = hashtable.get_candidate_from_cache_by_index(
        frame_center)
    x_frame: Point = Point()
    x_frame.coordinates = np.array(xcp_frame.coordinates)
    parent_index = active_barrier.parent_indexes[frame_center]
    x_parent: Point = Point()
    if parent_index != 0:
      x_parent.coordinates = hashtable.get_candidate_from_cache_by_index(
          parent_index).coordinates

    generated_candidates_during_step = None
    poll.create_poll_set(
        ub=param.ub, lb=param.lb, fc=xcp_frame, fci=frame_center, it=iteration,
        var_type=param.var_type, var_sets=param.var_sets, var_link=None,
        rich_direction=options.rich_direction, c_types=param.constraints_type)

    # poll.candidate_points_set
    if poll.candidate_points_set is not None and len(
            poll.candidate_points_set) > 0:
      poll.project_on_mesh_and_snap_to_bounds(
          m=mk, x_center=active_barrier.elements[frame_center].coordinates,
          lb=param.lb, ub=param.ub, hashtable=hashtable, stats=stats)

      peval = poll.bb_eval
      poll.omit_duplicates(peval, stats, hashtable=hashtable)

    generated_candidates_during_step = poll.candidate_points_set
    if (stats.noutbound_hits > poll.eval_budget and len(generated_candidates_during_step) == 0):
      state.stop_reason = STOP_TYPE.MAX_BB_OUTBOUND_REACHED

  return generated_candidates_during_step
