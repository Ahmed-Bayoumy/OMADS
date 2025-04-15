"""_summary_

# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                    #
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
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import sys
import os
import json
from multiprocessing import freeze_support
from dataclasses import dataclass
import copy
import numpy as np

from .barriers import Barrier, BarrierMO
from .cache import Cache
from .directions import Dirs2n
from .exploration import EfficientExploration, VNS
from ._globals import DESIGN_STATUS, SAMPLING_METHOD, SEARCH_TYPE
from ._globals import SUCCESS_TYPES, INSERTION_FLAG, VAR_TYPE, STOP_TYPE
from .pre_exploration import PreExploration
from .pre_poll import PrePoll
from .point import Point
from .candidate_point import CandidatePoint
from ._common import logger, MSG_TYPE
from .parameters import Parameters
from .options import Options
from .postprocess import Output, PostMADS
from .metrics import Metrics
from .optimizer import ConstraintsRelaxationParameters
from .evaluator import Evaluator
from .gmesh import Gmesh
from .optimizer import GenericSamplerBase
np.set_printoptions(legacy='1.21')

COLORS = {
    "pending": "\033[0m",  # Default terminal color
    "feasible": "\033[32m",  # Green for completed
    "infeasible": "\033[33m",    # Yellow for pending
    "error": "\033[31m",     # Red for failed
}


@dataclass
class MadsState:
  """
  A data class for the MADS session state and meta data
  """
  fk_frame_center: int = 0
  uk_frame_center: int = 0

  ordered_frame_centers: Tuple[int, int] = (-1, -1)
  h_max: Union[float, None] = None

  is_phase_one: bool = False

  last_success: SUCCESS_TYPES = SUCCESS_TYPES.US
  stop_reason: STOP_TYPE = STOP_TYPE.UNKNOWN_STOP_REASON


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


@dataclass
class MADS:
  """
  MADS class object that has both the search and poll steps
  where each considers secondary and primary frame centers
  """
  search: Optional[EfficientExploration] = None
  search_vns: VNS = None
  poll: Optional[Dirs2n] = None
  param: Optional[Parameters] = None
  evaluator: Optional[Evaluator] = None
  post: Optional[PostMADS] = None
  out: Optional[Output] = None
  out_p: Optional[Output] = None
  options: Optional[Options] = None
  data: Optional[dict] = None
  xmin: Optional[CandidatePoint] = None
  iteration: int = 0
  peval: int = 0
  hashtable: Cache = None
  log: Optional[logger] = None
  lambda_multipliers_k: float = 0.
  rho_k: float = 0.
  tic: Any = None
  toc: Any = None
  active_barrier: Optional[BarrierMO] = None
  state: MadsState = None
  stats: MadsStatistics = MadsStatistics()
  attributes: MadsIterationAttributes = MadsIterationAttributes()
  hv: float = 0.0

  def __init__(self, data: dict):
    """ Initialize the log file """
    self.log = logger()
    if not os.path.exists(data["param"]["post_dir"]):
      try:
        os.mkdir(data["param"]["post_dir"])
      except ValueError:
        os.makedirs(data["param"]["post_dir"], exist_ok=True)

    self.log.initialize(data["param"]["post_dir"] + "/OMADS.log")
    self.log.log_msg(msg="Preprocess the MADS algorithim...",
                     msg_type=MSG_TYPE.INFO)
    self.log.log_msg(msg="Preprocess the search step...",
                     msg_type=MSG_TYPE.INFO)
    _, _, self.search, _, _, _, _, _, _ = PreExploration(
        data).initialize_from_dict(log=self.log)
    self.log.log_msg(msg="Preprocess the POLL step...",
                     msg_type=MSG_TYPE.INFO)

    self.iteration, self.xmin, self.poll, self.options, self.param, \
        self.post, self.out, self.active_barrier, self.out_p = \
        PrePoll(data).initialize_from_dict(
            log=self.log, xs=self.search.xmin)

    self.out.step_name = "Poll"
    self.post.step_name = [f'Search: {self.search.type}']

    self.hashtable = copy.deepcopy(self.poll.hashtable)

    self.state = MadsState()

  def progress_bar(self, prefix="", length=40, fill="█"):
    """
    A progress bar for terminal

    :param prefix: a text that preceds the progress bar, defaults to ""
    :type prefix: str, optional
    :param length: length of progress bar, defaults to 40
    :type length: int, optional
    :param fill: the fill character, defaults to "█"
    :type fill: str, optional
    """
    filled_length = int(length * self.peval // self.options.budget)
    prog_bar = fill * filled_length + "-" * (length - filled_length)
    sys.stdout.write(
        f"\r{prefix} |{prog_bar}| {self.peval}/{self.options.budget}% Complete")
    sys.stdout.flush()

  def progress_bar_colored(self, prefix="", status_list=None):
    """
    A colored progress bar for terminal with each cell's color based on status.
    `status_list` should be a list of statuses like ["completed", "pending", "failed", ...]

    :param prefix: a text that preceds the progress bar, defaults to ""
    :type prefix: str, optional
    :param status_list: list of status names, defaults to None
    :type status_list: _type_, optional
    """
    length = 40
    if status_list is None:
      # Default: all cells are "pending"
      status_list = ["pending"] * self.options.budget
    idx = 0
    for i in range(self.options.budget):
      if self.active_barrier.elements[i] is None or \
              self.active_barrier.elements[i].status == DESIGN_STATUS.UNEVALUATED:
        status_list[idx] = "pending"
      elif self.active_barrier.elements[i].is_feasible():
        status_list[idx] = "feasible"
      elif self.active_barrier.elements[i].status == DESIGN_STATUS.INFEASIBLE:
        status_list[idx] = "infeasible"
      idx += 1
      if idx == self.options.budget:
        break

    prog_bar = ""
    filled_length = int(length * self.peval // self.options.budget)
    for i in range(filled_length):
      status = status_list[i]
      color = COLORS.get(status, COLORS["pending"])
      prog_bar += f"{color}█{COLORS['pending']}"  # Color each block
    for i in range(length, filled_length):
      prog_bar += "-"
    sys.stdout.write(f"\r{' ' * 100}")  # Clear the line
    pcolor = COLORS.get("pending", COLORS["pending"])
    fcolor = COLORS.get("feasible", COLORS["pending"])
    infcolor = COLORS.get("infeasible", COLORS["pending"])
    prefix = f"Pending: {pcolor}█{COLORS['pending']}" + \
        f", Feasible: {fcolor}█{COLORS['pending']}" f", Infeasible: {infcolor}█{COLORS['pending']}"
    sys.stdout.write(
        f"\r{prefix} |{prog_bar}| {self.peval}/{self.options.budget} #Evaluated")
    sys.stdout.flush()

  def check_mads_parameters(self):
    """ Initial exceptions check for parameters dict

    :raises IOError: ρ is a positive parameter
    :raises IOError: w_min is a positive parameter
    :raises IOError: h_init is a positive parameter
    :raises IOError: max_size is a strictly positive parameter
    """
    if self.param.rho <= 0:
      raise IOError("ρ is a positive parameter")
    if self.param.w_min < 0:
      raise IOError("w_min is a positive parameter")
    if self.param.h_init < 0:
      raise IOError("h_init is a positive parameter")
    if self.param.max_size <= 0:
      raise IOError("max_size is a strictly positive parameter")
    # reset rng in case someone changes the seed
    self.param.rng = np.random.Generator(
        np.random.MT19937(seed=self.options.seed))

  def check_mads_options(self):
    """ Initial exceptions check for options dict

    :raises IOError: _description_
    :raises IOError: _description_
    """
    if self.options.budget <= 0:
      raise IOError(
          "The number of blackbox evaluations cannot be negative !")
    if self.options.noutbound_hits_max <= 0:
      raise IOError("The number of blackbox tentative evaluations \
                    outside the bounds cannot be negative !")

  def check_mads_iteration_attributes(self):
    """TODO: Implement and check it
    """
    # if not any([step == self.attributes.first_center_steps[0] \
    # for step in available_poll_symbols]):
    #   raise IOError("A poll step must be performed around the first center")
    return

  def reach_nevals_bb_max(self) -> bool:
    """_summary_
    :return: Check whther exhausted eval budget
    :rtype: bool
    """
    return self.evaluator.bb_eval >= self.options.budget

  def reach_noutbound_hits_max(self) -> bool:
    """_summary_
    :return: Reached the maximum number of times hitting or exceeding any design variable bounds
    :rtype: bool
    """
    return self.evaluator.bb_eval >= self.options.noutbound_hits_max

  def init_phase(self):
    """Preliminary checks, validation and possibly search
    """
    # Preliminary checks
    self.check_mads_parameters()
    self.check_mads_options()
    self.check_mads_iteration_attributes()

    # Reset storing structures if change in parameters
    self.param.max_size = max(self.param.max_size, self.options.budget)
    if self.active_barrier is None:
      if self.param.is_pareto:
        self.active_barrier = BarrierMO(
            self.param, self.options, self.hashtable.get_cache_points())
      else:
        self.active_barrier = Barrier(self.param)
    self.state.stop_reason = STOP_TYPE.NO_INIT_CANDIDATES

    self.active_barrier.update_with_points(
        eval_point_list=self.hashtable.get_cache_candidate_points())

    # No init candidates: all evaluated points are above the h_max threshold.
    # Trigger phase one.
    if len(
            self.active_barrier.get_fk()) == 0 and len(
            self.active_barrier.get_uk()) == 0:
      self.state.stop_reason = STOP_TYPE.UNKNOWN_STOP_REASON
      self.state.is_phase_one = True
      return

    # Other cases
    if self.state.stop_reason not in \
            [STOP_TYPE.MAX_BB_EVAL_REACHED, STOP_TYPE.MAX_BB_OUTBOUND_REACHED]:
      self.state.stop_reason = STOP_TYPE.UNKNOWN_STOP_REASON
      return

  def search_step(self) -> CandidatePoint:
    """
    Run the search step

    :return: [Incumbent solution found for SOO, incumbent non-dominated solution]
    :rtype: CandidatePoint
    """    """"""
    # Set time
    tic = time.perf_counter()
    # Set mesh
    if self.state.fk_frame_center == -1:
      mk = self.active_barrier.meshes[self.state.ordered_frame_centers[0]]
    else:
      mk = self.active_barrier.meshes[self.state.fk_frame_center]

    if all(
        [mk.get_delta_frame_size().coordinates[pp] < self.options.tol
         for pp in range(self.param.n)]):
      self.state.stop_reason = STOP_TYPE.MIN_MESH_REACHED
      return

    # Generate candidates
    self.search.mesh = copy.deepcopy(mk)

    # self.search.mesh.update()
    # Create the candidate points
    self.search.constraints_rp.hmax = self.state.h_max
    self.search.constraints_handler.hmax = self.state.h_max
    self.search.hmax = self.state.h_max

    parent_index_candidates = []
    generated_during_search_step = []

    self.search.active_barrier = copy.deepcopy(self.active_barrier)
    if self.hashtable is not None:
      self.search.hashtable = self.hashtable

    for ci, _ in enumerate(self.state.ordered_frame_centers):
      center = self.state.ordered_frame_centers[ci]
      if center > -1:
        self.search.xmin = self.active_barrier.elements[center]
        self.search.mesh = copy.deepcopy(mk)

        if self.search.type == SEARCH_TYPE.VNS.name and self.search_vns is not None:
          self.search_vns.active_barrier = self.active_barrier
          self.search._candidate_points_set = self.search_vns.run()
          if self.search_vns.stop:
            print("Reached maximum number of VNS iterations!")
            self.hashtable = self.search.hashtable
            self.rho_k = self.search.RHO
            self.lambda_multipliers_k = self.search.LAMBDA
            return self.search.xmin
          self.search.map_samples_from_coords_to_points(
              samples=self.search._candidate_points_set)

        self.search.generate_sample_points(
            nsamples=int(
                ((self.search.dim + 1) / 2) *
                ((self.search.dim + 2) / 2))
            if self.search.ns is None else
            self.search.ns)

        self.search.constraints_handler.lambda_multipliers = self.lambda_multipliers_k
        self.search.constraints_handler.rho = self.rho_k

        self.search.project_on_mesh_and_snap_to_bounds(
            m=mk, x_center=self.active_barrier.elements[center].coordinates,
            lb=self.param.lb, ub=self.param.ub)

        self.search.omit_duplicates(self.peval)
        self.post.x_incumbent.append(self.search.xmin)
        for _ in self.search._candidate_points_set:
          parent_index_candidates.append(center)
          generated_during_search_step.append(False)

        #  Evaluation
        # Save current search directions and incumbent solution
        #   so they can be saved later in the post dir
        if self.options.save_coordinates:
          self.post.coords.append(self.search._candidate_points_set)
        self.post.x_incumbent.append(self.search.xmin)
        # """ Reset success boolean """
        self.search.success = SUCCESS_TYPES.US
        # """ Reset the BB output """
        self.search.bb_output = []
        xt = []
        self.search.bb_handle.xmin = self.active_barrier.elements[center]
        # """ Serial evaluation for points in the search candidates set """
        self.search.constraints_rp.lambda_multipliers = self.active_barrier.elements[
            center].lambda_multipliers
        self.search.constraints_rp.rho = self.active_barrier.elements[center].rho
        self.search.constraints_rp.constraints_type = \
            self.active_barrier.elements[center].constraints_type
        self.search.constraints_rp.hmax = self.state.h_max

        if not self.options.parallel_mode:
          xt, self.post, self.peval = self.search.bb_handle.run_callable_serial_local(
              iter=self.iteration, peval=self.peval,
              eval_set=self.search._candidate_points_set, options=self.options,
              post=self.post, psize=self.search.mesh.get_delta_frame_size().coordinates,
              step_name='Search Step',
              mesh=mk, constraints_relaxation=self.search.constraints_rp.__dict__,
              budget=self.options.budget)

        else:
          self.search.point_index = -1
          # """ Parallel evaluation for points in the samples set """
          self.search.bb_eval, xt, self.post, self.peval = \
              self.search.bb_handle.run_callable_parallel_local(
                  iter=self.iteration, peval=self.peval,
                  eval_set=self.search._candidate_points_set,
                  options=self.options, post=self.post, mesh=mk,
                  step_name='Search Step',
                  psize=self.search.mesh.get_delta_frame_size().coordinates,
                  constraints_relaxation=self.search.constraints_rp.__dict__,
                  budget=self.options.budget)

        if self.search.bb_handle.constraints_relaxation:
          temp: ConstraintsRelaxationParameters = \
              ConstraintsRelaxationParameters(
                  **self.search.bb_handle.constraints_relaxation)
          for i, _ in enumerate(temp.lambda_multipliers):
            self.search.constraints_rp.lambda_multipliers[i] = temp.lambda_multipliers[i]
          self.search.constraints_rp.rho = temp.rho
          self.search.constraints_rp.constraints_type = temp.constraints_type
          self.search.constraints_rp.hmax = temp.hmax

        self.lambda_multipliers_k = self.search.bb_handle.constraints_relaxation[
            "lambda_multipliers"]
        self.rho_k = self.search.bb_handle.constraints_relaxation["rho"]
        self.search.postprocess_evaluated_candidates(xt)

        idx = -1
        xpost: List[CandidatePoint] = []
        for i, _ in enumerate(xt):
          xpost.append(xt[i])
          self.post.poll_dirs.append(xpost[i])
        # assert len(self.HT.hash_id) - self.active_barrier.last_index + 1== 0
        for cp in xt:
          idx += 1
          self.stats.neval_bb_feasible += 1
          if cp.is_feasible():
            _, insertion_flag = self.active_barrier.update_feas_with_point(
                cp)
          else:
            self.stats.neval_bb_infeasible += 1
            _, insertion_flag = self.active_barrier.update_inf_with_point(
                cp)
          self.active_barrier.parent_indexes[self.active_barrier.last_index] = \
              parent_index_candidates[idx]
          success_flag = SUCCESS_TYPES.US if insertion_flag is None \
              else self.compute_success(insertion_flag=insertion_flag)
          if insertion_flag is not None and insertion_flag in \
                  [INSERTION_FLAG.DOMINATES, INSERTION_FLAG.EXTENDS]:
            x_parent = np.array(
                self.search.hashtable.get_cache_candidate_points()
                [parent_index_candidates[idx]].coordinates)
            self.active_barrier.meshes[self.active_barrier.last_index].enlarge_delta_frame_size(
                np.array(
                    self.search.hashtable.cache_dict[self.search.hashtable.hash_id[-1]].coordinates)
                - x_parent)
          # Update success flag
          if success_flag.value > (
                  self.state.last_success.value
                  if isinstance(self.state.last_success, SUCCESS_TYPES)
                  else SUCCESS_TYPES[self.state.last_success].value):
            self.state.last_success = success_flag

    self.lambda_multipliers_k = self.xmin.lambda_multipliers
    self.rho_k = self.search.xmin.rho

    self.post.xmin = self.search.xmin

    toc = time.perf_counter()

    self.post.xmin = self.search.xmin

    if self.log is not None:
      # log.log_msg(msg=" ---Run Summary--- ", msg_type=MSG_TYPE.INFO)
      self.log.log_msg(
          msg=f" Run completed in {toc - tic:.4f} seconds",
          msg_type=MSG_TYPE.INFO)
      self.log.log_msg(
          msg=f" Success status: {self.search.success}", msg_type=MSG_TYPE.INFO)
      self.log.log_msg(msg=self.post.__str__(), msg_type=MSG_TYPE.INFO)

    # self.active_barrier = B
    self.lambda_multipliers_k = self.search.xmin.lambda_multipliers
    self.search.hashtable.add_to_best_cache(
        self.active_barrier.get_all_points())

    self.hashtable = self.search.hashtable

    if self.state.last_success == SUCCESS_TYPES.FS \
            and self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
      self.search.update_local_region(region="expand")
    elif self.state.last_success == SUCCESS_TYPES.US \
            and self.search.sampling_t != SAMPLING_METHOD.ACTIVE.name:
      self.search.update_local_region(region="contract")
    return self.search.xmin

  def project_on_mesh_and_snap_to_bounds(self, m: Gmesh, proj: List[float],
                                         x_center: List[float],
                                         lb: List[float], ub: List[float]):
    """_summary_

    :param m: _description_
    :type m: Gmesh
    :param proj: _description_
    :type proj: List[float]
    :param x_center: _description_
    :type x_center: List[float]
    :param lb: _description_
    :type lb: List[float]
    :param ub: _description_
    :type ub: List[float]
    :raises ValueError: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """
    # Length check equivalent in Python
    if len(lb) != m.n or len(ub) != m.n:
      raise ValueError(f"Expected vectors of length {m.n}, \
                       but got lengths {len(lb)} and {len(ub)}")
    if not np.all(lb < ub):
      raise ValueError("Wrong bound constraints")
    if not np.all(lb <= x_center) or not np.all(ub >= x_center):
      raise ValueError(
          "mesh center values must satisfy: lb <= x^mesh <= ub")
    # 1. Project on the mesh
    candidate = m.project_on_mesh(point=proj, frame_center=x_center)
    # 2. Snap to bounds if necessary
    δ = m.get_delta_mesh_size()
    snapped_candidate = np.zeros(m.n)

    for i in range(m.n):
      if lb[i] <= candidate[i] <= ub[i]:
        snapped_candidate[i] = candidate[i]
      else:
        if candidate[i] < lb[i]:
          # Mesh center is supposed to be >= lb; normally,
          # this rounding is supposed to be in the box constraints
          snapped_candidate[i] = δ[i] * \
              np.ceil((lb[i] - x_center[i]) / δ[i]) + x_center[i]
        else:
          # Ref value is supposed to be <= ub; normally,
          # this rounding is supposed to be in the box constraints
          snapped_candidate[i] = δ[i] * \
              np.floor((ub[i] - x_center[i]) / δ[i]) + x_center[i]

        # Warnings as defined in Nomad 3
        if snapped_candidate[i] < lb[i]:
          print(
              f"Warning: snap_to_bounds: Error snapping {candidate[i]} to lower bound {lb[i]}")
          print(
              f"frameCenter = {x_center[i]}, δ = {δ[i]} : \
                it gave {snapped_candidate[i]} which is still lower than {lb[i]}")
          # TODO: Force the snapping?

        if snapped_candidate[i] > ub[i]:
          print(
              f"Warning: snap_to_bounds: Error snapping {candidate[i]} to upper bound {ub[i]}")
          print(
              f"frameCenter = {x_center[i]}, δ = {δ[i]} : \
                it gave {snapped_candidate[i]} which is still higher than {ub[i]}")
          # TODO: Force the snapping?

    return snapped_candidate

  def poll_step(self):
    """Run the poll step

    :raises ValueError: _description_
    :return: nondominated incumbent solution
    :rtype: CandidatePoint
    """
    # Set time
    tic = time.perf_counter()
    self.poll.prob_params = copy.deepcopy(self.param)
    # Set mesh
    if self.state.fk_frame_center == -1:
      mk = self.active_barrier.meshes[self.state.ordered_frame_centers[0]]
    else:
      mk = self.active_barrier.meshes[self.state.fk_frame_center]

    if all(
        [mk.get_delta_frame_size().coordinates[pp] < self.options.tol
         for pp in range(self.poll.dim)]):
      self.state.stop_reason = STOP_TYPE.MIN_MESH_REACHED
      return

    # Generate candidates
    # self.poll.mesh = Mk

    self.poll.mesh = copy.deepcopy(mk)
    # """ Create the set of poll directions """
    if (self.state.fk_frame_center >= 0 or self.state.uk_frame_center >= 0):
      hhm = self.poll.create_housholder(
          self.options.rich_direction,
          domain=self.param.var_type)
    else:
      raise ValueError(
          "Both primary and secondary frame centers are not \
            valid! Check the poll step configurations.")

    self.poll.lb = self.param.lb
    self.poll.ub = self.param.ub
    # xmin.mesh = copy.deepcopy(self.poll.mesh)
    # B = self.active_barrier
    if self.hashtable is not None:
      self.poll.hashtable = self.hashtable

    parent_index_candidates = []
    generated_during_search_step = []
    for ci, _ in enumerate(self.state.ordered_frame_centers):
      center = self.state.ordered_frame_centers[ci]
      if center > -1:
        if self.active_barrier.elements[center].is_feasible():
          self.poll.xmin = self.active_barrier.elements[center]
        else:
          self.poll.x_sc = self.active_barrier.elements[center]

        self.poll.create_poll_set(
            hhm=hhm, ub=self.param.ub, lb=self.param.lb, it=self.iteration,
            var_type=self.xmin.var_type, var_sets=self.xmin.sets,
            var_link=self.xmin.var_link, c_types=self.param.constraints_type,
            is_prim=self.active_barrier.elements[center].is_feasible())

        self.poll.LAMBDA = self.lambda_multipliers_k
        self.poll.RHO = self.rho_k

        self.poll.project_on_mesh_and_snap_to_bounds(
            m=mk, x_center=self.active_barrier.elements[center].coordinates,
            lb=self.param.lb, ub=self.param.ub)
        self.poll.omit_duplicates(self.peval)

        for _ in self.poll.poll_set:
          parent_index_candidates.append(center)
          generated_during_search_step.append(False)

        # Evaluation
        # """ Save current poll directions and incumbent solution
        #   so they can be saved later in the post dir """
        if self.options.save_coordinates:
          self.post.coords.append(self.poll.poll_set)
        self.post.x_incumbent.append(self.poll.xmin)
        # """ Reset success boolean """
        self.poll.success = SUCCESS_TYPES.US
        # """ Reset the BB output """
        self.poll.bb_output = []
        xt = []
        self.poll.bb_handle.xmin = self.active_barrier.elements[center]
        # """ Serial evaluation for points in the poll set """
        self.poll.constraints_rp.lambda_multipliers = self.active_barrier.elements[
            center].lambda_multipliers
        self.poll.constraints_rp.rho = self.active_barrier.elements[center].rho
        self.poll.constraints_rp.constraints_type = self.active_barrier.elements[
            center].constraints_type
        self.poll.constraints_rp.hmax = self.state.h_max

        if not self.options.parallel_mode:
          xt, self.post, self.peval = self.poll.bb_handle.run_callable_serial_local(
              iter=self.iteration, peval=self.peval,
              eval_set=self.poll._candidate_points_set,
              options=self.options, post=self.post,
              psize=self.poll.mesh.get_delta_frame_size().coordinates,
              step_name='Poll Step', mesh=mk,
              constraints_relaxation=self.poll.constraints_rp.__dict__,
              budget=self.options.budget)

        else:
          self.poll.point_index = -1
          # """ Parallel evaluation for points in the samples set """
          self.poll.bb_eval, xt, self.post, self.peval = \
              self.poll.bb_handle.run_callable_parallel_local(
                  iter=self.iteration, peval=self.peval,
                  eval_set=self.poll._candidate_points_set,
                  options=self.options,
                  post=self.post, mesh=mk,
                  step_name='Poll Step',
                  psize=self.poll.mesh.get_delta_frame_size().coordinates,
                  constraints_relaxation=self.poll.constraints_rp.__dict__,
                  budget=self.options.budget)

        if self.poll.bb_handle.constraints_relaxation:
          temp: ConstraintsRelaxationParameters = ConstraintsRelaxationParameters(
              **self.poll.bb_handle.constraints_relaxation)
          for i, _ in enumerate(temp.lambda_multipliers):
            self.poll.constraints_rp.lambda_multipliers[i] = temp.lambda_multipliers[i]
          self.poll.constraints_rp.rho = temp.rho
          self.poll.constraints_rp.constraints_type = temp.constraints_type
          self.poll.constraints_rp.hmax = temp.hmax

        self.lambda_multipliers_k = self.poll.bb_handle.constraints_relaxation[
            "lambda_multipliers"]
        self.rho_k = self.poll.bb_handle.constraints_relaxation["rho"]
        self.poll.postprocess_evaluated_candidates(xt)

        idx = -1
        xpost: List[CandidatePoint] = []
        for i, _ in enumerate(xt):
          xpost.append(xt[i])
          self.post.poll_dirs.append(xpost[i])
        # assert len(self.HT.hash_id) - self.active_barrier.last_index + 1== 0
        for cp in xt:
          idx += 1

          if cp.is_feasible():
            self.stats.neval_bb_feasible += 1
            _, insertion_flag = self.active_barrier.update_feas_with_point(
                cp)
          else:
            self.stats.neval_bb_infeasible += 1
            _, insertion_flag = self.active_barrier.update_inf_with_point(
                cp)
          self.active_barrier.parent_indexes[self.active_barrier.last_index] = \
              parent_index_candidates[idx]
          success_flag = SUCCESS_TYPES.US if insertion_flag is None else self.compute_success(
              insertion_flag=insertion_flag)

          if insertion_flag is not None and insertion_flag in [
                  INSERTION_FLAG.DOMINATES, INSERTION_FLAG.EXTENDS]:
            x_parent = np.array(self.poll.hashtable.get_cache_candidate_points()[
                                parent_index_candidates[idx]].coordinates)
            self.active_barrier.meshes[self.active_barrier.last_index].enlarge_delta_frame_size(np.array(
                self.poll.hashtable.cache_dict[self.poll.hashtable.hash_id[-1]].coordinates)-x_parent)

          # Update success flag
          if success_flag.value > (
                  self.state.last_success.value
                  if
                  isinstance(self.state.last_success, SUCCESS_TYPES) else
                  SUCCESS_TYPES[self.state.last_success].value):
            self.state.last_success = success_flag

    self.lambda_multipliers_k = self.poll.LAMBDA
    self.rho_k = self.poll.xmin.rho
    self.post.xmin = self.poll.xmin
    toc = time.perf_counter()

    if self.log is not None:
      # log.log_msg(msg=" ---Run Summary--- ", msg_type=MSG_TYPE.INFO)
      self.log.log_msg(
          msg=f" Run completed in {toc - tic:.4f} seconds",
          msg_type=MSG_TYPE.INFO)
      self.log.log_msg(
          msg=f" Success status: {self.poll.success}", msg_type=MSG_TYPE.INFO)
      self.log.log_msg(msg=self.post.__str__(), msg_type=MSG_TYPE.INFO)

    self.poll.hashtable.add_to_best_cache(
        self.active_barrier.get_all_points())
    self.hashtable = self.poll.hashtable
    # self.active_barrier = B
    self.lambda_multipliers_k = self.poll.xmin.lambda_multipliers
    return self.poll.xmin

  def update(self):
    """Update step
    """
    # Phase one is over.
    if self.state.stop_reason == STOP_TYPE.STOP_IF_FEASIBLE:
      self.state.is_phase_one = False
      self.state.stop_reason = STOP_TYPE.UNKNOWN_STOP_REASON

    # No need to update in this case
    if self.state.stop_reason in [
            STOP_TYPE.DELTA_M_MIN_REACHED, STOP_TYPE.MAX_BB_EVAL_REACHED,
            STOP_TYPE.MAX_BB_EVAL_REACHED]:
      return

    # There can remain some points in the set Uk which have better h-value,
    # To check if the flag is activated
    if not self.options.use_nomad_partial_success:
      if self.state.last_success == SUCCESS_TYPES.US and self.state.uk_frame_center != -1:
        tmp_hx_inf_min = min(
            elt.h for elt in self.active_barrier.x_filter_inf)
        if tmp_hx_inf_min < self.active_barrier.elements[self.state.uk_frame_center].h:
          self.state.last_success = SUCCESS_TYPES.PS

    # Update barrier and mesh
    if self.state.last_success == SUCCESS_TYPES.US:
      # Set to null the last success directions of the current incumbents.
      for frame_center in range(2):
        barrier_index = self.state.ordered_frame_centers[frame_center]
        if barrier_index != -1:
          self.active_barrier.parent_indexes[barrier_index] = 0

      # Update the mesh
      if self.state.fk_frame_center != -1:
        self.active_barrier.meshes[self.state.fk_frame_center].refine_delta_frame_size(
        )
      else:
        self.active_barrier.meshes[self.state.ordered_frame_centers[0]
                                   ].refine_delta_frame_size()

      # Update the barrier threshold
      if self.state.h_max is not None:
        below_hmax_elements = [
            elt for elt in self.active_barrier.elements
            if elt is not None and elt.h < self.state.h_max and elt in
            self.active_barrier.x_filter_inf]

        if self.state.uk_frame_center != -1 and len(below_hmax_elements) != 0:
          h_max_tmp = max(elt.h for elt in below_hmax_elements)
          if h_max_tmp > self.active_barrier.elements[self.state.uk_frame_center].h:
            self.active_barrier.update_barrier(h_max_tmp)
          else:
            self.active_barrier.update_barrier(
                self.active_barrier.elements[self.state.uk_frame_center].h)

      self.stats.nno_successes += 1

    elif self.state.last_success == SUCCESS_TYPES.PS:
      # Update the barrier threshold. This follows the implementation of Nomad 3.
      if self.state.h_max is not None:
        below_hxi_elements = [
            elt for elt in self.active_barrier.elements
            if elt is not None and elt.h < self.active_barrier.elements
            [self.state.uk_frame_center].h and self.active_barrier.within_uk
            [self.active_barrier.elements.index(elt)]]
        if below_hxi_elements is not None:
          self.active_barrier.update_barrier(
              max(elt.h for elt in below_hxi_elements))

      self.stats.npartial_successes += 1

    else:  # Full success
      # Update the barrier threshold

      if self.state.h_max is not None:
        below_hmax_elements = []
        for elt in self.active_barrier.elements:
          if elt is not None and elt.h < self.state.h_max:
            for elt2 in self.active_barrier.x_filter_inf:
              if elt.coordinates == elt2.coordinates:
                below_hmax_elements.append(elt)
        if self.state.uk_frame_center != -1 and below_hmax_elements:
          h_max_tmp = max(elt.h for elt in below_hmax_elements)
          if h_max_tmp > self.active_barrier.elements[self.state.uk_frame_center].h:
            self.active_barrier.update_barrier(h_max_tmp)
          else:
            self.active_barrier.update_barrier(
                self.active_barrier.elements[self.state.uk_frame_center].h)

      # In the case of phase one, update the mesh; otherwise, it was already set before
      if self.state.is_phase_one:
        emin = None
        cmin = 0
        c = 0
        for e in self.active_barrier.elements:
          if emin is None or (e is not None and e < emin):
            emin = copy.deepcopy(e)
            cmin = copy.deepcopy(c)
          c += 1
        new_incumbent_index = cmin
        x_parent = np.array((self.hashtable.get_cache_candidate_points()[
                            self.active_barrier.parent_indexes[new_incumbent_index]]).coordinates)
        self.active_barrier.meshes[new_incumbent_index].enlarge_delta_frame_size(np.array(
            (self.hashtable.get_cache_candidate_points()[new_incumbent_index]).coordinates)
            - np.array(x_parent))

      self.stats.nfull_successes += 1
    if self.param.is_pareto:
      assert self.active_barrier.last_index == len(self.hashtable.hash_id)-1

    # Reset state
    self.state.last_success = SUCCESS_TYPES.US

  def set_frame_centers_and_hvalues(self):
    """Select frame centers (prim and sec)
    """
    # First case: phase one has been triggered.
    # There is only one primary frame center, the one with minimum h value.
    if self.state.is_phase_one:
      emin = None
      cmin = 0
      c = 0
      for e in self.active_barrier.elements:
        if emin is None or (e is not None and e < emin):
          emin = copy.deepcopy(e)
          cmin = copy.deepcopy(c)
        c += 1

      self.state.ordered_frame_centers = (
          cmin,
          0
      )
    else:
      out = self.active_barrier.frame_centers(
          self.param.w_min, use_dom_selection=True)
      fk_index = out["feasible"]
      uk_index = out["infeasible"]
      # Set primary and secondary frame centers according to trigger conditions.
      if fk_index == -1:
        self.state.ordered_frame_centers = (uk_index, -1)
      else:
        if uk_index == -1:
          self.state.ordered_frame_centers = (fk_index, -1)
        else:
          if True:  # self.options.use_doM_trigger:
            # Using extent is slightly more efficient
            dom = min(
                [sum(elt.f - np.minimum(self.active_barrier.elements[uk_index].fs.coordinates,
                     elt.fs.coordinates)) for elt in self.active_barrier.get_fk()]
            )
            #  if doM >= self.params.ρ_trigger * self.bbproblem.meta.noutputs
            if dom >= self.param.rho * self.active_barrier.extent():
              self.state.ordered_frame_centers = (
                  uk_index, fk_index)
            else:
              self.state.ordered_frame_centers = (
                  fk_index, uk_index)
          else:  # Classic alternative based on dominance but slightly less efficient.
            if all(
                self.active_barrier.elements[fk_index].f -
                    self.param.rho >= self.active_barrier.elements[uk_index].f
            ):
              self.state.ordered_frame_centers = (
                  uk_index, fk_index)
            else:
              self.state.ordered_frame_centers = (
                  fk_index, uk_index)

      self.state.fk_frame_center = fk_index if isinstance(
          fk_index, int) else int(fk_index)
      self.state.uk_frame_center = uk_index if isinstance(
          uk_index, int) else int(uk_index)

      # set h_max
      if len(self.active_barrier.get_ik()) > 0:
        self.state.h_max = max(
            elt.h for elt in self.active_barrier.get_ik())

  def compute_success(self, insertion_flag) -> SUCCESS_TYPES:
    """Compute success type

    :param insertion_flag: _description_
    :type insertion_flag: _type_
    :return: success typr
    :rtype: SUCCESS_TYPES
    """
    v: CandidatePoint = self.active_barrier.elements[self.active_barrier.last_index]

    # Phase one
    if self.state.is_phase_one:
      if v.h < self.active_barrier.elements[self.state.ordered_frame_centers[0]].h:
        return SUCCESS_TYPES.FS
      else:
        return SUCCESS_TYPES.US
    else:
      success_flag = SUCCESS_TYPES.US

      # Feasible case: full success as soon as a new non-dominated point which dominates
      # the current feasible frame center is generated.
      if v.is_feasible():
        # The set of feasible points can be empty before the insertion of the new point.
        if self.state.fk_frame_center != -1:
          if self.options.use_dms_success and insertion_flag in [
                  'dominates', 'extends', 'improves']:
            success_flag = SUCCESS_TYPES.FS
          if v <= self.active_barrier.elements[self.state.fk_frame_center]:
            success_flag = SUCCESS_TYPES.FS
        # If Fk is empty, consider it as a full success, similar to the Nomad software.
        else:
          success_flag = SUCCESS_TYPES.FS
      else:
        # The first iterations are considered as a partial success when the progressive barrier
        # approach is chosen.
        if self.state.h_max is None:
          if self.active_barrier.h_max != 0:
            success_flag = SUCCESS_TYPES.PS
        else:
          if self.state.uk_frame_center != -1:
            # Partial success if h(x) below h(x_inf)
            if v.h < self.active_barrier.elements[self.state.uk_frame_center].h:
              success_flag = SUCCESS_TYPES.PS
          # Success if change in Iᵏ with h(x) <= h_max.
          if self.state.uk_frame_center != -1 and \
                  v <= self.active_barrier.elements[self.state.uk_frame_center]:
            success_flag = SUCCESS_TYPES.FS
            #  if v.h <= model.barrier.elements[model.state.Uk_frame_center].h
            # and insertion_flag in ['dominates', 'extends', 'improves']:
            #      success_flag = FULL_SUCCESS
            #  end

      return success_flag


def main(*args) -> Dict[str, Any]:
  """ Otho-MADS main algorithm """
  # COMPLETED: add more checks for more defensive code

  # """ Parse the parameters files """
  if isinstance(args[0], dict):
    data = args[0]
  elif isinstance(args[0], str):
    if os.path.exists(os.path.abspath(args[0])):
      _, file_extension = os.path.splitext(args[0])
      if file_extension == ".json":
        try:
          with open(args[0], encoding='utf-8') as file:
            data = json.load(file)
        except IOError as exc:
          raise IOError('invalid json file: ' + args[0]) from exc
      else:
        raise IOError(f"The input file {args[0]} is not a JSON dictionary. "
                      f"Currently, OMADS supports JSON files solely!")
    else:
      raise IOError(f"Couldn't find {args[0]} file!")
  else:
    raise IOError(
        "The first input argument couldn't be recognized. "
        "It should be either a dictionary object or a JSON file that holds "
        "the required input parameters.")
  # """ Initialize the log file """
  log = logger()
  if not os.path.exists(data["param"]["post_dir"]):
    try:
      os.mkdir(data["param"]["post_dir"])
    except KeyError:
      os.makedirs(data["param"]["post_dir"], exist_ok=True)

  log.initialize(data["param"]["post_dir"] + "/OMADS.log")

  # """ Run preprocessor for the setup of
  #  the optimization problem and for the initialization
  # of optimization process """
  mads_agent: MADS = MADS(data=data)
  # iteration: int

  # """ Set the random seed for results reproducibility """
  if len(args) < 4:
    np.random.seed(mads_agent.options.seed)
  else:
    np.random.seed(int(args[3]))

  # """ Start the count down for calculating the runtime indicator """
  tic = time.perf_counter()

  if mads_agent.search.type == SEARCH_TYPE.VNS.name:
    search_vn = VNS(active_barrier=mads_agent.active_barrier,
                    params=mads_agent.param)
    search_vn._ns_dist = [
        int(
            ((mads_agent.search.dim + 1) / 2) *
            ((mads_agent.search.dim + 2) / 2) /
            (len(search_vn._dist)))
        if mads_agent.search.ns is None else
        mads_agent.search.ns] * len(search_vn._dist)
    mads_agent.search.ns = sum(search_vn._ns_dist)
  else:
    search_vn = None

  mads_agent.state.h_max = mads_agent.active_barrier.h_max
  # mads_agent.lambda_multipliers_k = lambda_multipliers
  # original_st = copy.deepcopy(mads_agent.search.sampling_t)
  mads_agent.init_phase()
  while True:
    # """ Run search step (Optional) """
    # COMPLETED: This rule cannot be generalized -- needs further invistigation
    # if poll.dim > 10 and poll.mesh.psize >= 1E-4:
    #   canSearch = False
    # else:
    can_search = True

    # """ Run the poll step (mandatory step) """
    if mads_agent.state.last_success == SUCCESS_TYPES.US:
      mads_agent.log.log_msg(
          f"------- Iteration # {mads_agent.iteration}: Run the poll step -------",
          MSG_TYPE.INFO)
      mads_agent.set_frame_centers_and_hvalues()
      xmin = mads_agent.poll_step()
      xmin = mads_agent.poll.xmin
      mads_agent.search.mesh = copy.deepcopy(mads_agent.poll.mesh)
      mads_agent.search.psize = copy.deepcopy(mads_agent.poll.psize)
    # """ Run the search step (optional step) """
    if can_search and (mads_agent.state.last_success == SUCCESS_TYPES.US):
      mads_agent.log.log_msg(
          f"------- Iteration # {mads_agent.iteration}: Run the search step -------",
          MSG_TYPE.INFO)
      mads_agent.search.iter = mads_agent.iteration
      mads_agent.stats.niterations = mads_agent.iteration
      mads_agent.set_frame_centers_and_hvalues()
      xmin = mads_agent.search_step()
    # """ Check stopping criteria"""
    pt = (
        all(
            abs(mads_agent.poll.mesh.get_delta_frame_size().coordinates[pp]) <
            mads_agent.options.tol for pp in range(mads_agent.poll.n)))
    st = (
        all(
            abs(
                mads_agent.search.mesh.get_delta_mesh_size().coordinates
                [pp]) < mads_agent.options.tol
            for pp in range(mads_agent.search.mesh.n)))
    # bt = all(abs(mads_agent.active_barrier.meshes[mads_agent.state.fk_frame_center].get_delta_frame_size(
    # ).coordinates[pp]) < mads_agent.options.tol for pp in range(mads_agent.poll.n))
    if mads_agent.state.fk_frame_center == -1:
      mk = mads_agent.active_barrier.meshes[mads_agent.state.ordered_frame_centers[0]]
    else:
      mk = mads_agent.active_barrier.meshes[mads_agent.state.fk_frame_center]
    if all(
        [mk.get_delta_frame_size().coordinates[pp] < mads_agent.options.tol
         for pp in range(mads_agent.poll.n)]):
      mads_agent.state.stop_reason = STOP_TYPE.MIN_MESH_REACHED
    # bt = mads_agent.state.stop_reason == STOP_TYPE.MIN_MESH_REACHED
    # last_success = mads_agent.state.last_success
    # if not isinstance(mads_agent.state.last_success, SUCCESS_TYPES)
    # else mads_agent.state.last_success.name
    # """ Updates """
    mads_agent.update()
    if mads_agent.options.save_results:
      mads_agent.post.output_results(mads_agent.out, False)
      if mads_agent.param.is_pareto:
        mads_agent.post.nd_points = []
        for i in range(len(mads_agent.active_barrier.get_all_points())):
          mads_agent.post.nd_points.append(
              mads_agent.active_barrier.get_all_points()[i])
        mads_agent.post.output_nd_results(mads_agent.out_p)
    if (pt or st or mads_agent.state.stop_reason == STOP_TYPE.MIN_MESH_REACHED or mads_agent.search.bb_eval + mads_agent.poll.bb_eval
            >= mads_agent.options.budget):
      mads_agent.log.log_msg(
          "\n--------------- Termination of MADS  ---------------", MSG_TYPE.INFO)
      if pt:
        mads_agent.log.log_msg(
            "Termination criterion hit: the poll size is below the minimum threshold defined.",
            MSG_TYPE.INFO)
      if st:
        mads_agent.log.log_msg(
            "Termination criterion hit: the mesh size is below the minimum threshold defined.",
            MSG_TYPE.INFO)
      if mads_agent.search.bb_eval + mads_agent.poll.bb_eval >= mads_agent.options.budget:
        mads_agent.log.log_msg(
            "Termination criterion hit: Evaluation budget is exhausted.",
            MSG_TYPE.INFO)
      mads_agent.log.log_msg(
          "----------------------------------------------------\n", MSG_TYPE.INFO)
      break

    toc = time.perf_counter()
    if mads_agent.state.stop_reason == STOP_TYPE.STOP_IF_FEASIBLE:
      if mads_agent.options.display:
        mads_agent.log.log_msg(
            "Feasible point found: end of phase one.", MSG_TYPE.INFO)

    mads_agent.progress_bar_colored()
    mads_agent.iteration += 1

  if mads_agent.param.is_pareto:
    rp: Optional[CandidatePoint] = None
    if mads_agent.param.ref_point:
      rp = Point()
      rp.coordinates = mads_agent.param.ref_point
    perf_m = Metrics(
        nd_solutions=[elem
                      for elem in
                      mads_agent.active_barrier.get_fk()],
        nobj=mads_agent.active_barrier.nobj, ref_point=rp)
    mads_agent.hv = perf_m.hypervolume()

  out_step: Any = None
  if mads_agent.poll.xmin < mads_agent.search.xmin:
    out_step = mads_agent.poll
  elif mads_agent.search.xmin < mads_agent.poll.xmin:
    out_step = mads_agent.search
  else:
    out_step = mads_agent.poll

  if out_step is None:
    out_step = mads_agent.poll

  if mads_agent.options.display:
    print(" end of orthogonal MADS ")
    print(" Final objective value: " + str(out_step.xmin.f) +
          ", hmin= " + str(out_step.xmin.h))

  if mads_agent.options.save_coordinates:
    mads_agent.post.output_coordinates(mads_agent.out)

  if mads_agent.log is not None:
    mads_agent.log.log_msg(
        msg=" --- MADS Run Summary--- ", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" Run completed in {toc - tic:.4f} seconds",
        msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" # of successful search steps = {mads_agent.search.n_successes}",
        msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" # of successful poll steps = {mads_agent.poll.n_successes}",
        msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" Run completed in {toc - tic:.4f} seconds",
        msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" Random numbers generator's seed {mads_agent.options.seed}",
        msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" xmin = {mads_agent.poll.xmin} ",
        msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" hmin = {mads_agent.poll.xmin.h} ", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" fmin {mads_agent.poll.xmin.fobj}", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" Search step # BB evals =  {mads_agent.search.bb_eval} ",
        msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" Poll step # BB evals =  {mads_agent.poll.bb_eval} ",
        msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" Total # BB evals =  {mads_agent.poll.bb_eval + mads_agent.search.bb_eval} ",
        msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" #iterations =  {mads_agent.iteration} ", msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" psize = {mads_agent.poll.mesh.get_delta_frame_size().coordinates} ",
        msg_type=MSG_TYPE.INFO)
    mads_agent.log.log_msg(
        msg=f" psize_success = {mads_agent.poll.xmin.mesh.get_delta_frame_size().coordinates}",
        msg_type=MSG_TYPE.INFO)
    if mads_agent.param.is_pareto or isinstance(
            mads_agent.active_barrier, BarrierMO):
      mads_agent.log.log_msg(
          msg=f" Hypervolume metric = {mads_agent.hv}", msg_type=MSG_TYPE.INFO)
  if mads_agent.options.display:
    print("\n ---MADS Run Summary---")
    print(f" Run completed in {toc - tic:.4f} seconds")
    print(f" Random numbers generator's seed {mads_agent.options.seed}")
    print(" xmin = " + str(out_step.xmin))
    print(" hmin = " + str(out_step.xmin.h))
    print(" fmin = " + str(out_step.xmin.f))
    print(" #bb_eval = " + str(out_step.bb_eval))
    print(" #iteration = " + str(mads_agent.iteration))
    print(" nb_success = " +
          str(mads_agent.poll.nb_success + mads_agent.search.nb_success))
    print(" psize = " + str(mads_agent.poll.mesh.get_delta_frame_size().coordinates))
    print(" psize_success = " +
          str(mads_agent.poll.xmin.mesh.get_delta_frame_size().coordinates))

  xmin = out_step.xmin
  # """ Evaluation of the blackbox; get output responses """
  if xmin.sets is not None and isinstance(xmin.sets, dict):
    p: List[Any] = []
    for i, _ in enumerate(xmin.var_type):
      if (xmin.var_type[i] == VAR_TYPE.DISCRETE or xmin.var_type[i] == VAR_TYPE.CATEGORICAL) \
              and xmin.var_link[i] is not None:
        p.append(xmin.sets[xmin.var_link[i]][int(xmin.coordinates[i])])
      else:
        p.append(xmin.coordinates[i])
  else:
    p = xmin.coordinates
  output: Dict[str, Any] = {"xmin": p,
                            "fmin": out_step.xmin.f,
                            "hmin": out_step.xmin.h,
                            "nbb_evals": out_step.bb_eval,
                            "niterations": mads_agent.iteration,
                            "nb_success": mads_agent.poll.nb_success + mads_agent.search.nb_success,
                            "psize": mads_agent.poll.mesh.get_delta_frame_size().coordinates,
                            "psuccess": mads_agent.poll.xmin.mesh.get_delta_frame_size().coordinates,
                            # "pmax": poll.mesh.psize_max,
                            "msize": out_step.mesh.get_delta_mesh_size().coordinates,
                            "HV": mads_agent.hv if mads_agent.param.is_pareto else "NA"}
  return output, out_step


if __name__ == "__main__":
  freeze_support()
  p_file: str = os.path.abspath("")

  """ Check if an input argument is provided"""
  if len(sys.argv) > 1:
    p_file = os.path.abspath(sys.argv[1])
    main(p_file)

  if (p_file != "" and os.path.exists(p_file)):
    main(p_file)

  if p_file == "":
    raise IOError(
        "Undefined input args."
        " Please specify an appropriate input (parameters) jason file")
