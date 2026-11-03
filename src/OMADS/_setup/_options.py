"""
# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                               #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software          #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU General Public License for more details.          #
#                                                                                     #
#  You should have received a copy of the GNU General Public License along            #
#  with this program. If not, see <https://www.gnu.org/licenses/>.                    #
#                                                                                     #
#  You can find information on OMADS at                                               #
#  https://github.com/Ahmed-Bayoumy/OMADS                                             #
#  Copyright (C) 2026  Ahmed H. Bayoumy                                               #
# ------------------------------------------------------------------------------------#
"""

from typing import Any, Optional


class Options:
  """ The running study and algorithmic options of OMADS

    :param seed: Random generator seed
    :param budget: The evaluation budget
    :param tol: The threshold of the minimum frame (poll) or mesh (search)
      size at which the run is terminated
    :param psize_init: Initial frame size of the orthogonal mesh
    :param display: Print the evaluation of the starting point and a run summary
    :param opportunistic: Stop evaluating the current poll or search set as
      soon as a full success is found
    :param check_cache: Check the cache before evaluating points to avoid duplicates
    :param store_cache: Store evaluated points in the cache (required by the
      ``mads`` and ``search`` solvers to locate frame centres)
    :param collect_y: Reserved (collect coupling variables for DMDO)
    :param rich_direction: Use random Householder poll directions, which are
      asymptotically dense. None (default): True for multiobjective problems
      (``nobj > 1``, as DMulti-MADS requires for its convergence analysis),
      False otherwise
    :param precision: Floating-point precision: "high", "medium" or "low"
    :param save_results: Write every evaluation to
      ``<post_dir>/<name>_run/<name>_all.out``
    :param save_coordinates: Write the poll points of each iteration to
      ``coords.json`` (used to animate the poll trajectory; ``poll`` solver)
    :param save_all_best: Reserved
    :param parallel_mode: Evaluate candidate sets in parallel processes
    :param np: The number of worker processes (capped at the CPU count)
    :param extend: Reserved (a ``Dirs2n`` instance to continue a run)
    :param is_verbose: Write extra diagnostics to ``OMADS.log``
    :param anisotropy_factor: GMESH: threshold of the anisotropic frame
      enlargement rule
    :param anistropic_mesh: GMESH: enable anisotropic frame enlargement
    :param refine_freq: GMESH: refine the frame every ``refine_freq``
      unsuccessful iterations
    :param use_dms_success: Count any improvement of the non-dominated set as
      a full success
    :param use_omads_partial_success: When False, an unsuccessful iteration
      becomes a partial success if an infeasible point has a lower violation
      than the infeasible frame centre
    :param use_penalty_approach: Reserved
    :param noutbound_hits_max: Stop after this many trial points hit the bounds
    :param use_dom_trigger: Choose the primary frame centre with the
      dominance-distance trigger of the progressive barrier
    :param direction_type: Poll directions: "ORTHO_2N" (2n orthogonal
      directions) or "N+1" (n orthogonal directions plus their negative sum,
      which is not orthogonal to them; real variables only, falls back to
      ORTHO_2N otherwise). After a success, both sets are ordered by their
      angle with the last successful step, and the N+1 basis takes from each
      pair of opposite directions the one closer to it. None (default): "N+1"
      for multiobjective problems (``nobj > 1``, as in DMulti-MADS),
      "ORTHO_2N" otherwise. The former name "ORTHO_N+1" is still accepted
    :param snap_to_bounds: Replace a poll or speculative point outside the
      bounds by the closest mesh point inside them (as NOMAD does). False:
      drop it unevaluated, the extreme barrier of DMulti-MADS (f = +inf
      outside the bounds)
    :param search_every: In ``mads``, run the sampling search only after every
      ``search_every``-th consecutive unsuccessful poll (1: after every one)
    :param on_min_mesh: In ``mads``, what to do when the poll or mesh size
      falls below ``tol`` before the budget is used: "stop" ends the run;
      "search" keeps running the sampling search with the initial mesh around
      the incumbent, and resumes the MADS iterations at its first improvement
    :param mo_search: For multiobjective problems, run the quadratic model
      search (a point predicted to dominate the frame centre, and the
      minimizer of one objective's model from an extreme point of the front)
      and the middle-point search with the speculative search before each
      poll, in the spirit of the DMulti-MADS search strategies of NOMAD 4.
      False: the speculative search alone, as in DMulti-MADS (Bigeon et al.
      2021)
    :param constraint_strategy: "PB" (default): the progressive barrier
      handles the constraints declared ``PB``. "TWO_PHASE": the two-phase
      extreme barrier of DMulti-MADS (Bigeon, Le Digabel & Salomon):
      minimize the constraint violation (single objective) until a feasible
      point is found, then optimize from it with every constraint treated
      as an extreme barrier. Needs a callable blackbox
  """

  def __init__(
      self,
      seed: int = 0,
      budget: int = 1000,
      tol: float = 1e-9,
      psize_init: float = 1.0,
      display: bool = False,
      opportunistic: bool = False,
      check_cache: bool = False,
      store_cache: bool = False,
      collect_y: bool = False,
      rich_direction: Optional[bool] = None,
      precision: str = "high",
      save_results: bool = False,
      save_coordinates: bool = False,
      save_all_best: bool = False,
      parallel_mode: bool = False,
      np: int = 1,
      extend: Any = None,
      is_verbose: bool = False,
      anisotropy_factor: float = 0.1,
      anistropic_mesh: bool = True,
      refine_freq: int = 1,
      use_dms_success: bool = False,
      use_omads_partial_success: bool = True,
      use_penalty_approach: bool = False,
      noutbound_hits_max: int = 60000,
      use_dom_trigger: bool = True,
      direction_type: Optional[str] = None,
      search_every: int = 1,
      snap_to_bounds: bool = True,
      on_min_mesh: str = "stop",
      mo_search: bool = True,
      constraint_strategy: str = "PB",
  ):
    self.seed = seed
    self.budget = budget
    self.tol = tol
    self.psize_init = psize_init
    self.display = display
    self.opportunistic = opportunistic
    self.check_cache = check_cache
    self.store_cache = store_cache
    self.collect_y = collect_y
    self.rich_direction = rich_direction
    self.precision = precision
    self.save_results = save_results
    self.save_coordinates = save_coordinates
    self.save_all_best = save_all_best
    self.parallel_mode = parallel_mode
    self.np = np
    self.extend = extend
    self.is_verbose = is_verbose
    self.anisotropy_factor = anisotropy_factor
    self.anistropic_mesh = anistropic_mesh
    self.refine_freq = refine_freq
    self.use_dms_success = use_dms_success
    self.use_omads_partial_success = use_omads_partial_success
    self.use_penalty_approach = use_penalty_approach
    self.noutbound_hits_max = noutbound_hits_max
    self.use_dom_trigger = use_dom_trigger
    # "ORTHO_N+1" is the former name of the N+1 set, kept for old inputs.
    self.direction_type = "N+1" if direction_type == "ORTHO_N+1" else direction_type
    self.search_every = search_every
    self.snap_to_bounds = snap_to_bounds
    if on_min_mesh not in ("stop", "search"):
      raise ValueError(
          f'on_min_mesh must be "stop" or "search", not {on_min_mesh!r}')
    self.on_min_mesh = on_min_mesh
    self.mo_search = mo_search
    if str(constraint_strategy).upper() not in ("PB", "TWO_PHASE"):
      raise ValueError('constraint_strategy must be "PB" or "TWO_PHASE", '
                       f'not {constraint_strategy!r}')
    self.constraint_strategy = str(constraint_strategy).upper()

  def poll_rich_direction(self, nobj: int) -> bool:
    """`rich_direction`, or its default for a problem with `nobj` objectives."""
    if self.rich_direction is not None:
      return self.rich_direction
    return nobj > 1

  def poll_direction_type(self, nobj: int) -> str:
    """`direction_type`, or its default for a problem with `nobj` objectives."""
    if self.direction_type is not None:
      return self.direction_type
    return "N+1" if nobj > 1 else "ORTHO_2N"
