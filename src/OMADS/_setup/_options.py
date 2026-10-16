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

from typing import Any


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
      asymptotically dense
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
      directions) or "ORTHO_N+1" (n of them plus their negative sum; real
      variables only, falls back to ORTHO_2N otherwise)
    :param search_every: In ``mads``, run the sampling search only after every
      ``search_every``-th consecutive unsuccessful poll (1: after every one)
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
      rich_direction: bool = False,
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
      direction_type: str = "ORTHO_2N",
      search_every: int = 1,
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
    self.direction_type = direction_type
    self.search_every = search_every
