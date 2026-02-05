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
    :param tol: The threshold of the minimum poll size at which the run will be terminated
    :param psize_init: Initial poll size
    :param dispaly: Print the study progress during the run
    :param opportunistic: Loop on the points populated in the poll set until a better 
    minimum found, then stop the evaluation loop
    :param check_cache: Check the hash table before points evaluation to avoid duplicates
    :param store_cache: Enable storing evaluated points into the hash table
    :param collect_y: Collect dependent design variables (required for DMDO)
    :param rich_direction: Go with the rich direction (Impact the mesh size update)
    :param precision: Define the precision level
    :param save_results: A boolean flag that indicates saving results in a csv file
    :param save_coordinates: A boolean flag that indicates saving coordinates of the poll
      set in a JSON file (required to generate animations of the spinner)
    :param save_all_best: A boolean used to check whether saving best points only in
      the MADS.out file
    :param parallel_mode: A boolean to check whether evaluating the poll set 
    in parallel multiprocessing
    :param np: The number of CPUs
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
      use_nomad_partial_success: bool = True,
      use_penalty_approach: bool = False,
      noutbound_hits_max: int = 60000,
      use_dom_trigger: bool = True,
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
    self.use_nomad_partial_success = use_nomad_partial_success
    self.use_penalty_approach = use_penalty_approach
    self.noutbound_hits_max = noutbound_hits_max
    self.use_dom_trigger = use_dom_trigger
