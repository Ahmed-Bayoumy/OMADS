from dataclasses import dataclass
import logging
from typing import Any
import numpy as np
from .Point import Point
from ._globals import *

@dataclass
class Options:
  """ The running study and algorithmic options of OMADS
  
    :param seed: Random generator seed
    :param budget: The evaluation budget
    :param tol: The threshold of the minimum poll size at which the run will be terminated
    :param psize_init: Initial poll size
    :param dispaly: Print the study progress during the run
    :param opportunistic: Loop on the points populated in the poll set until a better minimum found, then stop the evaluation loop
    :param check_cache: Check the hash table before points evaluation to avoid duplicates
    :param store_cache: Enable storing evaluated points into the hash table
    :param collect_y: Collect dependent design variables (required for DMDO)
    :param rich_direction: Go with the rich direction (Impact the mesh size update)
    :param precision: Define the precision level
    :param save_results: A boolean flag that indicates saving results in a csv file
    :param save_coordinates: A boolean flag that indicates saving coordinates of the poll set in a JSON file (required to generate animations of the spinner)
    :param save_all_best: A boolean used to check whether saving best points only in the MADS.out file
    :param parallel_mode: A boolean to check whether evaluating the poll set in parallel multiprocessing
    :param np: The number of CPUs
  """
  seed: int = 0
  budget: int = 1000
  tol: float = 1e-9
  psize_init: float = 1.0
  display: bool = False
  opportunistic: bool = False
  check_cache: bool = False
  store_cache: bool = False
  collect_y: bool = False
  rich_direction: bool = False
  precision: str = "high"
  save_results: bool = False
  save_coordinates: bool = False
  save_all_best: bool = False
  parallel_mode: bool = False
  np: int = 1
  extend: Any = None
  isVerbose: bool = False
  anisotropyFactor: int = 0.1
  anistropicMesh: bool = True
  refineFreq: int = 1
