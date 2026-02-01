"""
# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                               #
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
import os
from typing import List, Dict
import warnings
import copy

import numpy as np


from ._globals import DType, VAR_TYPE, BARRIER_TYPES, MESH_TYPE
from .point import Point


class Parameters:
  """ Variables and algorithmic parameters 

    :param baseline: Baseline design point (initial point ``x0``)
    :param lb: The variables lower bound
    :param ub: The variables upper bound
    :param var_names: The variables name
    :param scaling: Scaling factor (can be defined as a list 
    (assigning a factor for each variable) or a scalar value 
    that will be applied on all variables)
    :param post_dir: The location and name of the post directory 
    where the output results file will live in (if any)
  """

  def __init__(
          self,
          baseline: List[float] = None,
          lb: List[float] = None,
          ub: List[float] = None,
          var_names: List[str] = None,
          fun_names: List[str] = None,
          function_weights: List[float] = None,
          scaling: float = 10.0,
          post_dir: str = os.path.abspath("./"),
          var_type: List[str] = None,
          var_sets: Dict = None,
          constants: List = None,
          constants_name: List = None,
          failure_stop: bool = None,
          problem_name: str = "unknown",
          best_known: List[float] = None,
          constraints_type: List[BARRIER_TYPES] = None,
          h_max: float = np.inf,
          rho: float = 0.00005,
          lambda_multipliers: List[float] = None,
          name: str = "undefined",
          mesh_type: str = MESH_TYPE.ORTHO.name,
          fixed_variables: List[float] = None,
          granularity: List[float] = None,
          min_mesh_size: List[float] = None,
          min_frame_size: List[float] = None,
          initial_mesh_size: List[float] = None,
          initial_frame_size: List[float] = None,
          is_pareto: bool = False,
          nobj: int = 1,
          incumbent_selection_param: int = 1,
          barrier_initialized_from_cache: bool = True,
          ref_point: List[float] = None,
          lhs_search_initialization: bool = False,
          mesh_adjustment: float = 2.0,
          w_min: int = 1):
    self.w_min = w_min
    self.incumbent_selection_param = incumbent_selection_param
    self.barrier_initialized_from_cache = barrier_initialized_from_cache
    self.nobj = nobj
    self.baseline = baseline
    is_list_of_list = all(isinstance(item, list) for item in self.baseline)
    self._n = len(self.baseline[0]) if is_list_of_list else len(baseline)
    self.x0 = Point(self._n)
    if isinstance(self.baseline, Point):
      self.x0 = copy.deepcopy(self.baseline)
    elif isinstance(self.baseline, List):
      if all(isinstance(item, list) for item in self.baseline):
        self.x0 = [Point(self._n)] * len(self.baseline)
        for i, _ in enumerate((self.baseline)):
          self.x0[i].coordinates = copy.deepcopy(self.baseline[i])
      elif all(isinstance(item, Point) for item in self.baseline):
        self.x0 = [Point(self._n)] * len(self.baseline)
        for i, _ in enumerate((self.baseline)):
          self.x0[i] = copy.deepcopy(self.baseline[i])
      else:
        self.x0.coordinates = self.baseline
    self.lb = lb
    self.ub = ub
    self.var_names = var_names if var_names else [
        f'x_{i}' for i in range(self._n)]
    self.fun_names = fun_names if fun_names else ["fobj"]
    self.function_weights = (np.divide(function_weights, np.sum(
        function_weights))).tolist() if function_weights else [1 / (self.nobj)] * self.nobj
    self.scaling = scaling
    self.post_dir = post_dir
    self.var_type = var_type
    self.constants = constants
    self.constants_name = constants_name
    self.failure_stop: bool = failure_stop
    self.problem_name = problem_name
    self.best_known = best_known
    self.constraints_type = constraints_type
    self.h_max = h_max
    self.rho = rho
    self.lambda_multipliers = lambda_multipliers
    self.name = name
    self.var_sets = var_sets
    self.is_pareto = is_pareto
    self.lhs_search_initialization = lhs_search_initialization
    # Mesh options
    self.mesh_type = mesh_type
    point_init = Point()
    point_init.reset(self._n, d=0)
    if fixed_variables:
      self.fixed_variables = point_init
      # self.fixed_variables
      self.fixed_variables.coordinates = fixed_variables
    else:
      self.fixed_variables = None

    self.granularity = copy.deepcopy(point_init)
    self.min_mesh_size = copy.deepcopy(point_init)
    self.min_frame_size = copy.deepcopy(point_init)
    self.initial_mesh_size = copy.deepcopy(point_init)
    self.initial_frame_size = copy.deepcopy(point_init)
    if granularity:
      self.granularity.coordinates = granularity
    if min_mesh_size:
      self.min_mesh_size.coordinates = min_mesh_size
    else:
      self.min_mesh_size.coordinates = [1E-9] * self._n
    if min_frame_size:
      self.min_frame_size.coordinates = min_frame_size
    else:
      self.min_frame_size.coordinates = [1E-9] * self._n
    if initial_mesh_size:
      self.initial_mesh_size.coordinates = initial_mesh_size
    if initial_frame_size:
      self.initial_frame_size.coordinates = initial_frame_size
    self.warning_initial_frame_size_reset: bool = True

    if self.constraints_type is not None and isinstance(
            self.constraints_type, list):
      for i, _ in enumerate((self.constraints_type)):
        if self.constraints_type[i] == BARRIER_TYPES.PB.name or\
                self.constraints_type[i] == BARRIER_TYPES.PB:
          self.constraints_type[i] = BARRIER_TYPES.PB
        elif self.constraints_type[i] == BARRIER_TYPES.RB.name or\
                self.constraints_type[i] == BARRIER_TYPES.RB:
          self.constraints_type[i] = BARRIER_TYPES.RB
        elif self.constraints_type[i] == BARRIER_TYPES.PEB.name or\
                self.constraints_type[i] == BARRIER_TYPES.PEB:
          self.constraints_type[i] = BARRIER_TYPES.PEB
        else:
          self.constraints_type[i] = BARRIER_TYPES.EB
    elif self.constraints_type is not None:
      self.constraints_type = BARRIER_TYPES(self.constraints_type)

    if self.lambda_multipliers is None:
      self.lambda_multipliers = [0]
    if not isinstance(self.lambda_multipliers, list):
      self.lambda_multipliers = [self.lambda_multipliers]

    if self.var_type is not None:
      c = 0
      for k in self.var_type:
        c += 1
        if k.lower()[0] == "d":
          if self.var_sets is not None and isinstance(self.var_sets, dict):
            if self.var_sets[k.split('_')[1]] is not None:
              if self.ub[c - 1] > len(self.var_sets[k.split('_')[1]]) - 1:
                self.ub[c - 1] = len(self.var_sets[k.split('_')[1]]) - 1
              if self.lb[c - 1] < 0:
                self.lb[c - 1] = 0
    if constants:
      self.fixed_variables = point_init
      self.fixed_variables.coordinates = constants

    if self.granularity and self.granularity.size != self._n:
      if self.granularity.size > 0 and self.granularity.is_all_defined():
        raise IOError(
            f'Parameter granularity has dimension {self.granularity.size} \
                which is different from problem dimension {self._n}')
      self.granularity.reset(self._n, 0.0)

    for i in range(self.n):
      if not self.granularity.defined[i]:
        self.granularity[i] = 0.
      elif self.granularity[i] < 0.:
        raise IOError("Check: invalid granular variables (negative values)")

    if self.var_type is None or len(self.var_type) <= 0:
      self.var_type = [VAR_TYPE.REAL.name] * self.n
    self.set_min_mesh_parameters()
    self.set_min_frame_parameters()
    self.set_initial_mesh_parameters()
    if isinstance(self.x0, list):
      for i, _ in enumerate((self.x0)):
        self.x0[i].check_for_granularity(g=self.granularity, name="baseline")
    else:
      self.x0.check_for_granularity(g=self.granularity, name="baseline")
    self.min_mesh_size.check_for_granularity(
        g=self.granularity, name="minMeshSize")
    self.min_frame_size.check_for_granularity(
        g=self.granularity, name="minFrameSize")
    self.initial_mesh_size.check_for_granularity(
        g=self.granularity, name="initialMeshSize")
    self.initial_frame_size.check_for_granularity(
        g=self.granularity, name="initialFrameSize")

    self._initialized_and_checked = True
    self.ref_point = ref_point
    self.mesh_adjustment = mesh_adjustment

  def set_initial_mesh_parameters(self):
    """Set the initial mesh parameters

    :raises IOError: _description_
    :raises IOError: _description_
    :raises IOError: _description_
    :raises IOError: _description_
    """
    if self.initial_mesh_size.is_all_defined() and self.initial_mesh_size.size != self.n:
      raise IOError(
          f"INITIAL_MESH_SIZE has dimension {self.initial_mesh_size.size} \
            which is different from problem dimension {self.n}")

    if self.initial_frame_size.is_all_defined() and self.initial_frame_size.size != self.n:
      raise IOError(
          f"INITIAL_FRAME_SIZE has dimension {self.initial_frame_size.size} \
            which is different from problem dimension {self.n}")

    if self.initial_mesh_size.is_all_defined() and self.initial_frame_size.is_all_defined():
      self.initial_mesh_size.reset(self._n)

    if not self.initial_mesh_size.is_all_defined():
      self.initial_mesh_size.reset(self.n, 0.)

    if not self.initial_frame_size.is_all_defined():
      self.initial_frame_size.reset(self.n, 0.)
    lb = self.lb
    ub = self.ub
    is_list_of_list = all(isinstance(item, list) for item in self.baseline)
    if is_list_of_list:
      min_bl = np.min(self.baseline, axis=0)
      max_bl = np.min(self.baseline, axis=0)
    else:
      min_bl = max_bl = self.baseline
    for j in range(self.n):
      if self.lb[j] is None or min_bl[j] < self.lb[j]:
        lb[j] = min_bl[j]
      if self.ub[j] is None or max_bl[j] > self.ub[j]:
        ub[j] = max_bl[j]

    for i in range(self.n):
      if self.initial_mesh_size.defined[i]:
        if self.initial_frame_size.defined[i] and self.warning_initial_frame_size_reset:
          self.warning_initial_frame_size_reset = False
          warnings.warn("Initial frame size reset from initial mesh")
        self.min_frame_size[i] = self.initial_mesh_size[i] * \
            np.power(self.n, 0.5)
        self.initial_frame_size[i] = self.initial_frame_size.next_mult(
            g=self.granularity[i], i=i)
        if self.initial_frame_size[i] < self.min_frame_size[i]:
          self.initial_frame_size[i] = self.min_frame_size[i]

      if not self.initial_frame_size.defined[i]:
        if lb[i] is not None and ub[i] is not None:
          self.initial_frame_size[i] = (ub[i] - lb[i]) / 10
        elif lb[i] is not None and self.lb[i] is not None and lb[i] != self.lb[i]:
          self.initial_frame_size[i] = (lb[i] - self.lb[i]) / 10.0
        elif (ub[i] is not None and self.ub[i] is not None and ub[i] != self.ub[i]):
          self.initial_frame_size[i] = (self.ub[i] - ub[i]) / 10.0
        else:
          if lb[i] is not None and abs(lb[i]) > DType("high")._zero * 10.0:
            self.initial_frame_size[i] = abs(lb[i]) / 10.0
          else:
            self.initial_frame_size[i] = 1.0
        # Adjust value with granularity
        self.initial_frame_size[i] = self.initial_frame_size.next_mult(
            g=self.granularity[i], i=i)
        # Adjust value with minFrameSize
        if self.initial_frame_size[i] < self.min_frame_size[i]:
          self.initial_frame_size[i] = self.min_frame_size[i]
      # Determine initial mesh size from initial frame size
      if not self.initial_mesh_size.defined[i]:
        self.initial_mesh_size[i] = self.initial_frame_size[i] * self.n**-0.5
        # Adjust value with granularity
        self.initial_mesh_size[i] = self.initial_mesh_size.next_mult(
            g=self.granularity[i], i=i)
        # Adjust value with minMeshSize
        if self.initial_mesh_size[i] < self.min_mesh_size[i]:
          self.initial_mesh_size[i] = self.min_mesh_size[i]

      if self.min_mesh_size[i] > self.initial_mesh_size[i]:
        raise IOError("Check: initial mesh size is lower than min mesh size.\n"
                      + f"INITIAL_MESH_SIZE  + {self.initial_mesh_size[i]} \n"
                      + f"MIN_MESH_SIZE {self.min_mesh_size[i]}")
      if self.min_frame_size[i] > self.min_frame_size[i]:
        raise IOError(
            "Check: initial frame size is lower than min frame size.\n" +
            f"INITIAL_FRAME_SIZE  + {self.min_frame_size[i]} \n" +
            f"MIN_FRAME_SIZE {self.min_frame_size[i]}")

  def set_min_mesh_parameters(self):
    """Set minimum mesh parameters

    :raises IOError: _description_
    :raises IOError: _description_
    :raises IOError: _description_
    """
    if not self.min_mesh_size.is_all_defined():
      for i in range(self.n):
        if self.granularity[i] > 0.0:
          self.min_mesh_size[i] = self.granularity[i]
    else:
      if self.min_mesh_size.size != self.n:
        raise IOError(
            f"minMeshSize parameter of size {self.min_frame_size.size} \
              doesn't match the parameters dimensionality of {self.n}")
      for i in range(self.n):
        if self.min_mesh_size.defined[i] and self.min_mesh_size[i] < 0.0:
          raise IOError(
              f"Invalid minMeshSize defined of value {self.min_mesh_size[i]}")
        elif (not self.min_mesh_size.defined[i]) or \
            (0.0 < self.granularity[i]
             and self.min_mesh_size[i] < self.granularity[i]):
          if self.granularity[i] > 0.0:
            self.min_mesh_size[i] = self.granularity[i]
          else:
            raise IOError(
                "Error: granularity is defined with a negative value.")

  def set_min_frame_parameters(self):
    """Minimum frame size parameters

    :raises IOError: _description_
    :raises IOError: _description_
    :raises IOError: _description_
    """
    if not self.min_frame_size.is_all_defined():
      for i in range(self.n):
        if self.granularity[i] > 0.0:
          self.min_frame_size[i] = self.granularity[i]
    else:
      if self.min_frame_size.size != self.n:
        raise IOError(
            f"minFrameSize parameter of size {self.min_frame_size.size} \
              doesn't match the parameters dimensionality of {self.n}")
      for i in range(self.n):
        if self.min_frame_size.defined[i] and self.min_frame_size[i] < 0.0:
          raise IOError(
              f"Invalid minFrameSize defined of value {self.min_frame_size[i]}")
        elif (not self.min_frame_size.defined[i]) or \
            (0.0 < self.granularity[i] and
                self.min_frame_size[i] < self.granularity[i]):
          if self.granularity[i] > 0.0:
            self.min_frame_size[i] = self.granularity[i]
          else:
            raise IOError(
                "Error: granularity is defined with a negative value.")

  def to_be_checked(self) -> bool:
    """Check whther parameters been initialized and checked

    :return: _description_
    :rtype: bool
    """
    return self._initialized_and_checked

  # TODO: give better control on variabls' resolution (mesh granularity)
  # var_type: List[str] = field(default_factory=["cont", "cont"])
  # resolution: List[int] = field(default_factory=["cont", "cont"])

  def get_barrier_type(self):
    """Get the constraints barrier type

    :return: _description_
    :rtype: _type_
    """
    if self.constraints_type is not None:
      if isinstance(self.constraints_type, list):
        for i, _ in enumerate((self.constraints_type)):
          if self.constraints_type[i] == BARRIER_TYPES.PB:
            return BARRIER_TYPES.PB
      else:
        if self.constraints_type == BARRIER_TYPES.PB:
          return BARRIER_TYPES.PB

    return BARRIER_TYPES.EB

  def get_h_max_0(self):
    """Get constraints violation margine

    :return: _description_
    :rtype: _type_
    """
    return self.h_max

  @property
  def n(self) -> int:
    """Get the problem number of dimensions

    :return: _description_
    :rtype: int
    """
    return self._n

  @n.setter
  def n(self, value: int) -> int:
    self._n = value
