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
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


from ._globals import DType, VAR_TYPE, GL_LIMITS
from .point import Point
from .mesh import Mesh
from .options import Options
from .parameters import Parameters


@dataclass
class Omesh(Mesh):
  """ Mesh coarsness update class

  :param _delta: mesh size
  :param _Delta: poll size
  :param _rho: poll size to mesh size ratio
  :param _exp:  manage the poll size granularity for discrete variables. 
  See Audet et. al, The mesh adaptive direct search algorithm for granular and discrete variable
  :param _mantissa: Same as ``_exp``
  :param psize_max: Maximum poll size
  :param psize_success: Poll size at successful evaluation
  :param _dtype: numpy double data type precision

  """
  _n: Optional[int] = None
  _mesh_size: Optional[Point] = None  # 1.0  # mesh size
  _frame_size: Optional[Point] = None  # poll size
  _rho: Optional[List[float]] = None  # poll size to mesh size ratio
  # Completed: manage the poll size granularity for discrete variables
  # A new class 'Gmesh' is now avialable.
  # Gmesh adapts mesh granularity and anistropy
  # See: Audet et. al, The mesh adaptive direct search algorithm for
  # granular and discrete variable
  _exp: Optional[Point] = None
  _mantissa: Optional[Point] = None
  _maximum_frame_size: Optional[Point] = None
  successful_frame_size: Optional[Point] = None

  # numpy double data type precision
  _dtype: Optional[DType] = None

  def __init__(self, pb_param: Parameters, run_options: Options):
    """ Constructor """
    super(
        Omesh, self).__init__(
        pb_params=pb_param, limit_max_mesh_index=-GL_LIMITS,
        limit_min_mesh_index=GL_LIMITS)
    self._n = len(pb_param.baseline)
    self.mesh_size = Point(self._n)
    self.frame_size = Point(self._n)
    self._exp = Point(self._n)
    self._mantissa = Point(self._n)
    self._maximum_frame_size = Point(self._n)
    self.successful_frame_size = Point(self._n)
    self.rho = [0] * self._n
    self.frame_size.coordinates = run_options.psize_init if isinstance(
        run_options.psize_init, list) else [run_options.psize_init] * self._n
    self.mesh_size.reset(n=self._n, d=0)
    self._r = Point(self._n)
    self._r.coordinates = [1]*self._n
    self._r_max = Point(self._n)
    self._r_max.coordinates = [1]*self._n
    self._r_min = Point(self._n)
    self._r_min.coordinates = [1]*self._n
    self.init()

  def init(self):
    self.update()

  def __post_init__(self):
    self._dtype = DType()

  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, other: DType):
    self._dtype = other

  @property
  def mesh_size(self):
    return self._mesh_size

  @mesh_size.setter
  def mesh_size(self, size):
    self._mesh_size = size

  @mesh_size.deleter
  def mesh_size(self):
    del self._mesh_size

  @property
  def frame_size(self):
    return self._frame_size

  @frame_size.setter
  def frame_size(self, size):
    self._frame_size = size

  @frame_size.deleter
  def frame_size(self):
    del self._frame_size

  @property
  def rho(self):
    return self._rho

  @rho.setter
  def rho(self, size):
    self._rho = size

  @rho.deleter
  def rho(self):
    del self._rho

  def update(self):
    for i in range(self._n):
      self.mesh_size[i] = np.minimum(np.power(
          self._frame_size[i],
          2.0, dtype=self.dtype.dtype),
          self._frame_size[i],
          dtype=self.dtype.dtype)
      self.rho[i] = np.divide(
          self._frame_size[i],
          self._mesh_size[i],
          dtype=self.dtype.dtype)

  def get_delta_mesh_size(self, i: int = None):
    if i is not None:
      return self.mesh_size[i]
    else:
      return self.mesh_size

  def get_delta_frame_size(self, i: int = None):
    if i is not None:
      return self.frame_size[i]
    else:
      return self.frame_size

  def get_rho(self):
    return self.rho

  def enlarge_delta_frame_size(self, direction: Point = None) -> bool:
    for i in range(self._n):
      self.frame_size[i] *= 2

  def refine_delta_frame_size(self) -> bool:
    for i in range(self._n):
      self.frame_size[i] /= 2

  def project_on_mesh(self, point: Point, frame_center: Point = None) -> Point:
    if frame_center is None:
      frame_center = [0.]*self._n
    if self._pb_params.var_type is None:
      self._pb_params.var_type = [VAR_TYPE.REAL.name] * self._n
    for i in range(self._n):
      if self._pb_params.var_type[i] != VAR_TYPE.CATEGORICAL.name:
        if self._pb_params.var_type[i] == VAR_TYPE.REAL.name:
          point[i] = frame_center[i] + (
              np.round((point[i]-frame_center[i])/self.mesh_size[i]) * self.mesh_size[i])
        else:
          point[i] = int(
              frame_center[i] +
              int(
                  int((point[i] - frame_center[i]) / self.mesh_size[i]) * self.mesh_size
                  [i]))
      else:
        point[i] = int(point[i])
      if point[i] < self._pb_params.lb[i]:
        point[i] = self._pb_params.lb[i] + (self._pb_params.lb[i] - point[i])
        if point[i] > self._pb_params.ub[i]:
          point[i] = self._pb_params.ub[i]
      if point[i] > self._pb_params.ub[i]:
        point[i] = self._pb_params.ub[i] - (point[i] - self._pb_params.ub[i])
        if point[i] < self._pb_params.lb[i]:
          point[i] = self._pb_params.lb[i]

    return point
