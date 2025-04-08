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
from typing import Protocol, Any, List, Optional
from ._globals import DType, M_INF_INT, P_INF_INT
from .point import Point
from .parameters import Parameters


@dataclass
class MeshData(Protocol):
  """ Orthognal Mesh class

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
  _initial_mesh_size: Optional[Point] = None  # mesh size
  _initial_frame_size: Optional[Point] = None  # poll size
  _min_mesh_size: Optional[Point] = None  # mesh size
  _min_frame_size: Optional[Point] = None  # poll size
  _lower_bound: Optional[Point] = None
  _upper_bound: Optional[Point] = None
  _is_finest: Optional[bool] = True
  _r: Optional[Point] = None
  _r_min: Optional[Point] = None
  _r_max: Optional[Point] = None
  _limit_min_mesh_index: int = M_INF_INT
  _limit_max_mesh_index: int = P_INF_INT
  _pb_params: Optional[Parameters] = None
  _rho: Optional[List[float]] = None  # poll size to mesh size ratio
  _dtype: Optional[DType] = None

  # COMPLETED: manage the poll size granularity for discrete variables
  # # See: Audet et. al, The mesh adaptive direct search algorithm for
  # # granular and discrete variable
  # _exp: int = 0
  # _mantissa: int = 1
  # psize_max: float = 0.0
  # psize_success: float = 0.0
  # numpy double data type precision

  def get_rho(self):
    ...

  # Update mesh size (small delta) based on frame size (big Delta)
  def updatedeltaMeshSize(self):
    ...

  def enlarge_delta_frame_size(self):
    ...

  def refine_delta_frame_size(self):
    ...

  def checkMeshForStopping(self):
    ...

  def get_delta_mesh_size(self):
    ...

  def get_delta_frame_size(self, i: int):
    ...

  def getDeltaFrameSizeCoarser(self):
    ...

  def setDeltas(
          self, i: int = None, delta_mesh_size: Any = None,
          delta_frame_size: Any = None):
    ...

  def scale_and_project_on_mesh(self, dir_in: Point = None):
    ...

  def project_on_mesh(self, point: Point, frame_center: Point):
    ...

  def verifyPointIsOnMesh(self, point: Point, frame_center: Point):
    ...

  def verifyDimension(self, name: str, dim: int):
    ...


@dataclass
class Mesh(MeshData):

  def __init__(
          self, pb_params: Parameters, limit_min_mesh_index: int,
          limit_max_mesh_index: int):
    self._n = pb_params._n
    self._initial_mesh_size = pb_params.initial_mesh_size
    self._min_mesh_size = pb_params.min_mesh_size
    self._initial_frame_size = pb_params.initial_frame_size
    self._min_frame_size = pb_params.min_frame_size
    self._lower_bound = Point(self._n, pb_params.lb)
    self._upper_bound = Point(self._n, pb_params.ub)
    self._is_finest = True
    self._r = Point(self._n).reset(n=self._n, d=0.)
    self._r_min = Point(self._n).reset(n=self._n, d=0.)
    self._r_max = Point(self._n).reset(n=self._n, d=0.)
    self._limit_min_mesh_index = limit_min_mesh_index
    self._limit_max_mesh_index = limit_max_mesh_index
    self._dtype = DType()
    self._pb_params = pb_params
    if not self._pb_params.to_be_checked():
      raise IOError(
          "Parameters::checkAndComply() needs to be called before constructing a mesh.")

  @property
  def n(self):
    return self._n

  @n.setter
  def n(self, value: int) -> int:
    self._n = value

  @property
  def rho(self):
    return self._rho

  @rho.setter
  def rho(self, rho):
    self.rho = rho

  @rho.deleter
  def rho(self):
    del self._rho

  def getSize(self):
    return self._n

  def getInitialMeshSize(self):
    return self._initial_mesh_size

  def getMinMeshSize(self):
    return self._min_mesh_size

  def getInitialFrameSize(self):
    return self._initial_frame_size

  def getMinFrameSize(self):
    return self._min_frame_size

  def getMeshIndex(self):
    return self._r

  def setMeshIndex(self, r: Point):
    self._r = r

  def isFinest(self):
    return self._is_finest

  def setLimitMeshIndices(self, limit_min_mesh_index: int,
                          limit_max_mesh_index: int):
    self._limit_max_mesh_index = limit_max_mesh_index
    self._limit_min_mesh_index = limit_min_mesh_index
