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
from .._include import DType
from .._points._point import Point
from .._setup._parameters import Parameters
from abc import ABC, abstractmethod


class Mesh(ABC):

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

  @abstractmethod
  def get_rho(self):
    pass

  # Update mesh size (small delta) based on frame size (big Delta)
  @abstractmethod
  def updatedeltaMeshSize(self):
    pass

  @abstractmethod
  def enlarge_delta_frame_size(self, direction: Point = None):
    pass

  @abstractmethod
  def refine_delta_frame_size(self):
    pass

  @abstractmethod
  def checkMeshForStopping(self):
    pass

  @abstractmethod
  def get_delta_mesh_size(self):
    pass

  @abstractmethod
  def get_delta_frame_size(self, i: int):
    pass

  @abstractmethod
  def getDeltaFrameSizeCoarser(self):
    pass

  @abstractmethod
  def setDeltas(
          self, i: int = None, delta_mesh_size: Any = None,
          delta_frame_size: Any = None):
    pass

  @abstractmethod
  def scale_and_project_on_mesh(self, dir_in: Point = None):
    pass

  @abstractmethod
  def project_on_mesh(self, point: Point, frame_center: Point):
    pass

  @abstractmethod
  def verifyPointIsOnMesh(self, point: Point, frame_center: Point):
    pass

  @abstractmethod
  def verifyDimension(self, name: str, dim: int):
    pass

  @abstractmethod
  def get_frame_size_parameter(self):
    pass
