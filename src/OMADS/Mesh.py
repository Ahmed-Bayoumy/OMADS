from dataclasses import dataclass
from typing import Protocol, Any, List, Optional
from ._globals import DType, M_INF_INT, P_INF_INT
from .Point import Point
from .Parameters import Parameters

@dataclass
class MeshData(Protocol):
  """ Orthognal Mesh class

  :param _delta: mesh size
  :param _Delta: poll size
  :param _rho: poll size to mesh size ratio
  :param _exp:  manage the poll size granularity for discrete variables. See Audet et. al, The mesh adaptive direct search algorithm for granular and discrete variable
  :param _mantissa: Same as ``_exp``
  :param psize_max: Maximum poll size
  :param psize_success: Poll size at successful evaluation
  :param _dtype: numpy double data type precision

  """
  _n: Optional[int] = None
  _initialMeshSize: Optional[Point] = None  # mesh size
  _initialFrameSize: Optional[Point] = None  # poll size
  _minMeshSize: Optional[Point] = None  # mesh size
  _minFrameSize: Optional[Point] = None  # poll size
  _lowerBound: Optional[Point] = None
  _upperBound: Optional[Point] = None
  _isFinest: Optional[bool] = True
  _r: Optional[Point] = None
  _rMin: Optional[Point] = None
  _rMax: Optional[Point] = None
  _limitMinMeshIndex: int = M_INF_INT
  _limitMaxMeshIndex: int = P_INF_INT
  _pbParams: Optional[Parameters] = None
  _rho: Optional[List[float] ]= None  # poll size to mesh size ratio
  _dtype: Optional[DType] = None

  # COMPLETED: manage the poll size granularity for discrete variables
  # # See: Audet et. al, The mesh adaptive direct search algorithm for
  # # granular and discrete variable
  # _exp: int = 0
  # _mantissa: int = 1
  # psize_max: float = 0.0
  # psize_success: float = 0.0
  # numpy double data type precision

  def getRho(self):
    ...
  
  # Update mesh size (small delta) based on frame size (big Delta)
  def updatedeltaMeshSize(self):
    ...
  
  def enlargeDeltaFrameSize(self):
    ...
  
  def refineDeltaFrameSize(self):
    ...
  
  def checkMeshForStopping(self):
    ... 
  
  def getdeltaMeshSize(self):
    ...
  
  def getDeltaFrameSize(self, i: int):
    ...

  def getDeltaFrameSizeCoarser(self):
    ...
  
  def setDeltas(self, i: int = None, delta_mesh_size: Any = None, delta_frame_size: Any = None):
    ...
  
  def scaleAndProjectOnMesh(self, dir: Point = None):
    ...

  def projectOnMesh(self, point: Point, frame_center: Point):
    ...

  def verifyPointIsOnMesh(self, point: Point, frame_center: Point):
    ...

  def verifyDimension(self, name: str, dim: int):
    ...


@dataclass
class Mesh(MeshData):

  def __init__(self, pb_params: Parameters, limit_min_mesh_index: int, limit_max_mesh_index: int):
    self._n = pb_params._n
    self._initialMeshSize = pb_params.initialMeshSize
    self._minMeshSize = pb_params.minMeshSize
    self._initialFrameSize = pb_params.initialFrameSize
    self._minFrameSize = pb_params.minFrameSize
    self._lowerBound = Point(self._n, pb_params.lb)
    self._upperBound = Point(self._n, pb_params.ub)
    self._isFinest = True
    self._r = Point(self._n).reset(n=self._n, d=0.)
    self._rMin = Point(self._n).reset(n=self._n, d=0.)
    self._rMax = Point(self._n).reset(n=self._n, d=0.)
    self._limitMinMeshIndex = limit_min_mesh_index
    self._limitMaxMeshIndex = limit_max_mesh_index
    self._dtype = DType()
    self._pbParams = pb_params
    if (not self._pbParams.to_be_checked()):
      raise IOError("Parameters::checkAndComply() needs to be called before constructing a mesh.")

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
    return self._initialMeshSize
  def getMinMeshSize(self):
    return self._minMeshSize
  def getInitialFrameSize(self):
    return self._initialFrameSize
  def getMinFrameSize(self):
    return self._minFrameSize
  
  def getMeshIndex(self):
    return self._r
  
  def setMeshIndex(self, r:Point):
    self._r = r

  def isFinest(self):
    return self._isFinest
  
  def setLimitMeshIndices(self, limit_min_mesh_index: int, limit_max_mesh_index: int):
    self._limitMaxMeshIndex = limit_max_mesh_index
    self._limitMinMeshIndex = limit_min_mesh_index

    
  

  
