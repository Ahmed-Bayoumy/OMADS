import copy
from typing import Protocol, Any, List
from ._globals import *
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
  _n: int = None
  # _anisotropy_factor: int = 0.1
  # _meshSize: Point = None  # mesh size
  # _frameSize: Point = None  # poll size
  _initialMeshSize: Point = None  # mesh size
  _initialFrameSize: Point = None  # poll size
  _minMeshSize: Point = None  # mesh size
  _minFrameSize: Point = None  # poll size
  _lowerBound: Point = None
  _upperBound: Point = None
  _isFinest: bool = True
  _r: Point = None
  _rMin: Point = None
  _rMax: Point = None
  _limitMinMeshIndex: int = M_INF_INT
  _limitMaxMeshIndex: int = P_INF_INT
  _pbParams: Parameters = None
  _rho: List[float] = None  # poll size to mesh size ratio
  _dtype: DType = None

  # TODO: manage the poll size granularity for discrete variables
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

  def getDeltaFrameSizeCoarser(self, i: int):
    ...
  
  def setDeltas(self, i: int = None, deltaMeshSize: Any = None, deltaFrameSize: Any = None):
    ...
  
  def scaleAndProjectOnMesh(self, i: int = None, l: float = None, dir: Point = None):
    ...

  def projectOnMesh(self, point: Point, frameCenter: Point):
    ...

  def verifyPointIsOnMesh(self, point: Point, frameCenter: Point):
    ...

  def verifyDimension(self, name: str, dim: int):
    ...


@dataclass
class Mesh(MeshData):

  def __init__(self, pbParams: Parameters, limitMinMeshIndex: int, limitMaxMeshIndex: int):
    self._n = pbParams._n
    self._initialMeshSize = pbParams.initialMeshSize
    self._minMeshSize = pbParams.minMeshSize
    self._initialFrameSize = pbParams.initialFrameSize
    self._minFrameSize = pbParams.minFrameSize
    self._lowerBound = Point(self._n, pbParams.lb)
    self._upperBound = Point(self._n, pbParams.ub)
    self._isFinest = True
    self._r = Point(self._n).reset(n=self._n, d=0.)
    self._rMin = Point(self._n).reset(n=self._n, d=0.)
    self._rMax = Point(self._n).reset(n=self._n, d=0.)
    self._limitMinMeshIndex = limitMinMeshIndex
    self._limitMaxMeshIndex = limitMaxMeshIndex
    self._dtype = DType()
    self._pbParams = pbParams
    if (not self._pbParams.toBeChecked()):
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
  
  def setLimitMeshIndices(self, limitMinMeshIndex: int, limitMaxMeshIndex: int):
    self._limitMaxMeshIndex = limitMaxMeshIndex
    self._limitMinMeshIndex = limitMinMeshIndex

    
  

  
