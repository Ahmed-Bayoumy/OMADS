import copy
from typing import List
from ._globals import *
from .Point import Point
from .Mesh import Mesh
from .Options import Options
from .Parameters import Parameters

@dataclass
class Omesh(Mesh):
  """ Mesh coarsness update class

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
  _meshSize: Point = None #1.0  # mesh size
  _frameSize: Point = 1.0  # poll size
  _rho: List[float] = 1.0  # poll size to mesh size ratio
  # Completed: manage the poll size granularity for discrete variables
  # A new class 'Gmesh' is now avialable. 
  # Gmesh adapts mesh granularity and anistropy
  # See: Audet et. al, The mesh adaptive direct search algorithm for
  # granular and discrete variable
  _exp: Point = None
  _mantissa: Point = None
  _maximumFrameSize: Point = None
  successfulFrameSize: Point = None

  # numpy double data type precision
  _dtype: DType = None

  def __init__(self, pbParam: Parameters, runOptions: Options):
    """ Constructor """
    super(Omesh, self).__init__(pbParams=pbParam, limitMaxMeshIndex=-GL_LIMITS, limitMinMeshIndex=GL_LIMITS)
    self._n = len(pbParam.baseline)
    self.meshSize = Point(self._n)
    self.frameSize = Point(self._n)
    self._exp = Point(self._n)
    self._mantissa = Point(self._n)
    self._maximumFrameSize = Point(self._n)
    self.successfulFrameSize = Point(self._n)
    self.rho = [0] * self._n
    self.frameSize.coordinates = runOptions.psize_init if isinstance(runOptions.psize_init, list) else [runOptions.psize_init] * self._n
    self.meshSize.reset(n=self._n, d=0)
    self._r = Point(self._n)
    self._r.coordinates = [1]*self._n
    self._rMax = Point(self._n)
    self._rMax.coordinates = [1]*self._n
    self._rMin = Point(self._n)
    self._rMin.coordinates = [1]*self._n
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
  def meshSize(self):
    return self._meshSize

  @meshSize.setter
  def meshSize(self, size):
    self._meshSize = size

  @meshSize.deleter
  def meshSize(self):
    del self._meshSize

  @property
  def frameSize(self):
    return self._frameSize

  @frameSize.setter
  def frameSize(self, size):
    self._frameSize = size

  @frameSize.deleter
  def frameSize(self):
    del self._frameSize

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
      self.meshSize[i] = np.minimum(np.power(self._frameSize[i], 2.0, dtype=self.dtype.dtype),
                self._frameSize[i], dtype=self.dtype.dtype)
      self.rho[i] = np.divide(self._frameSize[i], self._meshSize[i], dtype=self.dtype.dtype)


  def getdeltaMeshSize(self, i: int = None):
    if i is not None:
      return self.meshSize[i]
    else:
      return self.meshSize
  
  def getDeltaFrameSize(self, i: int = None):
    if i is not None:
      return self.frameSize[i]
    else:
      return self.frameSize
  
  def getRho(self):
    return self.rho
  
  def enlargeDeltaFrameSize(self, direction: Point = None) -> bool:
    for i in range(self._n):
      self.frameSize[i] *= 2
  
  def refineDeltaFrameSize(self) -> bool:
    for i in range(self._n):
      self.frameSize[i] /= 2
  
  def projectOnMesh(self, x: Point, ref: Point=None) -> Point:
    if ref is None:
      ref = [0.]*self._n
    if self._pbParams.var_type is None:
      self._pbParams.var_type = [VAR_TYPE.REAL.name] * self._n
    for i in range(self._n):
      if self._pbParams.var_type[i] != VAR_TYPE.CATEGORICAL.name:
        if self._pbParams.var_type[i] == VAR_TYPE.REAL.name:
          x[i] = ref[i] + (np.round((x[i]-ref[i])/self.meshSize[i]) * self.meshSize[i])
        else:
          x[i] = int(ref[i] + int(int((x[i]-ref[i])/self.meshSize[i]) * self.meshSize[i]))
      else:
        x[i] = int(x[i])
      if x[i] < self._pbParams.lb[i]:
        x[i] = self._pbParams.lb[i] + (self._pbParams.lb[i] - x[i])
        if x[i] > self._pbParams.ub[i]:
          x[i] = self._pbParams.ub[i]
      if x[i] > self._pbParams.ub[i]:
        x[i] = self._pbParams.ub[i] - (x[i] - self._pbParams.ub[i])
        if x[i] < self._pbParams.lb[i]:
          x[i] = self._pbParams.lb[i]

    return x