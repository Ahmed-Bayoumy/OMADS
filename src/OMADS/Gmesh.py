# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - ORTHO-MADS (MADS)                                    #
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

import copy
from typing import List
from ._globals import *
from .Point import Point
from .Mesh import Mesh
from .Options import Options
from .Parameters import Parameters

@dataclass
class Gmesh(Mesh):
  """ GMesh: Granular mesh """
  _initFrameSizeExp: Point = None
  _frameSizeMant: Point = None
  _frameSizeExp: Point = None
  _finestMeshSize: Point = None
  _granularity: Point = None
  _enforceSanityChecks: bool = None
  _allGranular: bool = None
  _anisotropyFactor: float = None
  _anisotropicMesh: bool = None
  _refineFreq: int = 1
  _refineCount: int = None
  _r: Point = None
  _r_min: Point = None
  _r_max: Point = None
  _Delta_0: Point = None
  _Delta_0_mant: Point = None
  _pos_mant_0: Point = None
  _HARD_MIN_MESH_INDEX: int = -300

  def __init__(self, pbParam: Parameters, runOptions: Options):
    """ Constructor """
    super(Gmesh, self).__init__(pbParams=pbParam, limitMaxMeshIndex=-GL_LIMITS, limitMinMeshIndex=GL_LIMITS)
    
    # if (self._limit_mesh_index>0):
    #   raise IOError("Limit mesh index must be <=0 ")
    
    self._initFrameSizeExp = Point()
    self._frameSizeMant = Point()
    self._frameSizeExp = Point()
    self._finestMeshSize = Point()
    self._granularity = pbParam.granularity
    self._enforceSanityChecks = True
    self._allGranular = True
    self._anisotropyFactor = runOptions.anisotropyFactor
    self._anisotropicMesh = runOptions.anistropicMesh
    self._refineFreq = runOptions.refineFreq
    self._refineCount = 0
    self._dtype = DType(runOptions.precision) 
    self.init()

  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, other: DType):
    self.dtype = other

  def initFrameSizeGranular(self, initialFrameSize: Point):
    if not initialFrameSize.is_all_defined() or initialFrameSize.size != self._n:
      raise IOError("GMesh: initFrameSizeGranular: inconsistent dimension of the frame size. \n" +
                    f"initial frame size defined: {initialFrameSize.is_all_defined()} \n" +
                    f"size: {initialFrameSize.size} \n" +
                    f"n: {self._n}")
    
    self._frameSizeExp.reset(n=self._n)
    self._frameSizeMant.reset(n=self._n)
    dMin: float = None
    for i in range(self._n):
      if self._granularity[i] > 0:
        dMin = self._granularity[i]
      else:
        dMin = 1
      
      div: float = initialFrameSize[i] / dMin
      exp: int = self.roundFrameSizeExp(np.log10(abs(div)))
      self._frameSizeExp[i] = exp
      self._frameSizeMant[i] = self.roundFrameSizeMant(div*10**-exp)

  def roundFrameSizeExp(self, exp: float) -> int:
    frameSizeExp: int = int(exp)
    return frameSizeExp
  
  def roundFrameSizeMant(self, mant: float):
    frameSizeMant: int = 0

    if mant < 1.5:
      frameSizeMant = 1
    elif mant >= 1.5 and mant < 3.5:
      frameSizeMant = 2
    else:
      frameSizeMant = 5

    return frameSizeMant
  
  def getRho(self, i: int = None) -> auto:
    
    if i is not None:
      rho: auto
      diff: float = self._frameSizeExp[i] - self._initFrameSizeExp[i]
      powDiff: float = 10.0 ** abs(diff)

      if self._granularity[i] > 0:
        rho = self._frameSizeMant[i] * min(10.0**self._frameSizeExp[i], powDiff)
      else:
        rho = self._frameSizeMant[i] * powDiff
    else:
      rho: auto = [None] * self._n
      for i in range(self._n):
        diff: float = self._frameSizeExp[i] - self._initFrameSizeExp[i]
        powDiff: float = 10.0 ** abs(diff)

        if self._granularity[i] > 0:
          rho[i] = self._frameSizeMant[i] * min(10.0**self._frameSizeExp[i], powDiff)
        else:
          rho[i] = self._frameSizeMant[i] * powDiff

    return rho
  
  def updatedeltaMeshSize(self):
    return

  def getdeltaMeshSize(self, i: int = None) -> Point:
    delta: Point = Point(self._n)
    delta.coordinates = [0] * self._n
    if i is None:
      for i in range(self._n):
        diff: float = self._frameSizeExp[i] - self._initFrameSizeExp[i]
        exp: float = self._frameSizeExp[i] - abs(diff)
        delta[i] = 10.0 ** exp

        if 0.0 < self._granularity[i]:
          delta[i] = self._granularity[i] * max(1.0, delta[i])
      
      return delta
    else:
      diff: float = self._frameSizeExp[i] - self._initFrameSizeExp[i]
      exp: float = self._frameSizeExp[i] - abs(diff)
      delta[i] = 10.0 ** exp

      if 0.0 < self._granularity[i]:
        delta[i] = self._granularity[i] * max(1.0, delta[i])
      return delta[i]
  
  def getDeltaFrameSize(self, i: int = None) -> Point:
    dMinGran = 1.0
    Delta: Point = Point(self._n)
    Delta.coordinates = [0] * self._n
    if i is None:
      for i in range(self._n):
        if self._granularity[i] > 0:
          dMinGran = self._granularity[i]
        Delta[i] = dMinGran * self._frameSizeMant[i] * 10 ** self._frameSizeExp[i]
      return Delta
    else:
      if self._granularity[i] > 0:
        dMinGran = self._granularity[i]
      Delta[i] = dMinGran * self._frameSizeMant[i] * 10 ** self._frameSizeExp[i]
      return Delta[i]

  def getDeltaFrameSizeCoarser(self) -> Point:
    Delta: Point = Point(self._n)
    Delta.coordinates = [0] * self._n
    for i in range(self._n):
      frameSizeMantOld = self._frameSizeMant[i]
      frameSizeExpOld = self._frameSizeExp[i]
      self._frameSizeMant[i], self._frameSizeExp[i] = self.getLargerMantExp(frameSizeMant=frameSizeMantOld, frameSizeExp=frameSizeExpOld, i=i)
      Delta[i] = self.getDeltaFrameSize(i=i)
      self._frameSizeMant[i] = frameSizeMantOld
      self._frameSizeExp[i] = frameSizeExpOld
    
    return Delta

  def getLargerMantExp(self, frameSizeMant: float, frameSizeExp: float, i: int):
    if frameSizeMant == 1:
      self._frameSizeMant[i] = 2
    elif frameSizeMant == 2:
      self._frameSizeMant[i] = 5
    else:
      self._frameSizeMant[i] = 1
      self._frameSizeExp[i] += 1
    return self._frameSizeMant[i], self._frameSizeExp[i]
  
  def checkDeltasGranularity(self, i: int, deltaMeshSize: float, deltaFrameSize: float):
    if self._granularity[i] > 0.0:
      hasError: bool = False
      err: str = "Error: setDeltas: "
      if not self.isMult(deltaMeshSize, self._granularity[i]):
        hasError = True
        err += f"deltaMeshSize at index {i}"
        err += f" is not a multiple of granularity {self._granularity[i]}"
      elif not self.isMult(deltaFrameSize, self._granularity[i]):
        hasError = True
        err += f"deltaFrameSize at index {i}"
        err += f" is not a multiple of granularity {self._granularity[i]}"
      if hasError:
        raise IOError(err)
  
  def setDeltas(self, i: int, deltaMeshSize: float, deltaFrameSize: float):
    # Input checks
    self.checkDeltasGranularity(i=i, deltaMeshSize=deltaMeshSize, deltaFrameSize=deltaFrameSize)
    # Value to use for granularity (division so default = 1.0)
    gran: float = 1.
    if 0. < self._granularity[i]:
      gran = self._granularity[i]
    
    mant: float
    exp: float
    # Compute mantisse first
    # There are only 3 cases: 1, 2, 5, so compute all
    # 3 possibilities and then assign the values that work.
    mant1: float = deltaFrameSize / (1.*gran)
    mant2: float = deltaFrameSize / (2.*gran)
    mant5: float = deltaFrameSize / (5. *gran)

    exp1: float = np.log10(mant1)
    exp2: float = np.log10(mant2)
    exp5: float = np.log10(mant5)

    # // deltaFrameSize = gran * mant * 10^exp  (where gran is 1.0 if granularity is not defined)
    # // => exp = log10(deltaFrameSize / (mant * gran))
    # // exp must be an integer so verify which one of the 3 values exp1, exp2, exp5
    # // is an integer and use that value for exp, and the corresponding value
    # // 1, 2 or 5 for mant.
    if exp1.is_integer():
      mant = 1
      exp = exp1
    elif exp2.is_integer():
      mant = 2
      exp = exp2
    else:
      mant = 5
      exp = exp5
    
    self._frameSizeExp[i] = self.roundFrameSizeExp(exp=exp)
    self._frameSizeMant[i] = mant

    # Sanity checks
    if self._enforceSanityChecks:
      self.checkFrameSizeIntegrity(frameSizeExp=self._frameSizeExp[i], 
                                   frameSizeMant=self._frameSizeMant[i])
      self.checkSetDeltas(i=i, deltaMeshSize=deltaMeshSize, deltaFrameSize=deltaFrameSize)
      self.checkDeltasGranularity(i, self.getdeltaMeshSize(i=i), self.getDeltaFrameSize(i=i))






  
  def checkFrameSizeIntegrity(self, frameSizeExp: float, frameSizeMant: float):
    # frameSizeExp must be an integer.
    # frameSizeMant must be 1, 2 or 5.
    hasError: bool = False
    err: str = "Error: Integrity check"
    if not isinstance(frameSizeExp, int):
      hasError = True
      err += f" of frameSizeExp ({frameSizeExp}): Should be integer."
    elif (frameSizeMant != 1.0 and frameSizeMant != 2.0 and frameSizeMant != 5.0):
      hasError = True
      err += f" of frameSizeMant ({frameSizeMant}): Should be integer."
    
    if hasError:
      raise IOError(err)

  def checkSetDeltas(self, i: int, deltaMeshSize: float, deltaFrameSize: float):
    hasError: bool = False
    err: str = "Warning: setDeltas did not give good value"

    # Something might be wrong with setDeltas(), so double check.
    if self.getdeltaMeshSize(i=i) != deltaMeshSize:
      hasError = True
      err += f" for deltaMeshSize at index {i}"
      err += f" Expected: {deltaMeshSize}"
      err += f" computed: {self.getdeltaMeshSize(i=i)}"
    elif self.getDeltaFrameSize(i=i) != deltaFrameSize:
      hasError = True
      err += f" for deltaFrameSize at index {i}"
      err += f" Expected: {deltaFrameSize}"
      err += f" computed: {self.getDeltaFrameSize(i=i)}"
    
    if (hasError):
      raise IOError(err)
    
  
  def scaleAndProjectOnMesh(self, dir: Point):
    proj: Point = Point(self._n)
    infiniteNorm: float = np.linalg.norm(dir.coordinates, np.inf)

    if 0 == infiniteNorm:
      err = "GMesh: scaleAndProjectOnMesh: Cannot handle an infinite norm of zero"
      raise IOError(err)
    
   

    if self._frameSizeMant.is_all_defined() and self._frameSizeExp.is_all_defined():
      for i in range(self._n):
        delta: float = self.getdeltaMeshSize(i=i)
        proj[i] = np.round(self.getRho(i=i)*dir[i]/infiniteNorm) * delta
    else:
      err = "GMesh: scaleAndProjectOnMesh cannot be performed."
      err += f" i = {i}"
      err += f" mantissa defined: {self._frameSizeMant.is_all_defined()}"
      err += f" exp defined: {self._frameSizeExp.is_all_defined()}"
      err += f"delta mesh size defined: {delta}"
      raise IOError(err)
    
    return proj

  def projectOnMesh(self, point: Point, frameCenter: Point):
    proj: Point = point
    delta: auto = self.getdeltaMeshSize()
    maxNbTry: int = 10
    verifValueI: Point = Point(self._n)
    verifValueI.coordinates = [0] * self._n
    for i in range(point.size):
      deltaI = delta[i]
      frameCenterIsOnMesh: bool = self.isMult(frameCenter[i], deltaI)

      diffProjFrameCenter: float = proj[i] - frameCenter[i]
      verifValueI[i] = proj[i] if (frameCenterIsOnMesh) else diffProjFrameCenter
      # // Force verifValueI to be a multiple of deltaI.
      # // nbTry = 0 means point is already on mesh.
      # // nbTry = 1 means the projection worked.
      # // nbTry > 1 means the process went hacky by forcing the value to work
      # // for verifyPointIsOnMesh.
      nbTry = 0
      while (not self.isMult(verifValueI[i], deltaI) and nbTry <= maxNbTry):
        newVerifValueI: float
        if (0==nbTry):
          # Use closest projection
          vHigh = verifValueI.nextMult(deltaI, i)
          p: Point = Point(self._n)
          p.coordinates = [-c for c in verifValueI.coordinates]
          vLow = - (p.nextMult(deltaI, i))
          diffHigh = vHigh - verifValueI[i]
          diffLow = verifValueI[i] - vLow
          verifValueI[i] = vLow if (diffLow < diffHigh) else (vHigh if (diffHigh < diffLow) else (vLow if (proj[i] < 0) else vHigh))
        else:
          p: Point = Point(self._n)
          p.coordinates = [-c for c in verifValueI.coordinates]
          verifValueI[i] =  verifValueI.nextMult(deltaI, i) if (diffProjFrameCenter >= 0) else (-(p.nextMult(deltaI, i)))
        proj[i] = verifValueI[i] if frameCenterIsOnMesh else verifValueI[i] + frameCenter[i]

        # Recompute verifValue for more precision
        newVerifValueI = proj[i] if frameCenterIsOnMesh else proj[i] - frameCenter[i]
        nbTry += 1

        #  Special cases
        while (newVerifValueI != verifValueI[i] and nbTry <= maxNbTry):
          if verifValueI[i] >= 0:
            verifValueI[i] = max(verifValueI[i], newVerifValueI)
            verifValueI[i] += self.dtype.zero
            verifValueI[i] = verifValueI.nextMult(deltaI, i)
          else:
            verifValueI[i] = min(verifValueI[i], newVerifValueI)
            verifValueI[i] -= self.dtype.zero
            p: Point = Point(self._n)
            p.coordinates = [-c for c in verifValueI.coordinates]
            verifValueI[i] = -(p.nextMult(deltaI, i))
          proj[i] = verifValueI[i] if frameCenterIsOnMesh else verifValueI[i] + frameCenter[i]
          # Recompute verifValue for more precision
          newVerifValueI = proj[i] if frameCenterIsOnMesh else proj[i] - frameCenter[i]
          nbTry += 1
        
        verifValueI[i] = newVerifValueI
      
      if (nbTry >= maxNbTry and not self.isMult(verifValueI[i], deltaI)):
        # TODO: print warning
        proj[i] = point[i]

    return proj


  def check_min_poll_size_criterion (self) -> bool:
    """ Check the minimal poll size criterion. """
    if not self._Delta_min_is_defined:
      return False
    S, D = self.get_Delta_object()
    return S
        
  def check_min_mesh_size_criterion (self) -> bool:
    """ Check the minimal mesh size criterion. """
    if not self._delta_min.is_all_defined():
      return False
    S, D = self.get_delta_object()
    return S
  
  def get_rho (self, i: int):
    """
    Access to the ratio of poll size / mesh size parameter rho^k.
    :param  rho The ratio poll/mesh size rho^k --  OUT.
    """
    rho: float = None
    if self._granularity[i] > 0:
      rho = self._frameSizeMant.coordinates[i] * min(10** self._frameSizeExp.coordinates[i], 10**abs(self._frameSizeExp.coordinates[i]-self._initFrameSizeExp.coordinates[i]))
    else:
      rho = self._frameSizeMant.coordinates[i] * 10** abs(self._frameSizeExp.coordinates[i]-self._initFrameSizeExp.coordinates[i])
    return rho


  def get_delta (self, i: int): 
    """
    Access to the mesh size parameter delta^k.
    :param  delta: The mesh size parameter delta^k --  OUT.
    :param  i: The index of the mesh size
  """
    delta: float = 10**(self._frameSizeExp.coordinates[i]-abs(self._frameSizeExp.coordinates[i]-self._initFrameSizeExp.coordinates[i]))
    if self._granularity.coordinates[i]:
      delta = self._granularity[i] * max(1.0, delta)
    return delta
    
  def  get_Delta (self, i: int): 
    """
      Access to the poll size parameter Delta^k.
      :param  Delta: The poll size parameter Delta^k --  OUT.
      :param  i: The index of the poll size
    """
    if self._granularity.coordinates[i]:
      d_min_gran = self._granularity[i]
    Delta: float = d_min_gran * self._frameSizeMant.coordinates[i] * 10**(self._frameSizeExp.coordinates[i])
    
    return Delta
  
  def init(self):
    """Initialization of granular poll size mantissa and exponent"""
    self._r = Point(self._n)
    self._r.coordinates = [0]*self._n
    self._rMax = Point(self._n)
    self._rMax.coordinates = [0]*self._n
    self._rMin = Point(self._n)
    self._rMin.coordinates = [0]*self._n
    self.initFrameSizeGranular(self._initialFrameSize)
    self._initFrameSizeExp.reset(self._n)
    self._finestMeshSize = self.getdeltaMeshSize()

    for i in range(self._n):
      if 0.0 == self._granularity[i]:
        self._allGranular = False
        break
    
    if not self._minMeshSize.is_complete():
      raise IOError("Expecting mesh minimum size to be fully defined.")
    
    if self._enforceSanityChecks:
      for i in range(self._n):
        self.checkFrameSizeIntegrity(frameSizeExp=self._frameSizeExp[i], frameSizeMant=self._frameSizeMant[i])
        self.checkDeltasGranularity(i=i, deltaMeshSize=self.getdeltaMeshSize(i=i), deltaFrameSize=self.getDeltaFrameSize(i=i))
  
  def isMult(self, v1, v2)->bool:
    return ((v1%v2) <= self.dtype.zero)

  def enlargeDeltaFrameSize(self, direction: Point) -> bool:
    oneFrameSizeChanged = False
    minRho = np.inf
    for i in range(self._n):
      if self._granularity[i] == 0:
        minRho = min(minRho, self.getRho(i=i))
    
    for i in range(self._n):
      frameSizeIChanged = False
      if (not self._anisotropicMesh or abs(direction[i])/self.getdeltaMeshSize(i=i)/self.getRho(i=i) > self._anisotropyFactor or (self._granularity[i] == 0 and self._frameSizeExp[i] < self._initFrameSizeExp[i] and self.getRho(i=i) > minRho*minRho)):
        self.getLargerMantExp(frameSizeMant=self._frameSizeMant[i], frameSizeExp=self._frameSizeExp[i], i=i)
        frameSizeIChanged = True
        oneFrameSizeChanged = True
        # update the mesh index
        self._r[i] += 1
        self._rMax[i] = max(self._r[i], self._rMax[i])

        # Sanity checks
        if self._enforceSanityChecks and frameSizeIChanged:
          self.checkFrameSizeIntegrity(self._frameSizeExp[i], self._frameSizeMant[i])
          self.checkDeltasGranularity(i=i, deltaMeshSize=self.getdeltaMeshSize(i=i), deltaFrameSize=self.getDeltaFrameSize(i=i))
        
    # When we enlarge the frame size we may keep the mesh size unchanged. So we need to test.
    msize = self.getdeltaMeshSize()
    if self._finestMeshSize < msize:
      self._isFinest = False
    
    return oneFrameSizeChanged

  def refineDeltaFrameSizeME(self, frameSizeMant: float, frameSizeExp:float, granularity: float):
    if frameSizeMant == 1:
      frameSizeMant = 5
      frameSizeExp -= 1
    elif frameSizeMant == 2:
      frameSizeMant = 1
    else:
      frameSizeMant = 2
    
    # When the mesh reaches granularity (exp = 1, mant = 1), make sure to remove the refinement
    if granularity > 0 and frameSizeExp < 0 and frameSizeMant == 5:
      frameSizeExp = 0
      frameSizeMant = 1
    
    return frameSizeMant, frameSizeExp
  
  def getdeltaMeshSizeF(self, frameSizeExp:int, initFrameSizeExp:int, granularity: int)->float:
    diff = frameSizeExp - initFrameSizeExp
    exp = frameSizeExp - abs(diff)
    delta = 10.0**exp
    if 0.0 < granularity:
      delta = granularity * max(1.0, delta)
    
    return delta
    

  def refineDeltaFrameSize(self) -> bool:
    # // Compute the new values frameSizeMant and frameSizeExp first.
    # // We will do some verifications before setting them.
    self._refineCount += 1
    if self._refineCount%self._refineFreq != 0:
      return
    
    for i in range(self._n):
      # // Compute the new values frameSizeMant and frameSizeExp first.
      # // We will do some verifications before setting them.
      frameSizeMant = self._frameSizeMant[i]
      frameSizeExp = self._frameSizeExp[i]
      frameSizeMant, frameSizeExp= self.refineDeltaFrameSizeME(frameSizeMant=frameSizeMant, frameSizeExp=frameSizeExp, granularity=self._granularity[i])
      # Verify delta mesh size does not go too small if we use the new values.
      olddeltaMeshSize = self.getdeltaMeshSizeF(frameSizeExp=self._frameSizeExp[i], initFrameSizeExp=self._initFrameSizeExp[i], granularity=self._granularity[i])
      if self._minMeshSize[i] <= olddeltaMeshSize:
        # update mesh index
        if self._granularity[i] == 0:
          self._r[i] -= 1
        else:
          # Update mesh index if not already at the min limit. When refining the frame, if mantissa and exponent stay the same, the min limit is reached (do not decrease).
          if (not (self._frameSizeMant[i] == frameSizeMant and self._frameSizeExp[i] == frameSizeExp)):
            self._r[i] -= 1
        # Update the minimal mesh index reached so far
        self._rMin[i] = min(self._r[i], self._rMin[i])

        # We can go lower
        self._frameSizeMant[i] = frameSizeMant
        self._frameSizeExp[i] = frameSizeExp

      # Sanity checks
      if self._enforceSanityChecks:
        self.checkFrameSizeIntegrity(frameSizeExp=self._frameSizeExp[i], 
                                     frameSizeMant=self._frameSizeMant[i])
        self.checkDeltasGranularity(i=i, deltaMeshSize=self.getdeltaMeshSize(i=i), deltaFrameSize=self.getDeltaFrameSize(i=i))
    msize = self.getdeltaMeshSize()
    if msize <= self._finestMeshSize:
      self._isFinest = True
      self._finestMeshSize = msize
    else:
      self._isFinest = False
    
  def update(self):
    return


# ###############################################
# ###############################################
# ###############################################
  
  # def init_poll_size_granular (self, cont_init_poll_size: Point ):
  #   """
  #   :param: cont_init_poll_size: continuous initial poll size   --  IN.
  #   """

  #   if not all(cont_init_poll_size.defined) or cont_init_poll_size.n_dimensions != self._n:
  #     raise IOError("Inconsistent dimension of the poll size!")
    
  #   self._frameSizeExp.reset(n=self._n)
  #   self._frameSizeMant.reset(n=self._n)
  #   self._pos_mant_0.reset(n=self._n)

  #   d_min: float

  #   for i in range(self._n):
  #     if self._granularity.defined[i] and self._granularity.coordinates[i] > 0:
  #       d_min = self._granularity[i]
  #     else:
  #       d_min=1.0
      
  #     exp: int = int(np.log10(abs(cont_init_poll_size.coordinates[i]/d_min)))
  #     if exp < 0:
  #       exp = 0

  #     self._frameSizeExp.coordinates[i]=exp
  #     cont_mant: float = cont_init_poll_size.coordinates[i] / d_min * 10.0**(-exp)

  #     if cont_mant < 1.5:
  #       self._frameSizeMant.coordinates[i] = 1
  #       self._pos_mant_0[i] = 0
  #     elif (cont_mant >= 1.5 and  cont_mant < 3.5):
  #       self._frameSizeMant.coordinates[i] = 2
  #       self._pos_mant_0.coordinates[i] = 1
  #     else:
  #       self._frameSizeMant.coordinates[i] = 5
  #       self._pos_mant_0.coordinates[i] = 2

      
  
  # def get_delta_object(self):
  #   """  """
  #   stop = True
  #   delta: Point = Point(self._n)
  #   for i in range(self._n):
  #     delta.coordinates[i] = self.get_delta(i=i)
  #     if stop and self._delta_min_is_defined and not self._fixed_variables.defined[i] and self._delta_min.defined[i] and delta.coordinates[i] >= self._delta_min[i]:
  #       stop = False
  #   return stop, delta
  
  # def get_delta_max(self)->Point:
  #   return self._delta_0
  
  # def get_Delta_object(self)->Point:
  #   """ """
  #   stop = True
  #   Delta: Point = Point(self._n)
  #   for i in range(self._n):
  #     Delta.coordinates[i] = self.get_Delta(i=i)
  #     if stop and self._granularity.coordinates[i] == 0 and not self._fixed_variables.defined[i] and (self._Delta_min_is_complete or Delta.coordinates[i] >= self._Delta_min[i]):
  #       stop = False
    
  #     if stop and self._granularity.coordinates[i] > 0 and not self._fixed_variables.defined[i] and (not self._Delta_min_is_complete or Delta.coordinates[i] > self._Delta_min[i]):
  #       stop = False


  #   return stop, Delta
  
  # def is_finer_than_initial(self):
  #   """ """
  #   for i in range(self._n):
  #     if not self._fixed_variables.defined[i]:
  #       # For continuous variables
  #       if self._granularity.coordinates[i]==0 and (self._frameSizeExp.coordinates[i] > self._initFrameSizeExp.coordinates[i] or ( self._frameSizeExp.coordinates[i] == self._initFrameSizeExp.coordinates[i] and self._frameSizeMant.coordinates[i] >= self._Delta_0_mant.coordinates[i] )):
  #         return False
  #       # For granular variables (case 1)
  #       if self._granularity.coordinates[i] > 0 and (self._frameSizeExp.coordinates[i] > self._initFrameSizeExp.coordinates[i] or ( self._frameSizeExp.coordinates[i] == self._initFrameSizeExp.coordinates[i] and self._frameSizeMant.coordinates[i] > self._Delta_0_mant.coordinates[i] )):
  #         return False
  #       # For continuous variables (case 2)
  #       if self._granularity.coordinates[i]>0 and (self._frameSizeExp.coordinates[i] == self._initFrameSizeExp.coordinates[i] and  self._frameSizeMant.coordinates[i] == self._Delta_0_mant.coordinates[i] and (self._frameSizeExp.coordinates[i] != 0 or self._frameSizeMant.coordinates[i] != 1) ):
  #         return False
    
  #   return True
  
  # def update(self, success: SUCCESS_TYPES, d: List[float]):
  #   if d and self._n != len(d):
  #     raise IOError("delta_0 and d have different sizes")
    
  #   if success == SUCCESS_TYPES.FS:
  #     for i in range(self._n):
  #       if (self._granularity.coordinates[i] == 0 and not self._fixed_variables.defined[i]):
  #         if i > 0:
  #           min_rho = min(min_rho, self.get_rho(i))
  #         else:
  #           min_rho = self.get_rho(i)
          
  #     for i in range(self._n):
  #       if (not d or not self._anisotropic_mesh or abs(d[i])/self.get_delta(i)/self.get_rho(i) > self._anisotropic_factor or ( self._granularity.coordinates[i] == 0  and self._frameSizeExp.coordinates[i] < self._initFrameSizeExp.coordinates[i] and self.get_rho(i) > min_rho*min_rho )):
  #         # Update the mesh index
  #         self._r.coordinates[i] += 1
  #         self._r_max.coordinates[i] = max(self._r.coordinates[i], self._r_max.coordinates[i])
  #         # update the mantissa and exponent
  #         if ( self._frameSizeMant.coordinates[i] == 1 ):
  #             self._frameSizeMant.coordinates[i]= 2
  #         elif ( self._frameSizeMant.coordinates[i] == 2 ):
  #             self._frameSizeMant.coordinates[i]=5
  #         else:
  #           self._frameSizeMant.coordinates[i]=1
  #           self._frameSizeExp.coordinates[i] += 1
  #   elif success == SUCCESS_TYPES.US:
  #     for i in range(self._n):
  #       if (not self._fixed_variables.defined[i]):
  #         # update the mesh index
  #         self._r.coordinates[i] -= 1
  #         # update the mesh mantissa and exponent
  #         if (self._frameSizeMant.coordinates[i]==1):
  #           self._frameSizeMant.coordinates[i] = 5
  #           self._frameSizeExp.coordinates[i] -= 1
  #         elif self._frameSizeMant.coordinates[i] == 2:
  #           self._frameSizeMant.coordinates[i] = 1
  #         else:
  #           self._frameSizeMant.coordinates[i] = 2
          
  #         if ( self._granularity.coordinates[i] > 0 and self._frameSizeExp.coordinates[i]==-1 and self._frameSizeMant.coordinates[i]==5 ):
  #           self._r.coordinates[i] += 1
  #           self._frameSizeExp.coordinates[i]=0
  #           self._frameSizeMant.coordinates[i]=1
  #       self._r_min.coordinates[i] = min(self._r.coordinates[i], self._r_min.coordinates[i])

  #     # for i in range(self._n):
  #     #   # Test for producing anisotropic mesh + correction to prevent mesh collapsing for some variables ( ifnot )
  #     #   if (not d or not self._anisotropic_mesh or d[i]/self.get_delta(i)):

  
  # def reset(self):
  #   """ """
  #   self.__init__()
  
  # def is_finest(self):
  #   """ """
  #   for i in range(self._n):
  #     if not self._fixed_variables.defined[i] and self._r.coordinates[i] > self._r_min.coordinates[i]:
  #       return False
  #   return True
  

  
  # def scale_and_project(self, i: int, l: float, round_up: bool):
  #   """ """
  #   delta: float = self.get_delta(i=i)
  #   if i<= self._n and self._frameSizeMant.is_all_defined() and self._frameSizeExp.is_all_defined() and delta is not None:
  #     d: float = self.get_rho(i=i) * l
  #     # round to double
  #     return np.round(d)*delta
  #   else:
  #     raise IOError("scale_and_project(): mesh scaling and projection cannot be performed!")



  
  # def check_min_mesh_sizes(self, stop: bool=None, stop_reason: STOP_TYPE = None):
  #   """_summary_
  #   """
  #   if stop:
  #     return
    
  #   stop = False
  #   # Coarse mesh stopping criterion
  #   for i in range(self._n):
  #     if self._r.coordinates[i] > -GL_LIMITS:
  #       stop = True
  #       break
  #   if stop:
  #     stop_reason = STOP_TYPE.GL_LIMITS_REACHED
  #     return
    
  #   stop = True

  #   # // Fine mesh stopping criterion. Do not apply when all variables have granularity.
  #   # // To trigger this stopping criterion:
  #   # //  - All mesh indices must be < _limit_mesh_index for all continuous variables (granularity==0), and
  #   # //  - mesh size == granularity for all granular variables.
  #   if self._all_granular:
  #     stop = False
    
  #   else:
  #     for i in range(self._n):
  #       # Skip fixed variables
  #       if self._fixed_variables.defined[i]:
  #         continue
  #       # Do not stop if the mesh size of a variable is strictly larger than its granularity
  #       if self._granularity.coordinates[i] > 0 and self.get_delta(i=i) > self._granularity.coordinates[i]:
  #         stop = False
  #         break
  #       # Do not stop if the mesh of a variable is above the limit mesh index
  #       if self._granularity.coordinates[i] == 0 and self._r.coordinates[i] >= self._granularity.coordinates[i]:
  #         stop = False
  #         break
    
  #   if stop:
  #     stop_reason = STOP_TYPE.GL_LIMITS_REACHED
  #     return
    
  #   # 2. delta^k (mesh size) tests:
  #   if self.check_min_poll_size_criterion():
  #     stop = True
  #     stop_reason = STOP_TYPE.DELTA_P_MIN_REACHED
  #     return

  #   # 3. delta^k (mesh size) tests:
  #   if self.check_min_mesh_size_criterion():
  #     stop = True
  #     stop_reason = STOP_TYPE.DELTA_M_MIN_REACHED
  #     return

    

  
  # def get_mesh_indices(self):
  #   """_summary_
  #   """
  #   return self._r

  
  # def get_min_mesh_indices(self):
  #   """_summary_
  #   """
  #   return self._r_min
  
  # def get_max_mesh_indices(self):
  #   """_summary_
  #   """
  #   return self._r_max
  
  # def set_mesh_indices(self, r: Point):
  #   """_summary_
  #   """
  #   if r.size != self._n:
  #     raise IOError("set_mesh_indices(): dimension of provided mesh indices must be consistent with their previous dimension")
    
  #   if r.coordinates[0] < HARD_MIN_MESH_INDEX:
  #     raise IOError("set_mesh_indices(): mesh index is too small")
    
  #   # Set the mesh indices
  #   self._r = copy.deepcopy(r)
  #   for i in range(self._n):
  #     if (r.coordinates[i]>self._r_max.coordinates[i]):
  #       self._r_max.coordinates[i] = r.coordinates[i]
  #     if (r.coordinates[i] < self._r_min.coordinates[i]):
  #       self._r_min.coordinates[i] = r.coordinates[i]
    
  #   # Set the mesh mantissas and exponents according to the mesh indices
  #   for i in range(self._n):
  #     shift: int = int(self._r.coordinates[i] + self._pos_mant_0.coordinates[i])
  #     pos: int = self.isMult((shift + 300), 3)

  #     self._frameSizeExp.coordinates[i] = np.floor((shift+300.0)/3.0) - 100.0 + self._initFrameSizeExp.coordinates[i]

  #     if pos == 0:
  #       self._frameSizeMant.coordinates[i] = 1
  #     elif pos == 1:
  #       self._frameSizeMant.coordinates[i] = 2
  #     elif pos == 2:
  #       self._frameSizeMant.coordinates[i] = 5
  #     else:
  #       raise IOError("set_mesh_indices(): something is wrong with conversion from index to mantissa and exponent")
  
  # def set_limit_mesh_index(self, l: int):
  #   """_summary_
  #   """
  #   if l > 0:
  #     raise IOError("set_limit_mesh_index(): the limit mesh index must be negative or null.")
    
  #   if l > HARD_MIN_MESH_INDEX:
  #     raise IOError("set_limit_mesh_index(): the limit mesh index is too small.")
    
  #   self._limit_mesh_index = l
  

  
  # def get_mesh_ratio_if_success(self):
  #   """_summary_
  #   """
  #   ratio: Point = Point(self._n)
  #   for i in range(self._n):
  #     power_of_tau: float = self._update_basis**(0 if self._r.coordinates[i] >= 0 else 2*self._r.coordinates[i])

  #     power_of_tau_if_success: float = self._update_basis**(0 if self._r.coordinates[i]+self._coarsening_step >= 0 else 2*(self._r.coordinates[i]+self._coarsening_step)) 

  #     ratio.coordinates[i] = power_of_tau_if_success/power_of_tau
    
  #   return ratio

  





  

