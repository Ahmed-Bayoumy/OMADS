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
from dataclasses import dataclass

import numpy as np
from ._globals import DType, GL_LIMITS
from .Point import Point
from .Mesh import Mesh
from .Options import Options
from .Parameters import Parameters
from typing import Any, Optional

@dataclass
class Gmesh(Mesh):
  """ GMesh: Granular mesh """
  _initFrameSizeExp: Optional[Point] = None
  _frameSizeMant: Optional[Point] = None
  _frameSizeExp: Optional[Point] = None
  _finestMeshSize: Optional[Point] = None
  _granularity: Optional[Point] = None
  _enforceSanityChecks: Optional[bool] = None
  _allGranular: Optional[bool] = None
  _anisotropyFactor: Optional[float] = None
  _anisotropicMesh: Optional[bool] = None
  _refineFreq: int = 1
  _refineCount: Optional[int] = None
  _r: Optional[Point] = None
  _r_min: Optional[Point] = None
  _r_max: Optional[Point] = None
  _Delta_0: Optional[Point] = None
  _Delta_0_mant: Optional[Point] = None
  _pos_mant_0: Optional[Point] = None
  _HARD_MIN_MESH_INDEX: int = -300

  def __init__(self, pb_param: Parameters, run_options: Options):
    """ Constructor """
    super(Gmesh, self).__init__(pb_params=pb_param, limit_max_mesh_index=-GL_LIMITS, limit_min_mesh_index=GL_LIMITS)
        
    self._initFrameSizeExp = Point()
    self._frameSizeMant = Point()
    self._frameSizeExp = Point()
    self._finestMeshSize = Point()
    self._granularity = pb_param.granularity
    self._enforceSanityChecks = True
    self._allGranular = True
    self._anisotropyFactor = run_options.anisotropyFactor
    self._anisotropicMesh = run_options.anistropicMesh
    self._refineFreq = run_options.refineFreq
    self._refineCount = 0
    self._dtype = DType(run_options.precision) 
    self.init()

  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, other: DType):
    self.dtype = other

  def initFrameSizeGranular(self, initial_frame_size: Point):
    if not initial_frame_size.is_all_defined() or initial_frame_size.size != self._n:
      raise IOError("GMesh: initFrameSizeGranular: inconsistent dimension of the frame size. \n" +
                    f"initial frame size defined: {initial_frame_size.is_all_defined()} \n" +
                    f"size: {initial_frame_size.size} \n" +
                    f"n: {self._n}")
    
    self._frameSizeExp.reset(n=self._n)
    self._frameSizeMant.reset(n=self._n)
    d_min: Optional[float] = None
    for i in range(self._n):
      if self._granularity[i] > 0:
        d_min = self._granularity[i]
      else:
        d_min = 1
      
      div: float = initial_frame_size[i] / d_min
      exp: int = self.roundFrameSizeExp(np.log10(abs(div)))
      self._frameSizeExp[i] = exp
      self._frameSizeMant[i] = self.roundFrameSizeMant(div*10**-exp)

  def roundFrameSizeExp(self, exp: float) -> int:
    frame_size_exp: int = int(exp)
    return frame_size_exp
  
  def roundFrameSizeMant(self, mant: float):
    frame_size_mant: int = 0

    if mant < 1.5:
      frame_size_mant = 1
    elif mant >= 1.5 and mant < 3.5:
      frame_size_mant = 2
    else:
      frame_size_mant = 5

    return frame_size_mant
  
  def getRho(self, i: int = None) -> Any:
    
    if i is not None:
      rho: Any
      diff: float = self._frameSizeExp[i] - self._initFrameSizeExp[i]
      pow_diff: float = 10.0 ** abs(diff)

      if self._granularity[i] > 0:
        rho = self._frameSizeMant[i] * min(10.0**self._frameSizeExp[i], pow_diff)
      else:
        rho = self._frameSizeMant[i] * pow_diff
    else:
      rho: auto = [None] * self._n
      for i in range(self._n):
        diff: float = self._frameSizeExp[i] - self._initFrameSizeExp[i]
        pow_diff: float = 10.0 ** abs(diff)

        if self._granularity[i] > 0:
          rho[i] = self._frameSizeMant[i] * min(10.0**self._frameSizeExp[i], pow_diff)
        else:
          rho[i] = self._frameSizeMant[i] * pow_diff

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
    d_min_gran = 1.0
    Delta: Point = Point(self._n)
    Delta.coordinates = [0] * self._n
    if i is None:
      for i in range(self._n):
        if self._granularity[i] > 0:
          d_min_gran = self._granularity[i]
        Delta[i] = d_min_gran * self._frameSizeMant[i] * 10 ** self._frameSizeExp[i]
      return Delta
    else:
      if self._granularity[i] > 0:
        d_min_gran = self._granularity[i]
      Delta[i] = d_min_gran * self._frameSizeMant[i] * 10 ** self._frameSizeExp[i]
      return Delta[i]

  def getDeltaFrameSizeCoarser(self) -> Point:
    Delta: Point = Point(self._n)
    Delta.coordinates = [0] * self._n
    for i in range(self._n):
      frame_size_mant_old = self._frameSizeMant[i]
      frame_size_exp_old = self._frameSizeExp[i]
      self._frameSizeMant[i], self._frameSizeExp[i] = self.getLargerMantExp(frame_size_mant=frame_size_mant_old, i=i)
      Delta[i] = self.getDeltaFrameSize(i=i)
      self._frameSizeMant[i] = frame_size_mant_old
      self._frameSizeExp[i] = frame_size_exp_old
    
    return Delta

  def getLargerMantExp(self, frame_size_mant: float, i: int):
    if frame_size_mant == 1:
      self._frameSizeMant[i] = 2
    elif frame_size_mant == 2:
      self._frameSizeMant[i] = 5
    else:
      self._frameSizeMant[i] = 1
      self._frameSizeExp[i] += 1
    return self._frameSizeMant[i], self._frameSizeExp[i]
  
  def checkDeltasGranularity(self, i: int, delta_mesh_size: float, delta_frame_size: float):
    if self._granularity[i] > 0.0:
      has_error: bool = False
      err: str = "Error: setDeltas: "
      if not self.isMult(delta_mesh_size, self._granularity[i]):
        has_error = True
        err += f"deltaMeshSize at index {i}"
        err += f" is not a multiple of granularity {self._granularity[i]}"
      elif not self.isMult(delta_frame_size, self._granularity[i]):
        has_error = True
        err += f"deltaFrameSize at index {i}"
        err += f" is not a multiple of granularity {self._granularity[i]}"
      if has_error:
        raise IOError(err)
  
  def setDeltas(self, i: int = None, delta_mesh_size: float = None, delta_frame_size: float= None):
    # Input checks
    self.checkDeltasGranularity(i=i, delta_mesh_size=delta_mesh_size, delta_frame_size=delta_frame_size)
    # Value to use for granularity (division so default = 1.0)
    gran: float = 1.
    if 0. < self._granularity[i]:
      gran = self._granularity[i]
    
    mant: float
    exp: float
    # Compute mantisse first
    # There are only 3 cases: 1, 2, 5, so compute all
    # 3 possibilities and then assign the values that work.
    mant1: float = delta_frame_size / (1.*gran)
    mant2: float = delta_frame_size / (2.*gran)
    mant5: float = delta_frame_size / (5. *gran)

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
      self.checkFrameSizeIntegrity(frame_size_exp=self._frameSizeExp[i], 
                                   frame_size_mant=self._frameSizeMant[i])
      self.checkSetDeltas(i=i, delta_mesh_size=delta_mesh_size, delta_frame_size=delta_frame_size)
      self.checkDeltasGranularity(i, self.getdeltaMeshSize(i=i), self.getDeltaFrameSize(i=i))
  
  def checkFrameSizeIntegrity(self, frame_size_exp: float, frame_size_mant: float):
    # frameSizeExp must be an integer.
    # frameSizeMant must be 1, 2 or 5.
    has_error: bool = False
    err: str = "Error: Integrity check"
    if not isinstance(frame_size_exp, int):
      has_error = True
      err += f" of frameSizeExp ({frame_size_exp}): Should be integer."
    elif (not np.isclose(frame_size_mant, 1.0, rtol=1e-09, atol=1e-09) and not np.isclose(frame_size_mant, 2.0, rtol=1e-09, atol=1e-09) and not np.isclose(frame_size_mant, 5.0, rtol=1e-09, atol=1e-09)):
      has_error = True
      err += f" of frameSizeMant ({frame_size_mant}): Should be integer."
    
    if has_error:
      raise IOError(err)

  def checkSetDeltas(self, i: int, delta_mesh_size: float, delta_frame_size: float):
    has_error: bool = False
    err: str = "Warning: setDeltas did not give good value"

    # Something might be wrong with setDeltas(), so double check.
    if self.getdeltaMeshSize(i=i) != delta_mesh_size:
      has_error = True
      err += f" for deltaMeshSize at index {i}"
      err += f" Expected: {delta_mesh_size}"
      err += f" computed: {self.getdeltaMeshSize(i=i)}"
    elif self.getDeltaFrameSize(i=i) != delta_frame_size:
      has_error = True
      err += f" for deltaFrameSize at index {i}"
      err += f" Expected: {delta_frame_size}"
      err += f" computed: {self.getDeltaFrameSize(i=i)}"
    
    if (has_error):
      raise IOError(err)

  def scaleAndProjectOnMesh(self, dir: Point = None):
    proj: Point = Point(self._n)
    infinite_norm: float = np.linalg.norm(dir.coordinates, np.inf)

    if 0 == infinite_norm:
      err = "GMesh: scaleAndProjectOnMesh: Cannot handle an infinite norm of zero"
      raise IOError(err)
    
   

    if self._frameSizeMant.is_all_defined() and self._frameSizeExp.is_all_defined():
      for i in range(self._n):
        delta: Any = self.getdeltaMeshSize(i=i)
        proj[i] = np.round(self.getRho(i=i)*dir[i]/infinite_norm) * delta
    else:
      err = "GMesh: scaleAndProjectOnMesh cannot be performed."
      err += f" i = {i}"
      err += f" mantissa defined: {self._frameSizeMant.is_all_defined()}"
      err += f" exp defined: {self._frameSizeExp.is_all_defined()}"
      err += f"delta mesh size defined: {self.getdeltaMeshSize()}"
      raise IOError(err)
    
    return proj

  def projectOnMesh(self, point: Point, frame_center: Point):
    proj: Point = point
    delta: Any = self.getdeltaMeshSize()
    max_nb_try: int = 10
    verif_value_i: Point = Point(self._n)
    verif_value_i.coordinates = [0] * self._n
    for i in range(point.size):
      delta_i = delta[i]
      frame_center_is_on_mesh: bool = self.isMult(frame_center[i], delta_i)

      diff_proj_frame_center: float = proj[i] - frame_center[i]
      verif_value_i[i] = proj[i] if (frame_center_is_on_mesh) else diff_proj_frame_center
      # // Force verifValueI to be a multiple of deltaI.
      # // nbTry = 0 means point is already on mesh.
      # // nbTry = 1 means the projection worked.
      # // nbTry > 1 means the process went hacky by forcing the value to work
      # // for verifyPointIsOnMesh.
      nb_try = 0
      while (not self.isMult(verif_value_i[i], delta_i) and nb_try <= max_nb_try):
        new_verif_value_i: float
        if (0==nb_try):
          # Use closest projection
          v_high = verif_value_i.next_mult(delta_i, i)
          p: Point = Point(self._n)
          p.coordinates = [-c for c in verif_value_i.coordinates]
          v_low = - (p.next_mult(delta_i, i))
          diff_high = v_high - verif_value_i[i]
          diff_low = verif_value_i[i] - v_low
          verif_value_i[i] = v_low if (diff_low < diff_high) else (v_high if (diff_high < diff_low) else (v_low if (proj[i] < 0) else v_high))
        else:
          p: Point = Point(self._n)
          p.coordinates = [-c for c in verif_value_i.coordinates]
          verif_value_i[i] =  verif_value_i.next_mult(delta_i, i) if (diff_proj_frame_center >= 0) else (-(p.next_mult(delta_i, i)))
        proj[i] = verif_value_i[i] if frame_center_is_on_mesh else verif_value_i[i] + frame_center[i]

        # Recompute verifValue for more precision
        new_verif_value_i = proj[i] if frame_center_is_on_mesh else proj[i] - frame_center[i]
        nb_try += 1

        #  Special cases
        while (new_verif_value_i != verif_value_i[i] and nb_try <= max_nb_try):
          if verif_value_i[i] >= 0:
            verif_value_i[i] = max(verif_value_i[i], new_verif_value_i)
            verif_value_i[i] += self.dtype.zero
            verif_value_i[i] = verif_value_i.next_mult(delta_i, i)
          else:
            verif_value_i[i] = min(verif_value_i[i], new_verif_value_i)
            verif_value_i[i] -= self.dtype.zero
            p: Point = Point(self._n)
            p.coordinates = [-c for c in verif_value_i.coordinates]
            verif_value_i[i] = -(p.next_mult(delta_i, i))
          proj[i] = verif_value_i[i] if frame_center_is_on_mesh else verif_value_i[i] + frame_center[i]
          # Recompute verifValue for more precision
          new_verif_value_i = proj[i] if frame_center_is_on_mesh else proj[i] - frame_center[i]
          nb_try += 1
        
        verif_value_i[i] = new_verif_value_i
      
      if (nb_try >= max_nb_try and not self.isMult(verif_value_i[i], delta_i)):
        # TODO: print warning
        proj[i] = point[i]

    return proj

  def check_min_poll_size_criterion (self) -> bool:
    """ Check the minimal poll size criterion. """
    if not self._Delta_min_is_defined:
      return False
    S, _ = self.get_Delta_object()
    return S
        
  def check_min_mesh_size_criterion(self) -> bool:
    """ Check the minimal mesh size criterion. """
    if not self._delta_min.is_all_defined():
      return False
    S, _ = self.get_delta_object()
    return S
  
  def get_rho(self, i: int):
    """
    Access to the ratio of poll size / mesh size parameter rho^k.
    :param  rho The ratio poll/mesh size rho^k --  OUT.
    """
    rho: Optional[float] = None
    if self._granularity[i] > 0:
      rho = self._frameSizeMant.coordinates[i] * min(10** self._frameSizeExp.coordinates[i], 10**abs(self._frameSizeExp.coordinates[i]-self._initFrameSizeExp.coordinates[i]))
    else:
      rho = self._frameSizeMant.coordinates[i] * 10** abs(self._frameSizeExp.coordinates[i]-self._initFrameSizeExp.coordinates[i])
    return rho

  def get_delta(self, i: int): 
    """
    Access to the mesh size parameter delta^k.
    :param  delta: The mesh size parameter delta^k --  OUT.
    :param  i: The index of the mesh size
  """
    delta: float = 10**(self._frameSizeExp.coordinates[i]-abs(self._frameSizeExp.coordinates[i]-self._initFrameSizeExp.coordinates[i]))
    if self._granularity.coordinates[i]:
      delta = self._granularity[i] * max(1.0, delta)
    return delta
    
  def get_Delta(self, i: int): 
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
      if np.isclose(0.0, self._granularity[i], rtol=1e-09, atol=1e-09):
        self._allGranular = False
        break
    
    if not self._minMeshSize.is_complete():
      raise IOError("Expecting mesh minimum size to be fully defined.")
    
    if self._enforceSanityChecks:
      for i in range(self._n):
        self.checkFrameSizeIntegrity(frame_size_exp=self._frameSizeExp[i], frame_size_mant=self._frameSizeMant[i])
        self.checkDeltasGranularity(i=i, delta_mesh_size=self.getdeltaMeshSize(i=i), delta_frame_size=self.getDeltaFrameSize(i=i))
  
  def isMult(self, v1, v2)->bool:
    return ((v1%v2) <= self.dtype.zero)

  def enlargeDeltaFrameSize(self, direction: Point = None) -> bool:
    one_frame_size_changed = False
    min_rho = np.inf
    for i in range(self._n):
      if self._granularity[i] == 0:
        min_rho = min(min_rho, self.getRho(i=i))
    
    for i in range(self._n):
      frame_size_i_changed = False
      if (not self._anisotropicMesh or abs(direction[i])/self.getdeltaMeshSize(i=i)/self.getRho(i=i) > self._anisotropyFactor or (self._granularity[i] == 0 and self._frameSizeExp[i] < self._initFrameSizeExp[i] and self.getRho(i=i) > min_rho*min_rho)):
        self.getLargerMantExp(frame_size_mant=self._frameSizeMant[i], i=i)
        frame_size_i_changed = True
        one_frame_size_changed = True
        # update the mesh index
        self._r[i] += 1
        self._rMax[i] = max(self._r[i], self._rMax[i])

        # Sanity checks
        if self._enforceSanityChecks and frame_size_i_changed:
          self.checkFrameSizeIntegrity(self._frameSizeExp[i], self._frameSizeMant[i])
          self.checkDeltasGranularity(i=i, delta_mesh_size=self.getdeltaMeshSize(i=i), delta_frame_size=self.getDeltaFrameSize(i=i))
        
    # When we enlarge the frame size we may keep the mesh size unchanged. So we need to test.
    msize = self.getdeltaMeshSize()
    if self._finestMeshSize < msize:
      self._isFinest = False
    
    return one_frame_size_changed

  def refineDeltaFrameSizeME(self, frame_size_mant: float, frame_size_exp:float, granularity: float):
    if frame_size_mant == 1:
      frame_size_mant = 5
      frame_size_exp -= 1
    elif frame_size_mant == 2:
      frame_size_mant = 1
    else:
      frame_size_mant = 2
    
    # When the mesh reaches granularity (exp = 1, mant = 1), make sure to remove the refinement
    if granularity > 0 and frame_size_exp < 0 and frame_size_mant == 5:
      frame_size_exp = 0
      frame_size_mant = 1
    
    return frame_size_mant, frame_size_exp
  
  def getdeltaMeshSizeF(self, frame_size_exp:int, init_frame_size_exp:int, granularity: int)->float:
    diff = frame_size_exp - init_frame_size_exp
    exp = frame_size_exp - abs(diff)
    delta = 10.0**exp
    if 0.0 < granularity:
      delta = granularity * max(1.0, delta)
    
    return delta

  def refineDeltaFrameSize(self):
    # // Compute the new values frameSizeMant and frameSizeExp first.
    # // We will do some verifications before setting them.
    self._refineCount += 1
    if self._refineCount%self._refineFreq != 0:
      return
    
    for i in range(self._n):
      # // Compute the new values frameSizeMant and frameSizeExp first.
      # // We will do some verifications before setting them.
      frame_size_mant = self._frameSizeMant[i]
      frame_size_exp = self._frameSizeExp[i]
      frame_size_mant, frame_size_exp= self.refineDeltaFrameSizeME(frame_size_mant=frame_size_mant, frame_size_exp=frame_size_exp, granularity=self._granularity[i])
      # Verify delta mesh size does not go too small if we use the new values.
      old_delta_mesh_size = self.getdeltaMeshSizeF(frame_size_exp=self._frameSizeExp[i], init_frame_size_exp=self._initFrameSizeExp[i], granularity=self._granularity[i])
      if self._minMeshSize[i] <= old_delta_mesh_size:
        # update mesh index
        if self._granularity[i] == 0:
          self._r[i] -= 1
        else:
          # Update mesh index if not already at the min limit. When refining the frame, if mantissa and exponent stay the same, the min limit is reached (do not decrease).
          if (not (self._frameSizeMant[i] == frame_size_mant and self._frameSizeExp[i] == frame_size_exp)):
            self._r[i] -= 1
        # Update the minimal mesh index reached so far
        self._rMin[i] = min(self._r[i], self._rMin[i])

        # We can go lower
        self._frameSizeMant[i] = frame_size_mant
        self._frameSizeExp[i] = frame_size_exp

      # Sanity checks
      if self._enforceSanityChecks:
        self.checkFrameSizeIntegrity(frame_size_exp=self._frameSizeExp[i], 
                                     frame_size_mant=self._frameSizeMant[i])
        self.checkDeltasGranularity(i=i, delta_mesh_size=self.getdeltaMeshSize(i=i), delta_frame_size=self.getDeltaFrameSize(i=i))
    msize = self.getdeltaMeshSize()
    if msize <= self._finestMeshSize:
      self._isFinest = True
      self._finestMeshSize = msize
    else:
      self._isFinest = False
    
  def update(self):
    return


  





  

