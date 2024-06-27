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
from .Points import Point
from .OrthoMesh import *

@dataclass
class Gmesh(OrthoMesh):
  """ GMesh: Granular mesh """
  _r: Point = None
  _r_min: Point = None
  _r_max: Point = None
  _Delta_0: Point = None
  _Delta_0_exp: Point = None
  _Delta_mant: Point = None
  _Delta_0_mant: Point = None
  _Delta_exp: Point = None
  _pos_mant_0: Point = None
  _HARD_MIN_MESH_INDEX: int = -300

  def __init__(self, anisotropic_mesh: bool = False,
                    anisotropy_factor: float = 0.1,
                    initial_poll_size: Point = None,
                    min_poll_size: Point = None,
                    min_mesh_size: Point = None,
                    fixed_variables: Point = None,
                    granularity: Point = None,
                    poll_update_basis: float = 0,  
                    poll_coarsening_step: int = 0,
                    poll_refining_step: int = 0,
                    limit_min_mesh_index: int = GL_LIMITS):
    """ Constructor """
    super(Gmesh, self).__init__(anisotropic_mesh = anisotropic_mesh,
                    anisotropy_factor = anisotropy_factor,
                    Delta_0 = initial_poll_size,
                    Delta_min = min_poll_size,
                    delta_min = min_mesh_size,
                    fixed_variables = fixed_variables,
                    granularity = granularity,
                    update_basis = poll_update_basis,  
                    coarsening_step = poll_coarsening_step,
                    refining_step = poll_refining_step,
                    limit_mesh_index = limit_min_mesh_index)
    
    if (self._limit_mesh_index>0):
      raise IOError("Limit mesh index must be <=0 ")
    

    # Set the mesh indices
    self._r.coordinates = [None]*self._n
    self._r_max.coordinates = [None]*self._n
    self._r_min.coordinates = [None]*self._n

    for k in range(self._n):
      self._r.coordinates[k] = 0
      self._r_max.coordinates[k] = 0
      self._r_min.coordinates[k] = 0
    
    self.init_poll_size_granular(self._Delta_0)
    self._Delta_0_exp = self._Delta_exp
    self._Delta_0_mant = self._Delta_mant
    _, self._Delta_0 = self.get_Delta_object()
    _, self._delta_0 = self.get_delta_object()



    
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
      rho = self._Delta_mant.coordinates[i] * min(10** self._Delta_exp.coordinates[i], 10**abs(self._Delta_exp.coordinates[i]-self._Delta_0_exp.coordinates[i]))
    else:
      rho = self._Delta_mant.coordinates[i] * 10** abs(self._Delta_exp.coordinates[i]-self._Delta_0_exp.coordinates[i])
    return rho


  def get_delta (self, i: int): 
    """
    Access to the mesh size parameter delta^k.
    :param  delta: The mesh size parameter delta^k --  OUT.
    :param  i: The index of the mesh size
  """
    delta: float = 10**(self._Delta_exp.coordinates[i]-abs(self._Delta_exp.coordinates[i]-self._Delta_0_exp.coordinates[i]))
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
    Delta: float = d_min_gran * self._Delta_mant.coordinates[i] * 10**(self._Delta_exp.coordinates[i])
    
    return Delta
  
  def init(self):
    """Initialization of granular poll size mantissa and exponent"""
    # Set the mesh indices
    self._r = [0] * self._n
    self._r_max = [0] * self._n
    self._r_min = [0] * self._n

    # Set the mesh mantissas and exponents
    self.init_poll_size_granular(self._Delta_0)

    # Update mesh and poll after granular sizing
    self._Delta_0_exp = self._Delta_exp
    self._Delta_0_mant = self._Delta_mant
    # Update mesh and poll after granular sizing
    self._Delta_0 = self.get_Delta_object()
    self._delta_0 = self.get_delta_object()



  
  def init_poll_size_granular (self, cont_init_poll_size: Point ):
    """
    :param: cont_init_poll_size: continuous initial poll size   --  IN.
    """

    if not all(cont_init_poll_size.defined) or cont_init_poll_size.n_dimensions != self._n:
      raise IOError("Inconsistent dimension of the poll size!")
    
    self._Delta_exp.reset(n=self._n)
    self._Delta_mant.reset(n=self._n)
    self._pos_mant_0.reset(n=self._n)

    d_min: float

    for i in range(self._n):
      if self._granularity.defined[i] and self._granularity.coordinates[i] > 0:
        d_min = self._granularity[i]
      else:
        d_min=1.0
      
      exp: int = int(np.log10(abs(cont_init_poll_size.coordinates[i]/d_min)))
      if exp < 0:
        exp = 0

      self._Delta_exp.coordinates[i]=exp
      cont_mant: float = cont_init_poll_size.coordinates[i] / d_min * 10.0**(-exp)

      if cont_mant < 1.5:
        self._Delta_mant.coordinates[i] = 1
        self._pos_mant_0[i] = 0
      elif (cont_mant >= 1.5 and  cont_mant < 3.5):
        self._Delta_mant.coordinates[i] = 2
        self._pos_mant_0.coordinates[i] = 1
      else:
        self._Delta_mant.coordinates[i] = 5
        self._pos_mant_0.coordinates[i] = 2

      
  
  def get_delta_object(self):
    """  """
    stop = True
    delta: Point = Point(self._n)
    for i in range(self._n):
      delta.coordinates[i] = self.get_delta(i=i)
      if stop and self._delta_min_is_defined and not self._fixed_variables.defined[i] and self._delta_min.defined[i] and delta.coordinates[i] >= self._delta_min[i]:
        stop = False
    return stop, delta
  
  def get_delta_max(self)->Point:
    return self._delta_0
  
  def get_Delta_object(self)->Point:
    """ """
    stop = True
    Delta: Point = Point(self._n)
    for i in range(self._n):
      Delta.coordinates[i] = self.get_Delta(i=i)
      if stop and self._granularity.coordinates[i] == 0 and not self._fixed_variables.defined[i] and (self._Delta_min_is_complete or Delta.coordinates[i] >= self._Delta_min[i]):
        stop = False
    
      if stop and self._granularity.coordinates[i] > 0 and not self._fixed_variables.defined[i] and (not self._Delta_min_is_complete or Delta.coordinates[i] > self._Delta_min[i]):
        stop = False


    return stop, Delta
  
  def is_finer_than_initial(self):
    """ """
    for i in range(self._n):
      if not self._fixed_variables.defined[i]:
        # For continuous variables
        if self._granularity.coordinates[i]==0 and (self._Delta_exp.coordinates[i] > self._Delta_0_exp.coordinates[i] or ( self._Delta_exp.coordinates[i] == self._Delta_0_exp.coordinates[i] and self._Delta_mant.coordinates[i] >= self._Delta_0_mant.coordinates[i] )):
          return False
        # For granular variables (case 1)
        if self._granularity.coordinates[i] > 0 and (self._Delta_exp.coordinates[i] > self._Delta_0_exp.coordinates[i] or ( self._Delta_exp.coordinates[i] == self._Delta_0_exp.coordinates[i] and self._Delta_mant.coordinates[i] > self._Delta_0_mant.coordinates[i] )):
          return False
        # For continuous variables (case 2)
        if self._granularity.coordinates[i]>0 and (self._Delta_exp.coordinates[i] == self._Delta_0_exp.coordinates[i] and  self._Delta_mant.coordinates[i] == self._Delta_0_mant.coordinates[i] and (self._Delta_exp.coordinates[i] != 0 or self._Delta_mant.coordinates[i] != 1) ):
          return False
    
    return True
  
  def update(self, success: SUCCESS_TYPES, d: List[float]):
    if d and self._n != len(d):
      raise IOError("delta_0 and d have different sizes")
    
    if success == SUCCESS_TYPES.FS:
      for i in range(self._n):
        if (self._granularity.coordinates[i] == 0 and not self._fixed_variables.defined[i]):
          if i > 0:
            min_rho = min(min_rho, self.get_rho(i))
          else:
            min_rho = self.get_rho(i)
          
      for i in range(self._n):
        if (not d or not self._anisotropic_mesh or abs(d[i])/self.get_delta(i)/self.get_rho(i) > self._anisotropic_factor or ( self._granularity.coordinates[i] == 0  and self._Delta_exp.coordinates[i] < self._Delta_0_exp.coordinates[i] and self.get_rho(i) > min_rho*min_rho )):
          # Update the mesh index
          self._r.coordinates[i] += 1
          self._r_max.coordinates[i] = max(self._r.coordinates[i], self._r_max.coordinates[i])
          # update the mantissa and exponent
          if ( self._Delta_mant.coordinates[i] == 1 ):
              self._Delta_mant.coordinates[i]= 2
          elif ( self._Delta_mant.coordinates[i] == 2 ):
              self._Delta_mant.coordinates[i]=5
          else:
            self._Delta_mant.coordinates[i]=1
            self._Delta_exp.coordinates[i] += 1
    elif success == SUCCESS_TYPES.US:
      for i in range(self._n):
        if (not self._fixed_variables.defined[i]):
          # update the mesh index
          self._r.coordinates[i] -= 1
          # update the mesh mantissa and exponent
          if (self._Delta_mant.coordinates[i]==1):
            self._Delta_mant.coordinates[i] = 5
            self._Delta_exp.coordinates[i] -= 1
          elif self._Delta_mant.coordinates[i] == 2:
            self._Delta_mant.coordinates[i] = 1
          else:
            self._Delta_mant.coordinates[i] = 2
          
          if ( self._granularity.coordinates[i] > 0 and self._Delta_exp.coordinates[i]==-1 and self._Delta_mant.coordinates[i]==5 ):
            self._r.coordinates[i] += 1
            self._Delta_exp.coordinates[i]=0
            self._Delta_mant.coordinates[i]=1
        self._r_min.coordinates[i] = min(self._r.coordinates[i], self._r_min.coordinates[i])

      # for i in range(self._n):
      #   # Test for producing anisotropic mesh + correction to prevent mesh collapsing for some variables ( ifnot )
      #   if (not d or not self._anisotropic_mesh or d[i]/self.get_delta(i)):

  
  def reset(self):
    """ """
    self.__init__()
  
  def is_finest(self):
    """ """
    for i in range(self._n):
      if not self._fixed_variables.defined[i] and self._r.coordinates[i] > self._r_min.coordinates[i]:
        return False
    return True
  

  
  def scale_and_project(self, i: int, l: float, round_up: bool):
    """ """
    delta: float = self.get_delta(i=i)
    if i<= self._n and self._Delta_mant.is_all_defined() and self._Delta_exp.is_all_defined() and delta is not None:
      d: float = self.get_rho(i=i) * l
      # round to double
      return np.round(d)*delta
    else:
      raise IOError("scale_and_project(): mesh scaling and projection cannot be performed!")



  
  def check_min_mesh_sizes(self, stop: bool=None, stop_reason: STOP_TYPE = None):
    """_summary_
    """
    if stop:
      return
    
    stop = False
    # Coarse mesh stopping criterion
    for i in range(self._n):
      if self._r.coordinates[i] > -GL_LIMITS:
        stop = True
        break
    if stop:
      stop_reason = STOP_TYPE.GL_LIMITS_REACHED
      return
    
    stop = True

    # // Fine mesh stopping criterion. Do not apply when all variables have granularity.
    # // To trigger this stopping criterion:
    # //  - All mesh indices must be < _limit_mesh_index for all continuous variables (granularity==0), and
    # //  - mesh size == granularity for all granular variables.
    if self._all_granular:
      stop = False
    
    else:
      for i in range(self._n):
        # Skip fixed variables
        if self._fixed_variables.defined[i]:
          continue
        # Do not stop if the mesh size of a variable is strictly larger than its granularity
        if self._granularity.coordinates[i] > 0 and self.get_delta(i=i) > self._granularity.coordinates[i]:
          stop = False
          break
        # Do not stop if the mesh of a variable is above the limit mesh index
        if self._granularity.coordinates[i] == 0 and self._r.coordinates[i] >= self._granularity.coordinates[i]:
          stop = False
          break
    
    if stop:
      stop_reason = STOP_TYPE.GL_LIMITS_REACHED
      return
    
    # 2. delta^k (mesh size) tests:
    if self.check_min_poll_size_criterion():
      stop = True
      stop_reason = STOP_TYPE.DELTA_P_MIN_REACHED
      return

    # 3. delta^k (mesh size) tests:
    if self.check_min_mesh_size_criterion():
      stop = True
      stop_reason = STOP_TYPE.DELTA_M_MIN_REACHED
      return

    

  
  def get_mesh_indices(self):
    """_summary_
    """
    return self._r

  
  def get_min_mesh_indices(self):
    """_summary_
    """
    return self._r_min
  
  def get_max_mesh_indices(self):
    """_summary_
    """
    return self._r_max
  
  def set_mesh_indices(self, r: Point):
    """_summary_
    """
    if r.size != self._n:
      raise IOError("set_mesh_indices(): dimension of provided mesh indices must be consistent with their previous dimension")
    
    if r.coordinates[0] < HARD_MIN_MESH_INDEX:
      raise IOError("set_mesh_indices(): mesh index is too small")
    
    # Set the mesh indices
    self._r = copy.deepcopy(r)
    for i in range(self._n):
      if (r.coordinates[i]>self._r_max.coordinates[i]):
        self._r_max.coordinates[i] = r.coordinates[i]
      if (r.coordinates[i] < self._r_min.coordinates[i]):
        self._r_min.coordinates[i] = r.coordinates[i]
    
    # Set the mesh mantissas and exponents according to the mesh indices
    for i in range(self._n):
      shift: int = int(self._r.coordinates[i] + self._pos_mant_0.coordinates[i])
      pos: int = (shift + 300) % 3

      self._Delta_exp.coordinates[i] = np.floor((shift+300.0)/3.0) - 100.0 + self._Delta_0_exp.coordinates[i]

      if pos == 0:
        self._Delta_mant.coordinates[i] = 1
      elif pos == 1:
        self._Delta_mant.coordinates[i] = 2
      elif pos == 2:
        self._Delta_mant.coordinates[i] = 5
      else:
        raise IOError("set_mesh_indices(): something is wrong with conversion from index to mantissa and exponent")
  
  def set_limit_mesh_index(self, l: int):
    """_summary_
    """
    if l > 0:
      raise IOError("set_limit_mesh_index(): the limit mesh index must be negative or null.")
    
    if l > HARD_MIN_MESH_INDEX:
      raise IOError("set_limit_mesh_index(): the limit mesh index is too small.")
    
    self._limit_mesh_index = l
  

  
  def get_mesh_ratio_if_success(self):
    """_summary_
    """
    ratio: Point = Point(self._n)
    for i in range(self._n):
      power_of_tau: float = self._update_basis**(0 if self._r.coordinates[i] >= 0 else 2*self._r.coordinates[i])

      power_of_tau_if_success: float = self._update_basis**(0 if self._r.coordinates[i]+self._coarsening_step >= 0 else 2*(self._r.coordinates[i]+self._coarsening_step)) 

      ratio.coordinates[i] = power_of_tau_if_success/power_of_tau
    
    return ratio

  





  

