import copy
from typing import List
from ._globals import *
from .Points import Point

@dataclass
class OrthoMesh:
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
  _anisotropic_mesh: bool = None
  _anisotropy_factor: int = 0.1
  _delta: Point = None  # mesh size
  _Delta: Point = None  # poll size
  _delta0: Point = None  # mesh size
  _Delta0: Point = None  # poll size
  _delta_min: Point = None  # mesh size
  _Delta_min: Point = None  # poll size
  _fixed_variables: Point = None
  _granularity: Point = None

  _all_granular: bool = None
  _Delta_min_is_defined: bool = None
  _Delta_min_is_complete: bool = None
  _delta_min_is_defined: bool = None
  _delta_min_is_complete: bool = None

  _update_basis: float = None
  _coarsening_step: int = None
  _refining_step: int = None
  _n_free_variables: int = None
  _limit_mesh_index: int = None
  _n: int = None

  _rho: float = 1.0  # poll size to mesh size ratio
  # TODO: manage the poll size granularity for discrete variables
  # See: Audet et. al, The mesh adaptive direct search algorithm for
  # granular and discrete variable
  _exp: int = 0
  _mantissa: int = 1
  psize_max: float = 0.0
  psize_success: float = 0.0
  # numpy double data type precision
  _dtype: DType = None

  def __init__(self, anisotropic_mesh: bool,
                    anisotropy_factor: float,
                    Delta_0: Point,
                    Delta_min: Point,
                    delta_min: Point,
                    fixed_variables: Point,
                    granularity: Point,
                    update_basis: float,  
                    coarsening_step: int,
                    refining_step: int,
                    limit_mesh_index: int):
    """ initialize """
    self._anisotropic_mesh = anisotropic_mesh
    self._Delta0 = Delta_0
    self._delta0 = Delta_0
    self._Delta_min = Delta_min
    self._delta_min = delta_min
    self._anisotropy_factor = anisotropy_factor
    self._fixed_variables = fixed_variables
    self._granularity = granularity
    self._update_basis = update_basis
    self._coarsening_step = coarsening_step
    self._refining_step = refining_step
    self._limit_mesh_index = limit_mesh_index

    self._n = Delta_0.size
    self._n_free_variables = self._n - self._fixed_variables.size

    


  

  def __post_init__(self):
    self._dtype = DType()
    self._Delta_min_is_defined = all(self._Delta_min.defined)
    self._delta_min_is_defined = all(self._delta_min.defined)
    self._n = len(self._Delta0)
    self._n_free_variables = 0
    for i in range(self._n): self._n_free_variables += 1 if self._Delta0.var_type[i] == VAR_TYPE.CONTINUOUS else 0
    self._all_granular = True if self._granularity.defined else False

    for i in range(self._n):
      if self._delta_min_is_defined and self._delta_min.defined[i] and self._delta0.coordinates[i] < self._delta_min.coordinates[i]:
        raise IOError("delta_0 < delta_min")
      
      if self._Delta_min_is_defined and self._Delta_min.defined[i] and self._Delta0.coordinates[i] < self._Delta_min.coordinates[i]:
        raise IOError("Delta_0 < Delta_min")
      
      if self._all_granular and not self._fixed_variables.defined[i] and self._granularity.coordinates[i] == 0:
        self._all_granular = False


  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, other: DType):
    self._dtype = other

  @property
  def msize(self):
    return self._delta

  @msize.setter
  def msize(self, size):
    self._delta = size

  @msize.deleter
  def msize(self):
    del self._delta

  @property
  def psize(self):
    return self._Delta

  @psize.setter
  def psize(self, size):
    self._Delta = size

  @psize.deleter
  def psize(self):
    del self._Delta

  @property
  def rho(self):
    return self._rho

  @rho.setter
  def rho(self, size):
    self._rho = size

  @rho.deleter
  def rho(self):
    del self._rho

  
  def is_finer_than_initial(self):
    delta: Point = self.get_delta()
    for i in range(self._n):
      if not self._fixed_variables.defined[i] and delta.coordinates[i] >= self._delta0.coordinates[i]:
        return False
    return True
    
  def set_min_mesh_sizes(self, delta_min: Point):
    if (not all(delta_min.defined)):
      self._delta_min_is_defined = False
      self._delta_min_is_complete = False
      return
    
    if self._n != delta_min.size:
      raise IOError("set_min_mesh_sizes(): delta_min has dimension different than mesh dimension")
    
    if not delta_min.is_complete():
      raise IOError("set_min_mesh_sizes(): delta_min has some defined and undefined values")
    
    self._delta_min.reset(self._n, None)
    self._delta_min_is_defined = True
    self._delta_min_is_complete = True
    self._delta_min = copy.deepcopy(delta_min)

    for k in range(self._n):
      if delta_min.defined[k] and self._delta0.coordinates[k] < delta_min.coordinates[k]:
        self._delta_min.coordinates[k] = self._delta0.coordinates[k]
      
      if delta_min.defined[k] and self._Delta0.coordinates[k] < delta_min.coordinates[k]:
        self._delta_min.coordinates[k] = self._Delta0.coordinates[k]

  def set_min_poll_sizes(self, Delta_min: Point):
    if not Delta_min.is_all_defined():
      self._Delta_min_is_defined = False
      self._Delta_min_is_complete = False
      return
    
    if Delta_min.size != self._n:
      raise IOError("set_min_poll_sizes() Delta_min has dimension different than mesh dimension")
    
    if not Delta_min.is_complete():
      raise IOError("set_min_poll_sizes() Delta_min has some defined and undefined values")
    
    self._Delta_min.reset(self._n)
    self._Delta_min = Delta_min
    self._Delta_min_is_defined = True
    self._Delta_min_is_complete = True
    
    for k in range(self._n):
      if (Delta_min.defined[k] and self._Delta0.coordinates[k] < Delta_min.coordinates[k]):
        self._Delta_min.coordinates[k] = self._Delta0.coordinates[k]

  def set_delta_0(self, d: Point):
    if self._delta0.size != d.size:
      raise IOError("set_delta_0(): dimension of provided delta_0 must be consistent with their previous dimension")
    
    self._delta0 = copy.deepcopy(d)

  def set_Delta_0(self, d: Point):
    if self._Delta0.size != d.size:
      raise IOError("set_delta_0(): dimension of provided Delta_0 must be consistent with their previous dimension")
    
    self._Delta0 = copy.deepcopy(d)