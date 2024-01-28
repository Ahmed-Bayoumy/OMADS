
from enum import Enum, auto
from dataclasses import dataclass, field
import warnings
import numpy as np
import platform
import pandas as pd
@dataclass
class DType:
  """A numpy data type delegator for decimal precision control

    :param prec: Default precision option can be set to "high", "medium" or "low" precision resolution, defaults to "medium"
    :type prec: str, optional
    :param dtype: Numpy double data type precision, defaults to np.float64
    :type dtype: np.dtype, optional
    :param itype: Numpy integer data type precision, defaults to np.int
    :type itype: np.dtype, optional
    :param zero: Zero value resolution, defaults to np.finfo(np.float64).resolution
    :type zero: float, optional
  """
  _prec: str = "medium"
  _dtype: np.dtype = np.float64
  _itype: np.dtype = np.int_
  _zero: float = (np.finfo(np.float64)).resolution
  _warned: bool = False


  @property
  def zero(self)->float:
    """This is the mantissa value of the machine zero

    :return: machine precision zero resolution
    :rtype: float
    """
    return self._zero

  @property
  def precision(self):
    """Set/get precision resolution level

    :return: float precision value
    :rtype: numpy float
    :note: MS Windows does not support precision with the {1e-18} high resolution of the python
            numerical library (numpy) so high precision will be
            changed to medium precision which supports {1e-15} resolution
            check: https://numpy.org/doc/stable/user/basics.types.html 
    """
    return self._prec

  @precision.setter
  def precision(self, val: str):
    self._prec = val
    self._prec = val
    isWin = platform.platform().split('-')[0] == 'Windows'
    if val == "high":
      if (isWin or not hasattr(np, 'float128')):
        'Warning: MS Windows does not support precision with the {1e-18} high resolution of the python numerical library (numpy) so high precision will be changed to medium precision which supports {1e-15} resolution check: https://numpy.org/doc/stable/user/basics.types.html '
        self.dtype = np.float64
        self._zero = np.finfo(np.float64).resolution
        self.itype = np.int_
        if not self._warned:
          warnings.warn("MS Windows does not support precision with the {1e-18} high resolution of the python numerical library (numpy) so high precision will be changed to medium precision which supports {1e-15} resolution check: https://numpy.org/doc/stable/user/basics.types.html")
        self._warned = True
      else:
        self.dtype = np.float128
        self._zero = np.finfo(np.float128).resolution
        self.itype = np.int_
    elif val == "medium":
      self.dtype = np.float64
      self._zero = np.finfo(np.float64).resolution
      self.itype = np.intc
    elif val == "low":
      self.dtype = np.float32
      self._zero = np.finfo(np.float32).resolution
      self.itype = np.short
    else:
      raise Exception("JASON parameters file; unrecognized textual"
              " input for the defined precision type. "
              "Please enter one of these textual values (high, medium, low)")

  @property
  def dtype(self):
    """Set/get the double formate type

    :return: float precision value
    :rtype: numpy float 
    """
    return self._dtype

  @dtype.setter
  def dtype(self, other: np.dtype):
    self._dtype = other

  @property
  def itype(self):
    """Set/Get for the integer formate type

    :return: integer precision value
    :rtype: numpy float 
    """
    return self._itype

  @itype.setter
  def itype(self, other: np.dtype):
    self._itype = other

class VAR_TYPE(Enum):
  CONTINUOUS = auto()
  INTEGER = auto()
  DISCRETE = auto()
  BINARY = auto()
  CATEGORICAL = auto()
  ORDINAL = auto()

class BARRIER_TYPES(Enum):
  EB = auto()
  PB = auto()
  PEB = auto()
  RB = auto()

class SUCCESS_TYPES(Enum):
  US = auto()
  PS = auto()
  FS = auto()
  
class MPP(Enum):
  LAMBDA = 0.
  RHO = 0.00005

class DESIGN_STATUS(Enum):
  FEASIBLE = auto()
  INFEASIBLE = auto()
  ERROR = auto()
  UNEVALUATED = auto()

class MSG_TYPE(Enum):
  DEBUG = auto()
  WARNING = auto()
  ERROR = auto()
  INFO = auto()
  CRITICAL = auto()