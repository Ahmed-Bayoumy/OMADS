import copy
from dataclasses import dataclass, field
from typing import List, Optional
from .CandidatePoint import CandidatePoint
from .Point import Point
from ._globals import DType, DESIGN_STATUS, EVAL_TYPE
import numpy as np
from typing import Protocol
from .Cache import Cache

@dataclass
class BarrierData(Protocol):
  _xFeas: Optional[List[CandidatePoint]] = None
  _xInf: Optional[List[CandidatePoint]] = None

  _xIncFeas: Optional[List[CandidatePoint]] = None
  _xIncInf: Optional[List[CandidatePoint]] = None

  _refBestFeas: Optional[CandidatePoint] = None
  _refBestInf: Optional[CandidatePoint] = None

  _dtype: Optional[DType] = None

  def init(self, eval_point_list: Optional[List[Point]] = None):
    ...

  def getAllXFeas(self):
    ...
  
  def clone(self):
    ...
  
  def getCurrentIncumbentFeas(self):
    ...
  
  def getAllXIncFeas(self):
    ...
  
  def getRefBestFeas(self):
    ...
  
  def setRefBestFeas(self):
    ...
  
  def updateRefBests(self):
    ...
  
  def nbXFeas(self):
    ...

  def clearXFeas(self):
    ...
  
  def getAllXInf(self):
    ...
  
  def getAllXIncInf(self):
    ...
  
  def getCurrentIncumbentInf(self):
    ...
  
  def getRefBestInf(self):
    ...
  
  def setRefBestInf(self):
    ...
  
  def nbXInf(self):
    ...
  
  def clearXInf(self):
    ...
  
  def getAllPoints(self) -> List[CandidatePoint]:
    ...
  
  def getFirstPoint(self) -> CandidatePoint:
    ...
  
  def getHMax(self):
    ...
  
  def setHMax(self):
    ...
  
  def getSuccessTypeOfPoints(self):
    ...
  
  def updateWithPoints(self):
    ...
  
  def findPoint(self, point: Point):
    ...
  
  def setN(self):
    ...
  
  def checkHMax(self):
    ...
  
  def checkCache(self):
    ...
  
  def findEvalPoint(self):
    ...



@dataclass
class BarrierBase(BarrierData):
  

  _h_max: float = np.inf

  _n: int = 0

  def __init__(self, h_max: float = np.inf):
    self._h_max = h_max
    self._n = 0
    self._dtype = DType()
    self._xInf = []
    self._xFeas = []
    self._xIncFeas = []
    self._xIncInf = []
  
  def setN(self):
    is_set: bool = False
    s: str

    for cp in self.getAllPoints():
      if not is_set:
        self._n = cp._n
        is_set = True
      elif cp._n != self._n:
        s = f"Barrier has points of size {self._n} and of size {cp._n}"
        raise IOError(s)
    if not is_set:
      raise IOError("Barrier could not set point size")
  
  def checkCache(self, cache: Cache = None):
    if cache == None:
      raise IOError("Cache must be instantiated before initializing Barrier.")
  
  def checkHMax(self):
    if self._h_max is None or self._h_max < self._dtype.zero:
      raise IOError("Barrier: hMax must be positive.")
  
  def clearXFeas(self):
    del self._xFeas
  
  def clearXInf(self):
    del self._xInf
    del self._xIncInf
  
  def getAllPoints(self) -> List[CandidatePoint]:
    all_points: List[CandidatePoint] = []
    if self._xFeas is None:
      self._xFeas = []
    for cp in self._xFeas:
      all_points.append(cp)
    if self._xInf is None:
      self._xInf = []
    for cp in self._xInf:
      all_points.append(cp)
    
    return all_points
  
  def getFirstPoint(self) -> Optional[CandidatePoint]:
    if self._xIncFeas and len(self._xIncFeas) > 0:
      return self._xIncFeas[0]
    elif self._xFeas and len(self._xFeas) > 0:
      return self._xFeas[0]
    elif self._xIncInf and len(self._xIncInf) > 0:
      return self._xIncInf[0]
    elif self._xInf and len(self._xInf) > 0:
      return self._xInf[0]
    else:
      return None

  def findEvalPoint(self, cps: List[CandidatePoint] = None, cp: CandidatePoint = None):
    for p in cps:
      if p.signature == cp.signature:
        return True, p
    
    return False, p
  
  def findPoint(self, point: Point) -> bool:
    found: bool = False

    eval_point_list: List[CandidatePoint] = self.getAllPoints()
    for cp in eval_point_list:
      if cp._n != point._n:
        raise IOError("Error: Eval points have different dimensions")
      if point == cp.coordinates:
        found = True
        break
    
    return found

  def checkXFeas(self, x_feas: CandidatePoint = None, eval_type: EVAL_TYPE = None):
    if x_feas.evaluated:
      self.checkXFeasIsFeas(x_feas=x_feas, eval_type=eval_type)

  def getAllXFeas(self): 
    return self._xFeas

  def checkXFeasIsFeas(self, x_feas: CandidatePoint=None, eval_type: EVAL_TYPE = None):
    if x_feas.evaluated and x_feas.status != DESIGN_STATUS.ERROR:
      h = x_feas.h
      if h is None or not np.isclose(h, 0.0, rtol=1e-09, atol=1e-09):
        raise IOError(f"Error: Barrier: xFeas' h value must be 0.0, got: {h}")

  def checkXInf(self, x_inf: CandidatePoint = None, eval_type: EVAL_TYPE = None):
    if not x_inf.evaluated:
      raise IOError("Barrier: xInf must be evaluated before being set.")
