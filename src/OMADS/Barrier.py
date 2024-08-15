import copy
from dataclasses import dataclass, field
from typing import List, Any
from .CandidatePoint import CandidatePoint
from .Point import Point
from ._globals import *
import numpy as np
from typing import Protocol
from .Parameters import Parameters
from .Cache import Cache

@dataclass
class BarrierData(Protocol):
  _xFeas: List[CandidatePoint] = None
  _xInf: List[CandidatePoint] = None

  _xIncFeas: List[CandidatePoint] = None
  _xIncInf: List[CandidatePoint] = None

  _refBestFeas: CandidatePoint = None
  _refBestInf: CandidatePoint = None

  _dtype: DType = None

  def init(self, xFeas: CandidatePoint = None, evalType: EVAL_TYPE = None, barrierInitializedFromCache: bool = True):
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
  
  def findPoint(self, Point: Point, foundEvalPoint: CandidatePoint):
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
  

  _hMax: float = np.inf

  _n: int = 0

  def __init__(self, hMax: float = np.inf):
    self._hMax = hMax
    self._n = 0
    self._dtype = DType()
    self._xInf = []
    self._xFeas = []
    self._xIncFeas = []
    self._xIncInf = []
  
  def setN(self):
    isSet: bool = False
    s: str

    for cp in self.getAllPoints():
      if not isSet:
        self._n = cp._n
        isSet = True
      elif cp._n != self._n:
        s = f"Barrier has points of size {self._n} and of size {cp._n}"
        raise IOError(s)
    if not isSet:
      raise IOError("Barrier could not set point size")
  
  def checkCache(self, cache: Cache):
    if cache == None:
      raise IOError("Cache must be instantiated before initializing Barrier.")
  
  def checkHMax(self):
    if self._hMax is None or self._hMax < self._dtype.zero:
      raise IOError("Barrier: hMax must be positive.")
  
  def clearXFeas(self):
    del self._xFeas
  
  def clearXInf(self):
    del self._xInf
    del self._xIncInf
  
  def getAllPoints(self) -> List[CandidatePoint]:
    allPoints: List[CandidatePoint] = []
    if self._xFeas is None:
      self._xFeas = []
    for cp in self._xFeas:
      allPoints.append(cp)
    if self._xInf is None:
      self._xInf = []
    for cp in self._xInf:
      allPoints.append(cp)
    
    return allPoints
  
  def getFirstPoint(self) -> CandidatePoint:
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
    ind = 0
    for p in cps:
      if p.signature == cp.signature:
        return True, p
      ind+=1
    
    return False, p
  
  def findPoint(self, Point: Point, foundEvalPoint: CandidatePoint) -> bool:
    found: bool = False

    evalPointList: List[CandidatePoint] = self.getAllPoints()
    for cp in evalPointList:
      if cp._n != Point._n:
        raise IOError("Error: Eval points have different dimensions")
      if Point == cp.coordinates:
        foundEvalPoint = copy.deepcopy(cp)
        found = True
        break
    
    return found

  def checkXFeas(self, xFeas: CandidatePoint = None, evalType: EVAL_TYPE = None):
    if xFeas.evaluated:
      self.checkXFeasIsFeas(xFeas=xFeas, evalType=evalType)


  def getAllXFeas(self): 
    return self._xFeas

  def checkXFeasIsFeas(self, xFeas: CandidatePoint=None, evalType: EVAL_TYPE = None):
    if xFeas.evaluated and xFeas.status != DESIGN_STATUS.ERROR:
      h = xFeas.h
      if h is None or h!= 0.0:
        raise IOError(f"Error: Barrier: xFeas' h value must be 0.0, got: {h}")

  
  def checkXInf(self, xInf: CandidatePoint = None, evalType: EVAL_TYPE = None):
    if not xInf.evaluated:
      raise IOError("Barrier: xInf must be evaluated before being set.")
