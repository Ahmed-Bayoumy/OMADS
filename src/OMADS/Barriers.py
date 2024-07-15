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
from dataclasses import dataclass, field
from typing import List, Any, Tuple
from .CandidatePoint import CandidatePoint
from .Point import Point
from ._globals import *
import numpy as np
from .Parameters import Parameters
from .Barrier import BarrierBase
from .Options import Options

@dataclass
class Barrier:
  _params: Parameters = None
  _eval_type: int = 1
  _h_max: float = 0
  _best_feasible: CandidatePoint = None
  _ref: CandidatePoint = None
  _filter: List[CandidatePoint] = None
  _prefilter: int = 0
  _rho_leaps: float = 0.1
  _prim_poll_center: CandidatePoint = None
  _sec_poll_center: CandidatePoint = None
  _peb_changes: int = 0
  _peb_filter_reset: int = 0
  _peb_lop: List[CandidatePoint] = None
  _all_inserted: List[CandidatePoint] = None
  _one_eval_succ: SUCCESS_TYPES = None
  _success: SUCCESS_TYPES = None

  def __init__(self, p: Parameters, eval_type: int = 1):
    self._h_max = p.get_h_max_0()
    self._params = p
    self._eval_type = eval_type


  def insert_feasible(self, x: CandidatePoint) -> SUCCESS_TYPES:
    fx: float
    fx_bf: float
    if self._best_feasible is not None:
      fx_bf = self._best_feasible.fobj
    else:
      self._best_feasible = copy.deepcopy(x)
      return SUCCESS_TYPES.FS
    fx = x.fobj

    if (fx is None or fx_bf is None):
      raise IOError("insert_feasible(): one point has no f value")
    
    if (fx < fx_bf):
      self._best_feasible = copy.deepcopy(x)
      return SUCCESS_TYPES.FS
    
    return SUCCESS_TYPES.US
  
  def filter_insertion(self, x:CandidatePoint) -> bool:
    if not x._is_EB_passed:
      return
    if self._filter is None:
      self._filter = []
      self._filter.append(x)
      insert = True
    else:
      insert = False
      it = 0
      while it != len(self._filter):
        if (x<self._filter[it]):
          del self._filter[it]
          insert = True
          continue
        it += 1
      
      if not insert:
        insert = True
        for it in range(len(self._filter)):
          if self._filter[it].fobj < x.fobj:
            insert = False
            break
      
      if insert:
        self._filter.append(x)
    
    return insert


  def insert_infeasible(self, x: CandidatePoint):
    insert: bool = self.filter_insertion(x=x)
    if not self._ref:
      return SUCCESS_TYPES.PS
    
    hx = x.h
    fx = x.fobj
    hr = self._ref.h
    fr = self._ref.fobj

    # Failure
    if hx > hr or (hx == hr and fx >= fr):
      return SUCCESS_TYPES.US
    
    # Partial success
    if (fx > fr):
      return SUCCESS_TYPES.PS
    
    #  FULL success
    return SUCCESS_TYPES.FS

  def get_best_infeasible(self):
    return self._filter[-1]
  
  def get_best_infeasible_min_viol(self):
    return self._filter[0]
  
  def select_poll_center(self):
    best_infeasible: CandidatePoint = self.get_best_infeasible()
    self._sec_poll_center = None
    if not self._best_feasible and not best_infeasible:
      self._prim_poll_center = None
      return
    if not best_infeasible:
      self._prim_poll_center = self._best_feasible
      return
    
    if not self._best_feasible:
      self._prim_poll_center = best_infeasible
      return
    
    last_poll_center: CandidatePoint = CandidatePoint()
    if self._params.get_barrier_type() == BARRIER_TYPES.PB:
      last_poll_center = self._prim_poll_center
      if best_infeasible.fobj < (self._best_feasible.fobj-self._rho_leaps):
        self._prim_poll_center = best_infeasible
        self._sec_poll_center = self._best_feasible
      else:
        self._prim_poll_center = self._best_feasible
        self._sec_poll_center = best_infeasible

      if last_poll_center is None or self._prim_poll_center != last_poll_center:
        self._rho_leaps += 1

  def set_h_max(self, h_max):
    self._h_max = np.round(h_max, 2)
    if self._filter is not None:
      if self._filter[0].h > self._h_max:
        self._filter = None
        return
    if self._filter is not None:
      it = 0
      while it != len(self._filter):
        if (self._filter[it].h>self._h_max):
          del self._filter[it]
          continue
        it += 1

  def insert(self, x: CandidatePoint):
    """/*---------------------------------------------------------*/
      /*         insertion of a candidate point in the barrier    */
      /*----------------------------------------------------------*/
    """
    if not x.evaluated:
      raise RuntimeError("This point hasn't been evaluated yet and cannot be inserted into the barrier object!")
    
    if (x.status == DESIGN_STATUS.ERROR):
      self._one_eval_succ = SUCCESS_TYPES.US
    if self._all_inserted is None:
      self._all_inserted = []
    self._all_inserted.append(x)
    h = x.h
    if x.status == DESIGN_STATUS.INFEASIBLE and (not x.is_EB_passed or x.h > self._h_max):
      self._one_eval_succ = SUCCESS_TYPES.US
      return
    
    # insert_feasible or insert_infeasible:
    self._one_eval_succ = self.insert_feasible(x) if x.status == DESIGN_STATUS.FEASIBLE else self.insert_infeasible(x)

    if self._success is None or self._one_eval_succ.value > self._success.value:
      self._success = self._one_eval_succ


  def insert_VNS(self):
    pass

  def update_and_reset_success(self):
    """/*------------------------------------------------------------*/
      /*  barrier update: invoked by Evaluator_Control::eval_lop()  */
      /*------------------------------------------------------------*/
    """
    if self._params.get_barrier_type() == BARRIER_TYPES.PB and self._success != SUCCESS_TYPES.US:
      if self._success == SUCCESS_TYPES.PS:
        if self._filter is None:
          raise RuntimeError("filter empty after a partial success")
        it = len(self._filter)-1
        while True:
          if (self._filter[it].h<self._h_max):
            self.set_h_max(self._filter[it].h)
            break
          if it == 0:
            break
            # raise RuntimeError("could not find a filter point with h < h_max after a partial success")
          it -= 1
      if self._filter is not None:
        self._ref = self.get_best_infeasible()
      if self._ref is not None:
        self.set_h_max(self._ref.h)
        if self._ref.status is DESIGN_STATUS.INFEASIBLE:
          self.insert_infeasible(self._ref)
        
        if self._ref.status is DESIGN_STATUS.FEASIBLE:
          self.insert_feasible(self._ref)
        
        if not (self._ref.status is DESIGN_STATUS.INFEASIBLE or self._ref.status is DESIGN_STATUS.INFEASIBLE):
          self.insert(self._ref)

        
    
    # reset success types:
    self._one_eval_succ = self._success = SUCCESS_TYPES.US

    

  def reset(self):
    """/*---------------------------------------------------------*/
      /*                    reset the barrier                    */
      /*---------------------------------------------------------*/"""

    self._prefilter = None
    self._filter = None
    # self._h_max = self._params._h_max_0()
    self._best_feasible   = None
    self._ref             = None
    self._rho_leaps       = 0
    self._poll_center     = None
    self._sec_poll_center = None
    
    # if ( self._peb_changes > 0 ):
    #     self._params.reset_PEB_changes()
    
    self._peb_changes      = 0
    self._peb_filter_reset = 0
    
    self._peb_lop = None
    self._all_inserted = None
    
    self._one_eval_succ = _success = SUCCESS_TYPES.US

@dataclass
class BarrierMO(BarrierBase):
  """ """
  _currentIncumbentFeas: CandidatePoint = None
  _currentIncumbentInf: CandidatePoint = None
  _fixedVariables: CandidatePoint = None
  _xFilterInf: List[CandidatePoint] = None
  _nobj: int = 0
  _bbInputsType: List[VAR_TYPE] = None
  _incumbentSelectionParam: int = 0

  def __init__(self, param: Parameters, options: Options, evalPointList: List[CandidatePoint]= None):
    super(BarrierBase, self).__init__(hMax=param.h_max)

    self._nobj = param.nobj
    self._fixedVariables = param.fixed_variables
    self._bbInputsType = param.var_type
    self._incumbentSelectionParam = param.incumbentincumbentSelectionParam
    self.barrierInitializedFromCache = param.barrierInitializedFromCache
    self._dtype = DType(options.precision)
    self._xFeas = []
    self._xInf = []
    self._xFilterInf = []
    

    self.checkHMax()
    if evalPointList:
      self.init(fixedVariables=self._fixedVariables, evalType=None,evalPointList=evalPointList)


  def init(self, fixedVariables: Point = None, evalType: EVAL_TYPE = None, evalPointList: List[Point] = None):
    updated: bool = self.updateWithPoints(evalPointList)

  def checkMeshParameters(self, x: CandidatePoint = None):
    mesh = copy.deepcopy(x.mesh)


    meshSizeCorrection: int = 0

    if mesh.getdeltaMeshSize().size != x._n:
      meshSizeCorrection = sum(self._fixedVariables.defined)
    
    if (mesh.getdeltaMeshSize().size + meshSizeCorrection != x._n 
        or mesh.getDeltaFrameSize().size + meshSizeCorrection != x._n
        or mesh.getMeshIndex().size + meshSizeCorrection != x._n):
      raise IOError("Error: Mesh parameters dimensions are not compatible with EvalPoint dimension.")
    
    if not mesh.getdeltaMeshSize().is_all_defined():
      raise IOError("Error: some MeshSize components of EvalPoint passed to MO Barrier are not defined.")
    
    if not mesh.getDeltaFrameSize().is_all_defined():
      raise IOError("Error: some FrameSize components of EvalPoint passed to MO Barrier are not defined.")
    
    if not mesh.getMeshIndex().is_all_defined():
      raise IOError("Error: some MeshIndex components of EvalPoint passed to MO Barrier ")


  def updateWithPoints(self, evalPointList: List[CandidatePoint], evalType: EVAL_TYPE = None, keepAllPoints: bool = None, updateInfeasibleIncumbentAndHmax : bool = None):
    updated = False
    updatedFeas = False
    updatedInf = False

    s: str
    xInfTmp: CandidatePoint

    for cp in evalPointList:
      self.checkMeshParameters(cp)

      if not cp.evaluated or cp.status == DESIGN_STATUS.ERROR:
        continue

      if cp.fs.size != self._nobj:
        raise IOError(f"Barrier update: number of objectives is equal to {self._nobj}. Trying to add this point with number of objectives {cp.fs.size}")
      
      updatedFeas = self.updateFeasWithPoint(evalPoint=cp, evalType=evalType, keepAllPoints=keepAllPoints) or updatedFeas

      # // Do separate loop on evalPointList
    # // Second loop update the bestInfeasible.
    # // Use the flag oneFeasEvalFullSuccess.
    # // If the flag is true hmax will not change. A point improving the best infeasible should not replace it.
    for cp in evalPointList:
      if not cp.evaluated or cp.status == DESIGN_STATUS.ERROR:
        continue
      updatedInf = self.updateInfWithPoint(evalPoint=cp, evalType=evalType, keepAllPoints=keepAllPoints, feasHasBeenUpdated=updatedFeas) or updatedInf
    
    updated = updated or updatedFeas or updatedInf

    if updated:
      self.setN()
      self.updateCurrentIncumbents()
    

    
    return updated
  
  def updateCurrentIncumbents(self):
    self.updateCurrentIncumbentFeas()
    self.updateCurrentIncumbentInf()

  def setHMax(self, hMax):
    oldHMax = self._hMax
    self.checkHMax()
    if self._hMax < oldHMax:
      self.updateXInfAndFilterInfAfterHMaxSet()
    self.updateCurrentIncumbentInf()

  def updateXInfAndFilterInfAfterHMaxSet(self):
    """ """
    if len(self._xInf) == 0:
      return
    
    currentInd = 0

    isInXInf = [True] * len(self._xInf)
    for xInf in self._xInf:
      h  = xInf.h
      if h > self._hMax:
        isInXInf[currentInd] = False
      currentInd += 1
  
    currentInd = 0
    for _ in self._xInf:
      if not isInXInf[currentInd]:
        self._xInf.pop(currentInd)
      currentInd += 1
    
    currentInd = 0
    isInXFilterInf = [True] *len(self._xFilterInf)

    for  xFilterInf in self._xFilterInf:
      h = xFilterInf.h
      if h >self._hMax:
        isInXFilterInf[currentInd] = False
      currentInd += 1
    
    currentInd = 0
    for _ in self._xFilterInf:
      if not isInXFilterInf[currentInd]:
        self._xFilterInf.pop(currentInd)
      currentInd += 1
    
    self._xFilterInf = self.non_dominated_sort(self._xFilterInf)

    # // And reinsert potential infeasible non dominated points into the set of infeasible
    # // solutions.
    currentInd = 0
    isInXinf = [False] * len(self._xFilterInf)

    for evalPoint in self._xFilterInf:
      if self.findEvalPoint(self._xFilterInf, evalPoint)[1] == self._xInf[-1]:
        currentIndTmp = 0
        insert = True
        for evalPointInf in self._xFilterInf:
          if currentIndTmp != currentInd:
            compFlag = evalPoint.__comMO__(evalPointInf, True)
            if compFlag == COMPARE_TYPE.DOMINATED:
              insert = False
              break
            elif compFlag == COMPARE_TYPE.DOMINATING:
              isInXInf[currentIndTmp] = False
          currentIndTmp += 1
        isInXInf[currentInd] = insert
      currentInd += 1
    
    for i in range(len(isInXInf)):
      if isInXInf[i]:
        self._xInf.append(self._xFilterInf[i])
    
    self._xInf = self.non_dominated_sort(self._xInf)

    return


  def clearXFeas(self):
    self._xFeas.clear()

    self.updateCurrentIncumbents()
  
  def clearXInf(self):
    self._xInf.clear()
    self._xFilterInf.clear()
    # Update the current incumbent inf. Only the infeasible one depends on XInf (not the case for the feasible one).
    self.updateCurrentIncumbentInf()
  
  def computeSuccessType(self, eval1: CandidatePoint=None, eval2: CandidatePoint=None, hMax: int=np.inf):
    """ """
    success: SUCCESS_TYPES = SUCCESS_TYPES.US
    if eval1 is not None:
      if eval2 is None:
        h = eval1.h
        if h > hMax or h == np.inf:
          success = SUCCESS_TYPES.US
        else:
          if eval1.status == DESIGN_STATUS.FEASIBLE:
            success = SUCCESS_TYPES.FS
          else:
            success = SUCCESS_TYPES.PS
      else:
        if eval1.__dominate__(eval2):
          # // Whether eval1 and eval2 are both feasible, or both
          # // infeasible, dominance means FULL_SUCCESS.
          success = SUCCESS_TYPES.FS
        elif eval1.status == DESIGN_STATUS.FEASIBLE and eval2.status == DESIGN_STATUS.FEASIBLE:
          success = SUCCESS_TYPES.US
        elif eval1.status != DESIGN_STATUS.FEASIBLE and eval2.status != DESIGN_STATUS.FEASIBLE:
          if eval1.h <= hMax and eval1.h < eval2.h and eval1.f > eval2.f:
            success = SUCCESS_TYPES.PS
        else:
          success = SUCCESS_TYPES.US
    return success

  
  def defaultComputeSuccessType(self, evalPoint1: CandidatePoint, evalPoint2: CandidatePoint, hMax: float):
    success: SUCCESS_TYPES = SUCCESS_TYPES.US
    if evalPoint1:
      if evalPoint2:
        h = evalPoint1.h
        if h > hMax or h == np.inf:
          # // Even if evalPoint2 is NULL, this case is still
          # // not a success.
          success = SUCCESS_TYPES.US
        elif evalPoint1.status == DESIGN_STATUS.FEASIBLE:
          success = SUCCESS_TYPES.FS
        else:
          success = self.defaultComputeSuccessType(evalPoint1, evalPoint2, hMax)
    return success

  def getSuccessTypeOfPoints(self, xFeas: CandidatePoint, xInf: CandidatePoint):
    successType = SUCCESS_TYPES.US
    successType2 = SUCCESS_TYPES.US
    newBestFeas: CandidatePoint = CandidatePoint()
    newBestInf: CandidatePoint = CandidatePoint()

    if self._currentIncumbentFeas != None or self._currentIncumbentInf != None:
      if not self._currentIncumbentFeas:
        successType = self.defaultComputeSuccessType(xFeas, self._currentIncumbentFeas, self._hMax)
      if not self._currentIncumbentInf:
        successType = self.defaultComputeSuccessType(xInf, self._currentIncumbentInf, self._hMax)
      if successType2.value > successType.value:
        successType = successType2
      
    return successType

  def checkXFeasIsFeas(self, xFeas: CandidatePoint, evalType: DESIGN_STATUS):
    if xFeas.evaluated and xFeas.status != DESIGN_STATUS.ERROR:
      h = xFeas.h
      if h != 0:
        raise IOError("Error: DMultiMadsBarrier: xFeas' h value must be 0.0")
      if xFeas.fs.size != self._nobj:
        raise IOError("Error: DMultiMadsBarrier: xFeas' F must be of size")



  def getMeshMaxFrameSize(self, pt:CandidatePoint):
    maxRealVal = -1.0
    maxIntegerVal = -1.0

    # Detect if mesh is sub dimension and pt are in full dimension.
    meshIsInSubDimension = False
    mesh = pt.mesh
    if pt.mesh._n < pt._n:
      meshIsInSubDimension = True
    
    shift = 0
    for i in range(pt._n):
      # Do not use access the frame size for fixed variables.
      if meshIsInSubDimension and self._fixedVariables.defined[i]:
        shift += 1
      if self._bbInputsType[i] == VAR_TYPE.REAL:
        maxRealVal = max(maxRealVal, mesh.getDeltaFrameSize(i-shift))
      elif self._bbInputsType[i] == VAR_TYPE.INTEGER:
        maxIntegerVal = max(maxIntegerVal, mesh.getDeltaFrameSize(i-shift))
    if maxRealVal > 0.0:
      return maxRealVal # Some values are real: get norm inf on these values only.
    elif maxIntegerVal > 0.0:
      return maxIntegerVal # No real value but some integer values: get norm inf on these values only
    else:
      return 1.0 # Only binary variables: any elements of the iterate list can be chosen


  def updateCurrentIncumbentFeas(self):
    if len(self._xFeas) == 0:
      self._currentIncumbentFeas = None
      return
    
    if len(self._xFeas) == 1:
      self._currentIncumbentFeas = self._xFeas[0]
      return
    
    maxFrameSizeFeasElts = -1.0

    # Set max frame size of all elements
    for xf in self._xFeas:
      maxFrameSizeFeasElts = max(self.getMeshMaxFrameSize(xf), maxFrameSizeFeasElts)
    
    # Select candidates
    canBeFrameCenter: List[bool] = [False] * len(self._xFeas)
    nbSelectedCandidates = 0

    # see article DMultiMads Algorithm 4.
    for i in range(len(self._xFeas)):
      maxFrameSizeElt = self.getMeshMaxFrameSize(self._xFeas[i])

    if (10**(-float(self._incumbentSelectionParam)) * maxFrameSizeFeasElts) <= maxFrameSizeElt:
      canBeFrameCenter[i] = True
      nbSelectedCandidates += 1

    # Only one point in the barrier.
    if (nbSelectedCandidates == 1):
      for it in range(len(canBeFrameCenter)):
        if canBeFrameCenter[it]:
          break
      if it == len(canBeFrameCenter):
        raise IOError("Error: DMultiMadsBarrier, should not reach this condition")
      else:
        selectedInd = it
        self._currentIncumbentFeas = self._xFeas[selectedInd]
    # Only two points in the barrier.
    elif ((nbSelectedCandidates == 2) and (len(self._xFeas) == 2)):
      eval1 = self._xFeas[0]
      eval2 = self._xFeas[1]

      objv1 = eval1.fs
      objv2 = eval2.fs

      if max(np.abs(objv1.coordinates)) > max(np.abs(objv2.coordinates)):
        self._currentIncumbentFeas = self._xFeas[0]
      else:
        self._currentIncumbentFeas = self._xFeas[1]
      
    # More than three points in the barrier.
    else:
      # First case: biobjective optimization. Points are already ranked by lexicographic order.
      if self._nobj:
        currentBestInd = 0
        maxGap = -1.0
        currentGap: float
        for obj in range(self._nobj):
          # Get extreme values value according to one objective
          fmin: float = self._xFeas[0].fs[obj]
          fmax: float = self._xFeas[len(self._xFeas)-1].fs[obj]
          # In this case, it means all elements of _xFeas are equal (return the first one)
          if fmin == fmax:
            break

          # Intermediate points
          for i in range(1, self._xFeas-1):
            currentGap = self._xFeas[i+1].fs[obj]-self._xFeas[i-1].fs[obj]
            self._xFeas[i-1].fs[obj]
            currentGap /= (fmax-fmin)
            if (canBeFrameCenter[i] and currentGap >= maxGap):
              maxGap = currentGap
              currentBestInd = i

          
          # Extreme points
          currentGap = 2 * (self._xFeas[len(self._xFeas)-1]).fs[obj] - (self._xFeas[len(self._xFeas)-2]).fs[obj]
          currentGap /= (fmax - fmin)
          if canBeFrameCenter[len(self._xFeas)-1] and currentGap >= maxGap:
            maxGap = currentGap
            currentBestInd = len(self._xFeas)-1
          
          currentGap = 2 * (self._xFeas[1]).fs[obj] - (self._xFeas[0]).fs[obj]
          currentGap /= (fmax -fmin)

          if canBeFrameCenter[0] and currentGap >= maxGap:
            maxGap = currentGap
            currentBestInd = 0
        self._currentIncumbentFeas = self._xFeas[currentBestInd]

      # // More than 2 objectives
      else:
        tmpXFeasPInd: List[Tuple[CandidatePoint, int]] = [(CandidatePoint(), 0)]*len(self._xFeas)
        for i in range(len(tmpXFeasPInd)):
          tmpXFeasPInd[i] = (self._xFeas[i], i)
        currentBestInd = 0
        maxGap = -1.0
        currentGap: float

        for obj in range(self._nobj):
          # Sort elements of tmpXFeasPInd according to objective obj (in ascending order)
          tmpXFeasPInd = sorted(tmpXFeasPInd, key=lambda x: x[0].fs[obj])

          # Get extreme values value according to one objective
          fmin = tmpXFeasPInd[0][0].fs[obj]
          fmax = tmpXFeasPInd[len(tmpXFeasPInd)-1][0].fs[obj]

          # Can happen for exemple when we have several minima or for more than three objectives
          if fmin == fmax:
            fmin = 0.
            fmax = 1.
          
          # Intermediate points
          for i in range(1, len(tmpXFeasPInd)-1):
            currentGap = tmpXFeasPInd[i+1][0].fs[obj]-tmpXFeasPInd[i-1][0].fs[obj]
            currentGap /= (fmax - fmin)
            if canBeFrameCenter[tmpXFeasPInd[i][1]] and currentGap >= maxGap:
              maxGap = currentGap
              currentBestInd = tmpXFeasPInd[i][1]
          
          # Extreme points
          currentGap = 2*(tmpXFeasPInd[len(tmpXFeasPInd)-1][0].fs[obj]) - tmpXFeasPInd[len(tmpXFeasPInd)-2][0].fs[obj]
          currentGap /= (fmax - fmin)

          if (canBeFrameCenter[tmpXFeasPInd[len(tmpXFeasPInd)-1][1]] and currentGap > maxGap):
            maxGap = currentGap
            currentBestInd = tmpXFeasPInd[len(tmpXFeasPInd)-1][1]
          
          currentGap = 2 * tmpXFeasPInd[1][0].fs[obj] - tmpXFeasPInd[0][0].fs[obj]
          currentGap /= (fmax -fmin)

          if (canBeFrameCenter[tmpXFeasPInd[0][1]] and currentGap > maxGap):
            maxGap = currentGap
            currentBestInd = tmpXFeasPInd[0][1]
        self._currentIncumbentFeas = self._xFeas[currentBestInd]

  def updateCurrentIncumbentInf(self):
    self._currentIncumbentInf = None
    if len(self._xFeas) > 0 and len(self._xInf) > 0:
      # // Get the infeasible solution with maximum dominance move below the _hMax threshold,
      # // according to the set of best feasible incumbent solutions.
      currentInd = 0
      maxDomMove = -np.inf

      for j in range(len(self._xInf)):
        # // Compute dominance move
        # // = min \sum_{1}^m max(fi(y) - fi(x), 0)
        # //   y \in Fk

        tmpDomMove = np.inf
        evalInf = self._xInf[j]
        h = evalInf.h

        if h <= self._hMax:
          for xFeas in self._xFeas:
            sumVal = 0.
            evalFeas = xFeas
            for i in range(self._nobj):
              sumVal += max(evalFeas.fs[i]-evalInf.fs[i], 0)
            if tmpDomMove > sumVal:
              tmpDomMove = sumVal
          
          # Get the maximum dominance move index
          if maxDomMove < tmpDomMove:
            maxDomMove = tmpDomMove
            currentInd = j
      
      # // In this case, all infeasible solutions are "dominated" in terms of fvalues
      # // by at least one element of Fk
      if maxDomMove == 0.:
        # // In this case, get the infeasible solution below the _hMax threshold which has
        # // minimal dominance move, when considered a maximization problem.
        minDomMove = np.inf
        currentInd = 0
        for j in range(len(self._xInf)):
          # // Compute dominance move
          # // = min \sum_{1}^m max(fi(x) - fi(y), 0)
          # //   y \in Fk
          tmpDomMove = np.inf
          evalInf = self._xInf[j]
          h = evalInf.h
          if h<= self._hMax:
            for xFeas in self._xFeas:
              sumVal = 0.
              evalFeas = xFeas
              # Compute \sum_{1}^m max (fi(x) - fi(y), 0)
              for i in range(self._nobj):
                sumVal += max(evalInf.fs[i] - evalFeas.fs[i], 0.)
              if tmpDomMove > sumVal:
                tmpDomMove = sumVal
            
            # Get the minimal dominance move index
            if minDomMove > tmpDomMove:
              minDomMove = tmpDomMove
              currentInd = j
      self._currentIncumbentInf = self._xInf[currentInd]
    else:
      self._currentIncumbentInf = self.getFirstXIncInfNoXFeas()
            

  def getXInfMinH(self):
    indXInfMinH = 0
    hMinVal = np.inf

    for i in range(len(self._xInf)):
      eval = self._xInf[i]
      h = eval.h

      # // By definition, all elements of _xInf or _xFilterInf have a well-defined
      # // h value. So, no need to check.

      if h <hMinVal:
        hMinVal = h
        indXInfMinH = i
    return self._xInf[indXInfMinH]

  def getFirstXIncInfNoXFeas(self):
    """ """
    xInf = None
    if len(self._xFilterInf) == 0:
      return xInf
    
    # Select candidates
    minFrameSizeInfElts: float = self.getMeshMaxFrameSize(self.getXInfMinH())
    canBeFrameCenter: List[bool] = [False] * len(self._xInf)
    nbSelectedCandidates = 0

    for i in range(len(self._xInf)):
      maxFrameSizeElt = self.getMeshMaxFrameSize(self._xInf[i])
      if minFrameSizeInfElts <= maxFrameSizeElt:
        canBeFrameCenter[i] = True
        nbSelectedCandidates += 1
    
    # The selection must always work
    if nbSelectedCandidates == 0:
      xInf = self._xInf[0]
    elif nbSelectedCandidates == 1:
      for it in range(len(canBeFrameCenter)):
        if canBeFrameCenter[it]:
          break
      if it == len(canBeFrameCenter):
        raise IOError("Error: DMultiMadsBarrier, should not reach this condition")
      else:
        selectedInd = it
        xInf = self._xInf[selectedInd]
    elif ((nbSelectedCandidates == 2) and (len(self._xInf) == 2)):
      eval1 = self._xInf[0]
      eval2 = self._xInf[1]

      objv1 = eval1.fs
      objv2 = eval2.fs

      if max(np.abs(objv1.coordinates)) > max(np.abs(objv2.coordinates)):
        xInf = self._xInf[0]
      else:
        xInf = self._xInf[1]
    else:
      if self._nobj == 2:
        currentBestInd = 0
        maxGap = -1.
        currentGap: float

        for obj in range(self._nobj):
          # Get extreme values value according to one objective
          fmin: float = self._xInf[0].fs[obj]
          fmax: float = self._xInf[len(self._xInf)-1].fs[obj]

          # In this case, it means all elements of _xFeas are equal (return the first one)
          if fmin == fmax:
            break

          # Intermediate points
          for i in range(1, self._xInf-1):
            currentGap = self._xInf[i+1].fs[obj]-self._xInf[i-1].fs[obj]
            self._xInf[i-1].fs[obj]
            currentGap /= (fmax-fmin)
            if (canBeFrameCenter[i] and currentGap >= maxGap):
              maxGap = currentGap
              currentBestInd = i
          
          # Extreme points
          currentGap = 2 * (self._xInf[len(self._xInf)-1]).fs[obj] - (self._xInf[len(self._xInf)-2]).fs[obj]
          currentGap /= (fmax - fmin)
          if canBeFrameCenter[len(self._xInf)-1] and currentGap > maxGap:
            maxGap = currentGap
            currentBestInd = len(self._xInf)-1
          
          currentGap = 2 * (self._xInf[1]).fs[obj] - (self._xInf[0]).fs[obj]
          currentGap /= (fmax -fmin)

          if canBeFrameCenter[0] and currentGap > maxGap:
            maxGap = currentGap
            currentBestInd = 0
        xInf = self._xInf[currentBestInd]
      # // More than 2 objectives
      else:
        tmpXInfPInd: List[Tuple[CandidatePoint, int]] = [(CandidatePoint(), 0)]*len(self._xInf)
        for i in range(len(tmpXInfPInd)):
          tmpXInfPInd[i] = (self._xInf[i], i)
        currentBestInd = 0
        maxGap = -1.0
        currentGap: float 

        for obj in range(self._nobj):
          # Sort elements of tmpXFeasPInd according to objective obj (in ascending order)
          tmpXInfPInd = sorted(tmpXInfPInd, key=lambda x: x[0].fs[obj])

          # Get extreme values value according to one objective
          fmin = tmpXInfPInd[0][0].fs[obj]
          fmax = tmpXInfPInd[len(tmpXInfPInd)-1][0].fs[obj]

          # Can happen for exemple when we have several minima or for more than three objectives
          if fmin == fmax:
            fmin = 0.
            fmax = 1.
          
          # Intermediate points
          for i in range(1, len(tmpXInfPInd)-1):
            currentGap = tmpXInfPInd[i+1][0].fs[obj]-tmpXInfPInd[i-1][0].fs[obj]
            currentGap /= (fmax - fmin)
            if canBeFrameCenter[tmpXInfPInd[i][1]] and currentGap >= maxGap:
              maxGap = currentGap
              currentBestInd = tmpXInfPInd[i][1]
          
          # Extreme points
          currentGap = 2*(tmpXInfPInd[len(tmpXInfPInd)-1][0].fs[obj]) - tmpXInfPInd[len(tmpXInfPInd)-2][0].fs[obj]
          currentGap /= (fmax - fmin)

          if (canBeFrameCenter[tmpXInfPInd[len(tmpXInfPInd)-1][1]] and currentGap > maxGap):
            maxGap = currentGap
            currentBestInd = tmpXInfPInd[len(tmpXInfPInd)-1][1]
          
          currentGap = 2 * tmpXInfPInd[1][0].fs[obj] - tmpXInfPInd[0][0].fs[obj]
          currentGap /= (fmax -fmin)

          if (canBeFrameCenter[tmpXInfPInd[0][1]] and currentGap > maxGap):
            maxGap = currentGap
            currentBestInd = tmpXInfPInd[0][1]
        xInf = self._xInf[currentBestInd]

    return xInf
  
  def updateInfWithPoint(self, evalPoint: CandidatePoint = None, evalType: EVAL_TYPE = None, keepAllPoints: bool = None, feasHasBeenUpdated: bool = False):
    updated = False

    if evalPoint.evaluated and evalPoint.status != DESIGN_STATUS.FEASIBLE:
      s: str
      h = evalPoint.h

      if h == np.inf or (self._hMax < np.inf and h > self._hMax):
        return False
      
      if self._xInf is None:
        self._xInf = []
      
      if self._xFilterInf is None:
        self._xFilterInf = []
      
      if len(self._xInf) <= 0:
        self._xInf.append(evalPoint)
        self._xFilterInf.append(evalPoint)
        self._currentIncumbentInf = self._xInf[0]
        updated = True
      else:
        insert = True
        isInXinfFilter: List[bool] = [True] * len(self._xFilterInf)
        currentInd = 0
        for xFilterInf in self._xFilterInf:
          compFlag = evalPoint.__comMO__(xFilterInf)
          if compFlag == COMPARE_TYPE.DOMINATED:
            insert = False
            break
          elif compFlag == COMPARE_TYPE.DOMINATING:
            updated = True
            isInXinfFilter[currentInd] = False
          elif compFlag == COMPARE_TYPE.EQUAL:
            if (not keepAllPoints):
              insert = False
              break
            if self.findEvalPoint(self._xFilterInf, evalPoint)[0]:
              insert = False
            else:
              updated = True
              break
          currentInd += 1
        
        if insert:
          indices_to_remove = []
          for i in range(len(self._xFilterInf)):
            if not isInXinfFilter[i]:
              indices_to_remove.append(i)
          
          self._xFilterInf.append(evalPoint)
          
          for index in sorted(indices_to_remove, reverse=True):
            del self._xFilterInf[index]
          
          self._xFilterInf = self.non_dominated_sort(self._xFilterInf)
          
          insert = True
          currentInd = 0
          isInXinf = [True * self._xInf]

          for xInf in self._xInf:
            compFlag = evalPoint.__comMO__(xInf, True)
            if compFlag == COMPARE_TYPE.DOMINATED:
              insert = False
              break
            elif compFlag == COMPARE_TYPE.DOMINATING or evalPoint.__comMO__(xInf):
              updated = True
              isInXinf[currentInd] = False
            currentInd += 1
          
          if insert:
            indices_to_remove = []
            for i in range(len(self._xInf)):
              if not isInXinf[i]:
                indices_to_remove.append(i)
            updated = True
            self._xInf.append(evalPoint)
            
            for index in sorted(indices_to_remove, reverse=True):
              del self._xInf[index]
            
            self._xInf = self.non_dominated_sort(self._xInf)

    return updated

            
      


  def non_dominated_sort(self, points: List[CandidatePoint] = None):
    """ Perform biobjective nondominated sorting """
    fronts = [[]]  # List to store different fronts
    dominated_count = [0] * len(points)  # Array to count number of points dominating each point
    
    for i, p in enumerate(points):
      for j, q in enumerate(points):
        if i != j and p.__comMO__(q) == COMPARE_TYPE.DOMINATED:
          dominated_count[i] += 1
    
      if dominated_count[i] == 0:
        fronts[0].append(p)
    
    # Sort each front lexicographically
    for front in fronts:
      front.sort()
    
    # Flatten the fronts into a single list
    sorted_points = [point for front in fronts for point in front]
    
    return sorted_points

  def updateFeasWithPoint(self, evalPoint: CandidatePoint = None, evalType: EVAL_TYPE = None, keepAllPoints: bool = None):
    updated = False

    if evalPoint.evaluated and evalPoint.status == DESIGN_STATUS.FEASIBLE:
      if evalPoint.fs.size != self._nobj:
        raise IOError(f"Barrier update: number of objectives is equal to {self._nobj}. Trying to add this point with number of objectives {evalPoint.fs.size}")
      
      if self._xFeas is None:
        self._xFeas = []
      
      if len(self._xFeas) == 0:
        self._xFeas.append(evalPoint)
        updated = True
        self._currentIncumbentFeas = self._xFeas[0]
      else:
        insert = True
        keepInXFeas = [True] * len(self._xFeas)
        currentInd = 0
        for xf in self._xFeas:
          compFlag: COMPARE_TYPE = evalPoint.__comMO__(xf)
          if compFlag == COMPARE_TYPE.DOMINATED:
            insert = False
            break
          elif compFlag == COMPARE_TYPE.DOMINATING:
            updated = True
            keepInXFeas[currentInd] = False
          elif compFlag == COMPARE_TYPE.EQUAL:
            if not keepAllPoints:
              insert = False
              break

            if self.findEvalPoint(self._xFeas, evalPoint)[0]:
              insert = False
            else:
              updated = True
            break
          currentInd += 1
        if insert:
          currentInd = 0
          for cp in self._xFeas:
            if cp.__comMO__(evalPoint) == COMPARE_TYPE.DOMINATED:
              self._xFeas.pop(currentInd)
            currentInd += 1
          updated = True
          dir = copy.deepcopy(evalPoint.direction)
          if dir is not None:
            evalPoint.mesh.enlargeDeltaFrameSize(direction=dir)
          
          self._xFeas.append(evalPoint)

          # Sort according to lexicographic order.
          self._xFeas = self.non_dominated_sort(self._xFeas)

    return updated






