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
from typing import List, Tuple, Optional
from .CandidatePoint import CandidatePoint
from .Point import Point
from ._globals import DType, VAR_TYPE, BARRIER_TYPES, SUCCESS_TYPES, DESIGN_STATUS, EVAL_TYPE, COMPARE_TYPE
import numpy as np
from .Parameters import Parameters
from .Barrier import BarrierBase
from .Options import Options

@dataclass
class Barrier:
  _params: Optional[Parameters] = None
  _eval_type: int = 1
  _h_max: float = 0
  _best_feasible: Optional[CandidatePoint] = None
  _ref: Optional[CandidatePoint] = None
  _filter: Optional[List[CandidatePoint]] = None
  _prefilter: int = 0
  _rho_leaps: float = 0.1
  _prim_poll_center: Optional[CandidatePoint] = None
  _sec_poll_center: Optional[CandidatePoint] = None
  _peb_changes: int = 0
  _peb_filter_reset: int = 0
  _peb_lop: Optional[List[CandidatePoint]] = None
  _all_inserted: Optional[List[CandidatePoint]] = None
  _one_eval_succ: Optional[SUCCESS_TYPES] = None
  _success: Optional[SUCCESS_TYPES] = None

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
      return False
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
    _ = self.filter_insertion(x=x)
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
    if self._filter:
      return self._filter[-1]
    else:
      return None
  
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
    
    last_poll_center: Optional[CandidatePoint] = None
    if self._params.get_barrier_type() == BARRIER_TYPES.PB:
      last_poll_center = self._prim_poll_center
      if best_infeasible.fobj[0] < (self._best_feasible.fobj[0]-self._rho_leaps):
        self._prim_poll_center = best_infeasible
        self._sec_poll_center = self._best_feasible
      else:
        self._prim_poll_center = self._best_feasible
        self._sec_poll_center = best_infeasible

      if last_poll_center is None or self._prim_poll_center != last_poll_center:
        self._rho_leaps += 1

  def set_h_max(self, h_max):
    self._h_max = np.round(h_max, 2)
    if self._filter is not None and self._filter[0].h > self._h_max:
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
    if x.status == DESIGN_STATUS.INFEASIBLE and (not x.is_EB_passed or x.h > self._h_max):
      self._one_eval_succ = SUCCESS_TYPES.US
      return
    
    # insert_feasible or insert_infeasible:
    self._one_eval_succ = self.insert_feasible(x) if x.status == DESIGN_STATUS.FEASIBLE else self.insert_infeasible(x)

    if self._success is None or self._one_eval_succ.value > self._success.value:
      self._success = self._one_eval_succ


  def insert_vns(self):
    """ Not required here
    """
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
          it -= 1
      if self._filter is not None:
        self._ref = self.get_best_infeasible()
      if self._ref is not None:
        self.set_h_max(self._ref.h)
        if self._ref.status is DESIGN_STATUS.INFEASIBLE:
          self.insert_infeasible(self._ref)
        
        if self._ref.status is DESIGN_STATUS.FEASIBLE:
          self.insert_feasible(self._ref)
        
        if not (self._ref.status is DESIGN_STATUS.INFEASIBLE or self._ref.status is DESIGN_STATUS.FEASIBLE):
          self.insert(self._ref)

        
    
    # reset success types:
    self._one_eval_succ = self._success = SUCCESS_TYPES.US

    

  def reset(self):
    """/*---------------------------------------------------------*/
      /*                    reset the barrier                    */
      /*---------------------------------------------------------*/"""

    self._prefilter = None
    self._filter = None
    self._best_feasible   = None
    self._ref             = None
    self._rho_leaps       = 0
    self._poll_center     = None
    self._sec_poll_center = None
    
    #     self._params.reset_PEB_changes()
    
    self._peb_changes      = 0
    self._peb_filter_reset = 0
    
    self._peb_lop = None
    self._all_inserted = None
    
    self._one_eval_succ = _success = SUCCESS_TYPES.US

@dataclass
class BarrierMO(BarrierBase):
  """ """
  _currentIncumbentFeas: Optional[CandidatePoint] = None
  _currentIncumbentInf: Optional[CandidatePoint] = None
  _fixedVariables: Optional[CandidatePoint] = None
  _xFilterInf: Optional[List[CandidatePoint]] = None
  _nobj: int = 0
  _bbInputsType: Optional[List[VAR_TYPE]] = None
  _incumbentSelectionParam: int = 0

  def __init__(self, param: Parameters, options: Options, eval_point_list: Optional[List[CandidatePoint]]= None):
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
    if eval_point_list:
      self.init(eval_point_list=eval_point_list)


  def init(self, eval_point_list: Optional[List[Point]] = None):
    _, _, _ = self.updateWithPoints(eval_point_list)

  def checkMeshParameters(self, x: CandidatePoint = None):
    mesh = copy.deepcopy(x.mesh)


    mesh_size_correction: int = 0

    if mesh.getdeltaMeshSize().size != x._n:
      mesh_size_correction = sum(self._fixedVariables.defined)
    
    if (mesh.getdeltaMeshSize().size + mesh_size_correction != x._n 
        or mesh.getDeltaFrameSize().size + mesh_size_correction != x._n
        or mesh.getMeshIndex().size + mesh_size_correction != x._n):
      raise IOError("Error: Mesh parameters dimensions are not compatible with EvalPoint dimension.")
    
    if not mesh.getdeltaMeshSize().is_all_defined():
      raise IOError("Error: some MeshSize components of EvalPoint passed to MO Barrier are not defined.")
    
    if not mesh.getDeltaFrameSize().is_all_defined():
      raise IOError("Error: some FrameSize components of EvalPoint passed to MO Barrier are not defined.")
    
    if not mesh.getMeshIndex().is_all_defined():
      raise IOError("Error: some MeshIndex components of EvalPoint passed to MO Barrier ")


  def updateWithPoints(self, eval_point_list: List[CandidatePoint]=None, keep_all_points: bool = None):
    updated = False
    updated_feas = False
    updated_inf = False

    for cp in eval_point_list:
      self.checkMeshParameters(cp)

      if not cp.evaluated or cp.status == DESIGN_STATUS.ERROR:
        continue

      if cp.fs.size != self._nobj:
        raise IOError(f"Barrier update: number of objectives is equal to {self._nobj}. Trying to add this point with number of objectives {cp.fs.size}")
      
      updated_feas = self.updateFeasWithPoint(eval_point=cp, keep_all_points=keep_all_points) or updated_feas

      # // Do separate loop on evalPointList
    # // Second loop update the bestInfeasible.
    # // Use the flag oneFeasEvalFullSuccess.
    # // If the flag is true hmax will not change. A point improving the best infeasible should not replace it.
    for cp in eval_point_list:
      if not cp.evaluated or cp.status == DESIGN_STATUS.ERROR:
        continue
      updated_inf = self.updateInfWithPoint(eval_point=cp, keep_all_points=keep_all_points) or updated_inf
    
    updated = updated or updated_feas or updated_inf

    if updated:
      self.setN()
      self.updateCurrentIncumbents()
    

    
    return updated, updated_feas, updated_inf
  
  def updateCurrentIncumbents(self):
    self.updateCurrentIncumbentFeas()
    self.updateCurrentIncumbentInf()

  def setHMax(self, h_max=np.inf):
    old_h_max = self._h_max
    self._h_max = h_max
    self.checkHMax()
    if h_max < old_h_max:
      self.updateXInfAndFilterInfAfterHMaxSet()
    self.updateCurrentIncumbentInf()

  def updateXInfAndFilterInfAfterHMaxSet(self):
    """ """
    if len(self._xInf) == 0:
      return
    
    current_ind = 0

    is_in_x_inf = [True] * len(self._xInf)
    for x_inf in self._xInf:
      h  = x_inf.h
      if h > self._h_max:
        is_in_x_inf[current_ind] = False
      current_ind += 1
  
    current_ind = 0
    for _ in self._xInf:
      if not is_in_x_inf[current_ind]:
        self._xInf.pop(current_ind)
      current_ind += 1
    
    current_ind = 0
    is_in_x_filter_inf = [True] *len(self._xFilterInf)

    for  x_filter_inf in self._xFilterInf:
      h = x_filter_inf.h
      if h >self._h_max:
        is_in_x_filter_inf[current_ind] = False
      current_ind += 1
    
    current_ind = 0
    for _ in self._xFilterInf:
      if not is_in_x_filter_inf[current_ind]:
        self._xFilterInf.pop(current_ind)
      current_ind += 1
    
    self._xFilterInf = self.non_dominated_sort(self._xFilterInf)

    # // And reinsert potential infeasible non dominated points into the set of infeasible
    # // solutions.
    current_ind = 0

    for eval_point in self._xFilterInf:
      if len(self._xInf) > 0 and self.findEvalPoint(self._xFilterInf, eval_point)[1] == self._xInf[-1]:
        current_ind_tmp = 0
        insert = True
        for eval_point_inf in self._xFilterInf:
          if current_ind_tmp != current_ind:
            comp_flag = eval_point.__comp_mo__(eval_point_inf, True)
            if comp_flag == COMPARE_TYPE.DOMINATED:
              insert = False
              break
            elif comp_flag == COMPARE_TYPE.DOMINATING:
              is_in_x_inf[current_ind_tmp] = False
          current_ind_tmp += 1
        is_in_x_inf[current_ind] = insert
      current_ind += 1
    
    for i in range(len(is_in_x_inf)):
      if is_in_x_inf[i]:
        self._xInf.append(self._xFilterInf[i])
    
    self._xInf = self.non_dominated_sort(self._xInf)

  def clearXFeas(self):
    self._xFeas.clear()

    self.updateCurrentIncumbents()
  
  def clearXInf(self):
    self._xInf.clear()
    self._xFilterInf.clear()
    # Update the current incumbent inf. Only the infeasible one depends on XInf (not the case for the feasible one).
    self.updateCurrentIncumbentInf()
  
  def computeSuccessType(self, eval1: CandidatePoint=None, eval2: CandidatePoint=None, h_max: int=np.inf):
    """ """
    success: SUCCESS_TYPES = SUCCESS_TYPES.US
    if eval1 is not None:
      if eval2 is None:
        h = eval1.h
        if h > h_max or h == np.inf:
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
          if eval1.h <= h_max and eval1.h < eval2.h and eval1.f > eval2.f:
            success = SUCCESS_TYPES.PS
        else:
          success = SUCCESS_TYPES.US
    return success

  
  def defaultComputeSuccessType(self, eval_point1: CandidatePoint, eval_point2: CandidatePoint, h_max: float):
    success: SUCCESS_TYPES = SUCCESS_TYPES.US
    if eval_point1 and eval_point2:
      h = eval_point1.h
      if h > h_max or h == np.inf:
        # // Even if evalPoint2 is NULL, this case is still
        # // not a success.
        success = SUCCESS_TYPES.US
      elif eval_point1.status == DESIGN_STATUS.FEASIBLE:
        success = SUCCESS_TYPES.FS
      else:
        success = self.defaultComputeSuccessType(eval_point1, eval_point2, h_max)
    return success

  def getSuccessTypeOfPoints(self, x_feas: CandidatePoint = None, x_inf: CandidatePoint = None):
    success_type = SUCCESS_TYPES.US
    success_type2 = SUCCESS_TYPES.US

    if self._currentIncumbentFeas != None or self._currentIncumbentInf != None:
      if not self._currentIncumbentFeas:
        success_type = self.defaultComputeSuccessType(x_feas, self._currentIncumbentFeas, self._h_max)
      if not self._currentIncumbentInf:
        success_type = self.defaultComputeSuccessType(x_inf, self._currentIncumbentInf, self._h_max)
      if success_type2.value > success_type.value:
        success_type = success_type2
      
    return success_type

  def checkXFeasIsFeas(self, x_feas: CandidatePoint = None, eval_type: DESIGN_STATUS = None):
    if x_feas.evaluated and x_feas.status != DESIGN_STATUS.ERROR:
      h = x_feas.h
      if h != 0:
        raise IOError("Error: DMultiMadsBarrier: xFeas' h value must be 0.0")
      if x_feas.fs.size != self._nobj:
        raise IOError("Error: DMultiMadsBarrier: xFeas' F must be of size")



  def getMeshMaxFrameSize(self, pt:CandidatePoint):
    max_real_val = -1.0
    max_integer_val = -1.0

    # Detect if mesh is sub dimension and pt are in full dimension.
    mesh_is_in_subdimension = False
    mesh = pt.mesh
    if pt.mesh._n < pt._n:
      mesh_is_in_subdimension = True
    
    shift = 0
    for i in range(pt._n):
      # Do not use access the frame size for fixed variables.
      if mesh_is_in_subdimension and self._fixedVariables.defined[i]:
        shift += 1
      if self._bbInputsType[i] == VAR_TYPE.REAL:
        max_real_val = max(max_real_val, mesh.getDeltaFrameSize(i-shift))
      elif self._bbInputsType[i] == VAR_TYPE.INTEGER:
        max_integer_val = max(max_integer_val, mesh.getDeltaFrameSize(i-shift))
    if max_real_val > 0.0:
      return max_real_val # Some values are real: get norm inf on these values only.
    elif max_integer_val > 0.0:
      return max_integer_val # No real value but some integer values: get norm inf on these values only
    else:
      return 1.0 # Only binary variables: any elements of the iterate list can be chosen


  def updateCurrentIncumbentFeas(self):
    if len(self._xFeas) == 0:
      self._currentIncumbentFeas = None
      return
    
    if len(self._xFeas) == 1:
      self._currentIncumbentFeas = self._xFeas[0]
      return
    
    max_frame_size_feas_elts = -1.0

    # Set max frame size of all elements
    for xf in self._xFeas:
      max_frame_size_feas_elts = max(self.getMeshMaxFrameSize(xf), max_frame_size_feas_elts)
    
    # Select candidates
    can_be_frame_center: List[bool] = [False] * len(self._xFeas)
    nb_selected_candidates = 0

    # see article DMultiMads Algorithm 4.
    for i in range(len(self._xFeas)):
      max_frame_size_elt = self.getMeshMaxFrameSize(self._xFeas[i])

    if (10**(-float(self._incumbentSelectionParam)) * max_frame_size_feas_elts) <= max_frame_size_elt:
      can_be_frame_center[i] = True
      nb_selected_candidates += 1

    # Only one point in the barrier.
    if (nb_selected_candidates == 1):
      for it in range(len(can_be_frame_center)):
        if can_be_frame_center[it]:
          break
      if it == len(can_be_frame_center):
        raise IOError("Error: DMultiMadsBarrier, should not reach this condition")
      else:
        selected_ind = it
        self._currentIncumbentFeas = self._xFeas[selected_ind]
    # Only two points in the barrier.
    elif ((nb_selected_candidates == 2) and (len(self._xFeas) == 2)):
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
        current_best_ind = 0
        max_gap = -1.0
        current_gap: float
        for obj in range(self._nobj):
          # Get extreme values value according to one objective
          fmin: float = self._xFeas[0].fs[obj]
          fmax: float = self._xFeas[len(self._xFeas)-1].fs[obj]
          # In this case, it means all elements of _xFeas are equal (return the first one)
          if fmin == fmax:
            break

          # Intermediate points
          for i in range(1, self._xFeas-1):
            current_gap = self._xFeas[i+1].fs[obj]-self._xFeas[i-1].fs[obj]
            self._xFeas[i-1].fs[obj]
            current_gap /= (fmax-fmin)
            if (can_be_frame_center[i] and current_gap >= max_gap):
              max_gap = current_gap
              current_best_ind = i

          
          # Extreme points
          current_gap = 2 * (self._xFeas[len(self._xFeas)-1]).fs[obj] - (self._xFeas[len(self._xFeas)-2]).fs[obj]
          current_gap /= (fmax - fmin)
          if can_be_frame_center[len(self._xFeas)-1] and current_gap >= max_gap:
            max_gap = current_gap
            current_best_ind = len(self._xFeas)-1
          
          current_gap = 2 * (self._xFeas[1]).fs[obj] - (self._xFeas[0]).fs[obj]
          current_gap /= (fmax -fmin)

          if can_be_frame_center[0] and current_gap >= max_gap:
            max_gap = current_gap
            current_best_ind = 0
        self._currentIncumbentFeas = self._xFeas[current_best_ind]

      # // More than 2 objectives
      else:
        tmp_x_feas_p_ind: List[Tuple[CandidatePoint, int]] = [(CandidatePoint(), 0)]*len(self._xFeas)
        for i in range(len(tmp_x_feas_p_ind)):
          tmp_x_feas_p_ind[i] = (self._xFeas[i], i)
        current_best_ind = 0
        max_gap = -1.0
        current_gap: float
        
        for obj in range(self._nobj):
          # Sort elements of tmpXFeasPInd according to objective obj (in ascending order)
          tmp_x_feas_p_ind = sorted(tmp_x_feas_p_ind, key=lambda x: x[0].fs[obj])

          # Get extreme values value according to one objective
          fmin = tmp_x_feas_p_ind[0][0].fs[obj]
          fmax = tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-1][0].fs[obj]

          # Can happen for exemple when we have several minima or for more than three objectives
          if fmin == fmax:
            fmin = 0.
            fmax = 1.
          
          # Intermediate points
          for i in range(1, len(tmp_x_feas_p_ind)-1):
            current_gap = tmp_x_feas_p_ind[i+1][0].fs[obj]-tmp_x_feas_p_ind[i-1][0].fs[obj]
            current_gap /= (fmax - fmin)
            if can_be_frame_center[tmp_x_feas_p_ind[i][1]] and current_gap >= max_gap:
              max_gap = current_gap
              current_best_ind = tmp_x_feas_p_ind[i][1]
          
          # Extreme points
          current_gap = 2*(tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-1][0].fs[obj]) - tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-2][0].fs[obj]
          current_gap /= (fmax - fmin)

          if (can_be_frame_center[tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-1][1]] and current_gap > max_gap):
            max_gap = current_gap
            current_best_ind = tmp_x_feas_p_ind[len(tmp_x_feas_p_ind)-1][1]
          
          current_gap = 2 * tmp_x_feas_p_ind[1][0].fs[obj] - tmp_x_feas_p_ind[0][0].fs[obj]
          current_gap /= (fmax -fmin)

          if (can_be_frame_center[tmp_x_feas_p_ind[0][1]] and current_gap > max_gap):
            max_gap = current_gap
            current_best_ind = tmp_x_feas_p_ind[0][1]
        self._currentIncumbentFeas = self._xFeas[current_best_ind]

  def updateCurrentIncumbentInf(self):
    self._currentIncumbentInf = None
    if len(self._xFeas) > 0 and len(self._xInf) > 0:
      # // Get the infeasible solution with maximum dominance move below the _hMax threshold,
      # // according to the set of best feasible incumbent solutions.
      current_ind = 0
      max_dom_move = -np.inf

      for j in range(len(self._xInf)):
        # // Compute dominance move
        # // = min \sum_{1}^m max(fi(y) - fi(x), 0)
        # //   y \in Fk

        tmp_dom_move = np.inf
        eval_inf = self._xInf[j]
        h = eval_inf.h

        if h <= self._h_max:
          for x_feas in self._xFeas:
            sum_val = 0.
            eval_feas = x_feas
            for i in range(self._nobj):
              sum_val += max(eval_feas.fs[i]-eval_inf.fs[i], 0)
            if tmp_dom_move > sum_val:
              tmp_dom_move = sum_val
          
          # Get the maximum dominance move index
          if max_dom_move < tmp_dom_move:
            max_dom_move = tmp_dom_move
            current_ind = j
      
      # // In this case, all infeasible solutions are "dominated" in terms of fvalues
      # // by at least one element of Fk
      if np.isclose(max_dom_move, 0., rtol=1e-09, atol=1e-09):
        # // In this case, get the infeasible solution below the _hMax threshold which has
        # // minimal dominance move, when considered a maximization problem.
        min_dom_move = np.inf
        current_ind = 0
        for j in range(len(self._xInf)):
          # // Compute dominance move
          # // = min \sum_{1}^m max(fi(x) - fi(y), 0)
          # //   y \in Fk
          tmp_dom_move = np.inf
          eval_inf = self._xInf[j]
          h = eval_inf.h
          if h<= self._h_max:
            for x_feas in self._xFeas:
              sum_val = 0.
              eval_feas = x_feas
              # Compute \sum_{1}^m max (fi(x) - fi(y), 0)
              for i in range(self._nobj):
                sum_val += max(eval_inf.fs[i] - eval_feas.fs[i], 0.)
              if tmp_dom_move > sum_val:
                tmp_dom_move = sum_val
            
            # Get the minimal dominance move index
            if min_dom_move > tmp_dom_move:
              min_dom_move = tmp_dom_move
              current_ind = j
      self._currentIncumbentInf = self._xInf[current_ind]
    else:
      self._currentIncumbentInf = self.getFirstXIncInfNoXFeas() if len(self._xInf) > 0 else None
            

  def getXInfMinH(self):
    ind_x_inf_min_h = 0
    h_min_val = np.inf

    for i in range(len(self._xInf)):
      my_eval = self._xInf[i]
      h = my_eval.h

      # // By definition, all elements of _xInf or _xFilterInf have a well-defined
      # // h value. So, no need to check.

      if h <h_min_val:
        h_min_val = h
        ind_x_inf_min_h = i
    return self._xInf[ind_x_inf_min_h]

  def getFirstXIncInfNoXFeas(self):
    """ """
    x_inf = None
    if len(self._xFilterInf) == 0:
      return x_inf
    
    # Select candidates
    min_frame_size_inf_elts: float = self.getMeshMaxFrameSize(self.getXInfMinH())
    can_be_frame_center: List[bool] = [False] * len(self._xInf)
    nb_selected_candidates = 0

    for i in range(len(self._xInf)):
      max_frame_size_elt = self.getMeshMaxFrameSize(self._xInf[i])
      if min_frame_size_inf_elts <= max_frame_size_elt:
        can_be_frame_center[i] = True
        nb_selected_candidates += 1
    
    # The selection must always work
    if nb_selected_candidates == 0:
      x_inf = self._xInf[0]
    elif nb_selected_candidates == 1:
      for it in range(len(can_be_frame_center)):
        if can_be_frame_center[it]:
          break
      if it == len(can_be_frame_center):
        raise IOError("Error: DMultiMadsBarrier, should not reach this condition")
      else:
        selected_ind = it
        x_inf = self._xInf[selected_ind]
    elif ((nb_selected_candidates == 2) and (len(self._xInf) == 2)):
      eval1 = self._xInf[0]
      eval2 = self._xInf[1]

      objv1 = eval1.fs
      objv2 = eval2.fs

      if max(np.abs(objv1.coordinates)) > max(np.abs(objv2.coordinates)):
        x_inf = self._xInf[0]
      else:
        x_inf = self._xInf[1]
    else:
      if self._nobj == 2:
        current_best_ind = 0
        max_gap = -1.
        current_gap: float

        for obj in range(self._nobj):
          # Get extreme values value according to one objective
          fmin: float = self._xInf[0].fs[obj]
          fmax: float = self._xInf[len(self._xInf)-1].fs[obj]

          # In this case, it means all elements of _xFeas are equal (return the first one)
          if fmin == fmax:
            break

          # Intermediate points
          for i in range(1, self._xInf-1):
            current_gap = self._xInf[i+1].fs[obj]-self._xInf[i-1].fs[obj]
            self._xInf[i-1].fs[obj]
            current_gap /= (fmax-fmin)
            if (can_be_frame_center[i] and current_gap >= max_gap):
              max_gap = current_gap
              current_best_ind = i
          
          # Extreme points
          current_gap = 2 * (self._xInf[len(self._xInf)-1]).fs[obj] - (self._xInf[len(self._xInf)-2]).fs[obj]
          current_gap /= (fmax - fmin)
          if can_be_frame_center[len(self._xInf)-1] and current_gap > max_gap:
            max_gap = current_gap
            current_best_ind = len(self._xInf)-1
          
          current_gap = 2 * (self._xInf[1]).fs[obj] - (self._xInf[0]).fs[obj]
          current_gap /= (fmax -fmin)

          if can_be_frame_center[0] and current_gap > max_gap:
            max_gap = current_gap
            current_best_ind = 0
        x_inf = self._xInf[current_best_ind]
      # // More than 2 objectives
      else:
        tmp_x_inf_p_ind: List[Tuple[CandidatePoint, int]] = [(CandidatePoint(), 0)]*len(self._xInf)
        for i in range(len(tmp_x_inf_p_ind)):
          tmp_x_inf_p_ind[i] = (self._xInf[i], i)
        current_best_ind = 0
        max_gap = -1.0
        current_gap: float 

        for obj in range(self._nobj):
          # Sort elements of tmpXFeasPInd according to objective obj (in ascending order)
          tmp_x_inf_p_ind = sorted(tmp_x_inf_p_ind, key=lambda x: x[0].fs[obj])

          # Get extreme values value according to one objective
          fmin = tmp_x_inf_p_ind[0][0].fs[obj]
          fmax = tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-1][0].fs[obj]

          # Can happen for exemple when we have several minima or for more than three objectives
          if fmin == fmax:
            fmin = 0.
            fmax = 1.
          
          # Intermediate points
          for i in range(1, len(tmp_x_inf_p_ind)-1):
            current_gap = tmp_x_inf_p_ind[i+1][0].fs[obj]-tmp_x_inf_p_ind[i-1][0].fs[obj]
            current_gap /= (fmax - fmin)
            if can_be_frame_center[tmp_x_inf_p_ind[i][1]] and current_gap >= max_gap:
              max_gap = current_gap
              current_best_ind = tmp_x_inf_p_ind[i][1]
          
          # Extreme points
          current_gap = 2*(tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-1][0].fs[obj]) - tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-2][0].fs[obj]
          current_gap /= (fmax - fmin)

          if (can_be_frame_center[tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-1][1]] and current_gap > max_gap):
            max_gap = current_gap
            current_best_ind = tmp_x_inf_p_ind[len(tmp_x_inf_p_ind)-1][1]
          
          current_gap = 2 * tmp_x_inf_p_ind[1][0].fs[obj] - tmp_x_inf_p_ind[0][0].fs[obj]
          current_gap /= (fmax -fmin)

          if (can_be_frame_center[tmp_x_inf_p_ind[0][1]] and current_gap > max_gap):
            max_gap = current_gap
            current_best_ind = tmp_x_inf_p_ind[0][1]
        x_inf = self._xInf[current_best_ind]

    return x_inf
  
  def updateInfWithPoint(self, eval_point: CandidatePoint = None, keep_all_points: bool = None):
    updated = False

    if eval_point.evaluated and eval_point.status != DESIGN_STATUS.FEASIBLE:
      s: str
      h = eval_point.h

      if h == np.inf or (self._h_max < np.inf and h > self._h_max):
        return False
      else:
        self.setHMax(h)
      
      if self._xInf is None:
        self._xInf = []
      
      if self._xFilterInf is None:
        self._xFilterInf = []
      
      if len(self._xInf) <= 0:
        self._xInf.append(eval_point)
        self._xFilterInf.append(eval_point)
        self._currentIncumbentInf = self._xInf[0]
        updated = True
      else:
        insert = True
        is_in_x_inf_filter: List[bool] = [True] * len(self._xFilterInf)
        current_ind = 0
        for x_filter_inf in self._xFilterInf:
          comp_flag = eval_point.__comp_mo__(x_filter_inf)
          if comp_flag == COMPARE_TYPE.DOMINATED:
            insert = False
            break
          elif comp_flag == COMPARE_TYPE.DOMINATING:
            updated = True
            is_in_x_inf_filter[current_ind] = False
          elif comp_flag == COMPARE_TYPE.EQUAL:
            if (not keep_all_points):
              insert = False
              break
            if self.findEvalPoint(self._xFilterInf, eval_point)[0]:
              insert = False
            else:
              updated = True
              break
          current_ind += 1
        
        if insert:
          indices_to_remove = []
          for i in range(len(self._xFilterInf)):
            if not is_in_x_inf_filter[i]:
              indices_to_remove.append(i)
          
          self._xFilterInf.append(eval_point)
          
          for index in sorted(indices_to_remove, reverse=True):
            del self._xFilterInf[index]
          
          self._xFilterInf = self.non_dominated_sort(self._xFilterInf)
          
          insert = True
          current_ind = 0
          is_in_x_inf = [True * self._xInf]

          for x_inf in self._xInf:
            comp_flag = eval_point.__comp_mo__(x_inf, True)
            if comp_flag == COMPARE_TYPE.DOMINATED:
              insert = False
              break
            elif comp_flag == COMPARE_TYPE.DOMINATING or eval_point.__comp_mo__(x_inf):
              updated = True
              is_in_x_inf[current_ind] = False
            current_ind += 1
          
          if insert:
            indices_to_remove = []
            for i in range(len(self._xInf)):
              if not is_in_x_inf[i]:
                indices_to_remove.append(i)
            updated = True
            self._xInf.append(eval_point)
            
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
        if i != j and p.__comp_mo__(q) == COMPARE_TYPE.DOMINATED:
          dominated_count[i] += 1
    
      if dominated_count[i] == 0:
        fronts[0].append(p)
    
    # Sort each front lexicographically
    for front in fronts:
      front.sort()
    
    # Flatten the fronts into a single list
    sorted_points = [point for front in fronts for point in front]
    
    return sorted_points

  def updateFeasWithPoint(self, eval_point: CandidatePoint = None, keep_all_points: bool = None):
    updated = False
    
    if eval_point.evaluated and eval_point.status == DESIGN_STATUS.FEASIBLE:
      if eval_point.fs.size != self._nobj:
        raise IOError(f"Barrier update: number of objectives is equal to {self._nobj}. Trying to add this point with number of objectives {eval_point.fs.size}")
      
      if self._xFeas is None:
        self._xFeas = []
      
      if len(self._xFeas) == 0:
        self._xFeas.append(eval_point)
        updated = True
        self._currentIncumbentFeas = self._xFeas[0]
      else:
        insert = True
        keep_in_x_feas = [True] * len(self._xFeas)
        current_ind = 0
        for xf in self._xFeas:
          comp_flag: COMPARE_TYPE = eval_point.__comp_mo__(xf)
          if comp_flag == COMPARE_TYPE.DOMINATED:
            insert = False
            break
          elif comp_flag == COMPARE_TYPE.DOMINATING:
            updated = True
            keep_in_x_feas[current_ind] = False
          elif comp_flag == COMPARE_TYPE.EQUAL:
            if not keep_all_points:
              insert = False
              break

            if self.findEvalPoint(self._xFeas, eval_point)[0]:
              insert = False
            else:
              updated = True
            break
          current_ind += 1
        if insert:
          current_ind = 0
          for cp in self._xFeas:
            if cp.__comp_mo__(eval_point) == COMPARE_TYPE.DOMINATED:
              self._xFeas.pop(current_ind)
            current_ind += 1
          updated = True
          my_dir = copy.deepcopy(eval_point.direction)
          if my_dir is not None:
            eval_point.mesh.enlargeDeltaFrameSize(direction=my_dir)
          
          self._xFeas.append(eval_point)

          # Sort according to lexicographic order.
          self._xFeas = self.non_dominated_sort(self._xFeas)

    return updated






