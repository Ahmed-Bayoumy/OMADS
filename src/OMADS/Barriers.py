import copy
from dataclasses import dataclass, field
from typing import List
from .Point import Point
from ._globals import *
import numpy as np
from ._common import Parameters

@dataclass
class Barrier:
  _params: Parameters = None
  _eval_type: int = 1
  _h_max: float = 0
  _best_feasible: Point = None
  _ref: Point = None
  _filter: List[Point] = None
  _prefilter: int = 0
  _rho_leaps: float = 0.1
  _prim_poll_center: Point = None
  _sec_poll_center: Point = None
  _peb_changes: int = 0
  _peb_filter_reset: int = 0
  _peb_lop: List[Point] = None
  _all_inserted: List[Point] = None
  _one_eval_succ: SUCCESS_TYPES = None
  _success: SUCCESS_TYPES = None

  def __init__(self, p: Parameters, eval_type: int = 1):
    self._h_max = p.get_h_max_0()
    self._params = p
    self._eval_type = eval_type


  def insert_feasible(self, x: Point) -> SUCCESS_TYPES:
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
  
  def filter_insertion(self, x:Point) -> bool:
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


  def insert_infeasible(self, x: Point):
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
    best_infeasible: Point = self.get_best_infeasible()
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
    
    last_poll_center: Point = Point()
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

  def insert(self, x: Point):
    """/*---------------------------------------------------------*/
      /*         insertion of an Eval_Point in the barrier       */
      /*---------------------------------------------------------*/
    """
    if not x.evaluated:
      raise RuntimeError("This points hasn't been evaluated yet and cannot be inserted to the barrier object!")
    
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
