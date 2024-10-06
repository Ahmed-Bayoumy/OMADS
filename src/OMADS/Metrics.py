import copy
from dataclasses import dataclass

import numpy as np
from .CandidatePoint import CandidatePoint
from typing import List, Optional
from deap.tools._hypervolume import pyhv as hv
from .Point import Point

@dataclass
class Metrics:
  nd_solutions: Optional[List[CandidatePoint]] = None
  nobj: int = 2
  ref_point: Optional[Point] = None
  

  def find_ref_point(self):
    if self.nd_solutions:
      self._nobj = len(self.nd_solutions[0].f)
      self.ref_point = Point()
      ftemp = []
      for i in range(self._nobj):
        f: List[float] = []
        for p in self.nd_solutions:
          f.append(p.fobj[i])
        ftemp.append(max(f)+abs(max(f))*0.025)
      self.ref_point.coordinates = copy.deepcopy(ftemp)

  def get_pareto_points(self):
    ftemp = []
    if self.nd_solutions:
      self._nobj = len(self.nd_solutions[0].fobj)
      for p in self.nd_solutions:
        f = ()
        for i in range(self._nobj):
          f += (p.fobj[i],)
        ftemp.append(f)
    return ftemp

  def normalize_data(self, pareto_front, reference_point):
    """
    Normalize Pareto points and the reference point.

    :param pareto_front: List of Pareto points where each point is a tuple (x, y).
    :param reference_point: The reference point (rx, ry).
    :return: Normalized Pareto points and reference point.
    """
    # Convert Pareto front and reference point to numpy arrays
    pareto_front = np.array(pareto_front)
    reference_point = np.array(reference_point)
    
    # Find min and max values for each objective
    min_vals = np.min(pareto_front, axis=0)
    max_vals = np.max(pareto_front, axis=0)
    
    # Ensure that min and max values are not the same to avoid division by zero
    if np.any(max_vals == min_vals):
        raise ValueError("Max and min values for at least one objective are the same. Normalization cannot be performed.")
    
    # Normalize Pareto points
    normalized_pareto_front = (pareto_front - min_vals) / (max_vals - min_vals)
    
    # Normalize reference point
    normalized_reference_point = (reference_point - min_vals) / (max_vals - min_vals)
    
    return normalized_pareto_front, normalized_reference_point

  def hypervolume(self):
    """
    Compute the hypervolume indicator of a Pareto front.

    Parameters:
    - self._ref_point: A list or array representing the reference point.
    - front: A list of tuples, where each tuple represents a solution's objective values.

    Returns:
    - The hypervolume indicator value.
    """
    if not self.ref_point:
      self.find_ref_point()
    ref_p = tuple(self.ref_point.coordinates)
    pf = self.get_pareto_points()
    pf_n, ref_n = self.normalize_data(pareto_front=pf, reference_point=ref_p)
    # Create a Hypervolume object with the reference point
    pf_n_list = []
    for i in range(len(pf_n)):
      pf_n_list.append(list(pf_n[i]))

    pf_n_list = np.array(pf_n_list)
    return hv.hypervolume(pointset=pf_n_list, ref=np.array(list(ref_n)))

    # # Ensure all objectives are minimized (convert to maximization problem)
    # nd_solutions = np.array([np.subtract(self._ref_point.f, xf.f) for xf in self.nd_solutions])
    
    # # Sort self.ND_solutionss lexicographically
    # nd_solutions.sort(axis=0)
    
    # hypervolume_value = 0.0
    # last_volume = [1.0]*self.nobj
    
    # for point in nd_solutions:
    #   current_volume = 1.0
    #   for i in range(len(self._ref_point.f)):
    #     current_volume *= max(last_volume[i], point[i]) - last_volume[i]
      
    #   hypervolume_value += current_volume
    #   last_volume = point
        
    # return hypervolume_value
  
  def normalize_data(self, pareto_front, reference_point):
    """
    Normalize Pareto points and the reference point.

    :param pareto_front: List of Pareto points where each point is a tuple (x, y).
    :param reference_point: The reference point (rx, ry).
    :return: Normalized Pareto points and reference point.
    """
    # Convert Pareto front and reference point to numpy arrays
    pareto_front = np.array(pareto_front)
    reference_point = np.array(reference_point)
    
    # Find min and max values for each objective
    min_vals = np.min(pareto_front, axis=0)
    max_vals = np.max(pareto_front, axis=0)
    
    # Normalize Pareto points
    normalized_pareto_front = (pareto_front - min_vals) / (max_vals - min_vals)
    
    # Normalize reference point
    normalized_reference_point = (reference_point - min_vals) / (max_vals - min_vals)
    
    return normalized_pareto_front, normalized_reference_point
  
  def calculate_hypervolume(self, pf, rp):
    """
    Calculate the hypervolume of a bi-objective Pareto front.

    :param pareto_front: A list of Pareto points where each point is a tuple (x, y).
    :param reference_point: The reference point (rx, ry) to compute the hypervolume against.
    :return: Hypervolume of the Pareto front.
    """
    # Sort Pareto front by the first objective (x-coordinate)
    pareto_front, reference_point = self.normalize_data(pf, rp)
    pareto_front = sorted(pareto_front, key=lambda point: point[0])
    
    # Initialize variables
    hypervolume = 0.0
    previous_y = reference_point[1]
    
    # Iterate through the sorted Pareto points
    for i in range(len(pareto_front)):
        x, y = pareto_front[i]
        # Compute the area between the current point and the previous point
        width = pareto_front[i][0] - (pareto_front[i - 1][0] if i > 0 else 0)
        height = previous_y - y
        hypervolume += width * height
        
        # Update previous_y to the current y
        previous_y = y
    
    # Account for the last segment up to the reference point
    width = reference_point[0] - pareto_front[-1][0]
    height = previous_y - reference_point[1]
    hypervolume += width * height
    
    return hypervolume

  def generational_distance(self, true_pareto_front, approximate_pareto_front):
    """
    Compute the generational distance (GD) metric between two Pareto fronts.

    Parameters:
    - true_pareto_front: A list of tuples representing the true Pareto front.
    - approximate_pareto_front: A list of tuples representing the approximate Pareto front.

    Returns:
    - The generational distance metric value.
    """
    gd_sum = 0.0
    
    for approx_point in approximate_pareto_front:
      min_distance = min(np.linalg.norm(np.array(approx_point) - np.array(true_point)) for true_point in true_pareto_front)
      gd_sum += min_distance
    
    gd = gd_sum / len(approximate_pareto_front)
    return gd
  
  def inverted_generational_distance(self, true_pareto_front, approximate_pareto_front):
    """
    Compute the inverted generational distance (IGD) metric between two Pareto fronts.

    Parameters:
    - true_pareto_front: A list of tuples representing the true Pareto front.
    - approximate_pareto_front: A list of tuples representing the approximate Pareto front.

    Returns:
    - The inverted generational distance metric value.
    """
    igd_sum = 0.0
    
    for true_point in true_pareto_front:
      min_distance = min(np.linalg.norm(np.array(true_point) - np.array(approx_point)) for approx_point in approximate_pareto_front)
      igd_sum += min_distance
    
    igd = igd_sum / len(true_pareto_front)
    return igd
  
  def dominates(self, a, b):
    return all(a <= b) and any(a < b)

  def ranking(self, solutions):
    # Initialize ranks
    n = len(solutions)
    rank = np.zeros(n, dtype=int)

    # Compare each solution with every other solution
    for i in range(n):
      for j in range(n):
        if i != j and self.dominates(solutions[j], solutions[i]):
            rank[i] += 1


