import copy
from ._globals import *
from .CandidatePoint import CandidatePoint
from typing import List

@dataclass
class Metrics:
  ND_solutions: List[CandidatePoint] = None
  nobj: int = 2
  _ref_point: CandidatePoint = None
  


  def find_ref_point(self):
    if self.ND_solutions:
      self._nobj = len(self.ND_solutions[0].f)
      self._ref_point = CandidatePoint(self.ND_solutions[0].n_dimensions)
      ftemp = []
      for i in range(self._nobj):
        f: List[float] = []
        for p in self.ND_solutions:
          f.append(p.f[i])
        ftemp.append(max(f))
      self._ref_point.f = copy.deepcopy(ftemp)

  def hypervolume(self):
    """
    Compute the hypervolume indicator of a Pareto front.

    Parameters:
    - self._ref_point: A list or array representing the reference point.
    - front: A list of tuples, where each tuple represents a solution's objective values.

    Returns:
    - The hypervolume indicator value.
    """
    self.find_ref_point()
    # self.ND_solutions = np.array(self.ND_solutions)
    # self._ref_point = np.array(self._ref_point)
    
    # Ensure all objectives are minimized (convert to maximization problem)
    ND_solutions = np.array([np.subtract(self._ref_point.f, xf.f) for xf in self.ND_solutions])
    
    # Sort self.ND_solutionss lexicographically
    ND_solutions.sort(axis=0)
    
    hypervolume_value = 0.0
    last_volume = [1.0]*self.nobj
    
    for point in ND_solutions:
      current_volume = 1.0
      for i in range(len(self._ref_point.f)):
        current_volume *= max(last_volume[i], point[i]) - last_volume[i]
      
      hypervolume_value += current_volume
      last_volume = point
        
    return hypervolume_value
  
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
  
  def dominates(self, A, B):
    return all(A <= B) and any(A < B)

  def ranking(self, solutions):
    # Initialize ranks
    n = len(solutions)
    rank = np.zeros(n, dtype=int)

    # Compare each solution with every other solution
    for i in range(n):
      for j in range(n):
        if i != j:
          if self.dominates(solutions[j], solutions[i]):
            rank[i] += 1


