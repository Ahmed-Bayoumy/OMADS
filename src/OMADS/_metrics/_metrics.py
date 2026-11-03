"""
# ------------------------------------------------------------------------------------#
#  Mesh Adaptive Direct Search - (MADS)                                               #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software          #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU General Public License for more details.          #
#                                                                                     #
#  You should have received a copy of the GNU General Public License along            #
#  with this program. If not, see <https://www.gnu.org/licenses/>.                    #
#                                                                                     #
#  You can find information on OMADS at                                               #
#  https://github.com/Ahmed-Bayoumy/OMADS                                             #
#  Copyright (C) 2026  Ahmed H. Bayoumy                                               #
# ------------------------------------------------------------------------------------#
"""
import copy
from typing import List, Optional
import numpy as np


from .._include import CandidatePoint
from .._include import Point



def exact_hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
  """Exact hypervolume of `points` (rows, minimization) with respect to the
  reference point `ref`; points that do not strictly dominate `ref` are
  ignored. A sweep in two objectives; in more, slices along the last
  objective, each the hypervolume of the points below it in one dimension
  less. Dominated points need no filtering: they add nothing to the union
  measured by the sweep."""
  P = np.asarray(points, dtype=float).reshape(-1, len(ref))
  ref = np.asarray(ref, dtype=float)
  P = np.unique(P[np.all(P < ref, axis=1)], axis=0)
  if len(P) == 0:
    return 0.0
  return _hv_sorted(P[np.argsort(P[:, -1], kind="stable")], ref)


def _hv_sorted(P: np.ndarray, ref: np.ndarray) -> float:
  # P sorted by its last column, every row strictly below `ref`.
  d = P.shape[1]
  if d == 1:
    return float(ref[0] - P[0, 0])
  if d == 2:
    # Sweep along the second objective: each point extends the covered
    # width to the left in the first one.
    heights = np.diff(np.append(P[:, 1], ref[1]))
    return float(np.sum(heights * (ref[0] - np.minimum.accumulate(P[:, 0]))))
  vol = 0.0
  for i in range(len(P)):
    depth = (P[i + 1, -1] if i + 1 < len(P) else ref[-1]) - P[i, -1]
    if depth > 0:
      Q = P[:i + 1, :-1]
      vol += _hv_sorted(Q[np.argsort(Q[:, -1], kind="stable")], ref[:-1]) * depth
  return float(vol)


class Metrics:

  def __init__(
          self, nd_solutions: Optional[List[CandidatePoint]] = None, nobj: int = 2,
          ref_point: Optional[Point] = None,
          ideal_point: Optional[List[float]] = None):
    self.nd_solutions: Optional[List[CandidatePoint]] = nd_solutions
    self.nobj: int = nobj
    self.ref_point: Optional[Point] = ref_point
    self.ideal_point: Optional[List[float]] = ideal_point

  def find_ref_point(self):
    if self.nd_solutions:
      self.nobj = len(self.nd_solutions[0].f)
      self.ref_point = Point()
      ftemp = []
      for i in range(self.nobj):
        f: List[float] = []
        for p in self.nd_solutions:
          f.append(p.fobj[i])
        ftemp.append(max(f) + abs(max(f)) * 0.025)
      self.ref_point.coordinates = copy.deepcopy(ftemp)

  def get_pareto_points(self):
    ftemp = []
    if self.nd_solutions:
      self.nobj = len(self.nd_solutions[0].fobj)
      for p in self.nd_solutions:
        f = ()
        for i in range(self.nobj):
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

    # An objective with a single value (e.g. a one-point front) is only
    # translated, to avoid dividing by zero.
    span = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)

    # Normalize Pareto points
    normalized_pareto_front = (pareto_front - min_vals) / span

    # Normalize reference point
    normalized_reference_point = (reference_point - min_vals) / span

    return normalized_pareto_front, normalized_reference_point

  def hypervolume(self):
    """
    Compute the hypervolume indicator of a Pareto front.

    - With a reference point and an ideal point: the hypervolume in the box
      they bound, scaled to [0, 1] (comparable across runs on a problem).
    - With a reference point only: the raw hypervolume (also comparable).
    - Without a reference point: the reference point is derived from the
      front itself and the front is scaled by its own extent, so the value
      only describes this front's shape and is NOT comparable across runs.

    Points that do not strictly dominate the reference point contribute
    nothing (Bigeon et al. 2021, Sect. 6.2).

    Returns:
    - The hypervolume indicator value.
    """
    pf = np.array(self.get_pareto_points(), dtype=float)
    if pf.size == 0:
      return 0.0
    self_referenced = not self.ref_point
    if self_referenced:
      self.find_ref_point()
    ref = np.array(self.ref_point.coordinates, dtype=float)
    if self.ideal_point is not None:
      ideal = np.array(self.ideal_point, dtype=float)
      if np.any(ref <= ideal):
        raise ValueError(
            "The reference point must be worse than the ideal point in every objective.")
      pf, ref = (pf - ideal) / (ref - ideal), np.ones_like(ref)
    elif self_referenced:
      pf, ref = self.normalize_data(pareto_front=pf, reference_point=ref)
    pf = pf[np.all(pf < ref, axis=1)]
    if len(pf) == 0:
      return 0.0
    return exact_hypervolume(pf, ref)

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
    for i, _ in enumerate((pareto_front)):
      _, y = pareto_front[i]
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
      min_distance = min(
          np.linalg.norm(np.array(approx_point) - np.array(true_point))
          for true_point in true_pareto_front)
      gd_sum += min_distance

    gd = gd_sum / len(approximate_pareto_front)
    return gd

  def inverted_generational_distance(
          self, true_pareto_front, approximate_pareto_front):
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
      min_distance = min(
          np.linalg.norm(np.array(true_point) - np.array(approx_point))
          for approx_point in approximate_pareto_front)
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
