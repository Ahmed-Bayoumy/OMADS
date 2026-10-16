Points and cache
================

.. autoclass:: OMADS._points._point.Point
   :no-members:

.. autoclass:: OMADS._points._candidate_point.CandidatePoint
   :members: f, fobj, h, c_ineq, is_feasible, status, coordinates

.. autoclass:: OMADS._points._cache.Cache
   :members: add_to_cache, is_duplicate, get_candidate_from_cache_by_index,
             get_feasible_points_and_f, get_points_f_and_c, save_cache, load_cache
