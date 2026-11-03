"""Unit tests for the DMulti-MADS barrier and success rules
(Bigeon, Le Digabel & Salomon, COAP 79:301-338, 2021)."""
import types

import pytest

from OMADS._analytics._metadata import MadsState
from OMADS._barriers._barriers import AdaptiveBarrier
from OMADS._include import DESIGN_STATUS, INSERTION_FLAG, SUCCESS_TYPES
from OMADS._metrics._success import compute_success
from OMADS._points._candidate_point import CandidatePoint
from OMADS._setup._options import Options


class _Mesh:
  """The barrier only reads the dimension of the mesh and stores a copy."""
  n = 2


def _point(x, f, h=0.0):
  p = CandidatePoint()
  p.coordinates = x
  p.f = list(f)
  p.fobj = list(f)
  p.h = h
  p.status = DESIGN_STATUS.FEASIBLE if h == 0.0 else DESIGN_STATUS.INFEASIBLE
  return p


def _barrier():
  param = types.SimpleNamespace(n=2, nobj=2, h_max=float("inf"))
  return AdaptiveBarrier(param, Options(), max_size=50)


def _ik(barrier):
  return [i for i in range(barrier.last_index + 1) if barrier.within_ik[i]]


def _state(fk_center):
  state = MadsState()
  state.fk_frame_center = fk_center
  state.uk_frame_center = -1
  return state


def test_ik_keeps_non_dominated_infeasible_points():
  b = _barrier()
  b.add_infeasible(_point([0, 0], [1, 5], h=1.0), _Mesh())
  b.add_infeasible(_point([1, 0], [5, 1], h=1.0), _Mesh())
  assert _ik(b) == [0, 1]
  b.add_infeasible(_point([2, 0], [4, 4], h=0.5), _Mesh())
  assert _ik(b) == [0, 1, 2]


def test_threshold_update_keeps_the_last_point_in_ik():
  # The last element is in uk but not ik (its f is worse). Lowering h_max
  # removes the only ik element; the last one must take its place, or the
  # infeasible frame centre falls back to element 0, above the threshold.
  class FramedMesh(_Mesh):
    def get_frame_size_parameter(self):
      return [1.0, 1.0]

  b = _barrier()
  b.add_infeasible(_point([0, 0], [1, 1], h=2.0), FramedMesh())
  b.add_infeasible(_point([1, 0], [5, 5], h=1.0), FramedMesh())
  assert _ik(b) == [0]
  b.update_barrier(1.0)
  assert _ik(b) == [1]
  assert b.infeasible_frame_center() == 1


def test_ik_drops_only_the_dominated_points():
  b = _barrier()
  b.add_infeasible(_point([0, 0], [1, 5], h=1.0), _Mesh())
  b.add_infeasible(_point([1, 0], [5, 1], h=1.0), _Mesh())
  # Dominates point 1 in f (and h), indifferent to point 0.
  b.add_infeasible(_point([2, 0], [4, 0.5], h=0.5), _Mesh())
  assert _ik(b) == [0, 2]


def test_ik_unchanged_by_point_dominated_in_f():
  b = _barrier()
  b.add_infeasible(_point([0, 0], [1, 5], h=1.0), _Mesh())
  b.add_infeasible(_point([1, 0], [5, 1], h=1.0), _Mesh())
  # Better h keeps it in uk, but it is dominated in f by point 0.
  b.add_infeasible(_point([2, 0], [2, 6], h=0.5), _Mesh())
  assert _ik(b) == [0, 1]


def test_success_requires_strict_dominance_of_frame_center():
  b = _barrier()
  b.add_feasible(_point([0, 0], [1, 1]), _Mesh())
  state, options = _state(0), Options()

  equal = _point([1, 1], [1, 1])
  assert compute_success(b, state, options, INSERTION_FLAG.IS_DOMINATED,
                         equal) == SUCCESS_TYPES.US
  weakly_better = _point([2, 2], [0.5, 1])
  assert compute_success(b, state, options, INSERTION_FLAG.DOMINATES,
                         weakly_better) == SUCCESS_TYPES.FS
  indifferent = _point([3, 3], [0.5, 2])
  assert compute_success(b, state, options, INSERTION_FLAG.EXTENDS,
                         indifferent) == SUCCESS_TYPES.US


def test_dms_success_counts_any_list_improvement():
  b = _barrier()
  b.add_feasible(_point([0, 0], [1, 1]), _Mesh())
  state = _state(0)
  options = Options(use_dms_success=True)

  indifferent = _point([3, 3], [0.5, 2])
  for flag in (INSERTION_FLAG.EXTENDS, INSERTION_FLAG.DOMINATES,
               INSERTION_FLAG.IMPROVES):
    assert compute_success(b, state, options, flag,
                           indifferent) == SUCCESS_TYPES.FS
  assert compute_success(b, state, options, INSERTION_FLAG.IS_DOMINATED,
                         _point([4, 4], [2, 2])) == SUCCESS_TYPES.US


# --- Paper conformance: poll directions, opportunism, speculative search ---

def test_poll_direction_defaults_follow_number_of_objectives():
  options = Options()
  assert options.poll_rich_direction(nobj=2) is True
  assert options.poll_direction_type(nobj=2) == "N+1"
  assert options.poll_rich_direction(nobj=1) is False
  assert options.poll_direction_type(nobj=1) == "ORTHO_2N"
  explicit = Options(rich_direction=False, direction_type="ORTHO_2N")
  assert explicit.poll_rich_direction(nobj=2) is False
  assert explicit.poll_direction_type(nobj=2) == "ORTHO_2N"
  assert Options(direction_type="ORTHO_N+1").poll_direction_type(1) == "N+1"


def _bi(x):
  return [[x[0] ** 2 + x[1] ** 2, (x[0] - 1) ** 2 + (x[1] - 1) ** 2], [0]]


def _biobjective_setup(tmp_path, baseline=(0.3, 0.6), scaling=None):
  """Poll sampler, parameters, barrier and cache after preprocessing."""
  from OMADS._include import SAMPLER_TYPE, logger
  from OMADS._pre._preprocess import preprocess

  data = {
      "evaluator": {"blackbox": _bi},
      "param": {"name": "bi", "baseline": list(baseline), "lb": [-2, -2],
                "ub": [2, 2], "var_names": ["x1", "x2"],
                "fun_names": ["f1", "f2"], "nobj": 2, "is_pareto": True,
                "post_dir": str(tmp_path)},
      "options": {"seed": 0, "budget": 50, "check_cache": True,
                  "store_cache": True, "display": False},
  }
  if scaling is not None:
    data["param"]["scaling"] = list(scaling)
  log = logger()
  log.initialize(str(tmp_path / "OMADS.log"))
  pre = preprocess(data=data, log=log, sampler_t=SAMPLER_TYPE.POLL)
  out = pre.initialize_from_dict()
  poll, param, barrier, hashtable = out[2], out[4], out[7], out[12]
  # As poll_step does before creating the poll set.
  poll.prob_params = param
  poll.mesh = barrier.meshes[0]
  return types.SimpleNamespace(
      poll=poll, param=param, barrier=barrier, hashtable=hashtable,
      fc=hashtable.get_candidate_from_cache_by_index(0))


def _biobjective_poll(tmp_path):
  setup = _biobjective_setup(tmp_path)
  return setup.poll, setup.param, setup.fc


@pytest.mark.parametrize("target", [[1.0, 0.2], [-1.0, 0.3], [0.2, -1.0]])
def test_n_plus_1_basis_is_oriented_and_ordered_by_target(tmp_path, target):
  import numpy as np
  poll, param, fc = _biobjective_poll(tmp_path)
  target = np.array(target)
  poll.create_poll_set(
      ub=param.ub, lb=param.lb, fc=fc, fci=0, it=1, var_type=param.var_type,
      var_sets=param.var_sets, var_link=None, rich_direction=True,
      direction_type="N+1", target_direction=target)
  steps = np.array([np.subtract(c.coordinates, fc.coordinates)
                    for c in poll.candidate_points_set])
  assert len(steps) == param.n + 1
  dots = steps @ target
  # The n basis directions point towards the target, the completing
  # (negative sum) direction away from it ...
  assert np.sum(dots >= 0) == param.n and np.sum(dots < 0) == 1
  # ... and the poll points come in decreasing cosine with the target.
  cosines = dots / np.linalg.norm(steps, axis=1)
  assert np.all(np.diff(cosines) <= 1e-12)


def test_opportunistic_evaluation_stops_on_step_success_rule():
  from unittest import mock
  from OMADS._evaluator._evaluator import Evaluator

  flags = [INSERTION_FLAG.EXTENDS, INSERTION_FLAG.DOMINATES,
           INSERTION_FLAG.EXTENDS, INSERTION_FLAG.EXTENDS]
  sampler = mock.MagicMock()
  sampler._candidate_points_set = [mock.MagicMock() for _ in flags]
  for c in sampler._candidate_points_set:
    c.is_feasible.return_value = True
  barrier = mock.MagicMock()
  barrier.add_feasible.side_effect = flags
  stats = types.SimpleNamespace(neval_bb=0, neval_bb_feasible=0,
                                nopportunistic_triggers=0)
  seen = []

  def full_success(candidate, flag):
    seen.append(flag)
    return flag == INSERTION_FLAG.DOMINATES

  flag_out, _ = Evaluator().run_callable_serial_local(
      sampler=sampler, centers=[0] * 4, options=Options(opportunistic=True),
      stats=stats, active_barrier=barrier, post=mock.MagicMock(),
      step_name="Poll", parent_indices=[0] * 4, out=None,
      hashtable=mock.MagicMock(), internal=True, full_success=full_success)
  assert stats.neval_bb == 2 and stats.nopportunistic_triggers == 1
  assert flag_out == flags[:2] + [None, None]
  assert seen == flags[:2]


def _cheap_search_points(nobj, x_parent, mo_search=False):
  import numpy as np
  from unittest import mock
  from OMADS._heuristics._exploration import EfficientExploration

  search = mock.MagicMock()
  search.prob_params = types.SimpleNamespace(
      constraints_type=None, is_pareto=nobj > 1, nobj=nobj,
      lb=[-2.0, -2.0], ub=[2.0, 2.0])
  search._quad_model_point.return_value = np.array([0.0, 0.0])
  fc = _point([0.5, 0.5], [1.0] * nobj)
  EfficientExploration.generate_quad_model_points(
      search, hashtable=None, fc=fc, x_parent=x_parent, mo_search=mo_search)
  if not search.map_samples_from_coords_to_points.called:
    return []
  return search.map_samples_from_coords_to_points.call_args[0][0].tolist()


def test_multiobjective_cheap_search_without_mo_search_is_speculative_only():
  # mo_search=False: repeats the last successful step x_k - x_parent, no
  # quadratic model (DMulti-MADS as published in 2021).
  assert _cheap_search_points(2, [0.25, 0.5]) == [[0.75, 0.5]]
  assert _cheap_search_points(2, None) == []
  # Single objective keeps both points.
  assert _cheap_search_points(1, [0.25, 0.5]) == [[0.75, 0.5], [0.0, 0.0]]


def _bowl_data(n=2, seed=0, npts=40):
  # f1 = |x - a|^2, f2 = |x - b|^2 sampled around xk.
  import numpy as np
  rng = np.random.default_rng(seed)
  a, b, xk = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])
  X = xk + rng.uniform(-0.5, 0.5, size=(npts, n))
  F = np.column_stack([np.sum((X - a) ** 2, 1), np.sum((X - b) ** 2, 1)])

  def f(x):
    return np.array([np.sum((x - a) ** 2), np.sum((x - b) ** 2)])

  return X, F, xk, f


def test_mo_model_point_dominates_the_frame_center():
  import numpy as np
  from OMADS._heuristics._mo_search import mo_model_points
  X, F, xk, f = _bowl_data()
  pts = mo_model_points(X, F, np.zeros((len(X), 0)), xk, f(xk), True,
                        radius=np.full(2, 0.25), lb=np.full(2, -2.),
                        ub=np.full(2, 2.), half_mesh=np.full(2, 1e-3))
  assert len(pts) == 1
  assert np.all(f(pts[0]) < f(xk))
  assert np.all(np.abs(pts[0] - xk) <= 0.25 + 1e-9)


def test_mo_model_point_respects_constraint_models():
  import numpy as np
  from OMADS._heuristics._mo_search import mo_model_points
  X, F, xk, f = _bowl_data()
  # c(x) = 0.9 - x1 <= 0 forbids moving x1 much below 0.9.
  C = (0.9 - X[:, 0])[:, None]
  pts = mo_model_points(X, F, C, xk, f(xk), True, radius=np.full(2, 0.25),
                        lb=np.full(2, -2.), ub=np.full(2, 2.),
                        half_mesh=np.full(2, 1e-3))
  assert len(pts) == 1 and pts[0][0] >= 0.9
  assert np.all(f(pts[0]) < f(xk))


def test_mo_model_points_extend_the_front_from_an_extreme_point():
  import numpy as np
  from OMADS._heuristics._mo_search import mo_model_points
  X, F, xk, f = _bowl_data()
  pts = mo_model_points(X, F, np.zeros((len(X), 0)), xk, f(xk), True,
                        radius=np.full(2, 0.25), lb=np.full(2, -2.),
                        ub=np.full(2, 2.), half_mesh=np.full(2, 1e-3),
                        extreme_objs=[0])
  assert len(pts) == 2
  assert f(pts[1])[0] < f(xk)[0]


def test_mo_model_points_need_enough_data():
  import numpy as np
  from OMADS._heuristics._mo_search import mo_model_points
  X, F, xk, f = _bowl_data(npts=3)
  assert mo_model_points(X, F, np.zeros((3, 0)), xk, f(xk), True,
                         np.full(2, .25), np.full(2, -2.), np.full(2, 2.),
                         np.full(2, 1e-3)) == []


def test_middle_point_takes_the_larger_gap():
  import numpy as np
  from OMADS._heuristics._mo_search import middle_point
  fx = np.array([[0., 0.], [1., 1.], [3., 3.]])
  ff = np.array([[0., 3.], [1., 2.], [3., 0.]])
  assert middle_point(fx, ff, fx[1]).tolist() == [2., 2.]
  assert middle_point(fx, ff, fx[0]).tolist() == [0.5, 0.5]
  assert middle_point(fx[:1], ff[:1], fx[0]) is None
  assert middle_point(fx, ff, np.array([9., 9.])) is None


def test_two_phase_finds_a_feasible_point_within_the_budget(tmp_path):
  import numpy as np
  from OMADS import poll
  evals = []

  def bb(x):
    x = np.asarray(x, float)
    c = [6 - (x[1] + 9 * x[0]), 1 - (-x[1] + 9 * x[0])]
    evals.append(max(c) <= 0)
    return [[x[0], (1 + x[1]) / x[0]], c]

  data = {"evaluator": {"blackbox": bb},
          "param": {"name": "t", "baseline": [0.15, 4.5], "lb": [0.1, 0],
                    "ub": [1, 5], "scaling": [0.9, 5], "nobj": 2,
                    "is_pareto": True, "fun_names": ["f1", "f2"],
                    "constraints_type": ["PB", "PB"],
                    "post_dir": str(tmp_path)},
          "options": {"seed": 1, "budget": 150, "check_cache": True,
                      "store_cache": True, "constraint_strategy": "TWO_PHASE"}}
  poll.main(data)
  assert len(evals) <= 150 and any(evals)


def test_constraint_strategy_is_validated():
  with pytest.raises(ValueError):
    Options(constraint_strategy="EB-ONLY")
  assert Options(constraint_strategy="two_phase").constraint_strategy == "TWO_PHASE"


# --- Hypervolume and results files ---

def _metrics(front, ref=None, ideal=None):
  from OMADS._metrics._metrics import Metrics
  from OMADS._points._point import Point
  rp = None
  if ref is not None:
    rp = Point()
    rp.coordinates = list(ref)
  return Metrics(nd_solutions=[_point([i, 0], f) for i, f in enumerate(front)],
                 nobj=2, ref_point=rp, ideal_point=ideal)


def test_hypervolume_is_comparable_across_fronts():
  full = [[0, 10], [5, 5], [10, 0]]
  partial = [[1, 9], [5, 5]]       # misses both extremes of `full`
  ref, ideal = [12, 12], [0, 0]
  raw_full = _metrics(full, ref).hypervolume()
  assert raw_full == pytest.approx(12 * 2 + 7 * 5 + 2 * 5)
  assert _metrics(partial, ref).hypervolume() < raw_full
  scaled = _metrics(full, ref, ideal).hypervolume()
  assert scaled == pytest.approx(raw_full / 144)
  assert _metrics(partial, ref, ideal).hypervolume() < scaled


def test_hypervolume_edge_cases():
  # One point (used to raise in the normalization), points beyond the
  # reference point (contribute nothing), empty front.
  assert _metrics([[1, 1]], [2, 2]).hypervolume() == pytest.approx(1.0)
  assert _metrics([[1, 1]]).hypervolume() >= 0.0
  assert _metrics([[1, 1], [3, 0]], [2, 2]).hypervolume() == pytest.approx(1.0)
  assert _metrics([[3, 3]], [2, 2]).hypervolume() == 0.0
  assert _metrics([], [2, 2]).hypervolume() == 0.0


def _read_out(path):
  import pandas as pd
  df = pd.read_csv(path, skipinitialspace=True)
  df.columns = df.columns.str.strip()
  return df


def _run_biobjective(tmp_path, solver, budget=150, **options):
  from OMADS import mads, poll
  data = {
      "evaluator": {"blackbox": _bi},
      "param": {"name": "bi", "baseline": [0.3, 0.6], "lb": [-2, -2],
                "ub": [2, 2], "var_names": ["x1", "x2"],
                "fun_names": ["f1", "f2"], "nobj": 2, "is_pareto": True,
                "ref_point": [10.0, 10.0], "ideal_point": [0.0, 0.0],
                "post_dir": str(tmp_path)},
      "options": dict({"seed": 0, "budget": budget, "check_cache": True,
                       "store_cache": True, "save_results": True,
                       "display": False}, **options),
      "search": {"type": "sampling", "s_method": "LH", "ns": 5,
                 "visualize": False},
  }
  out = (mads.main if solver == "mads" else poll.main)(data)[0]
  return (out, _read_out(tmp_path / "bi_run" / "bi_all.out"),
          _read_out(tmp_path / "bi_ND" / "bi_Pareto.out"))


@pytest.mark.parametrize("solver", ["mads", "poll"])
def test_results_files_of_multiobjective_runs(tmp_path, solver):
  import numpy as np
  out, all_out, pareto = _run_biobjective(tmp_path, solver)
  assert 0.0 < out["HV"] <= 1.0
  # One row per evaluation, numbered 1..N.
  evals = sorted(all_out["Evaluation #"])
  assert evals == list(range(1, len(evals) + 1)) and len(evals) > 100
  # The Pareto file holds the non-dominated rows of the results file
  # (one per objective vector), with the same iteration and frame size.
  # Compared on rounded values: mirrored points of this symmetric problem
  # tie up to the last bits, so either one (or both, when they trade off in
  # those bits) may represent the tie.
  F = np.round(all_out[["f1", "f2"]].to_numpy(float), 10)
  nd = [not np.any(np.all(F <= f, 1) & np.any(F < f, 1)) for f in F]
  expected = {tuple(f) for f in F[nd]}
  got = [tuple(f) for f in np.round(pareto[["f1", "f2"]].to_numpy(float), 10)]
  assert set(got) == expected and len(expected) > 1
  merged = pareto.merge(all_out, on="Evaluation #", suffixes=("", "_all"))
  assert len(merged) == len(pareto)
  for col in ("f1", "f2", "Iteration", "Delta", "Step"):
    assert (merged[col] == merged[col + "_all"]).all(), col


def test_poll_main_runs_the_speculative_search(tmp_path):
  _, all_out, _ = _run_biobjective(tmp_path, "poll")
  assert (all_out["Step"].str.strip() == "Speculative").sum() > 0


# --- Incumbent selection (Alg. 4) ---

class _FrameMesh:
  """Mesh stub with a given frame size."""
  n = 2

  def __init__(self, delta, converged=False):
    self.delta, self.converged = delta, converged

  def get_frame_size_parameter(self):
    return [self.delta, self.delta]

  def check_mesh_for_stopping(self):
    return self.converged

  def frame_size_after_refinements(self, value, k):
    return value * 0.5 ** k


def _barrier_with(points, deltas, converged=None):
  b = _barrier()
  for i, (f, d) in enumerate(zip(points, deltas)):
    b.add_feasible(_point([i, 0], f), _Mesh())
    b.meshes[b.last_index] = _FrameMesh(
        d, converged=bool(converged and converged[i]))
  return b


def test_two_point_case_takes_largest_objective_value():
  # max_i f_i is 0 for the first point and 1 for the second; the infinity
  # norm (10 vs 1) used to pick the first one.
  b = _barrier_with([[-10, 0], [1, -5]], [1.0, 1.0])
  assert b.feasible_frame_center() == 1


def test_fallback_is_the_largest_frame():
  b = _barrier_with([[0, 3], [1, 2], [2, 1]], [0.1, 0.4, 0.2],
                    converged=[True, True, True])
  assert b.feasible_frame_center() == 1


def test_w_plus_threshold():
  # With the halving stub, tau**3 * 1.0 = 0.125: the 0.1 frame is out and
  # the selection is between the two others.
  b = _barrier_with([[0, 3], [1, 2], [2, 1]], [1.0, 0.2, 0.1])
  assert b.feasible_frame_center(w_plus=3) in (0, 1)
  # w_plus = 0 keeps only the largest frame.
  b2 = _barrier_with([[0, 3], [1, 2], [2, 1]], [0.15, 0.2, 1.0])
  assert b2.feasible_frame_center(w_plus=0) == 2


def test_granular_mesh_refinements():
  from OMADS._mesh._gmesh import Gmesh
  mesh = Gmesh.__new__(Gmesh)
  for value in (1.0, 2.0, 5.0, 0.2, 50.0):
    assert mesh.frame_size_after_refinements(value, 3) == \
        pytest.approx(value / 10)
  assert [mesh.frame_size_after_refinements(1.0, k) for k in range(4)] == \
      pytest.approx([1.0, 0.5, 0.2, 0.1])


def test_no_infeasible_center_without_dominance_trigger():
  b = _barrier_with([[0, 3], [1, 2]], [1.0, 1.0])
  assert b.frame_centers(None, use_dom_selection=False)["infeasible"] == -1
  assert b.frame_centers(None, use_dom_selection=True)["infeasible"] == -1


def test_no_parent_is_minus_one():
  assert set(_barrier().parent_indexes) == {-1}


# --- Bounds ---

def _poll_points(setup, snap_to_bounds):
  import numpy as np
  setup.poll.create_poll_set(
      ub=setup.param.ub, lb=setup.param.lb, fc=setup.fc, fci=0, it=1,
      var_type=setup.param.var_type, var_sets=setup.param.var_sets,
      var_link=None, rich_direction=True, direction_type="ORTHO_2N",
      snap_to_bounds=snap_to_bounds)
  return np.array([c.coordinates for c in setup.poll.candidate_points_set])


def test_points_outside_bounds_are_snapped_or_dropped(tmp_path):
  import numpy as np
  setup = _biobjective_setup(tmp_path, baseline=(-2.0, -2.0))
  snapped = _poll_points(setup, snap_to_bounds=True)
  kept = _poll_points(setup, snap_to_bounds=False)
  assert np.all(snapped >= -2) and np.all(kept >= -2)
  # At the lower corner, the poll steps with a negative component leave the
  # bounds: snapped onto them, or dropped.
  assert 0 < len(kept) < len(snapped)


def test_snapped_points_stay_on_the_mesh(tmp_path):
  import numpy as np
  from OMADS._analytics._metadata import MadsStatistics
  # Poll steps of one mesh size (scaling = variable range), as in the
  # tutorials; with the default scaling the first poll collapses onto x0.
  setup = _biobjective_setup(tmp_path, baseline=(1.9, -1.9),
                             scaling=(4.0, 4.0))
  _poll_points(setup, snap_to_bounds=True)
  mesh = setup.poll.mesh
  setup.poll.project_on_mesh_and_snap_to_bounds(
      m=mesh, x_center=setup.fc.coordinates, lb=setup.param.lb,
      ub=setup.param.ub, hashtable=setup.hashtable, stats=MadsStatistics())
  delta = np.array(mesh.get_delta_mesh_size().coordinates, dtype=float)
  assert len(setup.poll.candidate_points_set) > 0
  for c in setup.poll.candidate_points_set:
    x = np.array(c.coordinates, dtype=float)
    assert np.all(x >= -2) and np.all(x <= 2)
    steps = (x - np.array(setup.fc.coordinates, dtype=float)) / delta
    assert np.allclose(steps, np.round(steps)), steps


# --- Parallel opportunistic evaluation ---

def test_parallel_stop_records_every_evaluation_that_ran():
  import concurrent.futures
  import threading
  from unittest import mock
  from OMADS._evaluator._evaluator import Evaluator

  n = 6
  ran, lock = [], threading.Lock()
  release = threading.Event()

  def evaluate(point):
    with lock:
      ran.append(point)
      first = len(ran) == 1
    if not first:
      release.wait(5)   # keep the other workers busy past the stop
    return point

  sampler = mock.MagicMock()
  sampler.candidate_points_set = sampler._candidate_points_set = \
      [mock.MagicMock() for _ in range(n)]
  for c in sampler.candidate_points_set:
    c.is_feasible.return_value = True
  barrier = mock.MagicMock()
  barrier.add_feasible.return_value = INSERTION_FLAG.DOMINATES
  stats = types.SimpleNamespace(neval_bb=0, neval_bb_feasible=0,
                                nopportunistic_triggers=0)
  ev = Evaluator()
  pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
  ev._get_executor = lambda workers: pool
  ev.evaluate_blackbox_parallel = evaluate

  def full_success(candidate, flag):
    release.set()
    return True

  ev.run_callable_parallel_local(
      sampler=sampler, centers=[0] * n,
      options=Options(opportunistic=True, parallel_mode=True, np=3),
      stats=stats, active_barrier=barrier, post=mock.MagicMock(),
      step_name="Poll", parent_indices=[0] * n, out=None,
      hashtable=mock.MagicMock(), full_success=full_success)
  pool.shutdown(wait=True)
  # Stopped early, and every evaluation that ran was recorded.
  assert stats.nopportunistic_triggers == 1
  assert len(ran) < n and stats.neval_bb == len(ran)


# --- Cache, mesh lattice, solvers' outputs ---

def test_signature_ignores_floating_point_noise():
  from OMADS._points._point import coordinates_signature
  assert coordinates_signature([0.3, 0.6]) == \
      coordinates_signature([0.30000000000000004, 0.6])
  assert coordinates_signature([0.3, 0.6]) != \
      coordinates_signature([0.3 + 1e-9, 0.6])
  a, b = _point([0.3, 0.6], [1, 1]), _point([0.1 + 0.2, 0.6], [1, 1])
  assert a.signature == b.signature


def test_first_poll_is_on_the_scaled_mesh(tmp_path):
  import numpy as np
  from OMADS._analytics._metadata import MadsStatistics
  # Default scaling (10): poll steps are in units of (ub - lb) / 10 = 0.4,
  # and the projection used to round them all back onto the center.
  setup = _biobjective_setup(tmp_path)
  _poll_points(setup, snap_to_bounds=True)
  mesh = setup.poll.mesh
  setup.poll.project_on_mesh_and_snap_to_bounds(
      m=mesh, x_center=setup.fc.coordinates, lb=setup.param.lb,
      ub=setup.param.ub, hashtable=setup.hashtable, stats=MadsStatistics())
  points = setup.poll.candidate_points_set
  assert len(points) == 2 * setup.param.n
  lattice = mesh.get_lattice_size()
  assert lattice == pytest.approx(
      np.asarray(mesh.get_delta_mesh_size().coordinates) * 0.4)
  for c in points:
    steps = (np.subtract(c.coordinates, setup.fc.coordinates)) / lattice
    assert np.allclose(steps, np.round(steps)) and np.any(steps != 0)


def _run_solver(tmp_path, solver, nobj, s_method="LH", budget=60, **options):
  from OMADS import mads, poll, search

  def rosen(x):
    return [[100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2], [0]]

  param = {"name": "s", "baseline": [0.3, 0.6], "lb": [-2, -2],
           "ub": [2, 2], "var_names": ["x1", "x2"], "post_dir": str(tmp_path)}
  if nobj == 2:
    param.update(fun_names=["f1", "f2"], nobj=2, is_pareto=True)
  data = {
      "evaluator": {"blackbox": _bi if nobj == 2 else rosen},
      "param": param,
      "options": dict({"seed": 0, "budget": budget, "check_cache": True,
                       "store_cache": True, "display": False,
                       "save_results": True}, **options),
      "search": {"type": "sampling", "s_method": s_method, "ns": 8,
                 "visualize": False},
  }
  out = {"mads": mads.main, "poll": poll.main, "search": search.main}[
      solver](data)
  return out[0] if isinstance(out, tuple) else out


@pytest.mark.parametrize("solver,nobj,s_method", [
    ("search", 1, "LH"), ("search", 2, "LH"), ("search", 2, "ACTIVE"),
    ("mads", 2, "ACTIVE"), ("poll", 1, "LH"), ("poll", 2, "LH")])
def test_solvers_run_and_report_their_evaluations(
        tmp_path, solver, nobj, s_method):
  out = _run_solver(tmp_path, solver, nobj, s_method)
  rows = _read_out(tmp_path / "s_run" / "s_all.out")
  assert out["nbb_evals"] == len(rows) == 60
  assert sorted(rows["Evaluation #"]) == list(range(1, 61))


def test_poll_main_reports_its_successes(tmp_path):
  out = _run_solver(tmp_path, "poll", 1, budget=60)
  assert out["nb_success"] > 0


def test_executable_reads_the_point_it_is_given(tmp_path, monkeypatch):
  import subprocess
  from OMADS._evaluator._evaluator import Evaluator

  def fake_run(cmd, **kwargs):
    # The "executable": f = x1 + 2 x2 from the input file, one constraint.
    x = [float(v) for v in (tmp_path / "in.txt").read_text().split()]
    (tmp_path / "out.txt").write_text(f"{x[0] + 2 * x[1]}\n-1\n")
    return subprocess.CompletedProcess(cmd, 0)

  ev = Evaluator(blackbox="bb.exe", path=str(tmp_path), input="in.txt",
                 output="out.txt")
  monkeypatch.setattr("os.access", lambda *a, **k: True)
  monkeypatch.setattr(subprocess, "run", fake_run)
  assert ev.eval([1.0, 2.0]) == ([5.0, [-1.0]], False)
  assert ev.eval([3.0, 0.5]) == ([4.0, [-1.0]], False)

  def failing_run(cmd, **kwargs):
    raise subprocess.TimeoutExpired(cmd, 1)

  monkeypatch.setattr(subprocess, "run", failing_run)
  f, err = ev.eval([0.0, 0.0])
  assert err and f[0] == [float("inf")]


def test_coordinates_file_survives_empty_batches(tmp_path):
  import json
  from OMADS._post._postprocess import Output, PostMADS
  out = Output(file_path=str(tmp_path), vnames=["x1", "x2"], fnames=["f"],
               pname="c", runfolder="c_run")
  x0, x1 = _point([0, 0], [1]), _point([1, 0], [0])
  post = PostMADS(x_incumbent=[x0])
  # Three batches (one empty) for a single recorded incumbent.
  for batch, center in (([x1], x0), ([], None), ([x0], x1)):
    post.coords.append(batch)
    post.coords_centers.append(center)
  post.output_coordinates(out)
  saved = json.load(open(tmp_path / "c_run" / "coords.json"))
  assert [saved[k]["x_incumbent"] for k in ("1", "2", "3")] == \
      [[0, 0], [0, 0], [1, 0]]


def test_callable_exception_is_a_failed_evaluation():
  from OMADS import StopEvaluation
  from OMADS._evaluator._evaluator import Evaluator
  inf = float("inf")

  def crash(x):
    raise ValueError("diverged")

  assert Evaluator(blackbox=crash).eval([0.0]) == ([[inf], [inf]], True)
  assert Evaluator(blackbox=crash, nobj=2).eval([0.0]) == \
      ([[inf, inf], [inf]], True)

  def stop(x):
    raise StopEvaluation

  with pytest.raises(StopEvaluation):
    Evaluator(blackbox=stop).eval([0.0])
  # A wrong signature is an input error, not a failed evaluation.
  with pytest.raises(IOError):
    Evaluator(blackbox=lambda x, y, z: 0.0).eval([0.0])


@pytest.mark.parametrize("solver,nobj", [
    ("poll", 1), ("mads", 1), ("poll", 2), ("mads", 2)])
def test_solvers_survive_a_failing_blackbox(tmp_path, solver, nobj):
  from OMADS import mads, poll

  def blackbox(x):
    if x[0] > 0.5:
      raise RuntimeError("simulation crashed")
    return _bi(x) if nobj == 2 else [[(x[0] - 0.2) ** 2 + x[1] ** 2], [0]]

  param = {"name": "s", "baseline": [0.3, 0.6], "lb": [-2, -2],
           "ub": [2, 2], "var_names": ["x1", "x2"], "post_dir": str(tmp_path)}
  if nobj == 2:
    param.update(fun_names=["f1", "f2"], nobj=2, is_pareto=True)
  data = {"evaluator": {"blackbox": blackbox}, "param": param,
          "options": {"seed": 0, "budget": 80, "check_cache": True,
                      "store_cache": True, "display": False,
                      "save_results": True},
          "search": {"type": "sampling", "s_method": "LH", "ns": 8,
                     "visualize": False}}
  out = {"mads": mads.main, "poll": poll.main}[solver](data)
  out = out[0] if isinstance(out, tuple) else out
  rows = _read_out(tmp_path / "s_run" / "s_all.out")
  assert out["nbb_evals"] == len(rows) == 80
  assert out["xmin"][0] <= 0.5
