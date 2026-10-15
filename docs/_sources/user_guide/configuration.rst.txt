Configuration reference
=======================

An OMADS run is described by one dictionary (or JSON file) with up to four sections:

.. code-block:: python

   data = {
       "evaluator": {...},   # how to evaluate a point          (required)
       "param":     {...},   # the problem: variables, bounds   (required)
       "options":   {...},   # algorithmic and run options      (required)
       "search":    {...},   # the search step                  (required by mads and search)
   }

Keys are passed straight to the constructors of :class:`~OMADS._evaluator._evaluator.Evaluator`,
:class:`~OMADS._setup._parameters.Parameters`, :class:`~OMADS._setup._options.Options` and
``search_sampling``, so a misspelled key raises a ``TypeError`` instead of being ignored.

.. contents:: On this page
   :local:
   :depth: 1

``evaluator``
-------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60
   :class: omads-ref

   * - Key
     - Default
     - Description
   * - ``blackbox``
     - ``"BB"``
     - A Python callable, or the file name of an executable (in ``path``).
   * - ``constants``
     - ``None``
     - A list passed as the second argument of the callable, ``blackbox(x, constants)``.
   * - ``path``
     - ``"..\\tests\\Rosen"``
     - Working directory of an executable blackbox.
   * - ``input``
     - ``"input.inp"``
     - Input file name of an executable blackbox (one design variable per line).
   * - ``output``
     - ``"output.out"``
     - Output file name of an executable blackbox: the objective on the first line, then
       one constraint per line.
   * - ``command_options``
     - ``None``
     - Extra arguments appended to the executable command (on Windows, the whole command).
   * - ``timeout``
     - ``1000000``
     - Seconds before an executable evaluation is killed and counted as a failure.
   * - ``internal``
     - ``None``
     - Must stay ``None``. The internal benchmarking suites (``"con"``, ``"uncon"``) of older
       releases were removed and now raise ``IOError``.

``param``
---------

Problem definition
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 18 60
   :class: omads-ref

   * - Key
     - Default
     - Description
   * - ``baseline``
     - *required*
     - Starting point :math:`x^0`. A list of lists gives several starting points; all are
       evaluated and the best one is used.
   * - ``lb``, ``ub``
     - *required*
     - Lower and upper bounds, one per variable.
   * - ``name``
     - ``"undefined"``
     - Run name, used for the output folders and files.
   * - ``post_dir``
     - current directory
     - Folder for ``OMADS.log`` and the result files (created if missing).
   * - ``var_names``
     - ``x_0, x_1, ...``
     - Variable names (column headers of the result files).
   * - ``fun_names``
     - ``["fobj"]``
     - Objective names (column headers of the result files). Give one per objective in
       multi-objective runs.
   * - ``scaling``
     - ``10.0``
     - Scalar or per-variable divisor :math:`s`: poll steps are expressed in units of
       :math:`(u-\ell)/s`. Use ``ub - lb`` to work in the original units.
   * - ``var_type``
     - all ``"R"``
     - Variable types: ``"R"`` real, ``"I"`` integer, ``"D_<set>"`` discrete and
       ``"C_<set>"`` categorical (values from ``var_sets``), ``"O"`` ordinal, ``"B"``
       binary. See the warning below.
   * - ``var_sets``
     - ``None``
     - Dictionary of named value lists used by ``D_`` and ``C_`` variables. The variable's
       coordinate is an index into its set.
   * - ``constants``
     - ``None``
     - Same as ``evaluator["constants"]`` (copied to the evaluator).

.. warning::

   Mixed-variable support (``"I"``, ``"D_*"``, ``"C_*"``, ``"B"``) is **experimental** in
   this release: the poll directions for non-real variables are not reliable yet. Prefer
   real variables and round (or index into a list of values) inside the blackbox.

Constraints
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 18 60
   :class: omads-ref

   * - Key
     - Default
     - Description
   * - ``constraints_type``
     - all ``"EB"``
     - One entry per constraint: ``"PB"`` progressive barrier (relaxable) or ``"EB"``
       extreme barrier (unrelaxable). ``"PEB"`` and ``"RB"`` are accepted and currently
       behave like ``"PB"``. Missing entries default to ``"EB"``.
   * - ``h_max``
     - ``inf``
     - Initial barrier threshold :math:`h^0_{\max}`.
   * - ``rho``
     - ``5e-05``
     - Trigger for picking the infeasible frame centre first, and penalty parameter for
       points with :math:`h \ge h_{\max}`.
   * - ``lambda_multipliers``
     - ``[0]``
     - Initial Lagrange multipliers (scalar or list) used by that penalty.

Mesh
^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 18 60
   :class: omads-ref

   * - Key
     - Default
     - Description
   * - ``mesh_type``
     - ``"ORTHO"``
     - ``"ORTHO"`` orthogonal mesh or ``"GMESH"`` granular, anisotropic mesh.
   * - ``granularity``
     - all ``0``
     - (GMESH) Per-variable granularity (``0`` for continuous variables). Non-zero values
       are experimental and currently fail when the mesh is built.
   * - ``initial_frame_size``, ``initial_mesh_size``
     - from bounds
     - (GMESH) Initial :math:`\Delta^0` and :math:`\delta^0`; by default
       :math:`\Delta^0_i = (u_i - \ell_i)/10`.
   * - ``min_frame_size``, ``min_mesh_size``
     - ``1e-9``
     - (GMESH) Lower limits for the frame and mesh sizes.
   * - ``fixed_variables``
     - ``None``
     - Reserved.
   * - ``mesh_adjustment``
     - ``2.0``
     - Reserved (the orthogonal mesh always doubles and halves).

Multi-objective and barrier
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 18 60
   :class: omads-ref

   * - Key
     - Default
     - Description
   * - ``nobj``
     - ``1``
     - Number of objectives returned by the blackbox.
   * - ``is_pareto``
     - ``False``
     - Run a Pareto study (required when ``nobj > 1``).
   * - ``ref_point``
     - ``None``
     - Reference point for the hypervolume indicator; computed from the front when omitted.
   * - ``function_weights``
     - equal
     - Objective weights (normalized to sum to one).
   * - ``w_min``
     - ``1``
     - A feasible point can become a frame centre only if its frame size is at least
       :math:`10^{-w}` times the largest one.
   * - ``lhs_search_initialization``
     - ``False``
     - Use :math:`(n+1)(n+2)` Latin hypercube samples at the first search step.
   * - ``design_criteria``, ``best_known``, ``problem_name``, ``failure_stop``,
       ``incumbent_selection_param``, ``barrier_initialized_from_cache``,
       ``constants_name``
     -
     - Informational or reserved.

``options``
-----------

.. list-table::
   :header-rows: 1
   :widths: 26 14 60
   :class: omads-ref

   * - Key
     - Default
     - Description
   * - ``seed``
     - ``0``
     - Random seed. The same seed and input give the same run.
   * - ``budget``
     - ``1000``
     - Maximum number of blackbox evaluations.
   * - ``tol``
     - ``1e-9``
     - Stop when every frame (poll) or mesh (search) size component is below this value.
   * - ``psize_init``
     - ``1.0``
     - Initial frame size of the orthogonal mesh (scalar or list).
   * - ``store_cache``
     - ``False``
     - Store evaluated points in the cache. **Set it to** ``True``: ``mads`` and ``search``
       need the cache to locate frame centres.
   * - ``check_cache``
     - ``False``
     - Skip candidates already in the cache. Recommended: ``True``.
   * - ``rich_direction``
     - ``False``
     - Random Householder directions (asymptotically dense). Recommended: ``True``.
   * - ``direction_type``
     - ``"ORTHO_2N"``
     - ``"ORTHO_2N"`` (:math:`2n` poll points) or ``"ORTHO_N+1"`` (:math:`n+1`; real
       variables only, otherwise falls back to :math:`2n`).
   * - ``opportunistic``
     - ``False``
     - Stop evaluating the current poll or search set at the first full success.
   * - ``parallel_mode``
     - ``False``
     - Evaluate candidate sets in parallel processes.
   * - ``np``
     - ``1``
     - Number of worker processes (capped at the number of CPU cores).
   * - ``precision``
     - ``"high"``
     - ``"high"`` (``float128`` where numpy has it, else ``float64``), ``"medium"``
       (``float64``) or ``"low"`` (``float32``).
   * - ``display``
     - ``False``
     - Print the evaluation of the starting point and a run summary.
   * - ``is_verbose``
     - ``False``
     - Write extra diagnostics (e.g. the sampling-search policy) to ``OMADS.log``.
   * - ``save_results``
     - ``False``
     - Write every evaluation to ``<post_dir>/<name>_run/<name>_all.out`` (see
       :doc:`outputs`).
   * - ``save_coordinates``
     - ``False``
     - Write the poll points of each iteration to ``coords.json`` (``poll.main`` only).
   * - ``save_all_best``
     - ``False``
     - Reserved.
   * - ``use_dom_trigger``
     - ``True``
     - Choose the primary frame centre with the dominance-distance trigger (see
       :numref:`A3`).
   * - ``use_omads_partial_success``
     - ``True``
     - When ``False``, an unsuccessful iteration becomes a partial success if some
       infeasible point has a lower :math:`h` than the infeasible centre.
   * - ``use_dms_success``
     - ``False``
     - Count any improvement of the non-dominated set as a full success (DMS rule).
   * - ``anisotropy_factor``
     - ``0.1``
     - (GMESH) Threshold of the anisotropic enlargement rule.
   * - ``anistropic_mesh``
     - ``True``
     - (GMESH) Enable anisotropic enlargement. Note the spelling of the key.
   * - ``refine_freq``
     - ``1``
     - (GMESH) Refine the frame every ``refine_freq`` unsuccessful iterations.
   * - ``noutbound_hits_max``
     - ``60000``
     - Stop after this many trial points hit the bounds.
   * - ``collect_y``, ``extend``, ``use_penalty_approach``
     -
     - Reserved.

``search``
----------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60
   :class: omads-ref

   * - Key
     - Default
     - Description
   * - ``type``
     - ``"SAMPLING"``
     - ``"sampling"`` (any value other than ``"VNS"``) or ``"VNS"`` (variable neighbourhood
       search).
   * - ``s_method``
     - ``"LH"``
     - ``"ACTIVE"`` (adaptive: BiTPE, diversifiers or Halton), ``"LH"`` Latin hypercube,
       ``"RS"`` random, ``"HALTON"`` or ``"FULLFACTORIAL"``.
   * - ``ns``
     - ``3``
     - Samples per search step. ``None`` uses :math:`(n+1)(n+2)/4`.
   * - ``criterion``
     - ``None``
     - Space-filling criterion of the Latin hypercube sampler (for example ``"maximin"``).
   * - ``weights``
     - ``None``
     - Level weights of the full-factorial sampler.
   * - ``visualize``
     - ``False``
     - Reserved.

.. tip::

   ``"ACTIVE"`` is the best general-purpose choice. On strongly multimodal problems a
   space-filling ``"LH"`` search with a few dozen samples often finds a better basin; see the
   Eggholder example in :doc:`../tutorials/toy_problems`.

Complete example
----------------

As a Python dictionary:

.. code-block:: python

   data = {
       "evaluator": {"blackbox": my_function},
       "param": {
           "name": "demo",
           "baseline": [0.5, 0.5],
           "lb": [0.0, 0.0],
           "ub": [1.0, 1.0],
           "var_names": ["x1", "x2"],
           "scaling": [1.0, 1.0],
           "constraints_type": ["PB", "EB"],
           "post_dir": "./post",
       },
       "options": {
           "seed": 0, "budget": 500, "tol": 1e-9,
           "check_cache": True, "store_cache": True, "rich_direction": True,
           "save_results": True,
       },
       "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 10},
   }

The same problem as a JSON file, with an executable blackbox:

.. code-block:: json

   {
     "evaluator": {"blackbox": "my_simulator", "path": "./sim"},
     "param": {
       "name": "demo",
       "baseline": [0.5, 0.5],
       "lb": [0.0, 0.0],
       "ub": [1.0, 1.0],
       "var_names": ["x1", "x2"],
       "scaling": [1.0, 1.0],
       "constraints_type": ["PB", "EB"],
       "post_dir": "./post"
     },
     "options": {
       "seed": 0, "budget": 500, "tol": 1e-9,
       "check_cache": true, "store_cache": true, "rich_direction": true,
       "save_results": true
     },
     "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 10}
   }
