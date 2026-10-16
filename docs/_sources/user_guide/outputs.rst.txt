Outputs
=======

Return value
------------

Every entry point returns a tuple whose first item is a results dictionary:

.. code-block:: python

   out, poll_step, search_step = mads.main(data)
   out, poll_step = poll.main(data)
   out, search_step = search.main(data)

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Content
   * - ``xmin``
     - Best point found. For ``D_`` and ``C_`` variables, the set values (not the indices).
   * - ``fmin``
     - Objective value(s) at ``xmin``, as a list (one entry per objective).
   * - ``hmin``
     - Constraint violation :math:`h` at ``xmin`` (``1e-15``, the machine zero, when feasible).
   * - ``nbb_evals``
     - Number of blackbox evaluations.
   * - ``niterations``
     - Number of iterations.
   * - ``nb_success``
     - Number of successful poll and search steps.
   * - ``psize``, ``psuccess``
     - Final frame size, and frame size of the best frame centre (one value per variable).
   * - ``msize``
     - Final mesh size of the best frame centre.
   * - ``HV``
     - Hypervolume of the Pareto front for multi-objective runs, ``"NA"`` otherwise.
   * - ``Secondary fmin``, ``Secondary hmin``
     - (``poll.main`` only) Objective and violation of the best infeasible point.

.. note::

   In this release ``poll.main`` reports ``nbb_evals = 1`` and ``nb_success = 0`` whatever
   the run did. Use ``niterations`` or the ``<name>_all.out`` file to count evaluations.

The second and third items are the poll (``Dirs2n``) and search (``EfficientExploration``)
objects at the end of the run, for users who want to inspect the final mesh or sampler.

Files in ``post_dir``
---------------------

.. code-block:: text

   post_dir/
   ├── OMADS.log                      # always: settings, per-iteration status, run summary
   ├── <name>_run/
   │   ├── <name>_all.out             # save_results=True: one row per evaluation
   │   └── coords.json                # save_coordinates=True (poll.main): poll points per iteration
   └── <name>_ND/
       └── <name>_Pareto.out          # multi-objective runs

``OMADS.log``
^^^^^^^^^^^^^

A plain-text log with one block per iteration (step, success status, best point so far) and
the termination reason. Set ``options["is_verbose"]`` for more detail, for example the
decisions of the active sampling search.

``<name>_all.out``
^^^^^^^^^^^^^^^^^^

A comma-separated file with right-aligned columns. Read it with pandas:

.. code-block:: python

   import pandas as pd

   df = pd.read_csv("post/rosen_run/rosen_all.out", skipinitialspace=True)
   df.columns = df.columns.str.strip()

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Content
   * - ``Runtime (Sec)``
     - Evaluation time.
   * - ``Iteration``, ``Evaluation #``
     - Iteration and evaluation counters.
   * - ``Step``
     - ``Poll_2n`` or ``Search``.
   * - ``Delta``
     - Smallest component of the frame size.
   * - ``Status``
     - ``FEASIBLE``, ``INFEASIBLE`` or ``ERROR``.
   * - ``phi``
     - Penalized objective used for comparisons.
   * - one column per ``fun_names`` entry
     - Objective value(s).
   * - ``max(c_in)``, ``hmax``
     - Barrier threshold at that time.
   * - ``Penalty_parameter``, ``Multipliers``
     - Penalty parameter and largest multiplier.
   * - one column per ``var_names`` entry
     - The design variables.

``coords.json``
^^^^^^^^^^^^^^^

The poll points of every iteration, with the incumbent, for animations of the poll
trajectory:

.. code-block:: json

   {
     "1": {
       "coord": [[-3.0, -3.0], [-3.0, -2.0]],
       "iter": 1,
       "x_incumbent": [-2.0, -2.0]
     },
     "2": {
       "coord": [[-2.25, -2.25], [-2.25, -1.5], [-1.75, -2.5], [-1.75, -1.75]],
       "iter": 2,
       "x_incumbent": [-2.0, -2.0]
     }
   }

.. note::

   ``save_coordinates`` works with ``poll.main``. With ``mads.main`` it raises an
   ``IndexError`` at the end of the run in this release.

Multi-objective files
^^^^^^^^^^^^^^^^^^^^^

``<name>_Pareto.out`` has the same columns as ``<name>_all.out``. In this release it only
keeps the starting point, so extract the front from ``<name>_all.out`` instead; see
:doc:`../tutorials/multiobjective`.
