Getting started
===============

Installation
------------

OMADS needs Python 3.10 or newer.

From the Python Package Index:

.. code-block:: console

   $ pip install OMADS

The optional plotting dependencies (``matplotlib``, ``plotly``) are installed with
``pip install "OMADS[visualization]"``. To install from source and run the test suite:

.. code-block:: console

   $ git clone https://github.com/Ahmed-Bayoumy/OMADS.git
   $ cd OMADS
   $ pip install -e ".[test]"
   $ pytest tests

OMADS pulls in ``numpy``, ``scipy``, ``pandas``, ``pydantic``, ``deap``, ``paramiko`` and
`samplersLib <https://pypi.org/project/samplerslib/>`_ (the design-of-experiments and
density-guided samplers used by the search step).

Your first optimization
-----------------------

A run consists of three steps:

#. write the blackbox, a Python callable that takes the design vector ``x`` and returns
   ``[f, [c1, c2, ...]]``;
#. describe the problem in a dictionary (or JSON file) with the ``evaluator``, ``param``,
   ``options`` and ``search`` sections;
#. call a solver: ``mads.main(data)``, ``poll.main(data)`` or ``search.main(data)``.

.. code-block:: python
   :caption: first_run.py
   :emphasize-lines: 6,23

   import numpy as np
   from OMADS import mads

   def rosenbrock(x):
       x = np.asarray(x)
       return [np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2), [0]]

   data = {
       "evaluator": {"blackbox": rosenbrock},
       "param": {
           "name": "rosen",                 # used to name the output folder
           "baseline": [-2.0, -2.0],        # starting point x0
           "lb": [-5, -5], "ub": [10, 10],  # bounds
           "scaling": [15, 15],             # see "Scaling" below
           "post_dir": "./post",            # log and result files go here
       },
       "options": {"seed": 0, "budget": 1000, "tol": 1e-9, "rich_direction": True,
                   "check_cache": True, "store_cache": True},
       "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 10},
   }

   if __name__ == "__main__":
       out, poll_step, search_step = mads.main(data)
       print(out["xmin"], out["fmin"], out["nbb_evals"])

.. code-block:: text

   [1.0, 1.0] [0.0] 223

.. important::

   Guard the call with ``if __name__ == "__main__":`` in scripts. OMADS uses
   :mod:`multiprocessing` when ``parallel_mode`` is on, and on Windows and macOS worker
   processes re-import your script.

The blackbox contract
---------------------

The evaluator calls your function with a list of floats, ordered like ``baseline``, and
expects back a two-item list:

.. code-block:: python

   def blackbox(x):
       f = ...                       # objective: a float, or a list of floats for
                                     # multi-objective problems
       c = [c1, c2, ...]             # inequality constraints, feasible when c_j(x) <= 0
       return [f, c]                 # unconstrained problems: return [f, [0]]

* A bare ``float`` return is accepted and treated as ``[f, [0]]``.
* Add a second positional argument to receive the ``evaluator["constants"]`` list:
  ``def blackbox(x, p): ...`` (see :doc:`tutorials/blackbox`).
* Return ``numpy.inf`` (or ``nan``) for a failed evaluation; OMADS marks the point
  infeasible or as an error and keeps going.

Choosing a solver
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Entry point
     - What it runs
     - Use it when
   * - ``OMADS.mads.main``
     - Model search, then poll, plus the sampling search after unsuccessful polls.
     - Default choice. Most robust on multimodal and constrained problems.
   * - ``OMADS.poll.main``
     - Poll step only (no search).
     - You want the plain, deterministic-step MADS behaviour, a multi-objective Pareto
       study, or the poll trajectory (``save_coordinates``).
   * - ``OMADS.search.main``
     - Sampling search only, on the mesh.
     - Exploring a design space, or comparing search strategies.

All three take the same input (a dictionary or the path to a JSON file) and return a tuple
whose first item is the results dictionary (see :doc:`user_guide/outputs`):

.. code-block:: python

   out, poll_step, search_step = mads.main(data)   # three items
   out, poll_step = poll.main(data)                # two items
   out, search_step = search.main(data)            # two items

Using a JSON file
-----------------

The same problem can live in a JSON file, with a string naming an executable in
``evaluator.blackbox`` instead of a Python callable:

.. code-block:: json
   :caption: problem.json

   {
     "evaluator": {"blackbox": "my_simulator", "path": "./sim",
                   "input": "input.inp", "output": "output.out"},
     "param":     {"name": "sim", "baseline": [0.5, 0.5], "lb": [0, 0], "ub": [1, 1],
                   "scaling": [1, 1], "post_dir": "./post"},
     "options":   {"seed": 0, "budget": 500, "check_cache": true, "store_cache": true},
     "search":    {"type": "sampling", "s_method": "LH", "ns": 10}
   }

.. code-block:: python

   from OMADS import mads
   out, *_ = mads.main("problem.json")

See :doc:`tutorials/blackbox` for the file protocol of external executables.

Scaling and the initial frame
-----------------------------

Poll steps are expressed in units of :math:`(u_i - \ell_i)/s_i` where :math:`s_i` is
``param["scaling"]`` (a scalar or one value per variable). Setting ``scaling`` to the
variable ranges, ``ub - lb``, makes the frame size equal to a step in the original units;
``options["psize_init"]`` (default ``1.0``) is then the size of the first poll step for the
default orthogonal mesh. With ``"mesh_type": "GMESH"`` the initial frame is instead one
tenth of each variable range.

.. tip::

   Problems whose variables span very different magnitudes (for example ``1e-6`` to
   ``1e6``) converge much better when ``scaling`` is set per variable.

Next steps
----------

* Walk through the :doc:`tutorials/index`.
* Look up any key in the :doc:`user_guide/configuration`.
* Read how the algorithm works in :doc:`background`.
