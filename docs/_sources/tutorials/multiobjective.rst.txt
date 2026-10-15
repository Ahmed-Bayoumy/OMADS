Multi-objective optimization
############################

OMADS can approximate the Pareto front of problems with several objectives. The blackbox
returns a list of objective values, and three parameters switch on the Pareto study:

.. code-block:: python

   def blackbox(x):
       return [[f1, f2], [c1, c2]]          # objectives first, then constraints

   param = {..., "nobj": 2, "is_pareto": True, "fun_names": ["f1", "f2"]}

Internally the adaptive barrier keeps the whole set of non-dominated points, each with its
own mesh, and the poll picks its frame centre where the front has the largest gaps (see
:numref:`A3`). The quality of the front is measured with the **hypervolume** indicator.

The Binh and Korn problem
=========================

.. math::

   \min_{x \in [0,5]\times[0,3]} \; \bigl(4x_1^2 + 4x_2^2,\; (x_1-5)^2 + (x_2-5)^2\bigr).

.. literalinclude:: scripts/multiobjective.py
   :language: python
   :caption: multiobjective.py

.. code-block:: text

   hypervolume = 0.8435
   380 non-dominated points, e.g.
         x1       x2        f1        f2
   0.000000 0.000000  0.000000 50.000000
   0.750000 0.718750  4.316406 36.391602
   1.593750 1.515625 19.348633 23.743408
   2.500000 2.453125 49.071289 12.736572
   3.484375 3.000000 84.563477  6.297119

The front runs from :math:`(0, 50)` at :math:`x = (0, 0)` to the region around
:math:`x_2 = 3`, where the upper bound becomes active. ``out["xmin"]`` and ``out["fmin"]``
hold one point of the front; the whole front is in the results file.

.. important::

   In this release, use ``poll.main`` for multi-objective problems. ``mads.main`` stops with
   an ``IndexError`` while writing the Pareto file, and the ``ACTIVE`` sampling search
   fails on multi-objective problems. Also, ``<name>_Pareto.out`` only keeps the starting
   point, which is why the script extracts the front from ``<name>_all.out``.

Gallery
=======

Pareto fronts obtained by OMADS on bi-objective test problems from the literature.

.. figure:: ../_static/ConstrainedMO.png
   :width: 100%

   Constrained bi-objective test problems.

.. figure:: ../_static/UnconstrainedMO.png
   :width: 100%

   Unconstrained bi-objective test problems.
