Constrained optimization
########################

OMADS handles inequality constraints :math:`c_j(x) \le 0` returned by the blackbox. Each
constraint gets a barrier type in ``param["constraints_type"]``:

Progressive barrier (``"PB"``)
   The constraint is **relaxable**. Infeasible points are retained and ranked by their
   violation :math:`h(x) = \sum_j \max\{c_j(x), 0\}^2`, and the threshold :math:`h_{\max}`
   decreases over the iterations to drive the search towards the feasible region
   :cite:p:`audet2009progressive`.
Extreme barrier (``"EB"``)
   The constraint is **unrelaxable** (for example, the simulation cannot be run when it is
   violated), and any point that violates it is rejected. This is the default for
   constraints without a declared type.

Prefer ``"PB"`` whenever the blackbox can still be evaluated at points that violate the
constraint. The barrier then also learns from infeasible points, and you can start from an
infeasible baseline.

A thin feasible region
======================

The feasible region of this problem is a thin band of height :math:`0.1` that follows a sine
wave, :math:`\sin(x_1) - 0.1 \le x_2 \le \sin(x_1)`. The objective is the distance to
:math:`(20, 1)`, which lies just outside the band, and the starting point
:math:`(0, -10)` is far from feasible.

.. literalinclude:: scripts/constrained.py
   :language: python
   :caption: constrained.py

.. code-block:: text

   xmin = [20.0288661159575, 0.9243430346250534]
   fmin = [0.08097671924829766]  hmin = 1e-15
   evaluations: 536

The run starts in *phase one*, minimizing :math:`h` until a feasible point is found, then
improves the objective along the band. The reported ``hmin`` of ``1e-15`` is the machine
zero, which means the solution is feasible. The best known value is :math:`f \approx 0.0989`
at the start of the band, so this run found an even better point near the band's upper
edge.

Reading the results
===================

* ``out["hmin"]`` is the violation of the returned point. A value above the machine zero
  means OMADS found no feasible point within the budget, and ``xmin`` is then the least
  infeasible point.
* ``poll.main`` also returns ``out["Secondary fmin"]`` and ``out["Secondary hmin"]``, the
  best infeasible point kept by the progressive barrier.
* With ``options["save_results"]`` the ``Status`` column of the results file shows
  ``FEASIBLE`` or ``INFEASIBLE`` for each evaluation; see :doc:`postprocessing`.

.. tip::

   * Scale the constraints so that their violations have comparable magnitudes, because
     :math:`h` sums their squares.
   * Mixing barrier types is allowed: ``"constraints_type": ["PB", "EB", "PB"]``.
   * Set ``param["h_max"]`` to a finite value to reject, from the start, points that are
     too infeasible.
