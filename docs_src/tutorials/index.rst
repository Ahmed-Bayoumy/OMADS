Tutorials
#########

This part presents complete, runnable examples. Each tutorial includes a script
from ``docs_src/tutorials/scripts`` together with the output it produces.

:doc:`toy_problems`
   Rosenbrock, Alpine, Ackley 3 and Eggholder in two dimensions; comparison of the three
   solvers; choice of the search step for multimodal functions.
:doc:`constrained`
   Progressive and extreme barriers, starting from an infeasible point, interpretation of
   ``hmin``.
:doc:`multiobjective`
   A Pareto study, the hypervolume indicator and the extraction of the non-dominated set.
:doc:`blackbox`
   User constants, external executables, failed evaluations, serial and parallel
   evaluation.
:doc:`postprocessing`
   The evaluation history, convergence plots and the poll-trajectory file.

.. toctree::
   :hidden:

   toy_problems
   constrained
   multiobjective
   blackbox
   postprocessing

.. note::

   All outputs shown in these tutorials were produced with OMADS 2610 on Windows (where
   ``"precision": "high"`` falls back to ``float64``). Runs are reproducible for a given
   seed and platform; other platforms may differ in the last digits.
