API reference
=============

The public entry points are the three solver modules. Everything else is used internally by
them, and is documented for users who want to extend OMADS or inspect a run.

:doc:`solvers`
   ``mads.main``, ``poll.main``, ``search.main``.
:doc:`configuration`
   ``Options``, ``Parameters``, ``Evaluator``, ``search_sampling``.
:doc:`components`
   Barrier, poll directions, search samplers, meshes, steps and updates.
:doc:`points`
   ``Point``, ``CandidatePoint``, ``Cache``.
:doc:`results`
   ``PostMADS``, ``Output``, ``Metrics`` and the run state.
:doc:`constants`
   Enumerations and global limits.

.. toctree::
   :hidden:

   solvers
   configuration
   components
   points
   results
   constants
