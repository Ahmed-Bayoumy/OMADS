Blackbox evaluation
###################

This tutorial covers the options that help you connect OMADS to real blackboxes and run them
efficiently.

User parameters
===============

Some blackboxes need constant parameters, such as material properties or a case name. Put
them in the ``constants`` list of the ``evaluator`` dictionary and add a second argument to
the callable. OMADS then calls ``blackbox(x, constants)``: the first argument is reserved
for the design variables and the second for the ``constants`` list.

.. literalinclude:: scripts/constants.py
   :language: python
   :caption: constants.py

.. code-block:: text

   xmin = [1.0, 1.0]  fmin = [0.0]

.. note::

   A callable may also be declared as ``def blackbox(x, *argv)``; the constants then arrive
   in ``argv[0]``. Any other signature with more than one required argument is rejected.

External executables
====================

Most engineering blackboxes are programs that read an input file and write an output file.
The most robust way to drive them is a small Python wrapper. It gives you full control of the
command line, the working directory, time-outs and error handling.

The "simulator" below stands in for your program. It reads one design variable per line
from ``input.inp`` and writes the objective, then one constraint per line, to
``output.out``:

.. literalinclude:: scripts/simulator.py
   :language: python
   :caption: simulator.py

The wrapper writes the input, runs the program and parses the output. When the run fails or
times out, it returns ``inf`` values, so OMADS rejects the point and carries on:

.. literalinclude:: scripts/executable.py
   :language: python
   :caption: executable.py
   :emphasize-lines: 12-21

.. code-block:: text

   xmin = [0.5 1.5]  fmin = [0.5]  evaluations = 157

The constrained minimum of :math:`(x_1-1)^2 + (x_2-2)^2` subject to :math:`x_1 + x_2 \le 2`
is :math:`f(0.5, 1.5) = 0.5`.

.. dropdown:: Built-in executable mode (``evaluator["blackbox"]`` as a file name)

   OMADS can also run an executable directly when ``blackbox`` is a file name. Set ``path``
   (the working directory), ``input``, ``output``, and optionally ``command_options`` and
   ``timeout``. OMADS runs ``./<blackbox> <command_options>`` (on Windows, the
   ``command_options`` string, or the blackbox name) and reads one value per line from the
   output file.

   .. warning::

      In this release the built-in mode does **not** write the input file before running
      the executable, so the program reads stale inputs. Use the wrapper approach above
      until this is fixed.

Serial mode
===========

Serial evaluation is the default. With ``"parallel_mode": False`` the evaluator runs the
blackbox one point at a time, whatever the number of processes ``np``.

.. code-block:: python

   "options": {
       "seed": 0, "budget": 10000, "tol": 1e-12,
       "check_cache": True, "store_cache": True, "rich_direction": True,
       "parallel_mode": False,
   }

Parallel mode
=============

OMADS can evaluate the poll and search candidate sets in parallel processes. This is
worthwhile when the blackbox is expensive. Set ``parallel_mode`` to ``True`` and ``np`` to
the number of worker processes. ``np`` is capped at the number of CPU cores, and the worker
pool is created once and reused for the whole run.

.. literalinclude:: scripts/parallel.py
   :language: python
   :caption: parallel.py

.. code-block:: text

   parallel=False  fmin=0.000e+00  evals=120  wall time=6.3s
   parallel=True   fmin=0.000e+00  evals=120  wall time=5.3s

The gain here is modest. Polls of a 4-variable problem only produce 8 candidates, the model
search evaluates its one or two points serially, and starting the worker processes takes
time. The longer each evaluation and the larger the candidate sets (higher dimension, larger
``ns``), the closer the speed-up gets to ``np``.

.. important::

   * The blackbox must be **picklable**: a module-level function, not a lambda or a nested
     function.
   * Guard the script's entry point with ``if __name__ == "__main__":``, because worker
     processes re-import it on Windows and macOS.
