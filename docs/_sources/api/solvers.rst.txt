Solvers
=======

.. code-block:: python

   from OMADS import mads, poll, search

All three functions take a single argument, the problem dictionary or the path to a JSON
file (see :doc:`../user_guide/configuration`), and return a tuple whose first item is the
results dictionary (see :doc:`../user_guide/outputs`).

``OMADS.mads``
--------------

.. autofunction:: OMADS._mads.mads.main

``OMADS.poll``
--------------

.. autofunction:: OMADS._poll.poll.main

``OMADS.search``
----------------

.. autofunction:: OMADS._search.search.main
