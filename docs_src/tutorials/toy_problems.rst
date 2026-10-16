Toy problems
############

The following subsections show how to use OMADS on selected 2D test functions.

.. tip::

   The videos below show the poll trajectory, frame size and mesh coarseness during an
   optimization. They were recorded with an earlier release of OMADS, so the path differs
   from what the current version takes, but the mechanics are the same.

2D Rosenbrock
=============

.. math::

   f(x_{1}, x_{2})=(1-x_{1})^{2}+100(x_{2}-x_{1}^{2})^{2}.

The Rosenbrock function has curved contours, and its global minimum
:math:`f(1, 1) = 0` lies in a parabolic, banana-shaped valley. Many algorithms find the
valley easily but struggle to reach the minimum, because the function values in the valley
barely change.

The script below solves it with each of the three OMADS solvers:

.. literalinclude:: scripts/rosenbrock.py
   :language: python
   :caption: rosenbrock.py

.. code-block:: text

     mads: xmin = [1.0, 1.0], fmin = 0.000e+00, iterations = 21
     poll: xmin = [0.9390255808830261, 0.8823902606964111], fmin = 3.756e-03, iterations = 253
   search: xmin = [-0.602294921875, 0.409423828125], fmin = 2.785e+00, iterations = 169

Full MADS reaches the exact minimum in 223 evaluations. Its quadratic model search makes
large, well-aimed steps along the valley. The poll alone crawls along the valley with
orthogonal steps and is still on its way when the 1000-evaluation budget runs out, and the
sampling search alone is an exploration tool, not a local solver. Use ``mads`` unless you
have a reason not to.

.. video:: ../media/videos/RB.mp4
   :width: 650
   :height: 400
   :autoplay:

2D Alpine
=========

.. math::

   f({\bf x})= \sum_{i=1}^{d} |x_{i} \sin(x_{i})+0.1x_{i}|.

The Alpine function is non-smooth, which can stop gradient-based algorithms from reaching
the global minimum. Its design space is non-smooth, non-convex, separable and multimodal,
and it has several global minima with :math:`f = 0`, for example at the origin and near
:math:`x_i \approx 6.18`.

.. literalinclude:: scripts/alpine.py
   :language: python
   :caption: alpine.py

.. code-block:: text

   xmin = [6.183017810806632, 6.183017225936055]
   fmin = [4.5235426833212244e-06] after 270 evaluations

.. video:: ../media/videos/Alpine.mp4
   :width: 650
   :height: 400
   :autoplay:

2D Ackley 3
===========

Ackley 3 combines a large funnel with a bumpy exponential term, which can trap
hill-climbing algorithms in one of its many sub-optimal solutions:

.. math::

   f(x_{1}, x_{2})= -200\, e^{-0.2 \sqrt{x^{2}_{1} + x^{2}_{2}}} + 5\, e^{\cos(3x_{1})+\sin(3x_{2})}.

Its design space is non-convex, non-separable and multimodal. The global minimum is
:math:`f \approx -186.4112` near :math:`(0, \pm 0.0068)`.

.. literalinclude:: scripts/ackley3.py
   :language: python
   :caption: ackley3.py

.. code-block:: text

   xmin = [4.852190613746643e-06, -0.006760597229003906]
   fmin = [-186.41121263368007] after 253 evaluations

.. video:: ../media/videos/Ackley3.mp4
   :width: 650
   :height: 400
   :autoplay:

2D Eggholder
============

The Eggholder function is a classic benchmark for global optimizers. Its many deep, irregular
basins make the single global minimum, :math:`f(512, 404.2319) \approx -959.6407`, hard
to find:

.. math::

   f(x_{1}, x_{2})= -(x_{2}+47)\sin\left(\sqrt{\left|\tfrac{x_{1}}{2}+x_{2}+47\right|}\right)-x_{1}\sin\left(\sqrt{|x_{1}-(x_{2}+47)|}\right).

This example runs MADS twice, with the default ``ACTIVE`` search and with a space-filling
Latin hypercube search of 30 points per step:

.. literalinclude:: scripts/eggholder.py
   :language: python
   :caption: eggholder.py

.. code-block:: text

   ACTIVE search: fmin = -493.9605 at [150.2320980578661, 301.3137746155262]
   LH search:     fmin = -959.6407 at [512.0, 404.2317980080843]

The ``ACTIVE`` search concentrates on the neighbourhood of the incumbent, which is efficient
on most problems but converges here to a local basin. The Latin hypercube search keeps
sampling the whole box, lands in the right basin and the poll finishes the job on the bound
:math:`x_1 = 512`.

.. admonition:: Choosing the search for multimodal functions

   On strongly multimodal functions, try ``"s_method": "LH"`` with ``ns`` between
   :math:`5n` and :math:`20n`, try several seeds, or start from several points by giving
   ``baseline`` as a list of points.

.. video:: ../media/videos/EggHolder.mp4
   :width: 650
   :height: 400
   :autoplay:
