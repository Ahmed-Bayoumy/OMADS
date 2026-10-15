"""A stand-in for an external simulation code.

Reads one design variable per line from input.inp, writes the objective and
then one constraint per line to output.out.
"""
import numpy as np

x = np.loadtxt("input.inp", ndmin=1)
f = (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2
c = x[0] + x[1] - 2.0                  # feasible when x1 + x2 <= 2
np.savetxt("output.out", [f, c])
