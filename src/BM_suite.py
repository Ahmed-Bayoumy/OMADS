# ------------------------------------------------------------------------------------#
#  Benchmarking suite for OMADS                                                       #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU Lesser General Public License as published by the Free Software   #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.   #
#                                                                                     #
#  You should have received a copy of the GNU Lesser General Public License along     #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on OMADS at                                               #
#  https://github.com/Ahmed-Bayoumy/RAF/OMADS                                         #
# ------------------------------------------------------------------------------------#

from dataclasses import dataclass, field
from typing import List
import copy
import numpy as np
from numpy import sum, subtract, cos, sin, prod, exp, pi, sqrt, e

import csv
import os


@dataclass
class DType:
    """ Data type delegator for decimal precision control """
    # default precision option
    _prec: str = "high"
    # numpy double data type precision
    _dtype: np.dtype = np.float64
    # numpy integer data type precision
    _itype: np.dtype = np.int_
    # Zero resolution value
    _zero: float = np.finfo(np.float64).resolution

    @property
    def zero(self):
        return self._zero

    @property
    def precision(self):
        return self._prec

    @precision.setter
    def precision(self, val: str):
        self._prec = val
        self._prec = val
        if val == "high":
            self.dtype = np.float128
            self._zero = np.finfo(np.float128).resolution
            self.itype = np.int_
        elif val == "medium":
            self.dtype = np.float64
            self._zero = np.finfo(np.float64).resolution
            self.itype = np.intc
        elif val == "low":
            self.dtype = np.float32
            self._zero = np.finfo(np.float32).resolution
            self.itype = np.short
        else:
            raise (IOError, " JASON parameters file; unrecognized textual input for the defined precision type. "
                            "Please enter one of these textual values (high, medium, low)")

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, other: np.dtype):
        self._dtype = other

    @property
    def itype(self):
        return self._itype

    @itype.setter
    def itype(self, other: np.dtype):
        self._itype = other


@dataclass
class UnconSO:
    """ This class contains benchmark uncostrained global test functions. """
    """ These tests are adopted from the following website (converted from their matlab version) ): """
    """ https://www.sfu.ca"""
    """ Note: Design variables used here are continuous, but in future other variable types shall be considered """
    # Design vector
    x: List[float]
    # function name
    _name: str = "rosenbrock"
    _dtype: DType = field(default_factory=DType)

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, fname: str):
        self._name = copy.deepcopy(fname)

    @name.deleter
    def name(self):
        del self._name

    def test_problem(self):
        if self.name == "ackley":
            self.ackley()
        elif self.name == "beale":
            self.beale()
        elif self.name == "dixonprice":
            self.dixonprice()
        elif self.name == "griewank":
            self.griewank()
        elif self.name == "levy":
            self.levy()
        elif self.name == "michalewicz":
            self.michalewicz()
        elif self.name == "perm":
            self.perm()
        elif self.name == "powell":
            self.powell()
        elif self.name == "powersum":
            self.powersum()
        elif self.name == "rastrigin":
            self.rastrigin()
        elif self.name == "rosenbrock":
            self.rosenbrock()
        elif self.name == "schwefel":
            self.schwefel()
        elif self.name == "sphere":
            self.sphere()
        elif self.name == "trid":
            self.trid()
        elif self.name == "zakharov":
            self.zakharov()
        else:
            raise RuntimeError(" Unnkown function name: please select one of the function names provided in "
                               "the ../bm/benchmark.txt file")

    def ackley(self) -> List[float]:
        """ https://www.sfu.ca/~ssurjano/ackley.html """
        return [-20.0 * exp(-0.2 * sqrt(0.5 * (self.x[0] ** 2 + self.x[1] ** 2), dtype=self._dtype.dtype),
                            dtype=self._dtype.dtype)
                - exp(0.5 * (cos(2 * pi * self.x[0], dtype=self._dtype.dtype) +
                             cos(2 * pi * self.x[1], dtype=self._dtype.dtype)), dtype=self._dtype.dtype) + e + 20,
                [0.0]]

    def beale(self) -> List[float]:
        """ https://www.sfu.ca/~ssurjano/beale.html """
        return [(((1.5 - self.x[0] + self.x[0] * self.x[1]) ** 2) +
                 ((2.25 - self.x[0] + self.x[0] * self.x[1] ** 2) ** 2) +
                 (2.625 - self.x[0] + self.x[0] * self.x[1] ** 3) ** 2), [0.0]]

    def dixonprice(self) -> List[float]:
        """ https://www.sfu.ca/~ssurjano/dixonpr.html """
        n = len(self.x)
        j = np.arange(2, n + 1)
        x2 = 2.0 * np.power(self.x, 2)
        return [(
                sum(j * subtract(x2[1:], self.x[:-1], dtype=self._dtype.dtype) ** 2, dtype=self._dtype.dtype) +
                (self.x[0] - 1) ** 2), [0.0]]

    def griewank(self, fr=4000) -> List[float]:
        """ https://www.sfu.ca/~ssurjano/griewank.html """
        n = len(self.x)
        j = np.arange(1., n + 1)
        s = sum(np.power(self.x, 2))
        p = prod(cos(np.divide(self.x, sqrt(j))))
        return [(s / fr - p + 1), [0.0]]

    def levy(self) -> List[float]:
        """ https://www.sfu.ca/~ssurjano/levy.html """
        n = len(self.x)
        z = 1 + (subtract(self.x, 1)) / 4
        t1 = sin(pi * z[0]) ** 2
        t2 = ((z[-1] - 1) ** 2) * (1 + (sin(2 * pi * z[-1])) ** 2)
        s = 0
        for i in range(n - 1):
            new = (z[i] - 1) ** 2 * (1 + 10 * (sin(pi * z[i] + 1)) ** 2)
            s += new
        return [(t1 + s + t2), [0.0]]

    def michalewicz(self) -> List[float]:
        """ https://www.sfu.ca/~ssurjano/michal.html """
        michalewicz_m = .5
        n = len(self.x)
        j = np.arange(1., n + 1)
        return [-(sum(np.multiply(sin(self.x, dtype=self._dtype.dtype),
                                  np.power(sin(np.multiply(j / pi,
                                                           np.power(self.x, 2, dtype=self._dtype.dtype),
                                                           dtype=self._dtype.dtype),
                                               dtype=self._dtype.dtype), (2 * michalewicz_m), dtype=self._dtype.dtype),
                                  dtype=self._dtype.dtype), dtype=self._dtype.dtype)), [0.0]]

    def perm(self, b=0.5) -> List[float]:
        """ https://www.sfu.ca/~ssurjano/permdb.html """
        n = len(self.x)
        outer = 0
        for i in range(n):
            inner = 0
            for j in range(n):
                xj = self.x[j]
                inner += (((j + 1) ** (i + 1)) + b) * (((xj / (j + 1)) ** (i + 1)) - 1)
            outer += inner ** 2
        return [outer, [0.0]]

    def powell(self) -> List[float]:
        """ https://www.sfu.ca/~ssurjano/permdb.html """
        n = int(len(self.x) / 4)
        s = 0
        for i in range(n):
            t1 = (self.x[4 * i - 3] + 10 * self.x[4 * i - 2]) ** 2
            t2 = 5 * (self.x[4 * i - 1] - self.x[4 * i]) ** 2
            t3 = (self.x[4 * i - 2] - 2 * self.x[4 * i - 1]) ** 4
            t4 = 10 * (self.x[4 * i - 3] - self.x[4 * i]) ** 4
            s += t1 + t2 + t3 + t4
        return [s, 0]

    def powersum(self) -> List[float]:
        """ https://www.sfu.ca/~ssurjano/powersum.html """
        b = [8, 18, 44, 114]
        n = 4
        s_out = 0
        for k in range(n):
            s_in = 0
            for j in range(n):
                s_in += self.x[j] ** (k + 1)
            s_out += (s_in - b[k]) ** 2

        return [s_out, [0.0]]

    def rastrigin(self) -> List[float]:
        """ https://www.sfu.ca/~ssurjano/rastr.html """
        n = len(self.x)
        s = sum(self.x[i] ** 2 - 10 * cos(2 * pi * self.x[i]) for i in range(n))
        return [10 * n + s,
                [0.0], [0.0]]

    def rosenbrock(self) -> List[float]:
        """ http://en.wikipedia.org/wiki/Rosenbrock_function """
        x0 = self.x[:-1]
        x1 = self.x[1:]
        return [sum((subtract(1, x0, dtype=self._dtype.dtype)) ** 2,
                    dtype=self._dtype.dtype) + 100 * sum(np.power(np.subtract(x1,
                                                                              np.power(x0, 2, dtype=self._dtype.dtype),
                                                                              dtype=self._dtype.dtype), 2,
                                                                  dtype=self._dtype.dtype),
                                                         dtype=self._dtype.dtype), [0.0]]

    def schwefel(self) -> List[float]:
        """ https://www.sfu.ca/~ssurjano/schwef.html """
        n = len(self.x)
        s = 0.0
        for i in range(n):
            s += self.x[i] * sin(sqrt(abs(self.x[i])), dtype=self._dtype.dtype)
        return [418.9829 * n - s, [0.0]]

    def sphere(self):
        """ https://www.sfu.ca/~ssurjano/spheref.html """
        return [sum(np.power(self.x, 2, dtype=self._dtype.dtype), dtype=self._dtype.dtype), [0.0]]

    def trid(self) -> List[float]:
        """ https://www.sfu.ca/~ssurjano/trid.html """

        return [sum(np.power(np.subtract(self.x, 1, dtype=self._dtype.dtype), 2, dtype=self._dtype.dtype),
                    dtype=self._dtype.dtype) -
                sum(np.multiply(self.x[:-1], self.x[1:], dtype=self._dtype.dtype),
                    dtype=self._dtype.dtype), [0.0]]

    def zakharov(self) -> List[float]:
        """ https://www.sfu.ca/~ssurjano/zakharov.html """
        n = len(self.x)
        j = np.arange(1., n + 1)
        s2 = sum(j * self.x) / 2
        return [sum(np.power(self.x, 2, dtype=self._dtype.dtype), dtype=self._dtype.dtype) + s2 ** 2 + s2 ** 4, [0.0]]


@dataclass
class ConSO:
    """ This class contains benchmark uncostrained global test functions. """
    """ These tests are adopted from the following website (converted from their matlab version) """
    """ https://www.sfu.ca"""
    """ Note: Design variables used here are continuous, but in future other variable types shall be considered """
    # Design vector
    x: List[float]
    # function name
    _name: str = "geom_prog"
    _dtype: DType = DType()

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, fname: str):
        self._name = copy.deepcopy(fname)

    @name.deleter
    def name(self):
        del self._name

    def test_problem(self):
        if self.name == "g1":
            self.g1()
        elif self.name == "g2":
            self.g2()
        elif self.name == "g3":
            self.g3()
        elif self.name == "geom_prog":
            self.geom_prog()
        elif self.name == "weight_min_speed_reducer":
            self.weight_min_speed_reducer()
        elif self.name == "tc_spring":
            self.tc_spring()
        elif self.name == "pressure_vessel":
            self.pressure_vessel()
        elif self.name == "himmelblau":
            self.himmelblau()
        elif self.name == "wbeam":
            self.wbeam()

    def g1(self):
        """ http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page506.htm """
        """ Note: MADS found better feasible solution than the best known one published for this problem """
        x1 = self.x[0:3]
        x2 = self.x[4:12]
        g = []
        f = (5 * sum(x1, dtype=self._dtype.dtype) - 5 * sum(np.square(x1, dtype=self._dtype.dtype),
                                                            dtype=self._dtype.dtype) - sum(x2, dtype=self._dtype.dtype))
        g.append(2 * self.x[0] + 2 * self.x[1] + self.x[9] + self.x[10] - 10)
        g.append(2 * self.x[0] + 2 * self.x[2] + self.x[9] + self.x[11] - 10)
        g.append(2 * self.x[1] + 2 * self.x[2] + self.x[10] + self.x[11] - 10)
        g.append(-8 * self.x[0] + self.x[9])
        g.append(-8 * self.x[1] + self.x[10])
        g.append(-8 * self.x[2] + self.x[11])
        g.append(-2 * self.x[3] - self.x[4] + self.x[9])
        g.append(-2 * self.x[5] - self.x[6] + self.x[10])
        g.append(-2 * self.x[7] - self.x[8] + self.x[11])

        return [f, g]

    def g2(self):
        """ http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2502.htm """
        n = len(self.x)
        s = 0
        c = []
        for j in range(n):
            s += (j + 1) * self.x[j] ** 2
        f = (-abs((sum(np.square(np.square(cos(self.x, dtype=self._dtype.dtype),
                                           dtype=self._dtype.dtype), dtype=self._dtype.dtype),
                       dtype=self._dtype.dtype) - 2 * prod(np.square(cos(self.x, dtype=self._dtype.dtype),
                                                                     dtype=self._dtype.dtype),
                                                           dtype=self._dtype.dtype)) / sqrt(s,
                                                                                            dtype=self._dtype.dtype)))
        c.append(-prod(self.x, dtype=self._dtype.dtype) + 0.75)
        c.append(sum(self.x, dtype=self._dtype.dtype) - 7.5 * n)
        return [f, c]

    def g3(self):
        """ http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2613.htm """
        """ The problem has been reformulated by defining the equality constraint as a response output """
        n = len(self.x) + 1
        c = [0]
        temp = np.subtract(1, sum(np.power(self.x, 2, dtype=self._dtype.dtype), dtype=self._dtype.dtype),
                           dtype=self._dtype.dtype)
        if temp > 0:
            x1 = sqrt(temp, dtype=self._dtype.dtype)
        else:
            x1 = np.nan
        f = (x1 * (-sqrt(n, dtype=self._dtype.dtype) ** n) * prod(self.x, dtype=self._dtype.dtype))
        return [f, c]

    def geom_prog(self) -> [float, list]:
        """ This problem has been reformulated as we assume that x0 , x1 , x2 , x5 are responses from analysis models
        r1 , r2 , r3 , r4 , the equality constraints can be regarded as analysis models
        and the overall problem can be implemented as shown below"""
        """ ref: [1] Target Cascading in Optimal System Design, 2003.  
                 [2] Numerical investigation of non-hierarchical coordination for distributed multidisciplinary 
                design optimization with fixed computational budget, 2017."""
        xx = self.x
        x2 = sqrt(xx[3] ** 2 + xx[4] ** -2 + xx[5] ** -2 + xx[6] ** 2)
        x5 = sqrt(xx[6] ** 2 + xx[7] ** 2 + xx[8] ** 2 + xx[9] ** 2)
        x0 = sqrt(x2 ** 2 + xx[0] ** -2 + xx[1] ** 2)
        x1 = sqrt(xx[1] ** 2 + x5 ** 2 + xx[2] ** 2)

        f = x0 ** 2 + x1 ** 2
        c = [x2 ** -2 + xx[0] ** 2 - xx[1] ** 2, xx[1] ** 2 + x5 ** -2 - xx[2] ** 2,
             xx[3] ** 2 + xx[4] ** 2 - xx[6] ** 2, xx[3] ** -2 + xx[5] ** 2 - xx[6] ** 2,
             xx[6] ** 2 + xx[7] ** -2 - xx[8] ** 2, xx[6] ** 2 + xx[7] ** 2 - xx[9] ** 2]
        return [f, c]

    def weight_min_speed_reducer(self):
        """ Weight minimization of a speed reducer """
        """ It involves the design of a speed reducer for a small aircraft engine. """
        """ ref:  
        [1] Integral Global Optimization: Theory, Implementation and Applications, 2012.
        [2] A test-suite of non-convex constrained optimization problems from the real-world and some 
        baseline results, 2020"""
        f = 0.7854 * self.x[1] ** 2 * self.x[0] * (14.9334 * self.x[2] - 43.0934 + 3.3333 * self.x[2] ** 2) \
            + 0.7854 * (self.x[4] * self.x[6] ** 2 + self.x[3] * self.x[5] ** 2) \
            - 1.508 * self.x[0] * (self.x[6] ** 2 + self.x[5] ** 2) + 7.477 * (self.x[6] ** 3 + self.x[5] ** 3)

        c = [-self.x[0] * self.x[1] ** 2 * self.x[2] + 27, -self.x[0] * self.x[1] ** 2 * self.x[2] ** 2 + 397.5,
             -self.x[1] * self.x[5] ** 4 * self.x[2] * self.x[3] ** -3 + 1.93,
             -self.x[1] * self.x[6] ** 4 * self.x[2] * self.x[4] ** -3 + 1.93,
             10 * self.x[5] ** -3 * sqrt(16.91 * 1e6 + (745 * self.x[3] * self.x[1] ** -1 * self.x[2] ** -1) ** 2)
             - 1100,
             10 * self.x[6] ** -3 * sqrt(157.5 * 1e6 + (745 * self.x[4] * self.x[1] ** -1 * self.x[2] ** -1) ** 2)
             - 850, self.x[1] * self.x[2] - 40, -self.x[0] * self.x[1] ** -1 + 5, self.x[0] * self.x[1] ** -1 - 12,
             1.5 * self.x[5] - self.x[3] + 1.9, 1.1 * self.x[6] - self.x[4] + 1.9]

        return [f, c]

    def tc_spring(self):
        """ Tension/compression spring design """
        """ The main objective of this problem is to optimize the weight of a tension or compression spring. 
        This problem contains four constraints and three variables are utilized to calculate the weight: the diameter 
        of the wire (x1), the mean of the diameter of coil (x2), and the number of active coils (x3). """
        """ ref:
        [1] A study of mathematical programming methods for structural optimization, 1985
        [2] A test-suite of non-convex constrained optimization problems from the real-world and some 
        baseline results, 2020"""

        f = self.x[0] ** 2 * self.x[1] * (2 + self.x[2])
        c = []
        a = (4 * self.x[1] ** 2 - self.x[0] * self.x[1]) / (12566 * (self.x[1] * self.x[0] ** 3 - self.x[0] ** 4))
        b = (1 / (5108 * self.x[0] ** 2))
        c.append(1 - ((self.x[1] ** 3 + self.x[2]) / (71785 * self.x[0] ** 4)))
        c.append(a + b - 1)
        c.append(1 - ((140.45 * self.x[0]) / (self.x[1] ** 2 * self.x[2])))
        c.append(((self.x[0] - self.x[1]) / 1.5) - 1)

        return [f, c]

    def pressure_vessel(self):
        """ Pressure vessel design  """
        """ The main objective of this problem is to optimize the welding cost, material, and forming of a vessel. 
        This problem contains four con- straints which are needed to be satisfied, and four variables are used to 
        calculate the objective function: shell thickness (z1), head thickness (z2), inner radius (x3), 
        and length of the vessel without including the head (x4). """
        """ ref:
        [1] Nonlinear integer and discrete programming in mechanical design, 1988.
        [2] A test-suite of non-convex constrained optimization problems from the real-world and some 
        baseline results, 2020. """
        self.x[2] = int(self.x[2])
        self.x[3] = int(self.x[3])
        z1 = 0.0625 * self.x[0]
        z2 = 0.0625 * self.x[1]
        f = 1.7781 * z2 * self.x[2] ** 2 + 0.6224 * z1 * self.x[2] * self.x[3] + 3.1661 * z1 ** 2 * self.x[3] \
            + 19.84 * z1 ** 2 * self.x[2]
        c = [0.00954 * self.x[2] - z2, 0.0193 * self.x[2] - z1, self.x[3] - 240,
             -pi * self.x[2] ** 2 * self.x[3] - (4 / 3) * pi * self.x[2] ** 3 + 1296000]

        return [f, c]

    def himmelblau(self):
        """ Proctor and Gamble Corporation proposes this problem to simulate the Process Design Problems
         and was cited by D. M. Himmelblau in Ref [1] which is used as common benchmark to analyze non-linear
         constrained optimization algorithms. This problem contains six nonlinear constraints and five variables.
         [1] A test-suite of non-convex constrained optimization problems from the real-world and some
        baseline results, 2020."""

        f = 5.3578547 * self.x[2] ** 2 + 0.8356891 * self.x[0] * self.x[4] + 37.293239 * self.x[0] - 40792.141

        g1_1 = 85.334407 + 0.0056858 * self.x[1] * self.x[4]
        g1_2 = 0.0006262 * self.x[0] * self.x[3] - 0.0022053 * self.x[2] * self.x[4]
        g1 = g1_1 + g1_2

        g2 = 80.51249 + 0.00713172 * self.x[1] * self.x[4] + 0.0029955 * self.x[0] * self.x[1] + 0.0021813 ** self.x[
            2] ** 2

        g3_1 = 9.300961 + 0.0047026 * self.x[2] * self.x[4] + 0.00125447 * self.x[0] * self.x[2]
        g3_2 = 0.0019085 * self.x[2] * self.x[3]

        g3 = g3_1 + g3_2

        c1 = -g1
        c2 = g1 - 92.0
        c3 = 90.0 - g2
        c4 = g2 - 110.0
        c5 = 20.0 - g3
        c6 = g3 - 25.0

        c = [c1, c2, c3, c4, c5, c6]

        return [f, c]

    def wbeam(self):
        """ The main objective of this problem is to design a welded beam with minimum cost.
        This problem contains five constraints, and four variables are used to develop a welded beam [1].
        [1] A test-suite of non-convex constrained optimization problems from the real-world and some
        baseline results, 2020.
        """
        P = 6000.0
        L = 14.0
        t_max = 13600.0
        s_max = 30000.0
        delta_max = 0.25
        G = 12 * 1e6
        E = 30 * 1e6
        R = sqrt((self.x[1] ** 2 / 4) + ((self.x[0] + self.x[2]) / 2) ** 2)
        M = P * (L + (self.x[1] / 2))
        J = 2 * sqrt(0.5) * self.x[0] * self.x[1] * ((self.x[1] ** 2 / 4.0) + 0.25 * (self.x[0] + self.x[2]) ** 2)
        t1 = P / (sqrt(2) * self.x[0] * self.x[1])
        t2 = M * R / J
        t = sqrt(t1 ** 2 + t2 ** 2 + (t1 * t2 * self.x[1] / R))
        s = 6.0 * P * L / (self.x[3] * self.x[2] ** 2)
        P_c = ((1 - (self.x[2] / (2 * L)) * sqrt(E / (4 * G))) * 4.013 * E * self.x[2] * self.x[3] ** 3) / (6 * L ** 2)
        delta = (6 * P * L ** 3) / (E * self.x[2] ** 2 * self.x[3])

        f = 0.04811 * self.x[2] * self.x[3] * (14.0 + self.x[1]) + 1.10471 * self.x[1] * self.x[0] ** 2

        g1 = self.x[0] - self.x[3]
        g2 = delta - delta_max
        g3 = P - P_c
        g4 = t - t_max
        g5 = s - s_max

        c = [g1, g2, g3, g4, g5]

        return [f, c]


@dataclass
class Run:
    """ See https://www.gerad.ca/Sebastien.Le.Digabel/MTH8418/2_benchmarking.pdf"""
    report_path: str
    field_names: List[str] = field(init=False)
    file_writer: csv.DictWriter = field(init=False)
    test_suite: str = "SO_UNCON"
    num_tests: int = 15
    tests_path: str = os.path.abspath("../examples/bm/constrained")

    def __post_init__(self):
        self.field_names = ['NAME', "#VAR", "#INEQCON", '#SUCCESS', '#ITER', '#FEVAL', 'PSIZE', "PS_SUCCESS",
                            "PS_MAX", 'HMIN', 'FMIN',
                            'FEX', 'FERR']

        with open(os.path.join(self.report_path, 'BM_report.csv'), 'w', newline='') as f:
            self.file_writer = csv.DictWriter(f, fieldnames=self.field_names)
            self.file_writer.writeheader()

    def add_row(self, name: str, nv: int, nc: int, nb_success: int, it: int, feval: int, psize: float,
                psize_success: float, psize_max: float, hmin: float, fmin: float):
        fex = 0.0
        if name == "michalewicz":
            fex = -1.8013
        elif name == "trid":
            fex = -2.0
        elif name == "geom_prog":
            fex = 17.0
        elif name == "pressure_vessel":
            fex = 5.8853327736E+03
        elif name == "weight_min_speed_reducer":
            fex = 2.9944244658E+03
        elif name == "tc_spring":
            fex = 1.2665232788E-02
        elif name == "g1":
            fex = -15
        elif name == "g2":
            fex = -0.803619
        elif name == "g3":
            fex = -1
        elif name == "himmelblau":
            fex = -3.0665538672e4
        elif name == "wbeam":
            fex = 1.6702177263E+00

        err = abs(fex-fmin)/max(abs(fex), abs(fmin)) * 100
        row = {'NAME': name, "#VAR": nv, "#INEQCON": nc, '#SUCCESS': nb_success, '#ITER': it,
               '#FEVAL': feval, 'PSIZE': psize, "PS_SUCCESS": psize_success,
               "PS_MAX": psize_max, 'HMIN': hmin, 'FMIN': fmin, 'FEX': fex,
               'FERR': err}

        with open(os.path.join(self.report_path, 'BM_report.csv'), 'a', newline='') as f:
            self.file_writer = csv.DictWriter(f, fieldnames=self.field_names)
            self.file_writer.writerow(row)

    def multi_seed_run(self):
        """ Repeate the tests using different random generator seeds"""
        # TODO: BM the code using different random seeds

    def analyze_bm_statistics(self):
        """ Calculate the performance indicators of the implemented algorithm using wilcoxon rank sum"""
        # TODO: Calculate the average performance and analyze its statistics with the aid of wilcoxon sign rank sum
