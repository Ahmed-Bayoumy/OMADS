"""The 100 bound-constrained multiobjective test problems of Custodio et al.

A. L. Custodio, J. F. A. Madeira, A. I. F. Vaz and L. N. Vicente, Direct
multisearch for multiobjective optimization, SIAM J. Optim. 21(3), 2011.
This is the problem set of the DMulti-MADS article (Bigeon, Le Digabel &
Salomon, Comput. Optim. Appl. 79, 2021), whose NOMAD implementation is
NOMAD 4's multiobjective solver: 69 problems with 2 objectives, 29 with 3
and 2 with 4, with 1 to 30 variables.

Ported to NumPy from the Julia versions distributed with DMultiMadsEB
(github.com/bbopt/DMultiMadsEB, problems/julia), keeping their formulas,
dimensions and bounds (from scripts/generate_nsgaii_caches.py); the constant
matrices of DPAM1, L1ZDT4, L2ZDT*, L3ZDT* and lovison5/6 are read from
``matrices.json``. Every function takes a 1-D array and returns the vector
of objectives (all minimized).

  PROBLEMS[name] = (function, n, m, lb, ub)
"""

import json
import os

import numpy as np

pi = np.pi
_MAT = {k: np.array(v, dtype=float) for k, v in json.load(open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "matrices.json"))).items()}


def _a(x):
  return np.asarray(x, dtype=float).ravel()


# --- Simple problems -------------------------------------------------------

def BK1(x):
  x = _a(x)
  return [x[0]**2 + x[1]**2, (x[0] - 5)**2 + (x[1] - 5)**2]


def CL1(x):
  x = _a(x)
  F, E, L = 10., 2e5, 200.
  return [L * (2 * x[0] + np.sqrt(2) * x[1] + np.sqrt(x[2]) + x[3]),
          F * L / E * (2 / x[0] + 2 * np.sqrt(2) / x[1] - 2 * np.sqrt(2) / x[2] + 2 / x[3])]


def DG01(x):
  x = _a(x)
  return [np.sin(x[0]), np.sin(x[0] + 0.7)]


def DPAM1(x):
  n = 10
  y = _MAT["M10"] @ _a(x)
  g = 1 + 10 * (n - 1) + np.sum(y[1:]**2 - 10 * np.cos(4 * pi * y[1:]))
  return [y[0], g * np.exp(-y[0] / g)]


def Deb41(x):
  x = _a(x)
  g = 2 - np.exp(-((x[1] - 0.2) / 0.004)**2) - 0.8 * np.exp(-((x[1] - 0.6) / 0.4)**2)
  return [x[0], g / x[0]]


def _deb512_g(x2):
  if x2 <= 0.4:
    return 4 - 3 * np.exp(-((x2 - 0.2) / 0.02)**2)
  return 4 - 2 * np.exp(-((x2 - 0.7) / 0.2)**2)


def _deb512(x, alpha=None):
  x = _a(x)
  f1 = 4 * x[0]
  g = _deb512_g(x[1])
  if alpha is None:              # Deb512c
    alpha = 0.25 + 3.75 * (g - 1)
  h = 1 - (f1 / g)**alpha if f1 <= g else 0.
  return [f1, g * h]


def Deb512a(x):
  return _deb512(x, 0.25)


def Deb512b(x):
  return _deb512(x, 4)


def Deb512c(x):
  return _deb512(x)


def Deb513(x):
  x = _a(x)
  f1, g = x[0], 1 + 10 * x[1]
  h = 1 - (f1 / g)**2 - (f1 / g) * np.sin(2 * pi * 4 * f1)
  return [f1, g * h]


def Deb521a(x):
  x = _a(x)
  g = 1 + x[1]**0.25
  return [x[0], g * (1 - (x[0] / g)**2)]


def Deb521b(x):
  x = _a(x)
  g = 1 + x[1]
  return [x[0], g * (1 - (x[0] / g)**2)]


def Deb53(x):
  x = _a(x)
  f1 = 1 - np.exp(-4 * x[0]) * np.sin(5 * pi * x[0])**4
  g = _deb512_g(x[1])
  h = 1 - (f1 / g)**4 if f1 <= g else 0.
  return [f1, g * h]


def ex005(x):
  x = _a(x)
  return [x[0]**2 - x[1]**2, x[0] / x[1]]


def Far1(x):
  x1, x2 = _a(x)
  f1 = (-2 * np.exp(15 * (-(x1 - 0.1)**2 - x2**2))
        - np.exp(20 * (-(x1 - 0.6)**2 - (x2 - 0.6)**2))
        + np.exp(20 * (-(x1 + 0.6)**2 - (x2 - 0.6)**2))
        + np.exp(20 * (-(x1 - 0.6)**2 - (x2 + 0.6)**2))
        + np.exp(20 * (-(x1 + 0.6)**2 - (x2 + 0.6)**2)))
  f2 = (2 * np.exp(20 * (-x1**2 - x2**2))
        + np.exp(20 * (-(x1 - 0.4)**2 - (x2 - 0.6)**2))
        - np.exp(20 * (-(x1 + 0.5)**2 - (x2 - 0.7)**2))
        - np.exp(20 * (-(x1 - 0.5)**2 - (x2 + 0.7)**2))
        + np.exp(20 * (-(x1 + 0.4)**2 - (x2 + 0.8)**2)))
  return [f1, f2]


_I10 = np.arange(1, 11)          # 1..n
_J10 = np.arange(0, 10)          # 0..n-1


def _fes_a(x):
  return np.sum(np.abs(x - np.exp((_I10 / 10)**2) / 3)**0.5)


def _fes_b(x):
  return np.sum((x - 0.5 * np.cos(10 * pi * _I10 / 10) - 0.5)**2)


def _fes_c(x):
  return np.sum(np.abs(x - np.sin(_J10)**2 * np.cos(_J10)**2)**0.5)


def _fes_d(x):
  return np.sum(np.abs(x - 0.25 * np.cos(_J10) * np.cos(2 * _J10) - 0.5)**0.5)


def FES1(x):
  x = _a(x)
  return [_fes_a(x), _fes_b(x)]


def FES2(x):
  x = _a(x)
  return [_fes_b(x), _fes_c(x), _fes_d(x)]


def FES3(x):
  x = _a(x)
  return [_fes_a(x), _fes_c(x), _fes_d(x),
          np.sum((x - 0.5 * np.sin(1000 * pi * _I10 / 10) - 0.5)**2)]


def Fonseca(x):
  x = _a(x)
  return [1 - np.exp(-(x[0] - 1)**2 - (x[1] + 1)**2),
          1 - np.exp(-(x[0] + 1)**2 - (x[1] - 1)**2)]


def IKK1(x):
  x = _a(x)
  return [x[0]**2, (x[0] - 20)**2, x[1]**2]


def IM1(x):
  x = _a(x)
  return [2 * np.sqrt(x[0]), x[0] * (1 - x[1]) + 5]


def Jin1(x):
  x = _a(x)
  return [np.sum(x**2) / 2, np.sum((x - 2)**2) / 2]


def _jin_g(x):
  return 1 + 9 * np.sum(x[1:]) / (len(x) - 1)


def Jin2(x):
  x = _a(x)
  g = _jin_g(x)
  return [x[0], g * (1 - np.sqrt(x[0] / g))]


def Jin3(x):
  x = _a(x)
  g = _jin_g(x)
  return [x[0], g * (1 - (x[0] / g)**2)]


def Jin4(x):
  x = _a(x)
  g = _jin_g(x)
  return [x[0], g * (1 - (x[0] / g)**0.25 - (x[0] / g)**4)]


def Kursawe(x):
  x = _a(x)
  return [np.sum(-10 * np.exp(-0.2 * np.sqrt(x[:-1]**2 + x[1:]**2))),
          np.sum(np.abs(x)**0.8 + 5 * np.sin(x)**3)]


def LE1(x):
  x = _a(x)
  return [(x[0]**2 + x[1]**2)**0.125, ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)**0.25]


def LRS1(x):
  x = _a(x)
  return [x[0]**2 + x[1]**2, (x[0] + 2)**2 + x[1]**2]


def lovison1(x):
  x = _a(x)
  f1 = -1.05 * x[0]**2 - 0.98 * x[1]**2
  f2 = -0.99 * (x[0] - 3)**2 - 1.03 * (x[1] - 2.5)**2
  return [-f1, -f2]


def lovison2(x):
  x = _a(x)
  return [x[1], -(x[1] - x[0]**3) / (x[0] + 1)]


def lovison3(x):
  x = _a(x)
  f1 = -x[0]**2 - x[1]**2
  f2 = -(x[0] - 6)**2 + (x[1] + 0.3)**2
  return [-f1, -f2]


def lovison4(x):
  x = _a(x)
  f1 = -x[0]**2 - x[1]**2 - 4 * (np.exp(-(x[0] + 2)**2 - x[1]**2) +
                                 np.exp(-(x[0] - 2)**2 - x[1]**2))
  f2 = -(x[0] - 6)**2 - (x[1] + 0.5)**2
  return [-f1, -f2]


def _lovison_tmp(x, C, alpha):
  # f_tmp[j] = -sum_i alpha[j, i] * (x[i] - C[i, j])^2
  return -np.sum(alpha * (x[None, :] - C.T)**2, axis=1)


def lovison5(x):
  x = _a(x)
  beta = [0.575496, 0.675617, 0.180332]
  gamma = [-0.593814, -0.492722, 0.0646786]
  t = _lovison_tmp(x, _MAT["lovison5_C"], _MAT["lovison5_alpha"])
  f1 = t[0]
  f2 = t[1] + beta[1] * np.sin(pi * (x[0] + x[1]) / gamma[1])
  f3 = t[2] + beta[2] * np.cos(pi * (x[0] - x[1]) / gamma[2])
  return [-f1, -f2, -f3]


def lovison6(x):
  x = _a(x)
  beta = [-0.666503, -0.945716, -0.334582, 0.611894]
  gamma = [0.281032, 0.508749, -0.0265389, -0.920133]
  t = _lovison_tmp(x, _MAT["lovison6_C"], _MAT["lovison6_alpha"])
  f1 = t[0] + beta[0] * np.exp(t[3] / gamma[0])
  f2 = t[1] + beta[1] * np.sin(pi * (x[0] + x[1]) / gamma[1])
  f3 = t[2] + beta[2] * np.cos(pi * (x[0] - x[1]) / gamma[2])
  return [-f1, -f2, -f3]


def MHHM1(x):
  x = _a(x)
  return [(x[0] - 0.8)**2, (x[0] - 0.85)**2, (x[0] - 0.9)**2]


def MHHM2(x):
  x = _a(x)
  return [(x[0] - 0.8)**2 + (x[1] - 0.6)**2,
          (x[0] - 0.85)**2 + (x[1] - 0.7)**2,
          (x[0] - 0.9)**2 + (x[1] - 0.6)**2]


def MLF1(x):
  x = _a(x)
  return [(1 + x[0] / 20) * np.sin(x[0]), (1 + x[0] / 20) * np.cos(x[0])]


def MLF2(x):
  x = _a(x)
  y = 2 * x
  return [-(5 - ((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2) / 200),
          -(5 - ((y[0]**2 + y[1] - 11)**2 + (y[0] + y[1]**2 - 7)**2) / 200)]


def MOP1(x):
  x = _a(x)
  return [x[0]**2, (x[0] - 2)**2]


def MOP2(x):
  x = _a(x)
  n = 4
  return [1 - np.exp(-np.sum((x - 1 / np.sqrt(n))**2)),
          1 - np.exp(-np.sum((x + 1 / np.sqrt(n))**2))]


def MOP3(x):
  x = _a(x)
  A1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
  A2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)
  B1 = 0.5 * np.sin(x[0]) - 2 * np.cos(x[0]) + np.sin(x[1]) - 1.5 * np.cos(x[1])
  B2 = 1.5 * np.sin(x[0]) - np.cos(x[0]) + 2 * np.sin(x[1]) - 0.5 * np.cos(x[1])
  f1 = -1 - (A1 - B1)**2 - (A2 - B2)**2
  f2 = -(x[0] + 3)**2 - (x[1] + 1)**2
  return [-f1, -f2]


def MOP4(x):
  x = _a(x)
  # sin(x^3), unlike Kursawe's sin(x)^3
  return [np.sum(-10 * np.exp(-0.2 * np.sqrt(x[0:2]**2 + x[1:3]**2))),
          np.sum(np.abs(x)**0.8 + 5 * np.sin(x**3))]


def MOP5(x):
  x = _a(x)
  r = x[0]**2 + x[1]**2
  return [0.5 * r + np.sin(r),
          (3 * x[0] - 2 * x[1] + 4)**2 / 8 + (x[0] - x[1] + 1)**2 / 27 + 15,
          1 / (r + 1) - 1.1 * np.exp(-x[0]**2 - x[1]**2)]


def MOP6(x):
  x = _a(x)
  g = 1 + 10 * x[1]
  return [x[0], g * (1 - (x[0] / g)**2 - x[0] / g * np.sin(8 * pi * x[0]))]


def MOP7(x):
  x = _a(x)
  return [(x[0] - 2)**2 / 2 + (x[1] + 1)**2 / 13 + 3,
          (x[0] + x[1] - 3)**2 / 36 + (-x[0] + x[1] + 2)**2 / 8 - 17,
          (x[0] + 2 * x[1] - 1)**2 / 175 + (-x[0] + 2 * x[1])**2 / 17 - 13]


def OKA1(x):
  x = _a(x)
  c, s = np.cos(pi / 12), np.sin(pi / 12)
  y1 = c * x[0] - s * x[1]
  y2 = s * x[0] + c * x[1]
  return [y1, np.sqrt(2 * pi) - np.sqrt(abs(y1)) + 2 * abs(y2 - 3 * np.cos(y1) - 3)**(1 / 3)]


def OKA2(x):
  x = _a(x)
  return [x[0], 1 - (x[0] + pi)**2 / (4 * pi**2) + abs(x[1] - 5 * np.cos(x[0]))**(1 / 3)
          + abs(x[2] - 5 * np.sin(x[0]))**(1 / 3)]


def QV1(x):
  x = _a(x)
  n = 10
  return [np.sum((x**2 - 10 * np.cos(2 * pi * x) + 10) / n)**0.25,
          np.sum(((x - 1.5)**2 - 10 * np.cos(2 * pi * (x - 1.5)) + 10) / n)**0.25]


def Sch1(x):
  x1 = _a(x)[0]
  if x1 <= 1:
    f1 = -x1
  elif x1 <= 3:
    f1 = -2 + x1
  elif x1 <= 4:
    f1 = 4 - x1
  else:
    f1 = -4 + x1
  return [f1, (x1 - 5)**2]


def SK1(x):
  x1 = _a(x)[0]
  f1 = -x1**4 - 3 * x1**3 + 10 * x1**2 + 10 * x1 + 10
  f2 = 0.5 * x1**4 + 2 * x1**3 + 10 * x1**2 - 10 * x1 + 5
  return [-f1, -f2]


def SK2(x):
  x = _a(x)
  f1 = -(x[0] - 2)**2 - (x[1] + 3)**2 - (x[2] - 5)**2 - (x[3] - 4)**2 + 5
  f2 = np.sum(np.sin(x)) / (1 + np.sum(x**2) / 100)
  return [-f1, -f2]


def SP1(x):
  x = _a(x)
  return [(x[0] - 1)**2 + (x[0] - x[1])**2, (x[1] - 3)**2 + (x[0] - x[1])**2]


def SSFYY1(x):
  x = _a(x)
  return [x[0]**2 + x[1]**2, (x[0] - 1)**2 + (x[1] - 2)**2]


def SSFYY2(x):
  x1 = _a(x)[0]
  return [10 + x1**2 - 10 * np.cos(x1 * pi / 2), (x1 - 4)**2]


def TKLY1(x):
  x = _a(x)
  z = x[1:4]
  return [x[0], np.prod(2.0 - np.exp(-((z - 0.1) / 0.004)**2)
                        - 0.8 * np.exp(-((z - 0.9) / 0.4)**2)) / x[0]]


def VFM1(x):
  x = _a(x)
  return [x[0]**2 + (x[1] - 1)**2, x[0]**2 + (x[1] + 1)**2 + 1,
          (x[0] - 1)**2 + x[1]**2 + 2]


def VU1(x):
  x = _a(x)
  return [1 / (x[0]**2 + x[1]**2 + 1), x[0]**2 + 3 * x[1]**2 + 1]


def VU2(x):
  x = _a(x)
  return [x[0] + x[1] + 1, x[0]**2 + 2 * x[1] - 1]


def ZLT1(x):
  x = _a(x)
  sq = np.sum(x**2)
  return [(x[m] - 1)**2 + sq - x[m]**2 for m in range(3)]


# --- DTLZ ------------------------------------------------------------------

def _dtlz_g1(x, M):
  k = len(x) - M + 1
  z = x[M - 1:]
  return 100 * (k + np.sum((z - 0.5)**2 - np.cos(20 * pi * (z - 0.5))))


def _dtlz_linear(x, M, g):
  ff = np.ones(M)
  ff[0] = 0.5 * (1 + g) * np.prod(x[:M - 1])
  for i in range(2, M + 1):
    ff[i - 1] = 0.5 * (1 + g) * np.prod(x[:M - i]) * (1 - x[M - i])
  return list(ff)


def _dtlz_sphere(y, M, g):
  ff = np.ones(M)
  ff[0] = (1 + g) * np.prod(np.cos(0.5 * pi * y[:M - 1]))
  for i in range(2, M + 1):
    ff[i - 1] = (1 + g) * np.prod(np.cos(0.5 * pi * y[:M - i])) * np.sin(0.5 * pi * y[M - i])
  return list(ff)


def _dtlz1(x, M):
  x = _a(x)
  return _dtlz_linear(x, M, _dtlz_g1(x, M))


def _dtlz2(x, M):
  x = _a(x)
  return _dtlz_sphere(x, M, np.sum((x[M - 1:] - 0.5)**2))


def _dtlz3(x, M):
  x = _a(x)
  return _dtlz_sphere(x, M, _dtlz_g1(x, M))


def _dtlz4(x, M):
  y = _a(x)**100
  return _dtlz_sphere(y, M, np.sum((y[M - 1:] - 0.5)**2))


def _dtlz5(x, M):
  x = _a(x)
  g = np.sum(x[M - 1:]**0.1)
  theta = (pi / 2) * (1 + 2 * g * x) / (2 * (1 + g))
  ff = np.ones(M)
  ff[0] = (1 + g) * np.cos(0.5 * pi * x[0]) * np.prod(np.cos(theta[1:M - 1]))
  for i in range(2, M):
    ff[i - 1] = (1 + g) * np.cos(0.5 * pi * x[0]) * np.prod(np.cos(theta[1:M - i])) \
        * np.sin(theta[M - i])
  ff[M - 1] = (1 + g) * np.sin(0.5 * pi * x[0])
  return list(ff)


def _dtlz6(x, M):
  x = _a(x)
  k = len(x) - M + 1
  g = 1 + (9 / k) * np.sum(x[M - 1:])
  ff = np.ones(M)
  ff[M - 1] = (1 + g) * (M - np.sum(x[:M - 1] / (1 + g) * (1 + np.sin(3 * pi * x[:M - 1]))))
  ff[:M - 1] = x[:M - 1]
  return list(ff)


def DTLZ1(x): return _dtlz1(x, 3)
def DTLZ1n2(x): return _dtlz1(x, 2)
def DTLZ2(x): return _dtlz2(x, 3)
def DTLZ2n2(x): return _dtlz2(x, 2)
def DTLZ3(x): return _dtlz3(x, 3)
def DTLZ3n2(x): return _dtlz3(x, 2)
def DTLZ4(x): return _dtlz4(x, 3)
def DTLZ4n2(x): return _dtlz4(x, 2)
def DTLZ5(x): return _dtlz5(x, 3)
def DTLZ5n2(x): return _dtlz5(x, 2)
def DTLZ6(x): return _dtlz6(x, 3)
def DTLZ6n2(x): return _dtlz6(x, 2)


# --- ZDT and rotated ZDT (L1/L2/L3ZDT) -------------------------------------

def _zdt(f1, y_rest, kind):
  m = len(y_rest) + 1
  if kind == "sum":
    g = 1 + 9 / (m - 1) * np.sum(y_rest)
  elif kind == "sumsq":
    g = 1 + 9 / (m - 1) * np.sum(y_rest**2)
  elif kind == "rastrigin":
    g = 1 + 10 * (m - 1) + np.sum(y_rest**2 - 10 * np.cos(4 * pi * y_rest))
  elif kind == "pow":
    g = 1 + 9 * (np.sum(y_rest) / (m - 1))**0.25
  elif kind == "powsq":
    g = 1 + 9 * (np.sum(y_rest**2) / (m - 1))**0.25
  return g


def _h(f1, g, shape):
  if shape == 1:
    return 1 - np.sqrt(f1 / g)
  if shape == 2:
    return 1 - (f1 / g)**2
  return 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * pi * f1)


def _zdt_problem(x, shape, kind):
  x = _a(x)
  f1 = x[0]
  g = _zdt(f1, x[1:], kind)
  return [f1, g * _h(f1, g, shape)]


def ZDT1(x): return _zdt_problem(x, 1, "sum")
def ZDT2(x): return _zdt_problem(x, 2, "sum")
def ZDT3(x): return _zdt_problem(x, 3, "sum")
def ZDT4(x): return _zdt_problem(x, 1, "rastrigin")


def ZDT6(x):
  x = _a(x)
  f1 = 1 - np.exp(-4 * x[0]) * np.sin(6 * pi * x[0])**6
  g = _zdt(f1, x[1:], "pow")
  return [f1, g * (1 - (f1 / g)**2)]


def _lzdt(x, mat, squared, shape, kind):
  x = _a(x)
  y = _MAT[mat] @ (x**2 if squared else x)
  f1 = y[0]**2
  g = _zdt(f1, y[1:], kind)
  return [f1, g * _h(f1, g, shape)]


def L1ZDT4(x): return _lzdt(x, "L1ZDT4", False, 1, "rastrigin")
def L2ZDT1(x): return _lzdt(x, "M30", False, 1, "sumsq")
def L2ZDT2(x): return _lzdt(x, "M30", False, 2, "sumsq")
def L2ZDT3(x): return _lzdt(x, "M30", False, 3, "sumsq")
def L2ZDT4(x): return _lzdt(x, "M30", False, 1, "rastrigin")
def L2ZDT6(x): return _lzdt(x, "M10", False, 2, "powsq")
def L3ZDT1(x): return _lzdt(x, "M30", True, 1, "sumsq")
def L3ZDT2(x): return _lzdt(x, "M30", True, 2, "sumsq")
def L3ZDT3(x): return _lzdt(x, "M30", True, 3, "sumsq")
def L3ZDT4(x): return _lzdt(x, "M30", True, 1, "rastrigin")
def L3ZDT6(x): return _lzdt(x, "M10", True, 2, "powsq")


# --- WFG toolkit (and its I1-I5 variants), M = 3, k = 4, l = 4 --------------
# Index arithmetic below follows the 1-based Julia code: groups of position
# variables are a..b = div((i-1)k, M-1)+1 .. div(ik, M-1).

_M, _K, _L = 3, 4, 4
_N = _K + _L


def _groups():
  return [((i - 1) * _K // (_M - 1) + 1, i * _K // (_M - 1)) for i in range(1, _M)]


def _s_linear(t):
  return np.abs(t - 0.35) / np.abs(np.floor(0.35 - t) + 0.35)


def _b_param(y, r):
  AA, BB, CC = 0.98 / 49.98, 0.02, 50.
  return y**(BB + (CC - BB) * (AA - (1 - 2 * r) * np.abs(np.floor(0.5 - r) + AA)))


def _s_multi(t, A, B, C):
  u = np.abs(t - C) / (2 * (np.floor(C - t) + C))
  return (1 + np.cos((4 * A + 2) * pi * (0.5 - u)) + 4 * B * u**2) / (B + 2)


def _s_decept(t, A, B, C):
  return 1 + (np.abs(t - A) - B) * (
      np.floor(t - A + B) * (1 - C + (A - B) / B) / (A - B)
      + np.floor(A + B - t) * (1 - C + (1 - A - B) / B) / (1 - A - B) + 1 / B)


def _r_sum_groups(t, w):
  """Weighted means of the position groups, then of t[k+1..n]."""
  out = [np.sum(w[a - 1:b] * t[a - 1:b]) / np.sum(w[a - 1:b]) for a, b in _groups()]
  out.append(np.sum(w[_K:_N] * t[_K:_N]) / np.sum(w[_K:_N]))
  return np.array(out)


def _r_nonsep_groups(t):
  """r_nonsep reduction of the position groups and of t[k+1..n]."""
  out = np.zeros(_M)
  for i, (a, b) in enumerate(_groups(), start=1):
    size = b - a + 1
    for ii in range(a, b + 1):
      out[i - 1] += t[ii - 1]
      for jj in range(0, _K // (_M - 1) - 1):
        out[i - 1] += abs(t[ii - 1] - t[a + (ii + jj - a + 1) % size - 1])
    kk = _K / (_M - 1)
    out[i - 1] /= (((i * _K / (_M - 1)) - ((i - 1) * _K / (_M - 1) + 1) + 1) / kk
                   * np.ceil(kk / 2) * (1 + 2 * kk - 2 * np.ceil(kk / 2)))
  for ii in range(_K + 1, _N + 1):
    out[_M - 1] += t[ii - 1]
    for jj in range(0, _L - 1):
      out[_M - 1] += abs(t[ii - 1] - t[_K + 1 + (ii + jj - (_K + 1) + 1) % (_N - _K) - 1])
  out[_M - 1] /= ((_N - _K) / _L) * np.ceil(_L / 2) * (1 + 2 * _L - 2 * np.ceil(_L / 2))
  return out


def _x_of_t(t, A):
  x = np.ones(_M)
  x[:_M - 1] = np.maximum(t[_M - 1], A) * (t[:_M - 1] - 0.5) + 0.5
  x[_M - 1] = t[_M - 1]
  return x


def _h_sin(x):
  """Shape used by I1-I5 and WFG4-WFG9."""
  h = np.ones(_M)
  h[0] = np.prod(np.sin(x[:_M - 1] * pi / 2))
  for m in range(2, _M):
    h[m - 1] = np.prod(np.sin(x[:_M - m] * pi / 2)) * np.cos(x[_M - m] * pi / 2)
  h[_M - 1] = np.cos(x[0] * pi / 2)
  return h


def _h_convex_head(x):
  h = np.ones(_M)
  h[0] = np.prod(1 - np.cos(pi / 2 * x[:_M - 1]))
  for m in range(2, _M):
    h[m - 1] = np.prod(1 - np.cos(pi / 2 * x[:_M - m])) * (1 - np.sin(x[_M - m] * pi / 2))
  return h


def _finish(x, h, S):
  return list(x[_M - 1] + S * h)


_ONES_N, _ONES_M = np.ones(_N), np.ones(_M)
_S2 = 2. * np.arange(1, _M + 1)
_Z2 = 2. * np.arange(1, _N + 1)
_A1 = np.ones(_M - 1)


def _r_sum_after(y, idx):
  # mean of y[i+1..n] (1-based i)
  return np.array([np.mean(y[i:]) for i in idx])


def _r_sum_before(y, idx):
  # mean of y[1..i-1] (1-based i)
  return np.array([np.mean(y[:i - 1]) for i in idx])


def _i_head(y, kind):
  t1 = y.copy()
  if kind == "after":            # I2, WFG9: positions 1..n-1 depend on later ones
    r = _r_sum_after(y, range(1, _N))
    t1[:_N - 1] = _b_param(y[:_N - 1], r)
  else:                          # I3, I5: positions 2..n depend on earlier ones
    r = _r_sum_before(y, range(2, _N + 1))
    t1[1:] = _b_param(y[1:], r)
  return t1


def _i_problem(x, kind, nonsep):
  y = _a(x)
  t1 = y if kind is None else _i_head(y, kind)
  t2 = t1.copy()
  t2[_K:] = _s_linear(t1[_K:])
  t3 = _r_nonsep_groups(t2) if nonsep else _r_sum_groups(t2, _ONES_N)
  xt = _x_of_t(t3, _A1)
  return _finish(xt, _h_sin(xt), _ONES_M)


def I1(x): return _i_problem(x, None, False)
def I2(x): return _i_problem(x, "after", False)
def I3(x): return _i_problem(x, "before", False)
def I4(x): return _i_problem(x, None, True)
def I5(x): return _i_problem(x, "before", True)


def WFG1(x):
  y = _a(x) / _Z2
  t1 = y.copy()
  t1[_K:] = _s_linear(y[_K:])
  AA, BB, CC = 0.8, 0.75, 0.85
  t2 = t1.copy()
  u = t1[_K:]
  t2[_K:] = AA + np.minimum(0, np.floor(u - BB)) * (AA * (BB - u)) / BB \
      - np.minimum(0, np.floor(CC - u)) * (1 - AA) * (u - CC) / (1 - CC)
  t3 = np.abs(t2)**0.02
  xt = _x_of_t(_r_sum_groups(t3, _Z2), _A1)
  h = _h_convex_head(xt)
  A5 = 5
  h[_M - 1] = 1 - xt[0] - np.cos(2 * A5 * pi * xt[0] + pi / 2) / (2 * A5 * pi)
  return _finish(xt, h, _S2)


def _wfg23_head(x):
  y = _a(x) / _Z2
  t1 = y.copy()
  t1[_K:] = _s_linear(y[_K:])
  AA = 2
  t2 = np.zeros(_K + _L // 2)
  t2[:_K] = t1[:_K]
  for i in range(_K + 1, _K + _L // 2 + 1):
    a, b = _K + 2 * (i - _K) - 1, _K + 2 * (i - _K)
    size = b - a + 1
    for ii in range(a, b + 1):
      t2[i - 1] += t1[ii - 1]
      for jj in range(0, AA - 1):
        t2[i - 1] += abs(t1[ii - 1] - t1[a + (ii + jj - a + 1) % size - 1])
    t2[i - 1] /= (size / AA * np.ceil(AA / 2) * (1 + 2 * AA - 2 * np.ceil(AA / 2)))
  t3 = np.ones(_M)
  for i, (a, b) in enumerate(_groups(), start=1):
    t3[i - 1] = np.mean(t2[a - 1:b])
  t3[_M - 1] = np.mean(t2[_K:_K + _L // 2])
  return t3


def WFG2(x):
  xt = _x_of_t(_wfg23_head(x), _A1)
  h = _h_convex_head(xt)
  h[_M - 1] = 1 - xt[0] * np.cos(5 * xt[0] * pi)**2
  return _finish(xt, h, _S2)


def WFG3(x):
  A = np.zeros(_M - 1)
  A[0] = 1
  xt = _x_of_t(_wfg23_head(x), A)
  h = np.ones(_M)
  h[0] = np.prod(xt[:_M - 1])
  for m in range(2, _M):
    h[m - 1] = np.prod(xt[:_M - m]) * (1 - xt[_M - m])
  h[_M - 1] = 1 - xt[0]
  return _finish(xt, h, _S2)


def _wfg_tail(t, nonsep=False):
  t3 = _r_nonsep_groups(t) if nonsep else _r_sum_groups(t, _ONES_N)
  xt = _x_of_t(t3, _A1)
  return _finish(xt, _h_sin(xt), _S2)


def WFG4(x):
  return _wfg_tail(_s_multi(_a(x) / _Z2, 30, 10, 0.35))


def WFG5(x):
  return _wfg_tail(_s_decept(_a(x) / _Z2, 0.35, 0.001, 0.05))


def WFG6(x):
  y = _a(x) / _Z2
  t1 = y.copy()
  t1[_K:] = _s_linear(y[_K:])
  return _wfg_tail(t1, nonsep=True)


def WFG7(x):
  y = _a(x) / _Z2
  t1 = y.copy()
  t1[:_K] = _b_param(y[:_K], _r_sum_after(y, range(1, _K + 1)))
  t2 = t1.copy()
  t2[_K:] = _s_linear(t1[_K:])
  return _wfg_tail(t2)


def WFG8(x):
  y = _a(x) / _Z2
  t1 = y.copy()
  t1[_K:] = _b_param(y[_K:], _r_sum_before(y, range(_K + 1, _N + 1)))
  t2 = t1.copy()
  t2[_K:] = _s_linear(t1[_K:])
  return _wfg_tail(t2)


def WFG9(x):
  y = _a(x) / _Z2
  t1 = _i_head(y, "after")
  t2 = t1.copy()
  t2[:_K] = _s_decept(t1[:_K], 0.35, 0.001, 0.05)
  t2[_K:] = _s_multi(t1[_K:], 30, 95, 0.35)
  return _wfg_tail(t2, nonsep=True)


# --- Problem table: (function, n, m, lb, ub) --------------------------------

def _b(n, lo, hi):
  return [float(lo)] * n, [float(hi)] * n


_s12, _c12 = np.sin(pi / 12), np.cos(pi / 12)
_TABLE = {
    "BK1": (2, 2, *_b(2, -5, 10)),
    "CL1": (4, 2, [1., np.sqrt(2), np.sqrt(2), 1.], [3.] * 4),
    "Deb41": (2, 2, [0.1, 0.], [1., 1.]),
    "Deb512a": (2, 2, *_b(2, 0, 1)), "Deb512b": (2, 2, *_b(2, 0, 1)),
    "Deb512c": (2, 2, *_b(2, 0, 1)), "Deb513": (2, 2, *_b(2, 0, 1)),
    "Deb521a": (2, 2, *_b(2, 0, 1)), "Deb521b": (2, 2, *_b(2, 0, 1)),
    "Deb53": (2, 2, *_b(2, 0, 1)),
    "DG01": (1, 2, *_b(1, -10, 13)),
    "DPAM1": (10, 2, *_b(10, -0.3, 0.3)),
    "DTLZ1": (7, 3, *_b(7, 0, 1)), "DTLZ1n2": (2, 2, *_b(2, 0, 1)),
    "DTLZ2": (12, 3, *_b(12, 0, 1)), "DTLZ2n2": (2, 2, *_b(2, 0, 1)),
    "DTLZ3": (12, 3, *_b(12, 0, 1)), "DTLZ3n2": (2, 2, *_b(2, 0, 1)),
    "DTLZ4": (12, 3, *_b(12, 0, 1)), "DTLZ4n2": (2, 2, *_b(2, 0, 1)),
    "DTLZ5": (12, 3, *_b(12, 0, 1)), "DTLZ5n2": (2, 2, *_b(2, 0, 1)),
    "DTLZ6": (22, 3, *_b(22, 0, 1)), "DTLZ6n2": (2, 2, *_b(2, 0, 1)),
    "ex005": (2, 2, [-1., 1.], [2., 2.]),
    "Far1": (2, 2, *_b(2, -1, 1)),
    "FES1": (10, 2, *_b(10, 0, 1)), "FES2": (10, 3, *_b(10, 0, 1)),
    "FES3": (10, 4, *_b(10, 0, 1)),
    "Fonseca": (2, 2, *_b(2, -4, 4)),
    "I1": (8, 3, *_b(8, 0, 1)), "I2": (8, 3, *_b(8, 0, 1)), "I3": (8, 3, *_b(8, 0, 1)),
    "I4": (8, 3, *_b(8, 0, 1)), "I5": (8, 3, *_b(8, 0, 1)),
    "IKK1": (2, 3, *_b(2, -50, 50)),
    "IM1": (2, 2, [1., 1.], [4., 2.]),
    "Jin1": (2, 2, *_b(2, 0, 1)), "Jin2": (2, 2, *_b(2, 0, 1)),
    "Jin3": (2, 2, *_b(2, 0, 1)), "Jin4": (2, 2, *_b(2, 0, 1)),
    "Kursawe": (3, 2, *_b(3, -5, 5)),
    "L1ZDT4": (10, 2, [0.] + [-5.] * 9, [1.] + [5.] * 9),
    "L2ZDT1": (30, 2, *_b(30, 0, 1)), "L2ZDT2": (30, 2, *_b(30, 0, 1)),
    "L2ZDT3": (30, 2, *_b(30, 0, 1)), "L2ZDT4": (30, 2, *_b(30, 0, 1)),
    "L2ZDT6": (10, 2, *_b(10, 0, 1)),
    "L3ZDT1": (30, 2, *_b(30, 0, 1)), "L3ZDT2": (30, 2, *_b(30, 0, 1)),
    "L3ZDT3": (30, 2, *_b(30, 0, 1)), "L3ZDT4": (30, 2, *_b(30, 0, 1)),
    "L3ZDT6": (10, 2, *_b(10, 0, 1)),
    "LE1": (2, 2, *_b(2, 0, 1)),
    "lovison1": (2, 2, *_b(2, 0, 3)),
    "lovison2": (2, 2, [-0.5, -0.5], [0., 0.5]),
    "lovison3": (2, 2, [0., -4.], [6., 4.]),
    "lovison4": (2, 2, [0., -1.], [6., 1.]),
    "lovison5": (3, 3, *_b(3, -1, 4)), "lovison6": (3, 3, *_b(3, -1, 4)),
    "LRS1": (2, 2, *_b(2, -50, 50)),
    "MHHM1": (1, 3, *_b(1, 0, 1)), "MHHM2": (2, 3, *_b(2, 0, 1)),
    "MLF1": (1, 2, *_b(1, 0, 20)), "MLF2": (2, 2, *_b(2, -2, 2)),
    "MOP1": (1, 2, *_b(1, -1e5, 1e5)), "MOP2": (4, 2, *_b(4, -4, 4)),
    "MOP3": (2, 2, *_b(2, -pi, pi)), "MOP4": (3, 2, *_b(3, -5, 5)),
    "MOP5": (2, 3, *_b(2, -30, 30)), "MOP6": (2, 2, *_b(2, 0, 1)),
    "MOP7": (2, 3, *_b(2, -400, 400)),
    "OKA1": (2, 2, [6 * _s12, -2 * pi * _s12], [6 * _s12 + 2 * pi * _c12, 6 * _c12]),
    "OKA2": (3, 2, [-pi, -5., -5.], [pi, 5., 5.]),
    "QV1": (10, 2, *_b(10, -5.12, 5.12)),
    "Sch1": (1, 2, *_b(1, 0, 5)),
    "SK1": (1, 2, *_b(1, -10, 10)), "SK2": (4, 2, *_b(4, -10, 10)),
    "SP1": (2, 2, *_b(2, -1, 5)),
    "SSFYY1": (2, 2, *_b(2, -100, 100)), "SSFYY2": (1, 2, *_b(1, -100, 100)),
    "TKLY1": (4, 2, [0.1, 0., 0., 0.], [1.] * 4),
    "VFM1": (2, 3, *_b(2, -2, 2)),
    "VU1": (2, 2, *_b(2, -3, 3)), "VU2": (2, 2, *_b(2, -3, 3)),
    **{f"WFG{i}": (8, 3, [0.] * 8, list(2. * np.arange(1, 9))) for i in range(1, 10)},
    "ZDT1": (30, 2, *_b(30, 0, 1)), "ZDT2": (30, 2, *_b(30, 0, 1)),
    "ZDT3": (30, 2, *_b(30, 0, 1)),
    "ZDT4": (10, 2, [0.] + [-5.] * 9, [1.] + [5.] * 9),
    "ZDT6": (10, 2, *_b(10, 0, 1)),
    "ZLT1": (10, 3, *_b(10, -1000, 1000)),
}

PROBLEMS = {name: (globals()[name], *spec) for name, spec in _TABLE.items()}


def start_points(name):
  """Starting points of the DMulti-MADS and DMS experiments: n points evenly
  spread on the diagonal of the box (the box centre when n = 1)."""
  _, n, _, lb, ub = PROBLEMS[name]
  lb, ub = np.asarray(lb), np.asarray(ub)
  if n == 1:
    return [list((lb + ub) / 2)]
  return [list(lb + j * (ub - lb) / (n - 1)) for j in range(n)]
