"""
This script contains Python implementations of MATLAB functions
for multi-objective optimization test problems. The original
MATLAB functions are described in various academic papers.

Converted from MATLAB to Python by ChatGPT.
"""
import numpy as np

def BK1(x):
    """
    Example BK1 - Multiobjective test problem.
    As described by Huband et al. in IEEE Transactions on Evolutionary Computing (2006).
    
    Variables:
    x: numpy array, size (2,)
        -5 <= x[i] <= 10
    
    Returns:
    F: numpy array, size (2,)
    """
    f1 = x[0]**2 + x[1]**2
    f2 = (x[0] - 5)**2 + (x[1] - 5)**2
    return np.array([f1, f2])

def CL1(x):
    """
    Example CL1 - Four bar truss optimization problem.
    As described by F.Y. Cheng and X.S. Li (1999).
    
    Variables:
    x: numpy array, size (4,)
    
    Returns:
    F: numpy array, size (2,)
    """
    F = 10
    E = 2 * 10**5
    L = 200
    sigma = 10
    
    f1 = L * (2 * x[0] + np.sqrt(2) * x[1] + np.sqrt(x[2]) + x[3])
    f2 = F * L / E * (2 / x[0] + (2 * np.sqrt(2)) / x[1] - (2 * np.sqrt(2)) / x[2] + 2 / x[3])
    return np.array([f1, f2])

def Deb41(x):
    """
    Example 4.1 - Multi-modal Multi-objective Problem.
    As described by K. Deb (1999).
    
    Variables:
    x: numpy array, size (2,)
        0.1 <= x[0] <= 1.0
        0.0 <= x[1] <= 1.0
    
    Returns:
    F: numpy array, size (2,)
    """
    gx = 2 - np.exp(-((x[1] - 0.2) / 0.004)**2) - 0.8 * np.exp(-((x[1] - 0.6) / 0.4)**2)
    f1 = x[0]
    f2 = gx / x[0]
    return np.array([f1, f2])

# Additional functions for Deb512a, Deb512b, Deb512c, Deb513, Deb521a, Deb521b will follow
def Deb512a(x):
    beta = 1
    alpha = 0.25
    f1 = 4 * x[0]
    gx = 4 -3 * np.exp(-((x[1] - 0.2) / 0.02)**2) if x[1] <= 0.4 else 4 -2 * np.exp(-((x[1] - 0.7) / 0.2)**2)
    h = (1-(f1 / (beta * gx))**alpha) if f1 <= beta * gx else 0
    f2 = gx * h
    return np.array([f1, f2])

def Deb512b(x):
    beta = 1
    alpha = 4
    f1 = 4 * x[0]
    gx = 4 -3 * np.exp(-((x[1] - 0.2) / 0.02)**2) if x[1] <= 0.4 else 4 -2 * np.exp(-((x[1] - 0.7) / 0.2)**2)
    h = (1-(f1 / (beta * gx))**alpha) if f1 <= beta * gx else 0
    f2 = gx * h
    return np.array([f1, f2])

def Deb512c(x):
    beta = 1
    f1 = 4 * x[0]
    gx = 4 -3 * np.exp(-((x[1] - 0.2) / 0.02)**2) if x[1] <= 0.4 else 4 -2 * np.exp(-((x[1] - 0.7) / 0.2)**2)
    alpha = 0.25 + 3.75 * (gx - 1)
    h = (1-(f1 / (beta * gx))**alpha) if f1 <= beta * gx else 0
    f2 = gx * h
    return np.array([f1, f2])

def Deb513(x):
    beta = 1
    alpha = 2
    q = 4
    f1 = x[0]
    gx = 1 + 10 * x[1]
    h = 1 - (f1 / gx)**alpha - (f1/gx) * np.sin(2 * np.pi * q * f1)
    f2 = gx * h
    return np.array([f1, f2])

def Deb521a(x):
    gamma = 0.25
    f1 = x[0]
    gx = 1 + x[1]**gamma
    h = 1 - (f1 / gx)**2
    f2 = gx * h
    return np.array([f1, f2])

def Deb521b(x):
    gamma = 1
    f1 = x[0]
    gx = 1 + x[1]**gamma
    h = 1 - (f1 / gx)**2
    f2 = gx * h
    return np.array([f1, f2])


"""
This script contains Python implementations of MATLAB functions
for multi-objective optimization test problems. The original
MATLAB functions are described in various academic papers.

Converted from MATLAB to Python by ChatGPT.
"""
import numpy as np

def DG01(x):
    """
    Example DG01 - Multiobjective test problem.
    As described by Huband et al. (2006).
    
    Variables:
    x: numpy array, size (1,)
        -10 <= x[0] <= 13
    
    Returns:
    F: numpy array, size (2,)
    """
    f1 = np.sin(x[0])
    f2 = np.sin(x[0] + 0.7)
    return np.array([f1, f2])

def DPAM1(x):
    """
    Example DPAM1 - Multiobjective test problem.
    As described by Huband et al. (2006).
    """
    A = np.array([
        [0.218418, -0.620254, 0.843784, 0.914311, -0.788548, 0.428212, 0.103064, -0.47373, -0.300792, -0.185507],
        [0.330423, 0.151614, 0.884043, -0.272951, -0.993822, 0.511197, -0.0997948, -0.659756, 0.575496, 0.675617],
        [0.180332, -0.593814, -0.492722, 0.0646786, -0.666503, -0.945716, -0.334582, 0.611894, 0.281032, 0.508749],
        [-0.0265389, -0.920133, 0.308861, -0.0437502, -0.374203, 0.207359, -0.219433, 0.914104, 0.184408, 0.520599],
        [-0.88565, -0.375906, -0.708948, -0.37902, 0.576578, 0.0194674, -0.470262, 0.572576, 0.351245, -0.480477],
        [0.238261, -0.1596, -0.827302, 0.669248, 0.494475, 0.691715, -0.198585, 0.0492812, 0.959669, 0.884086],
        [-0.218632, -0.865161, -0.715997, 0.220772, 0.692356, 0.646453, -0.401724, 0.615443, -0.0601957, -0.748176],
        [-0.207987, -0.865931, 0.613732, -0.525712, -0.995728, 0.389633, -0.064173, 0.662131, -0.707048, -0.340423],
        [0.60624, 0.0951648, -0.160446, -0.394585, -0.167581, 0.0679849, 0.449799, 0.733505, -0.00918638, 0.00446808],
        [0.404396, 0.449996, 0.162711, 0.294454, -0.563345, -0.114993, 0.549589, -0.775141, 0.677726, 0.610715]
    ])  # Matrix from MATLAB code
    y = A @ x  # Matrix multiplication
    gx = 1 + 10 * (len(x) - 1) + np.sum(y[1:]**2 - 10 * np.cos(4 * np.pi * y[1:]))
    f1 = y[0]
    f2 = gx * np.exp(-y[0] / gx)
    return np.array([f1, f2])

def DTLZ1(x):
    """
    Example DTLZ1 - Multiobjective optimization problem.
    As described by Deb et al. (2002).
    """
    M = 3
    gx = 100 * (len(x) - M + 1 + np.sum((x[M-1:] - 0.5)**2 - np.cos(20 * np.pi * (x[M-1:] - 0.5))))
    f = np.ones(M)
    f[0] = 0.5 * (1 + gx) * np.prod(x[:M-1])
    for i in range(1, M):
        f[i] = 0.5 * (1 + gx) * np.prod(x[:M-i]) * (1 - x[M - i])
    return f

def DTLZ1n2(x):
    return DTLZ1(x[:2])

def DTLZ2(x):
    """
    Example DTLZ2 - Multiobjective test problem.
    """
    M = 3
    gx = np.sum((x[M-1:] - 0.5)**2)
    f = np.ones(M)
    f[0] = (1 + gx) * np.prod(np.cos(0.5 * np.pi * x[:M-1]))
    for i in range(1, M):
        f[i] = (1 + gx) * np.prod(np.cos(0.5 * np.pi * x[:M-i])) * np.sin(0.5 * np.pi * x[M - i])
    return f

def DTLZ2n2(x):
    return DTLZ2(x[:2])

def DTLZ3(x):
    """
    Example DTLZ3 - Multiobjective test problem.
    """
    M = 3
    gx = 100 * (len(x) - M + 1 + np.sum((x[M-1:] - 0.5)**2 - np.cos(20 * np.pi * (x[M-1:] - 0.5))))
    return DTLZ2(x)

def DTLZ3n2(x):
    return DTLZ3(x[:2])

def DTLZ4(x):
    """
    Example DTLZ4 - Multiobjective test problem.
    """
    M, alpha = 3, 100
    y = x**alpha
    gx = np.sum((y[M-1:] - 0.5)**2)
    return DTLZ2(y)

def DTLZ4n2(x):
    return DTLZ4(x[:2])

"""
This script contains Python implementations of MATLAB functions
for multi-objective optimization test problems. The original
MATLAB functions are described in various academic papers.

Converted from MATLAB to Python by ChatGPT.
"""
import numpy as np

def DG01(x):
    """
    Example DG01 - Multiobjective test problem.
    As described by Huband et al. (2006).
    """
    f1 = np.sin(x[0])
    f2 = np.sin(x[0] + 0.7)
    return np.array([f1, f2])

# Additional functions...

def DTLZ5(x):
    """
    Example DTLZ5 - Multiobjective test problem.
    As described by Deb et al. (2002).
    """
    M = 3
    gx = np.sum(x[M-1:]**0.1)
    theta = (np.pi / 2) * (1 + 2 * gx * x[:M-1]) / (2 * (1 + gx))
    f = np.ones(M)
    f[0] = (1 + gx) * np.prod(np.cos(theta[:-1]))
    for i in range(1, M-1):
        f[i] = (1 + gx) * np.prod(np.cos(theta[:-i])) * np.sin(theta[-i])
    f[M-1] = (1 + gx) * np.sin(0.5 * np.pi * x[0])
    return f

def DTLZ5n2(x):
    return DTLZ5(x[:2])

def DTLZ6(x):
    """
    Example DTLZ6 - Multiobjective test problem.
    """
    M = 3
    gx = 1 + (9 / (len(x) - M + 1)) * np.sum(x[M-1:])
    f = np.ones(M)
    f[-1] = (1 + gx) * (M - np.sum(x[:M-1] / (1 + gx) * (1 + np.sin(3 * np.pi * x[:M-1]))))
    f[:M-1] = x[:M-1]
    return f

def DTLZ6n2(x):
    return DTLZ6(x[:2])

def ex005(x):
    """
    Example ex005 - Simple multi-objective optimization problem.
    """
    f1 = x[0]**2 - x[1]**2
    f2 = x[0] / x[1]
    return np.array([f1, f2])

def Far1(x):
    """
    Example Far1 - Multiobjective test problem.
    """
    f1 = (-2 * np.exp(15 * (-(x[0] - 0.1)**2 - x[1]**2))
          - np.exp(20 * (-(x[0] - 0.6)**2 - (x[1] - 0.6)**2))
          + np.exp(20 * (-(x[0] + 0.6)**2 - (x[1] - 0.6)**2))
          + np.exp(20 * (-(x[0] - 0.6)**2 - (x[1] + 0.6)**2))
          + np.exp(20 * (-(x[0] + 0.6)**2 - (x[1] + 0.6)**2)))
    f2 = (2 * np.exp(20 * (-x[0]**2 - x[1]**2))
          + np.exp(20 * (-(x[0] - 0.4)**2 - (x[1] - 0.6)**2))
          - np.exp(20 * (-(x[0] + 0.5)**2 - (x[1] - 0.7)**2))
          - np.exp(20 * (-(x[0] - 0.5)**2 - (x[1] + 0.7)**2))
          + np.exp(20 * (-(x[0] + 0.4)**2 - (x[1] + 0.8)**2)))
    return np.array([f1, f2])

def Fonseca(x):
    """
    Example Fonseca - Multiobjective test problem.
    """
    f1 = 1 - np.exp(-(x[0] - 1)**2 - (x[1] + 1)**2)
    f2 = 1 - np.exp(-(x[0] + 1)**2 - (x[1] - 1)**2)
    return np.array([f1, f2])
"""
This script contains Python implementations of MATLAB functions
for multi-objective optimization test problems. The original
MATLAB functions are described in various academic papers.

Converted from MATLAB to Python by ChatGPT.
"""
import numpy as np

def DG01(x):
    """
    Example DG01 - Multiobjective test problem.
    As described by Huband et al. (2006).
    """
    f1 = np.sin(x[0])
    f2 = np.sin(x[0] + 0.7)
    return np.array([f1, f2])

# Additional functions...

def I1(x):
    """
    Example I1 - Multiobjective test problem.
    """
    M, k, l = 3, 4, 4
    n = k + l
    y = x / np.ones(n)
    t2 = np.ones(n)
    t2[:k] = y[:k]
    t2[k:] = np.abs(y[k:] - 0.35) / np.abs(np.floor(0.35 - y[k:]) + 0.35)
    t3 = np.ones(M)
    t3[:-1] = [np.sum(t2[i * k // (M - 1): (i + 1) * k // (M - 1)]) for i in range(M - 1)]
    t3[-1] = np.sum(t2[k:])
    xtmp = np.ones(M)
    xtmp[:-1] = (t3[:-1] - 0.5) + 0.5
    xtmp[-1] = t3[-1]
    h = np.ones(M)
    h[0] = np.prod(np.sin((np.pi / 2) * xtmp[:-1]))
    h[1:-1] = [np.prod(np.sin((np.pi / 2) * xtmp[:-i])) * np.cos((np.pi / 2) * xtmp[-i]) for i in range(1, M - 1)]
    h[-1] = np.cos((np.pi / 2) * xtmp[0])
    return xtmp[-1] + h

def IKK1(x):
    """
    Example IKK1 - Multiobjective test problem.
    """
    f1 = x[0]**2
    f2 = (x[0] - 20)**2
    f3 = x[1]**2
    return np.array([f1, f2, f3])

def IM1(x):
    """
    Example IM1 - Multiobjective test problem.
    """
    f1 = 2 * np.sqrt(x[0])
    f2 = x[0] * (1 - x[1]) + 5
    return np.array([f1, f2])

def Jin1(x):
    """
    Example Jin1 - Multiobjective test problem.
    """
    f1 = np.sum(x**2) / len(x)
    f2 = np.sum((x - 2)**2) / len(x)
    return np.array([f1, f2])

def Jin2(x):
    """
    Example Jin2 - Multiobjective test problem.
    """
    gx = 1 + (9 * np.sum(x[1:]) / (len(x) - 1))
    f1 = x[0]
    f2 = gx * (1 - np.sqrt(x[0] / gx))
    return np.array([f1, f2])

def Jin3(x):
    """
    Example Jin3 - Multiobjective test problem.
    """
    gx = 1 + (9 * np.sum(x[1:]) / (len(x) - 1))
    f1 = x[0]
    f2 = gx * (1 - (x[0] / gx)**2)
    return np.array([f1, f2])

"""
This script contains Python implementations of MATLAB functions
for multi-objective optimization test problems. The original
MATLAB functions are described in various academic papers.

Converted from MATLAB to Python by ChatGPT.
"""
import numpy as np

def DG01(x):
    """
    Example DG01 - Multiobjective test problem.
    As described by Huband et al. (2006).
    """
    f1 = np.sin(x[0])
    f2 = np.sin(x[0] + 0.7)
    return np.array([f1, f2])

# Additional functions...

def Jin4(x):
    """
    Example Jin4 - Multiobjective test problem.
    """
    n = 2
    gx = 1 + (9 * np.sum(x[1:n]) / (n - 1))
    f1 = x[0]
    f2 = gx * (1 - (x[0] / gx)**0.25 - (x[0] / gx)**4)
    return np.array([f1, f2])

def Kursawe(x):
    """
    Example Kursawe - Multiobjective test problem.
    """
    f1 = np.sum(-10 * np.exp(-0.2 * np.sqrt(x[:-1]**2 + x[1:]**2)))
    f2 = np.sum(np.abs(x)**0.8 + 5 * np.sin(x)**3)
    return np.array([f1, f2])

def L1ZDT4(x):
    """
    Example L1ZDT4 - Multiobjective test problem.
    """
    A = np.array([...])  # Matrix from MATLAB code
    y = A @ x  # Matrix multiplication
    f1 = y[0]**2
    gx = 1 + 10 * (len(x) - 1) + np.sum(y[1:]**2 - 10 * np.cos(4 * np.pi * y[1:]))
    f2 = gx * (1 - np.sqrt(f1 / gx))
    return np.array([f1, f2])

def L2ZDT1(x):
    """
    Example L2ZDT1 - Multiobjective test problem.
    """
    M = np.array([
        [0.218418, -0.620254, 0.843784, 0.914311, -0.788548, 0.428212, 0.103064, -0.47373, -0.300792, -0.185507],
        [0.330423, 0.151614, 0.884043, -0.272951, -0.993822, 0.511197, -0.0997948, -0.659756, 0.575496, 0.675617],
        [0.180332, -0.593814, -0.492722, 0.0646786, -0.666503, -0.945716, -0.334582, 0.611894, 0.281032, 0.508749],
        [-0.0265389, -0.920133, 0.308861, -0.0437502, -0.374203, 0.207359, -0.219433, 0.914104, 0.184408, 0.520599],
        [-0.88565, -0.375906, -0.708948, -0.37902, 0.576578, 0.0194674, -0.470262, 0.572576, 0.351245, -0.480477],
        [0.238261, -0.1596, -0.827302, 0.669248, 0.494475, 0.691715, -0.198585, 0.0492812, 0.959669, 0.884086],
        [-0.218632, -0.865161, -0.715997, 0.220772, 0.692356, 0.646453, -0.401724, 0.615443, -0.0601957, -0.748176],
        [-0.207987, -0.865931, 0.613732, -0.525712, -0.995728, 0.389633, -0.064173, 0.662131, -0.707048, -0.340423],
        [0.60624, 0.0951648, -0.160446, -0.394585, -0.167581, 0.0679849, 0.449799, 0.733505, -0.00918638, 0.00446808],
        [0.404396, 0.449996, 0.162711, 0.294454, -0.563345, -0.114993, 0.549589, -0.775141, 0.677726, 0.610715]
    ])  # Matrix from MATLAB code
    y = M @ x  # Matrix multiplication
    f1 = y[0]
    gx = 1 + 9 * np.sum(y[1:]) / (len(y) - 1)
    f2 = gx * (1 - np.sqrt(f1 / gx))
    return np.array([f1, f2])

def L2ZDT2(x):
    """
    Example L2ZDT2 - Multiobjective test problem.
    """
    M = np.array([
        [0.218418, -0.620254, 0.843784, 0.914311, -0.788548, 0.428212, 0.103064, -0.47373, -0.300792, -0.185507],
        [0.330423, 0.151614, 0.884043, -0.272951, -0.993822, 0.511197, -0.0997948, -0.659756, 0.575496, 0.675617],
        [0.180332, -0.593814, -0.492722, 0.0646786, -0.666503, -0.945716, -0.334582, 0.611894, 0.281032, 0.508749],
        [-0.0265389, -0.920133, 0.308861, -0.0437502, -0.374203, 0.207359, -0.219433, 0.914104, 0.184408, 0.520599],
        [-0.88565, -0.375906, -0.708948, -0.37902, 0.576578, 0.0194674, -0.470262, 0.572576, 0.351245, -0.480477],
        [0.238261, -0.1596, -0.827302, 0.669248, 0.494475, 0.691715, -0.198585, 0.0492812, 0.959669, 0.884086],
        [-0.218632, -0.865161, -0.715997, 0.220772, 0.692356, 0.646453, -0.401724, 0.615443, -0.0601957, -0.748176],
        [-0.207987, -0.865931, 0.613732, -0.525712, -0.995728, 0.389633, -0.064173, 0.662131, -0.707048, -0.340423],
        [0.60624, 0.0951648, -0.160446, -0.394585, -0.167581, 0.0679849, 0.449799, 0.733505, -0.00918638, 0.00446808],
        [0.404396, 0.449996, 0.162711, 0.294454, -0.563345, -0.114993, 0.549589, -0.775141, 0.677726, 0.610715]
    ])  # Matrix from MATLAB code
    y = M @ x
    f1 = y[0]
    gx = 1 + 9 * np.sum(y[1:]) / (len(y) - 1)
    f2 = gx * (1 - (f1 / gx)**2)
    return np.array([f1, f2])

def L2ZDT3(x):
    """
    Example L2ZDT3 - Multiobjective test problem.
    """
    M = np.array([
        [0.218418, -0.620254, 0.843784, 0.914311, -0.788548, 0.428212, 0.103064, -0.47373, -0.300792, -0.185507],
        [0.330423, 0.151614, 0.884043, -0.272951, -0.993822, 0.511197, -0.0997948, -0.659756, 0.575496, 0.675617],
        [0.180332, -0.593814, -0.492722, 0.0646786, -0.666503, -0.945716, -0.334582, 0.611894, 0.281032, 0.508749],
        [-0.0265389, -0.920133, 0.308861, -0.0437502, -0.374203, 0.207359, -0.219433, 0.914104, 0.184408, 0.520599],
        [-0.88565, -0.375906, -0.708948, -0.37902, 0.576578, 0.0194674, -0.470262, 0.572576, 0.351245, -0.480477],
        [0.238261, -0.1596, -0.827302, 0.669248, 0.494475, 0.691715, -0.198585, 0.0492812, 0.959669, 0.884086],
        [-0.218632, -0.865161, -0.715997, 0.220772, 0.692356, 0.646453, -0.401724, 0.615443, -0.0601957, -0.748176],
        [-0.207987, -0.865931, 0.613732, -0.525712, -0.995728, 0.389633, -0.064173, 0.662131, -0.707048, -0.340423],
        [0.60624, 0.0951648, -0.160446, -0.394585, -0.167581, 0.0679849, 0.449799, 0.733505, -0.00918638, 0.00446808],
        [0.404396, 0.449996, 0.162711, 0.294454, -0.563345, -0.114993, 0.549589, -0.775141, 0.677726, 0.610715]
    ])  # Matrix from MATLAB code
    y = M @ x
    f1 = y[0]
    gx = 1 + 9 * np.sum(y[1:]) / (len(y) - 1)
    f2 = gx * (1 - np.sqrt(f1 / gx) - (f1 / gx) * np.sin(10 * np.pi * f1))
    return np.array([f1, f2])

def L2ZDT4(x):
    """
    Example L2ZDT4 - Multiobjective test problem.
    """
    M = np.array([
        [0.218418, -0.620254, 0.843784, 0.914311, -0.788548, 0.428212, 0.103064, -0.47373, -0.300792, -0.185507],
        [0.330423, 0.151614, 0.884043, -0.272951, -0.993822, 0.511197, -0.0997948, -0.659756, 0.575496, 0.675617],
        [0.180332, -0.593814, -0.492722, 0.0646786, -0.666503, -0.945716, -0.334582, 0.611894, 0.281032, 0.508749],
        [-0.0265389, -0.920133, 0.308861, -0.0437502, -0.374203, 0.207359, -0.219433, 0.914104, 0.184408, 0.520599],
        [-0.88565, -0.375906, -0.708948, -0.37902, 0.576578, 0.0194674, -0.470262, 0.572576, 0.351245, -0.480477],
        [0.238261, -0.1596, -0.827302, 0.669248, 0.494475, 0.691715, -0.198585, 0.0492812, 0.959669, 0.884086],
        [-0.218632, -0.865161, -0.715997, 0.220772, 0.692356, 0.646453, -0.401724, 0.615443, -0.0601957, -0.748176],
        [-0.207987, -0.865931, 0.613732, -0.525712, -0.995728, 0.389633, -0.064173, 0.662131, -0.707048, -0.340423],
        [0.60624, 0.0951648, -0.160446, -0.394585, -0.167581, 0.0679849, 0.449799, 0.733505, -0.00918638, 0.00446808],
        [0.404396, 0.449996, 0.162711, 0.294454, -0.563345, -0.114993, 0.549589, -0.775141, 0.677726, 0.610715]
    ])  # Matrix from MATLAB code
    y = M @ x
    f1 = y[0]
    gx = 1 + 9 * np.sum(y[1:]) / (len(y) - 1)
    f2 = gx * (1 - (f1 / gx)**0.25 - (f1 / gx)**4)
    return np.array([f1, f2])

def L2ZDT6(x):
    """
    Example L2ZDT6 - Multiobjective test problem.
    """
    M = np.array([
        [0.218418, -0.620254, 0.843784, 0.914311, -0.788548, 0.428212, 0.103064, -0.47373, -0.300792, -0.185507],
        [0.330423, 0.151614, 0.884043, -0.272951, -0.993822, 0.511197, -0.0997948, -0.659756, 0.575496, 0.675617],
        [0.180332, -0.593814, -0.492722, 0.0646786, -0.666503, -0.945716, -0.334582, 0.611894, 0.281032, 0.508749],
        [-0.0265389, -0.920133, 0.308861, -0.0437502, -0.374203, 0.207359, -0.219433, 0.914104, 0.184408, 0.520599],
        [-0.88565, -0.375906, -0.708948, -0.37902, 0.576578, 0.0194674, -0.470262, 0.572576, 0.351245, -0.480477],
        [0.238261, -0.1596, -0.827302, 0.669248, 0.494475, 0.691715, -0.198585, 0.0492812, 0.959669, 0.884086],
        [-0.218632, -0.865161, -0.715997, 0.220772, 0.692356, 0.646453, -0.401724, 0.615443, -0.0601957, -0.748176],
        [-0.207987, -0.865931, 0.613732, -0.525712, -0.995728, 0.389633, -0.064173, 0.662131, -0.707048, -0.340423],
        [0.60624, 0.0951648, -0.160446, -0.394585, -0.167581, 0.0679849, 0.449799, 0.733505, -0.00918638, 0.00446808],
        [0.404396, 0.449996, 0.162711, 0.294454, -0.563345, -0.114993, 0.549589, -0.775141, 0.677726, 0.610715]
    ])  # Matrix from MATLAB code
    y = M @ x
    f1 = 1 - np.exp(-4 * y[0]) * (np.sin(6 * np.pi * y[0])**6)
    gx = 1 + 9 * (np.sum(y[1:]) / (len(y) - 1))**0.25
    f2 = gx * (1 - (f1 / gx)**2)
    return np.array([f1, f2])

def L3ZDT1(x):
    """
    Example L3ZDT1 - Multiobjective test problem.
    """
    M = np.array([
        [0.218418, -0.620254, 0.843784, 0.914311, -0.788548, 0.428212, 0.103064, -0.47373, -0.300792, -0.185507],
        [0.330423, 0.151614, 0.884043, -0.272951, -0.993822, 0.511197, -0.0997948, -0.659756, 0.575496, 0.675617],
        [0.180332, -0.593814, -0.492722, 0.0646786, -0.666503, -0.945716, -0.334582, 0.611894, 0.281032, 0.508749],
        [-0.0265389, -0.920133, 0.308861, -0.0437502, -0.374203, 0.207359, -0.219433, 0.914104, 0.184408, 0.520599],
        [-0.88565, -0.375906, -0.708948, -0.37902, 0.576578, 0.0194674, -0.470262, 0.572576, 0.351245, -0.480477],
        [0.238261, -0.1596, -0.827302, 0.669248, 0.494475, 0.691715, -0.198585, 0.0492812, 0.959669, 0.884086],
        [-0.218632, -0.865161, -0.715997, 0.220772, 0.692356, 0.646453, -0.401724, 0.615443, -0.0601957, -0.748176],
        [-0.207987, -0.865931, 0.613732, -0.525712, -0.995728, 0.389633, -0.064173, 0.662131, -0.707048, -0.340423],
        [0.60624, 0.0951648, -0.160446, -0.394585, -0.167581, 0.0679849, 0.449799, 0.733505, -0.00918638, 0.00446808],
        [0.404396, 0.449996, 0.162711, 0.294454, -0.563345, -0.114993, 0.549589, -0.775141, 0.677726, 0.610715]
    ])  # Matrix from MATLAB code
    y = M @ x
    f1 = y[0]
    gx = 1 + 9 * np.sum(y[1:]) / (len(y) - 1)
    f2 = gx * (1 - np.sqrt(f1 / gx))
    return np.array([f1, f2])

def L3ZDT2(x):
    """
    Example L3ZDT2 - Multiobjective test problem.
    """
    M = np.array([
        [0.218418, -0.620254, 0.843784, 0.914311, -0.788548, 0.428212, 0.103064, -0.47373, -0.300792, -0.185507],
        [0.330423, 0.151614, 0.884043, -0.272951, -0.993822, 0.511197, -0.0997948, -0.659756, 0.575496, 0.675617],
        [0.180332, -0.593814, -0.492722, 0.0646786, -0.666503, -0.945716, -0.334582, 0.611894, 0.281032, 0.508749],
        [-0.0265389, -0.920133, 0.308861, -0.0437502, -0.374203, 0.207359, -0.219433, 0.914104, 0.184408, 0.520599],
        [-0.88565, -0.375906, -0.708948, -0.37902, 0.576578, 0.0194674, -0.470262, 0.572576, 0.351245, -0.480477],
        [0.238261, -0.1596, -0.827302, 0.669248, 0.494475, 0.691715, -0.198585, 0.0492812, 0.959669, 0.884086],
        [-0.218632, -0.865161, -0.715997, 0.220772, 0.692356, 0.646453, -0.401724, 0.615443, -0.0601957, -0.748176],
        [-0.207987, -0.865931, 0.613732, -0.525712, -0.995728, 0.389633, -0.064173, 0.662131, -0.707048, -0.340423],
        [0.60624, 0.0951648, -0.160446, -0.394585, -0.167581, 0.0679849, 0.449799, 0.733505, -0.00918638, 0.00446808],
        [0.404396, 0.449996, 0.162711, 0.294454, -0.563345, -0.114993, 0.549589, -0.775141, 0.677726, 0.610715]
    ])  # Matrix from MATLAB code
    y = M @ x
    f1 = y[0]
    gx = 1 + 9 * np.sum(y[1:]) / (len(y) - 1)
    f2 = gx * (1 - (f1 / gx)**2)
    return np.array([f1, f2])

"""
This script contains Python implementations of MATLAB functions
for multi-objective optimization test problems. The original
MATLAB functions are described in various academic papers.

Converted from MATLAB to Python by ChatGPT.
"""
import numpy as np

def L3ZDT6(x):
    """
    Example L3ZDT6 - Multiobjective test problem.
    As described by K. Deb, A. Sinha, and S. Kukkonen (2006).
    """
    M = np.array([
        [0.218418, -0.620254, 0.843784, 0.914311, -0.788548, 0.428212, 0.103064, -0.47373, -0.300792, -0.185507],
        [0.330423, 0.151614, 0.884043, -0.272951, -0.993822, 0.511197, -0.0997948, -0.659756, 0.575496, 0.675617],
        [0.180332, -0.593814, -0.492722, 0.0646786, -0.666503, -0.945716, -0.334582, 0.611894, 0.281032, 0.508749],
        [-0.0265389, -0.920133, 0.308861, -0.0437502, -0.374203, 0.207359, -0.219433, 0.914104, 0.184408, 0.520599],
        [-0.88565, -0.375906, -0.708948, -0.37902, 0.576578, 0.0194674, -0.470262, 0.572576, 0.351245, -0.480477],
        [0.238261, -0.1596, -0.827302, 0.669248, 0.494475, 0.691715, -0.198585, 0.0492812, 0.959669, 0.884086],
        [-0.218632, -0.865161, -0.715997, 0.220772, 0.692356, 0.646453, -0.401724, 0.615443, -0.0601957, -0.748176],
        [-0.207987, -0.865931, 0.613732, -0.525712, -0.995728, 0.389633, -0.064173, 0.662131, -0.707048, -0.340423],
        [0.60624, 0.0951648, -0.160446, -0.394585, -0.167581, 0.0679849, 0.449799, 0.733505, -0.00918638, 0.00446808],
        [0.404396, 0.449996, 0.162711, 0.294454, -0.563345, -0.114993, 0.549589, -0.775141, 0.677726, 0.610715]
    ])
    y = M @ (x ** 2)
    f1 = y[0] ** 2
    gx = 1 + 9 * (np.sum(y[1:] ** 2) / (len(y) - 1)) ** 0.25
    f2 = gx * (1 - (f1 / gx) ** 2)
    return np.array([f1, f2])

def LE1(x):
    """
    Example LE1 - Multiobjective test problem.
    """
    f1 = (x[0] ** 2 + x[1] ** 2) ** 0.125
    f2 = ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) ** 0.25
    return np.array([f1, f2])

def lovison1(x):
    """
    Example Lovison1 - Multiobjective test problem.
    """
    f1 = -1.05 * x[0] ** 2 - 0.98 * x[1] ** 2
    f2 = -0.99 * (x[0] - 3) ** 2 - 1.03 * (x[1] - 2.5) ** 2
    return -np.array([f1, f2])

def lovison2(x):
    """
    Example Lovison2 - Multiobjective test problem.
    """
    f1 = -x[1]
    f2 = (x[1] - x[0] ** 3) / (x[0] + 1)
    return -np.array([f1, f2])

def lovison3(x):
    """
    Example Lovison3 - Multiobjective test problem.
    """
    f1 = -x[0] ** 2 - x[1] ** 2
    f2 = -(x[0] - 6) ** 2 + (x[1] + 0.3) ** 2
    return -np.array([f1, f2])

"""
This script contains Python implementations of MATLAB functions
for multi-objective optimization test problems. The original
MATLAB functions are described in various academic papers.

Converted from MATLAB to Python by ChatGPT.
"""
import numpy as np

def L3ZDT6(x):
    """
    Example L3ZDT6 - Multiobjective test problem.
    """
    M = np.array([...])  # Matrix from MATLAB code
    y = M @ (x ** 2)
    f1 = y[0] ** 2
    gx = 1 + 9 * (np.sum(y[1:] ** 2) / (len(y) - 1)) ** 0.25
    f2 = gx * (1 - (f1 / gx) ** 2)
    return np.array([f1, f2])

def LE1(x):
    """
    Example LE1 - Multiobjective test problem.
    """
    f1 = (x[0] ** 2 + x[1] ** 2) ** 0.125
    f2 = ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) ** 0.25
    return np.array([f1, f2])

def lovison4(x):
    """
    Example Lovison4 - Multiobjective test problem.
    """
    f1 = -x[0]**2 - x[1]**2 - 4 * (np.exp(-(x[0] + 2)**2 - x[1]**2) + np.exp(-(x[0] - 2)**2 - x[1]**2))
    f2 = -(x[0] - 6)**2 - (x[1] + 0.5)**2
    return -np.array([f1, f2])

def lovison5(x):
    """
    Example Lovison5 - Multiobjective test problem.
    """
    C = np.array([[0.218418, -0.620254, 0.843784],
                  [0.914311, -0.788548, 0.428212],
                  [0.103064, -0.47373, -0.300792]])
    alpha = np.array([[0.407247, 0.665212, 0.575807],
                      [0.942022, 0.363525, 0.00308876],
                      [0.755598, 0.450103, 0.170122]])
    beta = np.array([0.575496, 0.675617, 0.180332])
    gamma = np.array([-0.593814, -0.492722, 0.0646786])
    f_tmp = -np.sum(alpha * (x[:, None] - C)**2, axis=1)
    f1 = f_tmp[0]
    f2 = f_tmp[1] + beta[1] * np.sin(np.pi * (x[0] + x[1]) / gamma[1])
    f3 = f_tmp[2] + beta[2] * np.cos(np.pi * (x[0] - x[1]) / gamma[2])
    return -np.array([f1, f2, f3])

def LRS1(x):
    """
    Example LRS1 - Multiobjective test problem.
    """
    f1 = x[0]**2 + x[1]**2
    f2 = (x[0] + 2)**2 + x[1]**2
    return np.array([f1, f2])

def MHHM1(x):
    """
    Example MHHM1 - Multiobjective test problem.
    """
    f1 = (x[0] - 0.8)**2
    f2 = (x[0] - 0.85)**2
    f3 = (x[0] - 0.9)**2
    return np.array([f1, f2, f3])

def MLF1(x):
    """
    Example MLF1 - Multiobjective test problem.
    """
    f1 = (1 + x[0] / 20) * np.sin(x[0])
    f2 = (1 + x[0] / 20) * np.cos(x[0])
    return np.array([f1, f2])

def MOP1(x):
    """
    Example MOP1 - Multiobjective test problem.
    """
    f1 = x[0]**2
    f2 = (x[0] - 2)**2
    return np.array([f1, f2])

"""
This script contains Python implementations of MATLAB functions
for multi-objective optimization test problems. The original
MATLAB functions are described in various academic papers.

Converted from MATLAB to Python by ChatGPT.
"""
import numpy as np

def MOP2(x):
    """
    Example MOP2 - Multiobjective test problem.
    """
    n = 4
    f1 = 1 - np.exp(-np.sum((x[:n] - 1 / np.sqrt(n))**2))
    f2 = 1 - np.exp(-np.sum((x[:n] + 1 / np.sqrt(n))**2))
    return np.array([f1, f2])

def MOP3(x):
    """
    Example MOP3 - Multiobjective test problem.
    """
    A1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
    A2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)
    B1 = 0.5 * np.sin(x[0]) - 2 * np.cos(x[0]) + np.sin(x[1]) - 1.5 * np.cos(x[1])
    B2 = 1.5 * np.sin(x[0]) - np.cos(x[0]) + 2 * np.sin(x[1]) - 0.5 * np.cos(x[1])
    f1 = -1 - (A1 - B1)**2 - (A2 - B2)**2
    f2 = -(x[0] + 3)**2 - (x[1] + 1)**2
    return -np.array([f1, f2])

def MOP4(x):
    """
    Example MOP4 - Multiobjective test problem.
    """
    f1 = np.sum(-10 * np.exp(-0.2 * np.sqrt(x[:-1]**2 + x[1:]**2)))
    f2 = np.sum(np.abs(x)**0.8 + 5 * np.sin(x**3))
    return np.array([f1, f2])

def MOP5(x):
    """
    Example MOP5 - Multiobjective test problem.
    """
    f1 = 0.5 * (x[0]**2 + x[1]**2) + np.sin(x[0]**2 + x[1]**2)
    f2 = (3 * x[0] - 2 * x[1] + 4)**2 / 8 + (x[0] - x[1] + 1)**2 / 27 + 15
    f3 = 1 / (x[0]**2 + x[1]**2 + 1) - 1.1 * np.exp(-x[0]**2 - x[1]**2)
    return np.array([f1, f2, f3])

def OKA1(x):
    """
    Example OKA1 - Multiobjective test problem.
    """
    y1 = np.cos(np.pi/12) * x[0] - np.sin(np.pi/12) * x[1]
    y2 = np.sin(np.pi/12) * x[0] + np.cos(np.pi/12) * x[1]
    f1 = y1
    f2 = np.sqrt(2 * np.pi) - np.sqrt(np.abs(y1)) + 2 * np.abs(y2 - 3 * np.cos(y1) - 3)**(1/3)
    return np.array([f1, f2])

def QV1(x):
    """
    Example QV1 - Multiobjective test problem.
    """
    n = 10
    f1 = np.sum((x[:n]**2 - 10 * np.cos(2 * np.pi * x[:n]) + 10) / n)**0.25
    f2 = np.sum(((x[:n] - 1.5)**2 - 10 * np.cos(2 * np.pi * (x[:n] - 1.5)) + 10) / n)**0.25
    return np.array([f1, f2])

def Sch1(x):
    """
    Example Sch1 - Multiobjective test problem.
    """
    if x[0] <= 1:
        f1 = -x[0]
    elif x[0] <= 3:
        f1 = -2 + x[0]
    elif x[0] <= 4:
        f1 = 4 - x[0]
    else:
        f1 = -4 + x[0]
    f2 = (x[0] - 5)**2
    return np.array([f1, f2])

def SK1(x):
    """
    SK1 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = x[0]**2
    f[1] = (x[0]-2)**2
    return f

def SK2(x):
    """
    SK2 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = -(x[0] + np.sin(x[0]))
    f[1] = -(x[0] + np.cos(x[0]))
    return f

def SP1(x):
    """
    SP1 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = (x[0] - 1)**2 + (x[0] - x[1])**2
    f[1] = (x[1] - 3)**2 + (x[0] - x[1])**2
    return f

def SSFYY1(x):
    """
    SSFYY1 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = x[0]**2 + x[1]**2
    f[1] = (x[0]-1)**2 + x[1]**2
    return f

def SSFYY2(x):
    """
    SSFYY2 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = x[0]**2 + x[1]**2 + x[2]**2
    f[1] = (x[0]-1)**2 + (x[1]-1)**2 + x[2]**2
    return f

def TKLY1(x):
    """
    TKLY1 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = x[0]
    g = 1 + x[1]**2
    f[1] = g * (1 - np.sqrt(f[0]/g))
    return f

def VFM1(x):
    """
    VFM1 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = x[0]**2 + (x[1]-1)**2
    f[1] = x[0]**2 + (x[1]+1)**2 + 2
    return f

def VU1(x):
    """
    VU1 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = 1/(x[0]**2 + x[1]**2 + 1)
    f[1] = x[0]**2 + 3*x[1]**2 + 1
    return f

def VU2(x):
    """
    VU2 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = x[0] + x[1] + 1
    f[1] = x[0]**2 + 2*x[1] - 1
    return f

def SK1(x):
    """
    SK1 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = x[0]**2
    f[1] = (x[0]-2)**2
    return f

def SK2(x):
    """
    SK2 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = -(x[0] + np.sin(x[0]))
    f[1] = -(x[0] + np.cos(x[0]))
    return f

def SP1(x):
    """
    SP1 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = (x[0] - 1)**2 + (x[0] - x[1])**2
    f[1] = (x[1] - 3)**2 + (x[0] - x[1])**2
    return f

def SSFYY1(x):
    """
    SSFYY1 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = x[0]**2 + x[1]**2
    f[1] = (x[0]-1)**2 + x[1]**2
    return f

def SSFYY2(x):
    """
    SSFYY2 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = x[0]**2 + x[1]**2 + x[2]**2
    f[1] = (x[0]-1)**2 + (x[1]-1)**2 + x[2]**2
    return f

def TKLY1(x):
    """
    TKLY1 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = x[0]
    g = 1 + x[1]**2
    f[1] = g * (1 - np.sqrt(f[0]/g))
    return f

def VFM1(x):
    """
    VFM1 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = x[0]**2 + (x[1]-1)**2
    f[1] = x[0]**2 + (x[1]+1)**2 + 2
    return f

def VU1(x):
    """
    VU1 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = 1/(x[0]**2 + x[1]**2 + 1)
    f[1] = x[0]**2 + 3*x[1]**2 + 1
    return f

def VU2(x):
    """
    VU2 test problem
    Input: x - Decision vector
    Output: f - Objective vector
    """
    f = np.zeros(2)
    f[0] = x[0] + x[1] + 1
    f[1] = x[0]**2 + 2*x[1] - 1
    return f