import numpy as np

def BK1(x):
    """Binh and Korn test problem BK1"""
    x1 = x[0]
    x2 = x[1]
    f1 = 4*x1**2 + 4*x2**2
    f2 = (x1-5)**2 + (x2-5)**2
    return np.array([f1, f2])

def CL1(x):
    """Chankong and Haimes test problem CL1"""
    x1 = x[0]
    x2 = x[1]
    f1 = 2 + (x1-2)**2 + (x2-1)**2
    f2 = 9*x1 - (x2-1)**2
    return np.array([f1, f2])

def Deb41(x):
    """Deb test problem 41"""
    x1 = x[0]
    f1 = x1
    g = 1 + x[1]**2
    h = 1 - (f1/g)**2
    f2 = g*h
    return np.array([f1, f2])

def Deb53(x):
    """Deb test problem 53"""
    x1 = x[0] 
    f1 = x1
    g = 1 + x[1]**2
    h = 1 - np.sqrt(f1/g)
    f2 = g*h
    return np.array([f1, f2])

def DTLZ1(x, M):
    """
    DTLZ1 test problem
    Args:
        x: Decision variables vector
        M: Number of objectives
    """
    k = len(x) - M + 1
    g = 100*(k + np.sum((x[M-1:] - 0.5)**2 - np.cos(20*np.pi*(x[M-1:] - 0.5))))
    f = np.zeros(M)
    for i in range(M):
        f[i] = 0.5*(1 + g)
        for j in range(M-1-i):
            f[i] *= x[j]
        if i > 0:
            f[i] *= (1 - x[M-1-i])
    return f

def DTLZ2(x, M):
    """DTLZ2 test problem"""
    k = len(x) - M + 1
    g = np.sum((x[M-1:] - 0.5)**2)
    f = np.zeros(M)
    for i in range(M):
        f[i] = (1 + g)
        for j in range(M-1-i):
            f[i] *= np.cos(x[j]*np.pi/2)
        if i > 0:
            f[i] *= np.sin(x[M-1-i]*np.pi/2)
    return f

def Fonseca(x):
    """Fonseca test problem"""
    n = len(x)
    f1 = 1 - np.exp(-np.sum((x - 1/np.sqrt(n))**2))
    f2 = 1 - np.exp(-np.sum((x + 1/np.sqrt(n))**2))
    return np.array([f1, f2])

def Kursawe(x):
    """Kursawe test problem"""
    n = len(x)
    f1 = np.sum(-10*np.exp(-0.2*np.sqrt(x[:-1]**2 + x[1:]**2)))
    f2 = np.sum(np.abs(x)**0.8 + 5*np.sin(x**3))
    return np.array([f1, f2])

def ZDT1(x):
    """ZDT1 test problem"""
    n = len(x)
    f1 = x[0]
    g = 1 + 9*np.sum(x[1:])/(n-1)
    h = 1 - np.sqrt(f1/g)
    f2 = g*h
    return np.array([f1, f2])

def ZDT2(x):
    """ZDT2 test problem"""
    n = len(x)
    f1 = x[0]
    g = 1 + 9*np.sum(x[1:])/(n-1)
    h = 1 - (f1/g)**2
    f2 = g*h
    return np.array([f1, f2])

def ZDT3(x):
    """ZDT3 test problem"""
    n = len(x)
    f1 = x[0]
    g = 1 + 9*np.sum(x[1:])/(n-1)
    h = 1 - np.sqrt(f1/g) - (f1/g)*np.sin(10*np.pi*f1)
    f2 = g*h
    return np.array([f1, f2])

def ZDT4(x):
    """ZDT4 test problem"""
    n = len(x)
    f1 = x[0]
    g = 1 + 10*(n-1) + np.sum(x[1:]**2 - 10*np.cos(4*np.pi*x[1:]))
    h = 1 - np.sqrt(f1/g)
    f2 = g*h
    return np.array([f1, f2])

def ZDT6(x):
    """ZDT6 test problem"""
    n = len(x)
    f1 = 1 - np.exp(-4*x[0])*np.sin(6*np.pi*x[0])**6
    g = 1 + 9*(np.sum(x[1:])/(n-1))**0.25
    h = 1 - (f1/g)**2
    f2 = g*h
    return np.array([f1, f2])

def WFG1(x, M):
    """WFG1 test problem"""
    z = x.copy()
    k = M - 1
    l = len(x) - k
    
    # First transition
    t1 = np.zeros_like(z)
    t1[:k] = z[:k]
    t1[k:] = np.abs(z[k:] - 0.35)/0.35
    
    # Second transition 
    t2 = t1.copy()
    for i in range(len(t2)):
        t2[i] = 1 + np.cos((20*t1[i] - 10)*np.pi/6)/20
    
    # Third transition
    t3 = t2.copy()
    for i in range(k, len(t3)):
        t3[i] = t2[i]**0.2
    
    # Define objectives
    f = np.zeros(M)
    for i in range(M):
        f[i] = t3[-1] + 2*i*np.sum(t3[:-1])/len(t3[:-1])
    return f

def WFG2(x, M):
    """WFG2 test problem"""
    z = x.copy()
    k = M - 1
    l = len(x) - k
    
    # First transition
    t1 = np.zeros_like(z)
    t1[:k] = z[:k]
    t1[k:] = np.abs(z[k:] - 0.35)/0.35
    
    # Second transition
    t2 = t1.copy()
    for i in range(k, len(t2), 2):
        if i+1 < len(t2):
            t2[i] = 2*t1[i] + t1[i+1]
            t2[i+1] = 2*t1[i+1] + t1[i]
    
    # Define objectives
    f = np.zeros(M)
    for i in range(M):
        f[i] = t2[-1] + 2*i*np.sum(t2[:-1])/len(t2[:-1])
    return f

def DTLZ1n2(x):
    """DTLZ1 with 2 objectives"""
    return DTLZ1(x, 2)

def MLF1(x):
    """MLF1 test problem"""
    x1 = x[0]
    x2 = x[1]
    f1 = (1 + x2)/x1
    f2 = x1*(1 + x2)
    return np.array([f1, f2])

def MLF2(x):
    """MLF2 test problem"""
    x1 = x[0]
    x2 = x[1]
    f1 = 2 - np.exp(-((x1-1)/1.2)**2 - ((x2-1)/0.8)**2)
    f2 = 2 - np.exp(-((x1+1)/1.2)**2 - ((x2+1)/0.8)**2)
    return np.array([f1, f2])

def VFM1(x):
    """VFM1 test problem"""
    x1 = x[0]
    x2 = x[1]
    f1 = x1**2 + (x2-1)**2
    f2 = x1**2 + (x2+1)**2 + 1
    return np.array([f1, f2])

def SK1(x):
    """SK1 test problem"""
    x1 = x[0]
    x2 = x[1]
    f1 = (x1-2)**2 + (x2-1)**2 + 2
    f2 = 9*x1 - (x2-1)**2
    return np.array([f1, f2])

def SK2(x):
    """SK2 test problem"""
    x1 = x[0]
    x2 = x[1]
    f1 = -(x1 + 3*x2 + 1)
    f2 = -(x1/2 + 2*x2 - 1)
    return np.array([f1, f2])

def SP1(x):
    """SP1 test problem"""
    x1 = x[0]
    x2 = x[1]
    f1 = (x1 - 1)**2 + (x1 - x2)**2
    f2 = (x2 - 3)**2 + (x1 - x2)**2
    return np.array([f1, f2])

def Sch1(x):
    """Schaffer test problem"""
    x1 = x[0]
    f1 = x1**2
    f2 = (x1-2)**2
    return np.array([f1, f2])

def WFG1(x, M):
    """WFG1 test problem"""
    z = x.copy()
    k = M - 1
    l = len(x) - k
    
    # First transition
    t1 = np.zeros_like(z)
    t1[:k] = z[:k]
    t1[k:] = np.abs(z[k:] - 0.35)/0.35
    
    # Second transition 
    t2 = t1.copy()
    for i in range(len(t2)):
        t2[i] = 1 + np.cos((20*t1[i] - 10)*np.pi/6)/20
    
    # Third transition
    t3 = t2.copy()
    for i in range(k, len(t3)):
        t3[i] = t2[i]**0.2
    
    # Define objectives
    f = np.zeros(M)
    for i in range(M):
        f[i] = t3[-1] + 2*i*np.sum(t3[:-1])/len(t3[:-1])
    return f

def WFG2(x, M):
    """WFG2 test problem"""
    z = x.copy()
    k = M - 1
    l = len(x) - k
    
    # First transition
    t1 = np.zeros_like(z)
    t1[:k] = z[:k]
    t1[k:] = np.abs(z[k:] - 0.35)/0.35
    
    # Second transition
    t2 = t1.copy()
    for i in range(k, len(t2), 2):
        if i+1 < len(t2):
            t2[i] = 2*t1[i] + t1[i+1]
            t2[i+1] = 2*t1[i+1] + t1[i]
    
    # Define objectives
    f = np.zeros(M)
    for i in range(M):
        f[i] = t2[-1] + 2*i*np.sum(t2[:-1])/len(t2[:-1])
    return f

def DTLZ1n2(x):
    """DTLZ1 with 2 objectives"""
    return DTLZ1(x, 2)

def MLF1(x):
    """MLF1 test problem"""
    x1 = x[0]
    x2 = x[1]
    f1 = (1 + x2)/x1
    f2 = x1*(1 + x2)
    return np.array([f1, f2])

def MLF2(x):
    """MLF2 test problem"""
    x1 = x[0]
    x2 = x[1]
    f1 = 2 - np.exp(-((x1-1)/1.2)**2 - ((x2-1)/0.8)**2)
    f2 = 2 - np.exp(-((x1+1)/1.2)**2 - ((x2+1)/0.8)**2)
    return np.array([f1, f2])

def VFM1(x):
    """VFM1 test problem"""
    x1 = x[0]
    x2 = x[1]
    f1 = x1**2 + (x2-1)**2
    f2 = x1**2 + (x2+1)**2 + 1
    return np.array([f1, f2])

def SK1(x):
    """SK1 test problem"""
    x1 = x[0]
    x2 = x[1]
    f1 = (x1-2)**2 + (x2-1)**2 + 2
    f2 = 9*x1 - (x2-1)**2
    return np.array([f1, f2])

def SK2(x):
    """SK2 test problem"""
    x1 = x[0]
    x2 = x[1]
    f1 = -(x1 + 3*x2 + 1)
    f2 = -(x1/2 + 2*x2 - 1)
    return np.array([f1, f2])

def SP1(x):
    """SP1 test problem"""
    x1 = x[0]
    x2 = x[1]
    f1 = (x1 - 1)**2 + (x1 - x2)**2
    f2 = (x2 - 3)**2 + (x1 - x2)**2
    return np.array([f1, f2])

def Sch1(x):
    """Schaffer test problem"""
    x1 = x[0]
    f1 = x1**2
    f2 = (x1-2)**2
    return np.array([f1, f2])

# Test functions can be called like this:
# x = np.array([0.5, 0.5])  # Example input vector
# result = WFG1(x, 2)  # Get objective function values for 2 objectives
