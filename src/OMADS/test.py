import random
import numpy as np

from matplotlib import pyplot as plt



lb = [-10, -15]
ub = [10, 15]

center = [0, 5]
ns = 10
nt = 20
scale = 0.9
samples_gauss = np.zeros((ns, 2))
samples_gamma = np.zeros((ns, 2))
samples_poisson = np.zeros((ns, 2))
samples_binomial = np.zeros((ns, 2))
samples_exp = np.zeros((ns, 2))

np.random.seed(5)
samples_gauss[:, 0] = np.random.normal(loc=center[0], scale=scale, size=(ns,))
samples_gauss[:, 1] = np.random.normal(loc=center[1], scale=scale, size=(ns,))

sgamma = scale
delta = np.zeros_like(center)
for i in range(len(center)):
  if -1. <= center[i] < 0.:
    delta[i] = -5.
  elif 0. <= center[i] < 1.:
    delta[i] = 5.
  else:
    delta[i] = 0.


samples_gamma[:, 0] = np.random.gamma(shape=(center[0]+delta[0])/sgamma, scale=sgamma, size=(ns,))-delta[0]
samples_gamma[:, 1] = np.random.gamma(shape=(center[1]+delta[1])/sgamma, scale=sgamma, size=(ns,))-delta[1]

samples_poisson[:, 0] = (np.random.poisson(lam=center[0]+delta[0], size=(ns,))-delta[0])*scale
samples_poisson[:, 1] = (np.random.poisson(lam=center[1]+delta[1], size=(ns,))-delta[1])*scale

samples_binomial[:, 0] = (np.random.binomial(n=(center[0]+delta[0])/scale, p=scale, size=ns))-delta[0]
samples_binomial[:, 1] = (np.random.binomial(n=(center[1]+delta[1])/scale, p=scale, size=ns))-delta[1]

samples_exp[:, 0] = (np.random.exponential(scale=scale, size=ns))+center[0]
samples_exp[:, 1] = (np.random.exponential(scale=scale, size=ns))+center[1]


temp1, = plt.plot(samples_gauss[:,0], samples_gauss[:, 1], 'ok', markersize=2)
temp2, = plt.plot(samples_gamma[:,0], samples_gamma[:, 1], 'og', markersize=2)
temp3, = plt.plot(samples_poisson[:,0], samples_poisson[:, 1], 'or', markersize=2)
temp4, = plt.plot(samples_binomial[:,0], samples_binomial[:, 1], 'ob', markersize=2)
temp5, = plt.plot(samples_exp[:,0], samples_exp[:, 1], 'oy', markersize=2)


plt.xlim(-10, 10)
plt.ylim(-15,15)

plt.show()