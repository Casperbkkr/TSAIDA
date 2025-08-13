from Data.Processes import Hawkes as Ha
from Data.Processes import Heston as He
from Data.Processes import Wiener as Wi
from Data.Processes import GBM as gbm
import matplotlib.pyplot as plt
import numpy as np
n_steps = 1000
T = 1
rate = 0.02*np.ones(n_steps)
decay_rate = 0.04*np.ones(n_steps)
excite_rate =0.6*np.ones(n_steps)
dim_min_max = [2,5]
dt = float(T) / n_steps

A, B, C = Ha.Hawkes(n_steps, rate, excite_rate , decay_rate, T, dim_min_max)

plt.plot(B)
plt.show()

plt.plot(A)
plt.show()
n_samples = 5

c_range=[[0,300], [300,600], [600,n_samples]]
rho = np.zeros(shape=(n_steps, n_samples))
rhos = [-0.9, 0.9, -0.9]

for y in range(len(c_range)):
    ranges = c_range[y]
    rho[ranges[0]:ranges[1],:] = rhos[y]

theta = 0.1*np.ones([n_steps, n_samples])
sigma = 0.2*np.ones([n_steps, n_samples])
mu = np.ones([n_steps, n_samples])
W = Wi.Correlated_Wiener(rho, T, n_steps,n_samples, c_range=c_range)
plt.plot(W)
plt.show()

#C = gbm.GBM(1, mu, sigma, T, n_samples, n_steps, W)[0]
C = He.CIR(theta, sigma, mu, T, n_samples, n_steps, 0.01, W)

plt.plot(C)
plt.show()
