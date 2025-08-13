import numpy as np
import matplotlib.pyplot as plt

def Wiener(T, n_steps, n_samples):
    dt = float(T) / n_steps
    rng = np.random.default_rng()
    rand = rng.normal(loc=0, scale=1, size=[n_steps, n_samples])
    return 1/np.sqrt(dt) * rand


def Correlated_Wiener(rho, T, n_steps, n_samples, c_range=[[0,1]]) -> np.ndarray:
    """
    Sample correlated discrete Brownian increments to given increments dW.
    """
    if c_range == [0,1]:
        c_range = [[0, n_steps]]
    rng = np.random.default_rng(42)

    dW2 = Wiener(T, n_steps, n_samples)
    dW = Wiener(T, n_steps, n_samples)

    #dW_uncorr = dW.copy()
    for ranges in c_range:
        for i in range(0 , n_samples-1):


            dW[ranges[0]:ranges[1],i+1] = rho[ranges[0]:ranges[1], i] * dW[ranges[0]:ranges[1],i] + np.sqrt(1 - rho[ranges[0]:ranges[1],i] ** 2) * dW2[ranges[0]:ranges[1],i]

    return dW#, dW_uncorr