import numpy as np
import matplotlib.pyplot as plt
def Wiener(T, n_steps, n_samples):
    dt = float(T) / n_steps
    rng = np.random.default_rng()
    rand = rng.normal(loc=0, scale=1, size=[n_steps, n_samples])
    return np.sqrt(dt) * rand


def Correlated_Wiener(rho, T, n_steps, n_samples, c_range=[0,1]) -> np.ndarray:
    """
    Sample correlated discrete Brownian increments to given increments dW.
    """
    if c_range == [0,1]:
        c_range = [0, n_steps]
    rng = np.random.default_rng()
    dW = Wiener(T, n_steps, n_samples)
    dW2 =  Wiener(T, n_steps, n_samples)
    #dW_uncorr = dW.copy()
    for i in range(1, n_samples):
        k = rng.choice([j for j in range(i)], size=1)[0]
        dW[c_range[0]:c_range[1],i] = rho[c_range[0]:c_range[1], k] * dW[c_range[0]:c_range[1],i] + np.sqrt(1 - rho[c_range[0]:c_range[1],k] ** 2) * dW2[c_range[0]:c_range[1],k]

    return dW#, dW_uncorr