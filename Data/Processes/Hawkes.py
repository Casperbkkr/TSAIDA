import numpy as np
from matplotlib import pyplot as plt

rng = np.random.default_rng()



n_steps = 10000
T = 3600
rate = 0.0008*np.ones(n_steps)
decay_rate = 0.3*np.ones(n_steps)
excite_rate =0.7*np.ones(n_steps)
dim_min_max = [2,5]
dt = float(T) / n_steps



def Hawkes(n_steps, rate, self_rate, decay, T, dim):
    dt = float(T) / n_steps
    rate_dt = rate*dt

    rng = np.random.default_rng()
    intensity =  np.zeros(n_steps)
    path = np.zeros(n_steps)
    indices = np.zeros(n_steps)
    event = np.array([])

    exitation_decay = decay*dt
    exitation_rate = self_rate*dt

    for i in range(0, n_steps):
        t = i*dt
        ts = t*np.ones_like(event)
        r = exitation_decay[i] *np.sum( np.exp(-1*exitation_rate[i]*(ts-event)))
        extra_intensity = np.maximum(0, r)
        total_intensity = rate_dt[i] + extra_intensity

        intensity[i] = total_intensity

        P0 = np.exp(-total_intensity)
        P1 = 1 - P0
        intensity[i] = P1
        IO = rng.choice([0,1], p=[P0,P1])
        UO = rng.normal(0,0.1, size=1)

        if IO == 1:
            event = np.append(event, t)
            indices[i] = indices[i-1]+UO

            path[i] = 1
        else:
            indices[i] = indices[i-1]

        event = event[event-i<24]

        a=1

    return indices, intensity, path


