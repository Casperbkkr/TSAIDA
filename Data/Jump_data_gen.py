import pathlib


import matplotlib.pyplot as plt
import numpy as np
import roughpy as rp

import time

from Data.Processes import Wiener as wien, Heston as hes, Events as ev
from Data.Processes import GBM as GBM
K=3
n_experiments = 100
n_samples = 10
n_steps = 1000
T = 1
rate = 4
dim_min_max = (1,3)
rebound = False
rebound_rate = 1
rng =  np.random.default_rng()
the = 0.01
thet = the*np.ones([n_steps, n_samples])
sigm = 0.01
Sigma = sigm*np.ones([n_steps, n_samples])
#Sigma = rng.uniform(0.01, 0.015, size=[n_experiments, n_samples])
muuu = 0.1
mu = muuu*np.ones([n_steps, n_samples])
Mu = rng.uniform(0.01, 0.02, size=[n_experiments, n_samples])

d=10

s0 = rng.uniform(1, 2, size=[n_experiments, n_samples])


c_range = [[0,300],[300,600],[600,n_steps]]
rho = np.ones([n_steps, n_samples])
rhos=[0.1, -0.1, 0.1]
for y in range(len(c_range)):
    ranges = c_range[y]
    rho[ranges[0]:ranges[1], :] = rhos[y]
dW = wien.Correlated_Wiener(rho, T, n_steps, n_samples, c_range=c_range)


data_set_standard = np.zeros(shape=[n_experiments, n_steps, n_samples])
data_set_jump = np.zeros(shape=[n_experiments, n_steps, n_samples])
anom_indices = np.zeros(shape=[n_experiments, n_steps])

epoch_time = int(time.time())
new_dir = pathlib.Path("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/Jump_gbm/", "jump_data_" + str(epoch_time))
new_dir.mkdir(parents=True, exist_ok=True)
new_file = new_dir / 'Parameters.txt'
new_file.write_text('Theta = ' + str(the) + '\nsigma = ' + str(sigm) +
                        '\nmu = ' + str(muuu) +
                        '\nS0 = ' + str(s0) +
                        '\nrate = ' + str(rate) +
                        '\nrebound_rate = ' + str(rebound_rate))

for i in range(n_experiments):
    S0 = s0[i,:]
    #mu = Mu[i,:]
    sigma = Sigma[i,:]
    print(i)
    dW = wien.Correlated_Wiener(rho, T, n_steps, n_samples, c_range)

    path = GBM.GBM(S0, mu, sigma, T, n_samples, n_steps, dW)[0]
    data_set_standard[i,:,:] = path
    data_set_jump[i, :, :] = path


    #event_info = ev.Event_location(n_samples, n_steps, T, rate, (1, 3), rebound=True, rebound_rate=rebound_rate)

   # A = np.zeros(shape=[n_steps])
    #for j in range(event_info[0].shape[0]):
     #   A[event_info[0][j]:event_info[1][j]] = 1

    #anom_indices[i, :] = A
    path_j = path#[:, 0][:,np.newaxis]
    pos_neg = rng.choice([1,0.9/1.2], p=[0.5, 0.5], size=d)
    path_jump = pos_neg*1.2*path[int(n_steps/2):int(n_steps/2+20), :d]
    data_set_jump[i,int(n_steps/2):int(n_steps/2+20),:d] = path_jump

    plt.plot(data_set_jump[i,:,:])
    plt.show()

total_out = np.zeros(shape=[n_experiments, n_steps, n_samples,2])
total_out[:,:,:,0] =data_set_standard
total_out[:,:,:,1] = data_set_jump

np.save("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/Jump_gbm/jump_data_" + str(epoch_time) + "/data", total_out)
np.save("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/Jump_gbm/jump_data_" + str(epoch_time) + "/anom", anom_indices)



