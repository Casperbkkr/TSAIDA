from cProfile import label
import pathlib


import matplotlib.pyplot as plt
import numpy as np
import roughpy as rp
from Processes import Events as ev
from Processes import Heston as hes
from Codes import Subsampling as ss
from Codes import Path_signature as ps
import time

from Processes import Wiener as wien

K=3

n_samples = 10
n_steps = 10000
T = 1
rate = 2
dim_min_max = (1,1)
rebound = False
rebound_rate = 1

the = 0.01
thet = the*np.ones([n_steps, n_samples])
sigm = 0.001
sigma = sigm*np.ones([n_steps, n_samples])
muuu = 0.03
mu = muuu*np.ones([n_steps, n_samples])
s0 = 0.03
S0 = s0*np.ones([1, n_samples])

c_range = [0,10000]
rho= np.ones([n_steps, n_samples])
dW = wien.Correlated_Wiener(rho, T, n_steps, n_samples, c_range=c_range)

n_experiments = 100
data_set_standard = np.zeros(shape=[n_experiments, n_steps, n_samples])
data_set_jump = np.zeros(shape=[n_experiments, n_steps, n_samples])
anom_indices = np.zeros(shape=[n_experiments, n_steps])

epoch_time = int(time.time())
new_dir = pathlib.Path('Synth_data', "jump_data_" + str(epoch_time))
new_dir.mkdir(parents=True, exist_ok=True)
new_file = new_dir / 'Parameters.txt'
new_file.write_text('Theta = ' + str(the) + '\nsigma = ' + str(sigm) +
                        '\nmu = ' + str(muuu) +
                        '\nS0 = ' + str(s0) +
                        '\nrate = ' + str(rate) +
                        '\nrebound_rate = ' + str(rebound_rate))

for i in range(n_experiments):
    print(i)
    dW = wien.Correlated_Wiener(rho, T, n_steps, n_samples, c_range)
    path = hes.CIR(thet, sigma, mu, T, n_samples, n_steps, S0, dW)
    data_set_standard[i,:,:] = path
    data_set_jump[i, :, :] = path

    event_info = ev.Event_location(n_samples, n_steps, T, rate, (1, 1), rebound=True, rebound_rate=rebound_rate)

    A = np.zeros(shape=[n_steps])
    for j in range(event_info[0].shape[0]):
        A[event_info[0][j]:event_info[1][j]] = 1

    anom_indices[i, :] = A
    path_j = path[:, 0][:,np.newaxis]
    path_jump = ev.Jump_rebound(path_j, event_info, 0, 0.0001)[:,0]
    data_set_jump[i,:,0] = path_jump



np.save("Synth_data/jump_data_" + str(epoch_time) + "/standard", data_set_standard)
np.save("Synth_data/jump_data_" + str(epoch_time) + "/jump", data_set_jump)
np.save("Synth_data/jump_data_" + str(epoch_time) + "/anom", anom_indices)



GAD = np.load("Synth_data/jump_data_"+ str(epoch_time) + "/jump.npy")

plt.plot(GAD[0,:,:])
plt.show()
R=1