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



K=3

n_samples = 10
n_steps = 10000
T = 1
rate = 2
dim_min_max = (1,1)
rebound = False
rebound_rate = 1

the = 0.06
thet = the*np.ones([n_steps, n_samples])
thet[5000:5200, 0] = 0.01
sigm = 0.06
sigma = sigm*np.ones([n_steps, n_samples])
sigma[5000:5200, 0] = 0.01
muuu = 0.03
mu = muuu*np.ones([n_steps, n_samples])
s0 = 4
S0 = s0*np.ones([1, n_samples])

n_experiments = 100
data_set_standard = np.zeros(shape=[n_experiments, n_steps, n_samples])
anom_indices = np.zeros(shape=[n_experiments, n_steps])

epoch_time = int(time.time())
new_dir = pathlib.Path('Synth_data', "var_data_" + str(epoch_time))
new_dir.mkdir(parents=True, exist_ok=True)
new_file = new_dir / 'Parameters.txt'
new_file.write_text('Theta = ' + str(the) + '\nsigma = ' + str(sigm) +
                        '\nmu = ' + str(muuu) +
                        '\nS0 = ' + str(s0) +
                        '\nrate = ' + str(rate) +
                        '\nrebound_rate = ' + str(rebound_rate))

for i in range(n_experiments):
    print(i)
    path = hes.CIR(thet, sigma, mu, T, n_samples, n_steps, S0)
    data_set_standard[i,:,:] = path

    np.save("Synth_data/var_data_" + str(epoch_time) + "/var", data_set_standard)

    np.save("Synth_data/var_data_" + str(epoch_time) + "/anom", anom_indices)


