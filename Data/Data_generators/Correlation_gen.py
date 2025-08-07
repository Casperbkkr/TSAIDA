import pathlib
import time
import numpy as np

from Data.Processes import Wiener as wien, Heston as hes

n_experiments = 100
n_samples = 20
n_steps = 10000
T = 3
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


data_set_standard = np.zeros(shape=[n_experiments, n_steps, n_samples])
data_set_jump = np.zeros(shape=[n_experiments, n_steps, n_samples])
anom_indices = np.zeros(shape=[n_experiments, n_steps])

rng = np.random.default_rng()
#rho = rng.uniform(low=-1, high=1, size=[n_steps, n_samples])
c_range = [1000,1500]
rho= np.ones([n_steps, n_samples])
rho[1000:1500,:] = 1


epoch_time = int(time.time())
new_dir = pathlib.Path('../Synth_data', "corr_data_" + str(epoch_time))
new_dir.mkdir(parents=True, exist_ok=True)
new_file = new_dir / 'Parameters.txt'
new_file.write_text('Theta = ' + str(the) + '\nsigma = ' + str(sigm) +
                        '\nmu = ' + str(muuu) +
                        '\nS0 = ' + str(s0) +
                        '\nrate = ' + str(rate) +
                        '\nrebound_rate = ' + str(rebound_rate))

data_out = np.zeros(shape=[n_experiments, n_steps, n_samples])

for experiment in range(n_experiments):
    print(experiment)
    dW = wien.Correlated_Wiener(rho, T, n_steps, n_samples, c_range=c_range)
    data_out[experiment,:,:] = hes.CIR(thet, sigma, mu, T, n_samples, n_steps, S0, dW)
    np.save("../Synth_data/corr_data_" + str(epoch_time)+"/cor_data", data_out)



