import pathlib
import time

import numpy as np

import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from Data.Processes import Hawkes as Ha
from Data.Processes import Wiener as Wiener
from Data.Processes import Heston as He
sns.set_theme()

T = 3600
dim = 10
n_steps = 10000
K=4
n_experiments=100

rate = 0.0008*np.ones(n_steps)
decay_rate = 0.3*np.ones(n_steps)
excite_rate =0.7*np.ones(n_steps)

sin_out = np.zeros(shape=[n_experiments, n_steps, dim, 2])
anom_out = np.zeros(shape=[n_experiments, n_steps])
start_anom = int(n_steps / 2)
end_anom = int(start_anom + n_steps / 50)

rng = np.random.default_rng()

for i in range(n_experiments):
    print(i)
    x1 = np.linspace(0, T, n_steps)
    x2 = x1[:,np.newaxis]
    x3 = np.repeat(x2, dim, axis=1)

    trans_para = 20
    trans = rng.uniform(0,trans_para, size=dim)
    trans = trans[np.newaxis, :]
    trans = np.repeat(trans, n_steps, axis=0)



    indices, intensity, _ = Ha.Hawkes(n_steps, rate, excite_rate, decay_rate, T, dim)
    intensity = intensity[:,np.newaxis]/100
    #plt.plot(intensity)

    #plt.show()
    intensity = np.repeat(intensity, dim, axis=1)

    c_range = [[0, start_anom], [start_anom, end_anom], [end_anom, -1]]
    #c_range = [[0, 300], [300, 600], [600, -1]]
    rho = np.zeros(shape=(n_steps, dim))
    rhos = [-0.9, 1, -0.9]
    for y in range(len(c_range)):
        ranges = c_range[y]
        rho[ranges[0]:ranges[1], :] = rhos[y]


    noise_scale = 0.05
    cauchy_scale = 0.01

    theta = 0.01 * np.ones([n_steps, dim])
    sigma = 0.1 * np.ones([n_steps, dim])
    mu = 0.01*np.ones([n_steps, dim])
    W = Wiener.Correlated_Wiener(rho, T, n_steps, dim, c_range=c_range)
    W2 = Wiener.Wiener(T, n_steps, dim)
    #plt.plot(W)
    #plt.show()
    A = He.CIR(theta, sigma, mu, T, dim, n_steps, 0.01, W)#rng.normal(loc=0.0, scale=noise_scale, size=x3.shape)
    noise1 = W#rng.normal(loc=0.0, scale=A, size=x3.shape)
    noise3 = stats.cauchy.rvs(loc=0.0, scale=intensity, size=x3.shape)

    #noise1[:,:] = 0

    noise = noise1 #+ noise2



    anom = np.zeros(shape=n_steps)
    anom_data = np.zeros(shape=(n_steps, dim))
    anom[start_anom:end_anom] = 1



    trans[:,:] = 0

        #x3[start_anom:end_anom, :5] = x3[start_anom, :5]
    sin_normal = W2 + trans
    sin_anom = W + trans # noise3

    #sin_anom[start_anom:end_anom, :] = noise2[start_anom:end_anom, :]

    sin_out[i,:,:,0] = sin_normal
    sin_out[i,:,:,1] = sin_anom
    anom_out[i,:] = anom

    epoch_time = int(time.time())

    plt.plot(sin_anom)
    plt.show()


new_dir = pathlib.Path("Noise_data_" + str(epoch_time))
new_dir.mkdir(parents=True, exist_ok=True)
new_file = new_dir / 'Parameters.txt'
new_file.write_text('transpose = ' + str(trans_para) +
                        '\nnoise_scale = ' + str(noise_scale)+
                        '\ncauchy_scale = ' + str(cauchy_scale)+
                        '\nRho='+ str(rhos))


np.save("../Noise/Noise_data_" + str(epoch_time) + "/data", sin_out)
np.save("../Noise/Noise_data_" + str(epoch_time) + "/anom", anom_out)