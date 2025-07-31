import pathlib
import time

import numpy as np

import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns



sns.set_theme()

T = 1000
dim = 10
steps = 10000
K=4
n_experiments=1000

sin_out = np.zeros(shape=[n_experiments, steps, dim, dim])
anom_out = np.zeros(shape=[n_experiments, steps])

rng = np.random.default_rng(42)

for i in range(n_experiments):
    x1 = np.linspace(0, T, steps)
    x2 = x1[:,np.newaxis]
    x3 = np.repeat(x2, dim, axis=1)

    trans_para = 20
    trans = rng.uniform(0,trans_para, size=dim)
    trans = trans[np.newaxis, :]
    trans = np.repeat(trans, steps, axis=0)



    noise_scale = 0.5
    cauchy_scale = 0.01
    noise1 = rng.normal(loc=0.0, scale=noise_scale, size=x3.shape)
    #noise1[:,:] = 0
    noise2 = stats.cauchy.rvs(loc=0.0, scale=cauchy_scale, size=x3.shape)
    noise2[noise2 > 10] = 10
    noise2[noise2 < -10] = 10
    noise = noise1 + noise2


    start_anom = int(steps/2)
    end_anom = int(start_anom+steps/50)
    anom = np.zeros(shape=steps)
    anom[start_anom:end_anom] = 1

    #period[start_anom:end_anom, :5] = period[start_anom:end_anom, :5]/5
    for d in range(0,dim):
        #noise1[start_anom:end_anom, :d] = noise1[start_anom:end_anom, :d]/5
        noise2[start_anom:end_anom, :d] = noise1[start_anom:end_anom, :d]
        trans[:,:]=0

        #x3[start_anom:end_anom, :5] = x3[start_anom, :5]
        sin = noise2 + trans

        sin_out[i,:,:,d] = sin
        anom_out[i,:] = anom

        epoch_time = int(time.time())

        #plt.plot(sin)
        #plt.show()
new_dir = pathlib.Path('Synth_data', "Noise_data_" + str(epoch_time))
new_dir.mkdir(parents=True, exist_ok=True)
new_file = new_dir / 'Parameters.txt'
new_file.write_text('transpose = ' + str(trans_para) +
                        '\nnoise_scale = ' + str(noise_scale)+
                        '\ncauchy_scale = ' + str(cauchy_scale)
                    )

np.save("Synth_data/Noise_data_" + str(epoch_time) + "/jump", sin_out)
np.save("Synth_data/Noise_data_" + str(epoch_time) + "/anom", anom_out)