import pathlib
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import seaborn as sns



sns.set_theme()
T = 1000
dim = 10
steps = 1000
K=4
n_experiments=100

sin_out = np.zeros(shape=[n_experiments, steps, dim, 2])
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

    period_low = 3
    period_high = 6*np.pi
    period = rng.uniform(period_low, period_high, size=dim)
    period = period[np.newaxis, :]
    period = np.repeat(period, steps, axis=0)

    shift_para = np.pi
    shift = rng.uniform(0, shift_para, size=dim)
    shift = shift[np.newaxis, :]
    shift = np.repeat(shift, steps, axis=0)

    amp_low = 1
    amp_high = 10
    amp = rng.uniform(amp_low, amp_high, size=dim)
    amp = amp[np.newaxis, :]
    amp = np.repeat(amp, steps, axis=0)

    noise_scale = 0.5
    cauchy_scale = 0#.01
    noise1 = rng.normal(loc=0.0, scale=noise_scale, size=x3.shape)
    noise2 = stats.cauchy.rvs(loc=0.0, scale=cauchy_scale, size=x3.shape)
    noise2[noise2 > 10] = 10
    noise2[noise2 < -10] = 10
    noise = noise1 + noise2


    start_anom = int(steps/2)
    end_anom = int(start_anom+steps/50)
    anom = np.zeros(shape=steps)
    anom[start_anom:end_anom] = 1



    sin = amp * np.sin(x3 / period + shift) + trans + noise
    sin_out[i, :, :, 0] = sin
    anom_out[i, :] = anom

    amp[start_anom:end_anom, :] = amp[start_anom:end_anom, :] / 5

    sin = amp * np.sin(x3 / period + shift) + trans + noise
    sin_out[i, :, :, 1] = sin

    #period[start_anom:end_anom, :5] = period[start_anom:end_anom, :5]/5
    #plt.plot(sin_out[i, :, :, 1])
    #plt.show()

epoch_time = int(time.time())


new_dir = pathlib.Path('/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/Period/', "period_data_" + str(epoch_time))
new_dir.mkdir(parents=True, exist_ok=True)
new_file = new_dir / 'Parameters.txt'
new_file.write_text('transpose = ' + str(trans_para) +
                    '\nperiod low = ' + str(period_low) +
                        '\nperiod_high = ' + str(period_high) +
                        '\nshift = ' + str(shift_para) +
                        '\namplitude low = ' + str(amp_low) +
                        '\namplitude high = ' + str(amp_high) +
                        '\nnoise_scale = ' + str(noise_scale)+
                        '\ncauchy_scale = ' + str(cauchy_scale)
                    )

np.save("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/Period/period_data_" + str(epoch_time) + "/data", sin_out)
np.save("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/Period/period_data_" + str(epoch_time) + "/anom", anom_out)


