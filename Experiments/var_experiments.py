import matplotlib.pyplot as plt
import numpy as np
import roughpy as rp
from Codes import Subsampling as ss
from Codes import Path_signature as ps
import gc
K=4

n_samples = 10
n_steps = 10000
T = 1

Jump_data1 = np.load("Experiments/Synth_data/jump_data_1747138536/dis_perf_N100_100.npy")
sig_data1 = np.load("Experiments/Synth_data/jump_data_1747138536/sig_perf_N100_100.npy")
dis_data2 = np.load("Experiments/Synth_data/jump_data_1747138536/dis_perf_N100.npy")
sig_data2 = np.load("Experiments/Synth_data/jump_data_1747138536/sig_perf_N100.npy")
plt.plot(dis_data1[4,:])
plt.plot(sig_data1[4,:])
plt.show()
plt.plot(dis_data2[4,:])
plt.plot(sig_data2[4,:])
plt.show()
"""
jump_data = np.load("Synth_data/var_data_1747732168/sig0005.npy")
"""
sigma = np.zeros([n_steps, n_samples])
sigma[5000:5500, :] = 1
anom_data = sigma


f = lambda X: X

data_sig1 = np.zeros(shape=[jump_data.shape[0],jump_data.shape[1]])
data_sig2 = np.zeros(shape=[jump_data.shape[0],jump_data.shape[1]])
data_sig3 = np.zeros(shape=[jump_data.shape[0],jump_data.shape[1]])
data_sig4 = np.zeros(shape=[jump_data.shape[0],jump_data.shape[1]])
data_dis = np.zeros(shape=[jump_data.shape[0],jump_data.shape[1]])

for experiment in range(100):
    print(experiment)
    jump_sample = jump_data[experiment, :, :4]
    #jump_sample = jump_sample[:, np.newaxis]
    anom_sample = anom_data[experiment, :]
    sample_info1 = ss.sub_sample(jump_sample, N=1000, n_samples_min_max=[20,50], length_min_max=[20, 101], dim_min_max=(1,3))
    sample_info2 = sample_info1#(sample_info1[0], sample_info1[1], sample_info1[2], [[0,1,2,3] for re in range(1000)])
    output_sig1 = ss.Score_aggregator(jump_sample, sample_info2, 1, T)[0]
    output_sig2 = ss.Score_aggregator(jump_sample, sample_info2, 2, T)[0]
    output_sig3 = ss.Score_aggregator(jump_sample, sample_info2, 3, T)[0]
    output_sig4 = ss.Score_aggregator(jump_sample, sample_info2, 4, T)[0]
    output_dis = ss.sub_sampler(jump_sample, sample_info2, f)[0]
    data_sig1[experiment, :] = output_sig1
    data_sig2[experiment, :] = output_sig2
    data_sig3[experiment, :] = output_sig3
    data_sig4[experiment, :] = output_sig4
    data_dis[experiment, :] = output_dis
    #if experiment % 20 == 0:
     #   np.save("Synth_data/var_data_1747732168" + "/standard_sig", data_sig)
      #  np.save("Synth_data/var_data_1747732168" + "/standard_dis", data_dis)
    gc.collect()
    plt.plot(jump_data[experiment, :, :7])
    plt.show()
    plt.plot(data_sig1[experiment, :], label="sig1")
    plt.plot(data_sig2[experiment, :], label="sig2")
    plt.plot(data_sig3[experiment, :], label="sig3")
    plt.plot(data_sig4[experiment, :], label="sig4")
    plt.plot(data_dis[experiment, :], label="dis")
    plt.legend()
    plt.show()

data_sig = np.zeros(shape=[jump_data.shape[0],jump_data.shape[1]])
data_dis = np.zeros(shape=[jump_data.shape[0],jump_data.shape[1]])
