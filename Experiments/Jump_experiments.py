from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()
import roughpy as rp
from Processes import Events as ev
from Processes import Heston as hes
from Codes import Subsampling as ss
from Codes import Path_signature as ps
import gc
K=3

n_samples = 10
n_steps = 10000
T = 1
"""
dis_data1 = np.load("Experiments/Synth_data/jump_data_1747138536/dis_perf_N100_100.npy")
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
jump_data = np.load("Data_gen/Synth_data/jump_data_1747138536/jump.npy")

anom_data = np.load("Data_gen/Synth_data/jump_data_1747138536/anom.npy")

f = lambda X: X

data_sig = np.zeros(shape=[5, jump_data.shape[0],jump_data.shape[1]])
data_dis = np.zeros(shape=[5, jump_data.shape[0],jump_data.shape[1]])

base_dis = np.zeros(shape=[5, jump_data.shape[1]])
base_sig = np.zeros(shape=[5, jump_data.shape[1]])
distance_sig = []
distance_dis = []
for experiment in range(5):
    print(experiment)
    jump_sample = jump_data[experiment, :, :4]
    # jump_sample = jump_sample[:, np.newaxis]
    anom_sample = anom_data[experiment, :]
    sample_info1 = ss.sub_sample(jump_sample, N=1000, n_samples_min_max=[50, 101], length_min_max=[50, 301],
                                 dim_min_max=(1, 3))
    #sample_info2 = (sample_info1[0], sample_info1[1], sample_info1[2], [[0, 1, 2, 3] for re in range(1000)])
    base_sig[experiment,:]= ss.Score_aggregator(jump_sample, sample_info1, K, T)[0]
    base_dis[experiment,:] = ss.sub_sampler(jump_sample, sample_info1, f)[0]
    x_a=[]
    for n in [50,100, 150, 200, 300, 400, 600]:
        x_a.append(n)
        dis_sig = []
        dis_dis = []
        for j in range(5):
            print(experiment,n,j)
            sample_info1 = ss.sub_sample(jump_sample, N=n, n_samples_min_max=[50, 101], length_min_max=[50, 301],
                                         dim_min_max=(1, 3))
            sample_info2 = (sample_info1[0], sample_info1[1], sample_info1[2], [[0, 1, 2, 3] for re in range(1000)])
            data_sig[j, experiment, :] = ss.Score_aggregator(jump_sample, sample_info1, K, T)[0]
            data_dis[j, experiment, :] = ss.sub_sampler(jump_sample, sample_info2, f)[0]

            dif = (np.abs(base_sig[experiment,:] -  data_sig[j, experiment, :])) ** 2
            S = (np.sum(dif))**(0.5)
            dis_sig.append(S)

            dif = (np.abs(base_dis[experiment, :] - data_dis[j, experiment, :])) ** 2
            S = (np.sum(dif)) ** (0.5)
            dis_dis.append(S)
        distance_sig.append(sum(dis_sig)/5)
        distance_dis.append(sum(dis_dis)/5)

        print(distance_sig)
        print(distance_dis)


        plt.plot(x_a, distance_dis, label="Distance")
        plt.plot(x_a, distance_sig, label="Signature")
        plt.legend()
        plt.xlabel("# subsamples")
        plt.ylabel("Distance to true")
        plt.show()

data_sig[experiment, :] = output_sig
data_dis[experiment, :] = output_dis
plt.plot(jump_sample)
plt.xlabel("Time")
plt.show()
plt.plot(data_dis[experiment, :], label="Distance")
plt.plot(data_sig[experiment, :], label="Signature")
plt.ylabel("Outlier score")
plt.xlabel("Time")
plt.legend()
plt.show()
#if experiment % 20 == 0:
 #   np.save("Synth_data/jump_data_1747138536" + "/data_sig_100_50300_d3_N100", data_sig)
  #  np.save("Synth_data/jump_data_1747138536" + "/data_dis_100_50300_d3_N100", data_dis)
gc.collect()

data_sig = np.zeros(shape=[jump_data.shape[0],jump_data.shape[1]])
data_dis = np.zeros(shape=[jump_data.shape[0],jump_data.shape[1]])

for experiment in range(jump_data.shape[0]):
    print(experiment)
    jump_sample = jump_data[experiment, :, 0]
    jump_sample = jump_sample[:, np.newaxis]
    anom_sample = anom_data[experiment, :]
    sample_info1 = ss.sub_sample(jump_sample, N=1000, n_samples_min_max=[100,101], length_min_max=[100, 201], dim_min_max=(1,2))
    sample_info2 = (sample_info1[0], sample_info1[1], sample_info1[2], [[0] for re in range(1000)])
    output_sig = ss.Score_aggregator(jump_sample, sample_info2, K, T)[0]
    output_dis = ss.sub_sampler(jump_sample, sample_info2, f)[0]
    data_sig[experiment, :] = output_sig
    data_dis[experiment, :] = output_dis
    plt.plot(output_dis, label="Distance")
    plt.plot(data_dis, label="Signature")
    plt.plot(output_sig)
    plt.ylabel("Outlier score")
    plt.xlabel("Time")
    plt.legend()
    plt.show()
    if experiment % 20 == 0:
        np.save("Experiments/Synth_data/jump_data_1747138536" + "/data_sig_100_200_d1_N1000", data_sig)
        np.save("Experiments/Synth_data/jump_data_1747138536" + "/data_dis_100_200_d1_N1000", data_dis)

