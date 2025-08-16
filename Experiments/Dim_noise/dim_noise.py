import numpy as np
import matplotlib.pyplot as plt
import roughpy as rp

import pandas as pd
from TS_AIDA import Subsampling as ss
from TS_AIDA import AIDA as AI

from TS_AIDA import Random_subsampling as ss2
import seaborn as sns

file_name = "period_data_1754391695"

data = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/data.npy")
anom = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/anom.npy")


sig_local_score = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/dim_noise_sig_local.npy")
dis_local_score = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/dim_noise_dis_local.npy")

sig_global_score = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/dim_noise_sig_global.npy")
dis_global_score = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/dim_noise_dis_global.npy")



T=100
K=4


N = 50
min_w = 30
max_w = 500
parameters = (min_w, max_w, 2, 10, 20, 50,  5)

tracker_file_name = "/Data/Synth_data/Dim_corr/period_data_1754391695/tracker.txt"
tracker_file = open(tracker_file_name)
start = int(tracker_file.read()) + 1
tracker_file.close()

for i in range(start, data.shape[0]):

    example_sample = data[0, :, :, 0]

    sample_info_local = ss2.Random_subsampler(example_sample, N=N, parameters=parameters, local=True, window_corrected=False,
                                              dim_corrected=False)

    sample_info_local_corr = ss2.Random_subsampler(example_sample, N=N, parameters=parameters, local=True, window_corrected=True,
                                             dim_corrected=False)

    #sample_info_corr = ss2.Random_subsampler(example_sample, N=N, parameters=parameters, local=False, window_corrected=True, dim_corrected=False)
#
    sample_info_stand = ss2.Random_subsampler(example_sample, N=N, parameters=parameters, local=False, window_corrected=False,
                                        dim_corrected=False)

    for d in range(10):
        print(d)
        sample = data[i, :, :, d]
        A, _, B = ss.Score_aggregator(sample, sample_info_local, K, T, sig=True)
        sig_local_score[i, :, d, 0], sig_global_score[i, :, d, 0]  = A[4000:6000], B[4000:6000]
        A, _, B = ss.Score_aggregator(sample, sample_info_local, K, T, sig=False)
        dis_local_score[i, :, d, 0], dis_global_score[i, :, d, 0]  = A[4000:6000], B[4000:6000]
        """
        A, _, B = ss.Score_aggregator(sample, sample_info_corr, K, T, sig=True)
        sig_local_score[i, :, d, 1], sig_global_score[i, :, d, 1]  = A[4000:6000], B[4000:6000]
        A, _, B = ss.Score_aggregator(sample, sample_info_corr, K, T, sig=False)
        dis_local_score[i, :, d, 1], dis_global_score[i, :, d, 1]  = A[4000:6000], B[4000:6000]
        """
        A, _, B = ss.Score_aggregator(sample, sample_info_local_corr, K, T, sig=True)
        sig_local_score[i, :, d, 2], sig_global_score[i, :, d, 2] = A[4000:6000], B[4000:6000]
        A, _, B = ss.Score_aggregator(sample, sample_info_local_corr, K, T, sig=False)
        dis_local_score[i, :, d, 2], dis_global_score[i, :, d, 2]  = A[4000:6000], B[4000:6000]

        A, _, B = ss.Score_aggregator(sample, sample_info_stand, K, T, sig=True)
        sig_local_score[i, :, d, 3], sig_global_score[i, :, d, 3]  = A[4000:6000], B[4000:6000]
        A, _, B = ss.Score_aggregator(sample, sample_info_stand, K, T, sig=False)
        dis_local_score[i, :, d, 3], dis_global_score[i, :, d, 3]  = A[4000:6000], B[4000:6000]


    np.save(
        "/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/dim_noise_sig_local.npy",
        sig_local_score)
    np.save(
        "/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/dim_noise_dis_local.npy",
        dis_local_score)
    np.save(
        "/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/dim_noise_sig_global.npy",
        sig_global_score)
    np.save(
        "/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/dim_noise_dis_global.npy",
        dis_global_score)

    with open(tracker_file_name, "w") as tracker_file:
        tracker_file.write(str(i))

    print("Completed the "+ str(i)+"-th experiment")


