import numpy as np
import matplotlib.pyplot as plt
import roughpy as rp

import pandas as pd
from TS_AIDA import Subsampling as ss
from TS_AIDA import AIDA as AI

from TS_AIDA import Random_subsampling as ss2
import seaborn as sns
sns.set_theme()
file_name = "period_data_1754391695"

sig_local_score = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/dim_noise_sig_local.npy")
dis_local_score = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/dim_noise_dis_local.npy")

sig_global_score = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/dim_noise_sig_global.npy")
dis_global_score = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/" + file_name + "/dim_noise_dis_global.npy")

tracker_file_name = "/Data/Synth_data/period_data_1754391695/tracker.txt"
tracker_file = open(tracker_file_name)
start = int(tracker_file.read())
tracker_file.close()

range_start = 800
range_end = 1200

def dis(data_slice, p=2):

    output = np.zeros(shape=10)
    for j in range(1,data_slice.shape[1]):
        A = data_slice[:,j]
        B = data_slice[:,j-1]
        dif = (np.abs(A - B)) ** p
        output[j] = np.sum(dif)

    return output

def distance_dim_noise(data, range_start, range_end, start):
    output = np.zeros(shape=(start+1, 10))
    for i in range(start+1):
        data_slice = data[i, range_start:range_end, :]

        output[i,:] = dis(data_slice)
        output[i, :] = output[i, :]#/output[i, :].max()
    return output

sig_local_dim_noise = np.average(distance_dim_noise(sig_local_score[:,:,:,0], range_start, range_end, start), axis=0)
sig_local_corr_dim_noise = np.average(distance_dim_noise(sig_local_score[:,:,:,2], range_start, range_end, start), axis=0)
sig_stand_dim_noise = np.average(distance_dim_noise(sig_local_score[:,:,:,3], range_start, range_end, start), axis=0)

dis_local_dim_noise = np.average(distance_dim_noise(dis_local_score[:,:,:,0], range_start, range_end, start), axis=0)
dis_local_corr_dim_noise = np.average(distance_dim_noise(dis_local_score[:,:,:,2], range_start, range_end, start), axis=0)
dis_stand_dim_noise = np.average(distance_dim_noise(dis_local_score[:,:,:,3], range_start, range_end, start), axis=0)

sig_local_dim_noise_gl = np.average(distance_dim_noise(sig_global_score[:,:,:,0], range_start, range_end, start), axis=0)
sig_local_corr_dim_noise_gl = np.average(distance_dim_noise(sig_global_score[:,:,:,2], range_start, range_end, start), axis=0)
sig_stand_dim_noise_gl = np.average(distance_dim_noise(sig_global_score[:,:,:,3], range_start, range_end, start), axis=0)

dis_local_dim_noise_gl = np.average(distance_dim_noise(dis_global_score[:,:,:,0], range_start, range_end, start), axis=0)
dis_local_corr_dim_noise_gl = np.average(distance_dim_noise(dis_global_score[:,:,:,2], range_start, range_end, start), axis=0)
dis_stand_dim_noise_gl = np.average(distance_dim_noise(dis_global_score[:,:,:,3], range_start, range_end, start), axis=0)

plt.plot(sig_local_corr_dim_noise, label="Signature local")
#plt.plot(sig_local_corr_dim_noise_gl, label="Signature global")

plt.plot(sig_local_dim_noise, label="Signature dim local")
#plt.plot(sig_local_dim_noise_gl, label="Signature dim global")

plt.plot(sig_stand_dim_noise, label="Signature standard local")
#plt.plot(sig_stand_dim_noise_gl, label="Signature standard global")


plt.xlabel("Dimension of anomaly occurence")
plt.ylabel("Distance to base case (normalized)")
plt.title("Dimensional noise effect on signature")
#plt.ylim(0,1)
plt.legend()
plt.show()

plt.plot(dis_local_corr_dim_noise, label="Distance local")
plt.plot(dis_local_corr_dim_noise_gl, label="Distance global")
plt.xlabel("Dimension of anomaly occurence")
plt.ylabel("Distance to base case (normalized)")
plt.title("Dimensional noise effect on signature")
#plt.ylim(0,1)
plt.legend()
plt.show()



plt.plot(sig_local_dim_noise, label="Local search")
plt.plot(sig_local_corr_dim_noise, label="Local search corrected")
plt.plot(sig_stand_dim_noise, label="Standard search")
plt.xlabel("Dimension of anomaly occurence")
plt.ylabel("Distance to base case (normalized)")
plt.title("Dimensional noise effect on signature")
#plt.ylim(0,1)
plt.legend()
plt.show()