import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()
import roughpy as rp
from TS_AIDA import Subsampling as ss
from TS_AIDA import Random_subsampling as ss2
from TS_AIDA import Path_signature as ps
from TS_AIDA import TSAIDA as TSA
from Data import Dim_mixer as dmm

file_name = "/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/dim_corr/jump_data_1755255199"
Data_np = np.load(file_name+"/data.npy")
Anom_np = np.load(file_name+"/anom.npy")

samp = 12
"""
t = Anom_np[samp,:]
start_anom = np.where(t == 1)[0][0]
end_anom = np.where(t == 1)[0][-1]
"""
Data_np = dmm.Mixer(Data_np[:,:,:,1], Data_np[:,:,:,0], 10)
#Data_np = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data/period_data_1754391695/data.npy")
Data_np = Data_np[1,:,:]
"""
len = end_anom-start_anom
"""
#A = TSA.Subsample_examiner(Data_np, start_anom,end_anom, [1,3], n_samples=20)

#B = TSA.Subsample_examiner(Data_np, 1000,1000+len, [1,3], n_samples=20)
Data_np = Data_np#(Data_np - np.mean(Data_np, axis=0)) / Data_np.std(axis=0)
plt.plot(Data_np)

"""
plt.plot(t*3)
"""
plt.title("Example")
plt.show()
print("Data loaded")


N = 100

parameters = (20, 50, 2, 5, 20, 30, 5)

sample_info_local = ss2.Random_subsampler(Data_np, N=N, parameters=parameters, local=True, window_corrected=True, dim_corrected=False)

sample_info_corr = ss2.Random_subsampler(Data_np, N=N, parameters=parameters, local=True, window_corrected=True, dim_corrected=True)

#sample_info_corr_dim = ss2.Random_subsampler(Data_np, N=N, parameters=parameters)

sample_info = ss2.Random_subsampler(Data_np, N=N, parameters=parameters, local=False, window_corrected=True, dim_corrected=True)



T = 1
K = 4

sig_local, _, sig_local_N = ss.Score_aggregator(Data_np, sample_info_local, K, T, sig=True, normalize=False)
dis_local, _, dis_local_N = ss.Score_aggregator(Data_np, sample_info_local, K, T, sig=False, normalize=True)

sig_corr, _, sig_corr_N = ss.Score_aggregator(Data_np, sample_info_corr, K, T, sig=True, normalize=False)
dis_corr, _, dis_corr_N = ss.Score_aggregator(Data_np, sample_info_corr, K, T, sig=False, normalize=True)

sig_stand, _, sig_stand_N = ss.Score_aggregator(Data_np, sample_info, K, T, sig=True, normalize=False)
dis_stand, _, dis_stand_N = ss.Score_aggregator(Data_np, sample_info, K, T, sig=False, normalize=True)

def moving_average(a, n=30):

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]

    return ret / n
av = moving_average(sig_local_N, n=30)
sig_local_N = sig_local_N - moving_average(sig_local_N, n=100)
plt.plot(sig_local)
plt.plot(sig_corr)
plt.plot(sig_stand)
plt.title("Signature")
plt.show()

plt.plot(dis_local)
plt.plot(dis_corr)
plt.plot(dis_stand)
plt.title("Distance")
plt.show()

plt.plot(sig_local_N - moving_average(sig_local_N, n=100))
plt.plot(sig_corr_N - moving_average(sig_corr_N, n=100))
plt.plot(sig_stand_N - moving_average(sig_stand_N, n=100), label="stand")
plt.legend()
plt.show()

plt.plot(dis_local_N)
plt.plot(dis_corr_N)
plt.plot(dis_stand_N, label="stand")
plt.legend()
plt.show()



