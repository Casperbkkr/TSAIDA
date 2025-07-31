
import pandas as pd
from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()
import roughpy as rp
from Processes import Events as ev
from Processes import Heston as hes
from Codes import Subsampling as ss
from Codes import Random_subsampling as ss2
from Codes import Path_signature as ps
import gc
from Codes import Classifier as cl


Data = pd.read_csv("/Users/casperbakker/Documents/PycharmProjects/Thesis/Experiments/FinData/FinData.csv")
Data.fillna(method='bfill', inplace=True)
Data_np = Data.to_numpy()[:,1:]
Data_np = Data_np.astype(float)

Data.plot()
plt.show()

Data_np = Data_np / Data_np.max(axis=0)
plt.plot(Data_np)
plt.show()
print("Data loaded")

N = 1000

parameters = (20, 100, 2, 4, 20, 50,  5)

sample_info_local = ss2.Random_subsampler(Data_np, N=N, parameters=parameters, local=True, window_corrected=True, dim_corrected=True)

sample_info_corr = ss2.Random_subsampler(Data_np, N=N, parameters=parameters, local=True, window_corrected=False, dim_corrected=False)

sample_info_corr_dim = ss2.Random_subsampler(Data_np, N=N, parameters=parameters)

sample_info = ss2.Random_subsampler(Data_np, N=N, parameters=parameters, local=True, window_corrected=True, dim_corrected=True)
"""
sample_info_local = ss.local_search(Data_np, N=N, n_samples_min_max=[20, 50], length_min_max=[20, 200],
                                 dim_min_max=(2,4))
sample_info_corr = ss.local_search_corr(Data_np, N=N, n_samples_min_max=[20, 50], length_min_max=[20, 200],
                                 dim_min_max=(2, 4))
sample_info = ss.sub_sample(Data_np, N=N, n_samples_min_max=[20, 50], length_min_max=[20, 200],
                                 dim_min_max=(2, 4))
"""

f = lambda X: X
T = 1
K = 3

sig_local, _, sig_local_N = ss.Score_aggregator(Data_np, sample_info_local, K, T, sig=True)
dis_local, _, dis_local_N= ss.Score_aggregator(Data_np, sample_info_local, K, T, sig=False)

sig_corr = ss.Score_aggregator(Data_np, sample_info_corr, K, T, sig=True)[0]
dis_corr = ss.Score_aggregator(Data_np, sample_info_corr, K, T, sig=False)[0]

sig_stand = ss.Score_aggregator(Data_np, sample_info, K, T, sig=True)[0]
dis_stand = ss.Score_aggregator(Data_np, sample_info, K, T, sig=False)[0]

df_sig = pd.DataFrame(index=Data["observation_date"])
df_sig["Signature local"] = sig_local
df_sig["dis_local"] = dis_local
df_sig["Signature local N"] = sig_local_N
df_sig["dis_local N "] = dis_local_N

df_sig.plot(
)
plt.legend()
plt.ylabel("Outlier score")
plt.xlabel("Time")
plt.show()


df_sig = pd.DataFrame(index=Data["observation_date"])
df_sig["Signature local"] = sig_local
df_sig["Signature corrected"] = sig_corr
df_sig["Signature standard"] = sig_stand

df_dis = pd.DataFrame(index=Data["observation_date"])
df_dis["Distance local"] = dis_local
df_dis["Distance corrected"] = dis_corr
df_dis["Distance standard"] = dis_stand


df_sig.plot(
)
plt.legend()
plt.ylabel("Outlier score")
plt.xlabel("Time")
plt.show()


df_dis.plot(
)
plt.ylabel("Outlier score")
plt.xlabel("Time")
plt.legend()
plt.show()



a=1