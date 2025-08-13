import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

from Codes import Subsampling as ss
from Codes import Classifier as cl

columns = pd.read_csv("esa-adb-challenge/target_channels.csv").to_numpy()[:,0]
data = pd.read_parquet("esa-adb-challenge/train.parquet")
data_test = pd.read_parquet("esa-adb-challenge/test.parquet")
anomaly = data["is_anomaly"]
anomaly_np = anomaly.to_numpy()

data = data[columns]

data_selection = data.iloc[160000:180000]
#data_selection = data_selection[["channel_12", "channel_13", "channel_14", "channel_15", "channel_16", "channel_17" , "channel_18", "channel_19", "channel_20", "channel_21"]]#, "channel_22", ]]
anomaly_selection = anomaly.iloc[160000:180000]
anom = anomaly_np[160000:180000]

Data = data_selection
Data_np = Data.to_numpy()
data_np = Data_np/Data_np.max()
plt.plot(Data_np)
plt.show()

print("Data loaded")


sample_info_local = ss.sub_sample(Data_np, N=1000, n_samples_min_max=[20, 50], length_min_max=[20, 300],
                                 dim_min_max=(3, 7))
sample_info_global = ss.local_search(Data_np, N=1000, n_samples_min_max=[20, 50], length_min_max=[20, 300],
                                 dim_min_max=(3, 7))
sample_info_corr = ss.local_search_corr(Data_np, N=1000, n_samples_min_max=[20, 50], length_min_max=[20, 300],
                                 dim_min_max=(3, 7))

f = lambda X: X
T = 1
K = 5

sig_local = ss.Score_aggregator(Data_np, sample_info_local, K, T)[0]
dis_local = ss.sub_sampler(Data_np, sample_info_local, f)[0]

sig_corr = ss.Score_aggregator(Data_np, sample_info_corr, K, T)[0]
dis_corr = ss.sub_sampler(Data_np, sample_info_corr, f)[0]

sig_global = ss.Score_aggregator(Data_np, sample_info_global, K, T)[0]
dis_global = ss.sub_sampler(Data_np, sample_info_global, f)[0]



plt.plot(sig_global, label="Signature global")
plt.plot(sig_local, label="Signature local")
plt.plot(sig_corr, label="Signature corrected")
plt.plot(anom)
plt.ylabel("Outlier score")
plt.legend()
plt.xlabel("Time")
plt.show()

plt.plot(dis_global, label="Distance global")
plt.plot(dis_local, label="Distance local")
plt.plot(dis_corr, label="Distance corrected")
plt.plot(anom)
plt.ylabel("Outlier score")
plt.legend()
plt.xlabel("Time")
plt.show()

sig_t, sig_score = cl.Max_perf(sig_local, anom)

dis_t, dis_score = cl.Max_perf(dis_local, anom)

plt.plot(dis_score)
plt.plot(sig_score)
plt.show()


print(np.max(dis_score))
print(np.max(sig_score))