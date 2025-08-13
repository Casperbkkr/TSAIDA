import numpy as np
import matplotlib.pyplot as plt
import roughpy as rp

import pandas as pd
from TS_AIDA import Subsampling as ss
from TS_AIDA import AIDA as AI

from TS_AIDA import Random_subsampling as ss2
import seaborn as sns
sns.set_theme()

data = np.load("/Data/Synth_data/Period/period_data_1754391695/data.npy")
anom = np.load("/Data/Synth_data/Period/period_data_1754391695/anom.npy")

def sig_rp(X, K, interval, indices, context):
    times = indices
    #context = rp.get_context(width=X.shape[0], depth=K, coeffs=rp.DPReal)
    stream = rp.LieIncrementStream.from_increments(X, indices=times, ctx=context)

    return stream.signature(interval)


T=100
K=4
f = lambda X: X
sig_scores = np.zeros(shape=100)
dis_scores = np.zeros(shape=100)

N = 50
min_w = 30
max_w = 500
parameters = (30, 500, 2, 10, 20, 50,  5)

for i in range(1):
    sig_out_local = np.zeros(shape=(data.shape[1]-400, data.shape[3]))
    sig_out_corr = np.zeros(shape=(data.shape[1]-400, data.shape[3]))
    sig_out_stand = np.zeros(shape=(data.shape[1]-400, data.shape[3]))
    dis_out_local = np.zeros(shape=(data.shape[1]-400, data.shape[3]))
    dis_out_corr = np.zeros(shape=(data.shape[1]-400, data.shape[3]))
    dis_out_stand = np.zeros(shape=(data.shape[1]-400, data.shape[3]))

    sample = data[0, :, :, 0]
    anom_sample = anom[i, :]

    sample_info_local = ss2.Random_subsampler(sample, N=N, parameters=parameters, local=True, window_corrected=False, dim_corrected=True)
    sample_info_corr = ss2.Random_subsampler(sample, N=N, parameters=parameters, local=True, window_corrected=True, dim_corrected=False)
    sample_info = ss2.Random_subsampler(sample, N=N, parameters=parameters, local=True, window_corrected=True, dim_corrected=True)

    for d in range(10):#data.shape[3]):
        sample = data[i, :, :, d]

        sig_out_local[:,d] = -1* ss.Score_aggregator(sample, sample_info_local, K, T, sig =True)[0][200:-200]
        dis_out_local[:,d] = ss.Score_aggregator(sample, sample_info_local, K, T, sig =False)[0][200:-200]

        sig_out_corr[:,d] = -1* ss.Score_aggregator(sample, sample_info_corr, K, T, sig =True)[0][200:-200]
        dis_out_corr[:,d] = ss.Score_aggregator(sample, sample_info_corr, K, T, sig =False)[0][200:-200]

        sig_out_stand[:,d] = -1* ss.Score_aggregator(sample, sample_info,K, T, sig =True)[0][200:-200]
        dis_out_stand[:,d] = ss.Score_aggregator(sample, sample_info, K, T, sig =False)[0][200:-200]
    y= [1 for i in range(10)]
    sig_out_local_p = AI.DistanceProfile(sig_out_local[:,0], sig_out_local.transpose())
    dis_out_local_p = AI.DistanceProfile(dis_out_local[:,0], dis_out_local.transpose())
    sig_out_corr_p = AI.DistanceProfile(sig_out_corr[:,0], sig_out_corr.transpose())
    dis_out_corr_p = AI.DistanceProfile(dis_out_corr[:,0], dis_out_corr.transpose())
    sig_out_stand_p = AI.DistanceProfile(sig_out_stand[:,0], sig_out_stand.transpose())
    dis_out_stand_p = AI.DistanceProfile(dis_out_stand[:,0], dis_out_stand.transpose())


    plt.scatter(sig_out_local_p/np.max(sig_out_local_p),[1 for i in range(10)], label=1)
    plt.scatter(sig_out_corr_p/np.max(sig_out_corr_p),[3 for i in range(10)], label=3)
    plt.scatter(sig_out_stand_p / np.max(sig_out_stand_p), [5 for i in range(10)], label=5)

    plt.legend()

    plt.show()
    plt.scatter(dis_out_local_p / np.max(dis_out_local_p), [2 for i in range(10)], label=2)
    plt.scatter(dis_out_corr_p/np.max(dis_out_corr_p),[4 for i in range(10)], label=4)
    plt.scatter(dis_out_stand_p/np.max(dis_out_stand_p),[6 for i in range(10)], label=6)
    plt.legend()

    plt.show()

    for d in range(10):
        plt.plot(sig_out_local[4000:6000,d], label=str(d))
    plt.legend()
    plt.title("sig_out_local")
    plt.show()
    for d in range(10):
        plt.plot(sig_out_corr[4000:6000,d], label=str(d))
    plt.title("sig_out_corr")
    plt.legend()
    plt.show()
    for d in range(10):
        plt.plot(sig_out_stand[4000:6000,d], label=str(d))
    plt.title("sig_out_stand")
    plt.legend()
    plt.show()
    for d in range(10):
        plt.plot(dis_out_local[4000:6000,d], label=str(d))
    plt.title("dis_out_local")
    plt.legend()
    plt.show()
    for d in range(10):
        plt.plot(dis_out_corr[4000:6000,d], label=str(d))
    plt.title("dis_out_corr")
    plt.legend()
    plt.show()
    for d in range(10):
        plt.plot(dis_out_stand[4000:6000,d], label=str(d))
    plt.title("dis_out_stand")
    plt.legend()
    plt.show()




print(sig_scores.avg())
print(dis_scores.avg())




