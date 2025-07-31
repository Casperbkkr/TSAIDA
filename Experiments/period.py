import numpy as np
import matplotlib.pyplot as plt
import roughpy as rp
import scipy.stats as stats
from networkx.algorithms.matching import min_weight_matching
import pandas as pd
from Codes import Subsampling as ss
from Codes import Classifier as cl
from Codes import Random_subsampling as ss2
import seaborn as sns
sns.set_theme()

data = np.load("/Users/casperbakker/Documents/PycharmProjects/Thesis/Experiments/Data_gen/Synth_data/Noise_data_1753872210/jump.npy")
anom = np.load("/Users/casperbakker/Documents/PycharmProjects/Thesis/Experiments/Data_gen/Synth_data/Noise_data_1753872210/anom.npy")

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

N = 100
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


        sig_out_local[:,d] = -1* ss.Score_aggregator(sample, sample_info_local, K, T)[0][200:-200]
        dis_out_local[:,d] = ss.sub_sampler(sample, sample_info_local, f)[0][200:-200]

        sig_out_corr[:,d] = -1* ss.Score_aggregator(sample, sample_info_corr, K, T)[0][200:-200]
        dis_out_corr[:,d] = ss.sub_sampler(sample, sample_info_corr, f)[0][200:-200]

        sig_out_stand[:,d] = -1* ss.Score_aggregator(sample, sample_info, K, T)[0][200:-200]
        dis_out_stand[:,d] = ss.sub_sampler(sample, sample_info, f)[0][200:-200]

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


    """
        df_sig = pd.DataFrame()
        df_sig["Signature local"] = sig_corr
        df_sig["Signature corrected"] = sig_local
        df_sig["Signature standard"] = sig_stand

        df_dis = pd.DataFrame()
        df_dis["Distance local"] = dis_local
        df_dis["Distance corrected"] = dis_corr
        df_dis["Distance standard"] = dis_stand

        df_sig.plot(subplots=True)
        plt.title("min_w=" + str(min_w) + ", max_w=" + str(max_w))
        plt.legend()
        plt.ylabel("Outlier score")
        plt.xlabel("Time")
        plt.show()

        df_dis.plot(subplots=True)
        plt.title("min_w=" + str(min_w) + ", max_w=" + str(max_w))
        plt.ylabel("Outlier score")
        plt.xlabel("Time")
        plt.legend()
        plt.show()
        """
    """
    sig_t, sig_score = cl.Max_perf(output_sig, anom_sample)

    dis_t, dis_score = cl.Max_perf(output_dis, anom_sample)
    sig_scores[i] = np.max(sig_score)
    dis_scores[i] = np.max(dis_score)
    print(np.max(sig_score))
    print(np.max(dis_score))
    """

print(sig_scores.avg())
print(dis_scores.avg())







plt.plot(output_dis, label="Distance")
plt.plot(output_sig, label="Signature")
plt.ylabel("Outlier score")
plt.xlabel("Time")
plt.legend()
plt.show()