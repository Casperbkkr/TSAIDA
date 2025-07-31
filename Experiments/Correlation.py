import numpy as np

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
from Codes import AIDA as aida
import gc

def sliding_window(path, size, step):
    windows = np.lib.stride_tricks.sliding_window_view(path, size, axis=0)
    windows2 = windows[1::step]
    return windows2

def sig_rp(X, K, interval, indices, context):
    times = indices
    #context = rp.get_context(width=X.shape[0], depth=K, coeffs=rp.DPReal)
    stream = rp.LieIncrementStream.from_increments(X, indices=times, ctx=context)


    return stream.signature(interval)

def sig_rp2(X, K, interval, indices, context):
    if len(X.shape)>2:
        out_d = sum([X.shape[1]**i for i in range(1,K+1)])+1
    else:
        out_d = K+1

    output = np.zeros(shape=[X.shape[0], out_d])

    for t in range(X.shape[0]):
        samp = X[t,:,:].copy()
        samp = samp.transpose()
        A = sig_rp(samp, K, interval, indices, context)
        B = np.array(A)
        output[t, :] = B

    return output

data = np.load("Data_gen/Synth_data/corr_data_1749640273/cor_data.npy")
data = data[:,1000:,:]



sig = np.zeros([data.shape[0], data.shape[1]])
dis = np.zeros([data.shape[0], data.shape[1]])
K = 3
interval = rp.RealInterval(0, 1)
indices = np.linspace(0.1, 1, 500)

d_max=10

output_sig = np.zeros(shape=[100, d_max])
output_dis = np.zeros(shape=[100, d_max])

iso_sig = np.zeros(shape=[100, d_max])
iso_dis = np.zeros(shape=[100, d_max])

for experiment in range(data.shape[0]):
    for d in range(1,d_max):
        sample = data[experiment, :-500, :]
        windows2 = sliding_window(sample, 500, 100)

        context = rp.get_context(width=d, depth=K, coeffs=rp.DPReal)
        windows = windows2[:, :d, :]
        sig = sig_rp2(windows, K, interval, indices, context)

        dp_sig = aida.DistanceProfile(sig[0, :], sig)
        dp_sig = dp_sig / np.max(dp_sig)
        iso_sig[experiment, d] = aida.Isolation(dp_sig)[1]

        # calc dp for standard
        dp_dis = aida.DistanceProfile(windows[0, :], windows)
        dp_dis = dp_dis / np.max(dp_dis)
        iso_dis[experiment, d] = aida.Isolation(dp_dis)[1]

av_sig_anom = np.average(iso_sig, axis=0)
av_dis_anom = np.average(iso_dis, axis=0)


for experiment in range(data.shape[0]):
    for d in range(1,d_max):
        sample = data[experiment, 500:, :]
        windows2 = sliding_window(sample, 500, 100)

        context = rp.get_context(width=d, depth=K, coeffs=rp.DPReal)
        windows = windows2[:, :d, :]
        sig = sig_rp2(windows, K, interval, indices, context)

        dp_sig = aida.DistanceProfile(sig[0, :], sig)
        dp_sig = dp_sig / np.max(dp_sig)
        iso_sig[experiment, d] = aida.Isolation(dp_sig)[1]

        # calc dp for standard
        dp_dis = aida.DistanceProfile(windows[0, :], windows)
        dp_dis = dp_dis / np.max(dp_dis)
        iso_dis[experiment, d] = aida.Isolation(dp_dis)[1]

av_sig_nom = np.average(iso_sig, axis=0)
av_dis_nom = np.average(iso_dis, axis=0)


plt.plot(av_sig_anom, label="Anom sig")
plt.plot(av_dis_anom, label="Anom dis")
plt.plot(av_sig_nom, label="Nom sig")
plt.plot(av_dis_nom, label="Nom dis")
plt.legend()
plt.show()




kadsf=1