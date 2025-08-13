
import numpy as np
import roughpy as rp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from scipy import stats
from Codes import AIDA as aida
import sys
outputs_sig = []
G = []
F = []
outputs_dis = []
for n in [50]:#, 100]:#, 150, 200, 250, 300, 350, 500]:
    A = np.load("Results/Window_robust_100/window_sig3_var" + str(n) + ".npy")[:50]
    B = np.load("Results/Window_robust_100/window_dis3_var" + str(n) + ".npy")[:50]#
    C = A/B
    D = np.mean(C)
    outputs_sig.append(D)
    G.append(np.mean(A))
    F.append(np.mean(B))





plt.plot(outputs_sig)
plt.show()
X = [0.5]#, 1, 1.5, 2, 2.5]
plt.plot(X,G, label="Signature")
plt.plot(X,F, label="Distance")
plt.xlabel("Window size as percentage of anomaly size")
plt.ylabel("Variation in isolation score")
plt.legend()
plt.show()

import gc as gc
a = gc.isenabled()
gc.enable()

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
#var_data_1750230793
data = np.load("Data_gen/Synth_data/var_data_1750404035/var.npy")[:,:,:4]
#anom = np.load("Data_gen/Synth_data/var_data_1749218431/sig0005.npy")

f = data[3,:,0]

plt.plot(f)
plt.show()
#data[:,2000:2100,0] = data[:,2000:2100,0] * 1.005

K=3
window_size = 100
step_size = 10
n_comp = 50

range_min = 5000+step_size - window_size
range_max = 5050-step_size + window_size
context = rp.get_context(width=4, depth=K, coeffs=rp.DPReal)
interval = rp.RealInterval(0, 1)
indices = np.linspace(0.1, 1, window_size)
windows_test = sliding_window(data[0, range_min:range_max,:], window_size, step_size)
comp_windows = sliding_window(data[0, range_max+1000:,:], window_size, step_size)

var_sig = np.zeros(shape=(data.shape[0]))
var_dis = np.zeros(shape=(data.shape[0]))

rng = np.random.default_rng()
LA = [i for i in range(comp_windows.shape[0])]
comp_index = rng.choice(LA, size=[comp_windows.shape[1], n_comp])
R = 5

for i in range(data.shape[0]):
    sample = data[i, range_min:range_max,:]
    comp_sample = data[i, range_max+1000:, :]
    windows = sliding_window(sample, window_size, step_size)
    comp_windows = sliding_window(comp_sample, window_size, step_size)
    comp_windows_select = [rng.choice(comp_windows, size=[n_comp]) for tr in range(R)]
    A = np.zeros(shape=[R, windows.shape[0]])
    B = np.zeros(shape=[R, windows.shape[0]])
    for k in range(R):
        sig_comp = sig_rp2(comp_windows_select[k], K, interval, indices, context)



        for j in range(windows.shape[0]):
            sig = sig_rp2(windows, K, interval, indices, context)[j,:]

            sig_comp_join = np.concatenate((sig[np.newaxis, :], sig_comp), axis=0)
            dp_sig = aida.DistanceProfile(sig, sig_comp_join)

            win = windows[j, :]
            win_join = np.concatenate((win[np.newaxis, :], comp_windows_select[k]), axis=0)
            dp_dis = aida.DistanceProfile(win, win_join)

            dp_sig = dp_sig / np.max(dp_sig)
            dp_dis = dp_dis / np.max(dp_dis)

            A[k,j] = -1*aida.Isolation(dp_sig)[1]
            B[k,j] = -1*aida.Isolation(dp_dis)[1]

    A1 = np.mean(A, axis=0)
    B1 = np.mean(B, axis=0)
    var_sig[i] = np.var(A1)
    var_dis[i] = np.var(B1)
    #mv_sig = np.mean(var_sig, axis=0)
    #mv_dis = np.mean(var_dis, axis=0)

    print(var_sig, var_dis)
    plt.plot(var_sig, label="sig")
    plt.plot(var_dis, label="dis")
    plt.legend()
    plt.show()

    np.save("Results/Window_robust_100/window_sig3_var"+str(window_size), var_sig)
    np.save("Results/Window_robust_100/window_dis3_var"+str(window_size), var_dis)




outputs_sig = []

outputs_dis = []
for n in [50, 100, 150, 200, 250, 300, 350, 500]:
    A = np.load("Results/Window_robust_100/window_sig_var" + str(n) + ".npy")
    B = np.load("Results/Window_robust_100/window_dis_var" + str(n) + ".npy")
    C = A-B
    D = np.mean(C)
    outputs_sig.append(D)

plt.plot(D)
plt.show()

a=2