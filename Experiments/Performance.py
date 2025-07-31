
import matplotlib.pyplot as plt
import numpy as np
from Processes import Events as ev
from Processes import Heston as hes
from Codes import Subsampling as ss
from Codes import Path_signature as ps
from Experiments import Score as sc

dis_data = np.load("Data_gen/Synth_data/var_data_1747732168/standard_sig.npy")
sig_data = np.load("Data_gen/Synth_data/var_data_1747732168/standard_sig.npy")
#anom_data = np.load("Synth_data/jump_data_1747138536/anom.npy")
sigma = np.zeros([10000, 1000])
sigma[5000:5500, :] = 1
anom_data = sigma

v = anom_data[:, 1000:-1000]
s = np.sum(anom_data[:, 1000:-1000], axis=1)
s1 = list(np.where(s != 0)[0])

#dis_data = dis_data[s1,:]
#sig_data = sig_data[s1,:]
#anom_data = anom_data[s1,:]
epochs = 100
thresh0 = 0.1
h = 0.01
learning_rate = [0.01*0.95**i for i in range(epochs)]

thresh_out_dis = np.zeros(shape=(120, epochs))
score_out_dis = np.zeros(shape=(120, epochs))
thresh_out_sig = np.zeros(shape=(120, epochs))
score_out_sig = np.zeros(shape=(120, epochs))

for i in range(0,120):

    score_dis = dis_data[i, :]
    score_sig = sig_data[i, :]
    y_true = anom_data[i, 1000:-1000]
    T0_dis = thresh0
    T0_sig = thresh0
    print(i)
    for j in range(epochs):
        y_dis = sc.Predict(score_dis, T0_dis)[0][1000:-1000,np.newaxis]
        y_sig = sc.Predict(score_sig, T0_sig)[0][1000:-1000,np.newaxis]

        cost_true_dis = sc.F(y_true, y_dis)
        cost_true_sig = sc.F(y_true, y_sig)

        y_dis_pert_dis = sc.Predict(score_dis, T0_dis + h)[0][1000:-1000,np.newaxis]
        y_dis_pert_sig = sc.Predict(score_sig, T0_sig + h)[0][1000:-1000,np.newaxis]

        cost_pert_dis = sc.F(y_true, y_dis_pert_dis)
        cost_pert_sig = sc.F(y_true, y_dis_pert_sig)

        grad_dis = (cost_pert_dis- cost_true_dis) / h
        grad_sig = (cost_pert_sig - cost_true_sig) / h

        thresh_out_dis[i, j] = T0_dis
        score_out_dis[i, j] = cost_true_dis
        T0_dis = T0_dis + learning_rate[j] * grad_dis

        thresh_out_sig[i, j] = T0_sig
        score_out_sig[i, j] = cost_true_sig
        T0_sig = T0_sig + learning_rate[j] * grad_sig
    plt.plot(score_dis, label="dis")
    plt.plot(score_sig, label="sig")
    plt.plot(anom_data[i, :])
    plt.legend()
    plt.show()
    plt.plot(score_out_dis[i,:], label="dis")
    plt.plot(score_out_sig[i,:], label="sig")
    plt.legend()
    plt.show()
    plt.plot(thresh_out_dis[i, :], label="dis")
    plt.plot(thresh_out_sig[i, :], label="sig")
    plt.legend()
    plt.show()
    if i %20 == 0:
        np.save("Synth_data/var_data_1747732168" + "/sig_perf_N100_200_d3", score_out_sig)
        np.save("Synth_data/var_data_1747732168" + "/dis_perf_N100_200_d3", score_out_dis)