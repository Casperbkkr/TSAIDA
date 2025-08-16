import pathlib
import sys
import numpy as np
from TS_AIDA import Subsampling as ss
from TS_AIDA import Random_subsampling as ss2
from Data import Dim_mixer as dm
import matplotlib.pyplot as plt
if __name__ == "__main__":
    file_name = sys.argv[1]
    parameter_number = int(sys.argv[2])
    N = int(sys.argv[3])

print(parameter_number, file_name, N)


Data_np = np.load(file_name+"/data.npy" )
Anom_np = np.load(file_name+"/data.npy")

Data_mix = dm.Mixer(Data_np[:,:,:,1],Data_np[:,:,:,0], 3)
#Data_mix = Data_np


N = 100
P1 = [20, 50, 2,5,10,50,5, 3, True, True, True]
P2 = [20, 50, 2,5,10,50,5, 3, True, True, False]





parameter_sets = [P1, P2]
current_parameters = parameter_sets[parameter_number]

w_min = P1[0]
w_max =  P1[1]
d_min =  P1[2]
d_max =  P1[3]
n_min =  P1[4]
n_max =  P1[5]
w_ratio =  P1[6]
sig_terms =  P1[7]

local = bool(P1[8])
window_corrected =  bool(P1[9])
dim_corrected =  bool(P1[10])

new_dir = pathlib.Path(file_name, "Score_" + "P"+str(parameter_number+1))
new_dir.mkdir(parents=True, exist_ok=True)
new_dir.mkdir(parents=True, exist_ok=True)
new_file = new_dir / 'Parameters.txt'
new_file.write_text("N =" + str(N) +
                    "\n w_min =" + str(w_min) +
                    "\n w_max =" + str(w_max)+
                    "\n d_min =" + str(d_min)+
                    "\n d_max =" + str(d_max)+
                    "\n n_min =" + str(n_min)+
                    "\n n_max =" + str(n_max)+
                    "\n w_ratio =" + str(w_ratio)+
                    "\n sig_terms =" + str(sig_terms)+
                    "\n local=" + str(local)+
                    "\n window corrected=" + str(window_corrected)+
                    "\n dim_corrected=" + str(dim_corrected))



parameters = (w_min, w_max, d_min, d_max, n_min, n_max, w_ratio)

T = 1
K = sig_terms
output = np.zeros(shape=[Data_np.shape[0], Data_np.shape[1], 4])


#Data_np = Data_np[:,:,:,1]
for i in range(0, Data_np.shape[0]):

    print("Scoring: " + str(i))
    sample = Data_mix[i,:,:]

    sample_info = ss2.Random_subsampler(sample, N=N, parameters=parameters,
                                        local=local,
                                        window_corrected=window_corrected,
                                        dim_corrected=dim_corrected)

    sample_info2 = ss2.Random_subsampler(sample, N=N, parameters=parameters,
                                         local=local,
                                         window_corrected=window_corrected,
                                         dim_corrected=dim_corrected)

    sig, _, sig_N = ss.Score_aggregator(sample, sample_info, K, T, sig=True, normalize=False, r=0.5)
    dis, _, dis_N = ss.Score_aggregator(sample, sample_info, K, T, sig=False, normalize=False, r=0.5)




    output[i, :, 0] = sig
    output[i, :, 1] = dis
    output[i, :, 2] = sig_N
    output[i, :, 3] = dis_N

plt.plot(Data_mix[1,:,0])
plt.plot(Data_mix[1,:,1])
plt.plot(Data_mix[1,:,2])
plt.plot(Data_mix[1,:,3])
plt.show()
np.save(
    str(new_dir) + "/score_N"+str(N)+".npy",
    output)
