import pathlib
import sys
import numpy as np
from TS_AIDA import Subsampling as ss
from TS_AIDA import Random_subsampling as ss2
from Data import Dim_mixer as dm

if __name__ == "__main__":
    file_name = sys.argv[1]
    parameter_number = sys.argv[2]
    dim_anom = int(sys.argv[3])

print(parameter_number, file_name, dim_anom)


Data_np = np.load(file_name + "/data.npy")
Anom_np = np.load(file_name + "/anom.npy")

F = dm.Mixer(Data_np[:,:,:,1],Data_np[:,:,:,0], dim_anom)

N = 100
P1 = [10, 100, 2,4,30,50,6, 3, True, True, True]
P2 = [10, 200, 2,4,30,50,6, 3, True, True, True]
P3 = [40, 100, 2,4,30,50,6, 3, True, True, True]
P4 = [40, 200, 2,4,30,50,6, 3, True, True, True]

P5 = [10, 100, 2,4,30,50,6, 3, False, True, True]
P6 = [10, 200, 2,4,30,50,6, 3, False, True, True]
P7 = [40, 100, 2,4,30,50,6, 3, False, True, True]
P8 = [40, 200, 2,4,30,50,6, 3, False, True, True]

parameter_sets = [P1, P2, P3, P4, P5, P6, P7, P8]
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

new_dir = pathlib.Path(file_name, "Score_" + "P1")
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


Data_np = Data_np[:,:,:,1]
for i in range(0, Data_np.shape[0]):

    print("Scoring: " + str(i))
    sample = Data_np[i,:,:]

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

    np.save(
       str(new_dir) + "/score.npy",
        output)
