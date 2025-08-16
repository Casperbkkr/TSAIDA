import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


Data = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Data/finished_Data/jump_data_1755255199/Total_data.npy")
Data_path = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Data/finished_Data/jump_data_1755255199/data.npy")


Data_sig = Data[0,:,:,0,:]
Data_dis = Data[0,:,:,1,:]

sample_n = 7

plt.plot(Data_path[sample_n,:,:,1])
plt.plot(Data_path[sample_n,:,:,1])
plt.show()

plt.plot(Data_sig[sample_n,:,0][100:-100])
plt.plot(Data_sig[sample_n,:,2][100:-100])
plt.plot(Data_sig[sample_n,:,4][100:-100])
plt.plot(Data_sig[sample_n,:,6][100:-100])
plt.show()
a=2