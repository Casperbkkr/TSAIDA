import os
import subprocess

import time


Data_dir = "/Users/casperbakker/PycharmProjects/PythonProject/Data/Convergence"

parameter_sets_file = "/Users/casperbakker/PycharmProjects/PythonProject/Data/Parameter_set"

score_calculator_script = '/Users/casperbakker/PycharmProjects/PythonProject/Data/Score_calculator.py'
for filename in os.listdir(Data_dir):
    for type in os.listdir(Data_dir + "/" + filename):
        for data in os.listdir(Data_dir + "/" + filename + "/" + type):
            data_name = Data_dir + "/"+ filename + "/" + type
            for parameter_set in range(0,2):#os.listdir(parameter_sets_file):
                for N in [50,100, 200,400, 800]:

                    start = time.time()

                    subprocess.call(['python',
                                      score_calculator_script,
                                      data_name,
                                      str(parameter_set),
                                      str(N)])
                    end = time.time()
                    print("elapsed_time is:" +str( end - start))