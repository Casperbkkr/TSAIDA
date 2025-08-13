import os
import subprocess



Data_dir = "/Users/casperbakker/PycharmProjects/PythonProject/Data/Synth_data"

parameter_sets_file = "/Users/casperbakker/PycharmProjects/PythonProject/Data/Parameter_set"

score_calculator_script = '/Users/casperbakker/PycharmProjects/PythonProject/Data/Score_calculator.py'
for filename in os.listdir(Data_dir):
    for type in os.listdir(Data_dir + "/" + filename):
        for data in os.listdir(Data_dir + "/" + filename + "/" + type):
            data_name = Data_dir + "/"+ filename + "/" + type + "/" + data
            for parameter_set in range(0,8):#os.listdir(parameter_sets_file):
                for dim_anom in range(10):
                    arguments = []
                    subprocess.call(['python',
                                      score_calculator_script,
                                      data_name,
                                      str(parameter_set),
                                      str(dim_anom)])
