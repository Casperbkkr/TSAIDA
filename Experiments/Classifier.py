import numpy as np
from Codes import Subsampling as Ss

from Codes import Path_signature as Ps
from Codes.Path_signature import path_signature
import Score as sc

def Train(T0, learning_rate, y_true, data):
    for A in range(5):
        h = 0.001
        thresh_out = []
        score_out = []
        for epoch in range(100):
            prediction = sc.Predict(score, T0)[0][2000:-2000]
            cost_true = sc.F(y_true, prediction)
            prediction_pert = sc.Predict(score, T0+h)[0][2000:-2000]
            cost_pert = sc.F(y_true, prediction_pert)
            grad = (cost_pert - cost_true)/h
            print(grad)
            T0 = T0 + learning_rate * grad
            thresh_out.append(T0)
            score_out.append(cost_true)
            print(T0, cost_true)

    return thresh_out, score_out

