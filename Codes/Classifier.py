import numpy as np
from Codes import Subsampling as Ss

from Codes import Path_signature as Ps
from Codes.Path_signature import path_signature


def Train(data, y_true):
    thresholds = np.arange(0, 1, 0.01)
    thresholds = np.repeat(thresholds, thresholds.shape[0])
    data = np.repeat(data, thresholds.shape[0], axis=0)

    pred = np.where(data>=thresholds, 1, 0)


    return


def Max_perf1(y_pred, y_true, learning_rate = 0.1, h=0.1):
    y_pred = y_pred[100:-100]
    y_true = y_true[100:-100]
    T0 =  0

    thresh_out = []
    score_out = []


    for epoch in range(1,100):

        prediction = Predict(y_pred, T0)[0]
        #h = 0.5* np.exp(-0.99 * epoch)
        cost_true = F(y_true, prediction-h)
        prediction_pert = Predict(y_pred, T0 + h)[0]
        cost_pert = F(y_true, prediction_pert)
        grad = (cost_pert - cost_true) / 2*h
        print(grad)
        #learning_rate = learning_rate * np.exp(-0.99*epoch)
        T0 = T0 + learning_rate * grad
        thresh_out.append(T0)
        score_out.append(cost_true)
        print(T0, cost_true)

    return thresh_out, score_out

def Max_perf(y_pred, y_true, learning_rate = 0.5, h=1):
    y_pred = y_pred[100:-100]
    y_true = y_true[100:-100]


    thresh_out = []
    score_out = []


    cost_true = np.zeros(shape=100)
    T0 = [0.1*i for i in range(40)]
    for i in range(40):
        prediction = Predict(y_pred, T0[i])[0]

        cost_true[i] = F(y_true, prediction)


    return  T0, cost_true


def Precision(tpe, fpe, fpt, nt):
    if (tpe + fpe) == 0:
        return 1
    A = (tpe)/(tpe + fpe)
    B = (1-fpt/nt)
    return A*B

def Recall(tpe, fne):
    return  (tpe)/(tpe + fne)

def F05(y_true, y_pred):
    true_events = Sparse_events(y_true)
    tpe, tpt, detected_events = TPE(y_true, y_pred)
    fpe, fpt = FPE(y_true, y_pred)
    fne, undetected = FNE(y_true, y_pred, true_events)

    fne_count = fne
    for i in range(undetected.shape[1]):
        for j in range(detected_events.shape[1]):
            A = detected_events[:,j] - undetected[:,i]
            k=1
            if detected_events[0,j] <= undetected[0,i] and detected_events[1,j] >= undetected[1,i]:
                fne_count -= 1


    precision = Precision(tpe, fpe, fpt, y_true.shape[0])
    recall = Recall(tpe, fne_count)

    top = (1.25) * precision * recall
    bottom = (0.25) * precision + recall

    return top/bottom

def FNE(y_true, y_pred, true_events):
    dif = y_pred - y_true
    fnt = np.where(dif == -1, 1, 0)
    fne = Sparse_events(fnt)

    return fne.shape[1], fne

def TPE(y_true, y_pred):
    tpt = y_true + y_pred
    #tpt = np.where(dif == 2, 1, 0)
    tpe = Sparse_events(tpt)
    return tpe.shape[1], np.sum(tpt), tpe

def FPE(y_true, y_pred):
    dif = y_true - y_pred
    fpt = np.where(dif == -1, 1, 0)
    fpe = Sparse_events(fpt)
    return fpe.shape[1], np.sum(fpt)

def Sparse_events(is_anomaly):
    event_start = []
    event_end = []
    in_event = False
    for i in range(is_anomaly.shape[0]):
        if in_event == False:
            if is_anomaly[i] == 1:
                event_start.append(i)
                in_event = True
            if is_anomaly[i] == 2:
                event_start.append(i)
                in_event = True
        if in_event == True:
            if is_anomaly[i] == 0:
                event_end.append(i)
                in_event = False

    if in_event == True:
        event_end.append(is_anomaly.shape[0])
    out = np.array([event_start, event_end])
    return out

def Predict(score, threshold):
    threshold_vec = threshold*np.ones_like(score)
    y_pred = np.where(score >= threshold_vec, 1, 0)

    pred_plot = np.where(score >= threshold_vec, 0.5, 0)
    return y_pred, pred_plot

def F(y_true, y_pred):


    true_events = Sparse_events(y_true)
    detected_events = Sparse_events(y_pred)

    detected = np.zeros(shape=true_events.shape[1])
    detection_counter = np.zeros(true_events.shape[1])
    event_det_true = y_true + y_pred
    for i in range(true_events.shape[1]):
        event_start = true_events[0,i]
        event_end = true_events[1,i]
        slic = y_pred[event_start:event_end]

        if np.any(slic == 1) == True:
            detected[i] = 1
            detection_counter[i] = Sparse_events(slic).shape[1]

    n_true_positive = detected.sum()
    n_false_negative = true_events.shape[1] - n_true_positive
    n_false_positive = detected_events.shape[1] - detection_counter.sum()


    dif = y_true - y_pred
    fpt = np.where(dif == -1, 1, 0)
    fpt = np.sum(fpt)

    precision = Precision(n_true_positive,
                          n_false_positive,
                          fpt,
                          y_true.shape[0]-np.sum(y_true))
    recall = Recall(n_true_positive, n_false_negative)

    top = (1.25) * precision * recall
    bottom = (0.25) * precision + recall

    return top/bottom


def F_old(y_true, y_pred):
    true_events = Sparse_events(y_true)
    detected_events = Sparse_events(y_pred)
    total_events = true_events.shape[1]
    event_detected = np.zeros(total_events)
    false_positive = np.zeros(detected_events.shape[1])
    for j in range(detected_events.shape[1]):
        p = detected_events[:,j]
        tp = False
        for i in range(true_events.shape[1]):
            t = true_events[:,i]
            I = np.arange(p[0],p[1])
            kg = I-t[0]
            r = np.any(I-t[0] > 0)
            if np.any(I-t[0] > 0) and np.any(I-t[1] < 0):
                event_detected[i] = 1
                tp=True

        if tp == False:
            if len(false_positive) > 0:
                false_positive[j] = 1

    n_false_positive = np.sum(false_positive)
    n_true_positive = np.sum(event_detected)
    n_false_negative = event_detected.shape[0] - n_true_positive

    dif = y_true - y_pred
    fpt = np.where(dif == -1, 1, 0)
    fpt = np.sum(fpt)

    precision = Precision(n_true_positive,
                          n_false_positive,
                          fpt,
                          y_true.shape[0])
    recall = Recall(n_true_positive, n_false_negative)

    top = (1.25) * precision * recall
    bottom = (0.25) * precision + recall

    return top/bottom
