import numpy as np

def Precision(tpe, fpe, fpt, n_normal):
    A = (tpe)/(tpe + fpe)
    B = 1-(fpt/n_normal)
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
    y_pred = y_pred[:,0]
    true_events = Sparse_events(y_true)
    detected_events = Sparse_events(y_pred)
    total_events = true_events.shape[1]
    event_detected = np.zeros(total_events)
    false_positive = np.zeros(detected_events.shape[1])
    true_positive2 = np.zeros(detected_events.shape[1])
    for j in range(detected_events.shape[1]):
        p = detected_events[:,j]
        tp = False
        for i in range(true_events.shape[1]):
            t = true_events[:,i]
            if t[0] >= p[0] and t[1] <= p[1]:
                event_detected[i] = 1
                true_positive2[j] = 1
                tp = True
            if t[0] <= p[0] and t[1] >= p[1]:
                event_detected[i] = 1
                true_positive2[j] = 1
                tp = True
            if t[0] <= p[0] <= t[1] and t[1] <= p[1]:
                event_detected[i] = 1
                true_positive2[j] = 1
                tp = True
            if t[0] >= p[0] and p[0]>=t[1] >= p[1]:
                event_detected[i] = 1
                true_positive2[j] = 1
                tp = True
        if tp == False:
            if len(false_positive) > 0:
                false_positive[j] = 1

    n_false_positive = np.sum(false_positive)
    n_true_positive = np.sum(true_positive2)
    n_false_negative = event_detected.shape[0] - np.sum(event_detected)

    dif = y_true - y_pred
    n_normal = y_true.shape[0] - np.sum(y_true)
    fpt = np.where(dif == -1, 1, 0)
    fpt = np.sum(fpt)

    precision = Precision(n_true_positive,
                          n_false_positive,
                          fpt,
                          n_normal)
    recall = Recall(n_true_positive, n_false_negative)

    top = (1.25) * precision * recall
    bottom = (0.25) * precision + recall

    return top/bottom
