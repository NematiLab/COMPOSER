# Copyright 2021
# Supreeth P. Shashikumar <spshashikumar@health.ucsd.edu>
# Shamim Nemati <snemati@health.ucsd.edu>
# Revision: 1.1 Date: 08/03/2021

import numpy as np

def compute_confusion_matrix_per_patient(labels, predictions, threshold, seqLength, includeMaskPerPatient, false_positive_prediction_horizon = 44, snooze_length = 6):   
    # Fill-in confusion matrix.
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    tp_modified = 0
    fp_modified = 0

    snooze = 0
    
    for t in range(seqLength):       
        # Only check labels and predictions when not snoozed.
        if snooze == 0:           
            if includeMaskPerPatient[t]==1:
                label = labels[t] == 1
                prediction = predictions[t] >= threshold 
           
                # Is there a positive label within the horizon?
                label_horizon = np.any(labels[t:min(seqLength, t+false_positive_prediction_horizon+1)])

                if label_horizon and prediction:
                    tp += 1
                elif not label_horizon and prediction:
                    fp += 1
                elif label and not prediction:
                    fn += 1
                elif not label and not prediction:
                    tn += 1

                if prediction:
                    snooze = snooze_length
                
        else:
            snooze -= 1
            
    return tp, fp, fn, tn


def compute_snoozed_auc(labels, predictions,  seqLength, includeMaskPerPatient, false_positive_prediction_horizon=44, snooze_length = 6, num_thresholds = 500):
    # Check input for errors.
    if len(labels) != len(predictions):
        raise Exception('Numbers of labels and predictions must be equal.')
    if any(len(X) != len(Y) for X, Y in zip(labels, predictions)):
        raise Exception('Numbers of labels and predictions must be equal for all samples.')
    if any(not (x == 0 or x == 1) for X in labels for x in X):
        raise Exception('Labels must be binary.')
    if any(not (0 <= y <= 1) for Y in predictions for y in Y):
        raise Exception('Predictions must be between 0 and 1, inclusive.')

    # Find prediction thresholds.
    #num_cohorts = len(labels)
    sorted_predictions = [np.unique(Y) for Y in predictions]

    if not num_thresholds:
        thresholds = np.unique(np.concatenate(sorted_predictions))[::-1]
    else:
        percentiles = np.linspace(0, 100, num_thresholds)
        thresholds = np.unique(np.percentile(np.concatenate(sorted_predictions), percentiles, interpolation='nearest'))[::-1]

    if thresholds[0] != 1:
        thresholds = np.insert(thresholds, 0, 1)
    if thresholds[-1] == 0:
        thresholds = thresholds[:-1]
    num_thresholds = len(thresholds)
    
    num_patients = labels.shape[0]

    # Populate confusion matrix for each prediction threshold.
    sample_tp = np.zeros(num_patients)
    sample_fp = np.zeros(num_patients)
    sample_fn = np.zeros(num_patients)
    sample_tn = np.zeros(num_patients)
    sample_indices = -np.ones(num_patients, dtype=np.int)

    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)
    tn = np.zeros(num_thresholds)

    for i in range(num_thresholds):
        for j in range(num_patients):
            k = np.searchsorted(sorted_predictions[j], thresholds[i])
            if k != sample_indices[j]:
                a, b, c, d = compute_confusion_matrix_per_patient(labels[j], predictions[j], thresholds[i], seqLength[j], includeMaskPerPatient[j], false_positive_prediction_horizon, snooze_length)
                sample_tp[j] = a
                sample_fp[j] = b
                sample_fn[j] = c
                sample_tn[j] = d
                sample_indices[j] = k

            tp[i] += sample_tp[j]
            fp[i] += sample_fp[j]
            fn[i] += sample_fn[j]
            tn[i] += sample_tn[j]

    tpr = np.zeros(num_thresholds)
    tnr = np.zeros(num_thresholds)
    ppv = np.zeros(num_thresholds)
    npv = np.zeros(num_thresholds)

    for i in range(num_thresholds):
        if tp[i] + fn[i]:
            tpr[i] = tp[i] / (tp[i] + fn[i])
        else:
            tpr[i] = 1
        if fp[i] + tn[i]:
            tnr[i] = tn[i] / (fp[i] + tn[i])
        else:
            tpr[i] = 0
        if tp[i] + fp[i]:
            ppv[i] = tp[i] / (tp[i] + fp[i])
        else:
            ppv[i] = 1
        
        if fn[i] + tn[i]:
            npv[i] = tn[i] / (fn[i] + tn[i])
        else:
            npv[i] = 0

    # Compute AUROC as the area under a piecewise linear function of TPR
    # (x-axis) and FPR (y-axis) and AUPRC as the area under a piecewise linear
    # function of recall (x-axis) and precision (y-axis).
    auroc = 0
    auprc = 0
    auprc_modified = 0
    for i in range(num_thresholds - 1):
        auroc += 0.5 * (tpr[i + 1] - tpr[i]) * (tnr[i + 1] + tnr[i])
        auprc += (tpr[i + 1] - tpr[i]) * (ppv[i + 1])

#     for i in range(num_thresholds):
#         print(i, thresholds[i], tp[i], fp[i], fn[i], tn[i])

    return auroc, auprc, tpr, tnr, fp, ppv, thresholds, npv
