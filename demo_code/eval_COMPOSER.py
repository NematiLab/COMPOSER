# Code accompanying the manuscript 'COMPOSER - Development and Validation of a Generalizable Model for Early Prediction of Sepsis'
# Copyright 2021
# Supreeth P. Shashikumar <spshashikumar@health.ucsd.edu>
# Shamim Nemati <snemati@health.ucsd.edu>
# Revision: 1.1 Date: 08/03/2021

import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join

import compute_snoozed_auc as snoozed_auc

# Directory that holds the sample files
baseDir = './demo_data/'

# Get the names of all the files in baseDir
fileList = [f for f in listdir(baseDir) if isfile(join(baseDir, f))]

N = len(fileList)

# Initialize the mask variable that will store the output of conformal prediction
# Mask of 0 corresponds to rejection by the conformal predictor
includeMaskPerPatient = 100*np.ones((N, 340), dtype = np.int32)
# Output risk score
SepsisPredScore = np.zeros((N, 340))
# Ground truth label
outcomePerPatient = np.zeros((N, 340))
# Record length
seqLength = np.zeros((N), dtype = np.int32)

# Loop through every .csv file and load data
for i in range(N):
    data_csv = (pd.read_csv(baseDir + fileList[i], header = None)).values
    lengthRecord = data_csv.shape[0]
    SepsisPredScore[i, 0:lengthRecord] = data_csv[:,0]
    includeMaskPerPatient[i, 0:lengthRecord] = data_csv[:,1]
    outcomePerPatient[i, 0:lengthRecord] = data_csv[:,2]
    seqLength[i] = lengthRecord

pTestClass = np.zeros((N))
for i in range(N):
    pTestClass[i] = int(np.any(outcomePerPatient[i]))

sepIndx = np.where(pTestClass==1)[0]

# Compute the AUC
auc, aucpr, tpr, tnr, fp, ppv, T, npv = snoozed_auc.compute_snoozed_auc(outcomePerPatient, SepsisPredScore, seqLength, includeMaskPerPatient) 

# Compute stats for a fixed decision threshold
threshold = 0.5617759227752686

tp = 0.
fp = 0.
fn = 0.
tn = 0.

tp_perPatient = 0.

for i in range(N):
    a, b, c, d = snoozed_auc.compute_confusion_matrix_per_patient(outcomePerPatient[i], SepsisPredScore[i], threshold, seqLength[i], includeMaskPerPatient[i])
    tp += a
    fp += b
    fn += c
    tn += d

    if a>0:
        tp_perPatient +=1

TPR = tp/(tp+fn)
TNR = tn/(fp+tn)
PPV = tp/(tp+fp)
NPV = tn/(tn+fn)
DOR = tp * tn / (fp * fn)
DOR_std = np.exp(1.96*np.sqrt(1/tp + 1/fn + 1/fp + 1/tn))

TPR_eff = tp_perPatient/(sepIndx.shape[0])

print "Total Patients: ", N, "Septic Patients: ", sepIndx.shape[0]

print "AUC: ", auc, "TPR: ", TPR, "SPC: ", TNR, "PPV: ", PPV, "NPV: ", NPV, "TPR_perPatient: ", TPR_eff, "DOR: ", DOR, "DOR_std: ", DOR_std, "#FP: ", fp
