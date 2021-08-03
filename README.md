# COMPOSER™
Paper Title: Artificial Intelligence Sepsis Prediction Algorithm Learns to Say “I don’t know”!

Authors: Supreeth P. Shashikumar, Gabriel Wardi, Atul Malhotra, and Shamim Nemati

## Instructions
1) A sample dataset of 500 patients is available in the folder 'demo_data'
2) Each csv file corresponds to one ICU stay of a patient, and contains the predicted risk score from the COMPOSER model, a mask indicating the output from conformal predictor, and the ground truth label.
3) The python code 'eval_COMPOSER.py' reads in the the sample csv files from the folder 'demo_data' and computes/print the Area Under the Curve (AUC), and other performance metrics.
4) The code 'eval_COMPOSER.py' should take about 30s-1 minute to run on a normal PC.
5) Required software: 
        python 2.7+
        numpy - pip install numpy
        pandas - pip install pandas
