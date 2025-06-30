# ------------------------------------------------------------------------------
# This code was originally written by Olha Shaposhnyk
# as part of a research study conducted in Biometric Technologies Laborytory,
# University of Calgary.
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def plot_poincare(rr):
    rr_n = rr[:-1]
    rr_n1 = rr[1:]
    sd1 = np.sqrt(0.5) * np.std(rr_n1 - rr_n)
    sd2 = np.sqrt(0.5) * np.std(rr_n1 + rr_n)
    mean = np.mean(rr)
    sd = np.std(rr)
    min_rr = np.min(rr)
    max_rr = np.max(rr)
    return sd1, sd2, sd, mean, min_rr, max_rr

def feature_extraction(arr):
    mean = np.mean(arr)
    sd = np.std(arr)
    min_rr = np.min(arr)
    max_rr = np.max(arr)
    return sd, mean, min_rr, max_rr

def extract_all_features(data):
    rr_feature = [plot_poincare(data['rr_filter'])]
    hr_feature = [feature_extraction(data['hr_filter'])]
    gsr_feature = [feature_extraction(data['gsr_filter'])]
    temp_feature = [feature_extraction(data['temperature_filter'])]
    
    df_rr = pd.DataFrame(rr_feature, columns=['rr_sd1','rr_sd2','rr_sd','rr_mean','rr_min','rr_max'])
    df_hr = pd.DataFrame(hr_feature, columns=['hr_sd', 'hr_mean', 'hr_min', 'hr_max'])
    df_gsr = pd.DataFrame(gsr_feature, columns=['gsr_sd', 'gsr_mean', 'gsr_min', 'gsr_max'])
    df_temp = pd.DataFrame(temp_feature, columns=['temp_sd', 'temp_mean', 'temp_min', 'temp_max'])

    result_data = pd.concat([df_rr, df_hr, df_gsr, df_temp], axis=1)
    return result_data

def combine_and_standardize(features, scaler=None, fit_scaler=False):

    if fit_scaler:
        scaler = StandardScaler().fit(features)
    features_scaled = scaler.transform(features)
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    return df_scaled, scaler
