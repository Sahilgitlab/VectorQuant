"""
Financial Data Quality
"""
from vectorquant.core.statistics import mean, standard_deviation
import math

def outlier_detection_zscore(time_series, sigma_threshold=10.0):
    """
    Detects extreme physical outliers using simple Z-score.
    Returns indices of clean data and indices of outliers.
    """
    if len(time_series) < 2: return list(range(len(time_series))), []
    
    m = mean(time_series)
    s = standard_deviation(time_series)
    
    if s == 0: return list(range(len(time_series))), []
    
    clean = []
    outliers = []
    
    for i, x in enumerate(time_series):
        if abs(x - m) / s > sigma_threshold:
            outliers.append(i)
        else:
            clean.append(i)
            
    return clean, outliers

def forward_fill_missing(time_series):
    """
    Fills None values with the last observed non-None value.
    """
    filled = []
    last_valid = None
    
    for x in time_series:
        if x is not None and not math.isnan(x):
            last_valid = x
            filled.append(x)
        else:
            if last_valid is not None:
                filled.append(last_valid)
            else:
                filled.append(0.0) # Assume 0 if starts with NaN depending on context
                
    return filled
