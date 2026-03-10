"""
Feature Engineering
"""
from vectorquant.core.statistics import mean, standard_deviation
from vectorquant.time_series.analysis import rolling_volatility

def cross_sectional_zscore(cross_section):
    """
    Normalizes a list of values at a single point in time across assets.
    """
    m = mean(cross_section)
    s = standard_deviation(cross_section)
    if s == 0: return [0.0] * len(cross_section)
    return [(v - m) / s for v in cross_section]

def rank_cross_section(cross_section):
    """
    Returns the percent rank [0, 1] of each asset in the cross section.
    """
    n = len(cross_section)
    if n == 0: return []
    
    indexed = [(val, i) for i, val in enumerate(cross_section)]
    indexed.sort(key=lambda x: x[0])
    
    ranks = [0.0] * n
    for rank_idx, (_, orig_idx) in enumerate(indexed):
        ranks[orig_idx] = rank_idx / (n - 1) if n > 1 else 1.0
        
    return ranks

def volatility_scaled_signal(signal_series, returns_series, lookback=21, target_vol=0.10):
    """
    Scales a raw signal inversely to recent volatility.
    """
    vols = rolling_volatility(returns_series, lookback)
    
    scaled_signal = [0.0] * len(signal_series)
    for i in range(len(signal_series)):
        if i < lookback - 1:
            scaled_signal[i] = signal_series[i] # Untouched if no vol data
        else:
            vol = vols[i - (lookback - 1)]
            if vol > 1e-6:
                # Assuming daily return vol provided, annualized target_vol
                import math
                ann_vol = vol * math.sqrt(252) 
                scalar = target_vol / ann_vol
                scaled_signal[i] = signal_series[i] * scalar
            else:
                scaled_signal[i] = 0.0
                
    return scaled_signal
