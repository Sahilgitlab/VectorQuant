"""
Extreme Value Theory
"""
import math

def gev_cdf(x, mu=0.0, sigma=1.0, xi=0.0):
    """
    Generalized Extreme Value Distribution CDF
    xi: shape parameter
    mu: location
    sigma: scale
    """
    if xi == 0:
        # Gumbel
        return math.exp(-math.exp(-(x - mu) / sigma))
    
    # Frechet / Weibull
    t = 1 + xi * ((x - mu) / sigma)
    if t <= 0:
        return 0.0 if xi > 0 else 1.0 # Depends on tail
        
    return math.exp(-(t ** (-1.0 / xi)))

def peaks_over_threshold(data, threshold):
    """
    Extracts excesses over a threshold.
    Returns the excesses data points.
    """
    return [x - threshold for x in data if x > threshold]
