"""
Extreme Value Theory & Information Theory
"""
import math

# Information Theory
def entropy(probabilities):
    """
    Calculates Shannon entropy: H(X) = - sum p(x) log p(x)
    """
    H = 0.0
    for p in probabilities:
        if p > 0:
            H -= p * math.log(p) # Natural log entropy (nats), use log2 for bits
    return H

def mutual_information(joint_probs, marginal_x, marginal_y):
    """
    I(X;Y) = sum_x sum_y p(x,y) log (p(x,y) / (p(x)p(y)))
    joint_probs: dict of (x,y): p
    marginal_x: dict of x: p
    marginal_y: dict of y: p
    """
    I = 0.0
    for (x, y), p_xy in joint_probs.items():
        if p_xy > 0:
            p_x = marginal_x.get(x, 0.0)
            p_y = marginal_y.get(y, 0.0)
            if p_x > 0 and p_y > 0:
                I += p_xy * math.log(p_xy / (p_x * p_y))
    return I

# Extreme Value Theory
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
