"""
Risk Models
"""
import math
from vectorquant.core.statistics import mean, standard_deviation
from vectorquant.core.probability import normal_inv_cdf

def historical_var(returns, confidence_level=0.95):
    """
    Calculates Value at Risk using historical simulation.
    confidence_level: typically 0.95 or 0.99
    Returns the loss value (positive number means loss).
    """
    if not returns: return 0.0
    sorted_returns = sorted(returns)
    idx = int((1.0 - confidence_level) * len(sorted_returns))
    return -sorted_returns[max(0, idx)]

def parametric_var(returns, confidence_level=0.95):
    """
    Calculates VaR assuming normal distribution of returns.
    VaR = -(mu + z * sigma)
    """
    if not returns: return 0.0
    mu = mean(returns)
    sigma = standard_deviation(returns)
    z = normal_inv_cdf(1.0 - confidence_level)
    return -(mu + z * sigma)

def monte_carlo_var(simulated_returns, confidence_level=0.95):
    """
    Calculates VaR given an array of monte carlo simulated final returns.
    """
    return historical_var(simulated_returns, confidence_level)

def cvar(returns, confidence_level=0.95):
    """
    Conditional Value at Risk (Expected Shortfall).
    Expected loss given that the loss exceeds VaR.
    """
    if not returns: return 0.0
    var_threshold = -historical_var(returns, confidence_level)
    tail_losses = [r for r in returns if r <= var_threshold]
    if not tail_losses:
        return -var_threshold
    return -mean(tail_losses)
