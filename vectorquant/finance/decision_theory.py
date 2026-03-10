"""
Agent Decision Theory & Portfolio Mathematics
"""
import math

def log_utility(wealth):
    if wealth <= 0: return float('-inf')
    return math.log(wealth)

def power_utility(wealth, gamma):
    """
    CRRA (Constant Relative Risk Aversion)
    U(W) = (W^(1-gamma) - 1) / (1-gamma) for gamma != 1
    """
    if wealth <= 0: return float('-inf')
    if gamma == 1.0: return log_utility(wealth)
    return (wealth ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

def kelly_criterion(win_prob, win_loss_ratio):
    """
    Computes optimal fraction of bankroll to wager.
    f* = (p(b+1) - 1) / b
    where p is win probability, b is net odds received on wager (win/loss ratio).
    """
    if win_loss_ratio <= 0: return 0.0
    f_star = (win_prob * (win_loss_ratio + 1.0) - 1.0) / win_loss_ratio
    return max(0.0, min(f_star, 1.0)) # Bounded between 0 and 100%

def kelly_continuous(expected_excess_return, variance):
    """
    Continuous Kelly approximation for financial assets.
    f* = (mu - r) / sigma^2
    """
    if variance <= 0: return 0.0
    return max(0.0, expected_excess_return / variance)
