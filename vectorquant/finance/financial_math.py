"""
Financial Mathematics Core
"""
import math
from vectorquant.core.statistics import mean, standard_deviation
from vectorquant.time_series.analysis import rolling_volatility

def simple_return(prices):
    """
    R_t = (P_t - P_{t-1}) / P_{t-1}
    """
    if len(prices) < 2: return []
    return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

def log_return(prices):
    """
    r_t = ln(P_t / P_{t-1})
    """
    if len(prices) < 2: return []
    return [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]

def cumulative_return(returns, is_log=False):
    """
    Given an array of returns, compute cumulative series (capital path).
    """
    if len(returns) == 0: return []
    cum = [1.0]
    for r in returns:
        if is_log:
            cum.append(cum[-1] * math.exp(r))
        else:
            cum.append(cum[-1] * (1.0 + r))
    return cum

def max_drawdown(capital_path):
    """
    Max Drop from peak.
    """
    if not capital_path: return 0.0
    peak = capital_path[0]
    max_dd = 0.0
    for val in capital_path:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd

def sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Sharpe = (Mean_Return - RF) / Standard_Deviation
    Returns should logically be annualized or keep in same frequency context.
    """
    if not returns: return 0.0
    excess_returns = [r - risk_free_rate for r in returns]
    m = mean(excess_returns)
    sd = standard_deviation(returns)
    if sd == 0: return 0.0
    return m / sd

def sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0):
    """
    Sortino = (Mean_Return - RF) / Downside_Deviation
    """
    if not returns: return 0.0
    excess_returns = [r - risk_free_rate for r in returns]
    m = mean(excess_returns)
    
    downside_returns = [min(0, r - target_return)**2 for r in returns]
    downside_var = sum(downside_returns) / len(returns)
    
    if downside_var == 0: return 0.0
    return m / math.sqrt(downside_var)

def calmar_ratio(returns, risk_free_rate=0.0):
    """
    Calmar = (Mean - RF) / Max_Drawdown
    """
    if not returns: return 0.0
    cum_path = cumulative_return(returns, is_log=False)
    mdd = max_drawdown(cum_path)
    excess_ret = mean([r - risk_free_rate for r in returns])
    if mdd == 0: return 0.0
    return excess_ret / mdd
