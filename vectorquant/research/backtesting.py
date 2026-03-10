"""
Backtesting Framework
"""
from vectorquant.core.statistics import mean, standard_deviation
from vectorquant.core.probability import normal_cdf
from .event_components import (
    EventDrivenBacktester, DataHandler, Strategy, 
    Portfolio, ExecutionHandler
)
from .events import MarketEvent, SignalEvent, OrderEvent, FillEvent
import math

def apply_transaction_costs(weights_t, weights_t_minus_1, bps_fee=0.0010):
    """
    Calculate return drag due to turnover transaction costs.
    bps_fee: 10 basis points default (0.0010)
    """
    if not weights_t_minus_1:
        # Full initial allocation cost
        turnover = sum(abs(w) for w in weights_t)
    else:
        turnover = sum(abs(weights_t[i] - weights_t_minus_1[i]) for i in range(len(weights_t)))
        
    return turnover * bps_fee

def rolling_window_backtest(returns_matrix, strategy_func, window_size=252, bps_fee=0.0010):
    """
    Simulates a strategy moving through time.
    strategy_func: callable that takes historical returns up to T and returns weights for T+1.
    returns_matrix: full chronological dataset
    Returns list of realized portfolio returns out of sample.
    """
    out_of_sample_returns = []
    T = len(returns_matrix)
    n_assets = len(returns_matrix[0])
    
    prev_weights = []
    
    for t in range(window_size, T):
        # Lookback window
        window_data = returns_matrix[t - window_size : t]
        
        # Strategy decides weights
        weights = strategy_func(window_data)
        
        # Realized given next period
        next_period_returns = returns_matrix[t]
        from vectorquant.finance.portfolio import portfolio_return
        gross_ret = portfolio_return(weights, next_period_returns)
        
        # Costs
        cost = apply_transaction_costs(weights, prev_weights, bps_fee)
        
        net_ret = gross_ret - cost
        out_of_sample_returns.append(net_ret)
        
        prev_weights = weights
        
    return out_of_sample_returns

def probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0):
    """
    Bailey and Lopez de Prado (2012).
    Adjusts Sharpe Ratio for non-normality (skew/kurtosis) and sample length.
    Returns the probability that the estimated Sharpe ratio is greater than the benchmark.
    """
    from vectorquant.core.statistics import skewness, kurtosis
    n = len(returns)
    if n < 3: return 0.0
    
    sr_est = mean(returns) / standard_deviation(returns) if standard_deviation(returns) > 0 else 0.0
    sk = skewness(returns)
    ku = kurtosis(returns) # Excess kurtosis
    
    numerator = (sr_est - benchmark_sharpe) * math.sqrt(n - 1)
    denominator = math.sqrt(1 - sk * sr_est + ((ku + 2) / 4) * (sr_est ** 2))
    
    if denominator == 0: return 1.0 if sr_est > benchmark_sharpe else 0.0
    
    z = numerator / denominator
    return normal_cdf(z)

def deflated_sharpe_ratio(prob_sharpe_ratio, num_trials=100, expected_variance=1.0):
    """
    Accounts for multiple testing bias.
    If multiple strategies are tested, the maximum expected Sharpe ratio increases.
    Returns the DSR (simplified probability adjustment relative to max expected SR).
    """
    # Simplified DSR approximation: adjust benchmark Sharpe via expected max
    # E[max(X)] approx sqrt(2 * ln(N)) for normal variables
    euler_mascheroni = 0.5772
    max_expected_sr = math.sqrt(2 * math.log(num_trials)) + euler_mascheroni / math.sqrt(2 * math.log(num_trials))
    
    # We would theoretically plug `max_expected_sr` back into PSR as the new benchmark.
    # We return the adjusted benchmark threshold for the user to compare.
    return max_expected_sr
