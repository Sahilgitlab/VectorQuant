"""
Model Validation & Bias Detection
"""
import math
from vectorquant.core.statistics import mean, standard_deviation
from vectorquant.research.backtesting import rolling_window_backtest

def walk_forward_validation(returns_matrix, strategy_func, train_size=252, test_size=63, bps_fee=0.0010):
    """
    Rolling train-test splits.
    Returns array of out-of-sample portfolio returns.
    """
    total_len = len(returns_matrix)
    oos_returns = []
    
    start = 0
    while start + train_size + test_size <= total_len:
        train_window = returns_matrix[start : start + train_size]
        test_window = returns_matrix[start + train_size : start + train_size + test_size]
        
        # Strategy determines weights from train, applied to test
        # We simulate a static weight vector over the test window for simplicity here
        target_weights = strategy_func(train_window)
        
        from vectorquant.finance.portfolio import portfolio_return
        for row in test_window:
            # Assume 0 cost over the static hold window after entry
            # In deep reality, cost applies on the initial rebalance
            oos_returns.append(portfolio_return(target_weights, row))
            
        start += test_size
        
    return oos_returns

def bootstrap_performance(returns, n_bootstraps=1000, seed=42):
    """
    Resamples return series with replacement to calculate mean sharpe ratio distribution.
    Returns (mean_sharpe, std_sharpe)
    """
    import random
    random.seed(seed) # standard lib is okay for utility scripts, but LCG available.
    n = len(returns)
    sharpes = []
    
    for _ in range(n_bootstraps):
        sample = [random.choice(returns) for _ in range(n)]
        m = mean(sample)
        s = standard_deviation(sample)
        sr = 0.0 if s == 0 else m / s
        sharpes.append(sr)
        
    return mean(sharpes), standard_deviation(sharpes)

def whites_reality_check(base_returns, strategy_returns_list, n_bootstraps=1000):
    """
    Checks if the best strategy out of N tested is genuinely better than the benchmark,
    or just lucky (data snooping bias).
    Returns p-value (low means it's genuinely better).
    """
    # Simplified stationary bootstrap
    import random
    n = len(base_returns)
    n_strats = len(strategy_returns_list)
    
    # Original perf differences
    base_mean = mean(base_returns)
    orig_diffs = [mean(s) - base_mean for s in strategy_returns_list]
    max_orig_diff = max(orig_diffs)
    
    # Resample
    beat_count = 0
    for _ in range(n_bootstraps):
        idx_sample = [random.randint(0, n-1) for _ in range(n)]
        
        # Centered bootstrap to force null hypothesis
        max_boot_diff = 0.0
        for i in range(n_strats):
            s_ret = strategy_returns_list[i]
            # mean of resampled strategy - mean of resampled base, centered around original
            b_diff = mean([s_ret[j] - base_returns[j] for j in idx_sample]) - orig_diffs[i]
            max_boot_diff = max(max_boot_diff, b_diff)
            
        if max_boot_diff > max_orig_diff:
            beat_count += 1
            
    return beat_count / n_bootstraps
