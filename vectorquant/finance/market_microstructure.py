"""
Market Microstructure
"""

def order_book_imbalance(bid_volume, ask_volume):
    """
    imbalance = (bid - ask) / (bid + ask)
    Positive means more buy pressure (bullish tick).
    Negative means more sell pressure.
    """
    total = bid_volume + ask_volume
    if total == 0: return 0.0
    return (bid_volume - ask_volume) / total

def kyles_lambda(price_changes, order_flows):
    """
    Measures price impact of trades.
    delta_P = lambda * order_flow
    Returns lambda via OLS (with intercept=0 assumption usually, but here simple univariate regression).
    """
    from vectorquant.core.statistics import linear_regression
    # X = [[flow] for flow in order_flows], y = price_changes
    # But usually without intercept. Let's do a strict proportional regression sum(xy)/sum(xx)
    n = len(order_flows)
    if n == 0: return 0.0
    
    sum_xy = sum(price_changes[i] * order_flows[i] for i in range(n))
    sum_xx = sum(order_flows[i]**2 for i in range(n))
    
    if sum_xx == 0: return 0.0
    return sum_xy / sum_xx

def expected_execution_cost(order_size, trade_rate, impact_lambda, spread):
    """
    Simplified expected execution cost based on linear price impact.
    Cost = (Spread/2) * Size + Lambda * trade_rate * Size
    """
    return (spread / 2.0) * order_size + impact_lambda * trade_rate * order_size

def square_root_market_impact(order_size, daily_volume, daily_volatility):
    """
    Square-root impact model.
    Impact = sigma * sqrt(Q / V)
    """
    import math
    if daily_volume <= 0: return float('inf')
    return daily_volatility * math.sqrt(order_size / daily_volume)

def almgren_chriss_optimum_trajectory(total_shares, risk_aversion, impact_lambda, volatility, periods):
    """
    Calculates an optimal execution schedule balancing price risk vs market impact.
    Returns array of subset sizes to execute at each period.
    """
    import math
    if periods <= 0: return []
    if risk_aversion <= 0 or volatility <= 0:
        # Risk neutral -> trade smoothly (TWAP)
        return [total_shares / periods] * periods
        
    # kappa = sqrt((risk_aversion * vol^2) / impact_lambda)
    if impact_lambda <= 0:
        impact_lambda = 1e-6
        
    kappa = math.sqrt((risk_aversion * volatility**2) / impact_lambda)
    
    trajectory = []
    T = periods
    
    for k in range(1, T + 1):
        # Simplified continuous approximation:
        # x_k = X * sinh(kappa * (T - k + 0.5)) / sinh(kappa * T)
        # Using exponential approximations for large kappa to avoid overflow:
        try:
            val = total_shares * (math.exp(kappa * (T - k + 0.5)) - math.exp(-kappa * (T - k + 0.5))) / \
                                 (math.exp(kappa * T) - math.exp(-kappa * T))
        except OverflowError:
            val = total_shares * math.exp(-kappa * (k - 0.5))
            
        trajectory.append(val)
        
    # Normalize to ensure sum is exactly total_shares
    s = sum(trajectory)
    if s > 0:
        trajectory = [t * (total_shares / s) for t in trajectory]
    else:
        trajectory = [total_shares / periods] * periods
        
    return trajectory
