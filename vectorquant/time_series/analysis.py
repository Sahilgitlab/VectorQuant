"""
Time Series Mathematics
"""
import math

def sma(data, n):
    """
    Simple Moving Average
    """
    if len(data) < n:
        return []
    return [sum(data[i-n+1:i+1]) / n for i in range(n-1, len(data))]

def ema(data, n):
    """
    Exponential Moving Average
    alpha = 2 / (n + 1)
    """
    if len(data) < n:
        return []
    alpha = 2.0 / (n + 1)
    res = []
    # Initialize with SMA for first window
    curr_ema = sum(data[:n]) / n
    for i in range(n-1, len(data)):
        if i == n-1:
            res.append(curr_ema)
        else:
            curr_ema = alpha * data[i] + (1 - alpha) * curr_ema
            res.append(curr_ema)
    return res

def wma(data, n):
    """
    Weighted Moving Average (linear weights)
    """
    if len(data) < n:
        return []
    weights = [i + 1 for i in range(n)]
    w_sum = sum(weights)
    res = []
    for i in range(n-1, len(data)):
        window = data[i-n+1:i+1]
        res.append(sum(x * w for x, w in zip(window, weights)) / w_sum)
    return res

def rolling_volatility(data, n, sample=True):
    """
    Rolling standard deviation
    """
    if len(data) < n:
        return []
    res = []
    for i in range(n-1, len(data)):
        window = data[i-n+1:i+1]
        mean_val = sum(window) / n
        var = sum((x - mean_val)**2 for x in window) / (n - 1 if sample else n)
        res.append(math.sqrt(var))
    return res

def ewma_volatility(returns, lmbda=0.94):
    """
    Exponentially Weighted Moving Average Volatility
    sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r^2
    """
    if not returns: return []
    var = returns[0]**2
    vols = [math.sqrt(var)]
    for i in range(1, len(returns)):
        var = lmbda * var + (1 - lmbda) * (returns[i]**2)
        vols.append(math.sqrt(var))
    return vols

def ar_1_model(y):
    """
    Fits an AR(1) model y_t = a + b * y_{t-1} using OLS
    Returns (intercept, phi1)
    """
    if len(y) < 2: return 0.0, 0.0
    Y = y[1:]
    X = y[:-1]
    
    # OLS for y_t = alpha + beta * y_{t-1}
    n = len(Y)
    mean_X = sum(X) / n
    mean_Y = sum(Y) / n
    
    cov = sum((X[i] - mean_X) * (Y[i] - mean_Y) for i in range(n))
    var_X = sum((X[i] - mean_X)**2 for i in range(n))
    
    if var_X == 0: return mean_Y, 0.0
    beta = cov / var_X
    alpha = mean_Y - beta * mean_X
    return alpha, beta
