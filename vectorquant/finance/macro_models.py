"""
Macroeconomic Models
"""

def yield_curve_slope(long_yield, short_yield):
    """
    10y - 2y spread.
    If < 0, recession signal.
    """
    return long_yield - short_yield

def is_recession_signal(long_yield, short_yield):
    return yield_curve_slope(long_yield, short_yield) < 0

def inflation_trend(current_cpi, past_cpi, n_years=1):
    if past_cpi == 0: return 0.0
    return (current_cpi - past_cpi) / past_cpi / n_years
