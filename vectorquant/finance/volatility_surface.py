"""
Volatility Surfaces & Implied Volatility
"""
from vectorquant.finance.derivatives import black_scholes_call, bs_vega
import math

def implied_volatility_call(S, K, r, T, market_price, initial_sigma=0.2, max_iter=100, tol=1e-5):
    """
    Newton-Raphson solver for implied volatility of a European Call option.
    """
    sigma = initial_sigma
    for _ in range(max_iter):
        price = black_scholes_call(S, K, r, sigma, T)
        diff = price - market_price
        
        if abs(diff) < tol:
            return sigma
            
        vega = bs_vega(S, K, r, sigma, T)
        if vega < 1e-8:
            # Drop back to bisection if Newton fails (simplified here)
            return sigma
            
        sigma = sigma - diff / vega
        
        # Keep positive
        if sigma <= 0.0:
            sigma = 1e-5
            
    return sigma

def interpolate_volatility_surface_2d(surface_points, target_K, target_T):
    """
    Basic bilinear interpolation for a volatility surface.
    surface_points: list of dicts [{'K': k, 'T': t, 'vol': v}, ...]
    target_K: target strike
    target_T: target maturity
    """
    if not surface_points: return 0.0
    
    # Very simplified nearest neighbor weightings (Inverse Distance Weighting)
    numerator = 0.0
    denominator = 0.0
    
    for point in surface_points:
        dk = point['K'] - target_K
        dt = point['T'] - target_T
        dist = math.sqrt(dk**2 + dt**2)
        
        if dist < 1e-6:
            return point['vol']
            
        weight = 1.0 / dist
        numerator += weight * point['vol']
        denominator += weight
        
    return numerator / denominator
