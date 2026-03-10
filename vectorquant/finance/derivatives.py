"""
Derivatives Mathematics (Option Pricing & Greeks)
"""
import math
from vectorquant.core.probability import normal_cdf, normal_pdf

def d1_d2(S, K, r, sigma, T):
    if T <= 0:
        return 0, 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def black_scholes_call(S, K, r, sigma, T):
    if T <= 0:
        return max(0.0, S - K)
    d1, d2 = d1_d2(S, K, r, sigma, T)
    return S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)

def black_scholes_put(S, K, r, sigma, T):
    if T <= 0:
        return max(0.0, K - S)
    d1, d2 = d1_d2(S, K, r, sigma, T)
    return K * math.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)

def bs_delta(S, K, r, sigma, T, option_type='call'):
    d1, _ = d1_d2(S, K, r, sigma, T)
    if option_type == 'call':
        return normal_cdf(d1)
    else:
        return normal_cdf(d1) - 1.0

def bs_gamma(S, K, r, sigma, T):
    d1, _ = d1_d2(S, K, r, sigma, T)
    return normal_pdf(d1) / (S * sigma * math.sqrt(T))

def bs_theta(S, K, r, sigma, T, option_type='call'):
    d1, d2 = d1_d2(S, K, r, sigma, T)
    term1 = -(S * normal_pdf(d1) * sigma) / (2 * math.sqrt(T))
    if option_type == 'call':
        term2 = -r * K * math.exp(-r * T) * normal_cdf(d2)
    else:
        term2 = r * K * math.exp(-r * T) * normal_cdf(-d2)
    return term1 + term2

def bs_vega(S, K, r, sigma, T):
    d1, _ = d1_d2(S, K, r, sigma, T)
    return S * normal_pdf(d1) * math.sqrt(T)

def bs_rho(S, K, r, sigma, T, option_type='call'):
    _, d2 = d1_d2(S, K, r, sigma, T)
    if option_type == 'call':
        return K * T * math.exp(-r * T) * normal_cdf(d2)
    else:
        return -K * T * math.exp(-r * T) * normal_cdf(-d2)
