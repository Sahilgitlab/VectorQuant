"""
Derivatives Mathematics (Option Pricing & Greeks)

All pricing functions return VQResult with proof traces.
Greeks continue to return raw floats (used as intermediate calculations).
"""
import math
from vectorquant.core.probability import normal_cdf, normal_pdf
from vectorquant.core.result import VQResult, ProofStep, _timed_result


def d1_d2(S, K, r, sigma, T):
    if T <= 0:
        return 0, 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


@_timed_result
def black_scholes_call(S, K, r, sigma, T):
    """Black-Scholes European call option price. Returns VQResult."""
    if T <= 0:
        return VQResult(
            value=max(0.0, S - K),
            formula_used="max(S - K, 0) [at expiry]",
            citation="Black & Scholes (1973)",
        )
    d1, d2 = d1_d2(S, K, r, sigma, T)
    call_price = S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)

    return VQResult(
        value=call_price,
        verified=True,
        confidence=1.0,
        formula_used=f"C = S·N(d1) - K·e^(-rT)·N(d2), d1={d1:.4f}, d2={d2:.4f}",
        formula_latex=r"C = S \Phi(d_1) - K e^{-rT} \Phi(d_2)",
        citation="Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy, 81(3), 637–654.",
        proof_trace=[
            ProofStep("d1", "[ln(S/K) + (r + σ²/2)T] / (σ√T)", r"d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}", d1, f"S={S}, K={K}, r={r}, σ={sigma}, T={T}"),
            ProofStep("d2", "d1 - σ√T", r"d_2 = d_1 - \sigma\sqrt{T}", d2, f"d1 - {sigma}×√{T}"),
            ProofStep("call", "S·N(d1) - K·e^(-rT)·N(d2)", r"S\Phi(d_1) - Ke^{-rT}\Phi(d_2)", call_price, "European call price"),
        ],
        backend="python",
    )


@_timed_result
def black_scholes_put(S, K, r, sigma, T):
    """Black-Scholes European put option price. Returns VQResult."""
    if T <= 0:
        return VQResult(
            value=max(0.0, K - S),
            formula_used="max(K - S, 0) [at expiry]",
            citation="Black & Scholes (1973)",
        )
    d1, d2 = d1_d2(S, K, r, sigma, T)
    put_price = K * math.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)

    return VQResult(
        value=put_price,
        verified=True,
        confidence=1.0,
        formula_used=f"P = K·e^(-rT)·N(-d2) - S·N(-d1), d1={d1:.4f}, d2={d2:.4f}",
        formula_latex=r"P = K e^{-rT} \Phi(-d_2) - S \Phi(-d_1)",
        citation="Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy, 81(3), 637–654.",
        proof_trace=[
            ProofStep("d1", "[ln(S/K) + (r + σ²/2)T] / (σ√T)", r"d_1", d1, f"S={S}, K={K}, r={r}, σ={sigma}, T={T}"),
            ProofStep("d2", "d1 - σ√T", r"d_2", d2, f"d1 - {sigma}×√{T}"),
            ProofStep("put", "K·e^(-rT)·N(-d2) - S·N(-d1)", r"Ke^{-rT}\Phi(-d_2) - S\Phi(-d_1)", put_price, "European put price"),
        ],
        backend="python",
    )


# ─── Greeks (return raw floats — used as intermediate calculations) ──────

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
