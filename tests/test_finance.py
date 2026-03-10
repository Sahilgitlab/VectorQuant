"""
VectorQuant — Finance Module Tests

Tests for portfolio optimization, risk models, derivatives, and stochastic.
"""

import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq


# ─── Portfolio ───────────────────────────────────────────────────────────────

def test_portfolio_return():
    weights = [0.5, 0.5]
    expected_returns = [0.10, 0.20]
    ret = vq.portfolio.portfolio_return(weights, expected_returns)
    assert abs(ret - 0.15) < 1e-8


def test_portfolio_variance():
    weights = [0.5, 0.5]
    cov = [[0.04, 0.01], [0.01, 0.09]]
    var = vq.portfolio.portfolio_variance(weights, cov)
    # w^T * Cov * w = 0.25*0.04 + 2*0.25*0.01 + 0.25*0.09 = 0.0375
    assert abs(var - 0.0375) < 1e-8


def test_optimize_max_sharpe_weights_sum_to_one():
    expected_returns = [0.12, 0.10, 0.07]
    cov = [[0.04, 0.006, 0.002],
           [0.006, 0.025, 0.004],
           [0.002, 0.004, 0.01]]
    weights = vq.portfolio.optimize_max_sharpe(expected_returns, cov)
    assert abs(sum(weights) - 1.0) < 1e-6
    assert all(w >= 0 for w in weights)


# ─── Risk Models ─────────────────────────────────────────────────────────────

def test_parametric_var_positive():
    returns = [-0.02, 0.01, -0.03, 0.005, -0.01, 0.02, -0.015, 0.01]
    var = vq.risk.parametric_var(returns, 0.95)
    assert var > 0  # VaR should be positive (loss amount)


def test_cvar_greater_than_var():
    returns = [-0.02, 0.01, -0.03, 0.005, -0.01, 0.02, -0.015, 0.01, -0.04, 0.005]
    var = vq.risk.historical_var(returns, 0.95)
    c_var = vq.risk.cvar(returns, 0.95)
    assert c_var >= var  # CVaR should be >= VaR


# ─── Derivatives ─────────────────────────────────────────────────────────────

def test_black_scholes_put_call_parity():
    """C - P = S - K * exp(-rT)"""
    S, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
    call = vq.derivatives.black_scholes_call(S, K, r, sigma, T)
    put = vq.derivatives.black_scholes_put(S, K, r, sigma, T)
    parity = call - put - (S - K * math.exp(-r * T))
    assert abs(parity) < 1e-6


def test_bs_greeks_sanity():
    S, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
    delta = vq.derivatives.bs_delta(S, K, r, sigma, T, 'call')
    assert 0 < delta < 1  # Call delta should be in (0, 1)

    gamma = vq.derivatives.bs_gamma(S, K, r, sigma, T)
    assert gamma > 0  # Gamma is always positive

    vega = vq.derivatives.bs_vega(S, K, r, sigma, T)
    assert vega > 0  # Vega is always positive


# ─── Stochastic ──────────────────────────────────────────────────────────────

def test_gbm_correct_shape():
    vq.prob.set_seed(42)
    paths = vq.stochastic.simulate_geometric_brownian_motion(
        S0=100, mu=0.05, sigma=0.2, T=1.0, dt=0.01, n_paths=10
    )
    assert len(paths) == 10
    assert len(paths[0]) == 101  # 1/0.01 + 1 = 101 steps


def test_gbm_starts_at_S0():
    vq.prob.set_seed(42)
    paths = vq.stochastic.simulate_geometric_brownian_motion(
        S0=100, mu=0.05, sigma=0.2, T=1.0, dt=0.01, n_paths=5
    )
    for path in paths:
        assert path[0] == 100


def test_monte_carlo_european_call():
    """MC European call should approximate Black-Scholes for large N."""
    vq.prob.set_seed(42)
    mc = vq.stochastic.MonteCarloEngine(n_paths=5000)
    mc_price, se = mc.european_call(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)
    bs_price = vq.derivatives.black_scholes_call(100, 100, 0.05, 0.2, 1.0)
    # Allow reasonable tolerance (MC has variance)
    assert abs(mc_price - bs_price) < 3.0  # Within $3


# ─── Time Series ─────────────────────────────────────────────────────────────

def test_sma():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = vq.timeseries.sma(data, 3)
    assert abs(result[0] - 2.0) < 1e-8  # (1+2+3)/3
    assert abs(result[-1] - 9.0) < 1e-8  # (8+9+10)/3


def test_ema():
    data = [1, 2, 3, 4, 5]
    result = vq.timeseries.ema(data, 3)
    assert len(result) == 3  # Only from position n-1 onward


# ─── AI Layer ────────────────────────────────────────────────────────────────

def test_score_strategy():
    score = vq.ai.score_strategy(sharpe=1.5, stability_prob=0.9, liquidity_score=1.0)
    assert abs(score - 1.35) < 1e-8


def test_score_strategy_negative_sharpe():
    score = vq.ai.score_strategy(sharpe=-0.5, stability_prob=0.9, liquidity_score=1.0)
    assert score == 0.0


def test_strategy_lifecycle():
    sl = vq.ai.StrategyLifecycle("S001", "TestStrategy")
    assert sl.state == vq.ai.LifecycleState.RESEARCH
    sl.evaluate_promotion(sharpe_ratio=1.5, periods_active=100, max_drawdown=0.10)
    assert sl.state == vq.ai.LifecycleState.PAPER_TRADING


