"""
VectorQuant — Public API Tests

Verifies that `import vectorquant as vq` works and all
namespace paths are accessible.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq


# ─── Public API accessibility ────────────────────────────────────────────────

def test_version():
    assert hasattr(vq, "__version__")
    assert vq.__version__ == "0.5.0"


def test_linalg_namespace():
    assert hasattr(vq, "linalg")
    assert callable(vq.linalg.matrix_multiply)
    assert callable(vq.linalg.determinant)
    assert callable(vq.linalg.svd)


def test_stats_namespace():
    assert hasattr(vq, "stats")
    assert callable(vq.stats.mean)
    assert callable(vq.stats.variance)
    assert callable(vq.stats.linear_regression)


def test_prob_namespace():
    assert hasattr(vq, "prob")
    assert callable(vq.prob.normal_pdf)
    assert callable(vq.prob.normal_cdf)
    assert callable(vq.prob.rnorm)


def test_stochastic_namespace():
    assert hasattr(vq, "stochastic")
    assert callable(vq.stochastic.simulate_geometric_brownian_motion)
    assert callable(vq.stochastic.MonteCarloEngine)


def test_timeseries_namespace():
    assert hasattr(vq, "timeseries")
    assert callable(vq.timeseries.sma)
    assert callable(vq.timeseries.ema)
    assert callable(vq.timeseries.viterbi_algorithm_hmm)


def test_portfolio_namespace():
    assert hasattr(vq, "portfolio")
    assert callable(vq.portfolio.optimize_max_sharpe)
    assert callable(vq.portfolio.portfolio_return)


def test_risk_namespace():
    assert hasattr(vq, "risk")
    assert callable(vq.risk.parametric_var)
    assert callable(vq.risk.cvar)


def test_derivatives_namespace():
    assert hasattr(vq, "derivatives")
    assert callable(vq.derivatives.black_scholes_call)
    assert callable(vq.derivatives.bs_delta)


def test_research_namespace():
    assert hasattr(vq, "research")
    assert callable(vq.research.probabilistic_sharpe_ratio)


def test_ai_namespace():
    assert hasattr(vq, "ai")
    assert callable(vq.ai.score_strategy)
    assert callable(vq.ai.explain_decision)


def test_infra_namespace():
    assert hasattr(vq, "infra")
    assert callable(vq.infra.forward_fill_missing)



