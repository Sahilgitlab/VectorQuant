"""
VectorQuant — Verified Quantitative Finance for AI Agents
==========================================================

The mathematical ground-truth layer for AI agents operating in quantitative
finance. Catches LLM hallucinations in four failure modes: wrong formula,
wrong convention, wrong units, and wrong distribution.

Usage::

    import vectorquant as vq

    # AI Verification (core purpose)
    result = vq.ai.verify_numeric("sharpe_ratio", llm_value=2.45,
                                   inputs={"returns": [...], "risk_free_rate": 0.02/252})
    result = vq.ai.conventions.lookup("treasury_bill", "USD")
    result = vq.ai.unit_checker.check(0.003, "sharpe_ratio")

    # Risk
    var = vq.risk.parametric_var(returns)

    # Portfolio
    weights = vq.portfolio.optimize_max_sharpe(expected_returns, cov)

    # Derivatives
    price = vq.derivatives.black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1.0)

    # Distributions
    t = vq.distributions.StudentT(df=5)
    t.var(confidence=0.99)
"""

from ._version import __version__

# ─── Core math layer ─────────────────────────────────────────────────────────

from .core import linear_algebra as linalg
from .core import statistics as stats
from .core import probability as prob
from .core import optimization as optim

# ─── Finance layer ───────────────────────────────────────────────────────────

from . import stochastic
from . import time_series as timeseries
from .finance import portfolio
from .finance import risk_models as risk
from .finance import derivatives

# ─── AI verification layer ───────────────────────────────────────────────────

from . import ai
from . import distributions

# ─── Convenience re-exports ─────────────────────────────────────────────────

from .finance import (
    black_scholes_call, black_scholes_put,
    historical_var, parametric_var, cvar,
    optimize_max_sharpe, black_litterman_returns,
)

# ─── Public API ──────────────────────────────────────────────────────────────

__all__ = [
    "__version__",
    "linalg",
    "stats",
    "prob",
    "optim",
    "stochastic",
    "timeseries",
    "portfolio",
    "risk",
    "derivatives",
    "ai",
    "distributions",
]
