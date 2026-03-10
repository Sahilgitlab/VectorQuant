"""
VectorQuant — High-Performance Quantitative Finance Engine for Python
=====================================================================

Ultra-fast quantitative finance research engine combining mathematical
kernel, financial modeling, research pipelines, and AI strategy discovery.

Usage::

    import vectorquant as vq

    # Statistics
    returns = vq.stats.mean(data)

    # Risk
    var = vq.risk.parametric_var(returns)

    # Portfolio
    weights = vq.portfolio.optimize_max_sharpe(expected_returns, cov)

    # Stochastic simulation
    paths = vq.stochastic.simulate_geometric_brownian_motion(
        S0=100, mu=0.05, sigma=0.2, T=1.0, dt=1/252, n_paths=10000
    )

    # Time series
    regime = vq.timeseries.viterbi_algorithm_hmm(obs, probs, trans, emission)

    # Derivatives
    price = vq.derivatives.black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1.0)
"""

from ._version import __version__

# ─── Layer Aliases ───────────────────────────────────────────────────────────

from .core import linear_algebra as linalg
from .core import statistics as stats
from .core import probability as prob
from .core import optimization as optim

from . import stochastic
from . import time_series as timeseries
from .finance import portfolio
from .finance import risk_models as risk
from .finance import derivatives
from . import research
from . import ai
from . import infrastructure as infra

# ─── Convenience re-exports ─────────────────────────────────────────────────

from .finance import (
    RiskMonitor,
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
    "research",
    "ai",
    "infra",
]
