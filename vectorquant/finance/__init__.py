"""
VectorQuant Finance — Portfolio, Risk, and Derivatives

Provides portfolio optimization, risk models (VaR/CVaR), derivatives pricing,
covariance estimation, and factor models.
"""

from .portfolio import (
    portfolio_return, portfolio_variance, portfolio_volatility,
    optimize_max_sharpe, black_litterman_returns,
)

from .risk_models import (
    historical_var, parametric_var, monte_carlo_var, cvar,
)

from .covariance import (
    ledoit_wolf_shrinkage, ewma_covariance,
    robust_covariance_mcd_approx,
)

from .derivatives import (
    black_scholes_call, black_scholes_put,
    bs_delta, bs_gamma, bs_theta, bs_vega, bs_rho,
)

from .factor_models import (
    capm_expected_return,
    fama_french_3_factor,
    fama_french_5_factor,
    estimate_factor_betas,
)
