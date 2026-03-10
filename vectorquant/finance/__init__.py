"""
VectorQuant Finance — Financial Theory & Risk Management

Provides portfolio optimization, risk models (VaR/CVaR), derivatives pricing,
covariance estimation, stress testing, risk attribution, and market microstructure.
"""

from .portfolio import (
    portfolio_return, portfolio_variance, portfolio_volatility,
    optimize_max_sharpe, black_litterman_returns,
)

from .risk_parity import (
    hrp_recursive_bisection,
)

from .risk_models import (
    historical_var, parametric_var, monte_carlo_var, cvar,
)

from .risk_attribution import (
    marginal_contribution_to_risk,
    risk_contribution,
    relative_risk_contribution,
    factor_risk_attribution,
)

from .risk_monitoring import RiskMonitor

from .covariance import (
    ledoit_wolf_shrinkage, ewma_covariance,
    robust_covariance_mcd_approx,
)

from .derivatives import (
    black_scholes_call, black_scholes_put,
    bs_delta, bs_gamma, bs_theta, bs_vega, bs_rho,
)

from .volatility_surface import (
    implied_volatility_call,
    interpolate_volatility_surface_2d,
)

from .stress_testing import (
    historical_stress_test, hypothetical_scenario,
    reverse_stress_test,
)

from .decision_theory import (
    log_utility, power_utility,
    kelly_criterion, kelly_continuous,
)

from .factor_models import (
    capm_expected_return,
    fama_french_3_factor,
    fama_french_5_factor,
    estimate_factor_betas,
)
