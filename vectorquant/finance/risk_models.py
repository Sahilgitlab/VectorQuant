"""
Risk Models — VaR, CVaR, Monte Carlo VaR

All functions return VQResult with proof traces and academic citations.
"""
import math
from vectorquant.core.statistics import mean, standard_deviation
from vectorquant.core.probability import normal_inv_cdf
from vectorquant.core.result import VQResult, ProofStep, _timed_result


@_timed_result
def historical_var(returns, confidence_level=0.95):
    """
    Calculates Value at Risk using historical simulation.

    Sorts returns and picks the (1-α) percentile.
    Returns VQResult where .value is the loss (positive number).
    """
    if not returns:
        return VQResult(value=0.0, formula_used="historical_var", warnings=["Empty returns"])

    sorted_returns = sorted(returns)
    idx = int((1.0 - confidence_level) * len(sorted_returns))
    var_value = -sorted_returns[max(0, idx)]

    return VQResult(
        value=var_value,
        verified=True,
        confidence=1.0,
        formula_used=f"VaR_hist = -percentile(returns, {(1-confidence_level)*100:.0f}%)",
        formula_latex=r"VaR_\alpha = -\text{percentile}(r, 1-\alpha)",
        citation="J.P. Morgan RiskMetrics Technical Document (1994)",
        proof_trace=[
            ProofStep("sort_returns", "sorted(returns)", "", sorted_returns[:5], "Sort returns ascending"),
            ProofStep("index", f"int({1-confidence_level} × {len(sorted_returns)})", "", idx, "Tail index"),
            ProofStep("var", f"-sorted_returns[{max(0, idx)}]", r"-r_{(\lfloor(1-\alpha)n\rfloor)}", var_value, "VaR as positive loss"),
        ],
        backend="python",
    )


@_timed_result
def parametric_var(returns, confidence_level=0.95):
    """
    Calculates VaR assuming normal distribution of returns.

    VaR = -(μ + z_α × σ)

    Returns VQResult where .value is the loss (positive number).
    """
    if not returns:
        return VQResult(value=0.0, formula_used="parametric_var", warnings=["Empty returns"])

    mu = mean(returns)
    sigma = standard_deviation(returns)
    z = normal_inv_cdf(1.0 - confidence_level)
    var_value = -(mu + z * sigma)

    return VQResult(
        value=var_value,
        verified=True,
        confidence=1.0,
        formula_used=f"VaR = -(μ + z_α × σ) = -({mu:.6f} + {z:.4f} × {sigma:.6f})",
        formula_latex=r"VaR_\alpha = -(\mu + z_\alpha \cdot \sigma)",
        citation="J.P. Morgan RiskMetrics Technical Document (1994)",
        proof_trace=[
            ProofStep("mean", "sum(r) / n", r"\mu = \frac{\sum r_i}{n}", mu, "Sample mean of returns"),
            ProofStep("std", "sqrt(var(r))", r"\sigma = \sqrt{\frac{\sum(r_i - \mu)^2}{n-1}}", sigma, "Sample standard deviation"),
            ProofStep("z_score", f"Φ⁻¹({1-confidence_level:.4f})", r"z_\alpha = \Phi^{-1}(1-\alpha)", z, "Normal inverse CDF"),
            ProofStep("var", "-(μ + z × σ)", r"-(μ + z_α σ)", var_value, "VaR as positive loss"),
        ],
        backend="python",
    )


@_timed_result
def monte_carlo_var(simulated_returns, confidence_level=0.95):
    """
    Calculates VaR given an array of Monte Carlo simulated final returns.
    Delegates to historical_var on the simulated distribution.
    """
    return historical_var(simulated_returns, confidence_level)


@_timed_result
def cvar(returns, confidence_level=0.95):
    """
    Conditional Value at Risk (Expected Shortfall).

    Expected loss given that the loss exceeds VaR.
    CVaR = -E[r | r ≤ -VaR]

    Returns VQResult where .value is the CVaR (positive number).
    """
    if not returns:
        return VQResult(value=0.0, formula_used="cvar", warnings=["Empty returns"])

    hist_var = historical_var(returns, confidence_level)
    var_threshold = -float(hist_var)
    tail_losses = [r for r in returns if r <= var_threshold]
    if not tail_losses:
        cvar_value = -var_threshold
    else:
        cvar_value = -mean(tail_losses)

    return VQResult(
        value=cvar_value,
        verified=True,
        confidence=1.0,
        formula_used=f"CVaR = -E[r | r ≤ {var_threshold:.6f}]",
        formula_latex=r"CVaR_\alpha = -E[r \mid r \leq -VaR_\alpha]",
        citation="Rockafellar & Uryasev (2000). Optimization of Conditional Value-at-Risk. Journal of Risk.",
        proof_trace=[
            ProofStep("var_threshold", "historical_var", "", float(hist_var), "VaR threshold"),
            ProofStep("tail_count", f"count(r ≤ {var_threshold:.6f})", "", len(tail_losses), "Number of tail losses"),
            ProofStep("cvar", "-mean(tail_losses)", r"-E[r \mid r \leq -VaR]", cvar_value, "Expected shortfall"),
        ],
        backend="python",
    )
