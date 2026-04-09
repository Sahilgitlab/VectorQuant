"""
Generalised Pareto Distribution (GPD) — Extreme Value / Tail Fitting

Used for Peaks-over-Threshold (POT) method in tail risk modelling.
When you care about the 99.9th percentile and beyond, GPD is the correct model.
Normal and even Student's t dramatically understate extreme losses.

Reference:
    Pickands, J. (1975). Statistical Inference Using Extreme Order Statistics.
    Annals of Statistics, 3(1), 119–131.
"""

from __future__ import annotations

import math

from vectorquant.core.result import VQResult, ProofStep
from vectorquant.distributions._math_utils import brent, aic, bic

CITATION = (
    "Pickands, J. (1975). Statistical Inference Using Extreme Order Statistics. "
    "Annals of Statistics, 3(1), 119–131. "
    "McNeil, A.J. et al. (2005). Quantitative Risk Management. Princeton University Press."
)


class GPD:
    """
    Generalised Pareto Distribution for extreme value / tail modelling.

    F(x) = 1 - (1 + ξx/β)^(-1/ξ)   for ξ ≠ 0
    F(x) = 1 - exp(-x/β)             for ξ = 0  (Exponential)

    Parameters:
        xi:    Shape parameter (ξ). ξ > 0 = heavy tail (Pareto),
               ξ = 0 = exponential tail, ξ < 0 = bounded tail.
               Typical financial returns: ξ = 0.1–0.5.
        beta:  Scale parameter (β > 0).
        mu:    Threshold (location). Default 0.

    Usage:
        gpd = GPD(xi=0.3, beta=0.02)
        gpd.fit(returns, threshold=0.02)  # Fit to exceedances above 2%
        tail_var = gpd.var(confidence=0.999)
    """

    def __init__(self, xi: float, beta: float, mu: float = 0.0):
        if beta <= 0:
            raise ValueError(f"Scale beta must be positive, got {beta}")
        self.xi = float(xi)
        self.beta = float(beta)
        self.mu = float(mu)

    def pdf(self, x: float) -> float:
        """PDF of GPD."""
        y = (x - self.mu) / self.beta
        if y < 0:
            return 0.0
        if abs(self.xi) < 1e-10:
            return math.exp(-y) / self.beta
        base = 1 + self.xi * y
        if base <= 0:
            return 0.0
        return base ** (-(1 / self.xi + 1)) / self.beta

    def logpdf(self, x: float) -> float:
        """Log PDF — numerically stable."""
        y = (x - self.mu) / self.beta
        if y < 0:
            return -math.inf
        if abs(self.xi) < 1e-10:
            return -y - math.log(self.beta)
        base = 1 + self.xi * y
        if base <= 0:
            return -math.inf
        return -(1 / self.xi + 1) * math.log(base) - math.log(self.beta)

    def cdf(self, x: float) -> float:
        """CDF of GPD."""
        y = (x - self.mu) / self.beta
        if y < 0:
            return 0.0
        if abs(self.xi) < 1e-10:
            return 1.0 - math.exp(-y)
        base = 1 + self.xi * y
        if base <= 0:
            return 1.0
        return 1.0 - base ** (-1 / self.xi)

    def ppf(self, q: float) -> float:
        """Quantile (inverse CDF)."""
        if not 0.0 <= q < 1.0:
            raise ValueError(f"q must be in [0, 1), got {q}")
        if abs(self.xi) < 1e-10:
            return self.mu - self.beta * math.log(1.0 - q)
        return self.mu + self.beta / self.xi * ((1.0 - q) ** (-self.xi) - 1.0)

    def var(self, confidence: float = 0.99, n_total: int = None,
            n_tail: int = None) -> VQResult:
        """
        VaR using Peaks-over-Threshold formula.

        If n_total and n_tail are provided (total observations and exceedances
        above the threshold), the formula is:
            VaR = threshold + (β/ξ) × [(n/n_u × (1-α))^(-ξ) - 1]

        Otherwise returns the unconditional GPD quantile.
        """
        alpha = 1.0 - confidence

        if n_total is not None and n_tail is not None and n_tail > 0:
            # Full POT formula
            ratio = n_total / n_tail * alpha
            if abs(self.xi) < 1e-10:
                var_value = self.mu - self.beta * math.log(ratio)
            else:
                var_value = self.mu + self.beta / self.xi * (ratio ** (-self.xi) - 1.0)
            formula = (f"VaR_POT = threshold + β/ξ × [(n/n_u × α)^(-ξ) - 1], "
                       f"n={n_total}, n_u={n_tail}")
        else:
            var_value = self.ppf(1.0 - alpha)
            formula = "VaR_GPD = GPD quantile at 1-α"

        return VQResult(
            value=round(var_value, 8),
            verified=True,
            confidence=1.0,
            formula_used=formula,
            formula_latex=(
                r"VaR_{POT} = \mu + \frac{\beta}{\xi}"
                r"\left[\left(\frac{n}{n_u}(1-\alpha)\right)^{-\xi} - 1\right]"
            ),
            citation=CITATION,
            proof_trace=[
                ProofStep("xi", "GPD shape", r"\xi", self.xi,
                          f"Shape parameter — heavy tail: xi={self.xi:.3f}"),
                ProofStep("beta", "GPD scale", r"\beta", self.beta,
                          f"Scale parameter: beta={self.beta:.6g}"),
                ProofStep("threshold", "Location / threshold", r"\mu", self.mu,
                          f"Exceedance threshold: {self.mu:.6g}"),
                ProofStep("var", formula, "", var_value, "VaR from GPD"),
            ],
            backend="python",
        )

    def cvar(self, confidence: float = 0.99) -> VQResult:
        """
        Expected Shortfall (CVaR) from GPD.

        ES = VaR/(1-ξ) + (β - ξ×threshold)/(1-ξ)

        Valid for ξ < 1 (which covers all practical cases).
        """
        if self.xi >= 1.0:
            return VQResult(
                value=float("inf"),
                verified=True,
                confidence=1.0,
                formula_used="CVaR_GPD = ∞ for ξ ≥ 1",
                warnings=["Expected Shortfall is infinite for GPD with ξ ≥ 1"],
            )

        var_result = self.var(confidence)
        var_val = float(var_result.value)
        es = (var_val + self.beta - self.xi * self.mu) / (1.0 - self.xi)

        return VQResult(
            value=round(es, 8),
            verified=True,
            confidence=1.0,
            formula_used="ES_GPD = (VaR + β - ξ×μ) / (1 - ξ)",
            formula_latex=(
                r"ES_\alpha = \frac{VaR_\alpha + \beta - \xi\mu}{1-\xi}"
            ),
            citation=CITATION,
            proof_trace=[
                ProofStep("var", "GPD VaR", "", var_val, f"VaR at {confidence:.2%}"),
                ProofStep("cvar", "(VaR + β - ξμ)/(1-ξ)", "", es, "GPD Expected Shortfall"),
            ],
            backend="python",
        )

    @classmethod
    def fit(cls, data: list[float], threshold: float = 0.0) -> "GPD":
        """
        Fit GPD to exceedances above threshold (Peaks-over-Threshold method).

        Args:
            data:      Full return series (losses as positive numbers, or raw returns)
            threshold: Only fit to values > threshold

        Returns:
            GPD instance with fitted (xi, beta, mu=threshold).
            Access ._fit_stats for AIC, BIC.
        """
        exceedances = [x - threshold for x in data if x > threshold]
        if len(exceedances) < 5:
            raise ValueError(
                f"Only {len(exceedances)} exceedances above threshold={threshold}. "
                "Need at least 5 for reliable GPD fitting."
            )

        n = len(exceedances)
        # Method of moments initial estimate
        mu_e = sum(exceedances) / n
        var_e = sum((x - mu_e) ** 2 for x in exceedances) / (n - 1)

        # MOM: xi = 0.5 × (1 - mu²/var), beta = 0.5 × mu × (1 + mu²/var)
        if var_e > 0 and mu_e > 0:
            xi_init = 0.5 * (1 - mu_e ** 2 / var_e)
            beta_init = 0.5 * mu_e * (1 + mu_e ** 2 / var_e)
        else:
            xi_init = 0.1
            beta_init = mu_e if mu_e > 0 else 0.01

        beta_init = max(1e-8, beta_init)

        def neg_ll(params):
            xi_, log_beta = params
            beta_ = math.exp(log_beta)
            try:
                g = cls(xi=xi_, beta=beta_, mu=0.0)
                return -sum(g.logpdf(x) for x in exceedances)
            except Exception:
                return 1e10

        # Simple coordinate descent optimisation
        xi = xi_init
        log_beta = math.log(beta_init)
        step = 0.1
        for _ in range(200):
            current = neg_ll([xi, log_beta])
            best_xi, best_lb = xi, log_beta
            improved = False
            for dxi in [-step, step]:
                for dlb in [-step, step]:
                    cand = neg_ll([xi + dxi, log_beta + dlb])
                    if cand < current:
                        current = cand
                        best_xi = xi + dxi
                        best_lb = log_beta + dlb
                        improved = True
            xi, log_beta = best_xi, best_lb
            if not improved:
                step *= 0.5
            if step < 1e-8:
                break

        beta = math.exp(log_beta)
        fitted = cls(xi=xi, beta=beta, mu=threshold)
        ll = sum(fitted.logpdf(x - threshold) for x in data if x > threshold)
        fitted._fit_stats = {
            "log_likelihood": ll,
            "aic": aic(ll, 2),
            "bic": bic(ll, 2, n),
            "n_exceedances": n,
            "threshold": threshold,
        }
        return fitted

    def __repr__(self) -> str:
        return f"GPD(xi={self.xi:.4f}, beta={self.beta:.6g}, mu={self.mu:.6g})"
