"""
Normal Distribution — baseline for comparison with fat-tail distributions.

Provided so users can directly compare VaR/CVaR under Normal vs Student's t.
"""

from __future__ import annotations
import math

from vectorquant.core.result import VQResult, ProofStep
from vectorquant.distributions._math_utils import (
    normal_pdf as _npdf, normal_cdf as _ncdf, normal_ppf as _nppf,
    aic, bic,
)

CITATION = (
    "Gauss, C.F. (1809). Theoria Motus Corporum Coelestium. "
    "J.P. Morgan RiskMetrics Technical Document (1994) for finance applications."
)


class Normal:
    """
    Normal (Gaussian) distribution.

    Provided as a baseline to compare against fat-tail distributions.
    For most financial return series, Student's t (ν=3–6) fits significantly
    better — Normal VaR understates risk at high confidence levels.

    Parameters:
        mu:    Mean (default 0)
        sigma: Standard deviation (default 1)
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.mu = float(mu)
        self.sigma = float(sigma)

    def pdf(self, x: float) -> float:
        return _npdf(x, self.mu, self.sigma)

    def cdf(self, x: float) -> float:
        return _ncdf(x, self.mu, self.sigma)

    def ppf(self, q: float) -> float:
        return _nppf(q, self.mu, self.sigma)

    def logpdf(self, x: float) -> float:
        z = (x - self.mu) / self.sigma
        return -0.5 * z**2 - math.log(self.sigma) - 0.5 * math.log(2 * math.pi)

    def var(self, confidence: float = 0.95) -> VQResult:
        """VaR at given confidence level (positive loss)."""
        alpha = 1.0 - confidence
        # ppf already returns the distribution quantile (mu + sigma * z_alpha)
        q = self.ppf(alpha)
        loss = -q  # VaR = -quantile (positive loss convention)
        z_std = (q - self.mu) / self.sigma  # standardised z-score for display

        return VQResult(
            value=round(loss, 8),
            verified=True,
            confidence=1.0,
            formula_used=f"VaR_Normal = -(μ + z_{{α={alpha:.4f}}} × σ)",
            formula_latex=r"VaR_\alpha = -(\mu + z_\alpha \cdot \sigma)",
            citation=CITATION,
            proof_trace=[
                ProofStep("alpha", "1 - confidence", r"1-\alpha", alpha, "Tail probability"),
                ProofStep("z_score", f"Φ⁻¹({alpha:.4f})", r"z_\alpha = \Phi^{-1}(\alpha)", z_std,
                          "Normal z-score (assumes normality — may understate tail risk)"),
                ProofStep("var", "-(μ + z × σ)", r"-(\mu + z\sigma)", loss,
                          "VaR as positive loss under Normal assumption"),
            ],
            warnings=[
                "Normal distribution may understate VaR at high confidence levels. "
                "Consider StudentT(df=4).var() for fat-tail adjustment."
            ],
            backend="python",
        )

    def cvar(self, confidence: float = 0.95) -> VQResult:
        """Expected Shortfall (CVaR) under Normal assumption."""
        alpha = 1.0 - confidence
        z = self.ppf(alpha)
        pdf_z = _npdf(z)
        es = self.mu + self.sigma * pdf_z / alpha
        loss = -es

        return VQResult(
            value=round(loss, 8),
            verified=True,
            confidence=1.0,
            formula_used=f"CVaR_Normal = -(μ + σ × φ(z_α)/α)",
            formula_latex=r"ES_\alpha = -\left(\mu + \sigma \frac{\phi(z_\alpha)}{\alpha}\right)",
            citation="Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. Journal of Risk.",
            proof_trace=[
                ProofStep("z_alpha", f"Φ⁻¹({alpha})", r"z_\alpha", z, "Normal quantile"),
                ProofStep("phi_z", "φ(z_α)", r"\phi(z_\alpha)", _npdf(z), "Normal PDF at quantile"),
                ProofStep("cvar", "-(μ + σ × φ(z)/α)", r"-(\mu + \sigma\phi(z)/\alpha)",
                          loss, "Normal CVaR"),
            ],
            warnings=[
                "Normal CVaR understates tail risk. Consider StudentT(df=4).cvar()."
            ],
            backend="python",
        )

    @classmethod
    def fit(cls, data: list[float]) -> "Normal":
        """MLE fit: μ = sample mean, σ = sample std."""
        n = len(data)
        mu = sum(data) / n
        var_ = sum((x - mu) ** 2 for x in data) / (n - 1)
        sigma = math.sqrt(var_)
        fitted = cls(mu=mu, sigma=sigma)
        ll = sum(fitted.logpdf(x) for x in data)
        fitted._fit_stats = {
            "log_likelihood": ll,
            "aic": aic(ll, 2),
            "bic": bic(ll, 2, n),
            "n_obs": n,
        }
        return fitted

    def __repr__(self) -> str:
        return f"Normal(mu={self.mu:.6g}, sigma={self.sigma:.6g})"
