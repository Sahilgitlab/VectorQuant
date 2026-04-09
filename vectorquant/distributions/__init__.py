"""
vq.distributions — Fat-Tail Finance Distributions

Zero external dependencies. Uses only Python stdlib math.

LLMs default to the normal distribution in virtually every financial
context. Student's t is the standard for anything with fat tails —
which is most of finance. This module provides:

  - StudentT   — pdf, cdf, ppf, var, cvar, fit
  - GPD        — Generalised Pareto for extreme value / tail fitting
  - Normal     — baseline for comparison
  - fit_all()  — rank distributions by AIC

All computations return VQResult with proof traces and citations.

Usage:
    import vectorquant.distributions as vqd

    t = vqd.StudentT(df=4)
    result = t.var(confidence=0.99)
    print(result.value)    # VaR — materially higher than Normal at 99%

    ranking = vqd.fit_all(returns)
    # [{"dist": "student_t", "df": 4.2, "aic": -312}, ...]
"""

from .student_t import StudentT
from .gpd import GPD
from .normal import Normal
from .fit import fit_all

__all__ = ["StudentT", "GPD", "Normal", "fit_all"]
