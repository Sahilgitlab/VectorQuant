"""
Internal math utilities for vq.distributions — zero external dependencies.

Implements:
  - Lanczos gamma approximation (for Student's t PDF/CDF)
  - Regularised incomplete beta function (for Student's t CDF)
  - Brent's method (root finding for ppf / quantile)
  - Normal PDF and CDF (for GPD and comparisons)
"""

from __future__ import annotations
import math


# ── Gamma function (Lanczos approximation) ────────────────────────────────────

_LANCZOS_G = 7
_LANCZOS_C = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
]


def lgamma(x: float) -> float:
    """Natural log of the gamma function (Lanczos)."""
    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - lgamma(1 - x)
    x -= 1
    a = _LANCZOS_C[0]
    t = x + _LANCZOS_G + 0.5
    for i in range(1, _LANCZOS_G + 2):
        a += _LANCZOS_C[i] / (x + i)
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def gamma(x: float) -> float:
    """Gamma function."""
    return math.exp(lgamma(x))


def lbeta(a: float, b: float) -> float:
    """Log of the beta function: lB(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b)."""
    return lgamma(a) + lgamma(b) - lgamma(a + b)


# ── Regularised incomplete beta (for Student's t CDF) ────────────────────────

def _betacf(a: float, b: float, x: float, max_iter: int = 200) -> float:
    """Continued fraction representation of regularised incomplete beta."""
    TINY = 1e-30
    fpmin = TINY
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-7:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x < 0.0 or x > 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    lbx = lgamma(a + b) - lgamma(a) - lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbx) * _betacf(a, b, x) / a
    else:
        return 1.0 - math.exp(lbx) * _betacf(b, a, 1.0 - x) / b


# ── Root finding (Brent's method) ────────────────────────────────────────────

def brent(f, a: float, b: float, tol: float = 1e-10, max_iter: int = 500) -> float:
    """
    Find root of f in [a, b] using Brent's method.
    Assumes f(a) and f(b) have opposite signs.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        # Expand bracket if needed
        for _ in range(50):
            if fa * fb <= 0:
                break
            a -= abs(b - a)
            b += abs(b - a)
            fa, fb = f(a), f(b)
        if fa * fb > 0:
            raise ValueError(
                f"Brent: f(a)={fa:.4g} and f(b)={fb:.4g} have same sign. "
                f"Could not bracket root in [{a}, {b}]."
            )
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c, fc = a, fa
    mflag = True
    s = b
    d = 0.0
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fa != fc and fb != fc:
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            s = b - fb * (b - a) / (fb - fa)
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = (not mflag) and abs(c - d) < tol
        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False
        fs = f(s)
        d = c
        c, fc = b, fb
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
    return b


# ── Normal distribution (stdlib) ──────────────────────────────────────────────

def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Standard normal PDF."""
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal CDF using stdlib erfc."""
    return 0.5 * math.erfc(-(x - mu) / (sigma * math.sqrt(2)))


def normal_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal quantile (inverse CDF) using rational approximation (Abramowitz & Stegun)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    # Use stdlib math.erfinv (Python 3.12+) or Brent fallback
    try:
        return mu + sigma * math.sqrt(2) * math.erfinv(2 * p - 1)
    except AttributeError:
        return mu + sigma * brent(lambda x: normal_cdf(x) - p, -20.0, 20.0)


# ── AIC / BIC ─────────────────────────────────────────────────────────────────

def aic(log_likelihood: float, n_params: int) -> float:
    """Akaike Information Criterion: AIC = -2 × logL + 2k."""
    return -2 * log_likelihood + 2 * n_params


def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Bayesian Information Criterion: BIC = -2 × logL + k × ln(n)."""
    return -2 * log_likelihood + n_params * math.log(n_obs)
