"""
fit_all() — Rank distributions by goodness-of-fit (AIC).

Fits Normal, Student's t (with df optimisation), and GPD to a dataset
and returns candidates ranked by AIC (lower = better fit).

LLMs nearly always pick the Normal distribution. fit_all() tells you
when that assumption is wrong — which is most of the time in finance.
"""

from __future__ import annotations

import math


def fit_all(data: list[float], threshold: float | None = None) -> list[dict]:
    """
    Fit Normal, Student's t, and GPD to data. Rank by AIC.

    Args:
        data:      List of numeric observations (e.g. daily returns)
        threshold: Optional threshold for GPD Peaks-over-Threshold fitting.
                   If None, GPD is fit to all positive observations.

    Returns:
        List of dicts sorted by AIC (ascending — lower = better fit):
        [
            {
                "dist":           "student_t",
                "params":         {"df": 4.2, "loc": 0.001, "scale": 0.012},
                "log_likelihood": -312.4,
                "aic":            630.8,
                "bic":            638.5,
                "n_obs":          252,
            },
            ...
        ]

    Example:
        ranking = fit_all(daily_returns)
        best = ranking[0]
        print(f"Best fit: {best['dist']} (AIC={best['aic']:.1f})")
        # Typically Student's t — not Normal
    """
    from vectorquant.distributions.student_t import StudentT
    from vectorquant.distributions.normal import Normal
    from vectorquant.distributions.gpd import GPD

    results = []
    n = len(data)

    # ── Normal ────────────────────────────────────────────────────────────────
    try:
        norm = Normal.fit(data)
        stats = norm._fit_stats
        results.append({
            "dist": "normal",
            "params": {"mu": norm.mu, "sigma": norm.sigma},
            "log_likelihood": stats["log_likelihood"],
            "aic": stats["aic"],
            "bic": stats["bic"],
            "n_obs": n,
            "instance": norm,
            "note": (
                "Normal distribution — often inappropriate for financial returns. "
                "Check Student's t AIC for comparison."
            ),
        })
    except Exception as e:
        results.append({"dist": "normal", "error": str(e)})

    # ── Student's t ───────────────────────────────────────────────────────────
    try:
        t = StudentT.fit(data)
        stats = t._fit_stats
        results.append({
            "dist": "student_t",
            "params": {"df": round(t.df, 3), "loc": t.loc, "scale": t.scale},
            "log_likelihood": stats["log_likelihood"],
            "aic": stats["aic"],
            "bic": stats["bic"],
            "n_obs": n,
            "instance": t,
            "note": (
                f"Student's t with ν={t.df:.1f}. "
                f"{'Heavy tails — Normal VaR will understate risk.' if t.df < 10 else 'Near-Normal — thin tails.'}"
            ),
        })
    except Exception as e:
        results.append({"dist": "student_t", "error": str(e)})

    # ── GPD (fit to upper tail) ───────────────────────────────────────────────
    try:
        # Use 90th percentile as threshold if not specified
        sorted_data = sorted(data)
        thr = threshold if threshold is not None else sorted_data[int(0.90 * n)]
        gpd = GPD.fit(data, threshold=thr)
        stats = gpd._fit_stats
        results.append({
            "dist": "gpd",
            "params": {"xi": round(gpd.xi, 4), "beta": gpd.beta, "threshold": thr},
            "log_likelihood": stats["log_likelihood"],
            "aic": stats["aic"],
            "bic": stats["bic"],
            "n_obs": stats["n_exceedances"],
            "instance": gpd,
            "note": (
                f"GPD tail fit (threshold={thr:.4f}). "
                f"Use for extreme quantiles (99.9%+). "
                f"ξ={gpd.xi:.3f}: "
                f"{'heavy Pareto tail' if gpd.xi > 0.2 else 'moderate tail' if gpd.xi > 0 else 'bounded tail'}."
            ),
        })
    except Exception as e:
        results.append({"dist": "gpd", "error": str(e)})

    # Sort by AIC (only rank distributions that succeeded)
    valid = [r for r in results if "aic" in r]
    failed = [r for r in results if "error" in r]
    valid.sort(key=lambda x: x["aic"])

    # Remove the instance objects before returning (not serialisable)
    for r in valid:
        r.pop("instance", None)

    return valid + failed


def compare_var(
    data: list[float],
    confidence: float = 0.99,
) -> dict:
    """
    Compare VaR estimates across Normal and Student's t distributions.

    Shows concretely why the distributional assumption matters.
    At 99% confidence, Student's t (ν=4) gives ~53% higher VaR than Normal.

    Args:
        data:       Return series
        confidence: VaR confidence level

    Returns:
        Dict with VaR from each distribution and the ratio.
    """
    from vectorquant.distributions.student_t import StudentT
    from vectorquant.distributions.normal import Normal

    norm = Normal.fit(data)
    t = StudentT.fit(data)

    var_normal = float(norm.var(confidence).value)
    var_t = float(t.var(confidence).value)

    ratio = var_t / var_normal if var_normal > 0 else float("nan")
    pct_diff = (ratio - 1.0) * 100

    return {
        "confidence": confidence,
        "var_normal": round(var_normal, 6),
        "var_student_t": round(var_t, 6),
        "ratio_t_to_normal": round(ratio, 3),
        "pct_higher_t": round(pct_diff, 1),
        "t_df": round(t.df, 2),
        "interpretation": (
            f"Student's t VaR is {pct_diff:.1f}% {'higher' if pct_diff > 0 else 'lower'} "
            f"than Normal VaR at {confidence:.0%} confidence "
            f"(fitted df={t.df:.1f})."
        ),
    }
