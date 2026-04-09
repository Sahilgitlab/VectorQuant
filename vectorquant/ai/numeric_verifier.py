"""
VectorQuant Numeric Verifier — Level 2/3 Hallucination Detection

Level 2 — Expression check:
    check_expression(formula_name, llm_expression)
    → compares the LLM's formula string against the registry using
      bidirectional token-overlap matching

Level 3 — Numeric spot check:
    verify_numeric(formula_name, llm_value, inputs)
    → recomputes the ground-truth value with VectorQuant's deterministic
      engine and compares it to the LLM's claimed number

Both functions return VQResult with:
  - value:        the correct computed value (Level 3) or similarity (Level 2)
  - verified:     True when the LLM is correct within tolerance
  - confidence:   0–1
  - failure_mode: "formula" | "unit" | None
  - proof_trace:  step-by-step derivation for the ground-truth computation
  - warnings:     plain-English explanation of any detected error
"""

from __future__ import annotations

import math
import time
from typing import Any

from vectorquant.core.result import VQResult, ProofStep, UnitCheckResult
from vectorquant.ai.formula_registry import (
    FORMULA_REGISTRY, get_formula, match_expression as _registry_match,
)


# ── Dispatch table: formula_name → callable ────────────────────────────────
# Each callable accepts **inputs and returns a numeric value (or VQResult).

def _compute_sharpe(returns, risk_free_rate=0.0, **_):
    from vectorquant.core.statistics import mean, standard_deviation
    mu = mean(returns)
    sigma = standard_deviation(returns)
    if sigma == 0:
        raise ValueError("Standard deviation is zero — Sharpe undefined")
    return (mu - risk_free_rate) / sigma


def _compute_parametric_var(returns, confidence_level=0.95, **_):
    from vectorquant.core.statistics import mean, standard_deviation
    from vectorquant.core.probability import normal_inv_cdf
    mu = mean(returns)
    sigma = standard_deviation(returns)
    z = normal_inv_cdf(1.0 - confidence_level)
    return -(mu + z * sigma)


def _compute_historical_var(returns, confidence_level=0.95, **_):
    sorted_r = sorted(returns)
    idx = max(0, int((1.0 - confidence_level) * len(sorted_r)))
    return -sorted_r[idx]


def _compute_cvar(returns, confidence_level=0.95, **_):
    from vectorquant.core.statistics import mean
    var_val = _compute_historical_var(returns, confidence_level)
    tail = [r for r in returns if r <= -var_val]
    if not tail:
        return var_val
    return -mean(tail)


def _compute_max_drawdown(prices, **_):
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _compute_black_scholes_call(S, K, r, sigma, T, **_):
    from vectorquant.finance.derivatives import black_scholes_call
    result = black_scholes_call(S, K, r, sigma, T)
    return float(result)


def _compute_black_scholes_put(S, K, r, sigma, T, **_):
    from vectorquant.finance.derivatives import black_scholes_put
    result = black_scholes_put(S, K, r, sigma, T)
    return float(result)


def _compute_portfolio_return(weights, returns, **_):
    return sum(w * r for w, r in zip(weights, returns))


def _compute_portfolio_variance(weights, cov_matrix, **_):
    n = len(weights)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += weights[i] * weights[j] * cov_matrix[i][j]
    return total


def _compute_cagr(start_value, end_value, years, **_):
    return (end_value / start_value) ** (1.0 / years) - 1.0


def _compute_annualized_vol(daily_returns, **_):
    from vectorquant.core.statistics import standard_deviation
    return standard_deviation(daily_returns) * math.sqrt(252)


def _compute_log_return(P_t, P_t_minus_1, **_):
    return math.log(P_t / P_t_minus_1)


def _compute_correlation_pearson(X, Y, **_):
    from vectorquant.core.statistics import mean, standard_deviation
    n = len(X)
    mu_x, mu_y = mean(X), mean(Y)
    cov = sum((X[i] - mu_x) * (Y[i] - mu_y) for i in range(n)) / (n - 1)
    sx, sy = standard_deviation(X), standard_deviation(Y)
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def _compute_modified_duration(macaulay_duration, ytm, frequency=2, **_):
    return macaulay_duration / (1 + ytm / frequency)


def _compute_dv01(modified_duration, bond_price, **_):
    return modified_duration * bond_price / 10000.0


def _compute_kelly(mean_return, rf, variance, **_):
    return (mean_return - rf) / variance


def _compute_treynor(mean_return, rf, beta, **_):
    return (mean_return - rf) / beta


def _compute_jensen_alpha(r_portfolio, rf, beta, r_market, **_):
    return r_portfolio - (rf + beta * (r_market - rf))


def _compute_calmar(cagr, max_drawdown, **_):
    return cagr / max_drawdown


def _compute_sortino(returns, risk_free_rate=0.0, threshold=0.0, **_):
    from vectorquant.core.statistics import mean
    mu = mean(returns)
    downside = [min(r - threshold, 0.0) for r in returns]
    n = len(downside)
    downside_var = sum(d ** 2 for d in downside) / (n - 1) if n > 1 else 0.0
    downside_std = math.sqrt(downside_var)
    if downside_std == 0:
        return float("inf")
    return (mu - risk_free_rate) / downside_std


def _beta_coefficient(asset_returns, market_returns, **_):
    from vectorquant.core.statistics import mean
    n = len(asset_returns)
    mu_a = mean(asset_returns)
    mu_m = mean(market_returns)
    cov = sum((asset_returns[i] - mu_a) * (market_returns[i] - mu_m) for i in range(n)) / (n - 1)
    var_m = sum((market_returns[i] - mu_m) ** 2 for i in range(n)) / (n - 1)
    if var_m == 0:
        raise ValueError("Market variance is zero")
    return cov / var_m


_COMPUTE_MAP: dict[str, Any] = {
    "sharpe_ratio":           _compute_sharpe,
    "sortino_ratio":          _compute_sortino,
    "parametric_var":         _compute_parametric_var,
    "historical_var":         _compute_historical_var,
    "cvar":                   _compute_cvar,
    "maximum_drawdown":       _compute_max_drawdown,
    "calmar_ratio":           _compute_calmar,
    "annualized_vol":         _compute_annualized_vol,
    "cagr":                   _compute_cagr,
    "black_scholes_call":     _compute_black_scholes_call,
    "black_scholes_put":      _compute_black_scholes_put,
    "portfolio_return":       _compute_portfolio_return,
    "portfolio_variance":     _compute_portfolio_variance,
    "beta_coefficient":       _beta_coefficient,
    "jensen_alpha":           _compute_jensen_alpha,
    "treynor_ratio":          _compute_treynor,
    "kelly_criterion":        _compute_kelly,
    "log_return":             _compute_log_return,
    "correlation_pearson":    _compute_correlation_pearson,
    "modified_duration":      _compute_modified_duration,
    "dv01":                   _compute_dv01,
}


# ── Public API ─────────────────────────────────────────────────────────────────

def verify_numeric(
    formula_name: str,
    llm_value: float,
    inputs: dict,
    tolerance: float | None = None,
) -> VQResult:
    """
    Level 3 Hallucination Detection — Numeric Spot Check.

    Recomputes the ground-truth value using VectorQuant's deterministic
    engine and compares it to the LLM's claimed numeric value.

    Args:
        formula_name: Registry name, e.g. "sharpe_ratio"
        llm_value:    The numeric value the LLM produced
        inputs:       Dict of named inputs required by the formula
        tolerance:    Override default tolerance (taken from registry if None)

    Returns:
        VQResult with:
          .value          → correct computed value
          .verified       → True if llm_value is within tolerance of correct value
          .confidence     → 0–1; 1.0 = exact match, 0.0 = wildly wrong
          .failure_mode   → "formula" if wrong, None if correct
          .warnings       → plain-English diagnosis

    Example:
        result = verify_numeric(
            "sharpe_ratio",
            llm_value=2.45,
            inputs={"returns": [0.01, -0.02, 0.015, 0.02, -0.005],
                    "risk_free_rate": 0.02 / 252}
        )
        # result.value = -0.8094   (correct)
        # result.verified = False
        # result.confidence = 0.0  (LLM is 402% off)
        # result.failure_mode = "formula"
    """
    t0 = time.perf_counter()

    spec = get_formula(formula_name)
    if spec is None:
        return VQResult(
            value=None,
            verified=False,
            confidence=0.0,
            formula_used=formula_name,
            warnings=[f"Formula '{formula_name}' not found in registry. "
                      f"Available: {list(FORMULA_REGISTRY.keys())}"],
            failure_mode="formula",
        )

    # Determine tolerance
    tol: float = tolerance if tolerance is not None else float(
        spec.get("test_case", {}).get("tolerance", 1e-4)
    )

    # Compute ground truth
    compute_fn = _COMPUTE_MAP.get(formula_name)
    if compute_fn is None:
        return VQResult(
            value=None,
            verified=False,
            confidence=0.0,
            formula_used=spec["correct_expression"],
            formula_latex=spec.get("latex", ""),
            citation=spec["citation"],
            warnings=[f"No compute function registered for '{formula_name}'. "
                      "Cannot perform numeric verification."],
        )

    try:
        correct_value = float(compute_fn(**inputs))
    except Exception as exc:
        return VQResult(
            value=None,
            verified=False,
            confidence=0.0,
            formula_used=spec["correct_expression"],
            formula_latex=spec.get("latex", ""),
            citation=spec["citation"],
            warnings=[f"Computation failed: {exc}"],
            failure_mode="formula",
        )

    # Compare
    abs_error = abs(llm_value - correct_value)
    rel_error = abs_error / (abs(correct_value) + 1e-12)
    is_correct = abs_error <= tol

    # Confidence: 1.0 for exact match, decays with relative error
    if is_correct:
        confidence = 1.0
    elif rel_error < 0.01:
        confidence = 0.95
    elif rel_error < 0.05:
        confidence = 0.80
    elif rel_error < 0.20:
        confidence = 0.50
    elif rel_error < 1.0:
        confidence = 0.20
    else:
        confidence = 0.0

    # Diagnosis
    warnings = []
    failure_mode = None

    if not is_correct:
        failure_mode = "formula"
        pct_error = rel_error * 100
        warnings.append(
            f"Correct value: {correct_value:.6g} | LLM value: {llm_value:.6g} | "
            f"Relative error: {pct_error:.1f}%"
        )
        # Check if the LLM used variance instead of std (common Sharpe error)
        if formula_name == "sharpe_ratio" and rel_error > 10:
            warnings.append(
                "Likely error: denominator is variance instead of standard deviation. "
                "Sharpe = (mean - rf) / std(r), NOT / var(r)."
            )
        # Check annualisation
        unit_spec = spec.get("unit", {})
        if unit_spec.get("annualisation_factor") == "sqrt(252)":
            annualised = llm_value * math.sqrt(252)
            if abs(annualised - correct_value) < tol * math.sqrt(252) * 10:
                warnings.append(
                    f"Possible unit error: LLM returned daily value ({llm_value:.6g}). "
                    f"Annualised = {annualised:.6g}. Did you forget ×sqrt(252)?"
                )
                failure_mode = "unit"

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return VQResult(
        value=correct_value,
        verified=is_correct,
        confidence=confidence,
        formula_used=spec["correct_expression"],
        formula_latex=spec.get("latex", ""),
        citation=spec["citation"],
        proof_trace=[
            ProofStep(
                name="ground_truth",
                formula=spec["correct_expression"],
                formula_latex=spec.get("latex", ""),
                value=correct_value,
                explanation=f"VectorQuant deterministic computation: {correct_value:.6g}",
            ),
            ProofStep(
                name="llm_claim",
                formula="(LLM output)",
                formula_latex="",
                value=llm_value,
                explanation=f"LLM claimed value: {llm_value:.6g}",
            ),
            ProofStep(
                name="comparison",
                formula=f"|llm - correct| = {abs_error:.4g}, tolerance = {tolerance}",
                formula_latex="",
                value=is_correct,
                explanation="VERIFIED" if is_correct else f"HALLUCINATION — {rel_error * 100:.1f}% error",
            ),
        ],
        failure_mode=failure_mode,
        warnings=warnings,
        backend="python",
        computation_time_ms=elapsed_ms,
    )  # type: ignore[return-value]


def check_expression(
    formula_name: str,
    llm_expression: str,
) -> VQResult:
    """
    Level 2 Hallucination Detection — Expression Equivalence Check.

    Compares the LLM's formula string against the registry's correct
    expression and known-wrong variants using token-overlap similarity.

    Args:
        formula_name:    Registry name, e.g. "sharpe_ratio"
        llm_expression:  The formula string the LLM produced

    Returns:
        VQResult with:
          .value        → similarity score (0–1, 1.0 = matches correct expression)
          .verified     → True if expression matches correct form
          .confidence   → similarity score
          .failure_mode → "formula" if matched a known-wrong variant
          .warnings     → explanation of what the LLM got wrong

    Example:
        result = check_expression(
            "sharpe_ratio",
            "(mean(returns) - rf) / variance(returns)"
        )
        # result.verified = False
        # result.failure_mode = "formula"
        # result.warnings = ["wrong_denominator: 6000%+ error ..."]
    """
    t0 = time.perf_counter()

    spec = get_formula(formula_name)
    if spec is None:
        return VQResult(
            value=0.0,
            verified=False,
            confidence=0.0,
            formula_used=formula_name,
            warnings=[f"Formula '{formula_name}' not in registry."],
            failure_mode="formula",
        )

    from vectorquant.ai.formula_registry import _tokenize, _jaccard

    tokens_llm = _tokenize(llm_expression)
    tokens_correct = _tokenize(spec["correct_expression"])
    sim_correct = _jaccard(tokens_llm, tokens_correct)

    CORRECT_THRESHOLD = 0.55
    ERROR_THRESHOLD = 0.45

    # ── Score against known-wrong variants ───────────────────────────────────
    best_error_match = None
    best_error_sim = 0.0

    for error in spec.get("common_errors", []):
        if error.get("severity") is None:
            continue
        tokens_error = _tokenize(error["expression"])
        sim = _jaccard(tokens_llm, tokens_error)
        if sim > best_error_sim:
            best_error_sim = sim
            best_error_match = error

    # Error pattern wins only when it scores strictly higher than the correct
    # expression — prevents the correct expression from being flagged as an error
    # because it is a superset of an error-pattern's token set.
    error_wins = (
        best_error_match is not None
        and best_error_sim >= ERROR_THRESHOLD
        and best_error_sim > sim_correct
    )

    if error_wins:
        warnings = [
            f"Matched known LLM error pattern: {best_error_match['error_type']}",
            f"Magnitude: {best_error_match.get('magnitude', 'unknown')}",
            f"Why LLMs make this: {best_error_match.get('why_llms_make_this', '')}",
            f"Correct expression: {spec['correct_expression']}",
        ]
        if best_error_match.get("severity") == "critical":
            warnings.insert(0, "CRITICAL: This error causes large numeric errors.")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return VQResult(
            value=best_error_sim,
            verified=False,
            confidence=0.0,
            formula_used=spec["correct_expression"],
            formula_latex=spec.get("latex", ""),
            citation=spec["citation"],
            proof_trace=[
                ProofStep(
                    "expression_match_to_error",
                    llm_expression,
                    "",
                    best_error_sim,
                    f"Matched error pattern '{best_error_match['error_type']}' with similarity {best_error_sim:.2f}",
                )
            ],
            failure_mode="formula",
            warnings=warnings,
            backend="python",
            computation_time_ms=elapsed_ms,
        )

    # ── Matches correct expression ────────────────────────────────────────────
    if sim_correct >= CORRECT_THRESHOLD:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return VQResult(
            value=sim_correct,
            verified=True,
            confidence=sim_correct,
            formula_used=spec["correct_expression"],
            formula_latex=spec.get("latex", ""),
            citation=spec["citation"],
            proof_trace=[
                ProofStep(
                    "expression_match",
                    llm_expression,
                    "",
                    sim_correct,
                    f"Similarity to correct expression = {sim_correct:.2f} — CORRECT",
                )
            ],
            failure_mode=None,
            warnings=[],
            backend="python",
            computation_time_ms=elapsed_ms,
        )

    # ── No match ─────────────────────────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return VQResult(
        value=sim_correct,
        verified=False,
        confidence=sim_correct,
        formula_used=spec["correct_expression"],
        formula_latex=spec.get("latex", ""),
        citation=spec["citation"],
        proof_trace=[
            ProofStep(
                "expression_match",
                llm_expression,
                "",
                sim_correct,
                f"Similarity to correct expression = {sim_correct:.2f} — UNRECOGNISED",
            )
        ],
        failure_mode="formula",
        warnings=[
            f"Expression did not match correct form (similarity={sim_correct:.2f}) "
            f"or any known error pattern.",
            f"Correct expression: {spec['correct_expression']}",
        ],
        backend="python",
        computation_time_ms=elapsed_ms,
    )


def check_formula(formula_name: str, llm_formula: str) -> VQResult:
    """Alias for check_expression — Level 1/2 combined check."""
    return check_expression(formula_name, llm_formula)
