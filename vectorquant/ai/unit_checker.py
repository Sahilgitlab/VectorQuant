"""
VectorQuant Unit Checker — Dimensional Analysis for Finance

Level 6 Hallucination Detection.

Catches the class of errors where the formula is correct but the scale or
annualisation is wrong:
  - Returning 0.003 when the question asked for an annualised Sharpe ratio
    (forgot sqrt(252) — off by 15.9x)
  - Returning 76.0 as a Sharpe ratio (percentage, not ratio)
  - Returning 0.0045 as DV01 for a $1M position (per $1 not per $1M)

The checker runs automatically on every verify_numeric() call and is also
available standalone via vq.ai.unit_checker.check().

Usage:
    from vectorquant.ai.unit_checker import check, UNITS, VQUnit, Scale

    result = check(value=0.003, formula="sharpe_ratio",
                   question="What is the annualised Sharpe ratio?")
    # result.verified = False
    # result.warnings = ["0.003 is 1000x smaller than typical Sharpe range (-3,3)..."]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from vectorquant.core.result import VQResult, UnitCheckResult, ProofStep


class Scale(Enum):
    """Dimensional scale for a financial quantity."""
    RATIO       = "ratio"        # unitless, e.g. Sharpe (-3 to 3)
    PERCENT     = "percent"      # e.g. return as 5.2%
    DECIMAL     = "decimal"      # e.g. return as 0.052
    BPS         = "basis_points" # e.g. spread in bps
    DOLLAR      = "dollar"       # e.g. option price, DV01 in $ terms
    YEARS       = "years"        # e.g. duration
    PROBABILITY = "probability"  # e.g. delta, N(d1) → [0, 1]
    COUNT       = "count"        # e.g. number of observations


@dataclass
class VQUnit:
    """Dimensional annotation for a formula result."""
    scale: Scale
    annualised: bool
    typical_range: tuple              # (lo, hi) — what healthy values look like
    annualisation_factor: Optional[str] = None   # e.g. "sqrt(252)"
    per_unit: Optional[str] = None               # e.g. "per $1M notional"
    description: str = ""


# ── Unit specs for every formula in the registry ──────────────────────────────

UNITS: dict[str, VQUnit] = {
    "sharpe_ratio": VQUnit(
        Scale.RATIO, annualised=True, typical_range=(-3.0, 3.0),
        annualisation_factor="sqrt(252)",
        description="Annualised Sharpe ratio — unitless, typically -3 to 3",
    ),
    "sortino_ratio": VQUnit(
        Scale.RATIO, annualised=True, typical_range=(-5.0, 5.0),
        annualisation_factor="sqrt(252)",
        description="Annualised Sortino ratio — unitless",
    ),
    "parametric_var": VQUnit(
        Scale.DECIMAL, annualised=False, typical_range=(0.005, 0.20),
        description="VaR as a decimal fraction of portfolio (e.g. 0.02 = 2%)",
    ),
    "historical_var": VQUnit(
        Scale.DECIMAL, annualised=False, typical_range=(0.005, 0.20),
        description="Historical VaR as decimal fraction",
    ),
    "cvar": VQUnit(
        Scale.DECIMAL, annualised=False, typical_range=(0.005, 0.30),
        description="CVaR/Expected Shortfall as decimal fraction",
    ),
    "maximum_drawdown": VQUnit(
        Scale.DECIMAL, annualised=False, typical_range=(0.01, 0.99),
        description="Maximum drawdown as decimal fraction (e.g. 0.33 = 33%)",
    ),
    "calmar_ratio": VQUnit(
        Scale.RATIO, annualised=True, typical_range=(0.0, 10.0),
        description="CAGR / max drawdown — unitless ratio",
    ),
    "annualized_vol": VQUnit(
        Scale.DECIMAL, annualised=True, typical_range=(0.01, 0.80),
        annualisation_factor="sqrt(252)",
        description="Annualised volatility as decimal (e.g. 0.20 = 20%)",
    ),
    "cagr": VQUnit(
        Scale.DECIMAL, annualised=True, typical_range=(-0.30, 0.50),
        description="Compound annual growth rate as decimal",
    ),
    "black_scholes_call": VQUnit(
        Scale.DOLLAR, annualised=False, typical_range=(0.01, 500.0),
        description="Call option price in dollars",
    ),
    "black_scholes_put": VQUnit(
        Scale.DOLLAR, annualised=False, typical_range=(0.01, 500.0),
        description="Put option price in dollars",
    ),
    "bs_delta_call": VQUnit(
        Scale.PROBABILITY, annualised=False, typical_range=(0.0, 1.0),
        description="Call delta — probability-like, in [0, 1]",
    ),
    "bs_delta_put": VQUnit(
        Scale.PROBABILITY, annualised=False, typical_range=(-1.0, 0.0),
        description="Put delta — in [-1, 0]",
    ),
    "bs_gamma": VQUnit(
        Scale.RATIO, annualised=False, typical_range=(0.0, 0.20),
        description="Gamma — rate of change of delta per $1 move in underlying",
    ),
    "bond_price": VQUnit(
        Scale.DOLLAR, annualised=False, typical_range=(50.0, 150.0),
        per_unit="per $100 face value",
        description="Bond clean price (percentage of par or dollar amount)",
    ),
    "macaulay_duration": VQUnit(
        Scale.YEARS, annualised=False, typical_range=(0.5, 30.0),
        description="Macaulay duration in years",
    ),
    "modified_duration": VQUnit(
        Scale.YEARS, annualised=False, typical_range=(0.5, 28.0),
        description="Modified duration in years",
    ),
    "dv01": VQUnit(
        Scale.DOLLAR, annualised=False, typical_range=(0.01, 1000.0),
        per_unit="per $1M notional",
        description="Dollar value of 1bp — typically $50–$1000 per $1M for standard bonds",
    ),
    "portfolio_return": VQUnit(
        Scale.DECIMAL, annualised=False, typical_range=(-0.30, 0.50),
        description="Portfolio return as decimal",
    ),
    "portfolio_variance": VQUnit(
        Scale.DECIMAL, annualised=False, typical_range=(0.0001, 0.10),
        description="Portfolio variance (decimal squared — e.g. 0.04 = 20% vol squared)",
    ),
    "beta_coefficient": VQUnit(
        Scale.RATIO, annualised=False, typical_range=(-2.0, 3.0),
        description="Beta — unitless; 1.0 = market-like",
    ),
    "jensen_alpha": VQUnit(
        Scale.DECIMAL, annualised=True, typical_range=(-0.10, 0.10),
        description="Jensen's alpha as decimal annual return",
    ),
    "treynor_ratio": VQUnit(
        Scale.RATIO, annualised=True, typical_range=(-0.20, 0.20),
        description="Treynor ratio — excess return per unit beta",
    ),
    "information_ratio": VQUnit(
        Scale.RATIO, annualised=True, typical_range=(-2.0, 2.0),
        annualisation_factor="sqrt(252)",
        description="Information ratio — active return per unit tracking error",
    ),
    "kelly_criterion": VQUnit(
        Scale.RATIO, annualised=False, typical_range=(0.0, 5.0),
        description="Kelly fraction — optimal allocation as multiple of capital",
    ),
    "log_return": VQUnit(
        Scale.DECIMAL, annualised=False, typical_range=(-0.30, 0.30),
        description="Log return as decimal",
    ),
    "geometric_mean_return": VQUnit(
        Scale.DECIMAL, annualised=False, typical_range=(-0.30, 0.50),
        description="Geometric mean return as decimal",
    ),
    "correlation_pearson": VQUnit(
        Scale.RATIO, annualised=False, typical_range=(-1.0, 1.0),
        description="Pearson correlation — unitless, in [-1, 1]",
    ),
    "ytm": VQUnit(
        Scale.DECIMAL, annualised=True, typical_range=(0.001, 0.20),
        description="Yield to maturity as decimal annual rate (e.g. 0.05 = 5%)",
    ),
}


# ── Public API ─────────────────────────────────────────────────────────────────

def check(
    value: float,
    formula: str,
    question: str = "",
) -> VQResult:
    """
    Level 6 Hallucination Detection — Unit / Dimensional Check.

    Determines whether a numeric value is plausible for the given formula.
    Detects annualisation errors, scale errors (ratio vs %, bps vs $), and
    order-of-magnitude errors.

    Args:
        value:    The numeric value to check
        formula:  Registry name, e.g. "sharpe_ratio"
        question: Optional plain-English description of what was asked
                  (used to detect annualisation intent)

    Returns:
        VQResult with:
          .verified     → True if value is within the expected range
          .value        → the value itself (or corrected value estimate)
          .unit_check   → UnitCheckResult with detailed diagnosis
          .failure_mode → "unit" if out-of-range, None otherwise

    Example:
        result = check(0.003, "sharpe_ratio",
                       "What is the annualised Sharpe ratio?")
        # result.verified = False
        # result.unit_check.likely_error = "forgot sqrt(252) annualisation"
        # result.unit_check.corrected_value = 0.0476
    """
    spec = UNITS.get(formula)
    if spec is None:
        unit_check = UnitCheckResult(
            passed=True,
            expected_scale="unknown",
            actual_interpretation="No unit spec registered for this formula",
            likely_error=None,
            corrected_value=None,
        )
        return VQResult(
            value=value,
            verified=True,
            confidence=0.5,
            formula_used=formula,
            unit_check=unit_check,
            warnings=[f"No unit spec for '{formula}' — cannot perform dimensional check"],
        )

    lo, hi = spec.typical_range

    # ── Annualisation check runs even inside range ────────────────────────────
    # A value of 0.003 for Sharpe is technically in (-3, 3) but is 1000x smaller
    # than a realistic annualised Sharpe. Check if × sqrt(252) lands in the
    # "more typical" interior of the range (0.1, 3) to catch this.
    annualisation_warning: str | None = None
    annualisation_corrected: float | None = None

    if spec.annualised and spec.annualisation_factor == "sqrt(252)":
        annualised_candidate = value * math.sqrt(252)
        # If the raw value is very small but the annualised version is plausible,
        # flag it — the LLM likely forgot to annualise.
        typical_lo = max(abs(lo), 0.05) if lo < 0 else 0.05
        if abs(value) < typical_lo and lo <= annualised_candidate <= hi:
            annualisation_warning = (
                f"Possible annualisation error: {value:.6g} × sqrt(252) = "
                f"{annualised_candidate:.4f}, which is in the expected range. "
                f"Did you forget to multiply by sqrt(252)?"
            )
            annualisation_corrected = annualised_candidate

    in_range = lo <= value <= hi

    # ── In range and no annualisation suspicion — pass ────────────────────────
    if in_range and annualisation_warning is None:
        unit_check = UnitCheckResult(
            passed=True,
            expected_scale=f"{spec.scale.value} {spec.typical_range}",
            actual_interpretation=f"{value:.6g} is within typical range {spec.typical_range}",
            likely_error=None,
            corrected_value=None,
        )
        return VQResult(
            value=value,
            verified=True,
            confidence=1.0,
            formula_used=formula,
            unit_check=unit_check,
            failure_mode=None,
        )

    # ── In range but suspicious annualisation ────────────────────────────────
    if in_range and annualisation_warning is not None:
        unit_check = UnitCheckResult(
            passed=False,
            expected_scale=f"{spec.scale.value} {spec.typical_range}",
            actual_interpretation=(
                f"{value:.6g} is technically in range but suspiciously small "
                f"for an annualised {formula}."
            ),
            likely_error="forgot sqrt(252) annualisation factor",
            corrected_value=annualisation_corrected,
        )
        return VQResult(
            value=annualisation_corrected,
            verified=False,
            confidence=0.2,
            formula_used=formula,
            unit_check=unit_check,
            failure_mode="unit",
            warnings=[annualisation_warning],
        )

    # ── Out of range — diagnose ───────────────────────────────────────────────
    likely_error: Optional[str] = None
    corrected_value: Optional[float] = None
    warnings = []

    # Check 1: forgot annualisation
    if spec.annualised and spec.annualisation_factor == "sqrt(252)":
        annualised = value * math.sqrt(252)
        if lo <= annualised <= hi:
            likely_error = f"forgot annualisation — multiply by sqrt(252)≈15.87"
            corrected_value = annualised
            warnings.append(
                f"{value:.6g} looks like a daily value. "
                f"Annualised = {annualised:.4f}. "
                f"Did you forget to multiply by sqrt(252)?"
            )

    # Check 2: percentage vs decimal (value × 100 falls in range)
    if likely_error is None:
        pct_form = value * 100
        if lo <= pct_form <= hi:
            likely_error = "value is in decimal form but percent was expected (or vice versa)"
            corrected_value = pct_form
            warnings.append(
                f"{value:.6g} × 100 = {pct_form:.4f} which falls in the expected range. "
                f"Possible unit error: decimal vs percent."
            )

    # Check 3: divide by 100 (value ÷ 100 falls in range)
    if likely_error is None:
        dec_form = value / 100.0
        if lo <= dec_form <= hi:
            likely_error = "value appears to be a percentage — should be a decimal ratio"
            corrected_value = dec_form
            warnings.append(
                f"{value:.6g} ÷ 100 = {dec_form:.6f} which falls in the expected range. "
                f"Did you return a percentage (e.g. 5.2%) when a decimal (0.052) was expected?"
            )

    # Check 4: per-unit mismatch (e.g. DV01 per $1 vs per $1M)
    if likely_error is None and spec.per_unit:
        warnings.append(
            f"Check per-unit convention: this formula is typically expressed '{spec.per_unit}'. "
            f"Scale mismatch could account for the out-of-range value {value:.6g}."
        )
        likely_error = f"possible per-unit mismatch ({spec.per_unit})"

    # Generic out-of-range message
    if likely_error is None:
        likely_error = "value is outside the expected range — check formula and inputs"

    warnings.insert(0,
        f"UNIT CHECK FAILED: {formula} = {value:.6g} is outside typical range {spec.typical_range}. "
        f"Expected: {spec.scale.value}. {spec.description}"
    )

    # Check if question mentions annualised (intent detection)
    if question and "annual" in question.lower() and not spec.annualised:
        warnings.append(
            "Note: the question mentions 'annualised' but this formula result "
            "is typically expressed as a per-period value. Check your annualisation."
        )

    unit_check = UnitCheckResult(
        passed=False,
        expected_scale=f"{spec.scale.value}, typical range {spec.typical_range}",
        actual_interpretation=(
            f"{value:.6g} is outside the expected range for {formula}. "
            f"Typical range: {spec.typical_range}."
        ),
        likely_error=likely_error,
        corrected_value=corrected_value,
    )

    return VQResult(
        value=corrected_value if corrected_value is not None else value,
        verified=False,
        confidence=0.0,
        formula_used=formula,
        proof_trace=[
            ProofStep(
                name="range_check",
                formula=f"{lo} <= {value:.6g} <= {hi}",
                formula_latex="",
                value=in_range,
                explanation=f"Value {value:.6g} vs expected range {spec.typical_range}",
            ),
        ],
        failure_mode="unit",
        unit_check=unit_check,
        warnings=warnings,
    )


def get_unit(formula: str) -> Optional[VQUnit]:
    """Return the VQUnit spec for a formula, or None if not registered."""
    return UNITS.get(formula)


def check_with_convention(
    value: float,
    formula: str,
    instrument: str,
    market: str,
) -> VQResult:
    """
    Combined unit + convention check.

    Runs the unit check and also verifies that the value is consistent
    with the market convention for the instrument (e.g. YTM as decimal vs %).

    Args:
        value:      Numeric value to check
        formula:    Registry name (e.g. "ytm")
        instrument: Convention instrument (e.g. "treasury_note")
        market:     Convention market (e.g. "USD")

    Returns:
        VQResult combining unit and convention validation.
    """
    unit_result = check(value, formula)

    try:
        from vectorquant.ai.conventions_db import lookup
        conv = lookup(instrument, market)
        quote_type = conv.get("quote_type", "")

        extra_warnings = []

        # YTM: some markets quote as percentage, others as decimal
        if formula == "ytm":
            if 0.5 < value < 25.0:
                extra_warnings.append(
                    f"YTM = {value:.4g}: this looks like a percentage (e.g. 4.5%). "
                    f"VectorQuant uses decimal form (0.045). "
                    f"Convention for ({instrument}, {market}): {conv.get('day_count', 'N/A')}, "
                    f"{conv.get('compounding', 'N/A')} compounding."
                )
            elif 0.001 <= value <= 0.25:
                extra_warnings.append(
                    f"YTM = {value:.4g}: decimal form — consistent with convention "
                    f"({conv.get('compounding', 'N/A')} compounding, {conv.get('day_count', 'N/A')})."
                )

        unit_result.warnings.extend(extra_warnings)

    except Exception:
        pass  # Convention lookup failure is non-fatal

    return unit_result
