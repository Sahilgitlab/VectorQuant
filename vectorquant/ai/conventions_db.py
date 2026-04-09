"""
VectorQuant Financial Conventions Database

Machine-readable database of financial market conventions: day count fractions,
compounding frequencies, settlement rules, and yield quote types.

This is where LLMs make their most dangerous silent errors — correct formula,
wrong convention. An LLM implementing Black-Scholes exactly can still return a
wrong bond yield by assuming Act/365 when the market uses Act/360.

Usage:
    from vectorquant.ai.conventions_db import lookup, check, day_count_fraction

    conv = lookup("treasury_bill", "USD")
    # {"day_count": "Act/360", "quote_type": "discount_rate", ...}

    result = check("corporate_bond", "USD",
                   {"day_count": "Act/365", "compounding": "continuous"})
    # VQResult with errors field describing what the LLM got wrong
"""

from __future__ import annotations

from vectorquant.core.result import VQResult, ProofStep


# ── Convention database ────────────────────────────────────────────────────────
# Key: (instrument, market) — both lower/upper normalised on lookup
# Value: dict of convention fields with citation

CONVENTIONS: dict[tuple[str, str], dict] = {

    # ── US Treasury ───────────────────────────────────────────────────────────

    ("treasury_bill", "USD"): {
        "day_count": "Act/360",
        "quote_type": "discount_rate",      # NOT yield — the most common LLM error
        "compounding": "simple",
        "settlement_days": 1,
        "citation": "US Treasury, Uniform Offering Circular, 31 CFR 356",
        "notes": (
            "T-bills are quoted as a discount rate, not a yield. "
            "LLMs almost universally assume yield and Act/365. Both are wrong."
        ),
    },

    ("treasury_note", "USD"): {
        "day_count": "Act/Act (ICMA)",
        "quote_type": "yield",
        "compounding": "semi-annual",
        "settlement_days": 1,
        "citation": "US Treasury, Uniform Offering Circular, 31 CFR 356",
        "notes": "2yr, 3yr, 5yr, 7yr, 10yr T-notes. Semi-annual coupon, Act/Act.",
    },

    ("treasury_bond", "USD"): {
        "day_count": "Act/Act (ICMA)",
        "quote_type": "yield",
        "compounding": "semi-annual",
        "settlement_days": 1,
        "citation": "US Treasury, Uniform Offering Circular, 31 CFR 356",
        "notes": "30yr T-bonds. Same conventions as T-notes.",
    },

    ("tips", "USD"): {
        "day_count": "Act/Act (ICMA)",
        "quote_type": "real_yield",
        "compounding": "semi-annual",
        "settlement_days": 1,
        "citation": "US Treasury, Inflation-Protected Securities conventions",
        "notes": "Treasury Inflation-Protected Securities. Principal adjusts with CPI.",
    },

    # ── US Corporate Bonds ────────────────────────────────────────────────────

    ("corporate_bond", "USD"): {
        "day_count": "30/360",
        "quote_type": "yield",
        "compounding": "semi-annual",
        "settlement_days": 2,
        "citation": "SIFMA US corporate bond market conventions (Street convention)",
        "notes": (
            "USD corporate bonds use 30/360 (not Act/360 or Act/365) and "
            "semi-annual compounding. LLMs commonly use annual compounding."
        ),
    },

    ("high_yield_bond", "USD"): {
        "day_count": "30/360",
        "quote_type": "yield",
        "compounding": "semi-annual",
        "settlement_days": 2,
        "citation": "SIFMA US high-yield bond market conventions",
        "notes": "Same conventions as investment-grade corporate bonds.",
    },

    ("municipal_bond", "USD"): {
        "day_count": "30/360",
        "quote_type": "yield",
        "compounding": "semi-annual",
        "settlement_days": 2,
        "citation": "MSRB (Municipal Securities Rulemaking Board) rules",
        "notes": "US municipal bonds. Tax-exempt; same day-count as corporates.",
    },

    # ── European Government Bonds ─────────────────────────────────────────────

    ("government_bond", "EUR"): {
        "day_count": "Act/Act (ICMA)",
        "quote_type": "yield",
        "compounding": "annual",            # annual, NOT semi-annual — common LLM error
        "settlement_days": 3,
        "citation": "ICMA Rule Book, European government bond conventions",
        "notes": (
            "EUR government bonds pay coupons annually (unlike US semi-annual). "
            "Settlement is T+3 in most European markets."
        ),
    },

    ("government_bond", "GBP"): {
        "day_count": "Act/Act (ICMA)",
        "quote_type": "yield",
        "compounding": "semi-annual",
        "settlement_days": 1,
        "citation": "UK Debt Management Office (DMO), Gilt market conventions",
        "notes": "UK Gilts. Semi-annual coupon, T+1 settlement.",
    },

    ("government_bond", "JPY"): {
        "day_count": "Act/365 (Fixed)",
        "quote_type": "yield",
        "compounding": "semi-annual",
        "settlement_days": 2,
        "citation": "Japan Ministry of Finance, JGB market conventions",
        "notes": "Japanese Government Bonds. Act/365 Fixed (not Act/Act).",
    },

    ("government_bond", "USD"): {
        "day_count": "Act/Act (ICMA)",
        "quote_type": "yield",
        "compounding": "semi-annual",
        "settlement_days": 1,
        "citation": "SIFMA/FINRA US Treasury market conventions",
        "notes": "US Treasury notes and bonds.",
    },

    ("government_bond", "AUD"): {
        "day_count": "Act/Act (ICMA)",
        "quote_type": "yield",
        "compounding": "semi-annual",
        "settlement_days": 2,
        "citation": "Australian Office of Financial Management (AOFM) conventions",
    },

    ("government_bond", "CAD"): {
        "day_count": "Act/365 (Fixed)",
        "quote_type": "yield",
        "compounding": "semi-annual",
        "settlement_days": 2,
        "citation": "Bank of Canada, Government of Canada bond conventions",
    },

    # ── Interest Rate Swaps ───────────────────────────────────────────────────

    ("interest_rate_swap", "USD"): {
        "fixed_leg": {
            "day_count": "30/360",
            "frequency": "semi-annual",
        },
        "float_leg": {
            "day_count": "Act/360",
            "frequency": "quarterly",
            "index": "SOFR",
        },
        "settlement_days": 2,
        "citation": "ISDA 2006 Definitions, USD swap market conventions",
        "notes": (
            "Post-LIBOR transition: USD swaps now reference SOFR (not LIBOR). "
            "Fixed leg: 30/360 semi-annual. Float: Act/360 quarterly."
        ),
    },

    ("interest_rate_swap", "EUR"): {
        "fixed_leg": {
            "day_count": "30/360",
            "frequency": "annual",
        },
        "float_leg": {
            "day_count": "Act/360",
            "frequency": "semi-annual",
            "index": "EURIBOR",
        },
        "settlement_days": 2,
        "citation": "ISDA 2006 Definitions, EUR swap market conventions",
        "notes": "EUR IRS: fixed annual vs EURIBOR semi-annual.",
    },

    ("interest_rate_swap", "GBP"): {
        "fixed_leg": {
            "day_count": "Act/365 (Fixed)",
            "frequency": "semi-annual",
        },
        "float_leg": {
            "day_count": "Act/365 (Fixed)",
            "frequency": "semi-annual",
            "index": "SONIA",
        },
        "settlement_days": 0,
        "citation": "ISDA 2006 Definitions, GBP swap market conventions",
        "notes": "GBP swaps use Act/365 Fixed on both legs. SONIA is overnight.",
    },

    ("interest_rate_swap", "JPY"): {
        "fixed_leg": {
            "day_count": "Act/365 (Fixed)",
            "frequency": "semi-annual",
        },
        "float_leg": {
            "day_count": "Act/360",
            "frequency": "semi-annual",
            "index": "TONAR",
        },
        "settlement_days": 2,
        "citation": "ISDA 2006 Definitions, JPY swap market conventions",
    },

    # ── Money Market ──────────────────────────────────────────────────────────

    ("certificate_of_deposit", "USD"): {
        "day_count": "Act/360",
        "quote_type": "yield",
        "compounding": "simple",
        "settlement_days": 0,
        "citation": "Federal Reserve Regulation D; ISDA money market conventions",
        "notes": "USD CDs: Act/360 simple interest. Quote as yield (unlike T-bills).",
    },

    ("commercial_paper", "USD"): {
        "day_count": "Act/360",
        "quote_type": "discount_rate",
        "compounding": "simple",
        "settlement_days": 0,
        "citation": "Federal Reserve / DTCC commercial paper conventions",
        "notes": "Quoted on a discount basis like T-bills.",
    },

    ("repo", "USD"): {
        "day_count": "Act/360",
        "quote_type": "repo_rate",
        "compounding": "simple",
        "settlement_days": 0,
        "citation": "SIFMA/ICMA Global Master Repurchase Agreement",
        "notes": "Overnight and term repos. Act/360 simple interest.",
    },

    ("libor", "USD"): {
        "day_count": "Act/360",
        "quote_type": "rate",
        "compounding": "simple",
        "settlement_days": 2,
        "citation": "ICE Benchmark Administration (IBA). Now largely replaced by SOFR.",
        "notes": "USD LIBOR was discontinued June 2023. Use SOFR for new contracts.",
    },

    ("sofr", "USD"): {
        "day_count": "Act/360",
        "quote_type": "rate",
        "compounding": "simple",
        "settlement_days": 1,
        "citation": "Federal Reserve Bank of New York, SOFR methodology",
        "notes": "Secured Overnight Financing Rate. USD LIBOR replacement.",
    },

    ("euribor", "EUR"): {
        "day_count": "Act/360",
        "quote_type": "rate",
        "compounding": "simple",
        "settlement_days": 2,
        "citation": "European Money Markets Institute (EMMI), EURIBOR methodology",
    },

    # ── Equity Derivatives ────────────────────────────────────────────────────

    ("equity_option", "US"): {
        "exercise": "American",
        "expiry": "3rd Friday of expiry month",
        "settlement": "T+1",
        "day_count": "Act/365",
        "quote_convention": "premium_in_dollars",
        "citation": "OCC Options Clearing Corporation, options contract specifications",
        "notes": (
            "Listed US equity options: American exercise, 3rd Friday expiry. "
            "Mini options (1/10 lot) and weekly options also exist."
        ),
    },

    ("equity_option", "EU"): {
        "exercise": "European",
        "expiry": "3rd Friday of expiry month",
        "settlement": "T+2",
        "day_count": "Act/365",
        "quote_convention": "premium_in_local_currency",
        "citation": "Eurex / Euronext options specifications",
        "notes": "European-style (no early exercise) for most European index options.",
    },

    ("index_option", "US"): {
        "exercise": "European",
        "expiry": "3rd Friday of expiry month",
        "settlement": "cash",
        "settlement_days": 1,
        "day_count": "Act/365",
        "citation": "CBOE index option specifications (SPX, NDX, etc.)",
        "notes": "US index options (SPX, NDX) are European-style and cash-settled.",
    },

    # ── FX ────────────────────────────────────────────────────────────────────

    ("fx_spot", "USDJPY"): {
        "settlement_days": 2,
        "quote_convention": "indirect",   # JPY per 1 USD
        "pip_size": 0.01,
        "citation": "FX market conventions (BIS)",
        "notes": "USD/JPY is quoted as JPY per USD. Settlement T+2.",
    },

    ("fx_spot", "EURUSD"): {
        "settlement_days": 2,
        "quote_convention": "direct",     # USD per 1 EUR
        "pip_size": 0.0001,
        "citation": "FX market conventions (BIS)",
        "notes": "EUR/USD is quoted as USD per EUR (EUR is base currency).",
    },

    ("fx_forward", "USDJPY"): {
        "day_count": "Act/360",
        "settlement_days": 2,
        "citation": "ISDA FX definitions",
        "notes": "FX forwards: use Act/360 for USD leg, Act/365 for JPY leg when computing carry.",
    },

    ("fx_forward", "EURUSD"): {
        "day_count": "Act/360",
        "settlement_days": 2,
        "citation": "ISDA FX definitions",
    },

    # ── Futures ───────────────────────────────────────────────────────────────

    ("treasury_futures", "USD"): {
        "contract_size": 100000,
        "quote_convention": "percentage_of_par",
        "settlement": "physical",
        "day_count": "Act/Act (ICMA)",
        "citation": "CME Group Treasury futures specifications",
        "notes": "T-note/T-bond futures: cheapest-to-deliver bond delivered at expiry.",
    },

    ("equity_futures", "US"): {
        "settlement": "cash",
        "day_count": "Act/365",
        "margin": "initial_margin_plus_variation",
        "citation": "CME/CBOE equity index futures specifications",
        "notes": "E-mini S&P 500 (ES), Nasdaq 100 (NQ), etc. Cash settled.",
    },

    # ── Credit ────────────────────────────────────────────────────────────────

    ("cds", "USD"): {
        "day_count": "Act/360",
        "premium_frequency": "quarterly",
        "settlement_days": 3,
        "recovery_rate_assumption": 0.40,
        "citation": "ISDA 2003/2014 Credit Derivatives Definitions",
        "notes": (
            "CDS premium (spread) paid quarterly on Act/360. "
            "Standard recovery rate assumption: 40% for senior unsecured."
        ),
    },

    ("cds", "EUR"): {
        "day_count": "Act/360",
        "premium_frequency": "quarterly",
        "settlement_days": 3,
        "recovery_rate_assumption": 0.40,
        "citation": "ISDA 2003/2014 Credit Derivatives Definitions",
    },
}


# ── Public API ─────────────────────────────────────────────────────────────────

class VQConventionNotFound(KeyError):
    """Raised when no convention exists for the given (instrument, market) pair."""
    pass


def lookup(instrument: str, market: str) -> dict:
    """
    Return the convention record for (instrument, market).

    Args:
        instrument: e.g. "treasury_bill", "corporate_bond", "interest_rate_swap"
        market:     e.g. "USD", "EUR", "GBP"

    Returns:
        Convention dict with fields: day_count, compounding, settlement_days,
        quote_type, citation, notes, and instrument-specific fields.

    Raises:
        VQConventionNotFound: if the (instrument, market) pair is not in the database.
    """
    key = (instrument.lower().replace(" ", "_").replace("-", "_"),
           market.upper())
    if key not in CONVENTIONS:
        available = [f"({i}, {m})" for (i, m) in CONVENTIONS.keys()]
        raise VQConventionNotFound(
            f"No convention for ({instrument}, {market}). "
            f"Available: {available}"
        )
    return dict(CONVENTIONS[key])


def list_all() -> list[dict]:
    """Return all convention entries as a list of dicts."""
    return [
        {"instrument": i, "market": m, **v}
        for (i, m), v in CONVENTIONS.items()
    ]


def check(
    instrument: str,
    market: str,
    llm_assumptions: dict,
) -> VQResult:
    """
    Level 5 Hallucination Detection — Convention Check.

    Compare LLM-assumed conventions against the authoritative database.

    Args:
        instrument:      e.g. "treasury_bill"
        market:          e.g. "USD"
        llm_assumptions: Dict of convention fields the LLM used, e.g.
                         {"day_count": "Act/365", "compounding": "annual"}

    Returns:
        VQResult with:
          .verified     → True if all supplied fields match the database
          .value        → number of errors found
          .warnings     → list of (field, llm_value, correct_value) mismatches
          .failure_mode → "convention" if errors found, None otherwise

    Example:
        result = check("corporate_bond", "USD",
                       {"day_count": "Act/365", "compounding": "continuous"})
        # Two errors: day_count (should be 30/360), compounding (should be semi-annual)
    """
    try:
        correct = lookup(instrument, market)
    except VQConventionNotFound as e:
        return VQResult(
            value=0,
            verified=False,
            confidence=0.0,
            formula_used=f"conventions.check({instrument}, {market})",
            warnings=[str(e)],
            failure_mode="convention",
        )

    errors = []
    for field, llm_val in llm_assumptions.items():
        if field in correct and correct[field] != llm_val:
            errors.append({
                "field": field,
                "llm_assumed": llm_val,
                "correct": correct[field],
            })

    is_correct = len(errors) == 0
    confidence = 1.0 if is_correct else max(0.0, 1.0 - len(errors) * 0.25)

    warnings = []
    for err in errors:
        warnings.append(
            f"CONVENTION ERROR — {err['field']}: "
            f"LLM used '{err['llm_assumed']}', correct is '{err['correct']}'"
        )
    if not is_correct:
        warnings.append(f"Reference: {correct.get('citation', 'N/A')}")

    return VQResult(
        value=len(errors),
        verified=is_correct,
        confidence=confidence,
        formula_used=f"conventions({instrument}, {market})",
        citation=correct.get("citation", ""),
        proof_trace=[
            ProofStep(
                name="convention_lookup",
                formula=f"CONVENTIONS[({instrument}, {market})]",
                formula_latex="",
                value=correct,
                explanation=f"Database record for ({instrument}, {market})",
            ),
            ProofStep(
                name="field_comparison",
                formula=str(llm_assumptions),
                formula_latex="",
                value=errors,
                explanation=f"{len(errors)} convention error(s) found",
            ),
        ],
        failure_mode=None if is_correct else "convention",
        warnings=warnings,
    )


def day_count_fraction(
    start_date: str,
    end_date: str,
    convention: str,
) -> VQResult:
    """
    Compute the day count fraction between two dates under a given convention.

    Args:
        start_date: ISO date string "YYYY-MM-DD"
        end_date:   ISO date string "YYYY-MM-DD"
        convention: One of "Act/360", "Act/365", "Act/Act", "30/360",
                    "Act/Act (ICMA)", "Act/365 (Fixed)"

    Returns:
        VQResult with .value = the day count fraction (float)

    Example:
        dcf = day_count_fraction("2026-01-15", "2026-07-15", "Act/360")
        # 0.5028  (not 0.5000 — the difference matters for money market instruments)
    """
    import datetime

    def _parse(s: str) -> datetime.date:
        return datetime.date.fromisoformat(s)

    d1 = _parse(start_date)
    d2 = _parse(end_date)
    actual_days = (d2 - d1).days

    conv_upper = convention.upper().replace(" ", "").replace("(", "").replace(")", "")

    if conv_upper in ("ACT/360",):
        dcf = actual_days / 360.0
        formula = f"{actual_days} / 360"
    elif conv_upper in ("ACT/365", "ACT/365FIXED", "ACT/365F"):
        dcf = actual_days / 365.0
        formula = f"{actual_days} / 365"
    elif conv_upper in ("ACT/ACT", "ACT/ACTICMA", "ACTACT"):
        # Simplified Act/Act: count days in each coupon period
        # Full Act/Act (ICMA) requires coupon schedule; this is the bond-year approximation
        year_start = datetime.date(d1.year, 1, 1)
        year_end = datetime.date(d1.year + 1, 1, 1)
        days_in_year = (year_end - year_start).days
        dcf = actual_days / days_in_year
        formula = f"{actual_days} / {days_in_year} (days in year from start)"
    elif conv_upper in ("30/360", "30360"):
        # 30/360 (US / Bond Basis)
        y1, m1, d1v = d1.year, d1.month, min(d1.day, 30)
        y2, m2, d2v = d2.year, d2.month, d2.day
        if d2v == 31 and d1v >= 30:
            d2v = 30
        if d1v == 31:
            d1v = 30
        days_30_360 = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2v - d1v)
        dcf = days_30_360 / 360.0
        formula = f"30/360: ({days_30_360}) / 360"
    else:
        return VQResult(
            value=None,
            verified=False,
            confidence=0.0,
            formula_used=f"day_count_fraction({convention})",
            warnings=[f"Unknown day count convention: '{convention}'. "
                      "Supported: Act/360, Act/365, Act/Act, 30/360"],
        )

    return VQResult(
        value=round(dcf, 8),
        verified=True,
        confidence=1.0,
        formula_used=f"DCF({convention}) = {formula} = {dcf:.6f}",
        citation="ISDA 2006 Definitions, Section 4.16 (Day Count Fractions)",
        proof_trace=[
            ProofStep(
                name="actual_days",
                formula=f"({end_date}) - ({start_date})",
                formula_latex="",
                value=actual_days,
                explanation=f"Calendar days between dates: {actual_days}",
            ),
            ProofStep(
                name="day_count_fraction",
                formula=formula,
                formula_latex="",
                value=dcf,
                explanation=f"Day count fraction under {convention} convention",
            ),
        ],
        backend="python",
    )
