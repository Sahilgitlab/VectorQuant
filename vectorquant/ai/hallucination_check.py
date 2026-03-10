"""
AI Hallucination Detection

Allows AI systems to self-correct by checking claims against
known mathematical truths and VectorQuant's deterministic engine.
"""

import math


class HallucinationResult:
    """Structured result of a hallucination check."""
    def __init__(self, is_correct, correct_formula, computed_value, 
                 confidence, details=None):
        self.is_correct = is_correct
        self.correct_formula = correct_formula
        self.computed_value = computed_value
        self.confidence = confidence
        self.details = details or ""

    def to_dict(self):
        return {
            "is_correct": self.is_correct,
            "correct_formula": self.correct_formula,
            "computed_value": self.computed_value,
            "confidence": self.confidence,
            "details": self.details,
        }

    def __repr__(self):
        status = "CORRECT" if self.is_correct else "HALLUCINATION"
        return f"HallucinationResult({status}, confidence={self.confidence})"


# ─── Known correct formulas ────────────────────────────────────────────────

_KNOWN_FORMULAS = {
    "sharpe_ratio": {
        "formula": "(mu - r_f) / sigma",
        "description": "Sharpe Ratio = (mean return - risk-free rate) / standard deviation of returns",
    },
    "parametric_var": {
        "formula": "-(mu + z * sigma)",
        "description": "VaR = -(mean + z_score * std_dev)",
    },
    "black_scholes_call": {
        "formula": "S*N(d1) - K*exp(-rT)*N(d2)",
        "description": "Black-Scholes Call = S*N(d1) - K*e^(-rT)*N(d2)",
    },
    "black_scholes_put": {
        "formula": "K*exp(-rT)*N(-d2) - S*N(-d1)",
        "description": "Black-Scholes Put = K*e^(-rT)*N(-d2) - S*N(-d1)",
    },
    "portfolio_variance": {
        "formula": "w' * Cov * w",
        "description": "Portfolio Variance = weights^T * Covariance_Matrix * weights",
    },
    "capm": {
        "formula": "r_f + beta * (r_m - r_f)",
        "description": "CAPM: E(R) = risk-free rate + beta * market risk premium",
    },
    "d1": {
        "formula": "(ln(S/K) + (r + sigma^2/2)*T) / (sigma*sqrt(T))",
        "description": "d1 in Black-Scholes formula",
    },
    "d2": {
        "formula": "d1 - sigma*sqrt(T)",
        "description": "d2 in Black-Scholes formula",
    },
}


def check_formula(claim_name, claim_formula):
    """
    Compares a claimed formula against VectorQuant's known correct formulas.

    Args:
        claim_name:    Name of the formula (e.g. "sharpe_ratio").
        claim_formula: The claimed formula string.

    Returns:
        HallucinationResult
    """
    claim_name_lower = claim_name.lower().replace(" ", "_")

    if claim_name_lower not in _KNOWN_FORMULAS:
        return HallucinationResult(
            is_correct=False,
            correct_formula=None,
            computed_value=None,
            confidence=0.0,
            details=f"Unknown formula name: {claim_name}. "
                    f"Known formulas: {list(_KNOWN_FORMULAS.keys())}"
        )

    known = _KNOWN_FORMULAS[claim_name_lower]
    correct = known["formula"]

    # Normalize for comparison (strip spaces, lowercase)
    norm_claim = claim_formula.lower().replace(" ", "")
    norm_correct = correct.lower().replace(" ", "")

    is_correct = norm_claim == norm_correct

    return HallucinationResult(
        is_correct=is_correct,
        correct_formula=correct,
        computed_value=None,
        confidence=1.0 if is_correct else 0.0,
        details=f"Claimed: {claim_formula} | Correct: {correct}. "
                f"{known['description']}"
    )


def check_numerical_claim(computation_name, claimed_value, params, tolerance=1e-4):
    """
    Recomputes a named computation and checks the claimed value.

    Args:
        computation_name: One of "sharpe_ratio", "parametric_var", 
                          "black_scholes_call", "black_scholes_put".
        claimed_value:    The value being checked.
        params:           Dict of parameters.
        tolerance:        Allowable error.

    Returns:
        HallucinationResult
    """
    from vectorquant.ai.verify import verify_finance_formula
    
    result = verify_finance_formula(computation_name, params, claimed_value, tolerance)
    
    return HallucinationResult(
        is_correct=result.verified,
        correct_formula=_KNOWN_FORMULAS.get(computation_name, {}).get("formula", "N/A"),
        computed_value=result.computed_value,
        confidence=result.confidence,
        details=result.details
    )


def validate_prediction(hypothesis, S0, mu, sigma, T, target_price, n_simulations=10000):
    """
    Uses Monte Carlo simulation to probabilistically validate
    a forward-looking financial prediction.

    Args:
        hypothesis:    Description of the claim (for labeling).
        S0:            Current price.
        mu:            Expected return.
        sigma:         Volatility.
        T:             Time horizon in years.
        target_price:  The price threshold in the hypothesis.
        n_simulations: Number of MC paths.

    Returns:
        HallucinationResult with probability and expected value.
    """
    from vectorquant.stochastic.processes import simulate_geometric_brownian_motion
    from vectorquant.core.statistics import mean

    paths = simulate_geometric_brownian_motion(S0, mu, sigma, T, T, n_simulations)
    terminal_prices = [p[-1] for p in paths]

    prob_above = sum(1 for p in terminal_prices if p >= target_price) / n_simulations
    expected_val = mean(terminal_prices)

    if prob_above >= 0.7:
        confidence_label = "high"
    elif prob_above >= 0.4:
        confidence_label = "moderate"
    else:
        confidence_label = "low"

    return HallucinationResult(
        is_correct=prob_above >= 0.5,
        correct_formula=f"P(S_T >= {target_price}) via {n_simulations} GBM simulations",
        computed_value=round(prob_above, 4),
        confidence=round(prob_above, 4),
        details=(f"Hypothesis: '{hypothesis}'. "
                 f"P(reach target) = {prob_above:.2%}. "
                 f"E[S_T] = {expected_val:.2f}. "
                 f"Confidence: {confidence_label}.")
    )
