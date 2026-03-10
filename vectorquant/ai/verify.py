"""
AI Verification Layer

Provides deterministic verification of mathematical, probabilistic,
and financial computations. AI systems use this to check reasoning
against ground-truth calculations.
"""

import math


class VerificationResult:
    """Structured result of a verification check."""
    def __init__(self, verified, computed_value, expected_value, 
                 confidence, method, details=None):
        self.verified = verified
        self.computed_value = computed_value
        self.expected_value = expected_value
        self.confidence = confidence
        self.method = method
        self.details = details or ""

    def to_dict(self):
        return {
            "verified": self.verified,
            "computed_value": self.computed_value,
            "expected_value": self.expected_value,
            "confidence": self.confidence,
            "method": self.method,
            "details": self.details,
        }

    def __repr__(self):
        status = "VERIFIED" if self.verified else "FAILED"
        return (f"VerificationResult({status}, computed={self.computed_value}, "
                f"expected={self.expected_value}, confidence={self.confidence})")


# ─── Registry of known safe math operations ────────────────────────────────

_SAFE_MATH = {
    "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "abs": abs, "pow": pow, "pi": math.pi, "e": math.e,
}


def verify_calculation(expression, expected, tolerance=1e-6):
    """
    Evaluates a math expression string using safe builtins
    and compares to the expected value.

    Args:
        expression: A string math expression, e.g. "sqrt(2) * 3"
        expected:   The expected numeric result.
        tolerance:  Allowable absolute error.
    
    Returns:
        VerificationResult
    """
    try:
        computed = eval(expression, {"__builtins__": {}}, _SAFE_MATH)
        diff = abs(computed - expected)
        verified = diff <= tolerance
        confidence = 1.0 if verified else max(0.0, 1.0 - diff / (abs(expected) + 1e-12))
        return VerificationResult(
            verified=verified,
            computed_value=computed,
            expected_value=expected,
            confidence=round(confidence, 4),
            method="safe_eval",
            details=f"Expression: {expression}, Absolute error: {diff:.2e}"
        )
    except Exception as e:
        return VerificationResult(
            verified=False, computed_value=None,
            expected_value=expected, confidence=0.0,
            method="safe_eval", details=f"Evaluation error: {str(e)}"
        )


def verify_probability(dist, params, x, expected, tolerance=1e-4):
    """
    Validates a probability computation against VectorQuant's
    pure-Python probability module.

    Args:
        dist:     Distribution name ("normal_pdf", "normal_cdf").
        params:   Dict of distribution parameters, e.g. {"mu": 0, "sigma": 1}.
        x:        The point to evaluate.
        expected: The expected probability value.
        tolerance: Allowable error.
    
    Returns:
        VerificationResult
    """
    from vectorquant.core.probability import normal_pdf, normal_cdf

    _prob_funcs = {
        "normal_pdf": normal_pdf,
        "normal_cdf": normal_cdf,
    }

    if dist not in _prob_funcs:
        return VerificationResult(
            verified=False, computed_value=None,
            expected_value=expected, confidence=0.0,
            method=dist, details=f"Unknown distribution: {dist}"
        )

    func = _prob_funcs[dist]
    mu = params.get("mu", 0.0)
    sigma = params.get("sigma", 1.0)

    try:
        computed = func(x, mu, sigma)
        diff = abs(computed - expected)
        verified = diff <= tolerance
        confidence = 1.0 if verified else max(0.0, 1.0 - diff)
        return VerificationResult(
            verified=verified,
            computed_value=round(computed, 6),
            expected_value=expected,
            confidence=round(confidence, 4),
            method=dist,
            details=f"P({dist}(x={x}, mu={mu}, sigma={sigma})) = {computed:.6f}"
        )
    except Exception as e:
        return VerificationResult(
            verified=False, computed_value=None,
            expected_value=expected, confidence=0.0,
            method=dist, details=f"Computation error: {str(e)}"
        )


def verify_finance_formula(formula_name, params, expected, tolerance=1e-4):
    """
    Validates a financial formula against VectorQuant's implementations.

    Supported formulas:
        - "black_scholes_call"
        - "black_scholes_put"
        - "parametric_var"
        - "sharpe_ratio"

    Args:
        formula_name: Name of the formula.
        params:       Dict of parameters.
        expected:     Expected result.
        tolerance:    Allowable error.
    
    Returns:
        VerificationResult
    """
    from vectorquant.finance.derivatives import black_scholes_call, black_scholes_put
    from vectorquant.finance.risk_models import parametric_var
    from vectorquant.core.statistics import mean, standard_deviation

    try:
        computed = None
        method = formula_name

        if formula_name == "black_scholes_call":
            computed = black_scholes_call(**params)
        elif formula_name == "black_scholes_put":
            computed = black_scholes_put(**params)
        elif formula_name == "parametric_var":
            computed = parametric_var(**params)
        elif formula_name == "sharpe_ratio":
            returns = params["returns"]
            rf = params.get("risk_free_rate", 0.0)
            mu = mean(returns)
            sigma = standard_deviation(returns)
            computed = (mu - rf) / sigma if sigma > 0 else 0.0
        else:
            return VerificationResult(
                verified=False, computed_value=None,
                expected_value=expected, confidence=0.0,
                method=formula_name,
                details=f"Unknown formula: {formula_name}"
            )

        diff = abs(computed - expected)
        verified = diff <= tolerance
        confidence = 1.0 if verified else max(0.0, 1.0 - diff / (abs(expected) + 1e-12))

        return VerificationResult(
            verified=verified,
            computed_value=round(computed, 6),
            expected_value=expected,
            confidence=round(confidence, 4),
            method=method,
            details=f"Computed {formula_name} = {computed:.6f}"
        )
    except Exception as e:
        return VerificationResult(
            verified=False, computed_value=None,
            expected_value=expected, confidence=0.0,
            method=formula_name,
            details=f"Computation error: {str(e)}"
        )
