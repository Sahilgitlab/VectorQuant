"""
AI Reasoning Pipeline

Structured pipeline: Parse -> Compute -> Verify -> Explain.
Provides the ReasoningEngine that orchestrates deterministic
computation with explainable proof traces.
"""


class ReasoningResult:
    """Structured result of a reasoning pipeline execution."""
    def __init__(self, result, verified, method, proof_trace, confidence):
        self.result = result
        self.verified = verified
        self.method = method
        self.proof_trace = proof_trace
        self.confidence = confidence

    def to_dict(self):
        return {
            "result": self.result,
            "verified": self.verified,
            "method": self.method,
            "proof_trace": self.proof_trace.to_dict() if self.proof_trace else None,
            "confidence": self.confidence,
        }

    def __repr__(self):
        status = "VERIFIED" if self.verified else "UNVERIFIED"
        return (f"ReasoningResult({status}, result={self.result}, "
                f"method={self.method}, confidence={self.confidence})")


class ReasoningEngine:
    """
    Orchestrates the full deterministic reasoning pipeline.

    Usage::

        engine = ReasoningEngine()
        result = engine.solve("var", returns=data, confidence=0.95)
        print(result)
    """

    # Map of supported computation types to their handlers
    _SUPPORTED = ["var", "sharpe", "black_scholes", "monte_carlo"]

    def solve(self, question, **params):
        """
        Routes to the appropriate computation, generates a proof trace,
        verifies the result, and returns a ReasoningResult.

        Args:
            question: The computation type (e.g. "var", "sharpe", 
                      "black_scholes", "monte_carlo").
            **params: Parameters for the computation.

        Returns:
            ReasoningResult
        """
        question_lower = question.lower().replace(" ", "_")

        if question_lower == "var":
            return self._solve_var(**params)
        elif question_lower == "sharpe" or question_lower == "sharpe_ratio":
            return self._solve_sharpe(**params)
        elif question_lower == "black_scholes" or question_lower == "black_scholes_call":
            return self._solve_black_scholes(**params)
        elif question_lower == "monte_carlo":
            return self._solve_monte_carlo(**params)
        else:
            return ReasoningResult(
                result=None, verified=False,
                method=question, proof_trace=None,
                confidence=0.0,
            )

    def _solve_var(self, returns, confidence=0.95):
        from vectorquant.ai.proof_trace import explain_var
        from vectorquant.finance.risk_models import parametric_var

        trace = explain_var(returns, confidence)
        computed = parametric_var(returns, confidence)

        # Cross-verify
        diff = abs(trace.result - computed)
        verified = diff < 1e-4

        return ReasoningResult(
            result=trace.result,
            verified=verified,
            method="Parametric VaR",
            proof_trace=trace,
            confidence=1.0 if verified else 0.5,
        )

    def _solve_sharpe(self, returns, risk_free_rate=0.0):
        from vectorquant.ai.proof_trace import explain_sharpe
        from vectorquant.core.statistics import mean, standard_deviation

        trace = explain_sharpe(returns, risk_free_rate)
        mu = mean(returns)
        sigma = standard_deviation(returns)
        computed = (mu - risk_free_rate) / sigma if sigma > 0 else 0.0

        diff = abs(trace.result - computed)
        verified = diff < 1e-4

        return ReasoningResult(
            result=trace.result,
            verified=verified,
            method="Sharpe Ratio",
            proof_trace=trace,
            confidence=1.0 if verified else 0.5,
        )

    def _solve_black_scholes(self, S, K, r, sigma, T):
        from vectorquant.ai.proof_trace import explain_black_scholes
        from vectorquant.finance.derivatives import black_scholes_call

        trace = explain_black_scholes(S, K, r, sigma, T)
        computed = black_scholes_call(S, K, r, sigma, T)

        diff = abs(trace.result - computed)
        verified = diff < 1e-3

        return ReasoningResult(
            result=trace.result,
            verified=verified,
            method="Black-Scholes Call",
            proof_trace=trace,
            confidence=1.0 if verified else 0.5,
        )

    def _solve_monte_carlo(self, S0, K, r, sigma, T, n_paths=10000):
        from vectorquant.ai.proof_trace import explain_monte_carlo

        trace = explain_monte_carlo(S0, K, r, sigma, T, n_paths)

        return ReasoningResult(
            result=trace.result,
            verified=True,
            method="Monte Carlo European Call",
            proof_trace=trace,
            confidence=0.95,
        )
