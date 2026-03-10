"""
AI Reasoning Engine Tests

Comprehensive tests for the verification layer, proof traces,
hallucination detection, LLM tool interface, and reasoning pipeline.
"""

import sys
import os
import math
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq

# ═══════════════════════════════════════════════════════════
# 1. Verification Layer Tests
# ═══════════════════════════════════════════════════════════

def test_verify_calculation_correct():
    result = vq.ai.verify_calculation("sqrt(4) * 3", expected=6.0)
    assert result.verified == True
    assert result.confidence == 1.0

def test_verify_calculation_incorrect():
    result = vq.ai.verify_calculation("sqrt(4) * 3", expected=7.0)
    assert result.verified == False

def test_verify_calculation_complex():
    result = vq.ai.verify_calculation("exp(0)", expected=1.0)
    assert result.verified == True

def test_verify_probability_normal_pdf():
    result = vq.ai.verify_probability(
        "normal_pdf", {"mu": 0, "sigma": 1}, x=0, expected=0.3989, tolerance=0.001
    )
    assert result.verified == True

def test_verify_probability_unknown_dist():
    result = vq.ai.verify_probability(
        "unknown_dist", {}, x=0, expected=0.5
    )
    assert result.verified == False

def test_verify_finance_formula_bs_call():
    result = vq.ai.verify_finance_formula(
        "black_scholes_call",
        {"S": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0},
        expected=10.4506,
        tolerance=0.01
    )
    assert result.verified == True

def test_verify_finance_formula_sharpe():
    returns = [0.01, 0.02, -0.005, 0.015, 0.008]
    from vectorquant.core.statistics import mean, standard_deviation
    mu = mean(returns)
    sigma = standard_deviation(returns)
    expected_sharpe = mu / sigma
    
    result = vq.ai.verify_finance_formula(
        "sharpe_ratio",
        {"returns": returns},
        expected=expected_sharpe,
        tolerance=1e-4
    )
    assert result.verified == True


# ═══════════════════════════════════════════════════════════
# 2. Proof Trace Tests
# ═══════════════════════════════════════════════════════════

def test_explain_var_steps():
    returns = [0.01, -0.02, 0.015, -0.005, 0.008, -0.01, 0.02]
    trace = vq.ai.explain_var(returns, confidence=0.95)
    
    assert trace.method == "Parametric VaR"
    assert len(trace.steps) == 4
    assert trace.steps[0]["step"].startswith("mu")
    assert isinstance(trace.result, float)

def test_explain_sharpe_steps():
    returns = [0.01, 0.02, -0.005, 0.015, 0.008]
    trace = vq.ai.explain_sharpe(returns, risk_free_rate=0.0)
    
    assert trace.method == "Sharpe Ratio"
    assert len(trace.steps) == 4
    assert isinstance(trace.result, float)

def test_explain_black_scholes_steps():
    trace = vq.ai.explain_black_scholes(S=100, K=100, r=0.05, sigma=0.2, T=1.0)
    
    assert trace.method == "Black-Scholes Call"
    assert len(trace.steps) == 5
    assert trace.result > 0


# ═══════════════════════════════════════════════════════════
# 3. Hallucination Detection Tests
# ═══════════════════════════════════════════════════════════

def test_check_formula_correct():
    result = vq.ai.check_formula("sharpe_ratio", "(mu - r_f) / sigma")
    assert result.is_correct == True
    assert result.confidence == 1.0

def test_check_formula_hallucination():
    # Common hallucination: Sharpe = mean / variance (wrong!)
    result = vq.ai.check_formula("sharpe_ratio", "mu / variance")
    assert result.is_correct == False
    assert result.confidence == 0.0

def test_check_formula_unknown():
    result = vq.ai.check_formula("unknown_formula", "x + y")
    assert result.is_correct == False

def test_check_numerical_claim():
    result = vq.ai.check_numerical_claim(
        "black_scholes_call",
        claimed_value=10.45,
        params={"S": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0},
        tolerance=0.01
    )
    assert result.is_correct == True

def test_validate_prediction():
    result = vq.ai.validate_prediction(
        hypothesis="Stock will reach 200 from 100 in 1 year",
        S0=100, mu=0.05, sigma=0.2, T=1.0, target_price=200,
        n_simulations=5000
    )
    # Very unlikely to double with 5% drift and 20% vol
    assert result.is_correct == False
    assert result.confidence < 0.1


# ═══════════════════════════════════════════════════════════
# 4. LLM Tool Interface Tests
# ═══════════════════════════════════════════════════════════

def test_tool_registry_exists():
    registry = vq.ai.get_tool_registry()
    assert "calculate_var" in registry
    assert "price_call_option" in registry
    assert "optimize_portfolio" in registry
    assert "simulate_gbm" in registry
    assert "compute_sharpe" in registry

def test_execute_tool_var():
    returns = [0.01, -0.02, 0.015, -0.005, 0.008]
    result = vq.ai.execute_tool("calculate_var", returns=returns, confidence_level=0.95)
    assert isinstance(result, float)

def test_execute_tool_unknown():
    with pytest.raises(ValueError):
        vq.ai.execute_tool("nonexistent_tool", x=1)


# ═══════════════════════════════════════════════════════════
# 5. Reasoning Pipeline Tests
# ═══════════════════════════════════════════════════════════

def test_reasoning_engine_var():
    returns = [0.01, -0.02, 0.015, -0.005, 0.008, -0.01, 0.02]
    engine = vq.ai.ReasoningEngine()
    result = engine.solve("var", returns=returns, confidence=0.95)
    
    assert result.verified == True
    assert result.confidence == 1.0
    assert result.method == "Parametric VaR"
    assert result.proof_trace is not None

def test_reasoning_engine_sharpe():
    returns = [0.01, 0.02, -0.005, 0.015, 0.008]
    engine = vq.ai.ReasoningEngine()
    result = engine.solve("sharpe", returns=returns)
    
    assert result.verified == True
    assert result.method == "Sharpe Ratio"

def test_reasoning_engine_black_scholes():
    engine = vq.ai.ReasoningEngine()
    result = engine.solve("black_scholes", S=100, K=100, r=0.05, sigma=0.2, T=1.0)
    
    assert result.verified == True
    assert result.result > 0

def test_reasoning_engine_unknown():
    engine = vq.ai.ReasoningEngine()
    result = engine.solve("unknown_question")
    
    assert result.verified == False
    assert result.confidence == 0.0
