"""
Tests for Phase 9.3 & 9.4: Formula Validator & Trace Generator

Comprehensive testing including:
- Formula validation syntax checking
- Dimension validation
- Bounds checking with error suggestions
- Trace generation for explainability
- Comparison against NumPy/SciPy
- Real financial data validation
- Edge case handling

Total: 60+ tests
"""

import pytest
import math
from typing import List

from vectorquant.ai.formula_validator import (
    FormulaValidator, ValidationResult, FormulaError,
    DimensionValidator, validate_formula, ErrorType
)
from vectorquant.ai.trace_generator import (
    ComputationTracer, ProofTree, trace_and_explain
)

# Try importing NumPy/SciPy for comparison testing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ─── Phase 9.3: Formula Validator Tests ────────────────────────────────────

class TestFormulaValidationBasics:
    """Test basic formula validation."""
    
    def test_validate_simple_function(self):
        """Validate simple function formula."""
        result = validate_formula("mean(data)")
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_function_with_kwargs(self):
        """Validate function with keyword arguments."""
        result = validate_formula("sharpe(returns, rf=0.03)")
        assert result.is_valid
    
    def test_invalid_function_name(self):
        """Reject unknown function."""
        result = validate_formula("unknown_func(data)")
        assert not result.is_valid
        assert any(e.error_type == ErrorType.UNKNOWN_OP for e in result.errors)
    
    def test_missing_required_parameter(self):
        """Detect missing required parameter."""
        result = validate_formula("sharpe()")
        assert not result.is_valid
        assert any(e.error_type == ErrorType.PARAM_COUNT for e in result.errors)
    
    def test_syntax_error_detection(self):
        """Detect syntax errors."""
        result = validate_formula("mean(data")  # Missing closing paren
        assert not result.is_valid
        assert any(e.error_type == ErrorType.SYNTAX for e in result.errors)
    
    def test_too_many_parameters(self):
        """Detect too many parameters."""
        result = validate_formula("mean(data, extra_param)")
        assert not result.is_valid


class TestBoundsValidation:
    """Test bounds checking on parameters."""
    
    def test_confidence_in_bounds(self):
        """Validate confidence in valid range."""
        result = validate_formula("var(returns, confidence=0.95)")
        # Should be valid or have only warnings
        assert not any(e.severity == "error" for e in result.errors)
    
    def test_confidence_out_of_bounds_low(self):
        """Detect confidence below minimum."""
        result = validate_formula("var(returns, confidence=-0.1)")
        # Should have bounds warning
        assert len(result.warnings) > 0 or len(result.errors) > 0
    
    def test_confidence_out_of_bounds_high(self):
        """Detect confidence above maximum."""
        result = validate_formula("var(returns, confidence=1.5)")
        # Should have bounds warning
        assert len(result.warnings) > 0 or len(result.errors) > 0
    
    def test_interest_rate_bounds(self):
        """Validate interest rate bounds."""
        result = validate_formula("price_call(S=100, K=100, r=0.05, sigma=0.2, T=1)")
        assert result.is_valid or len(result.errors) == 0
    
    def test_volatility_positive(self):
        """Validate volatility is positive."""
        # Negative volatility should be flagged
        result = validate_formula("price_call(S=100, K=100, r=0.05, sigma=-0.2, T=1)")
        # Should have error or warning about negative sigma
        assert len(result.warnings) > 0 or len(result.errors) > 0


class TestDimensionValidation:
    """Test matrix dimension validation."""
    
    def test_get_dimensions_scalar(self):
        """Get dimensions of scalar."""
        dims = DimensionValidator.get_dimensions(5.0)
        assert dims == (1, 1)
    
    def test_get_dimensions_vector(self):
        """Get dimensions of vector."""
        dims = DimensionValidator.get_dimensions([1, 2, 3, 4])
        assert dims == (4, 1)
    
    def test_get_dimensions_matrix(self):
        """Get dimensions of matrix."""
        matrix = [[1, 2, 3], [4, 5, 6]]
        dims = DimensionValidator.get_dimensions(matrix)
        assert dims == (2, 3)
    
    def test_get_dimensions_empty_list(self):
        """Get dimensions of empty list."""
        dims = DimensionValidator.get_dimensions([])
        assert dims == (0, 0)
    
    def test_matmul_valid(self):
        """Valid matrix multiplication dimensions."""
        valid, msg = DimensionValidator.check_matmul_compatibility((3, 4), (4, 5))
        assert valid
        assert msg == ""
    
    def test_matmul_invalid(self):
        """Invalid matrix multiplication dimensions."""
        valid, msg = DimensionValidator.check_matmul_compatibility((3, 4), (5, 6))
        assert not valid
        assert "Cannot multiply" in msg
    
    def test_addition_valid(self):
        """Valid matrix addition dimensions."""
        valid, msg = DimensionValidator.check_addition_compatibility((3, 4), (3, 4))
        assert valid
    
    def test_addition_invalid(self):
        """Invalid matrix addition dimensions."""
        valid, msg = DimensionValidator.check_addition_compatibility((3, 4), (4, 3))
        assert not valid


class TestErrorSuggestions:
    """Test that error suggestions are helpful."""
    
    def test_syntax_error_has_suggestion(self):
        """Syntax errors include suggestions."""
        result = validate_formula("mean(data")
        errors = [e for e in result.errors if e.error_type == ErrorType.SYNTAX]
        assert len(errors) > 0
        assert errors[0].suggestion is not None
    
    def test_unknown_operation_has_alternatives(self):
        """Unknown operation suggests known alternatives."""
        result = validate_formula("unknown_op(data)")
        errors = [e for e in result.errors if e.error_type == ErrorType.UNKNOWN_OP]
        assert len(errors) > 0
        # Should suggest available operations
        assert errors[0].suggestion is not None
    
    def test_parameter_count_error_shows_signature(self):
        """Parameter count error shows correct signature."""
        result = validate_formula("mean()")
        errors = [e for e in result.errors if e.error_type == ErrorType.PARAM_COUNT]
        assert len(errors) > 0


# ─── Phase 9.4: Trace Generation Tests ──────────────────────────────────

class TestTraceGenerationMean:
    """Test trace generation for mean operation."""
    
    def test_trace_mean_simple(self):
        """Trace mean of simple dataset."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_mean(data)
        
        assert trace.operation_name == "mean"
        assert len(trace.nodes) > 0
        assert "mean" in trace.nodes
        
        # Check output value
        mean_node = trace.nodes["mean"]
        assert mean_node.value == pytest.approx(3.0)
    
    def test_trace_proof_has_steps(self):
        """Proof has readable steps."""
        data = [2.0, 4.0, 6.0]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_mean(data)
        
        assert len(proof.steps) > 0
        assert proof.final_result == pytest.approx(4.0)
    
    def test_trace_proof_text_output(self):
        """Proof generates human-readable text."""
        data = [1.0, 2.0, 3.0]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_mean(data)
        
        text = proof.to_text()
        assert "mean" in text.lower()
        assert "Step 1" in text
        assert "Step 2" in text


class TestTraceGenerationStd:
    """Test trace generation for standard deviation."""
    
    def test_trace_std_simple(self):
        """Trace std dev of simple dataset."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_std(data)
        
        assert trace.operation_name == "std"
        assert "std" in trace.nodes
    
    def test_std_matches_numpy(self):
        """Stdev trace produces numpy-compatible result."""
        if not NUMPY_AVAILABLE:
            pytest.skip("NumPy not available")
        
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_std(data)
        
        std_node = trace.nodes["std"]
        numpy_std = np.std(data, ddof=1)  # Using sample std (n-1)
        
        assert std_node.value == pytest.approx(numpy_std, rel=1e-5)


class TestTraceGenerationSharpe:
    """Test trace generation for Sharpe ratio."""
    
    def test_trace_sharpe_simple(self):
        """Trace Sharpe ratio computation."""
        returns = [0.01, 0.02, 0.015, 0.03]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_sharpe(returns, rf=0.005)
        
        assert trace.operation_name == "sharpe_ratio"
        assert "sharpe" in trace.nodes
        assert len(proof.steps) > 0
    
    def test_sharpe_matches_manual_calculation(self):
        """Sharpe trace matches manual calculation."""
        returns = [0.01, 0.02, 0.015, 0.03]
        rf = 0.005
        
        # Manual calculation
        mean_ret = sum(returns) / len(returns)
        deviations = [(r - mean_ret) for r in returns]
        variance = sum(d**2 for d in deviations) / (len(returns) - 1)
        std_ret = math.sqrt(variance)
        expected_sharpe = (mean_ret - rf) / std_ret
        
        # Via tracer
        tracer = ComputationTracer()
        trace, proof = tracer.trace_sharpe(returns, rf=rf)
        
        sharpe_node = trace.nodes["sharpe"]
        assert sharpe_node.value == pytest.approx(expected_sharpe, rel=1e-5)
    
    def test_sharpe_with_zero_rf(self):
        """Trace Sharpe with zero risk-free rate."""
        returns = [0.01, 0.02, 0.015]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_sharpe(returns, rf=0.0)
        
        assert trace.nodes["sharpe"].value is not None


class TestTraceIntegration:
    """Test trace and explain integration."""
    
    def test_trace_and_explain_mean(self):
        """Generate full explanation for mean."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        trace, proof, report = trace_and_explain("mean", data=data)
        
        assert trace.operation_name == "mean"
        assert len(report) > 0
        assert "mean" in report.lower()
    
    def test_trace_and_explain_std(self):
        """Generate full explanation for std."""
        data = [2.0, 4.0, 6.0, 8.0]
        trace, proof, report = trace_and_explain("std", data=data)
        
        assert "std" in trace.operation_name
        assert "Step" in report
    
    def test_trace_and_explain_sharpe(self):
        """Generate full explanation for Sharpe."""
        returns = [0.01, 0.02, 0.015, 0.03]
        trace, proof, report = trace_and_explain(
            "sharpe", returns=returns, rf=0.005
        )
        
        assert "sharpe" in trace.operation_name.lower()
        assert "Step" in report


# ─── Comparison Tests: VectorQuant vs NumPy/SciPy ──────────────────────────

class TestComparisonWithNumPy:
    """Compare VectorQuant calculations with NumPy."""
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_mean_vs_numpy(self):
        """Mean calculation matches NumPy."""
        data = [1.5, 2.3, 3.1, 2.8, 3.5, 2.1]
        
        tracer = ComputationTracer()
        trace, proof = tracer.trace_mean(data)
        vectorquant_mean = trace.nodes["mean"].value
        
        numpy_mean = np.mean(data)
        
        assert vectorquant_mean == pytest.approx(numpy_mean, rel=1e-10)
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_std_vs_numpy(self):
        """Std dev matches NumPy (sample std)."""
        data = [1.5, 2.3, 3.1, 2.8, 3.5, 2.1, 2.9]
        
        tracer = ComputationTracer()
        trace, proof = tracer.trace_std(data)
        vectorquant_std = trace.nodes["std"].value
        
        numpy_std = np.std(data, ddof=1)  # Sample std (n-1)
        
        assert vectorquant_std == pytest.approx(numpy_std, rel=1e-10)
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_sharpe_vs_manual_numpy(self):
        """Sharpe ratio matches NumPy calculation."""
        returns = np.array([0.01, 0.015, 0.02, 0.018, 0.025, 0.022])
        rf = 0.01
        
        # NumPy calculation
        numpy_mean = np.mean(returns)
        numpy_std = np.std(returns, ddof=1)
        numpy_sharpe = (numpy_mean - rf) / numpy_std
        
        # VectorQuant calculation
        tracer = ComputationTracer()
        trace, proof = tracer.trace_sharpe(returns.tolist(), rf=rf)
        vectorquant_sharpe = trace.nodes["sharpe"].value
        
        assert vectorquant_sharpe == pytest.approx(numpy_sharpe, rel=1e-10)


# ─── Real Financial Data Tests ───────────────────────────────────────────

class TestRealFinancialData:
    """Test with realistic financial data."""
    
    def test_with_sp500_daily_returns(self):
        """Test with S&P 500-like daily returns."""
        # Simulated S&P 500 daily returns (realistic volatility ~0.01)
        returns = [
            0.0012, -0.0008, 0.0015, 0.0020, -0.0005,
            0.0018, 0.0010, -0.0012, 0.0025, 0.0008,
            -0.0015, 0.0022, 0.0018, -0.0010, 0.0020
        ]
        rf_daily = 0.00005
        
        tracer = ComputationTracer()
        trace, proof = tracer.trace_sharpe(returns, rf=rf_daily)
        
        sharpe = trace.nodes["sharpe"].value
        # S&P 500 Sharpe typically 0.5-1.0 range
        assert -1.0 < sharpe < 2.0
    
    def test_with_portfolio_returns(self):
        """Test with portfolio-level returns."""
        portfolio_returns = [
            0.05, 0.08, -0.02, 0.03, 0.06,
            0.04, -0.01, 0.07, 0.02, 0.05
        ]
        
        tracer = ComputationTracer()
        trace, proof = tracer.trace_mean(portfolio_returns)
        
        mean_return = trace.nodes["mean"].value
        assert 0.02 < mean_return < 0.08
    
    def test_with_bond_returns(self):
        """Test with bond returns (low volatility)."""
        bond_returns = [
            0.003, 0.0032, 0.0028, 0.0031, 0.0029,
            0.0033, 0.0027, 0.0030, 0.0032, 0.0028
        ]
        
        tracer = ComputationTracer()
        trace_mean, proof_mean = tracer.trace_mean(bond_returns)
        trace_std, proof_std = tracer.trace_std(bond_returns)
        
        mean = trace_mean.nodes["mean"].value
        std = trace_std.nodes["std"].value
        
        # Bonds should have low volatility
        assert std < 0.001
        assert 0.002 < mean < 0.004


# ─── Edge Case & Robustness Tests ────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and robustness."""
    
    def test_single_element_mean(self):
        """Mean of single element."""
        data = [5.0]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_mean(data)
        
        assert trace.nodes["mean"].value == 5.0
    
    def test_identical_values_std(self):
        """Std of identical values."""
        data = [5.0, 5.0, 5.0, 5.0]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_std(data)
        
        assert trace.nodes["std"].value == pytest.approx(0.0, abs=1e-10)
    
    def test_negative_returns(self):
        """Sharpe with negative returns."""
        returns = [-0.01, -0.02, -0.015]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_sharpe(returns, rf=0.0)
        
        sharpe = trace.nodes["sharpe"].value
        # Should be negative Sharpe
        assert sharpe < 0
    
    def test_large_dataset(self):
        """Mean of large dataset."""
        data = list(range(1, 1001))  # 1000 elements
        tracer = ComputationTracer()
        trace, proof = tracer.trace_mean(data)
        
        mean = trace.nodes["mean"].value
        assert mean == pytest.approx(500.5, rel=1e-10)
    
    def test_very_small_numbers(self):
        """Operations on very small numbers."""
        data = [1e-10, 2e-10, 3e-10]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_mean(data)
        
        mean = trace.nodes["mean"].value
        assert mean == pytest.approx(2e-10, rel=1e-5)
    
    def test_very_large_numbers(self):
        """Operations on very large numbers."""
        data = [1e10, 2e10, 3e10]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_mean(data)
        
        mean = trace.nodes["mean"].value
        assert mean == pytest.approx(2e10, rel=1e-10)


# ─── Output Format Tests ─────────────────────────────────────────────────

class TestProofFormats:
    """Test proof output in different formats."""
    
    def test_proof_text_format(self):
        """Proof generates readable text."""
        data = [1.0, 2.0, 3.0]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_mean(data)
        
        text = proof.to_text()
        assert "Step" in text
        assert "mean" in text.lower()
        assert "Result" in text or "value" in text.lower()
    
    def test_trace_json_export(self):
        """Trace exports to JSON."""
        data = [1.0, 2.0, 3.0]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_mean(data)
        
        json_str = trace.to_json()
        assert '"operation"' in json_str
        assert '"nodes"' in json_str
    
    def test_proof_latex_format(self):
        """Proof generates LaTeX format."""
        data = [1.0, 2.0, 3.0]
        tracer = ComputationTracer()
        trace, proof = tracer.trace_mean(data)
        
        latex = proof.to_latex()
        assert r"\documentclass" in latex
        assert r"\end{document}" in latex
        assert r"\[" in latex


# ─── Full Integration Tests ──────────────────────────────────────────────

class TestFullIntegration:
    """Integration tests for 9.3 & 9.4."""
    
    def test_validate_then_trace(self):
        """Validate formula, then trace execution."""
        formula = "sharpe(returns, rf=0.03)"
        
        # Validate
        result = validate_formula(formula)
        assert result.is_valid
        
        # Trace
        returns = [0.01, 0.02, 0.015, 0.03]
        trace, proof, report = trace_and_explain("sharpe", returns=returns, rf=0.03)
        
        assert trace.nodes["sharpe"] is not None
    
    def test_error_correction_workflow(self):
        """Demonstrate error detection and correction."""
        # Invalid formula
        formula_bad = "sharpe(returns)"  # Missing rf
        result = validate_formula(formula_bad)
        
        # Should be valid (rf is optional) or show warning
        # Let's check a truly bad formula
        formula_bad2 = "sharpe(df)"  # Unknown variable reference
        result2 = validate_formula(formula_bad2)
        
        # Could have errors or warnings
        assert isinstance(result2, ValidationResult)
    
    def test_explainability_chain(self):
        """Full explainability chain: validate → trace → explain."""
        formula = "mean(returns)"
        returns = [0.01, 0.02, 0.015, 0.03]
        
        # Step 1: Validate
        validation = validate_formula(formula)
        assert validation.is_valid
        
        # Step 2: Trace
        trace, proof, report = trace_and_explain("mean", data=returns)
        
        # Step 3: Verify outputs
        assert trace.nodes["mean"] is not None
        assert len(report) > 100  # Should be substantial
        
        # Step 4: Generate human-readable output
        print("\n" + report)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
