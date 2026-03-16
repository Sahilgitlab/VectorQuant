"""
Verification Pipeline (5-Stage) for AI Hallucination Detection

Detects and prevents AI numerical hallucinations by comparing LLM
reasoning against deterministic VectorQuant computations.

5-Stage Pipeline:
1. EXTRACT: Pull mathematical expressions from LLM reasoning
2. PARSE: Convert expressions to VectorQuant operations
3. EXECUTE: Run in deterministic VectorQuant (C backend)
4. COMPARE: Check LLM output vs ground truth
5. REPORT: Generate verification confidence scores

Example:
    LLM says: "The Sharpe ratio is 1.25"
    User calls: verify_llm_statement(
        "The Sharpe ratio is 1.25",
        returns=[0.01, 0.02, ...],
        risk_free_rate=0.03
    )
    Result: {verified: True, confidence: 0.987, error: 0.0001}
"""

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional, Union
from enum import Enum

from .expression_parser import (
    parse_expression, ParsedExpression, FunctionCall
)


class VerificationStage(Enum):
    """Verification pipeline stages."""
    EXTRACT = "extract"
    PARSE = "parse"
    EXECUTE = "execute"
    COMPARE = "compare"
    REPORT = "report"


@dataclass
class ExtractionResult:
    """Result of Stage 1: Expression extraction."""
    raw_text: str
    extracted_expressions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    found_numbers: List[float] = field(default_factory=list)
    extraction_method: str = "regex"
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ParseResult:
    """Result of Stage 2: Expression parsing."""
    expression: str
    parsed_ast: Optional[Dict] = None
    is_valid: bool = False
    error_message: str = ""
    operations: List[str] = field(default_factory=list)
    variables_needed: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExecutionResult:
    """Result of Stage 3: Execution in VectorQuant."""
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    computed_value: Optional[float] = None
    backend: str = "python"
    latency_ms: float = 0.0
    error: Optional[str] = None
    success: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ComparisonResult:
    """Result of Stage 4: Comparing values."""
    expression: str
    llm_value: float
    computed_value: float
    absolute_error: float = 0.0
    relative_error: float = 0.0
    tolerance: float = 1e-6
    matches: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VerificationReport:
    """Result of Stage 5: Final verification report."""
    timestamp: str
    original_statement: str
    extracted_expressions: List[str]
    parse_results: List[ParseResult]
    execution_results: List[ExecutionResult]
    comparison_results: List[ComparisonResult]
    overall_verified: bool
    confidence_score: float
    hallucination_detected: bool
    error_summary: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "original_statement": self.original_statement,
            "extracted_expressions": self.extracted_expressions,
            "parse_results": [p.to_dict() for p in self.parse_results],
            "execution_results": [e.to_dict() for e in self.execution_results],
            "comparison_results": [c.to_dict() for c in self.comparison_results],
            "overall_verified": self.overall_verified,
            "confidence_score": round(self.confidence_score, 4),
            "hallucination_detected": self.hallucination_detected,
            "error_summary": self.error_summary
        }


class ExpressionExtractor:
    """Stage 1: Extract mathematical expressions from text."""
    
    # Patterns to extract expressions
    PATTERNS = [
        # "result is X" patterns
        r'(?:is|equals?|=)\s*([-+]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
        # "function(args)" patterns
        r'([a-zA-Z_]\w*\s*\([^)]*\))',
        # Mathematical expressions with operators
        r'([\w_]+\s*(?:[+\-*/^]\s*)?[\(\w\.\d]+[\)]*)',
    ]
    
    @staticmethod
    def extract(text: str) -> ExtractionResult:
        """
        Extract mathematical expressions and numeric values from text.
        
        Args:
            text: LLM output text containing expressions
        
        Returns:
            ExtractionResult with extracted expressions and values
        """
        result = ExtractionResult(raw_text=text)
        
        # Extract numeric values
        numbers = re.findall(r'[-+]?\d+\.?\d*(?:[eE][+-]?\d+)?', text)
        result.found_numbers = [float(n) for n in numbers]
        
        # Extract expressions: Look for function calls and equations
        # Pattern 1: function(something)
        func_pattern = r'([a-zA-Z_]\w*)\s*\(\s*([^)]+(?:\([^)]*\))?[^)]*)\)'
        matches = re.finditer(func_pattern, text)
        
        for match in matches:
            func_name = match.group(1)
            args = match.group(2)
            expression = f"{func_name}({args})"
            result.extracted_expressions.append(expression)
        
        # Pattern 2: "is/equals X" where X is a number
        eq_pattern = r'(?:is|equal\s+to|equals|=)\s*([-+]?\d+\.?\d*(?:[eE][+-]?\d+)?)'
        matches = re.finditer(eq_pattern, text, re.IGNORECASE)
        
        for match in matches:
            value_str = match.group(1)
            # Extract preceding context to find variable name
            start = max(0, match.start() - 100)
            context = text[start:match.start()]
            
            # Try to find variable name before "is"
            var_match = re.search(r'(\b[a-zA-Z_]\w*)\s+(?:is|equal)', context, re.IGNORECASE)
            if var_match:
                var_name = var_match.group(1)
                result.extracted_expressions.append(f"{var_name}={value_str}")
        
        # Remove duplicates while preserving order
        result.extracted_expressions = list(dict.fromkeys(result.extracted_expressions))
        
        return result


class StageExecutor:
    """Stage 3: Execute operations in VectorQuant."""
    
    # Map operation names to VectorQuant functions
    OPERATION_MAP = {
        # Statistics
        "mean": ("vectorquant.core.statistics", "mean"),
        "compute_mean": ("vectorquant.core.statistics", "mean"),
        "std": ("vectorquant.core.statistics", "standard_deviation"),
        "compute_std": ("vectorquant.core.statistics", "standard_deviation"),
        "variance": ("vectorquant.core.statistics", "variance"),
        "compute_variance": ("vectorquant.core.statistics", "variance"),
        "covariance": ("vectorquant.core.statistics", "covariance"),
        "compute_covariance": ("vectorquant.core.statistics", "covariance"),
        
        # Risk
        "sharpe": ("vectorquant.ai.agent_interface", "_sharpe_ratio_wrapper"),
        "sharpe_ratio": ("vectorquant.ai.agent_interface", "_sharpe_ratio_wrapper"),
        "compute_sharpe": ("vectorquant.ai.agent_interface", "_sharpe_ratio_wrapper"),
        "var": ("vectorquant.finance.risk_models", "parametric_var"),
        "value_at_risk": ("vectorquant.finance.risk_models", "parametric_var"),
        "compute_var": ("vectorquant.finance.risk_models", "parametric_var"),
        
        # Derivatives
        "call": ("vectorquant.finance.derivatives", "black_scholes_call"),
        "price_call": ("vectorquant.finance.derivatives", "black_scholes_call"),
        "put": ("vectorquant.finance.derivatives", "black_scholes_put"),
        "price_put": ("vectorquant.finance.derivatives", "black_scholes_put"),
    }
    
    @staticmethod
    def execute(operation: str, parameters: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a VectorQuant operation.
        
        Args:
            operation: Operation name (e.g., "mean", "sharpe_ratio")
            parameters: Dict of parameters for the operation
        
        Returns:
            ExecutionResult with computed value or error
        """
        result = ExecutionResult(operation=operation, parameters=parameters)
        
        # Check if operation is registered
        if operation not in StageExecutor.OPERATION_MAP:
            result.success = False
            result.error = f"Unknown operation: {operation}"
            return result
        
        try:
            import time
            start_time = time.perf_counter()
            
            # Load and execute operation
            module_name, func_name = StageExecutor.OPERATION_MAP[operation]
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name)
            
            # Execute with parameters
            value = func(**parameters)
            
            end_time = time.perf_counter()
            result.computed_value = float(value) if value is not None else None
            result.latency_ms = (end_time - start_time) * 1000
            result.success = True
            result.backend = "c" if hasattr(func, "__c_func__") else "python"
            
        except ImportError as e:
            result.success = False
            result.error = f"Module not found: {str(e)}"
        except AttributeError as e:
            result.success = False
            result.error = f"Function not found: {str(e)}"
        except TypeError as e:
            result.success = False
            result.error = f"Parameter error: {str(e)}"
        except Exception as e:
            result.success = False
            result.error = f"Execution error: {str(e)}"
        
        return result


class StageComparator:
    """Stage 4: Compare computed vs LLM values."""
    
    @staticmethod
    def compare(expression: str, llm_value: float, computed_value: float,
                tolerance: float = 1e-6) -> ComparisonResult:
        """
        Compare LLM-claimed value with computed value.
        
        Args:
            expression: The expression being verified
            llm_value: Value claimed by LLM
            computed_value: Deterministically computed value
            tolerance: Absolute error tolerance
        
        Returns:
            ComparisonResult
        """
        result = ComparisonResult(
            expression=expression,
            llm_value=llm_value,
            computed_value=computed_value,
            tolerance=tolerance
        )
        
        # Calculate errors
        result.absolute_error = abs(llm_value - computed_value)
        
        # Relative error (avoid division by zero)
        if abs(computed_value) > 1e-10:
            result.relative_error = result.absolute_error / abs(computed_value)
        else:
            result.relative_error = float('inf') if result.absolute_error > 1e-10 else 0.0
        
        # Check if within tolerance
        result.matches = result.absolute_error <= tolerance
        
        return result


class VerificationPipeline:
    """5-Stage Verification Pipeline."""
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize the pipeline.
        
        Args:
            tolerance: Absolute error tolerance for comparison
        """
        self.tolerance = tolerance
        self.extractor = ExpressionExtractor()
        self.executor = StageExecutor()
        self.comparator = StageComparator()
    
    def verify(self, llm_statement: str, expression: str, 
               llm_value: float, variables: Optional[Dict[str, Any]] = None) -> VerificationReport:
        """
        Run the full 5-stage verification pipeline.
        
        Args:
            llm_statement: The original LLM output/reasoning
            expression: Mathematical expression to verify (e.g., "sharpe(returns, rf=0.03)")
            llm_value: Numeric value claimed by LLM
            variables: Dict of variable values needed for execution (e.g., {"returns": [...]})
        
        Returns:
            VerificationReport with full pipeline results
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        extracted_expressions = []
        parse_results = []
        execution_results = []
        comparison_results = []
        errors = []
        
        try:
            # STAGE 1: EXTRACT
            # Extract expressions from the raw LLM statement
            extraction = self.extractor.extract(llm_statement)
            extracted_expressions = extraction.extracted_expressions
            
            if not extracted_expressions and expression:
                # Use provided expression if extraction found nothing
                extracted_expressions = [expression]
            
            # STAGE 2: PARSE
            # For each extracted expression, parse it
            for expr_str in extracted_expressions:
                success, result = parse_expression(expr_str)
                
                if success:
                    parse_result = ParseResult(
                        expression=expr_str,
                        parsed_ast=result.to_dict(),
                        is_valid=True,
                        operations=result.operations,
                        variables_needed=list(result.variable_types.keys())
                    )
                else:
                    parse_result = ParseResult(
                        expression=expr_str,
                        is_valid=False,
                        error_message=str(result)
                    )
                    errors.append(f"Parse error in '{expr_str}': {result}")
                
                parse_results.append(parse_result)
            
            # STAGE 3: EXECUTE
            # For valid parsed expressions, execute them
            if variables is None:
                variables = {}
            
            # Extract the primary operation from the parsed expression
            for parse_result in parse_results:
                if parse_result.is_valid and parse_result.operations:
                    primary_op = parse_result.operations[0]  # First operation
                    
                    # Prepare parameters for execution
                    exec_params = dict(variables)
                    
                    # Execute the operation
                    exec_result = self.executor.execute(primary_op, exec_params)
                    execution_results.append(exec_result)
                    
                    if not exec_result.success:
                        errors.append(f"Execution error for '{primary_op}': {exec_result.error}")
            
            # STAGE 4: COMPARE
            # Compare computed values with LLM values
            for exec_result in execution_results:
                if exec_result.success and exec_result.computed_value is not None:
                    comparison = self.comparator.compare(
                        exec_result.operation,
                        llm_value,
                        exec_result.computed_value,
                        self.tolerance
                    )
                    comparison_results.append(comparison)
            
        except Exception as e:
            errors.append(f"Pipeline error: {str(e)}")
        
        # STAGE 5: REPORT
        # Generate final report
        overall_verified = (
            len(comparison_results) > 0 and
            all(c.matches for c in comparison_results)
        )
        
        hallucination_detected = (
            len(comparison_results) > 0 and
            any(not c.matches for c in comparison_results)
        )
        
        # Confidence score calculation
        if not comparison_results:
            confidence_score = 0.0 if errors else 0.5  # Neutral if can't verify
        else:
            # Average confidence based on error magnitudes
            confidences = []
            for comp in comparison_results:
                if comp.matches:
                    confidences.append(1.0)
                else:
                    # Reduce confidence based on relative error
                    conf = max(0.0, 1.0 - min(comp.relative_error, 1.0))
                    confidences.append(conf)
            confidence_score = sum(confidences) / len(confidences)
        
        report = VerificationReport(
            timestamp=timestamp,
            original_statement=llm_statement,
            extracted_expressions=extracted_expressions,
            parse_results=parse_results,
            execution_results=execution_results,
            comparison_results=comparison_results,
            overall_verified=overall_verified,
            confidence_score=confidence_score,
            hallucination_detected=hallucination_detected,
            error_summary="; ".join(errors) if errors else ""
        )
        
        return report


# Global pipeline instance
_pipeline = None


def get_verifier() -> VerificationPipeline:
    """Get or create the global verification pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = VerificationPipeline()
    return _pipeline


def verify_llm_statement(llm_statement: str, expression: str, 
                        llm_value: float, variables: Optional[Dict[str, Any]] = None,
                        tolerance: float = 1e-6) -> Dict:
    """
    Quick verification of an LLM statement.
    
    Args:
        llm_statement: Original LLM reasoning/output
        expression: Mathematical expression to compute (e.g., "sharpe(returns, rf=0.03)")
        llm_value: Value claimed by LLM
        variables: Dict of variable values (e.g., {"returns": [0.01, 0.02, ...]})
        tolerance: Absolute error tolerance
    
    Returns:
        Dict with verification result
    """
    verifier = get_verifier()
    verifier.tolerance = tolerance
    report = verifier.verify(llm_statement, expression, llm_value, variables)
    return report.to_dict()
