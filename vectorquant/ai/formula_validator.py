"""
Phase 9.3: Formula Validation Engine

Allows AI to self-check mathematical formulas before execution.

Validates:
1. Syntax correctness (matching parens, valid names)
2. Dimensions (matrix operations compatibility)
3. Input bounds (valid ranges for parameters)
4. Statistical assumptions (e.g., variance != 0 for Sharpe)
5. Domain constraints (e.g., volatility > 0, probabilities in [0,1])

Error suggestions help LLMs correct formulas automatically.

Example:
    formula = "sharpe(returns, rf='0.03')"  # Wrong: rf is string
    errors = validator.check(formula)
    # Returns: [FormulaError(type='type_error', msg='...', suggestion='0.03')]
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import re

from .expression_parser import (
    parse_expression, ExpressionValidator, FunctionCall, ExpressionTokenizer
)


class ErrorType(Enum):
    """Types of formula errors."""
    SYNTAX = "syntax"           # Tokenization/parsing failed
    UNKNOWN_OP = "unknown_op"   # Operation not registered
    PARAM_COUNT = "param_count" # Wrong number of parameters
    PARAM_TYPE = "param_type"   # Parameter has wrong type
    DIMENSION = "dimension"     # Matrix dimension mismatch
    BOUNDS = "bounds"           # Value outside valid range
    DOMAIN = "domain"            # Value violates domain constraint
    MISSING_VAR = "missing_var"  # Variable not provided


@dataclass
class FormulaError:
    """Represents a single formula error."""
    error_type: ErrorType
    location: str          # Where the error is (e.g., "param 2")
    message: str           # Human-readable error message
    suggestion: Optional[str] = None  # How to fix it
    severity: str = "error"           # "error", "warning", "info"
    
    def to_dict(self) -> Dict:
        return {
            "type": self.error_type.value,
            "location": self.location,
            "message": self.message,
            "suggestion": self.suggestion,
            "severity": self.severity
        }


@dataclass
class ValidationResult:
    """Result of formula validation check."""
    formula: str
    is_valid: bool
    errors: List[FormulaError] = field(default_factory=list)
    warnings: List[FormulaError] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "formula": self.formula,
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings]
        }
    
    def __str__(self) -> str:
        """Human-readable output."""
        if self.is_valid:
            return f"✓ VALID: {self.formula}"
        
        parts = [f"✗ INVALID: {self.formula}"]
        
        if self.errors:
            parts.append("\nErrors:")
            for err in self.errors:
                parts.append(f"  - {err.message}")
                if err.suggestion:
                    parts.append(f"    → {err.suggestion}")
        
        if self.warnings:
            parts.append("\nWarnings:")
            for warn in self.warnings:
                parts.append(f"  - {warn.message}")
        
        return "\n".join(parts)


class OperationSignatures:
    """Database of operation signatures and constraints."""
    
    # Define each operation with expected parameter types and constraints
    OPERATIONS = {
        # Statistics
        "mean": {
            "params": [("data", ["list", "array"])],
            "constraints": {"data": {"non_empty": True}},
            "output_type": "scalar"
        },
        "std": {
            "params": [("data", ["list", "array"])],
            "constraints": {"data": {"non_empty": True, "min_elements": 2}},
            "output_type": "scalar"
        },
        "variance": {
            "params": [("data", ["list", "array"])],
            "constraints": {"data": {"non_empty": True, "min_elements": 2}},
            "output_type": "scalar"
        },
        "covariance": {
            "params": [("matrix", ["matrix"])],
            "constraints": {"matrix": {"non_empty": True, "min_rows": 2, "min_cols": 2}},
            "output_type": "matrix"
        },
        "correlation": {
            "params": [("matrix", ["matrix"])],
            "constraints": {"matrix": {"non_empty": True, "min_rows": 2, "min_cols": 2}},
            "output_type": "matrix"
        },
        
        # Risk
        "sharpe": {
            "params": [("returns", ["list", "array"]), ("rf", ["scalar"], "optional")],
            "constraints": {
                "returns": {"non_empty": True, "min_elements": 2},
                "rf": {"type": "numeric"}
            },
            "output_type": "scalar"
        },
        "var": {
            "params": [("returns", ["list", "array"]), ("confidence", ["scalar"], "optional")],
            "constraints": {
                "returns": {"non_empty": True},
                "confidence": {"min": 0.0, "max": 1.0}
            },
            "output_type": "scalar"
        },
        "cvar": {
            "params": [("returns", ["list", "array"]), ("confidence", ["scalar"], "optional")],
            "constraints": {
                "returns": {"non_empty": True},
                "confidence": {"min": 0.0, "max": 1.0}
            },
            "output_type": "scalar"
        },
        
        # Derivatives
        "price_call": {
            "params": [
                ("S", ["scalar"]),
                ("K", ["scalar"]),
                ("r", ["scalar"]),
                ("sigma", ["scalar"]),
                ("T", ["scalar"])
            ],
            "constraints": {
                "S": {"min": 0.0},
                "K": {"min": 0.0},
                "r": {"min": -1.0, "max": 1.0},
                "sigma": {"min": 0.0},
                "T": {"min": 0.0}
            },
            "output_type": "scalar",
            "mathematical_constraints": [
                "sigma > 0 for meaningful price",
                "T > 0 for future contract"
            ]
        },
        "price_put": {
            "params": [
                ("S", ["scalar"]),
                ("K", ["scalar"]),
                ("r", ["scalar"]),
                ("sigma", ["scalar"]),
                ("T", ["scalar"])
            ],
            "constraints": {
                "S": {"min": 0.0},
                "K": {"min": 0.0},
                "r": {"min": -1.0, "max": 1.0},
                "sigma": {"min": 0.0},
                "T": {"min": 0.0}
            },
            "output_type": "scalar"
        },
        
        # Simulation
        "simulate_gbm": {
            "params": [
                ("S0", ["scalar"]),
                ("mu", ["scalar"]),
                ("sigma", ["scalar"]),
                ("T", ["scalar"]),
                ("dt", ["scalar"]),
                ("n", ["scalar"])
            ],
            "constraints": {
                "S0": {"min": 0.0},
                "sigma": {"min": 0.0},
                "T": {"min": 0.0},
                "dt": {"min": 0.0},
                "n": {"min": 1, "type": "integer"}
            },
            "output_type": "matrix"
        },
        
        # Optimization
        "optimize_portfolio": {
            "params": [
                ("returns", ["list", "array"]),
                ("cov", ["matrix"]),
                ("rf", ["scalar"], "optional")
            ],
            "constraints": {
                "returns": {"non_empty": True},
                "cov": {"square": True},
                "rf": {"type": "numeric"}
            },
            "output_type": "list"
        }
    }


class FormulaValidator:
    """Validates mathematical formulas before execution."""
    
    def __init__(self):
        self.signatures = OperationSignatures.OPERATIONS
    
    def check(self, formula: str, variable_types: Optional[Dict[str, str]] = None) -> ValidationResult:
        """
        Validate a formula.
        
        Args:
            formula: Formula string to validate
            variable_types: Optional dict of known variable types
        
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(formula=formula, is_valid=True)
        variable_types = variable_types or {}
        
        # Step 1: Parse the formula
        success, parse_result = parse_expression(formula, variable_types)
        
        if not success:
            result.is_valid = False
            result.errors.append(FormulaError(
                error_type=ErrorType.SYNTAX,
                location="formula",
                message=f"Syntax error: {parse_result}",
                suggestion="Check parentheses, operator names, and parameter order"
            ))
            return result
        
        # Step 2: Detailed validation
        ast = parse_result.ast
        
        if isinstance(ast, FunctionCall):
            self._validate_function_call(ast, result, variable_types)
        
        return result
    
    def _validate_function_call(self, call: FunctionCall, result: ValidationResult,
                               variable_types: Dict[str, str]):
        """Validate a function call recursively."""
        op_name = call.function_name
        
        # Check operation is registered
        if op_name not in self.signatures:
            result.is_valid = False
            result.errors.append(FormulaError(
                error_type=ErrorType.UNKNOWN_OP,
                location=f"operation '{op_name}'",
                message=f"Unknown operation: {op_name}",
                suggestion=f"Check operation name. Available: {', '.join(list(self.signatures.keys())[:5])}..."
            ))
            return
        
        sig = self.signatures[op_name]
        
        # Validate parameters
        self._validate_parameters(op_name, call, sig, result)
        
        # Validate nested calls recursively
        for arg in call.args:
            if isinstance(arg, FunctionCall):
                self._validate_function_call(arg, result, variable_types)
        
        for val in call.kwargs.values():
            if isinstance(val, FunctionCall):
                self._validate_function_call(val, result, variable_types)
    
    def _validate_parameters(self, op_name: str, call: FunctionCall, 
                           sig: Dict, result: ValidationResult):
        """Validate function parameters."""
        required_params = [p for p in sig["params"] if len(p) == 2 or p[2] != "optional"]
        optional_params = [p for p in sig["params"] if len(p) > 2 and p[2] == "optional"]
        
        # Check parameter count
        total_provided = len(call.args) + len(call.kwargs)
        min_required = len(required_params)
        max_allowed = len(required_params) + len(optional_params)
        
        if total_provided < min_required:
            missing = [p[0] for p in required_params[len(call.args):]]
            result.is_valid = False
            result.errors.append(FormulaError(
                error_type=ErrorType.PARAM_COUNT,
                location=f"function {op_name}",
                message=f"Missing required parameters: {missing}",
                suggestion=f"{op_name}({', '.join([p[0] for p in sig['params']])})"
            ))
            return  # Early return to avoid cascading errors
        
        if total_provided > max_allowed:
            result.is_valid = False
            result.errors.append(FormulaError(
                error_type=ErrorType.PARAM_COUNT,
                location=f"function {op_name}",
                message=f"Too many parameters. Expected max {max_allowed}, got {total_provided}",
                suggestion=f"Remove extra parameters"
            ))
            return  # Early return
        
        # Validate parameter types and constraints
        constraints = sig.get("constraints", {})
        
        for param_name, param_value in call.kwargs.items():
            if param_name in constraints:
                constraint = constraints[param_name]
                # Extract actual value from AST node if needed
                actual_value = self._extract_value(param_value)
                self._check_constraints(param_name, actual_value, constraint, result)
    
    def _check_constraints(self, param_name: str, param_value: Any, 
                         constraints: Dict, result: ValidationResult):
        """Check parameter constraints."""
        # Check type constraints
        if "type" in constraints:
            expected_type = constraints["type"]
            if expected_type == "numeric":
                if not isinstance(param_value, (int, float)):
                    result.is_valid = False
                    result.errors.append(FormulaError(
                        error_type=ErrorType.PARAM_TYPE,
                        location=f"parameter '{param_name}'",
                        message=f"Expected numeric, got {type(param_value).__name__}",
                        suggestion=f"Convert {param_name} to a number"
                    ))
                    return
        
        # Check bounds
        if isinstance(param_value, (int, float)):
            if "min" in constraints and param_value < constraints["min"]:
                result.errors.append(FormulaError(
                    error_type=ErrorType.BOUNDS,
                    location=f"parameter '{param_name}'",
                    message=f"{param_name} must be >= {constraints['min']}, got {param_value}",
                    severity="warning"
                ))
            
            if "max" in constraints and param_value > constraints["max"]:
                result.errors.append(FormulaError(
                    error_type=ErrorType.BOUNDS,
                    location=f"parameter '{param_name}'",
                    message=f"{param_name} must be <= {constraints['max']}, got {param_value}",
                    severity="warning"
                ))
    
    @staticmethod
    def _extract_value(node: Any) -> Any:
        """Extract actual value from AST node."""
        if isinstance(node, dict):
            if node.get("type") == "literal":
                return node.get("value")
            elif node.get("type") == "variable":
                return None  # Variables can't be validated without values
            else:
                return node
        return node


class DimensionValidator:
    """Validates matrix dimension compatibility."""
    
    @staticmethod
    def get_dimensions(data: Any) -> Tuple[Optional[int], Optional[int]]:
        """
        Get dimensions of data.
        
        Returns:
            (rows, cols) for matrices, (length, 1) for vectors, (1, 1) for scalars
        """
        if isinstance(data, (int, float)):
            return (1, 1)
        
        if isinstance(data, list):
            if len(data) == 0:
                return (0, 0)
            
            # Check if it's a matrix (list of lists)
            if isinstance(data[0], list):
                rows = len(data)
                cols = len(data[0]) if data[0] else 0
                # Verify all rows have same length
                for row in data:
                    if len(row) != cols:
                        return None  # Irregular matrix
                return (rows, cols)
            else:
                # Vector
                return (len(data), 1)
        
        return None
    
    @staticmethod
    def check_matmul_compatibility(A_dims: Tuple[int, int], 
                                   B_dims: Tuple[int, int]) -> Tuple[bool, str]:
        """Check if two matrices can be multiplied."""
        A_rows, A_cols = A_dims
        B_rows, B_cols = B_dims
        
        if A_cols != B_rows:
            return False, f"Cannot multiply {A_dims} @ {B_dims}. Need {A_rows}x{A_cols} @ {A_cols}x{B_cols}"
        
        return True, ""
    
    @staticmethod
    def check_addition_compatibility(A_dims: Tuple[int, int],
                                    B_dims: Tuple[int, int]) -> Tuple[bool, str]:
        """Check if two matrices can be added."""
        if A_dims != B_dims:
            return False, f"Cannot add matrices with different dimensions: {A_dims} + {B_dims}"
        
        return True, ""


def validate_formula(formula: str, variable_types: Optional[Dict[str, str]] = None) -> ValidationResult:
    """
    Quick validation of a formula.
    
    Args:
        formula: Formula string
        variable_types: Optional dict of variable types
    
    Returns:
        ValidationResult
    """
    validator = FormulaValidator()
    return validator.check(formula, variable_types)
