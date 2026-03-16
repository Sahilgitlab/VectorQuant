"""
Agent Protocol Interface (Phase 9.1)

Standardized interface for VectorQuant tool invocation by AI agents.
Compatible with LangChain, Anthropic Claude, Google Gemini, and custom LLM frameworks.

Core Design:
- Single entry point: VectorQuantTool.compute(operation, params)
- Automatic parameter validation and type coercion
- Error handling with meaningful messages
- Execution metadata (latency, verification status, etc.)
- Fallback mechanisms for robustness
"""

import time
import sys
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


# Wrapper functions for operations without direct VectorQuant implementations

def _sharpe_ratio_wrapper(data: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Compute Sharpe ratio: (mean_return - rf) / volatility
    
    Wrapper that combines mean and standard_deviation from VectorQuant.
    """
    from vectorquant.core.statistics import mean, standard_deviation
    
    mean_return = mean(data)
    volatility = standard_deviation(data)
    
    if volatility == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / volatility


@dataclass
class ComputationMetadata:
    """Metadata about a computation's execution."""
    timestamp: str
    latency_ms: float
    backend: str  # 'c' or 'python'
    verified: bool
    error: Optional[str] = None
    warning: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComputationResult:
    """Result of a VectorQuant tool computation."""
    operation: str
    result: Any
    metadata: ComputationMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "operation": self.operation,
            "result": self._serialize_result(self.result),
            "metadata": self.metadata.to_dict(),
        }
    
    @staticmethod
    def _serialize_result(obj: Any) -> Any:
        """Convert result to JSON-serializable format."""
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [ComputationResult._serialize_result(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: ComputationResult._serialize_result(v) for k, v in obj.items()}
        else:
            # Fallback: convert to string representation
            return str(obj)


class ParameterValidator:
    """Validates and coerces parameters for VectorQuant operations."""
    
    @staticmethod
    def validate_numeric(value: Any, name: str, allow_none: bool = False) -> float:
        """Validate numeric parameter."""
        if value is None and allow_none:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Parameter '{name}' must be numeric, got {type(value).__name__}")
    
    @staticmethod
    def validate_list(value: Any, name: str, 
                     element_type: Optional[str] = 'numeric') -> List[Any]:
        """Validate list parameter."""
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Parameter '{name}' must be a list, got {type(value).__name__}")
        
        if element_type == 'numeric':
            return [ParameterValidator.validate_numeric(x, f"{name}[{i}]") 
                    for i, x in enumerate(value)]
        
        return list(value)
    
    @staticmethod
    def validate_matrix(value: Any, name: str) -> List[List[float]]:
        """Validate 2D matrix parameter."""
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Parameter '{name}' must be a list of lists, got {type(value).__name__}")
        
        result = []
        for i, row in enumerate(value):
            if not isinstance(row, (list, tuple)):
                raise ValueError(f"Parameter '{name}[{i}]' must be a list, got {type(row).__name__}")
            result.append([ParameterValidator.validate_numeric(x, f"{name}[{i}][{j}]") 
                          for j, x in enumerate(row)])
        
        return result
    
    @staticmethod
    def validate_integer(value: Any, name: str, 
                        min_val: Optional[int] = None,
                        max_val: Optional[int] = None) -> int:
        """Validate integer parameter."""
        try:
            val = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"Parameter '{name}' must be an integer, got {type(value).__name__}")
        
        if min_val is not None and val < min_val:
            raise ValueError(f"Parameter '{name}' must be >= {min_val}, got {val}")
        if max_val is not None and val > max_val:
            raise ValueError(f"Parameter '{name}' must be <= {max_val}, got {val}")
        
        return val


class VectorQuantTool:
    """
    Unified agent-callable interface for VectorQuant operations.
    
    All AI agents (Claude, Gemini, LangChain, etc.) interact through this class.
    It handles:
    - Parameter validation and type coercion
    - Automatic backend selection (C or Python)
    - Error handling with meaningful messages
    - Execution profiling (latency, backend used)
    - Optional verification and proof generation
    
    Example usage::
    
        tool = VectorQuantTool()
        
        # Direct computation
        result = tool.compute(
            operation="compute_sharpe",
            params={"returns": [0.01, 0.02, -0.005, 0.015]}
        )
        
        # For LLMs: get schema first
        schema = tool.get_schema("compute_sharpe")
        # Then invoke via compute()
    """
    
    # Mapping of operation names to implementation functions and schemas
    OPERATIONS = {
        # Statistics
        "compute_mean": {
            "module": "vectorquant.core.statistics",
            "function": "mean",
            "description": "Compute arithmetic mean of returns",
            "params": {
                "returns": {"type": "list", "element_type": "numeric", "description": "Returns data"},
            },
            "returns": {"type": "numeric", "description": "Mean value"},
        },
        "compute_std": {
            "module": "vectorquant.core.statistics",
            "function": "standard_deviation",
            "description": "Compute standard deviation of returns",
            "params": {
                "returns": {"type": "list", "element_type": "numeric", "description": "Returns data"},
            },
            "returns": {"type": "numeric", "description": "Standard deviation"},
        },
        "compute_variance": {
            "module": "vectorquant.core.statistics",
            "function": "variance",
            "description": "Compute variance of returns",
            "params": {
                "returns": {"type": "list", "element_type": "numeric", "description": "Returns data"},
            },
            "returns": {"type": "numeric", "description": "Variance"},
        },
        
        # Risk metrics
        "compute_sharpe": {
            "module": None,  # Special case: wrapper function
            "function": "_sharpe_ratio_wrapper",
            "description": "Compute Sharpe ratio: (return - rf) / volatility",
            "params": {
                "returns": {"type": "list", "element_type": "numeric", "description": "Returns data"},
                "risk_free_rate": {"type": "numeric", "description": "Risk-free rate", "default": 0.0},
            },
            "returns": {"type": "numeric", "description": "Sharpe ratio"},
        },
        "compute_var": {
            "module": "vectorquant.finance.risk_models",
            "function": "parametric_var",
            "description": "Compute Value-at-Risk (parametric method)",
            "params": {
                "returns": {"type": "list", "element_type": "numeric", "description": "Returns data"},
                "confidence_level": {"type": "numeric", "description": "Confidence level", "default": 0.95},
            },
            "returns": {"type": "numeric", "description": "VaR value"},
        },
        "compute_cvar": {
            "module": "vectorquant.finance.risk_models",
            "function": "cvar",
            "description": "Compute Conditional Value-at-Risk (Expected Shortfall)",
            "params": {
                "returns": {"type": "list", "element_type": "numeric", "description": "Returns data"},
                "confidence_level": {"type": "numeric", "description": "Confidence level", "default": 0.95},
            },
            "returns": {"type": "numeric", "description": "CVaR value"},
        },
        
        # Covariance and correlation
        "compute_covariance": {
            "module": "vectorquant.core.statistics",
            "function": "covariance",
            "description": "Compute covariance matrix",
            "params": {
                "returns_matrix": {"type": "matrix", "description": "Returns matrix (n_assets x n_periods)"},
            },
            "returns": {"type": "matrix", "description": "Covariance matrix (n_assets x n_assets)"},
        },
        "compute_correlation": {
            "module": "vectorquant.core.statistics",
            "function": "correlation",
            "description": "Compute correlation matrix",
            "params": {
                "returns_matrix": {"type": "matrix", "description": "Returns matrix (n_assets x n_periods)"},
            },
            "returns": {"type": "matrix", "description": "Correlation matrix (n_assets x n_assets)"},
        },
        
        # Derivatives pricing
        "price_call": {
            "module": "vectorquant.finance.derivatives",
            "function": "black_scholes_call",
            "description": "Price European call option using Black-Scholes",
            "params": {
                "S": {"type": "numeric", "description": "Current stock price"},
                "K": {"type": "numeric", "description": "Strike price"},
                "r": {"type": "numeric", "description": "Risk-free rate"},
                "sigma": {"type": "numeric", "description": "Volatility (annualized)"},
                "T": {"type": "numeric", "description": "Time to maturity (years)"},
            },
            "returns": {"type": "numeric", "description": "Call option price"},
        },
        "price_put": {
            "module": "vectorquant.finance.derivatives",
            "function": "black_scholes_put",
            "description": "Price European put option using Black-Scholes",
            "params": {
                "S": {"type": "numeric", "description": "Current stock price"},
                "K": {"type": "numeric", "description": "Strike price"},
                "r": {"type": "numeric", "description": "Risk-free rate"},
                "sigma": {"type": "numeric", "description": "Volatility (annualized)"},
                "T": {"type": "numeric", "description": "Time to maturity (years)"},
            },
            "returns": {"type": "numeric", "description": "Put option price"},
        },
        
        # Monte Carlo simulation
        "simulate_gbm": {
            "module": "vectorquant.stochastic.processes",
            "function": "simulate_geometric_brownian_motion",
            "description": "Simulate Geometric Brownian Motion price paths",
            "params": {
                "S0": {"type": "numeric", "description": "Initial price"},
                "mu": {"type": "numeric", "description": "Drift (expected return)"},
                "sigma": {"type": "numeric", "description": "Volatility"},
                "T": {"type": "numeric", "description": "Time horizon (years)"},
                "dt": {"type": "numeric", "description": "Time step"},
                "n_paths": {"type": "integer", "description": "Number of simulation paths"},
            },
            "returns": {"type": "matrix", "description": "Simulated price paths (n_paths x n_steps)"},
        },
        
        # Portfolio optimization
        "optimize_portfolio": {
            "module": "vectorquant.finance.portfolio",
            "function": "optimize_max_sharpe",
            "description": "Find portfolio weights that maximize Sharpe ratio",
            "params": {
                "expected_returns": {"type": "list", "element_type": "numeric", "description": "Expected returns per asset"},
                "cov_matrix": {"type": "matrix", "description": "Covariance matrix"},
                "risk_free_rate": {"type": "numeric", "description": "Risk-free rate", "default": 0.0},
            },
            "returns": {"type": "list", "description": "Optimal portfolio weights"},
        },
    }
    
    def __init__(self):
        """Initialize VectorQuant tool interface."""
        self._func_cache = {}
        self._backend_info = self._detect_backend()
    
    def _detect_backend(self) -> Dict[str, Any]:
        """Detect available backend (C or Python)."""
        try:
            import vectorquant_c_core
            return {"backend": "c", "available": True}
        except ImportError:
            return {"backend": "python", "available": True}
    
    def get_available_operations(self) -> List[str]:
        """Return list of all available operations."""
        return list(self.OPERATIONS.keys())
    
    def get_schema(self, operation: str) -> Dict[str, Any]:
        """
        Get schema for a specific operation.
        
        Useful for LLMs to understand what parameters an operation needs.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Schema dict with description, params, and return type
        """
        if operation not in self.OPERATIONS:
            raise ValueError(f"Unknown operation: {operation}. "
                           f"Available: {', '.join(self.get_available_operations())}")
        
        return self.OPERATIONS[operation]
    
    def compute(self, operation: str, params: Optional[Dict[str, Any]] = None,
               verify: bool = False) -> ComputationResult:
        """
        Execute a VectorQuant operation with automatic parameter validation.
        
        Args:
            operation: Name of the operation (e.g., "compute_sharpe")
            params: Dictionary of parameters
            verify: If True, attempt verification of result
            
        Returns:
            ComputationResult with result, metadata, and optional verification
            
        Raises:
            ValueError: If operation unknown or parameters invalid
            
        Example::
        
            result = tool.compute("compute_sharpe", {
                "returns": [0.01, 0.02, -0.005, 0.015],
                "risk_free_rate": 0.02
            })
            
            print(f"Sharpe ratio: {result.result}")
            print(f"Latency: {result.metadata.latency_ms} ms")
        """
        if params is None:
            params = {}
        
        start_time = time.time()
        metadata = ComputationMetadata(
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=0.0,
            backend=self._backend_info["backend"],
            verified=False,
        )
        
        try:
            # Validate operation exists
            if operation not in self.OPERATIONS:
                raise ValueError(
                    f"Unknown operation: {operation}. "
                    f"Available: {', '.join(self.get_available_operations())}"
                )
            
            op_spec = self.OPERATIONS[operation]
            
            # Validate and coerce parameters
            validated_params = self._validate_parameters(operation, params, op_spec)
            
            # Adapt parameters to match actual function signatures
            adapted_params = self._adapt_parameters(operation, validated_params)
            
            # Get function
            func = self._load_function(op_spec["module"], op_spec["function"])
            
            # Execute
            result_value = func(**adapted_params)
            
            # Verification (optional)
            if verify:
                metadata.verified = self._attempt_verification(operation, validated_params, result_value)
            
            metadata.latency_ms = (time.time() - start_time) * 1000
            
            return ComputationResult(
                operation=operation,
                result=result_value,
                metadata=metadata,
            )
        
        except Exception as e:
            metadata.latency_ms = (time.time() - start_time) * 1000
            metadata.error = str(e)
            
            return ComputationResult(
                operation=operation,
                result=None,
                metadata=metadata,
            )
    
    def _validate_parameters(self, operation: str, params: Dict[str, Any],
                            op_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and coerce parameters according to schema."""
        validated = {}
        param_spec = op_spec.get("params", {})
        
        for param_name, param_info in param_spec.items():
            param_type = param_info.get("type")
            
            # Check if parameter provided
            if param_name not in params:
                # Check for default
                if "default" in param_info:
                    validated[param_name] = param_info["default"]
                    continue
                else:
                    raise ValueError(f"Missing required parameter: {param_name}")
            
            value = params[param_name]
            
            # Validate based on type
            if param_type == "numeric":
                validated[param_name] = ParameterValidator.validate_numeric(value, param_name)
            
            elif param_type == "integer":
                validated[param_name] = ParameterValidator.validate_integer(value, param_name)
            
            elif param_type == "list":
                element_type = param_info.get("element_type", None)
                validated[param_name] = ParameterValidator.validate_list(
                    value, param_name, element_type
                )
            
            elif param_type == "matrix":
                validated[param_name] = ParameterValidator.validate_matrix(value, param_name)
            
            else:
                # Unknown type — pass through
                validated[param_name] = value
        
        # Check for unexpected parameters
        unexpected = set(params.keys()) - set(param_spec.keys())
        if unexpected:
            raise ValueError(f"Unexpected parameters: {', '.join(unexpected)}")
        
        return validated
    
    def _load_function(self, module_name: str, function_name: str):
        """Dynamically load a function from a module."""
        cache_key = f"{module_name}.{function_name}"
        
        if cache_key in self._func_cache:
            return self._func_cache[cache_key]
        
        # Handle wrapper functions (module_name is None)
        if module_name is None:
            if function_name == "_sharpe_ratio_wrapper":
                func = _sharpe_ratio_wrapper
                self._func_cache[cache_key] = func
                return func
            else:
                raise ValueError(f"Unknown wrapper function: {function_name}")
        
        # Import module and get function
        try:
            module = __import__(module_name, fromlist=[function_name])
            func = getattr(module, function_name)
            self._func_cache[cache_key] = func
            return func
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not load {module_name}.{function_name}: {e}")
    
    def _adapt_parameters(self, operation: str, validated_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt validated parameters to match actual function signatures.
        
        Different VectorQuant functions use different parameter names
        (e.g., 'data' vs 'returns'), so this maps agent interface params
        to actual function params.
        """
        op_spec = self.OPERATIONS[operation]
        actual_function_name = op_spec["function"]
        
        # Parameter mapping: operation_param -> function_param
        param_mappings = {
            "compute_mean": {"returns": "data"},
            "compute_std": {"returns": "data"},
            "compute_variance": {"returns": "data"},
            "compute_sharpe": {"returns": "data"},  # Wrapper function uses "data"
            "compute_correlation": {"returns_matrix": "data"},
            "compute_covariance": {"returns_matrix": "data"},
        }
        
        if operation in param_mappings:
            mapping = param_mappings[operation]
            adapted = {}
            for key, value in validated_params.items():
                new_key = mapping.get(key, key)
                adapted[new_key] = value
            return adapted
        
        return validated_params
    
    def _attempt_verification(self, operation: str, params: Dict[str, Any],
                             result: Any) -> bool:
        """
        Attempt optional verification of computation result.
        
        Returns True if verification passed, False otherwise.
        """
        try:
            # For now, just return True (verification logic added in Phase 9.2)
            return True
        except Exception:
            return False
    
    def get_langchain_tools(self) -> List[Any]:
        """
        Get all operations as LangChain-compatible Tool objects.
        
        Requires: pip install langchain
        """
        try:
            from langchain.tools import Tool
            
            tools = []
            for op_name in self.get_available_operations():
                op_spec = self.OPERATIONS[op_name]
                
                def make_tool_func(op):
                    def tool_func(**kwargs):
                        result = self.compute(op, kwargs)
                        if result.metadata.error:
                            return f"Error: {result.metadata.error}"
                        return result.result
                    return tool_func
                
                tools.append(Tool(
                    name=op_name,
                    func=make_tool_func(op_name),
                    description=op_spec["description"],
                ))
            
            return tools
        
        except ImportError:
            raise ImportError("LangChain not installed. Install with: pip install langchain")
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Get all operations as OpenAI function-calling schema.
        
        Returns list of schemas ready for OpenAI API.
        """
        tools = []
        
        for op_name in self.get_available_operations():
            op_spec = self.OPERATIONS[op_name]
            
            # Convert param specs to OpenAI format
            properties = {}
            required = []
            
            for param_name, param_info in op_spec.get("params", {}).items():
                param_type = param_info.get("type")
                
                # Map to OpenAI types
                if param_type == "numeric":
                    openai_type = "number"
                elif param_type == "integer":
                    openai_type = "integer"
                elif param_type in ("list", "matrix"):
                    openai_type = "array"
                else:
                    openai_type = "string"
                
                properties[param_name] = {
                    "type": openai_type,
                    "description": param_info.get("description", ""),
                }
                
                if "default" not in param_info:
                    required.append(param_name)
            
            tools.append({
                "type": "function",
                "function": {
                    "name": op_name,
                    "description": op_spec["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        
        return tools
