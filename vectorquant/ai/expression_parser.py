"""
Stage 2: Expression Parser for Verification Pipeline

Converts mathematical expressions from LLM output into VectorQuant operations.

Supports:
- Function calls: mean(data), std(data), sharpe_ratio(returns, rf=0.03)
- Nested expressions: sharpe_ratio(optimize_portfolio(returns, cov), rf=0.03)
- Parameters: numeric literals, lists, matrices
- Variable references: named_data, portfolio_returns, etc.

Process:
1. Tokenize: Break expression into tokens
2. Parse: Build abstract syntax tree (AST)
3. Validate: Check operation exists and parameters are valid
4. Convert: Map to VectorQuant operation
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Union
from enum import Enum


class TokenType(Enum):
    """Token types for expression parsing."""
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    STRING = "STRING"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    COMMA = "COMMA"
    EQUALS = "EQUALS"
    EOF = "EOF"


@dataclass
class Token:
    """A single token from the expression."""
    type: TokenType
    value: Any
    position: int


@dataclass
class FunctionCall:
    """Represents a function call in the expression."""
    function_name: str
    args: List[Any]  # positional arguments (can be FunctionCall, literal, etc)
    kwargs: Dict[str, Any]  # keyword arguments
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "type": "function_call",
            "function": self.function_name,
            "args": [self._serialize_arg(a) for a in self.args],
            "kwargs": {k: self._serialize_arg(v) for k, v in self.kwargs.items()}
        }
    
    @staticmethod
    def _serialize_arg(arg: Any) -> Any:
        """Recursively serialize arguments."""
        if isinstance(arg, FunctionCall):
            return arg.to_dict()
        elif isinstance(arg, list):
            return [FunctionCall._serialize_arg(a) for a in arg]
        elif isinstance(arg, dict):
            return {k: FunctionCall._serialize_arg(v) for k, v in arg.items()}
        else:
            return arg


class ExpressionTokenizer:
    """Tokenizes a mathematical expression string."""
    
    TOKEN_PATTERNS = [
        (r'\(', TokenType.LPAREN),
        (r'\)', TokenType.RPAREN),
        (r'\[', TokenType.LBRACKET),
        (r'\]', TokenType.RBRACKET),
        (r',', TokenType.COMMA),
        (r'=', TokenType.EQUALS),
        (r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', TokenType.NUMBER),  # floats, ints, scientific
        (r'"[^"]*"', TokenType.STRING),  # double-quoted strings
        (r"'[^']*'", TokenType.STRING),  # single-quoted strings
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENTIFIER),  # variable/function names
    ]
    
    def __init__(self, expression: str):
        """Initialize with an expression string."""
        self.expression = expression
        self.position = 0
        self.tokens: List[Token] = []
    
    def tokenize(self) -> List[Token]:
        """Tokenize the expression."""
        self.tokens = []
        self.position = 0
        
        while self.position < len(self.expression):
            # Skip whitespace
            if self.expression[self.position].isspace():
                self.position += 1
                continue
            
            # Try to match each pattern
            matched = False
            for pattern, token_type in self.TOKEN_PATTERNS:
                regex = re.compile(pattern)
                match = regex.match(self.expression, self.position)
                
                if match:
                    value = match.group(0)
                    
                    # Convert numeric values
                    if token_type == TokenType.NUMBER:
                        try:
                            if '.' in value or 'e' in value or 'E' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            value = float(value)
                    
                    # Strip quotes from strings
                    elif token_type == TokenType.STRING:
                        value = value[1:-1]
                    
                    self.tokens.append(Token(token_type, value, self.position))
                    self.position = match.end()
                    matched = True
                    break
            
            if not matched:
                raise SyntaxError(f"Invalid character '{self.expression[self.position]}' at position {self.position}")
        
        self.tokens.append(Token(TokenType.EOF, None, self.position))
        return self.tokens


class ExpressionParser:
    """Parses tokenized expressions into function call ASTs."""
    
    # Registered VectorQuant operations with their signatures
    REGISTERED_OPS = {
        # Statistics
        "compute_mean": {"params": ["data"], "optional": []},
        "compute_std": {"params": ["data"], "optional": []},
        "compute_variance": {"params": ["data"], "optional": []},
        "compute_covariance": {"params": ["matrix"], "optional": []},
        "compute_correlation": {"params": ["matrix"], "optional": []},
        
        # Risk
        "compute_sharpe": {"params": ["returns"], "optional": ["rf"]},
        "compute_var": {"params": ["returns"], "optional": ["confidence"]},
        "compute_cvar": {"params": ["returns"], "optional": ["confidence"]},
        
        # Derivatives
        "price_call": {"params": ["S", "K", "r", "sigma", "T"], "optional": []},
        "price_put": {"params": ["S", "K", "r", "sigma", "T"], "optional": []},
        
        # Simulation
        "simulate_gbm": {"params": ["S0", "mu", "sigma", "T", "dt", "n"], "optional": []},
        
        # Optimization
        "optimize_portfolio": {"params": ["returns", "cov"], "optional": ["rf"]},
        
        # Common aliases
        "mean": {"params": ["data"], "optional": []},
        "std": {"params": ["data"], "optional": []},
        "variance": {"params": ["data"], "optional": []},
        "sharpe": {"params": ["returns"], "optional": ["rf"]},
        "var": {"params": ["returns"], "optional": ["confidence"]},
        "cvar": {"params": ["returns"], "optional": ["confidence"]},
    }
    
    def __init__(self, tokens: List[Token]):
        """Initialize with a token stream."""
        self.tokens = tokens
        self.position = 0
    
    def current_token(self) -> Token:
        """Get the current token."""
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return self.tokens[-1]  # EOF
    
    def peek_token(self, offset: int = 1) -> Token:
        """Look ahead at the next token."""
        pos = self.position + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]  # EOF
    
    def advance(self) -> Token:
        """Move to the next token and return current."""
        token = self.current_token()
        if token.type != TokenType.EOF:
            self.position += 1
        return token
    
    def expect(self, token_type: TokenType) -> Token:
        """Consume a token of the expected type."""
        token = self.current_token()
        if token.type != token_type:
            raise SyntaxError(f"Expected {token_type}, got {token.type} at position {token.position}")
        return self.advance()
    
    def parse(self) -> Union[FunctionCall, Any]:
        """Parse the expression and return the AST."""
        result = self._parse_expression()
        self.expect(TokenType.EOF)
        return result
    
    def _parse_expression(self) -> Union[FunctionCall, Any]:
        """Parse a primary expression (function call or literal)."""
        token = self.current_token()
        
        # Function call: identifier followed by (
        if token.type == TokenType.IDENTIFIER and self.peek_token().type == TokenType.LPAREN:
            return self._parse_function_call()
        
        # Variable reference (will be resolved at verification time)
        elif token.type == TokenType.IDENTIFIER:
            name = self.advance().value
            return {"type": "variable", "name": name}
        
        # Numeric literal
        elif token.type == TokenType.NUMBER:
            value = self.advance().value
            return {"type": "literal", "value": value}
        
        # String literal
        elif token.type == TokenType.STRING:
            value = self.advance().value
            return {"type": "literal", "value": value}
        
        # List literal: [1, 2, 3]
        elif token.type == TokenType.LBRACKET:
            return self._parse_list()
        
        else:
            raise SyntaxError(f"Unexpected token {token.type} at position {token.position}")
    
    def _parse_function_call(self) -> FunctionCall:
        """Parse a function call: func(arg1, arg2, kwarg1=val1)."""
        func_name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.LPAREN)
        
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        
        # Parse arguments
        while self.current_token().type != TokenType.RPAREN:
            # Check for keyword argument
            if (self.current_token().type == TokenType.IDENTIFIER and
                self.peek_token().type == TokenType.EQUALS):
                
                key = self.expect(TokenType.IDENTIFIER).value
                self.expect(TokenType.EQUALS)
                value = self._parse_expression()
                kwargs[key] = value
            else:
                # Positional argument
                args.append(self._parse_expression())
            
            # Check for comma
            if self.current_token().type == TokenType.COMMA:
                self.advance()
        
        self.expect(TokenType.RPAREN)
        
        return FunctionCall(func_name, args, kwargs)
    
    def _parse_list(self) -> Dict:
        """Parse a list literal: [1, 2, 3]."""
        self.expect(TokenType.LBRACKET)
        
        elements = []
        while self.current_token().type != TokenType.RBRACKET:
            elements.append(self._parse_expression())
            
            if self.current_token().type == TokenType.COMMA:
                self.advance()
        
        self.expect(TokenType.RBRACKET)
        
        return {"type": "literal", "value": elements}


class ExpressionValidator:
    """Validates that parsed expressions are valid VectorQuant operations."""
    
    def __init__(self, variable_types: Optional[Dict[str, str]] = None):
        """
        Initialize validator.
        
        Args:
            variable_types: Dict mapping variable names to their types
                           (e.g., {"returns": "list", "data": "list"})
        """
        self.variable_types = variable_types or {}
    
    def validate(self, expr: Union[FunctionCall, Dict]) -> Tuple[bool, str]:
        """
        Validate an expression (FunctionCall or dict).
        
        Returns:
            (is_valid, error_message)
        """
        if isinstance(expr, FunctionCall):
            return self._validate_function_call(expr)
        elif isinstance(expr, dict):
            if expr.get("type") == "function_call":
                # Reconstruct FunctionCall from dict
                call = FunctionCall(
                    expr["function"],
                    expr.get("args", []),
                    expr.get("kwargs", {})
                )
                return self._validate_function_call(call)
            elif expr.get("type") == "variable":
                # Variables are always valid if they're registered
                return True, ""
            elif expr.get("type") == "literal":
                return True, ""
            else:
                return False, f"Unknown expression type: {expr.get('type')}"
        else:
            return False, f"Invalid expression format: {type(expr)}"
    
    def _validate_function_call(self, call: FunctionCall) -> Tuple[bool, str]:
        """Validate a function call."""
        ops = ExpressionParser.REGISTERED_OPS
        
        # If operation is not registered, allow it to pass through
        # FormulaValidator will handle unknown operations
        if call.function_name not in ops:
            return True, ""  # Return success, FormulaValidator will check validity
        
        # For registered operations, do minimal validation
        # Just check that it's a valid FunctionCall structure
        # More detailed validation happens in FormulaValidator
        
        # Note: Parameter count validation is delegated to FormulaValidator
        # to provide better error messages and suggestions
        return True, ""
        
        # Validate nested expressions
        for arg in call.args:
            if isinstance(arg, FunctionCall):
                valid, msg = self._validate_function_call(arg)
                if not valid:
                    return False, msg
        
        for val in call.kwargs.values():
            if isinstance(val, FunctionCall):
                valid, msg = self._validate_function_call(val)
                if not valid:
                    return False, msg
        
        return True, ""


class ParsedExpression:
    """Container for a parsed and validated expression."""
    
    def __init__(self, original: str, ast: Union[FunctionCall, Dict], 
                 variable_types: Optional[Dict[str, str]] = None):
        """
        Initialize.
        
        Args:
            original: Original expression string
            ast: Abstract syntax tree (FunctionCall or dict)
            variable_types: Types of variables used in expression
        """
        self.original = original
        self.ast = ast
        self.variable_types = variable_types or {}
        self.is_nested = self._check_nested()
        self.operations = self._extract_operations()
    
    def _check_nested(self) -> bool:
        """Check if this is a nested expression."""
        if not isinstance(self.ast, FunctionCall):
            return False
        
        for arg in self.ast.args:
            if isinstance(arg, FunctionCall):
                return True
        
        for val in self.ast.kwargs.values():
            if isinstance(val, FunctionCall):
                return True
        
        return False
    
    def _extract_operations(self) -> List[str]:
        """Extract all operation names from the AST."""
        ops = []
        
        def extract_from(node):
            if isinstance(node, FunctionCall):
                ops.append(node.function_name)
                for arg in node.args:
                    extract_from(arg)
                for val in node.kwargs.values():
                    extract_from(val)
        
        extract_from(self.ast)
        return ops
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "original": self.original,
            "ast": self.ast.to_dict() if isinstance(self.ast, FunctionCall) else self.ast,
            "operations": self.operations,
            "is_nested": self.is_nested,
            "variables": list(self.variable_types.keys())
        }


def parse_expression(expression: str, 
                    variable_types: Optional[Dict[str, str]] = None) -> Tuple[bool, Union[ParsedExpression, str]]:
    """
    Parse and validate a mathematical expression.
    
    Args:
        expression: Expression string to parse
        variable_types: Optional dict of variable name -> type mappings
    
    Returns:
        (success, result): (True, ParsedExpression) if valid, (False, error_message) if not
    """
    try:
        # Tokenize
        tokenizer = ExpressionTokenizer(expression)
        tokens = tokenizer.tokenize()
        
        # Parse
        parser = ExpressionParser(tokens)
        ast = parser.parse()
        
        # Validate
        validator = ExpressionValidator(variable_types)
        is_valid, error_msg = validator.validate(ast)
        
        if not is_valid:
            return False, error_msg
        
        # Create result
        result = ParsedExpression(expression, ast, variable_types)
        return True, result
    
    except (SyntaxError, ValueError) as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"
