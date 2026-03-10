"""
LLM Tool Interface

Exposes VectorQuant functions as structured tool definitions
compatible with LangChain, LlamaIndex, and OpenAI function calling.
"""


def get_tool_registry():
    """
    Returns a dictionary of all callable VectorQuant tools
    with name, description, parameters, and function reference.
    """
    from vectorquant.finance.risk_models import parametric_var, cvar, historical_var
    from vectorquant.finance.derivatives import black_scholes_call, black_scholes_put
    from vectorquant.finance.portfolio import optimize_max_sharpe
    from vectorquant.stochastic.processes import simulate_geometric_brownian_motion
    from vectorquant.stochastic.monte_carlo import MonteCarloEngine
    from vectorquant.core.statistics import mean, standard_deviation
    from vectorquant.finance.factor_models import (
        fama_french_3_factor, fama_french_5_factor, estimate_factor_betas
    )

    registry = {
        "calculate_var": {
            "function": parametric_var,
            "description": "Calculate Parametric Value-at-Risk for a returns series.",
            "parameters": {
                "returns": "List of float returns",
                "confidence_level": "float, default 0.95"
            },
        },
        "calculate_cvar": {
            "function": cvar,
            "description": "Calculate Conditional Value-at-Risk (Expected Shortfall).",
            "parameters": {
                "returns": "List of float returns",
                "confidence_level": "float, default 0.95"
            },
        },
        "price_call_option": {
            "function": black_scholes_call,
            "description": "Price a European call option using Black-Scholes.",
            "parameters": {
                "S": "Current stock price",
                "K": "Strike price",
                "r": "Risk-free rate",
                "sigma": "Volatility",
                "T": "Time to maturity in years"
            },
        },
        "price_put_option": {
            "function": black_scholes_put,
            "description": "Price a European put option using Black-Scholes.",
            "parameters": {
                "S": "Current stock price",
                "K": "Strike price",
                "r": "Risk-free rate",
                "sigma": "Volatility",
                "T": "Time to maturity in years"
            },
        },
        "optimize_portfolio": {
            "function": optimize_max_sharpe,
            "description": "Find the optimal portfolio weights that maximize the Sharpe ratio.",
            "parameters": {
                "expected_returns": "List of expected returns per asset",
                "cov_matrix": "Covariance matrix (list of lists)",
                "risk_free_rate": "float, default 0.0"
            },
        },
        "simulate_gbm": {
            "function": simulate_geometric_brownian_motion,
            "description": "Simulate Geometric Brownian Motion paths for asset prices.",
            "parameters": {
                "S0": "Initial price",
                "mu": "Drift (expected return)",
                "sigma": "Volatility",
                "T": "Time horizon in years",
                "dt": "Time step (e.g. 1/252 for daily)",
                "n_paths": "Number of simulation paths"
            },
        },
        "compute_sharpe": {
            "function": lambda returns, risk_free_rate=0.0: (
                (mean(returns) - risk_free_rate) / standard_deviation(returns)
                if standard_deviation(returns) > 0 else 0.0
            ),
            "description": "Compute the Sharpe Ratio of a returns series.",
            "parameters": {
                "returns": "List of float returns",
                "risk_free_rate": "float, default 0.0"
            },
        },
        "estimate_factor_betas": {
            "function": estimate_factor_betas,
            "description": "Estimate factor sensitivities (betas) via OLS regression.",
            "parameters": {
                "asset_returns": "List of asset returns",
                "factor_returns": "List of lists of factor returns",
                "add_intercept": "bool, default True"
            },
        },
    }

    return registry


def execute_tool(tool_name, **params):
    """
    Universal tool dispatcher. Routes a tool call to the
    appropriate VectorQuant function.

    Args:
        tool_name: Name of the tool (must exist in registry).
        **params:  Keyword arguments passed to the function.

    Returns:
        The result of the function call.

    Raises:
        ValueError: If the tool name is not found in the registry.
    """
    registry = get_tool_registry()

    if tool_name not in registry:
        raise ValueError(
            f"Unknown tool: '{tool_name}'. "
            f"Available tools: {list(registry.keys())}"
        )

    func = registry[tool_name]["function"]
    return func(**params)


def get_tool_schemas():
    """
    Returns OpenAI function-calling compatible JSON schemas
    for all VectorQuant tools.
    
    Each schema follows the OpenAI format:
    {
        "type": "function",
        "function": {
            "name": "...",
            "description": "...",
            "parameters": { JSON Schema }
        }
    }
    """
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "calculate_var",
                "description": "Compute Parametric Value-at-Risk for a returns series.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "returns": {"type": "array", "items": {"type": "number"}, "description": "List of float returns"},
                        "confidence_level": {"type": "number", "default": 0.95, "description": "Confidence level (0-1)"}
                    },
                    "required": ["returns"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_cvar",
                "description": "Compute Conditional Value-at-Risk (Expected Shortfall).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "returns": {"type": "array", "items": {"type": "number"}, "description": "List of float returns"},
                        "confidence_level": {"type": "number", "default": 0.95, "description": "Confidence level (0-1)"}
                    },
                    "required": ["returns"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "price_call_option",
                "description": "Price a European call option using Black-Scholes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "S": {"type": "number", "description": "Current stock price"},
                        "K": {"type": "number", "description": "Strike price"},
                        "r": {"type": "number", "description": "Risk-free rate"},
                        "sigma": {"type": "number", "description": "Volatility"},
                        "T": {"type": "number", "description": "Time to maturity in years"}
                    },
                    "required": ["S", "K", "r", "sigma", "T"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "price_put_option",
                "description": "Price a European put option using Black-Scholes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "S": {"type": "number", "description": "Current stock price"},
                        "K": {"type": "number", "description": "Strike price"},
                        "r": {"type": "number", "description": "Risk-free rate"},
                        "sigma": {"type": "number", "description": "Volatility"},
                        "T": {"type": "number", "description": "Time to maturity in years"}
                    },
                    "required": ["S", "K", "r", "sigma", "T"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "optimize_portfolio",
                "description": "Find optimal portfolio weights maximizing the Sharpe ratio.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expected_returns": {"type": "array", "items": {"type": "number"}, "description": "Expected returns per asset"},
                        "cov_matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Covariance matrix"},
                        "risk_free_rate": {"type": "number", "default": 0.0, "description": "Risk-free rate"}
                    },
                    "required": ["expected_returns", "cov_matrix"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "simulate_gbm",
                "description": "Simulate Geometric Brownian Motion paths for asset prices.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "S0": {"type": "number", "description": "Initial price"},
                        "mu": {"type": "number", "description": "Drift (expected return)"},
                        "sigma": {"type": "number", "description": "Volatility"},
                        "T": {"type": "number", "description": "Time horizon in years"},
                        "dt": {"type": "number", "description": "Time step (e.g. 1/252 for daily)"},
                        "n_paths": {"type": "integer", "description": "Number of simulation paths"}
                    },
                    "required": ["S0", "mu", "sigma", "T", "dt", "n_paths"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compute_sharpe",
                "description": "Compute the Sharpe Ratio of a returns series.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "returns": {"type": "array", "items": {"type": "number"}, "description": "List of float returns"},
                        "risk_free_rate": {"type": "number", "default": 0.0, "description": "Risk-free rate"}
                    },
                    "required": ["returns"]
                }
            }
        },
    ]
    return schemas
