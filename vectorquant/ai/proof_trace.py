"""
AI Proof Trace Engine

Provides step-by-step computation traces so AI systems can explain
HOW a result was derived, not just WHAT it is.
Each function returns an ExplanationTrace with traceable intermediate steps.
"""

import math


class ExplanationTrace:
    """Structured proof trace for a computation."""
    def __init__(self, result, method, formula, steps):
        self.result = result
        self.method = method
        self.formula = formula
        self.steps = steps  # List of {"step": str, "value": float}

    def to_dict(self):
        return {
            "result": self.result,
            "method": self.method,
            "formula": self.formula,
            "steps": self.steps,
        }

    def __repr__(self):
        step_str = "\n".join(f"  {i+1}. {s['step']} = {s['value']}" 
                             for i, s in enumerate(self.steps))
        return (f"ExplanationTrace(result={self.result}, method={self.method})\n"
                f"Steps:\n{step_str}")


def explain_var(returns, confidence=0.95):
    """
    Step-by-step derivation of Parametric Value-at-Risk.
    VaR = -(mu + z * sigma)
    """
    from vectorquant.core.statistics import mean, standard_deviation
    from vectorquant.core.probability import normal_inv_cdf

    steps = []

    mu = mean(returns)
    steps.append({"step": "mu = mean(returns)", "value": round(mu, 6)})

    sigma = standard_deviation(returns)
    steps.append({"step": "sigma = std(returns)", "value": round(sigma, 6)})

    z = normal_inv_cdf(1 - confidence)
    steps.append({"step": f"z = InvNorm(1 - {confidence})", "value": round(z, 6)})

    var = -(mu + z * sigma)
    steps.append({"step": "VaR = -(mu + z * sigma)", "value": round(var, 6)})

    return ExplanationTrace(
        result=round(var, 6),
        method="Parametric VaR",
        formula="VaR = -(mu + z * sigma)",
        steps=steps
    )


def explain_sharpe(returns, risk_free_rate=0.0):
    """
    Step-by-step derivation of the Sharpe Ratio.
    Sharpe = (mu - r_f) / sigma
    """
    from vectorquant.core.statistics import mean, standard_deviation

    steps = []

    mu = mean(returns)
    steps.append({"step": "mu = mean(returns)", "value": round(mu, 6)})

    sigma = standard_deviation(returns)
    steps.append({"step": "sigma = std(returns)", "value": round(sigma, 6)})

    excess = mu - risk_free_rate
    steps.append({"step": f"excess_return = mu - r_f ({risk_free_rate})", "value": round(excess, 6)})

    sharpe = excess / sigma if sigma > 0 else 0.0
    steps.append({"step": "Sharpe = excess_return / sigma", "value": round(sharpe, 6)})

    return ExplanationTrace(
        result=round(sharpe, 6),
        method="Sharpe Ratio",
        formula="Sharpe = (mu - r_f) / sigma",
        steps=steps
    )


def explain_black_scholes(S, K, r, sigma, T):
    """
    Step-by-step Black-Scholes call option price derivation.
    """
    from vectorquant.core.probability import normal_cdf

    steps = []

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    steps.append({"step": "d1 = (ln(S/K) + (r + sigma^2/2)*T) / (sigma*sqrt(T))", "value": round(d1, 6)})

    d2 = d1 - sigma * math.sqrt(T)
    steps.append({"step": "d2 = d1 - sigma*sqrt(T)", "value": round(d2, 6)})

    nd1 = normal_cdf(d1)
    steps.append({"step": "N(d1)", "value": round(nd1, 6)})

    nd2 = normal_cdf(d2)
    steps.append({"step": "N(d2)", "value": round(nd2, 6)})

    call_price = S * nd1 - K * math.exp(-r * T) * nd2
    steps.append({"step": "C = S*N(d1) - K*e^(-rT)*N(d2)", "value": round(call_price, 6)})

    return ExplanationTrace(
        result=round(call_price, 6),
        method="Black-Scholes Call",
        formula="C = S*N(d1) - K*e^(-rT)*N(d2)",
        steps=steps
    )


def explain_monte_carlo(S0, K, r, sigma, T, n_paths=10000):
    """
    Step-by-step Monte Carlo European Call pricing with standard error.
    """
    from vectorquant.stochastic.processes import simulate_geometric_brownian_motion
    from vectorquant.core.statistics import mean, standard_deviation

    steps = []

    steps.append({"step": f"Simulate {n_paths} GBM paths (S0={S0}, r={r}, sigma={sigma}, T={T})", "value": n_paths})

    paths = simulate_geometric_brownian_motion(S0, r, sigma, T, T, n_paths)
    terminal_prices = [p[-1] for p in paths]

    avg_terminal = mean(terminal_prices)
    steps.append({"step": "E[S_T] = mean of terminal prices", "value": round(avg_terminal, 4)})

    payoffs = [max(st - K, 0.0) for st in terminal_prices]
    expected_payoff = mean(payoffs)
    steps.append({"step": "E[max(S_T - K, 0)]", "value": round(expected_payoff, 4)})

    discount_factor = math.exp(-r * T)
    steps.append({"step": "Discount factor = e^(-rT)", "value": round(discount_factor, 6)})

    price = discount_factor * expected_payoff
    steps.append({"step": "Price = discount * E[payoff]", "value": round(price, 4)})

    se = standard_deviation(payoffs) / math.sqrt(n_paths)
    steps.append({"step": "Standard error = std(payoffs) / sqrt(N)", "value": round(se, 6)})

    return ExplanationTrace(
        result=round(price, 4),
        method="Monte Carlo European Call",
        formula="C = e^(-rT) * E[max(S_T - K, 0)]",
        steps=steps
    )
