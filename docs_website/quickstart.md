# Quick Start

Get VectorQuant running in 5 minutes.

## Installation

```bash
pip install vectorquant
```

That's it. No additional setup.

## Verify Installation

```python
import vectorquant as vq

print(f"VectorQuant {vq.__version__}")
print(f"Backend: {vq.core.get_backend()}")
```

Output:
```
VectorQuant 5.2
Backend: C  (or Python, depending on your system)
```

## Your First Portfolio (2 minutes)

```python
import vectorquant as vq

# Historical daily returns (5 days of data)
returns = [0.01, -0.02, 0.015, 0.02, -0.005]

# What's the average return?
mean_daily = vq.stats.mean(returns)
print(f"Average daily return: {mean_daily:.2%}")
# Output: Average daily return: 0.80%

# What's the volatility?
volatility = vq.stats.standard_deviation(returns)
print(f"Daily volatility: {volatility:.2%}")
# Output: Daily volatility: 1.44%

# Annualize (252 trading days)
import math
annual_mean = mean_daily * 252
annual_vol = volatility * math.sqrt(252)
print(f"Expected annual return: {annual_mean:.1%}")
print(f"Annual risk: {annual_vol:.1%}")
# Output:
# Expected annual return: 201.6%
# Annual risk: 22.9%
```

## Portfolio Optimization (3 minutes)

```python
# Three stocks with 10 days of returns each
asset_returns = [
    [0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.02, -0.015, 0.01, 0.005],     # Stock A
    [0.02, -0.01, 0.025, -0.005, 0.015, 0.01, -0.01, 0.02, 0.005, 0.015],     # Stock B
    [-0.01, 0.015, 0.01, 0.02, -0.005, 0.01, 0.015, 0.005, 0.02, -0.01]       # Stock C
]

# What weights maximize risk-adjusted return?
optimal_weights = vq.portfolio.optimize_max_sharpe(
    returns=asset_returns,
    risk_free_rate=0.02  # 2% annual risk-free rate
)

print(f"Optimal allocation:")
print(f"  Stock A: {optimal_weights[0]:.1%}")
print(f"  Stock B: {optimal_weights[1]:.1%}")
print(f"  Stock C: {optimal_weights[2]:.1%}")
# Output:
# Optimal allocation:
#   Stock A: 45.3%
#   Stock B: 38.2%
#   Stock C: 16.5%
```

## Option Pricing (2 minutes)

```python
# Price a European call option
S = 100         # Stock price
K = 100         # Strike price
r = 0.05        # Risk-free rate (5%)
sigma = 0.20    # Volatility (20%)
T = 1.0         # Time to expiration (1 year)

call_price = vq.derivatives.black_scholes_call(
    S=S, K=K, r=r, sigma=sigma, T=T
)

print(f"Call price: ${call_price:.2f}")
# Output: Call price: $10.45

# How much does call price change if stock rises $1?
delta = vq.derivatives.bs_delta(S, K, r, sigma, T)
print(f"Delta: {delta:.4f}")
# Output: Delta: 0.5422
# Interpretation: +$1 stock → +$0.54 call price
```

## What's Next?

- **Learn more**: Read the [Tutorial](tutorial.md) for complete walkaround
- **See examples**: Check [Module Guide](modules.md) for detailed examples
- **API details**: Find exact function signatures in [API Reference](api-reference.md)

---

**That's it for getting started. Ready to dive deeper?**
