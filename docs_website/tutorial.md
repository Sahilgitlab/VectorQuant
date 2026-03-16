# Tutorial

30-minute complete walkthrough of VectorQuant.

---

## Phase 1: Foundations (10 minutes)

Learn core concepts and set up your first computation.

### Step 1: Installation & Verification (2 minutes)

```bash
pip install vectorquant
```

Verify installation:

```python
import vectorquant as vq

# Check which backend is active
backend = vq.core.get_backend()
print(f"Backend: {backend}")  # "C" (fast) or "Python" (slower)

# Load safe parameters
from vectorquant.core.mc_config import get_safe_test_params
params = get_safe_test_params()
print(f"Safe MC params: {params}")
```

### Step 2: Basic Statistics (3 minutes)

Analyze returns data:

```python
# Sample daily returns (0.5% to 2% per day)
returns = [0.005, 0.012, -0.008, 0.015, 0.003, -0.002, 0.010, 0.008]

# Calculate basic metrics
mean = vq.stats.mean(returns)
std = vq.stats.standard_deviation(returns)

print(f"Mean return: {mean:.4f}")    # ~0.0068 (0.68%)
print(f"Volatility: {std:.4f}")      # ~0.0099 (0.99%)

# Annualize (252 trading days)
annual_mean = mean * 252
annual_std = std * (252 ** 0.5)

print(f"Annual return: {annual_mean:.2%}")    # ~171%
print(f"Annual volatility: {annual_std:.2%}") # ~157%
```

**Key insight:** Always annualize for portfolio decisions.

### Step 3: Correlation (2 minutes)

Compare two assets:

```python
# Two assets over same period
asset1_returns = [0.005, 0.012, -0.008, 0.015, 0.003]
asset2_returns = [0.010, 0.008, 0.005, 0.020, -0.005]

# Compute correlation matrix
corr = vq.stats.correlation([asset1_returns, asset2_returns])
print("Correlation matrix:")
print(corr)
# [[1.0, 0.7],
#  [0.7, 1.0]]
# → Assets move together (70% correlation)
```

---

## Phase 2: Portfolio Optimization (10 minutes)

Build and optimize a portfolio.

### Step 4: Portfolio Metrics (3 minutes)

Compute return and volatility:

```python
# Three asset portfolio
weights = [0.5, 0.3, 0.2]  # 50% A, 30% B, 20% C
asset_returns = [0.08, 0.10, 0.06]  # Expected annual returns

# Portfolio return (weighted average)
port_return = vq.portfolio.portfolio_return(weights, asset_returns)
print(f"Portfolio return: {port_return:.2%}")  # 7.8%

# Now add volatility using historical data
historical_returns = [
    [0.01, 0.02, -0.01, 0.015, 0.008],  # Asset A daily returns
    [0.02, -0.01, 0.03, -0.005, 0.012], # Asset B daily returns
    [-0.01, 0.03, 0.02, 0.010, -0.008]  # Asset C daily returns
]

# Compute covariance
cov = vq.stats.covariance(historical_returns)

# Portfolio volatility
port_vol = vq.portfolio.portfolio_volatility(weights, cov)
print(f"Portfolio volatility: {port_vol:.2%}")  # ~1.2%

# Annualize
annual_port_vol = port_vol * (252 ** 0.5)
print(f"Annual volatility: {annual_port_vol:.2%}")  # ~19%
```

### Step 5: Risk-Adjusted Returns (2 minutes)

Sharpe ratio measures return per unit of risk:

```python
# Daily returns of a portfolio
daily_returns = [0.005, 0.012, -0.008, 0.015, 0.003, -0.002, 0.010]

# Risk-free rate (e.g., Treasury): 2% annual = 2%/252 daily
daily_rf = 0.02 / 252

# Calculate Sharpe ratio
sharpe = vq.portfolio.sharpe_ratio(daily_returns, risk_free_rate=daily_rf)
print(f"Sharpe ratio: {sharpe:.3f}")  # Higher is better (> 1.0 is good)
```

**Interpretation:**
- Sharpe > 1: Risk-adjusted returns are good
- Sharpe > 2: Exceptional
- Sharpe < 0: Worse than holding cash

### Step 6: Find Optimal Weights (3 minutes)

Let VectorQuant find the best portfolio:

```python
# Historical returns for 3 assets (many time periods)
asset_returns = [
    [0.01, 0.02, -0.01, 0.015, 0.008, ...],  # Asset A
    [0.02, -0.01, 0.03, -0.005, 0.012, ...], # Asset B
    [-0.01, 0.03, 0.02, 0.010, -0.008, ...] # Asset C
]

# Risk-free rate (2% annual)
rf = 0.02

# Find optimal weights that maximize Sharpe ratio
optimal_weights = vq.portfolio.optimize_max_sharpe(asset_returns, risk_free_rate=rf)

print("Optimal weights:")
for i, w in enumerate(optimal_weights):
    print(f"  Asset {i+1}: {w:.1%}")

# Typically: [0.45, 0.38, 0.17]
# → Concentrate in higher-return assets
```

### Step 7: 2 minutes - Compute metrics on optimized portfolio

```python
# Use optimal weights on original assets
port_return = vq.portfolio.portfolio_return(optimal_weights, [0.08, 0.10, 0.06])
cov = vq.stats.covariance(asset_returns)
port_vol = vq.portfolio.portfolio_volatility(optimal_weights, cov)
sharpe = vq.portfolio.sharpe_ratio(daily_returns_from_weights, rf)

print(f"Optimized - Return: {port_return:.2%}, Volatility: {port_vol:.2%}")
```

---

## Phase 3: Options Pricing (5 minutes)

Price options using Black-Scholes.

### Step 8: European Call Valuation (2 minutes)

```python
# Call option (right to buy at strike K)
S = 100      # Current stock price
K = 100      # Strike price (at-the-money)
r = 0.05     # Risk-free rate (5% annual)
sigma = 0.20 # Volatility (20% annual)
T = 1.0      # 1 year to expiration

# Price the option
call_price = vq.derivatives.black_scholes_call(S, K, r, sigma, T)
print(f"Call price: ${call_price:.2f}")  # ~$10.45

# What if stock goes up?
S_up = 110
call_price_up = vq.derivatives.black_scholes_call(S_up, K, r, sigma, T)
print(f"Call price if S=110: ${call_price_up:.2f}")  # ~$15.67
print(f"Change: ${call_price_up - call_price:.2f}")  # ~$5.22

# What if volatility increases?
sigma_high = 0.30
call_price_high_vol = vq.derivatives.black_scholes_call(S, K, r, sigma_high, T)
print(f"Call price if σ=30%: ${call_price_high_vol:.2f}")  # Higher
```

### Step 9: Greeks for Risk Management (2 minutes)

Greeks tell you how option value changes:

```python
# Delta: Change per $1 stock move
delta = vq.derivatives.bs_delta(100, 100, 0.05, 0.20, 1.0)
print(f"Delta: {delta:.4f}")  # ~0.6368
# → Option price increases $0.64 per $1 stock increase

# Gamma: How fast delta changes
gamma = vq.derivatives.bs_gamma(100, 100, 0.05, 0.20, 1.0)
print(f"Gamma: {gamma:.4f}")  # Small number
# → Delta stays relatively stable (gamma is low)

# Vega: Change per 1% volatility move
vega = vq.derivatives.bs_vega(100, 100, 0.05, 0.20, 1.0)
print(f"Vega: {vega:.2f}")  # ~$39.89
# → Option price increases $39.89 per 1% volatility increase

# Theta: Daily time decay
theta = vq.derivatives.bs_theta(100, 100, 0.05, 0.20, 1.0)
print(f"Theta: {theta:.2f}")  # Negative
# → Seller gains this per day (time decay benefit)
```

### Step 10: 1 minute - Real-world example

Hedge a short position:

```python
# You sold 100 shares short at $100
position = -100 * 100  # -$10,000

# Hedge with call options (buy the right to buy)
call_price = vq.derivatives.black_scholes_call(100, 100, 0.05, 0.20, 1.0)
num_contracts = 100  # 1 contract = 100 shares
hedge_cost = num_contracts * call_price
print(f"Hedge cost: ${hedge_cost:.2f}")  # ~$1,045

# Maximum loss: hedge_cost
# Upside: Unlimited
```

---

## Phase 4: Risk Analysis (3 minutes)

Measure and understand portfolio risk.

### Step 11: Value-at-Risk (1.5 minutes)

```python
# Daily returns of your portfolio
portfolio_returns = [0.005, 0.012, -0.008, 0.015, 0.003, 
                     -0.002, 0.010, 0.008, -0.020, 0.011,
                     0.007, -0.005, 0.013, -0.011, 0.009]

# 95% Value-at-Risk: Worst 5% of days
var_95 = vq.risk.parametric_var(portfolio_returns, confidence_level=0.95)
print(f"95% Daily VaR: {var_95:.2%}")  # ~1.8%
# Interpretation: 95% of days, loss < 1.8%
#                 5% of days, loss > 1.8% (bad days)

# Annualize
var_annual = var_95 * (252 ** 0.5)
print(f"95% Annual VaR: {var_annual:.2%}")  # ~28.6%
```

### Step 12: Conditional Value-at-Risk (1.5 minutes)

When things go wrong, how bad?

```python
# CVaR: Average loss when loss > VaR
cvar_95 = vq.risk.cvar(portfolio_returns, confidence_level=0.95)
print(f"95% Daily CVaR: {cvar_95:.2%}")  # ~2.5%
# Interpretation: On bad days (bottom 5%), average loss is 2.5%

# Compare to VaR
print(f"VaR:  {var_95:.2%}")   # 1.8%
print(f"CVaR: {cvar_95:.2%}")  # 2.5%
# Tail risk (CVaR > VaR) shows dangerous days are really dangerous
```

---

## Phase 5: Monte Carlo Simulation (2 minutes)

Forecast future prices under uncertainty.

### Step 13: Simulate Stock Paths (1 minute)

```python
from vectorquant.core.mc_config import get_safe_test_params

# Safety parameters (prevents PC hanging)
params = get_safe_test_params()  # {n_paths: 1000, n_steps: 50, dt: 0.02}

# Simulate 1 year of stock prices
paths = vq.stochastic.simulate_geometric_brownian_motion(
    S0=100,                    # Start at $100
    mu=0.08,                   # 8% expected return
    sigma=0.20,                # 20% volatility
    T=1.0,                      # 1 year
    n_paths=params['n_paths'],  # 1000 paths
    dt=params['dt']             # 0.02 year = 5-day steps
)

print(f"Shape: {len(paths)} paths × {len(paths[0])} steps")
print(f"Final prices: min=${min(p[-1] for p in paths):.2f}, "
      f"max=${max(p[-1] for p in paths):.2f}")

# Typical outcome: min ~$65, max ~$180
```

### Step 14: Price Options via Monte Carlo (1 minute)

```python
# Use Monte Carlo instead of Black-Scholes
price_mc = vq.stochastic.MonteCarloEngine.european_call(
    S0=100,
    K=100,
    r=0.05,
    sigma=0.20,
    T=1.0,
    n_paths=10000
)

# Compare with exact Black-Scholes
price_bs = vq.derivatives.black_scholes_call(100, 100, 0.05, 0.20, 1.0)

print(f"Black-Scholes: ${price_bs:.2f}")
print(f"Monte Carlo:   ${price_mc:.2f}")
# Should be very close (~$10.45)
```

---

## Complete Example: Build a Quant Model

10-minute end-to-end workflow.

```python
import vectorquant as vq
from vectorquant.core.mc_config import get_safe_test_params

# 1. Load data (pretend 3 assets, 100 days of returns)
asset_returns = [
    [0.01, 0.02, -0.01, ..., 0.005],  # Asset A
    [0.02, -0.01, 0.03, ..., 0.008],  # Asset B
    [-0.01, 0.03, 0.02, ..., -0.002]  # Asset C
]

# 2. Optimize portfolio
optimal_w = vq.portfolio.optimize_max_sharpe(asset_returns, 0.02)
print(f"Optimal weights: {optimal_w}")

# 3. Compute metrics
port_return = vq.portfolio.portfolio_return(optimal_w, [0.08, 0.10, 0.06])
cov = vq.stats.covariance(asset_returns)
port_vol = vq.portfolio.portfolio_volatility(optimal_w, cov)
sharpe = vq.portfolio.sharpe_ratio(
    [optimal_w[i] * returns[i] for i, returns in enumerate(asset_returns)],
    0.02
)

print(f"Expected return: {port_return:.2%}")
print(f"Volatility: {port_vol:.2%}")
print(f"Sharpe ratio: {sharpe:.3f}")

# 4. Simulate forward paths and compute VaR
params = get_safe_test_params()
paths = vq.stochastic.simulate_geometric_brownian_motion(
    S0=100, mu=port_return, sigma=port_vol, T=5, 
    n_paths=params['n_paths'], dt=params['dt']
)

# Final values after 5 years
final_values = [p[-1] for p in paths]
var_95 = vq.risk.parametric_var(
    [(v - 100) / 100 for v in final_values],  # Convert to returns
    0.95
)

print(f"5-year portfolio value range: ${min(final_values):.0f}-${max(final_values):.0f}")
print(f"95% worst-case loss: {var_95:.2%}")

# 5. Verify using AI
verification = vq.ai.verify_calculation(
    f"sqrt({port_vol**2}) * sqrt(252)",
    expected=port_vol * (252 ** 0.5),
    tolerance=1e-10
)
print(f"Annual vol calculation verified: {verification.verified}")
```

---

## Summary

**You've learned:**

✓ Installation and backend selection  
✓ Basic statistics (mean, std, correlation)  
✓ Portfolio optimization and Sharpe ratio  
✓ Options pricing and Greeks  
✓ Value-at-Risk and tail risk  
✓ Monte Carlo simulation  
✓ Complete quant workflow  

**Next steps:**

- Explore [Module Guide](modules.md) for deeper dives
- Check [API Reference](api-reference.md) for all functions
- Review [Benchmarks](benchmarks.md) for performance details
- See [FAQ](faq.md) for troubleshooting

**Running time:** 30 minutes start → working code

---

**Ready to use VectorQuant for your problems?**

See [Quick Start](quickstart.md) for minimal 5-minute example, or start with your own data.
