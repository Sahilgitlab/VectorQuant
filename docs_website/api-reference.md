# API Reference

Complete function documentation for VectorQuant 5.2.

---

## Statistics Module (`vq.stats`)

### mean(data: list[float]) → float

Calculate arithmetic mean of values.

**Parameters:**
- `data` (list[float]): Values to average

**Returns:** float - Arithmetic mean

**Example:**
```python
import vectorquant as vq
result = vq.stats.mean([1, 2, 3, 4, 5])
# Returns: 3.0
```

**Notes:**
- Deterministic computation
- Works on any size list
- O(n) time complexity

---

### standard_deviation(data: list[float]) → float

Calculate sample standard deviation.

**Parameters:**
- `data` (list[float]): Values to analyze

**Returns:** float - Sample standard deviation

**Example:**
```python
result = vq.stats.standard_deviation([1, 2, 3, 4, 5])
# Returns: 1.5811...
```

**Formula:** √(Σ(x - mean)² / (n-1))

---

### variance(data: list[float]) → float

Calculate sample variance.

**Parameters:**
- `data` (list[float]): Values to analyze

**Returns:** float - Sample variance

**Example:**
```python
result = vq.stats.variance([1, 2, 3, 4, 5])
# Returns: 2.5
```

**Notes:**
- Variance = Std Dev²
- Uses Bessel's correction (n-1 denominator)

---

### skewness(data: list[float]) → float

Measure of asymmetry in distribution.

**Parameters:**
- `data` (list[float]): Returns or prices

**Returns:** float - Skewness coefficient (-∞ to +∞)

**Interpretation:**
- < 0: Left-skewed (negative tail)
- = 0: Symmetric
- > 0: Right-skewed (positive tail)

**Example:**
```python
result = vq.stats.skewness(daily_returns)
# -0.45 means slight negative skew (losses worse than gains)
```

---

### kurtosis(data: list[float]) → float

Measure of tail fatness.

**Parameters:**
- `data` (list[float]): Returns or prices

**Returns:** float - Excess kurtosis

**Interpretation:**
- < 3: Thin tails (fewer extreme moves)
- ≈ 3: Normal distribution
- > 3: Fat tails (more extreme moves)

**Example:**
```python
result = vq.stats.kurtosis(daily_returns)
# 5.2 means fatter tails than normal (more crashes)
```

---

### covariance(series: list[list[float]]) → list[list[float]]

Covariance matrix of multiple return series.

**Parameters:**
- `series` (list[list[float]]): Multiple time series

**Returns:** list[list[float]] - Covariance matrix (n×n)

**Example:**
```python
returns = [
    [0.01, 0.02, -0.01],  # Asset 1
    [0.02, -0.01, 0.03]   # Asset 2
]
cov = vq.stats.covariance(returns)
# Returns 2×2 covariance matrix
```

**Notes:**
- Symmetric matrix
- Diagonal = variance
- Off-diagonal = covariance between assets

---

### correlation(series: list[list[float]]) → list[list[float]]

Correlation matrix (normalized covariance).

**Parameters:**
- `series` (list[list[float]]): Multiple time series

**Returns:** list[list[float]] - Correlation matrix (-1 to +1)

**Example:**
```python
corr = vq.stats.correlation([series1, series2])
# Returns correlation coefficient (-1, 0, or +1)
```

**Notes:**
- Range: -1 (perfect negative) to +1 (perfect positive)
- 0 = no correlation

---

## Portfolio Module (`vq.portfolio`)

### portfolio_return(weights: list[float], returns: list[float]) → float

Weighted average return of portfolio.

**Parameters:**
- `weights` (list[float]): Asset weights (must sum to 1.0)
- `returns` (list[float]): Average return per asset

**Returns:** float - Portfolio return

**Example:**
```python
weights = [0.4, 0.3, 0.3]      # 40% A, 30% B, 30% C
returns = [0.08, 0.10, 0.06]   # Expected returns
port_ret = vq.portfolio.portfolio_return(weights, returns)
# Returns: 0.082 (8.2%)
```

---

### portfolio_volatility(weights: list[float], covariance: list[list[float]]) → float

Portfolio standard deviation given asset weights and covariance.

**Parameters:**
- `weights` (list[float]): Asset weights
- `covariance` (list[list[float]]): Covariance matrix

**Returns:** float - Portfolio volatility

**Example:**
```python
cov = vq.stats.covariance(returns)
weights = [0.4, 0.3, 0.3]
vol = vq.portfolio.portfolio_volatility(weights, cov)
# Returns: 0.135 (13.5% annual volatility)
```

---

### sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) → float

Risk-adjusted return metric.

**Parameters:**
- `returns` (list[float]): Historical returns
- `risk_free_rate` (float, optional): Risk-free rate (default 0.0)

**Returns:** float - Sharpe ratio

**Formula:** (mean(returns) - risk_free_rate) / std(returns)

**Example:**
```python
sharpe = vq.portfolio.sharpe_ratio(daily_returns, risk_free_rate=0.02/252)
# Returns: 1.23
# Interpretation: 1.23x excess return per unit of risk
```

---

### optimize_max_sharpe(returns: list[list[float]], risk_free_rate: float) → list[float]

Find portfolio weights that maximize Sharpe ratio.

**Parameters:**
- `returns` (list[list[float]]): Historical returns per asset
- `risk_free_rate` (float): Annual risk-free rate

**Returns:** list[float] - Optimal weights

**Example:**
```python
asset_returns = [
    [0.01, 0.02, -0.01, ...],  # Asset A returns
    [0.02, -0.01, 0.03, ...],  # Asset B returns
    [-0.01, 0.03, 0.02, ...]   # Asset C returns
]
optimal = vq.portfolio.optimize_max_sharpe(asset_returns, risk_free_rate=0.02)
# Returns: [0.45, 0.38, 0.17]  (45% A, 38% B, 17% C)
```

---

### black_litterman_returns(observed: list[float], views: list[float], confidence: float) → list[float]

Adjusted expected returns incorporating views and confidence.

**Parameters:**
- `observed` (list[float]): Market-implied expected returns
- `views` (list[float]): Your return estimates
- `confidence` (float): Confidence in views (0 to 1)

**Returns:** list[float] - Adjusted expected returns

**Example:**
```python
market = [0.08, 0.10, 0.06]     # Market consensus
my_views = [0.09, 0.10, 0.07]   # Your estimates
adjusted = vq.portfolio.black_litterman_returns(
    observed=market,
    views=my_views,
    confidence=0.8
)
# Returns: Blend of market and your views
```

---

## Derivatives Module (`vq.derivatives`)

### black_scholes_call(S: float, K: float, r: float, sigma: float, T: float) → float

Price European call option.

**Parameters:**
- `S` (float): Current stock price
- `K` (float): Strike price
- `r` (float): Risk-free rate (annual)
- `sigma` (float): Volatility (annual)
- `T` (float): Time to expiration (years)

**Returns:** float - Call option price

**Example:**
```python
call_price = vq.derivatives.black_scholes_call(
    S=100, K=100, r=0.05, sigma=0.20, T=1.0
)
# Returns: 10.45
```

---

### black_scholes_put(S: float, K: float, r: float, sigma: float, T: float) → float

Price European put option.

**Parameters:** Same as call

**Returns:** float - Put option price

---

### bs_delta(S: float, K: float, r: float, sigma: float, T: float) → float

Option delta (price sensitivity to stock).

**Interpretation:** Option price changes by delta per $1 stock move.

**Range:** 0 to 1 for calls, -1 to 0 for puts

**Example:**
```python
delta = vq.derivatives.bs_delta(100, 100, 0.05, 0.20, 1.0)
# Returns: 0.5422
# Call price increases $0.54 per $1 stock increase
```

---

### bs_gamma(S: float, K: float, r: float, sigma: float, T: float) → float

Change in delta per $1 stock move.

**Interpretation:** Higher gamma = delta changes faster

---

### bs_vega(S: float, K: float, r: float, sigma: float, T: float) → float

Option sensitivity to volatility.

**Interpretation:** Option price changes by vega per 1% volatility change

---

### bs_theta(S: float, K: float, r: float, sigma: float, T: float) → float

Time decay (daily option loss as expiration approaches).

**Interpretation:** Option loses approximately theta per day

---

### bs_rho(S: float, K: float, r: float, sigma: float, T: float) → float

Interest rate sensitivity.

**Interpretation:** Option price changes by rho per 1% rate change

---

## Risk Module (`vq.risk`)

### parametric_var(returns: list[float], confidence_level: float = 0.95) → float

Value-at-Risk assuming normal distribution.

**Parameters:**
- `returns` (list[float]): Historical returns
- `confidence_level` (float): Confidence (default 0.95 = 95%)

**Returns:** float - VaR loss (positive number)

**Interpretation:** 95% VaR = 0.032 means 95% of the time loss < 3.2%

**Example:**
```python
var = vq.risk.parametric_var(daily_returns, confidence_level=0.95)
# Returns: 0.032
# Interpretation: Worst 5% of days lose more than 3.2%
```

---

### historical_var(returns: list[float], confidence_level: float = 0.95) → float

Value-at-Risk from empirical distribution.

**Parameters:** Same as parametric_var

**Returns:** float - VaR loss

**Notes:**
- More robust than parametric (doesn't assume normal)
- Requires sufficient historical data

---

### cvar(returns: list[float], confidence_level: float = 0.95) → float

Conditional Value-at-Risk (expected loss when worse than VaR).

**Parameters:** Same as VaR

**Returns:** float - CVaR loss

**Interpretation:** When loss > VaR, expected loss is CVaR

**Example:**
```python
cvar = vq.risk.cvar(daily_returns, confidence_level=0.95)
# Returns: 0.045
# Interpretation: In worst 5% days, average loss is 4.5%
```

---

## Stochastic Module (`vq.stochastic`)

### simulate_geometric_brownian_motion(S0: float, mu: float, sigma: float, T: float, n_paths: int, dt: float) → list[list[float]]

Monte Carlo simulation of stock prices.

**Parameters:**
- `S0` (float): Initial stock price
- `mu` (float): Drift (expected return)
- `sigma` (float): Volatility
- `T` (float): Time horizon (years)
- `n_paths` (int): Number of paths to simulate
- `dt` (float): Time step (typically 1/252 for daily)

**Returns:** list[list[float]] - Price paths (n_paths × n_steps)

**Example:**
```python
from vectorquant.core.mc_config import get_safe_test_params
params = get_safe_test_params()

paths = vq.stochastic.simulate_geometric_brownian_motion(
    S0=100, mu=0.05, sigma=0.20, T=1.0,
    n_paths=params['n_paths'],
    dt=params['dt']
)
# Returns: 1000 price paths over 1 year
```

**Notes:**
- Use `get_safe_test_params()` for reasonable defaults
- Each row is one simulation path
- Each column is one time step

---

### MonteCarloEngine.european_call(S0, K, r, sigma, T, n_paths)

Price European call via Monte Carlo.

**Parameters:**
- `S0` (float): Initial stock price
- `K` (float): Strike price
- `r` (float): Risk-free rate
- `sigma` (float): Volatility
- `T` (float): Time to expiration
- `n_paths` (int): Simulation paths

**Returns:** float - Call option price

**Example:**
```python
price = vq.stochastic.MonteCarloEngine.european_call(
    S0=100, K=100, r=0.05, sigma=0.20, T=1.0,
    n_paths=10000
)
# Returns: ~10.45 (matches Black-Scholes)
```

---

## AI Verification Module (`vq.ai`)

### verify_calculation(expression: str, expected: float, tolerance: float = 1e-6) → VerificationResult

Verify a mathematical expression.

**Parameters:**
- `expression` (str): Math expression (e.g., "sqrt(4) * 3")
- `expected` (float): Expected result
- `tolerance` (float): Acceptable error

**Returns:** VerificationResult
- `.verified` (bool): Did it match?
- `.computed_value` (float): Actual value
- `.confidence` (float): Confidence (0-1)

**Example:**
```python
result = vq.ai.verify_calculation("sqrt(4) * 3", expected=6.0)
print(f"Verified: {result.verified}")         # True
print(f"Confidence: {result.confidence}")     # 1.0
```

---

### explain_sharpe(returns: list[float], risk_free_rate: float = 0.0) → ExplanationTrace

Step-by-step computation of Sharpe ratio.

**Parameters:**
- `returns` (list[float]): Historical returns
- `risk_free_rate` (float): Risk-free rate

**Returns:** ExplanationTrace
- `.result` (float): Final Sharpe ratio
- `.steps` (list[dict]): Each computation step
- `.formula` (str): Mathematical formula used

**Example:**
```python
trace = vq.ai.explain_sharpe([0.01, -0.02, 0.015], risk_free_rate=0.02)
for step in trace.steps:
    print(f"{step['step']} = {step['value']:.6f}")
# Prints step-by-step calculation
```

---

### HallucinationProofPipeline.process(intent: str, **params) → PipelineResult

Full verification pipeline: compute → verify → explain.

**Parameters:**
- `intent` (str): What to compute ("var", "sharpe", "call_option", etc.)
- `**params`: Parameters for the computation

**Returns:** PipelineResult
- `.result` (float): Computed value
- `.verified` (bool): Is it reliable?
- `.confidence` (float): Confidence score (0-1)
- `.proof_trace` (dict): Step-by-step proof

**Example:**
```python
pipeline = vq.ai.HallucinationProofPipeline()
result = pipeline.process(
    "sharpe",
    returns=[0.01, -0.02, 0.015],
    risk_free_rate=0.02
)
print(f"Result: {result.result:.4f}")
print(f"Verified: {result.verified}")
print(f"Confidence: {result.confidence:.0%}")
```

---

## Core Module (`vq.core`)

### create_rng(seed: int) → RNG

Create deterministic random number generator.

**Parameters:**
- `seed` (int): Random seed

**Returns:** RNG - Seeded generator

**Example:**
```python
rng = vq.core.create_rng(seed=42)
# Same seed always produces same random numbers
```

---

### get_backend() → str

Check which backend is active.

**Returns:** "C" or "Python"

**Example:**
```python
backend = vq.core.get_backend()
if backend == "C":
    print("165x speedup active")
else:
    print("Using pure Python fallback")
```

---

### gradient_descent(f, grad, x0, lr: float, max_iter: int) → list[float]

Minimize objective function using gradient descent.

**Parameters:**
- `f` (callable): Objective function
- `grad` (callable): Gradient function
- `x0` (list[float]): Starting point
- `lr` (float): Learning rate (step size)
- `max_iter` (int): Maximum iterations

**Returns:** list[float] - Optimal point

**Example:**
```python
def objective(x):
    return (x[0] - 3)**2 + (x[1] + 2)**2

def gradient(x):
    return [2*(x[0] - 3), 2*(x[1] + 2)]

x_opt = vq.core.gradient_descent(
    f=objective,
    grad=gradient,
    x0=[0, 0],
    lr=0.01,
    max_iter=100
)
# Returns: [3.0, -2.0]
```

---

## Configuration

### get_safe_test_params() → dict

Returns safe Monte Carlo parameters (prevents PC hanging).

**Returns:**
```python
{
    'n_paths': 1000,    # Simulation paths
    'n_steps': 50,      # Time steps
    'dt': 0.02          # Time step size
}
```

**Usage:**
```python
from vectorquant.core.mc_config import get_safe_test_params
params = get_safe_test_params()
paths = vq.stochastic.simulate_gbm(..., n_paths=params['n_paths'], dt=params['dt'])
```

---

## Type Hints

All functions have type hints. Example:

```python
def mean(data: list[float]) -> float: ...
def optimize_max_sharpe(
    returns: list[list[float]], 
    risk_free_rate: float
) -> list[float]: ...
```

These match the signatures throughout this reference.
