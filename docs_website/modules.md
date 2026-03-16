# Modules Guide

Complete reference for each VectorQuant module.

---

## Statistics Module (`vq.stats`)

**Purpose:** Fundamental statistical operations

### Functions

#### `mean(data: list[float]) → float`

Arithmetic average.

```python
result = vq.stats.mean([1, 2, 3])
# Returns: 2.0
```

#### `standard_deviation(data: list[float]) → float`

Sample standard deviation ($\sqrt{\frac{\sum (x_i - \mu)^2}{n-1}}$).

```python
result = vq.stats.standard_deviation([1, 2, 3])
# Returns: 1.0
```

**Note:** Uses $n-1$ denominator (sample, not population).

#### `variance(data: list[float]) → float`

Sample variance ($\frac{\sum (x_i - \mu)^2}{n-1}$).

```python
result = vq.stats.variance([1, 2, 3])
# Returns: 1.0
```

**Relationship:** variance = std²

#### `skewness(data: list[float]) → float`

Measure of distribution asymmetry.

```python
# Left-skewed (negative tail)
result1 = vq.stats.skewness([-2, -1, 0, 1, 5])
# Returns: ~0.5 (right tail)

# Right-skewed
result2 = vq.stats.skewness([-5, -1, 0, 1, 2])
# Returns: ~-0.5 (left tail)
```

**Finance interpretation:**
- Negative skew: More frequent small gains, rare large losses (stocks have this)
- Positive skew: More frequent small losses, rare large gains

#### `kurtosis(data: list[float]) → float`

Tail fatness (excess kurtosis, normal = 0).

```python
# Normal-like distribution
result1 = vq.stats.kurtosis(generate_normal(1000))
# Returns: ~0

# Fat tails (crash risk)
result2 = vq.stats.kurtosis(stock_returns)
# Returns: 3.5
```

**Interpretation:**
- Excess kurtosis > 0: Fatter tails than normal (more crashes possible)

#### `covariance(series: list[list[float]]) → list[list[float]]`

Covariance between multiple series.

```python
series = [
    [0.01, 0.02, -0.01, 0.015],  # Series 1
    [0.02, -0.01, 0.03, -0.005]   # Series 2
]
result = vq.stats.covariance(series)
# Returns: 2×2 matrix
# [[cov(1,1), cov(1,2)],
#  [cov(2,1), cov(2,2)]]
```

**Key insight:** Diagonal = variances, off-diagonal = covariances

#### `correlation(series: list[list[float]]) → list[list[float]]`

Normalized covariance (ranges -1 to +1).

```python
result = vq.stats.correlation(series)
# Returns: [[1.0, 0.65],
#           [0.65, 1.0]]
```

**Interpretation:**
- 1.0: Perfect positive correlation (move together)
- 0.0: No correlation (independent)
- -1.0: Perfect negative correlation (opposite moves)

---

## Portfolio Module (`vq.portfolio`)

**Purpose:** Portfolio construction and optimization

### Core Concepts

**Weight:** Allocation to each asset (must sum to 1.0)

```python
weights = [0.4, 0.3, 0.3]  # 40%, 30%, 30%
sum(weights) == 1.0  # Must be true
```

**Return:** Expected or historical return per asset (annual).

```python
returns = [0.08, 0.10, 0.06]  # 8%, 10%, 6% annual
```

### Functions

#### `portfolio_return(weights: list[float], returns: list[float]) → float`

Weighted average expected return.

```python
weights = [0.4, 0.3, 0.3]
returns = [0.08, 0.10, 0.06]
result = vq.portfolio.portfolio_return(weights, returns)
# Returns: 0.082 (8.2%)
# Calculation: 0.4*0.08 + 0.3*0.10 + 0.3*0.06
```

#### `portfolio_volatility(weights: list[float], covariance: list[list[float]]) → float`

Portfolio risk (standard deviation).

```python
weights = [0.4, 0.3, 0.3]
cov = [[0.001, 0.0005, 0.0002],
       [0.0005, 0.0015, 0.0003],
       [0.0002, 0.0003, 0.0008]]
result = vq.portfolio.portfolio_volatility(weights, cov)
# Returns: 0.035 (3.5% daily)
```

**Formula:** $\sigma_p = \sqrt{w^T \Sigma w}$

#### `sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) → float`

Risk-adjusted return metric.

```python
daily_returns = [0.001, 0.002, -0.001, 0.003]
rf_daily = 0.02 / 252  # 2% annual risk-free rate
result = vq.portfolio.sharpe_ratio(daily_returns, risk_free_rate=rf_daily)
# Returns: 1.23
```

**Interpretation:**
- > 1.0: Good
- > 2.0: Exceptional
- < 0: Worse than bonds

**Formula:** $S = \frac{\mu - r_f}{\sigma}$

#### `optimize_max_sharpe(returns: list[list[float]], risk_free_rate: float) → list[float]`

Find optimal portfolio weights.

```python
asset_returns = [
    [0.01, 0.02, -0.01, ...],   # Asset A daily returns
    [0.02, -0.01, 0.03, ...],   # Asset B daily returns
]
weights = vq.portfolio.optimize_max_sharpe(asset_returns, risk_free_rate=0.02/252)
# Returns: [0.45, 0.55]
```

**What it does:**
1. Tries weight combinations
2. Computes Sharpe ratio for each
3. Returns weights with highest Sharpe

**Behind scenes:** L-BFGS optimizer

#### `black_litterman_returns(observed: list[float], views: list[float], confidence: float) → list[float]`

Blend market estimates with your views.

```python
market = [0.08, 0.10, 0.06]       # Market consensus
my_view = [0.09, 0.10, 0.07]      # Your estimates
adjusted = vq.portfolio.black_litterman_returns(
    observed=market,
    views=my_view,
    confidence=0.7  # 70% confidence in your view
)
# Returns: blend of market and your view
```

**Interpretation:**
- Confidence = 1.0: Use your estimate entirely
- Confidence = 0.5: 50/50 blend
- Confidence = 0.0: Ignore your view (use market)

---

## Derivatives Module (`vq.derivatives`)

**Purpose:** Option pricing and risk metrics

### Black-Scholes Model

**Assumptions:**
- European options (exercise only at expiration)
- No dividends
- Constant volatility
- Log-normal stock prices

### Pricing Functions

#### `black_scholes_call(S, K, r, sigma, T) → float`

European call option price.

```python
S = 100      # Stock price
K = 100      # Strike price
r = 0.05     # Risk-free rate (5% annual)
sigma = 0.20 # Volatility (20% annual)
T = 1.0      # 1 year to expiration

price = vq.derivatives.black_scholes_call(S, K, r, sigma, T)
# Returns: ~10.45
```

**Intuition:**
- At-the-money (S=K): Price based on volatility
- In-the-money (S>K): Floor is S-K
- Out-of-the-money (S<K): Decreases with time

#### `black_scholes_put(S, K, r, sigma, T) → float`

European put option price.

```python
price = vq.derivatives.black_scholes_put(100, 100, 0.05, 0.20, 1.0)
# Returns: ~5.57
```

**Relationship:** Call - Put = S - K*e^(-rT) (put-call parity)

### Greeks (Sensitivities)

#### `bs_delta(S, K, r, sigma, T) → float`

Price change per $1 stock move.

```python
delta = vq.derivatives.bs_delta(100, 100, 0.05, 0.20, 1.0)
# Returns: ~0.64 (call)
```

**Ranges:**
- Call delta: 0 to 1
- Put delta: -1 to 0

**Management use:**
- Delta 0.5: 50% of full stock movement
- Delta 1.0: Fully hedged by owning stock

#### `bs_gamma(S, K, r, sigma, T) → float`

Delta change per $1 stock move.

```python
gamma = vq.derivatives.bs_gamma(100, 100, 0.05, 0.20, 1.0)
# Returns: ~0.03
```

**Interpretation:**
- If stock moves $1, delta increases by 0.03
- High gamma: Delta changes rapidly (risky)
- Low gamma: Delta stable (safe)

#### `bs_vega(S, K, r, sigma, T) → float`

Price change per 1% volatility increase.

```python
vega = vq.derivatives.bs_vega(100, 100, 0.05, 0.20, 1.0)
# Returns: ~39.89
```

**Management use:**
- If volatility increases 1%: option gains $39.89
- Volatility traders focus on vega

#### `bs_theta(S, K, r, sigma, T) → float`

Daily time decay.

```python
theta = vq.derivatives.bs_theta(100, 100, 0.05, 0.20, 1.0)
# Returns: ~-0.03
```

**Interpretation:**
- Negative for calls (seller gains daily)
- Positive for puts (seller loses daily)
- Accelerates near expiration

#### `bs_rho(S, K, r, sigma, T) → float`

Interest rate sensitivity.

```python
rho = vq.derivatives.bs_rho(100, 100, 0.05, 0.20, 1.0)
# Returns: ~20.73
```

**Interpretation:**
- Option gains $20.73 per 1% rate increase
- Minor impact except for far-out-of-money options

---

## Risk Module (`vq.risk`)

**Purpose:** Quantifying and managing portfolio risk

### Value-at-Risk (VaR)

**Definition:** Maximum expected loss at given confidence level.

```
95% VaR = 2% means:
- 95% of the time: loss < 2%
- 5% of the time: loss > 2%
```

#### `parametric_var(returns, confidence_level) → float`

VaR assuming normal distribution.

```python
daily_returns = [0.001, 0.002, -0.001, ...]
var_95 = vq.risk.parametric_var(daily_returns, confidence_level=0.95)
# Returns: 0.032 (3.2%)
```

**Pros:**
- Fast (one formula)
- Works with limited data

**Cons:**
- Assumes normal (not realistic for stocks)
- Underestimates tail risk

#### `historical_var(returns, confidence_level) → float`

VaR from actual data distribution.

```python
var_95 = vq.risk.historical_var(daily_returns, confidence_level=0.95)
# Returns: empirical percentile
```

**Pros:**
- No distribution assumption
- Reflects reality better

**Cons:**
- Needs more data
- Can't extrapolate beyond data

### Conditional Value-at-Risk (CVaR)

#### `cvar(returns, confidence_level) → float`

Expected loss **when worse than VaR**.

```python
returns = [0.001, 0.002, -0.001, -0.03, ...]
var_95 = vq.risk.parametric_var(returns, 0.95)    # 3.2%
cvar_95 = vq.risk.cvar(returns, 0.95)              # 4.5%
```

**Interpretation:**
- VaR: Threshold of bad days
- CVaR: Average of bad days

**Better for risk management:** CVaR > VaR shows tail risk

---

## Stochastic Module (`vq.stochastic`)

**Purpose:** Simulating uncertain future scenarios

### Monte Carlo Architecture

**Process:**
```
1. Generate random numbers (seeded)
2. Simulate individual paths
3. Aggregate to get distribution
4. Estimate statistics
```

### Safe Configuration

**Always use:**

```python
from vectorquant.core.mc_config import get_safe_test_params

params = get_safe_test_params()
# {'n_paths': 1000, 'n_steps': 50, 'dt': 0.02}
```

**Never:**

```python
# ❌ Don't do this
n_paths = 1_000_000  # Will hang
n_steps = 10_000
```

### Geometric Brownian Motion

#### `simulate_geometric_brownian_motion(S0, mu, sigma, T, n_paths, dt) → list[list[float]]`

Simulate stock price paths.

```python
paths = vq.stochastic.simulate_geometric_brownian_motion(
    S0=100,           # Start at 100
    mu=0.08,          # 8% expected return
    sigma=0.20,       # 20% volatility
    T=1.0,            # 1 year
    n_paths=1000,
    dt=0.02           # 0.02 years = 5 days
)
# Returns: list of 1000 paths
# Each path: list of ~50 prices
```

**Formula:** $dS = \mu S dt + \sigma S dW$

**Interpretation:**
- Drift ($\mu dt$): Expected return
- Diffusion ($\sigma dW$): Random shock

### Monte Carlo Valuation

#### `MonteCarloEngine.european_call(S0, K, r, sigma, T, n_paths) → float`

Option price via simulation.

```python
price = vq.stochastic.MonteCarloEngine.european_call(
    S0=100, K=100, r=0.05, sigma=0.20, T=1.0,
    n_paths=10000
)
# Returns: ~10.45 (matches Black-Scholes)
```

**Process:**
1. Simulate 10,000 final prices
2. Compute payoff: max(S_T - K, 0)
3. Average: E[payoff] = price

**Pros:**
- Flexible (works for any payoff)
- Intuitive

**Cons:**
- Noisy (variance decreases as 1/√n)
- Slower than closed-form

---

## AI Verification Module (`vq.ai`)

**Purpose:** Detect hallucinations and verify computations

### Verification System

#### `verify_calculation(expression, expected, tolerance) → VerificationResult`

Check if computation is correct.

```python
result = vq.ai.verify_calculation(
    expression="mean([1, 2, 3])",
    expected=2.0,
    tolerance=1e-10
)

print(result.verified)      # True
print(result.confidence)    # 1.0
print(result.computed_value) # 2.0 (actual)
```

**Workflow:**
1. Parse expression
2. Evaluate in VectorQuant
3. Compare with expected
4. Return confidence score

### Explanation System

#### `explain_sharpe(returns, risk_free_rate) → ExplanationTrace`

Step-by-step computation breakdown.

```python
trace = vq.ai.explain_sharpe([0.01, 0.02], risk_free_rate=0.02)

for step in trace.steps:
    print(f"{step['step']}: {step['value']}")

# Output:
# Step 1 - mean: 0.015
# Step 2 - std: 0.007071
# Step 3 - sharpe: 1.978
```

**Uses:**
- Debugging (what went wrong?)
- Verification (see the work)
- Education (learn the formula)

### Proof Verification

#### `HallucinationProofPipeline.process(intent, **params) → PipelineResult`

Full verification pipeline.

```python
pipeline = vq.ai.HallucinationProofPipeline()
result = pipeline.process(
    intent="sharpe",
    returns=[0.01, 0.02, -0.01],
    risk_free_rate=0.02
)

print(f"Result: {result.result}")           # Computed value
print(f"Verified: {result.verified}")       # Correct?
print(f"Confidence: {result.confidence}")   # How sure (0-1)?
```

---

## Core Module (`vq.core`)

**Purpose:** Backend management and utilities

### Backend Management

#### `get_backend() → str`

Check active computation engine.

```python
backend = vq.core.get_backend()
if backend == "C":
    print("✓ Fast (50-200x speedup)")
else:
    print("⚠ Fallback (slower)")
```

### Random Number Generation

#### `create_rng(seed: int) → RNG`

Seeded random number generator.

```python
rng = vq.core.create_rng(seed=42)
seq1 = [rng.next() for _ in range(5)]

rng = vq.core.create_rng(seed=42)
seq2 = [rng.next() for _ in range(5)]

seq1 == seq2  # True (deterministic)
```

### Optimization

#### `gradient_descent(f, grad, x0, lr, max_iter) → list[float]`

Minimize objective function.

```python
def objective(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return [2*x[0], 2*x[1]]

x_optimal = vq.core.gradient_descent(
    f=objective,
    grad=gradient,
    x0=[5.0, -3.0],
    lr=0.1,
    max_iter=100
)
# Returns: [0, 0] (minimum)
```

---

## Module Dependencies

```
All modules depend on:
    core/ (backend, RNG)
        ↓
    stats/ ← Used by portfolio, risk
        ↓
    portfolio/ ← Uses stats
      ↑ ← Also: derivatives
      
    derivatives/ (independent)
    
    stochastic/ ← Uses core.rng
      ↑ ← Can use in risk
      
    risk/ (independent)
    
    ai/ ← Uses all above for verification
```

---

## Module Selection Guide

**For portfolio work:** Use `stats` + `portfolio`
**For option traders:** Use `derivatives`
**For risk managers:** Use `risk` + `stats`
**For simulation:** Use `stochastic`
**For testing models:** Use `ai` verification

**All modules together:** Full quant analytics platform
