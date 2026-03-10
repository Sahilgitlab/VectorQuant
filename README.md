# VectorQuant

**A deterministic reasoning engine for AI and quantitative finance.**

*Created by Sahil Gupta*

Zero-dependency · Pure Python · Numba JIT · GPU Accelerated · LLM-Ready

---

## Why VectorQuant Exists

LLMs hallucinate numbers.
Finance libraries are fragmented.
Scientific stacks are heavy.

VectorQuant solves this by combining **deterministic math**, **financial models**, and **AI verification tools**. This makes it the perfect ground-truth layer for algorithmic trading and agentic AI.

## Capabilities

- ✓ **Linear algebra & statistics**
- ✓ **Monte Carlo simulation**
- ✓ **Portfolio optimization**
- ✓ **Risk models (VaR, CVaR)**
- ✓ **Derivatives pricing**
- ✓ **Event-driven backtesting**
- ✓ **AI hallucination detection**
- ✓ **LLM tool integration**

## Real Use Cases

- **AI systems** that must avoid hallucinated math.
- **Quant research pipelines** looking for a zero-dependency foundation.
- **Algorithmic trading research** and event-driven backtesting.
- **Financial education tools** that need clean, readable math.
- **Monte Carlo simulation engines** requiring massive GPU scale.

---

## What is VectorQuant?

Imagine you're building a calculator, but not a regular one — a calculator that can:

- **Price stock options** using the same math Wall Street banks use
- **Simulate thousands of possible futures** for a stock price
- **Build the best possible investment portfolio** from a group of stocks
- **Tell an AI system "your math is wrong"** when it hallucinates

That's what VectorQuant does. It's a **complete quantitative finance library** written entirely in pure Python (no NumPy, no Pandas, no SciPy required). You install it, and it just works — on any machine, anywhere.

### Why Would You Use This?

| Problem                                     | How VectorQuant Solves It                                                   |
| ------------------------------------------- | --------------------------------------------------------------------------- |
| "I need to calculate risk for my portfolio" | `vq.risk.parametric_var(returns, 0.95)` — one line                       |
| "I want to simulate stock prices"           | `vq.stochastic.simulate_geometric_brownian_motion(...)`                   |
| "I need to price an option"                 | `vq.derivatives.black_scholes_call(S, K, r, sigma, T)`                    |
| "My AI is making up numbers"                | `vq.ai.check_formula("sharpe_ratio", "mu/variance")` → `HALLUCINATION` |
| "I want it to run faster"                   | `pip install vectorquant[perf]` → 15x instant speedup                    |
| "I want GPU speed"                          | `pip install vectorquant[gpu]` → 200x+ speedup                           |

---

## Installation

### Basic (Zero Dependencies)

```bash
pip install vectorquant
```

This gives you **everything**. No NumPy, no SciPy, no C compiler needed. Works on Windows, Mac, Linux, even Raspberry Pi.

### With Numba JIT Acceleration (15x Faster)

```bash
pip install vectorquant[perf]
```

This installs [Numba](https://numba.pydata.org/), which automatically compiles your hot loops to machine code. You don't change any code — it just gets faster.

### With GPU Acceleration (200x+ Faster)

```bash
pip install vectorquant[gpu]
```

This installs [CuPy](https://cupy.dev/) for NVIDIA GPU acceleration. Monte Carlo simulations go from 6,500 paths/sec to **1,500,000+ paths/sec**.

### For Development

```bash
pip install vectorquant[dev]
```

---

## Quick Start (Your First 5 Minutes)

```python
import vectorquant as vq

# 1. Basic statistics
data = [0.01, -0.02, 0.015, -0.005, 0.008, 0.012, -0.01]
print("Mean:", vq.stats.mean(data))           # 0.0014...
print("Std Dev:", vq.stats.standard_deviation(data))

# 2. Calculate risk
var = vq.risk.parametric_var(data, confidence_level=0.95)
print(f"95% VaR: {var:.4f}")    # If you lose money, it won't be worse than this 95% of the time

# 3. Price an option
call_price = vq.derivatives.black_scholes_call(S=100, K=105, r=0.05, sigma=0.2, T=1.0)
print(f"Call Option Price: ${call_price:.2f}")

# 4. Simulate stock prices
paths = vq.stochastic.simulate_geometric_brownian_motion(
    S0=100, mu=0.05, sigma=0.2, T=1.0, dt=1/252, n_paths=5
)
for i, path in enumerate(paths):
    print(f"  Path {i+1}: ${path[0]:.0f} → ${path[-1]:.2f}")

# 5. Prevent AI Hallucinations
llm = vq.ai.LLMInterface()
result = llm.execute("calculate_var", returns=data, confidence_level=0.95)
print(f"Verified: {result['verified']}")
# Verified: True
# Proof Trace: mean → std → z-score → VaR
```

---

## Architecture

```text
                ┌──────────────┐
                │      AI      │
                │ verification │
                └───────▲──────┘
                        │
                ┌───────┴──────┐
                │   research   │
                │ backtesting  │
                └───────▲──────┘
                        │
                ┌───────┴──────┐
                │    finance   │
                │ models/risk  │
                └───────▲──────┘
                        │
                ┌───────┴──────┐
                │  stochastic  │
                │ simulation   │
                └───────▲──────┘
                        │
                ┌───────┴──────┐
                │     core     │
                │ math engine  │
                └──────────────┘
```

VectorQuant is designed in strict vertical layers. Upper layers depend on lower layers, ensuring the core math engine remains pure and fast.

---

## Module-by-Module Guide

### `vq.linalg` — Linear Algebra

The math foundation. Matrix operations, decompositions, and solvers.

```python
import vectorquant as vq

# Matrix multiply
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
result = vq.linalg.matrix_multiply(A, B)
# [[19, 22], [43, 50]]

# Determinant
det = vq.linalg.determinant(A)
# -2.0

# Transpose
T = vq.linalg.transpose(A)
# [[1, 3], [2, 4]]

# Matrix inverse
inv = vq.linalg.invert(A)

# Singular Value Decomposition
U, S, Vt = vq.linalg.svd(A)

# Dot product
result = vq.linalg.dot([1, 2, 3], [4, 5, 6])
# 32
```

---

### `vq.stats` — Statistics

Everything from basic descriptive stats to multivariate regression.

```python
import vectorquant as vq

data = [0.01, -0.02, 0.015, -0.005, 0.008, 0.012, -0.01]

# Descriptive statistics
vq.stats.mean(data)                # Average
vq.stats.variance(data)            # How spread out
vq.stats.standard_deviation(data)  # Square root of variance
vq.stats.skewness(data)            # Is it lopsided?
vq.stats.kurtosis(data)            # Fat tails?

# Correlation and covariance
returns_matrix = [
    [0.01, 0.005],
    [0.02, -0.01],
    [-0.005, 0.015],
]
cov = vq.stats.covariance_matrix(returns_matrix)

# Linear regression (y = a + bx)
x = [1, 2, 3, 4, 5]
y = [2.1, 3.9, 6.2, 7.8, 10.1]
slope, intercept = vq.stats.linear_regression(x, y)
print(f"y = {slope:.2f}x + {intercept:.2f}")
```

---

### `vq.prob` — Probability

Probability distributions, random number generation, and inverse CDF.

```python
import vectorquant as vq

# Normal distribution
pdf_val = vq.prob.normal_pdf(0, mu=0, sigma=1)     # ≈ 0.3989
cdf_val = vq.prob.normal_cdf(1.96, mu=0, sigma=1)  # ≈ 0.975

# Inverse CDF (for VaR calculations)
z = vq.prob.normal_inv_cdf(0.95)  # ≈ 1.645

# Random number generation
vq.prob.set_seed(42)              # Reproducible results
rand_normal = vq.prob.rnorm()     # Random normal draw
rand_uniform = vq.prob.runif()    # Random uniform [0,1)

# Other distributions
vq.prob.lognormal_pdf(1.0, mu=0, sigma=1)
vq.prob.exponential_pdf(1.0, lmbda=2.0)
vq.prob.poisson_pmf(3, lmbda=2.5)
```

---

### `vq.stochastic` — Stochastic Simulation

Simulate random processes used in finance: stock prices, interest rates, volatility.

```python
import vectorquant as vq

# Geometric Brownian Motion (stock prices)
# "Simulate 1000 possible stock price paths for 1 year"
paths = vq.stochastic.simulate_geometric_brownian_motion(
    S0=100,      # Starting price: $100
    mu=0.08,     # Expected 8% annual return
    sigma=0.2,   # 20% volatility
    T=1.0,       # 1 year
    dt=1/252,    # Daily steps
    n_paths=1000
)
# Each path is a list of 253 prices (252 trading days + start)

# Heston Model (stochastic volatility)
s_paths, v_paths = vq.stochastic.simulate_heston(
    S0=100, v0=0.04, mu=0.05, kappa=2.0,
    theta=0.04, sigma_v=0.3, rho=-0.7,
    T=1.0, dt=1/252, n_paths=100
)

# Vasicek Model (interest rates)
rate_paths = vq.stochastic.simulate_vasicek_model(
    r0=0.03, a=0.5, b=0.05, sigma=0.01,
    T=1.0, dt=1/12, n_paths=100
)

# Monte Carlo Options Pricing
engine = vq.stochastic.MonteCarloEngine(n_paths=50000)
price, std_error = engine.european_call(S0=100, K=105, r=0.05, sigma=0.2, T=1.0)
print(f"European Call: ${price:.4f} ± ${std_error:.4f}")

asian_price, se = engine.asian_call(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, dt=1/252)
print(f"Asian Call: ${asian_price:.4f}")
```

---

### `vq.timeseries` — Time Series Analysis

Moving averages, volatility estimation, and regime detection.

```python
import vectorquant as vq

prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]

# Moving Averages
sma_5 = vq.timeseries.sma(prices, n=5)   # Simple Moving Average (5-period)
ema_5 = vq.timeseries.ema(prices, n=5)   # Exponential Moving Average
wma_5 = vq.timeseries.wma(prices, n=5)   # Weighted Moving Average

# Volatility
returns = [0.02, -0.01, 0.02, 0.02, -0.01, 0.02, 0.02, -0.01, 0.02]
rolling_vol = vq.timeseries.rolling_volatility(returns, n=5)
ewma_vol = vq.timeseries.ewma_volatility(returns, lmbda=0.94)

# AR(1) Model
intercept, phi = vq.timeseries.ar_1_model(prices)
print(f"AR(1): price_t = {intercept:.2f} + {phi:.4f} * price_(t-1)")

# Hidden Markov Model (Regime Detection)
# Detect bull/bear market regimes from observed data
```

---

### `vq.portfolio` — Portfolio Optimization

Build the best portfolio from a set of assets.

```python
import vectorquant as vq

# Expected returns for 3 assets
expected_returns = [0.08, 0.12, 0.06]  # 8%, 12%, 6%

# Covariance matrix
cov_matrix = [
    [0.04, 0.006, 0.002],
    [0.006, 0.09, 0.004],
    [0.002, 0.004, 0.01],
]

# Find the optimal weights (maximize Sharpe Ratio)
weights = vq.portfolio.optimize_max_sharpe(expected_returns, cov_matrix, risk_free_rate=0.02)
print(f"Optimal Weights: {[f'{w:.1%}' for w in weights]}")

# Calculate portfolio return and risk
import math
port_ret = vq.portfolio.portfolio_return(weights, expected_returns)
port_var = vq.portfolio.portfolio_variance(weights, cov_matrix)
print(f"Expected Return: {port_ret:.2%}")
print(f"Portfolio Volatility: {math.sqrt(port_var):.2%}")
```

---

### `vq.risk` — Risk Models

Measure how much money you could lose.

```python
import vectorquant as vq

daily_returns = [0.01, -0.02, 0.015, -0.005, 0.008, -0.03, 0.02, -0.01, 0.005, -0.015]

# Parametric VaR (assumes normal distribution)
# "With 95% confidence, your worst daily loss won't exceed this"
var_95 = vq.risk.parametric_var(daily_returns, confidence_level=0.95)
print(f"95% VaR: {var_95:.4f}")

# Historical VaR (directly from past data)
hist_var = vq.risk.historical_var(daily_returns, confidence_level=0.95)

# CVaR (Expected Shortfall)
# "If you DO lose more than VaR, how bad is it on average?"
cvar_95 = vq.risk.cvar(daily_returns, confidence_level=0.95)
print(f"95% CVaR: {cvar_95:.4f}")
```

---

### `vq.derivatives` — Options Pricing & Greeks

Price options and compute risk sensitivities using Black-Scholes.

```python
import vectorquant as vq

S = 100    # Stock price
K = 105    # Strike price
r = 0.05   # Risk-free rate (5%)
sigma = 0.2  # Volatility (20%)
T = 1.0    # Time to expiry (1 year)

# Option prices
call = vq.derivatives.black_scholes_call(S, K, r, sigma, T)
put = vq.derivatives.black_scholes_put(S, K, r, sigma, T)
print(f"Call: ${call:.4f}  |  Put: ${put:.4f}")

# The Greeks (how the price changes when inputs change)
delta = vq.derivatives.bs_delta(S, K, r, sigma, T)     # Price sensitivity
gamma = vq.derivatives.bs_gamma(S, K, r, sigma, T)     # Delta sensitivity
theta = vq.derivatives.bs_theta(S, K, r, sigma, T)     # Time decay
vega = vq.derivatives.bs_vega(S, K, r, sigma, T)       # Volatility sensitivity
rho = vq.derivatives.bs_rho(S, K, r, sigma, T)         # Interest rate sensitivity

print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Theta: {theta:.4f}")
print(f"Vega:  {vega:.4f}")
print(f"Rho:   {rho:.4f}")
```

---

### `vq.research` — Backtesting Framework

Test trading strategies on historical data.

```python
import vectorquant as vq

# Simple rolling-window backtest
def equal_weight_strategy(historical_returns):
    n_assets = len(historical_returns[0])
    return [1.0 / n_assets] * n_assets

returns_matrix = [
    [0.01, 0.005], [0.02, -0.01], [-0.005, 0.015],
    [0.008, 0.003], [0.015, -0.005], [-0.01, 0.02],
    [0.012, 0.008], [0.003, -0.002], [0.01, 0.01],
    [0.005, 0.003], [-0.008, 0.012], [0.02, -0.005],
]

results = vq.research.rolling_window_backtest(
    returns_matrix, equal_weight_strategy, window_size=5, bps_fee=0.001
)
print(f"Out-of-sample returns: {len(results)} periods")
print(f"Average return: {sum(results)/len(results):.6f}")

# Probabilistic Sharpe Ratio (is your Sharpe statistically significant?)
psr = vq.research.probabilistic_sharpe_ratio(results, benchmark_sharpe=0.0)
print(f"PSR: {psr:.2%}")
```

---

### `vq.ai` — AI Verification & Reasoning Engine

The most powerful module. Prevents AI hallucinations and provides deterministic reasoning.

```python
import vectorquant as vq

# ── 1. Verify a math expression ──
result = vq.ai.verify_calculation("sqrt(4) * 3", expected=6.0)
print(result)  # VerificationResult(VERIFIED, confidence=1.0)

# ── 2. Catch a hallucination ──
result = vq.ai.check_formula("sharpe_ratio", "mu / variance")
print(f"Correct: {result.is_correct}")           # False!
print(f"Real formula: {result.correct_formula}")  # (mu - r_f) / sigma

# ── 3. Get step-by-step proof ──
returns = [0.01, -0.02, 0.015, -0.005, 0.008, -0.01, 0.02]
trace = vq.ai.explain_var(returns, confidence=0.95)
for step in trace.steps:
    print(f"  {step['step']} = {step['value']}")

# ── 4. Full reasoning pipeline ──
engine = vq.ai.ReasoningEngine()
answer = engine.solve("sharpe", returns=returns)
print(f"Result: {answer.result}, Verified: {answer.verified}")

# ── 5. LLM Tool Interface ──
schemas = vq.ai.get_tool_schemas()  # OpenAI function-calling format
llm = vq.ai.LLMInterface()
result = llm.execute("calculate_var", returns=returns, confidence_level=0.95)
print(f"VaR: {result['value']:.4f}, Verified: {result['verified']}")

# ── 6. Hallucination-proof pipeline ──
pipeline = vq.ai.HallucinationProofPipeline()
result = pipeline.process("var", returns=returns, confidence_level=0.95)
print(result)  # PipelineResult(VERIFIED, confidence=100%)

# ── 7. Validate a prediction with Monte Carlo ──
result = vq.ai.validate_prediction(
    hypothesis="Stock reaches 200 from 100 in 1 year",
    S0=100, mu=0.05, sigma=0.2, T=1.0, target_price=200,
    n_simulations=10000
)
print(f"Probability: {result.computed_value:.2%}")
```

---

### `vq.infra` — Infrastructure

Data cleaning and parallel computing.

```python
import vectorquant as vq

# Forward-fill missing data (None values)
dirty_data = [1.0, None, None, 4.0, None, 6.0]
clean = vq.infra.forward_fill_missing(dirty_data)
# [1.0, 1.0, 1.0, 4.0, 4.0, 6.0]
```

---

## Making It Faster

VectorQuant has **three speed tiers** — you choose based on your needs:

### Tier 1: Pure Python (Default)

```bash
pip install vectorquant
```

- **Speed**: ~6,500 Monte Carlo paths/sec
- **Dependencies**: Zero
- **Best for**: Learning, prototyping, any machine

### Tier 2: Numba JIT (`[perf]`)

```bash
pip install vectorquant[perf]
```

- **Speed**: ~97,500 paths/sec (**15x faster**)
- **Dependencies**: numba
- **Best for**: Production research, serious backtesting
- **How it works**: The `@njit_fallback` decorator automatically JIT-compiles stochastic functions. You change zero code.

### Tier 3: GPU (`[gpu]`)

```bash
pip install vectorquant[gpu]
```

- **Speed**: ~1,500,000+ paths/sec (**200x+ faster**)
- **Dependencies**: cupy (requires NVIDIA GPU)
- **Best for**: Institutional-scale Monte Carlo, massive option grids
- **How it works**: Pass `gpu=True` to the Monte Carlo engine:

```python
engine = vq.stochastic.MonteCarloEngine(n_paths=1_000_000, gpu=True)
price, se = engine.european_call(S0=100, K=105, r=0.05, sigma=0.2, T=1.0)
```

### Monte Carlo Simulation Speed

```text
Pure Python    ███                 6,500 paths/sec
Numba JIT      █████████████      97,500 paths/sec
GPU (CuPy)     █████████████████████████████████ 1,500,000+ paths/sec
```

---

## Running the Examples

VectorQuant includes ready-to-run example scripts:

```bash
# Monte Carlo simulation and options pricing
python examples/01_monte_carlo.py

# Portfolio optimization with risk analysis
python examples/02_portfolio_optimization.py

# AI hallucination prevention demo
python examples/03_llm_verification.py
```

---

## Why VectorQuant?

### Advantages Over Existing Libraries

| Feature                     | VectorQuant | NumPy/SciPy | QuantLib |
| --------------------------- | :---------: | :---------: | :------: |
| Zero dependencies           |     ✅     |     ❌     |    ❌    |
| Pure Python (no C compiler) |     ✅     |     ❌     |    ❌    |
| AI hallucination detection  |     ✅     |     ❌     |    ❌    |
| LLM tool interface          |     ✅     |     ❌     |    ❌    |
| Proof traces for AI         |     ✅     |     ❌     |    ❌    |
| GPU acceleration            |     ✅     |     ❌     |    ❌    |
| Event-driven backtesting    |     ✅     |     ❌     |    ❌    |
| Fama-French factor models   |     ✅     |     ❌     |    ❌    |
| Works on any machine        |     ✅     |    ⚠️    |    ❌    |

### Key Advantages

1. **Zero Dependencies** — Install on any machine instantly. No build tools, no C compiler, no conda.
2. **Three Speed Tiers** — Start with pure Python, then add Numba (15x) or GPU (200x+) when you need speed. Same API, same code.
3. **AI-Native** — The only quant library with built-in hallucination detection, proof traces, and LLM tool schemas. Your AI never guesses a number again.
4. **Complete Stack** — From linear algebra to portfolio optimization to event-driven backtesting in one library. No gluing 10 packages together.
5. **Readable Code** — Every function is written in clear Python you can read, understand, and modify. No black boxes.
6. **Institutional Architecture** — Layered design (core → stochastic → finance → research → ai) mirrors how real hedge funds organize their code.

---

## Project Structure

```
VectorQuant/
├── vectorquant/          # Main library
│   ├── core/             # Mathematical kernel
│   ├── stochastic/       # Simulation engines
│   ├── time_series/      # Signal processing
│   ├── finance/          # Financial modeling
│   ├── research/         # Backtesting framework
│   ├── ai/               # AI reasoning engine
│   └── infrastructure/   # Engineering tools
├── tests/                # 89+ automated tests
├── benchmarks/           # Performance benchmarks
├── examples/             # Ready-to-run scripts
├── pyproject.toml        # Package configuration
└── README.md             # This file
```

---

## Contributing

VectorQuant is open to contributions. Areas where help is deeply appreciated:

- Adding new financial models (e.g., fixed income, exotic options).
- Expanding the AI reasoning tools and proof traces.
- GPU acceleration improvements and kernel optimizations.
- Documentation, tutorials, and real-world examples.

Feel free to open an issue or submit a Pull Request!

---

## Author

**Sahil Gupta**

- **Email:** [linkedin.sahil.gupta07@gmail.com](mailto:linkedin.sahil.gupta07@gmail.com)
- **LinkedIn:** [https://www.linkedin.com/in/sahilg007/](https://www.linkedin.com/in/sahilg007/)

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Sahil Gupta. Use it however you want.
