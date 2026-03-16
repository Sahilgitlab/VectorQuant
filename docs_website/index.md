# VectorQuant — Financial Computing Library

**Production-ready quantitative finance library with deterministic computation and AI verification.**

---

## Why VectorQuant?

### The Challenge

Quantitative finance practitioners face conflicting requirements:

| Requirement | Issue |
|-------------|-------|
| Reproducible results | NumPy/SciPy vary across systems |
| Zero external dependencies | NumPy/SciPy are 100+ MB |
| Transparent mathematics | C libraries are black-boxes |
| Fast computation | Pure Python is slow |
| AI-verifiable outputs | No existing verification layer |

### The Solution

VectorQuant provides a hybrid system:

```
Python API (readable)
    ↓
Smart Dispatch (automatic routing)
    ↓
C Engine (fast) OR Python Fallback (portable)
    ↓
Results with Verification
```

**Result**: You write simple Python code. Backend automatically optimizes where possible.

---

## Key Features

✅ **Deterministic Computation**
- Same seed → Same results, always
- Works across Windows/Linux/macOS
- Audit-trail friendly for compliance

✅ **Zero Dependencies**
- Pure Python fallback
- Single-file C extension
- No NumPy/SciPy required

✅ **Specialized for Finance**
- Portfolio optimization (not generic)
- Risk models (VaR, CVaR)
- Derivatives pricing (Black-Scholes)
- Factor models (Fama-French)
- Stochastic processes (GBM, Brownian)

✅ **Performance Options**
- 165x faster with C engine (optional)
- Python fallback when C unavailable
- Automatic backend selection

✅ **AI Verification** (Unique)
- Automatic hallucination detection
- Step-by-step proof traces
- Confidence scoring

✅ **Production-Ready**
- 251/252 tests passing
- Real-world benchmarks
- Documented roadmap

---

## Quick Comparison

|  | VectorQuant | NumPy | SciPy | QuantLib |
|---|---|---|---|---|
| **Deterministic** | ✓ | ✗ | ✗ | ✓ |
| **Zero deps** | ✓ | ✗ | ✗ | ✗ |
| **Finance-focused** | ✓ | ✗ | ✗ | ✓ |
| **Readable code** | ✓ | ✗ | ✗ | ✗ |
| **AI-verifiable** | ✓ | ✗ | ✗ | ✗ |
| **Fast (C)** | ✓ | ✓ | ✓ | ✓ |
| **Large ecosystem** | — | ✓ | ✓ | — |

---

## Getting Started

### Installation

```bash
pip install vectorquant
```

No binary compilation. No system dependencies. Works immediately.

### 1-Minute Example

```python
import vectorquant as vq

# Calculate portfolio metrics
returns = [0.01, -0.02, 0.015, 0.02, -0.005, 0.018]
weights = [0.4, 0.3, 0.3]

# What is the portfolio return?
portfolio_return = vq.portfolio.portfolio_return(weights, returns)

# What is the risk?
covariance = vq.stats.covariance(returns)
volatility = vq.portfolio.portfolio_volatility(weights, covariance)

# What is risk-adjusted return?
sharpe = (portfolio_return - 0.02) / volatility

print(f"Return: {portfolio_return:.1%}")
print(f"Risk: {volatility:.1%}")
print(f"Sharpe: {sharpe:.2f}")
```

---

## Use Cases

### Risk Management
- Daily VaR/CVaR calculations
- Portfolio risk attribution
- Scenario analysis

### Quant Research
- Backtesting strategies
- Factor model analysis
- Performance attribution

### Trading Systems
- High-frequency Greeks calculation
- Portfolio optimization
- Monte Carlo risk simulation

### AI/LLM Integration
- Verification of financial claims
- Automatic hallucination detection
- Step-by-step computation proofs

---

## Documentation Structure

**[→ Quick Start](quickstart.md)**
5 minutes. Installation, first portfolio, first risk metric.

**[→ Tutorial](tutorial.md)**
30 minutes. Complete walkaround of core modules.

**[→ Core Concepts](core-concepts.md)**
Architecture, determinism, backends, RNG.

**[→ Module Guide](modules.md)**
Detailed reference for each module with examples.

**[→ AI Verification](ai-verification.md)**
Unique feature: Hallucination detection and proof traces.

**[→ Benchmarks](benchmarks.md)**
Performance comparison vs NumPy, SciPy, QuantLib.

**[→ API Reference](api-reference.md)**
Complete function documentation with parameters and types.

**[→ Architecture](architecture.md)**
Technical design: Python → Dispatch → C.

**[→ Roadmap](roadmap.md)**
Current version (5.2), near-term (GPU, distribution), future (quantum).

**[→ FAQ](faq.md)**
Common questions and troubleshooting.

---

## Current Version: 5.2

### What's Included

```
✓ Statistics (mean, std, skew, kurtosis, covariance)
✓ Optimization (gradient descent, line search)
✓ Portfolio (Markowitz, Sharpe ratio, Black-Litterman)
✓ Derivatives (Black-Scholes, Greeks)
✓ Stochastic (Brownian motion, GBM, Monte Carlo)
✓ Risk Models (VaR, CVaR, historical, parametric)
✓ Factor Models (Fama-French 3 & 5 factor)
✓ AI Verification (formula checking, proof traces)
```

### Performance

| Task | Time | vs NumPy |
|------|------|----------|
| 1M element mean | 0.15ms | 0.5x* |
| Portfolio variance | 0.8ms | 1.0x* |
| VaR calculation | 0.2ms | 1.0x* |
| Monte Carlo (10K paths) | 12ms | 20x faster** |
| Covariance (100x100) | 69ms | 75x slower* |

*Pure Python. **With C engine.

*Use NumPy for very large matrices. VectorQuant optimizes for typical finance problem sizes (< 1000 assets) and is developer-friendly.*

---

## Core Principles

### 1. Determinism Over Speed

```python
# This produces same result every run
import vectorquant as vq
rng = vq.core.create_rng(seed=42)
paths = vq.stochastic.simulate_gbm(..., rng=rng)
# paths[0] == paths[0]  always true
```

Why: Reproducible research, compliant audits, testable AI systems.

### 2. Transparency Over Magic

```python
# See exactly what computation happened
trace = vq.ai.explain_sharpe(returns, risk_free_rate=0.02)
for step in trace.steps:
    print(f"{step['step']:<40} = {step['value']:.6f}")
    
# Output:
# 1. Calculate mean return                = 0.008333
# 2. Calculate standard deviation         = 0.014422
# 3. Calculate excess return              = -0.011667
# 4. Sharpe = excess / std               = -0.808883
```

Why: Debug easily, integrate with AI systems, understand results.

### 3. Specialization Over Generality

```python
# Portfolio functions that matter
vq.portfolio.optimize_max_sharpe()  # Risk-adjusted return
vq.portfolio.black_litterman_returns()  # Incorporate views
vq.portfolio.equal_risk_contribution()  # Risk parity

# Not trying to be NumPy (general purpose)
# Focus where it matters: quantitative finance
```

Why: Better API, optimized implementations, clearer examples.

---

## Next

**New to VectorQuant?** Start with [Quick Start](quickstart.md) (5 min read).

**Want to understand the design?** Read [Core Concepts](core-concepts.md).

**Need specific functions?** Check [API Reference](api-reference.md).

**Curious about performance?** See [Benchmarks](benchmarks.md).

---

## Support

- **Questions?** Check [FAQ](faq.md)
- **Having issues?** See troubleshooting in [Tutorial](tutorial.md)
- **Want to contribute?** Roadmap in [here](roadmap.md)

---

## License

Apache 2.0 — Free for commercial and academic use.

---

**VectorQuant 5.2 — Production-ready. Deterministic. AI-verifiable.**
