# FAQ

Frequently asked questions about VectorQuant.

---

## Installation & Setup

### Q: How do I install VectorQuant?

A: Use pip:

```bash
pip install vectorquant
```

That's it. Check if C engine is available:

```python
import vectorquant as vq
print(vq.core.get_backend())  # Returns "C" or "Python"
```

---

### Q: Why is my installation using Python backend instead of C?

A: The C engine wasn't compiled on your system. VectorQuant still works (Python backend), but slower.

**Solutions:**

1. **Install with development tools:**
   ```bash
   # macOS
   xcode-select --install
   
   # Ubuntu/Debian
   sudo apt-get install build-essential python3-dev
   
   # Windows (use gcc via MinGW or MSVC)
   ```

2. **Reinstall:**
   ```bash
   pip install --force-reinstall --no-cache-dir vectorquant
   ```

3. **Check compiler:**
   ```bash
   gcc --version        # Should work
   python -m pip install wheel
   ```

If C engine still won't build, Python backend is guaranteed to work (just slower).

---

### Q: Does VectorQuant work on Apple Silicon (M1/M2)?

A: Yes. VectorQuant detects your CPU architecture and uses appropriate SIMD instructions.

**Performance:** Slightly lower than x86-64 (M1 doesn't have AVX2), but still 30-50x over pure Python.

---

### Q: I'm on Windows. Any special requirements?

A: VectorQuant works on Windows with either:

1. **MSVC (Visual Studio)** — Native compilation
2. **MinGW (GCC)** — Open-source compilation
3. **WSL2 (Linux subsystem)** — Linux environment in Windows

All three work equally well.

---

## Performance Questions

### Q: How much faster is VectorQuant than NumPy?

A: Depends on the operation:

| Operation | Against NumPy | Note |
|-----------|--------------|------|
| Basic math | 1-2x | Not much faster (Python translation overhead) |
| Covariance | 50x | NumPy uses system BLAS (varies) |
| Optimization | 170x | L-BFGS is well-optimized |
| Monte Carlo | 170x | Parallel + tight loops |

**Average:** 50-100x for practical workflows.

See [benchmarks documentation](benchmarks.md) for detailed comparison.

---

### Q: My computation is hanging. What do I do?

A: Likely issue: Unsafe Monte Carlo parameters. VectorQuant runs **safe defaults** by default:

```python
from vectorquant.core.mc_config import get_safe_test_params
params = get_safe_test_params()
# n_paths: 1000, n_steps: 50, dt: 0.02
```

**If you need more simulations:**

```python
# Increase gradually
paths = vq.stochastic.simulate_gbm(
    S0=100, mu=0.05, sigma=0.20, T=1.0,
    n_paths=10000,    # Safe to use
    dt=1/252          # Daily timesteps
)
# ~5 seconds on modern CPU
```

**Hanging usually means:**
- Too many paths (>1M without GPU)
- Too many timesteps (>10K)
- Old computer (< 2GB RAM)

See [Monte Carlo troubleshooting](#monte-carlo-hanging).

---

### Q: Is VectorQuant slower than QuantLib?

A: Similar speed, with different strengths:

| Metric | VectorQuant | QuantLib |
|--------|------------|----------|
| Speed | 50-200x over Python | Native C++, similar |
| Ease of use | Simple Python API | C++ complexity |
| Determinism | Guaranteed | Depends on BLAS |
| Hallucination detection | ✓ | ✗ |

Choose VectorQuant if you want ease + determinism. Choose QuantLib for established enterprise library.

---

### Q: Can VectorQuant use my GPU?

A: **Not yet** (v5.2). GPU acceleration is planned for Q3 2025.

**Workaround:** Use CPU with safe parameters:

```python
# This runs fast enough on CPU (5 sec for 1M paths)
from vectorquant.core.mc_config import get_safe_test_params
params = get_safe_test_params()
vq.stochastic.simulate_gbm(..., n_paths=1_000_000, dt=params['dt'])
```

---

## Functionality Questions

### Q: Does VectorQuant support constraint optimization?

A: Yes, but limited in v5.2:

```python
# Linear portfolio constraints
constraints = [
    {"type": "eq", "fun": lambda w: sum(w) - 1},  # Sum to 1
    {"type": "ineq", "fun": lambda w: w[i] - 0.05}  # Min 5% per asset
]
```

**Full constraint support:** Planned for v6.0

---

### Q: Can I use VectorQuant for real options?

A: Partially:

✅ **Supported:**
- Black-Scholes European options
- Monte Carlo American option approximation

❌ **Not supported:**
- Exotic derivatives (Asian, Barrier, Rainbow)
- Stochastic interest rates
- Jump diffusion models

**Exotic support:** Under evaluation for v6.0

---

### Q: How do I integrate VectorQuant with my trading system?

A: VectorQuant is a **computation library**, not a trading system:

✅ **VectorQuant does:**
```python
# Compute portfolio optimal weights
weights = vq.portfolio.optimize_max_sharpe(returns, risk_free_rate)

# Compute option prices and Greeks
price = vq.derivatives.black_scholes_call(...)
delta = vq.derivatives.bs_delta(...)
```

❌ **VectorQuant does NOT:**
- Execute trades
- Connect to exchanges
- Manage capital allocation
- Handle compliance

**Integration pattern:**

```python
# Your trading system
while market_is_open:
    data = fetch_market_data()
    
    # Use VectorQuant for computation
    weights = vq.portfolio.optimize_max_sharpe(data)
    
    # Your system executes
    execute_trades(weights)
```

---

### Q: Can I use VectorQuant with machine learning?

A: Yes:

**Integration:**

```python
# Step 1: VectorQuant computes portfolio metrics
var = vq.risk.parametric_var(returns)
sharpe = vq.portfolio.sharpe_ratio(returns)

# Step 2: Your ML model uses these features
ml_model.fit([var, sharpe, ...], targets)
```

**AI Verification (unique feature):**

```python
# Check if LLM's computation is correct
result = vq.ai.verify_calculation(
    expression="sharpe_ratio * sqrt(252)",
    expected=my_expected_value
)
```

---

## Determinism & Reproducibility

### Q: Why is VectorQuant deterministic?

A: **Problem:** NumPy uses system BLAS, which varies across platforms/architectures.

**Solution:** VectorQuant uses custom C implementation with seeded RNG.

```python
# Same seed always gives same result
rng1 = vq.core.create_rng(seed=42)
rng2 = vq.core.create_rng(seed=42)
# rng1 and rng2 produce identical sequences
```

---

### Q: Does determinism make results less realistic?

A: No. Determinism just means **reproducible**, not **artificial**:

```python
# Setup: Same data, same seed
data = [0.01, 0.02, -0.01, 0.03]
rng = vq.core.create_rng(seed=42)

# First run
paths1 = vq.stochastic.simulate_gbm(..., n_paths=1000)

# Second run (identical)
rng = vq.core.create_rng(seed=42)
paths2 = vq.stochastic.simulate_gbm(..., n_paths=1000)

# paths1 == paths2 (exactly)
```

**Why reproducibility matters:**
- Test results remain consistent
- Debugging becomes easier
- Regulatory compliance (audit trail)
- Sharing analysis with others

---

### Q: When should I use different seeds?

A: Always change seed when you want different simulation:

```python
import time

# Option 1: Time-based (non-reproducible)
seed = int(time.time())

# Option 2: Fixed for testing (reproducible)
seed = 42

# Option 3: Incremental (multiple scenarios)
for scenario in range(100):
    rng = vq.core.create_rng(seed=scenario)
    paths = vq.stochastic.simulate_gbm(...)
```

---

## AI Verification

### Q: How does hallucination detection work?

A: VectorQuant re-computes the result and compares:

```python
# LLM says the answer is X
# VectorQuant verification:
1. Extract the formula/computation
2. Run in our engines
3. Compare with LLM's answer
4. Return confidence score

result = vq.ai.verify_calculation(
    expression="mean_return / std_return",
    expected=1.23
)
# Returns: {verified: True, confidence: 0.95}
```

---

### Q: Can I use this to verify my own code?

A: Yes, exactly. Use it like a unit test:

```python
# Your code computed something
your_result = my_portfolio_return(weights, returns)

# Verify it
verified = vq.ai.verify_calculation(
    expression="sum(w*r for w,r in zip(weights, returns))",
    expected=your_result,
    tolerance=1e-10
)

if not verified.verified:
    print(f"Bug found! Expected {verified.computed_value}, got {your_result}")
```

---

### Q: What about accuracy? Does it catch subtle bugs?

A: VectorQuant catches **computation errors**, not **logic errors**:

✅ **Catches:**
```python
# Wrong formula
sharpe = mean / var  # Should be mean / std
verified = vq.ai.verify_calculation(...)  # Detects this
```

✅ **Catches:**
```python
# Wrong parameters
result = vq.derivatives.black_scholes_call(S=100, K=100, r=0.05, sigma=0.20, T=-1)
# Negative time caught as error
```

❌ **Doesn't catch:**
```python
# Logic error (computation is correct, but why you're computing it is wrong)
weights = [0.5, 0.5]  # Verified correct percentages
# But you meant [0.3, 0.7]
```

---

## Comparison Questions

### Q: How does VectorQuant compare to NumPy + SciPy?

A:

| Feature | VectorQuant | NumPy | SciPy |
|---------|------------|-------|-------|
| Speed | 50-200x baseline | 1x (reference) | Varies |
| Determinism | ✓ Guaranteed | ✗ Platform-dependent | ✗ Varies |
| Finance functions | ✓ Built-in | ✗ Must implement | ✓ Some available |
| Dependencies | 0 | 1 (BLAS/LAPACK) | 2+ |
| Hallucination detection | ✓ | ✗ | ✗ |

**Summary:** NumPy/SciPy for general math. VectorQuant for deterministic finance.

---

### Q: Why not just use NumPy for finance?

A: Good question. You *can*, but:

| Issue | NumPy | VectorQuant |
|-------|--------|-------------|
| Determinism across machines | No | Yes |
| Optimization algorithms | Manual | Built-in |
| Finance domain knowledge | No | Yes |
| Hallucination checking | No | Yes |

---

### Q: Does VectorQuant work with pandas DataFrames?

A: Not directly, but simple conversion:

```python
import pandas as pd
import vectorquant as vq

df = pd.read_csv("returns.csv")

# Convert to list[list[float]]
returns = df[["AAPL", "MSFT", "GOOGL"]].values.tolist()

# Use VectorQuant
weights = vq.portfolio.optimize_max_sharpe(returns, risk_free_rate=0.02)

# Convert back
result_df = pd.DataFrame({"asset": df.columns, "weight": weights})
```

---

## Troubleshooting

### Q: I get ImportError: No module named 'vectorquant'

A: Not installed. Do:

```bash
pip install vectorquant
```

If that fails:

```bash
pip install --verbose vectorquant  # See what's happening
```

---

### Q: Results differ between runs

A: You're using different seeds. Try:

```python
# Before your computation
rng = vq.core.create_rng(seed=42)

# Now results are deterministic
```

---

### Q: My optimized portfolio has negative weights

A: Constraints not enforced. Current version (5.2) doesn't enforce bounds.

**Workaround:**

```python
# After optimization, force to non-negative
weights = vq.portfolio.optimize_max_sharpe(returns, risk_free_rate)
weights = [max(0, w) for w in weights]
weights = [w / sum(weights) for w in weights]  # Re-normalize
```

**Fix:** Full constraint support in v6.0

---

### Q: Computing VaR gives unexpected result

A: Check if using enough historical data:

```python
# VaR is statistical - needs sufficient data
# Minimum: 250 observations (1 year of daily data)
# Better: 1000+ observations (4+ years)

if len(returns) < 250:
    print("⚠ Warning: Not enough data for reliable VaR")
```

---

### Q: Why is my Monte Carlo estimate noisy?

A: All Monte Carlo has variance. Increase paths:

```python
# Fewer paths (noisy)
price1 = vq.stochastic.MonteCarloEngine.european_call(..., n_paths=100)

# More paths (less noisy)
price2 = vq.stochastic.MonteCarloEngine.european_call(..., n_paths=10000)

# Variance ~ 1/sqrt(n_paths)
# 100x more paths = 10x less noise
```

---

## Advanced Questions

### Q: Can I extend VectorQuant with custom models?

A: Yes:

```python
# Use VectorQuant as computation library
import vectorquant as vq

def my_custom_model(data):
    # Use VectorQuant functions
    mean = vq.stats.mean(data)
    std = vq.stats.standard_deviation(data)
    
    # Add your logic
    return mean + 2*std

result = my_custom_model([1, 2, 3, 4, 5])
```

---

### Q: How do I contribute to VectorQuant?

A: Welcome! Steps:

1. Fork repository on GitHub
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes
4. Run tests: `pytest tests/`
5. Submit pull request

**High-priority areas:**
- GPU acceleration (CUDA)
- Distributed computing
- Additional stochastic models
- Documentation

---

### Q: Is VectorQuant suitable for production?

A: Yes, with caveats:

✅ **Use in production if:**
- You've tested thoroughly
- Results are validated against benchmarks
- Determinism is important
- You can tolerate lack of breaking changes until v6.0

⚠ **Not suitable yet for:**
- Real-time processing (add GPU first)
- Massive portfolios (>100K assets)
- Exotic derivatives (use QuantLib instead)

---

### Q: What's the license?

A: MIT License — use freely, commercially or otherwise.

---

### Q: Can I use VectorQuant in closed-source products?

A: Yes. MIT license allows commercial use without disclosing source.

---

## Getting Help

### Where to ask questions?

1. **This FAQ** — Most common issues answered here
2. **GitHub Issues** — Bugs and feature requests
3. **Documentation** — See [docs](index.md)
4. **Examples** — See `examples/` folder in repository

### How to report bugs

Include:
1. Python version (`python --version`)
2. VectorQuant version (`pip show vectorquant`)
3. Backend used (`vq.core.get_backend()`)
4. Minimal reproducible example
5. Expected vs. actual behavior

---

**Last Updated:** Q1 2025
