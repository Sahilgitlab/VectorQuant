# Core Concepts

Understanding VectorQuant's design philosophy and key ideas.

---

## The Problem VectorQuant Solves

### Reproducibility Crisis in Finance

**Issue:** Different computers, different results.

```python
import numpy as np

# Same data, same code
data = np.array([1, 2, 3, 4, 5])

# On Machine A (Intel, MKL BLAS):
result_A = np.cov(data)  # 2.5

# On Machine B (ARM, OpenBLAS):
result_B = np.cov(data)  # 2.5000000000000004

# Different results due to BLAS floating point differences
```

**Why it matters:**
- Regulatory audits require reproducible computation
- Model validation across teams fails
- Back-testing results don't match
- Can't publish reproducible research

### VectorQuant Solution

**Same seed → Same result, always:**

```python
import vectorquant as vq

# Deterministic RNG with seed
rng = vq.core.create_rng(seed=42)

# Machine A
result_A = vq.stats.mean([1, 2, 3, 4, 5])  # 3.0

# Machine B (hours later, different IP)
result_B = vq.stats.mean([1, 2, 3, 4, 5])  # 3.0

# Bit-identical across all platforms
```

---

## Determinism in Detail

### The RNG Story

**Xoroshiro128+ Generator:**

```c
// Super-fast (2 CPU cycles) and high-quality
state = [x, y]
return (x + y) rotated by sqrt
update x and y deterministically
```

**Why custom RNG?**
- NumPy uses BLAS internal RNG (platform-dependent)
- MKL, OpenBLAS, vecLib give different sequences
- Xoroshiro is faster and consistent

**Usage:**

```python
rng = vq.core.create_rng(seed=42)

# Same seed always produces same sequence
path1 = vq.stochastic.simulate_gbm(..., n_paths=1000)
path2 = vq.stochastic.simulate_gbm(..., n_paths=1000)  # Different paths!

# Only identical if you reseed
rng = vq.core.create_rng(seed=42)
path1 = ...

rng = vq.core.create_rng(seed=42)
path2 = ...  # Identical to path1
```

---

## The Three-Layer Architecture

### Why Three Layers?

**Problem:**
```python
# If we write everything in Python:
import time
start = time.time()
result = some_expensive_computation()
elapsed = time.time() - start
# 5 seconds ❌
```

**Solution: Offload heavy work to C:**

```
User Code (Python)
    ↓ (lightweight call)
Dispatch Layer (which engine?)
    ↓
C Engine (165x speedup) OR Python Fallback (always works)
```

### Layer 1: Python API

**Purpose:** User-friendly interface

```python
def portfolio_return(weights, returns):
    """Simple list comprehension - no heavy work"""
    return sum(w * r for w, r in zip(weights, returns))
```

**Characteristics:**
- No numerical loops
- Delegates to backend for heavy lifting
- Transparent to user

### Layer 2: Smart Dispatch

**Purpose:** Route to optimal backend

```python
def get_backend():
    try:
        import vectorquant_c  # Try C engine
        return CBackend()
    except ImportError:
        return PythonBackend()

backend = get_backend()  # Decided once at import
```

**Why smart?**
- Decision made once (not per call)
- No runtime overhead
- Clean fallback (always works)

### Layer 3: Engines

**C Engine (High Performance):**

```c
// vectorquant-c/src/linalg.c
void matrix_multiply(double *A, int m, int n,
                     double *B, int k, double *C) {
    // SIMD loops - 50-200x Python
}
```

**Python Backend (Fallback):**

```python
# Same interface, pure Python
def py_matrix_multiply(A, m, n, B, k):
    C = [[0]*k for _ in range(m)]
    for i in range(m):
        for j in range(k):
            for p in range(n):
                C[i][j] += A[i*n+p] * B[p*k+j]
    return C
```

---

## Zero Dependencies Philosophy

### Why No NumPy?

**Argument for NumPy:**
- Mature, well-tested
- BLAS/LAPACK optimized
- Everyone uses it

**Counter-argument:**

| Issue | NumPy | VectorQuant |
|-------|--------|-------------|
| Determinism | Platform-dependent BLAS | Custom implementation |
| Dependencies | Many (BLAS, LAPACK, GFORTRAN) | Zero |
| Customization | Hard - you're locked in | Easy - we own the code |

### The Cost

```
NumPy + SciPy approach:
- Pip install numpy scipy (two packages)
- Automatic BLAS from system (varies)
- Many potential version conflicts
- Not guaranteed reproducible

VectorQuant approach:
- Pip install vectorquant (one package)
- C from our source code
- Zero external dependencies
- Deterministic all the time
```

---

## Optimization Strategies

### When to Use Gradient Descent

**Good for:**
- Smooth, continuous problems
- Portfolio optimization
- Maximum likelihood estimation

**Bad for:**
- Integer programming (asset selection)
- Discrete problems (which factors to include)

### L-BFGS Algorithm

VectorQuant uses Limited-memory BFGS:

```
Start: x = [initial guess]

Repeat until convergence:
  1. Compute gradient
  2. Approximate Hessian (using past steps)
  3. Update: x ← x + direction
  4. Check convergence
```

**Why BFGS?**
- Second-order (converges in ~log(n) steps)
- Memory-efficient (limited history)
- Robust (handles ill-conditioned problems)

**Example: Portfolio Optimization**

```python
# Minimize: -Sharpe(weights)
# Subject to: sum(weights) = 1

weights = vq.portfolio.optimize_max_sharpe(returns, rf)
# Behind scenes: BFGS optimizer finding optimal portfolio
```

---

## Memory Model

### Assumptions

All VectorQuant kernels assume:

1. **Contiguous memory**
   ```python
   # Good: contiguous row-major array
   data = [[a, b, c],
           [d, e, f],
           [g, h, i]]
   # Bad: jagged array
   data = [[a, b], [c, d, e, f], ...]
   ```

2. **Row-major layout**
   ```
   In memory: [a, b, c, d, e, f, g, h, i]
   Not:       [a, d, g, b, e, h, c, f, i]  (column-major)
   ```

3. **SIMD alignment**
   ```
   16-byte boundaries for AVX2 operations
   Automatic in Python lists → internal arrays
   ```

### Why This Matters

**Poor memory layout:**
```python
# Iterates by column - cache misses!
for col in range(n):
    for row in range(m):
        value = data[row][col]
```

**Good memory layout:**
```python
# Iterates by row - cache hits!
for row in range(m):
    for col in range(n):
        value = data[row][col]
```

---

## Parallelization Rules

### Rule: Parallelize Outer Loops

**Correct:**

```c
#pragma omp parallel for
for (int i = 0; i < n_paths; ++i) {
    // Each thread: full path simulation
    for (int step = 0; step < n_steps; ++step) {
        // Nested loop stays sequential
        simulate_step(i, step);
    }
}
```

**Why?**
- No synchronization needed between paths
- Each thread works independently
- Great NUMA scaling (multi-socket systems)

**Incorrect:**

```c
#pragma omp parallel for
for (int i = 0; i < n_paths; ++i) {
    #pragma omp parallel for  // ❌ Nested parallel
    for (int step = 0; step < n_steps; ++step) {
        // Too much overhead, potential deadlock
    }
}
```

### Speedup Scaling

```
Ideal scaling (linear): 4 cores → 4x speedup
Practical scaling: 4 cores → 3.5x speedup (overhead)
Rule of thumb: Each outer loop parallelizes ~35% overhead

For n_paths = 10,000:
- 1 core: 850ms
- 4 cores: 850ms / 3.5 = 242ms
```

---

## Verification Philosophy

### Hallucination Detection

**Problem:** LLMs can make up answers that sound plausible.

```python
# LLM says:
"The Sharpe ratio of [0.01, 0.02, -0.01] with rf=0.02 is 0.707"

# Verify:
result = vq.ai.verify_calculation(
    "mean([0.01, 0.02, -0.01]) / std([0.01, 0.02, -0.01])",
    expected=0.707
)
# Catches: Actually should be ~0.196 (very different!)
```

### Confidence Scoring

```python
error_rate = |actual - expected| / |expected|

confidence = max(0, 1 - error_rate)

# Examples:
error = 0%    → confidence = 1.0 ✓ Perfect
error = 10%   → confidence = 0.9 ✓ Good
error = 50%   → confidence = 0.5 ⚠ Questionable
error = 100%+ → confidence = 0.0 ✗ Wrong
```

### Trace Generation

```python
result = vq.ai.explain_sharpe([0.01, 0.02], rf=0.02)

for step in result.steps:
    print(step)

# Output:
# Step 1: mean([0.01, 0.02]) = 0.015
# Step 2: std([0.01, 0.02]) = 0.007071
# Step 3: sharpe = (0.015 - 0.0008) / 0.007071 = 1.978
# Step 4: confidence = 1.0
```

---

## Comparisons with Alternatives

### VectorQuant vs NumPy

| Aspect | VectorQuant | NumPy |
|--------|------------|-------|
| Speed | 50x (our implementations) | 1x (reference) |
| Determinism | ✓ Guaranteed | ✗ Platform-dependent |
| Ease | Simple Python API | More flexible |
| Dependencies | 0 | Many |
| Specialization | Finance-focused | General math |

**Use NumPy when:** You need flexibility and determinism doesn't matter

**Use VectorQuant when:** You need reproducible finance computations

### VectorQuant vs QuantLib

| Aspect | VectorQuant | QuantLib |
|--------|------------|----------|
| Language | Python | C++ |
| Easy setup | Pip install | Complex build |
| Learning curve | Hours | Days |
| Hallucination detection | ✓ | ✗ |
| Exotic derivatives | Limited | Extensive |
| Enterprise ready | Newer | Battle-tested |

**Use QuantLib when:** You need extensive derivatives (exotic options)

**Use VectorQuant when:** You want ease + modern features (verification)

---

## Design Principles

### 1. Fast by Default

C engine is standard. Python is fallback.

```python
backend = vq.core.get_backend()
# Usually "C" (165x faster)
# Falls back to "Python" if C unavailable
```

### 2. Always Deterministic

Same input + seed = same output, always, everywhere.

```python
result1 = vq.stats.mean([1, 2, 3])  # 2.0
result2 = vq.stats.mean([1, 2, 3])  # 2.0 (identical)
```

### 3. Transparent

User always knows exactly what's happening.

```python
print(vq.core.get_backend())  # Shows active engine
```

### 4. Specialized

Focused on quant/financial workflows.

```
✓ Portfolio optimization
✓ Options pricing
✓ Risk metrics
✗ General linear algebra (use NumPy)
✗ Machine learning (use TensorFlow)
```

### 5. Correct First, Fast Second

Accuracy > Speed.

```python
# Careful numerical handling
# Stable algorithms (backward stable)
# Deterministic rounding
```

---

## Performance Physics

### Roofline Model

Performance limited by:

```
1. Compute bandwidth (FLOPs/sec)
2. Memory bandwidth (bytes/sec)

Actual performance = min(compute_limit, memory_limit)
```

### VectorQuant Strategy

**Observation:**
- Modern CPUs: ~100 GFLOPS compute
- Memory bandwidth: ~50 GB/sec
- Ratio: 100 GFLOPS / (50 GB/sec) = 16 operations per byte

**Implication:**
- Operations with small data = memory-bound
- Must reduce memory accesses

**Example: Matrix Multiply**

```c
// Naive (memory-bound)
for (int k = 0; k < n; ++k) {
    C[i][j] += A[i][k] * B[k][j];  // 3 memory accesses per FLOP
}

// SIMD tiling (compute-bound)
for (...) {
    // Accumulate in registers
    // Use SIMD (4x more parallel)
    // Reuse data in cache
}
// Result: Better roofline efficiency
```

---

## When to Use VectorQuant

### Good Fit

✓ Portfolio optimization
✓ Greeks calculation
✓ VaR computation
✓ Deterministic Monte Carlo
✓ Factor analysis
✓ Hallucination detection for LLM

### Not a Good Fit

✗ General machine learning (use TensorFlow)
✗ Image/video processing (use OpenCV)
✗ Signal processing (use SciPy)
✗ High-dimensional sampling (use PyMC)

### Mixed Fit

⚠ Time series analysis: VectorQuant stats + your forecasting
⚠ Deep learning + finance: PyTorch + VectorQuant for metrics
⚠ Real-time trading: VectorQuant for signal, your engine for execution

---

## Summary of Key Ideas

| Concept | Why It Matters |
|---------|---|
| Determinism | Reproducible across platforms |
| Three layers | Fast by default, always works |
| Zero dependencies | Simple installation, fewer conflicts |
| Smart dispatch | No slowdown from flexibility |
| Custom RNG | Bit-identical results |
| Verification | Catch LLM hallucinations |
| Specialization | Best finance implementations |

**Master these concepts and you understand VectorQuant's philosophy.**
