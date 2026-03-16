# VectorQuant Benchmark Suite

Comprehensive performance comparison between VectorQuant's C core and industry-standard libraries:
- **NumPy**: Numerical computing with BLAS/LAPACK
- **SciPy**: Scientific computing toolkit
- **QuantLib**: Quantitative finance library

## Quick Start

### Run All Benchmarks
```bash
cd benchmarks/
python bench_runner.py
```

This will execute all benchmark suites and generate comprehensive reports.

### Run Individual Benchmarks

**Comprehensive Comparison** (Main benchmarks)
```bash
python benchmarks/bench_comprehensive_comparison.py
```

Tests:
- Matrix Multiplication (various sizes)
- LU Decomposition
- QR Decomposition
- Cholesky Decomposition
- SVD (Singular Value Decomposition)
- Eigendecomposition
- Covariance Matrix Computation
- OLS Regression
- Monte Carlo Simulation (GBM)
- FFT (Fast Fourier Transform)
- BFGS Optimization

---

**Performance Metrics & Analysis** (Advanced metrics)
```bash
python benchmarks/bench_performance_metrics.py
```

Tests:
- Dense Matrix Operations (scaling analysis)
- Matrix Decompositions (LU, QR, SVD)
- Covariance Computation (dimension scaling)
- FFT Performance (size scaling)
- Throughput Analysis (operations per second)

---

## Output

The benchmark suite generates:

1. **Console Output**: Real-time performance metrics with speedup comparisons
2. **JSON Reports**:
   - `bench_comprehensive_results.json` - Detailed benchmark results
   - `bench_performance_metrics.json` - Statistical metrics (mean, median, std dev, percentiles)
   - `benchmark_report.json` - Unified report with all results

## Performance Metrics Explained

- **Mean (avg)**: Average execution time across all runs
- **Median**: Middle value of execution times
- **Std Dev**: Standard deviation (lower = more consistent)
- **Min/Max**: Best and worst observed times
- **P95/P99**: 95th and 99th percentile times
- **Speedup**: Ratio of comparison library vs VectorQuant C (>1 means C is faster)

## Requirements

### Required
- Python 3.8+
- VectorQuant with C backend: `pip install vectorquant-c`

### Optional (for full comparisons)
```bash
pip install numpy scipy
pip install QuantLib  # Optional, complex installation
```

### Checking Installation
```python
import vectorquant.core.backend as backend
print(f"C Backend Available: {backend.C_AVAILABLE}")

import numpy
import scipy
import QuantLib  # Optional
```

## Benchmark Details

### 1. Dense Matrix Multiplication
- **Sizes Tested**: 50×50, 100×100, 200×200, 500×500
- **Algorithm**: O(n³) naive algorithm
- **Critical Performance**: Memory bandwidth, cache efficiency
- **SIMD Optimization**: AVX2 vectorization in C backend

### 2. LU Decomposition
- **Size**: 100×100
- **Method**: Gaussian elimination with partial pivoting
- **Applications**: Linear system solving, determinant computation
- **VectorQuant Advantage**: Zero external dependencies, direct C implementation

### 3. QR Decomposition
- **Size**: 100×100
- **Method**: Householder reflections
- **Applications**: Least squares problems, eigenvalue computation
- **Critical Test**: Numerical stability under ill-conditioned matrices

### 4. Cholesky Decomposition
- **Size**: 100×100
- **Precondition**: Symmetric positive definite matrix
- **Speed Factor**: Faster than LU due to symmetry exploitation
- **Applications**: Covariance matrix factorization, Monte Carlo sampling

### 5. SVD (Singular Value Decomposition)
- **Size**: 200×100
- **Method**: Power iteration or Lanczos
- **Applications**: PCA, dimensionality reduction, matrix rank computation
- **Numerical Challenge**: Requires high precision for small singular values

### 6. Eigendecomposition
- **Size**: 50×50
- **Precondition**: Symmetric matrix
- **Method**: QR iteration with shifts
- **Applications**: Principal component analysis (PCA), risk decomposition

### 7. Covariance Matrix
- **Dimensions**: 
  - 10 variables × 1,000 observations
  - 50 variables × 2,000 observations
  - 100 variables × 2,000 observations
  - 200 variables × 3,000 observations
- **Algorithm**: Two-pass or online computation
- **VectorQuant Feature**: Parallel column-wise computation with OpenMP
- **Performance Gain**: Highly optimized for real-time portfolio risk

### 8. OLS Regression
- **Size**: 50 features × 1,000 observations
- **Method**: Normal equations or SVD
- **Critical Path**: Matrix multiplication and linear solver
- **Comparison**: SciPy uses optimized BLAS routines

### 9. Monte Carlo GBM
- **Paths**: 50,000
- **Time Steps**: 252 (daily returns for 1 year)
- **Model**: Geometric Brownian Motion
- **Parameters**: S₀=100, μ=0.05, σ=0.2, T=1
- **VectorQuant Strategy**: Vectorized Box-Muller, parallel path generation
- **QuantLib**: Includes day-count conventions and QuantLib calendars

### 10. FFT (Fast Fourier Transform)
- **Sizes**: 128, 256, 512, 1024, 2048
- **Algorithm**: Radix-2 Cooley-Tukey
- **Applications**: Option pricing (Carr-Madan), volatility surface interpolation
- **Bottleneck**: Complex arithmetic, memory access patterns

### 11. BFGS Optimization
- **Problem**: Minimize f(x) = (x₀-2)² + (x₁-3)²
- **Method**: Quasi-Newton with BFGS updates
- **Applications**: Maximum likelihood estimation, portfolio optimization
- **Convergence**: VectorQuant uses two-loop recursion for memory efficiency

---

## Interpreting Results

### What's Good?
- **VectorQuant C Speedup > 1.5×**: Indicates effective SIMD and cache optimization
- **Consistent Times**: Low standard deviation = predictable performance
- **Scaling**: O(n³) algorithms should show cubic growth with problem size

### What's Expected?
- **Matrix Multiply**: NumPy uses optimized BLAS; C speedup ≈ 0.8-1.2×
- **Decompositions**: C might be slower due to general implementations vs highly tuned libraries
- **Covariance**: C can be faster due to cache-aware column-wise parallelization
- **Stochastic**: C dominates (vectorized RNG, parallel paths)

### Troubleshooting
```bash
# If C backend shows no speedup:
pip install --upgrade vectorquant-c

# If seeing high variance in times:
- Close background applications
- Run on dedicated CPU cores
- Increase iterations/rounds in benchmark

# If libraries are missing:
pip install numpy scipy
```

## Performance Expectations by Operation

| Operation | VectorQuant C | NumPy | SciPy | Notes |
|-----------|---------------|-------|-------|-------|
| MatMul (150×150) | 1.0× | 0.8-1.0× | N/A | C uses cache-blocked ikj kernel |
| Covariance (100×2000) | 1.0× | 0.4-0.5× | N/A | Parallel column-wise outstanding |
| GBM (50k paths) | 1.0× | 1.5-2.5× | Slow | Vectorized Box-Muller wins |
| FFT (1024) | 0.9-1.1× | 0.8-1.0× | 0.8-1.0× | Radix-2 vs faster FFTPACK |
| SVD (200×100) | Variable | 0.7-1.0× | Lower precision | Depends on condition number |

---

## Advanced Usage

### Custom Benchmark
```python
from benchmarks.bench_performance_metrics import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
metric = analyzer.measure_operation(
    "My Custom Operation",
    my_function,
    iterations=10,
    rounds=5
)
print(f"Mean: {metric['mean_ms']:.6f} ms")
```

### Extract Results Programmatically
```python
import json

with open("bench_comprehensive_results.json", 'r') as f:
    results = json.load(f)

for test_name, impl_results in results['results'].items():
    print(f"\n{test_name}:")
    for impl, times in impl_results.items():
        print(f"  {impl}: {times['avg_ms']:.6f} ms")
```

### Comparing Across Versions
```bash
# Save baseline
python benchmarks/bench_runner.py
mv benchmark_report.json baseline_report.json

# After optimization
python benchmarks/bench_runner.py

# Compare manually
python -c "
import json
with open('baseline_report.json') as f1, open('benchmark_report.json') as f2:
    baseline, current = json.load(f1), json.load(f2)
    # Custom comparison logic
"
```

---

## Performance Tuning Guide

### For C Backend
1. **Compile Flags**: Ensure `-O3 -march=native -mavx2` in setup.py
2. **OpenMP**: Enable `#pragma omp parallel for` directives
3. **Cache**: Use cache-line alignment (64 bytes) for arrays
4. **SIMD**: Auto-vectorization with compiler flags

### For Algorithms
1. **Blocking**: Use cache-oblivious blocking for matrix multiply
2. **Parallelism**: Outer-loop parallelization (rows for MatMul, cols for Cov)
3. **Precision**: Use `float` for GBM, `double` for linear algebra
4. **Memory**: Pre-allocate, avoid reallocation in tight loops

---

## Known Limitations

- **NumPy/SciPy**: May be cached or optimized by system BLAS (Intel MKL, OpenBLAS)
- **QuantLib**: Complex installation; may not be available in all environments
- **Warm-up**: First run may be slower due to JIT, caching, or library initialization
- **Problem Scale**: Benchmarks use moderate sizes; larger problems may show different patterns

---

## References

- [VectorQuant Documentation](../README.md)
- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [SciPy Documentation](https://scipy.org/)
- [QuantLib Guide](https://www.quantlib.org/docs/)

---

## Contributing

To add new benchmarks:

1. Add test function to `BenchmarkSuite` class
2. Call it from `main()`
3. Ensure JSON output is consistent
4. Update this README with operation details

Example:
```python
def benchmark_my_op(self, size=100):
    """Benchmark my operation."""
    print(f"\n📊 My Operation ({size})")
    print("-" * 70)
    
    # Prepare data
    data = ...
    
    # Test implementations
    results = {}
    if C_AVAILABLE:
        result = self.run_timed_operation(...)
        results["vectorquant_c"] = result
    
    self.results["my_operation"] = results
    self._print_speedup(results, "vectorquant_c")
```

---

Generated: 2024+
VectorQuant Zero-Dependency Finance Engine
