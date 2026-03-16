# VectorQuant Benchmark Suite - Complete Overview

## 📊 Benchmark Files Overview

This directory contains a comprehensive benchmark suite comparing VectorQuant's zero-dependency C core against industry-standard libraries (NumPy, SciPy, QuantLib).

### File Structure

```
benchmarks/
├── bench_comprehensive_comparison.py      # Main benchmark suite (11 operations)
├── bench_performance_metrics.py           # Advanced statistical analysis
├── bench_speedup_analysis.py              # Detailed speedup analysis & insights
├── bench_runner.py                        # Unified runner for all benchmarks
├── BENCHMARK_GUIDE.md                     # Detailed documentation
├── BENCHMARK_SUITE_OVERVIEW.md            # This file
│
└── [Existing benchmarks for compatibility]
    ├── bench_c_vs_python.py               # C vs pure Python fallback
    ├── bench_monte_carlo.py               # Monte Carlo specific tests
    ├── bench_covariance.py                # Covariance computation
    ├── bench_portfolio_optimization.py    # Portfolio optimization
    ├── bench_regression.py                # Regression analysis
    └── run_benchmarks.py                  # Legacy runner
```

---

## 🚀 Quick Start

### One Command to Run Everything
```bash
cd benchmarks/
python bench_runner.py
```

### Run Individual Benchmarks
```bash
# Comprehensive comparison across 11 operations
python bench_comprehensive_comparison.py

# Advanced metrics and statistical analysis
python bench_performance_metrics.py

# Detailed speedup analysis with insights
python bench_speedup_analysis.py
```

---

## 📋 Benchmark Files Details

### 1️⃣ `bench_comprehensive_comparison.py` (Main Suite)

**Purpose**: Comprehensive comparison across 11 core quantitative finance operations.

**What It Tests**:
- ✓ Matrix Multiplication (150×150)
- ✓ LU Decomposition (100×100)
- ✓ QR Decomposition (100×100)
- ✓ Cholesky Decomposition (100×100)
- ✓ SVD - Singular Value Decomposition (200×100)
- ✓ Eigendecomposition (50×50)
- ✓ Covariance Matrix (100 vars × 2,000 obs)
- ✓ OLS Regression (50 features × 1,000 obs)
- ✓ Monte Carlo GBM Simulation (50,000 paths)
- ✓ FFT - Fast Fourier Transform (1,024 samples)
- ✓ BFGS Optimization

**Implementations Compared**:
- VectorQuant C Core
- NumPy
- SciPy
- QuantLib (where applicable)

**Output**:
```
Console: Real-time performance metrics with speedup comparisons
JSON: bench_comprehensive_results.json (detailed results for each operation)
```

**Example Output**:
```
📊 Matrix Multiplication (150x150)
──────────────────────────────────────────────────────────
✓ VectorQuant C: 12.3456 ms
✓ NumPy: 11.8900 ms
  NumPy: 1.04x slower

Results Saved
```

---

### 2️⃣ `bench_performance_metrics.py` (Advanced Analysis)

**Purpose**: Statistical performance analysis with detailed metrics.

**What It Analyzes**:
- Dense matrix operations across multiple sizes (50×50 to 500×500)
- Matrix decompositions with statistical measures
- Covariance computation scaling analysis
- FFT performance with size scaling
- Throughput analysis (operations per second)

**Metrics Computed**:
- **Mean**: Average execution time
- **Median**: Middle value
- **Std Dev**: Consistency measure
- **Min/Max**: Range of observations
- **P95/P99**: Percentile performance
- **Throughput**: Ops/second

**Output**:
```
Console: Formatted tables with statistical breakdowns
JSON: bench_performance_metrics.json (all metrics for analysis)
```

**Use Case**: 
- Understand performance consistency
- Identify if implementation is stable or variable
- Check 95th/99th percentile for production SLAs

---

### 3️⃣ `bench_speedup_analysis.py` (Detailed Insights)

**Purpose**: Deep analysis of speedup factors and efficiency.

**What It Shows**:

1. **Scaling Behavior Analysis**
   - How time grows with problem size
   - Validates O(n³) complexity for matrix ops
   - Shows if algorithm has expected scaling

2. **Algorithm Efficiency Analysis**
   - Computational GFLOPs achieved
   - Percentage of theoretical peak
   - Indicates SIMD utilization

3. **Cache Behavior Analysis**
   - Working set vs execution time
   - Cache efficiency metrics
   - Memory bandwidth utilization

4. **Comprehensive Operation Comparison**
   - Side-by-side timing of major operations
   - Formatted comparison tables
   - Quick visual reference

5. **Performance Insights**
   - Recommendations for which library to use
   - Strengths and weaknesses highlighted
   - Best practices for implementation

**Output**:
```
Console: Formatted tables and insights
JSON: bench_speedup_analysis.json (structured comparison data)
```

---

### 4️⃣ `bench_runner.py` (Unified Orchestrator)

**Purpose**: Single entry point to run all benchmarks sequentially.

**Features**:
- Executes all benchmark suites
- Collects JSON outputs
- Generates unified report
- Handles errors gracefully
- 10-minute timeout per benchmark

**Usage**:
```bash
python bench_runner.py
```

**Generated Outputs**:
```
bench_comprehensive_results.json      # Results from comprehensive suite
bench_performance_metrics.json        # Statistical metrics
bench_speedup_analysis.json           # Detailed analysis
benchmark_report.json                 # Unified report (all data combined)
```

**Report Structure**:
```json
{
  "timestamp": "2024-03-12T...",
  "benchmarks": {
    "comprehensive": { /* results */ },
    "metrics": { /* metrics */ },
    "speedup_analysis": { /* analysis */ }
  }
}
```

---

## 📊 Output Files Explained

### `bench_comprehensive_results.json`
```json
{
  "timestamp": "2024-03-12T10:30:00",
  "availability": {
    "c_backend": true,
    "numpy": true,
    "scipy": true,
    "quantlib": false
  },
  "results": {
    "matrix_multiply": {
      "vectorquant_c": { "avg_ms": 12.34, "min_ms": 11.99, "max_ms": 12.89 },
      "numpy": { "avg_ms": 11.88, "min_ms": 11.45, "max_ms": 12.34 }
    }
  }
}
```

### `bench_performance_metrics.json`
```json
{
  "measurements": {
    "matrix_operations": {
      "100x100": {
        "vectorquant_c": {
          "mean_ms": 2.345,
          "median_ms": 2.340,
          "std_dev": 0.045,
          "p95_ms": 2.398,
          "p99_ms": 2.410
        }
      }
    }
  }
}
```

### `bench_speedup_analysis.json`
```json
{
  "timestamp": "2024-03-12T...",
  "comparison_matrix": {
    "LU Decomposition (100×100)": {
      "vectorquant_c": 5.67,
      "scipy": 4.23
    }
  },
  "insights": {
    "c_available": true,
    "numpy_available": true
  }
}
```

---

## 🎯 Use Cases

### For Performance Verification
```bash
# Run after code changes to verify no regressions
python bench_runner.py

# Compare with baseline
diff baseline_report.json benchmark_report.json
```

### For Documentation
```bash
# Generate performance statistics for README
python bench_comprehensive_comparison.py  # Copy output to docs
```

### For Tuning
```bash
# Run detailed analysis to find bottlenecks
python bench_performance_metrics.py

# Analyze scaling behavior
python bench_speedup_analysis.py
```

### For Library Selection
```bash
# See which library performs best for your use case
python bench_speedup_analysis.py  # Reviews recommendations
```

---

## 🔍 Performance Interpretation Guide

### Expected Performance Profiles

**Matrix Multiplication** (Good C Performance)
- VectorQuant C: Cache-blocked kernel, AVX2 SIMD
- NumPy: Optimized BLAS (Intel MKL, OpenBLAS)
- **Expected**: 0.8-1.2x (C can be competitive or slightly slower)

**Covariance** (VectorQuant C Advantage)
- VectorQuant C: Parallel column-wise computation
- NumPy: Standard two-pass algorithm
- **Expected**: 1.5-3.0x (C significantly faster)

**Monte Carlo GBM** (VectorQuant C Dominance)
- VectorQuant C: Vectorized Box-Muller, parallel paths
- NumPy: Pure Python loops with NumPy ops
- **Expected**: 2.0-5.0x (C much faster)

**Linear Algebra** (Variable)
- VectorQuant C: Zero-dependency algorithms
- SciPy: Highly optimized LAPACK/BLAS
- **Expected**: 0.5-1.5x (SciPy may be faster for large problems)

---

## ⚙️ Configuration

### Adjusting Test Sizes
Edit the benchmark files to change problem dimensions:

```python
# In bench_comprehensive_comparison.py
suite.benchmark_matrix_multiply(size=500)  # Increase from 150
suite.benchmark_gbm_monte_carlo(paths=100000)  # Increase from 50k

# In bench_performance_metrics.py
sizes = [100, 200, 300, 400, 500]  # Add or remove sizes
```

### Adjusting Iteration Counts
For faster benchmarks (less accurate):
```python
iterations=1, rounds=1  # Quick test
```

For slower but more accurate:
```python
iterations=10, rounds=5  # Thorough test
```

---

## 🐛 Troubleshooting

### C Backend Not Found
```bash
pip install vectorquant-c
python -c "from vectorquant.core.backend import C_AVAILABLE; print(C_AVAILABLE)"
```

### High Variance in Results
- Close background applications
- Run on isolated CPU cores
- Increase iterations/rounds
- Check system clock source

### Library Import Errors
```bash
# Install missing libraries
pip install numpy scipy
pip install QuantLib  # Complex, optional
```

### Benchmark Timeout
- Reduce problem sizes
- Reduce iterations/rounds
- Check for infinite loops in custom operations

---

## 📈 Historical Tracking

### Create Baseline
```bash
python bench_runner.py
cp benchmark_report.json baseline_2024_03_12.json
```

### Compare Versions
```bash
# After optimization or code changes
python bench_runner.py
python -c "
import json
with open('baseline_2024_03_12.json') as f1, open('benchmark_report.json') as f2:
    baseline = json.load(f1)
    current = json.load(f2)
    # Custom comparison logic
"
```

---

## 📚 Integration with CI/CD

### GitHub Actions Example
```yaml
name: Performance Benchmarks
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install numpy scipy quantlib vectorquant-c
      - run: cd benchmarks && python bench_runner.py
      - uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/benchmark_report.json
```

---

## 🔗 Related Files

- [BENCHMARK_GUIDE.md](./BENCHMARK_GUIDE.md) - Detailed operation descriptions
- [../Claude.md](../Claude.md) - Project architecture & philosophy
- [../README.md](../README.md) - Main project documentation

---

## 📞 Support

For issues running benchmarks:
1. Check that all dependencies are installed
2. Review BENCHMARK_GUIDE.md for detailed information
3. Run individual benchmarks to isolate issues
4. Check JSON outputs for detailed error information

---

## 📝 Notes

- Benchmarks are designed for Ubuntu/Linux and Windows
- macOS may show different results due to different BLAS implementations
- Results are dependent on system hardware and current load
- Always run multiple times and check consistency
- Use P95/P99 metrics for production performance estimation

---

Generated: 2024+
VectorQuant Zero-Dependency Finance Engine
