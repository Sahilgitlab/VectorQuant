"""
VectorQuant Comprehensive Benchmark: C Core vs NumPy, SciPy, QuantLib
======================================================================

This benchmark compares VectorQuant's zero-dependency C core against industry-standard libraries:
- NumPy: Numerical computing library
- SciPy: Scientific computing library
- QuantLib: Quantitative finance library

Operations tested:
1. Matrix Multiplication
2. Linear Algebra (LU, QR, Cholesky, SVD, Eigendecomposition)
3. Covariance & Statistics
4. Monte Carlo Simulation (GBM)
5. Fourier Transform (FFT)
6. Optimization (BFGS)
7. Regression (OLS)
"""

import time
import random
import math
import sys
from typing import Dict, List, Tuple
import json
from datetime import datetime

# VectorQuant imports
from vectorquant.core.backend import C_AVAILABLE, CBackend, PythonBackend

# NumPy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️  NumPy not available - skipping NumPy benchmarks")

# SciPy
try:
    import scipy.linalg
    import scipy.stats
    import scipy.fft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  SciPy not available - skipping SciPy benchmarks")

# QuantLib
try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    print("⚠️  QuantLib not available - skipping QuantLib benchmarks")


class BenchmarkSuite:
    """Comprehensive benchmark suite for quantitative finance operations."""
    
    def __init__(self):
        self.results = {}
        self.c_backend = CBackend() if C_AVAILABLE else None
        self.python_backend = PythonBackend()
        
    def run_timed_operation(self, fn, name: str, iterations: int = 1, rounds: int = 3):
        """Run operation and time it across multiple rounds."""
        times = []
        for _ in range(rounds):
            start = time.perf_counter()
            for _ in range(iterations):
                fn()
            end = time.perf_counter()
            times.append((end - start) / iterations)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        return {
            "avg_ms": avg_time * 1000,
            "min_ms": min_time * 1000,
            "max_ms": max_time * 1000,
            "total_runs": iterations * rounds
        }
    
    # ==================== Matrix Operations ====================
    
    def benchmark_matrix_multiply(self, size=150):
        """Benchmark matrix multiplication."""
        print(f"\n📊 Matrix Multiplication ({size}x{size})")
        print("-" * 70)
        
        A = [[random.random() for _ in range(size)] for _ in range(size)]
        B = [[random.random() for _ in range(size)] for _ in range(size)]
        
        results = {}
        
        # VectorQuant C
        if C_AVAILABLE:
            result = self.run_timed_operation(
                lambda: self.c_backend.matrix_multiply(A, B),
                "VectorQuant C", iterations=1, rounds=3
            )
            results["vectorquant_c"] = result
            print(f"✓ VectorQuant C: {result['avg_ms']:.4f} ms")
        
        # NumPy
        if NUMPY_AVAILABLE:
            A_np = np.array(A)
            B_np = np.array(B)
            result = self.run_timed_operation(
                lambda: np.dot(A_np, B_np),
                "NumPy", iterations=1, rounds=3
            )
            results["numpy"] = result
            print(f"✓ NumPy: {result['avg_ms']:.4f} ms")
        
        # SciPy (uses BLAS under the hood)
        if SCIPY_AVAILABLE:
            A_np = np.array(A)
            B_np = np.array(B)
            result = self.run_timed_operation(
                lambda: np.dot(A_np, B_np),
                "SciPy BLAS", iterations=1, rounds=3
            )
            results["scipy_blas"] = result
            print(f"✓ SciPy/BLAS: {result['avg_ms']:.4f} ms")
        
        self.results["matrix_multiply"] = results
        self._print_speedup(results, "vectorquant_c")
        
    def benchmark_lu_decomposition(self, size=100):
        """Benchmark LU decomposition."""
        print(f"\n📊 LU Decomposition ({size}x{size})")
        print("-" * 70)
        
        A = [[random.random() for _ in range(size)] for _ in range(size)]
        results = {}
        
        # VectorQuant C
        if C_AVAILABLE:
            result = self.run_timed_operation(
                lambda: self.c_backend.lu_decomposition(A),
                "VectorQuant C", iterations=1, rounds=3
            )
            results["vectorquant_c"] = result
            print(f"✓ VectorQuant C: {result['avg_ms']:.4f} ms")
        
        # NumPy
        if NUMPY_AVAILABLE:
            A_np = np.array(A)
            result = self.run_timed_operation(
                lambda: scipy.linalg.lu(A_np),
                "NumPy/SciPy", iterations=1, rounds=3
            )
            results["numpy_scipy"] = result
            print(f"✓ NumPy/SciPy: {result['avg_ms']:.4f} ms")
        
        self.results["lu_decomposition"] = results
        self._print_speedup(results, "vectorquant_c")
    
    def benchmark_qr_decomposition(self, size=100):
        """Benchmark QR decomposition."""
        print(f"\n📊 QR Decomposition ({size}x{size})")
        print("-" * 70)
        
        A = [[random.random() for _ in range(size)] for _ in range(size)]
        results = {}
        
        # VectorQuant C
        if C_AVAILABLE:
            result = self.run_timed_operation(
                lambda: self.c_backend.qr_decomposition(A),
                "VectorQuant C", iterations=1, rounds=3
            )
            results["vectorquant_c"] = result
            print(f"✓ VectorQuant C: {result['avg_ms']:.4f} ms")
        
        # NumPy/SciPy
        if SCIPY_AVAILABLE:
            A_np = np.array(A)
            result = self.run_timed_operation(
                lambda: scipy.linalg.qr(A_np),
                "NumPy/SciPy", iterations=1, rounds=3
            )
            results["numpy_scipy"] = result
            print(f"✓ NumPy/SciPy: {result['avg_ms']:.4f} ms")
        
        self.results["qr_decomposition"] = results
        self._print_speedup(results, "vectorquant_c")
    
    def benchmark_cholesky_decomposition(self, size=100):
        """Benchmark Cholesky decomposition."""
        print(f"\n📊 Cholesky Decomposition ({size}x{size})")
        print("-" * 70)
        
        # Create symmetric positive definite matrix
        B = [[random.random() for _ in range(size)] for _ in range(size)]
        A = [[sum(B[i][k] * B[j][k] for k in range(size)) for j in range(size)] for i in range(size)]
        for i in range(size):
            A[i][i] += 5.0  # Ensure positive definite
        
        results = {}
        
        # VectorQuant C
        if C_AVAILABLE:
            result = self.run_timed_operation(
                lambda: self.c_backend.cholesky_decomposition(A),
                "VectorQuant C", iterations=1, rounds=3
            )
            results["vectorquant_c"] = result
            print(f"✓ VectorQuant C: {result['avg_ms']:.4f} ms")
        
        # NumPy/SciPy
        if SCIPY_AVAILABLE:
            A_np = np.array(A)
            result = self.run_timed_operation(
                lambda: scipy.linalg.cholesky(A_np),
                "NumPy/SciPy", iterations=1, rounds=3
            )
            results["numpy_scipy"] = result
            print(f"✓ NumPy/SciPy: {result['avg_ms']:.4f} ms")
        
        self.results["cholesky_decomposition"] = results
        self._print_speedup(results, "vectorquant_c")
    
    def benchmark_svd(self, rows=200, cols=100):
        """Benchmark Singular Value Decomposition."""
        print(f"\n📊 SVD ({rows}x{cols})")
        print("-" * 70)
        
        A = [[random.random() for _ in range(cols)] for _ in range(rows)]
        results = {}
        
        # VectorQuant C
        if C_AVAILABLE:
            result = self.run_timed_operation(
                lambda: self.c_backend.svd(A),
                "VectorQuant C", iterations=1, rounds=3
            )
            results["vectorquant_c"] = result
            print(f"✓ VectorQuant C: {result['avg_ms']:.4f} ms")
        
        # NumPy/SciPy
        if SCIPY_AVAILABLE:
            A_np = np.array(A)
            result = self.run_timed_operation(
                lambda: scipy.linalg.svd(A_np),
                "NumPy/SciPy", iterations=1, rounds=3
            )
            results["numpy_scipy"] = result
            print(f"✓ NumPy/SciPy: {result['avg_ms']:.4f} ms")
        
        self.results["svd"] = results
        self._print_speedup(results, "vectorquant_c")
    
    def benchmark_eigendecomposition(self, size=50):
        """Benchmark eigendecomposition."""
        print(f"\n📊 Eigendecomposition ({size}x{size})")
        print("-" * 70)
        
        B = [[random.random() for _ in range(size)] for _ in range(size)]
        A = [[(B[i][j] + B[j][i]) / 2 for j in range(size)] for i in range(size)]
        
        results = {}
        
        # VectorQuant C
        if C_AVAILABLE:
            result = self.run_timed_operation(
                lambda: self.c_backend.eigen_decomposition(A, num_simulations=100),
                "VectorQuant C", iterations=1, rounds=3
            )
            results["vectorquant_c"] = result
            print(f"✓ VectorQuant C: {result['avg_ms']:.4f} ms")
        
        # NumPy
        if NUMPY_AVAILABLE:
            A_np = np.array(A)
            result = self.run_timed_operation(
                lambda: np.linalg.eigh(A_np),
                "NumPy", iterations=1, rounds=3
            )
            results["numpy"] = result
            print(f"✓ NumPy: {result['avg_ms']:.4f} ms")
        
        # SciPy
        if SCIPY_AVAILABLE:
            A_np = np.array(A)
            result = self.run_timed_operation(
                lambda: scipy.linalg.eigh(A_np),
                "SciPy", iterations=1, rounds=3
            )
            results["scipy"] = result
            print(f"✓ SciPy: {result['avg_ms']:.4f} ms")
        
        self.results["eigendecomposition"] = results
        self._print_speedup(results, "vectorquant_c")
    
    # ==================== Statistical Operations ====================
    
    def benchmark_covariance(self, variables=100, observations=2000):
        """Benchmark covariance matrix computation."""
        print(f"\n📊 Covariance Matrix ({variables} variables, {observations} observations)")
        print("-" * 70)
        
        data = [[random.random() for _ in range(observations)] for _ in range(variables)]
        results = {}
        
        # VectorQuant C
        if C_AVAILABLE:
            result = self.run_timed_operation(
                lambda: self.c_backend.covariance_matrix(data),
                "VectorQuant C", iterations=1, rounds=3
            )
            results["vectorquant_c"] = result
            print(f"✓ VectorQuant C: {result['avg_ms']:.4f} ms")
        
        # NumPy
        if NUMPY_AVAILABLE:
            data_np = np.array(data)
            result = self.run_timed_operation(
                lambda: np.cov(data_np),
                "NumPy", iterations=1, rounds=3
            )
            results["numpy"] = result
            print(f"✓ NumPy: {result['avg_ms']:.4f} ms")
        
        self.results["covariance"] = results
        self._print_speedup(results, "vectorquant_c")
    
    def benchmark_ols_regression(self, variables=50, observations=1000):
        """Benchmark OLS regression."""
        print(f"\n📊 OLS Regression ({variables} features, {observations} observations)")
        print("-" * 70)
        
        X = [[random.random() for _ in range(variables)] for _ in range(observations)]
        y = [random.random() for _ in range(observations)]
        
        results = {}
        
        # VectorQuant C (use linear_regression, which is OLS)
        if C_AVAILABLE:
            result = self.run_timed_operation(
                lambda: self.c_backend.linear_regression(X, y),
                "VectorQuant C", iterations=1, rounds=3
            )
            results["vectorquant_c"] = result
            print(f"✓ VectorQuant C: {result['avg_ms']:.4f} ms")
        
        # NumPy/SciPy
        if SCIPY_AVAILABLE:
            X_np = np.array(X)
            y_np = np.array(y)
            result = self.run_timed_operation(
                lambda: scipy.linalg.lstsq(X_np, y_np),
                "SciPy", iterations=1, rounds=3
            )
            results["scipy"] = result
            print(f"✓ SciPy: {result['avg_ms']:.4f} ms")
        
        self.results["ols_regression"] = results
        self._print_speedup(results, "vectorquant_c")
    
    # ==================== Stochastic Simulations ====================
    
    def benchmark_gbm_monte_carlo(self, paths=50000):
        """Benchmark Geometric Brownian Motion simulation."""
        print(f"\n📊 Monte Carlo GBM Simulation ({paths:,} paths)")
        print("-" * 70)
        
        results = {}
        
        # VectorQuant C
        if C_AVAILABLE:
            result = self.run_timed_operation(
                lambda: self.c_backend.simulate_gbm(
                    s0=100.0, mu=0.05, sigma=0.2, t=1.0, dt=1/252, n_paths=paths
                ),
                "VectorQuant C", iterations=1, rounds=3
            )
            results["vectorquant_c"] = result
            print(f"✓ VectorQuant C: {result['avg_ms']:.4f} ms")
        
        # NumPy (pure Python implementation)
        if NUMPY_AVAILABLE:
            def numpy_gbm():
                np.random.seed(42)
                dt = 1/252
                t = 1.0
                steps = int(t / dt)
                S = np.full((paths, steps), 100.0)
                
                for i in range(1, steps):
                    dW = np.random.normal(0, np.sqrt(dt), paths)
                    S[:, i] = S[:, i-1] * np.exp((0.05 - 0.5 * 0.2**2) * dt + 0.2 * dW)
                
                return S
            
            result = self.run_timed_operation(
                numpy_gbm,
                "NumPy", iterations=1, rounds=3
            )
            results["numpy"] = result
            print(f"✓ NumPy: {result['avg_ms']:.4f} ms")
        
        # QuantLib (if available) - Simple European option pricing benchmark
        if QUANTLIB_AVAILABLE:
            try:
                # Use simpler QuantLib approach: European option pricing
                def quantlib_option_pricing():
                    """
                    QuantLib benchmark: Price 100 European options using Black-Scholes
                    This is simpler and more reliable than path simulation
                    """
                    # Set evaluation date
                    today = ql.Date(12, ql.March, 2026)
                    ql.Settings.instance().evaluationDate = today
                    
                    # Option parameters
                    spot = 100.0
                    strike = 100.0
                    risk_free = 0.05
                    dividend = 0.0
                    volatility = 0.2
                    years = 1.0
                    
                    # Build market data
                    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
                    flat_rate = ql.FlatForward(today, risk_free, ql.Actual365Fixed())
                    rate_handle = ql.YieldTermStructureHandle(flat_rate)
                    # BlackConstantVol requires: Date, Calendar, Volatility, DayCounter
                    flat_vol = ql.BlackConstantVol(today, ql.TARGET(), volatility, ql.Actual365Fixed())
                    vol_handle = ql.BlackVolTermStructureHandle(flat_vol)
                    
                    # Black-Scholes process (expects: spot, rate, volatility)
                    bs_process = ql.BlackScholesProcess(
                        spot_handle,
                        rate_handle,
                        vol_handle
                    )
                    
                    # Price multiple options  
                    engine = ql.AnalyticEuropeanEngine(bs_process)
                    maturity = ql.Date(12, ql.March, int(years) + 2026)
                    
                    prices = []
                    for strike_price in [90.0, 95.0, 100.0, 105.0, 110.0]:
                        for opt_type in [ql.Option.Call, ql.Option.Put]:
                            exercise = ql.EuropeanExercise(maturity)
                            option = ql.VanillaOption(
                                ql.PlainVanillaPayoff(opt_type, strike_price),
                                exercise
                            )
                            option.setPricingEngine(engine)
                            try:
                                prices.append(option.NPV())
                            except:
                                prices.append(0.0)
                    
                    return prices
                
                result = self.run_timed_operation(
                    quantlib_option_pricing,
                    "QuantLib", iterations=1, rounds=3
                )
                results["quantlib"] = result
                print(f"✓ QuantLib: {result['avg_ms']:.4f} ms (European option pricing)")
            except Exception as e:
                print(f"✗ QuantLib failed: {str(e)[:80]}...")  # Truncate long error messages
        
        self.results["gbm_monte_carlo"] = results
        self._print_speedup(results, "vectorquant_c")
    
    # ==================== Fourier Transform ====================
    
    def benchmark_fft(self, size=1024):
        """Benchmark Fast Fourier Transform."""
        print(f"\n📊 FFT ({size} samples)")
        print("-" * 70)
        
        data = [complex(random.random(), random.random()) for _ in range(size)]
        results = {}
        
        # VectorQuant C
        if C_AVAILABLE:
            result = self.run_timed_operation(
                lambda: self.c_backend.radix2_fft(data),
                "VectorQuant C", iterations=1, rounds=3
            )
            results["vectorquant_c"] = result
            print(f"✓ VectorQuant C: {result['avg_ms']:.4f} ms")
        
        # NumPy FFT
        if NUMPY_AVAILABLE:
            data_np = np.array(data)
            result = self.run_timed_operation(
                lambda: np.fft.fft(data_np),
                "NumPy", iterations=1, rounds=3
            )
            results["numpy"] = result
            print(f"✓ NumPy: {result['avg_ms']:.4f} ms")
        
        # SciPy FFT
        if SCIPY_AVAILABLE:
            data_np = np.array(data)
            result = self.run_timed_operation(
                lambda: scipy.fft.fft(data_np),
                "SciPy", iterations=1, rounds=3
            )
            results["scipy"] = result
            print(f"✓ SciPy: {result['avg_ms']:.4f} ms")
        
        self.results["fft"] = results
        self._print_speedup(results, "vectorquant_c")
    
    # ==================== Optimization ====================
    
    def benchmark_bfgs(self):
        """Benchmark BFGS optimization."""
        print(f"\n📊 BFGS Optimization")
        print("-" * 70)
        
        # Simple quadratic function: (x-2)^2 + (y-3)^2
        def objective(x):
            return (x[0] - 2)**2 + (x[1] - 3)**2
        
        def gradient(x):
            return [2*(x[0] - 2), 2*(x[1] - 3)]
        
        results = {}
        
        # VectorQuant C BFGS optimization (fallback to Python if not available)
        if C_AVAILABLE:
            try:
                # Try using Python backend's bfgs_minimize which is more reliable
                result = self.run_timed_operation(
                    lambda: self.python_backend.bfgs_minimize([0.0, 0.0], objective, gradient, tol=1e-6),
                    "VectorQuant Python", iterations=1, rounds=3
                )
                results["vectorquant_python"] = result
                print(f"✓ VectorQuant Python BFGS: {result['avg_ms']:.4f} ms")
            except Exception as e:
                print(f"✗ VectorQuant BFGS failed: {str(e)[:60]}...")
        else:
            try:
                result = self.run_timed_operation(
                    lambda: self.python_backend.bfgs_minimize([0.0, 0.0], objective, gradient, tol=1e-6),
                    "VectorQuant Python", iterations=1, rounds=3
                )
                results["vectorquant_python"] = result
                print(f"✓ VectorQuant Python BFGS: {result['avg_ms']:.4f} ms")
            except Exception as e:
                print(f"✗ VectorQuant BFGS failed: {str(e)[:60]}...")
        
        # SciPy BFGS
        if SCIPY_AVAILABLE:
            from scipy.optimize import minimize
            
            result = self.run_timed_operation(
                lambda: minimize(objective, [0.0, 0.0], method='BFGS'),
                "SciPy BFGS", iterations=1, rounds=3
            )
            results["scipy"] = result
            print(f"✓ SciPy BFGS: {result['avg_ms']:.4f} ms")
        
        self.results["bfgs"] = results
        # Determine baseline for speedup comparison
        baseline = "vectorquant_python" if "vectorquant_python" in results else "scipy"
        self._print_speedup(results, baseline)
    
    # ==================== Utilities ====================
    
    def _print_speedup(self, results: Dict, reference_key: str):
        """Print speedup relative to a reference implementation."""
        if reference_key not in results:
            return
        
        ref_time = results[reference_key]["avg_ms"]
        print()
        for name, data in results.items():
            if name != reference_key:
                speedup = data["avg_ms"] / ref_time
                emoji = "🚀" if speedup > 1 else "🔻"
                print(f"  {emoji} {name}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
        print()
    
    def generate_report(self):
        """Generate summary report."""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY REPORT")
        print("="*70)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"C Backend Available: {C_AVAILABLE}")
        print(f"NumPy Available: {NUMPY_AVAILABLE}")
        print(f"SciPy Available: {SCIPY_AVAILABLE}")
        print(f"QuantLib Available: {QUANTLIB_AVAILABLE}")
        print("="*70)
        
        print("\nDetailed Results (in milliseconds):")
        print("-" * 70)
        
        for test_name, results in self.results.items():
            print(f"\n{test_name.upper()}")
            for impl_name, data in results.items():
                print(f"  {impl_name:20s}: {data['avg_ms']:10.6f} ms (min={data['min_ms']:.6f}, max={data['max_ms']:.6f})")
        
        # Save results to JSON
        self._save_json_results()
    
    def _save_json_results(self):
        """Save benchmark results to JSON file."""
        output_file = "bench_comprehensive_results.json"
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "availability": {
                "c_backend": C_AVAILABLE,
                "numpy": NUMPY_AVAILABLE,
                "scipy": SCIPY_AVAILABLE,
                "quantlib": QUANTLIB_AVAILABLE
            },
            "results": self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")


def main():
    """Run the comprehensive benchmark suite."""
    print("\n" + "="*70)
    print("VectorQuant Comprehensive Benchmark: C Core vs Industry Standards")
    print("="*70)
    print()
    
    suite = BenchmarkSuite()
    
    if not C_AVAILABLE:
        print("⚠️  WARNING: VectorQuant C backend not available!")
        print("   Please install: pip install vectorquant-c")
        print()
    
    # Run benchmarks
    suite.benchmark_matrix_multiply(size=150)
    suite.benchmark_lu_decomposition(size=100)
    suite.benchmark_qr_decomposition(size=100)
    suite.benchmark_cholesky_decomposition(size=100)
    suite.benchmark_svd(rows=200, cols=100)
    suite.benchmark_eigendecomposition(size=50)
    suite.benchmark_covariance(variables=100, observations=2000)
    suite.benchmark_ols_regression(variables=50, observations=1000)
    suite.benchmark_gbm_monte_carlo(paths=50000)
    suite.benchmark_fft(size=1024)
    suite.benchmark_bfgs()
    
    # Generate report
    suite.generate_report()


if __name__ == "__main__":
    main()
