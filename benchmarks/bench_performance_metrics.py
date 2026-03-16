"""
VectorQuant Performance Metrics & Analysis
===========================================

Advanced benchmark suite with detailed performance metrics, statistical analysis,
and comparative speedup analysis.
"""

import time
import random
import math
import json
from datetime import datetime
from typing import Dict, List, Tuple
import statistics

from vectorquant.core.backend import C_AVAILABLE, CBackend, PythonBackend

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import scipy.linalg
    import scipy.stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class PerformanceAnalyzer:
    """Detailed performance analysis and metrics calculation."""
    
    def __init__(self):
        self.measurements = {}
        self.c_backend = CBackend() if C_AVAILABLE else None
        self.python_backend = PythonBackend()
    
    def measure_operation(self, operation_name: str, fn, iterations: int = 10, rounds: int = 5) -> Dict:
        """Measure operation performance with detailed metrics."""
        all_times = []
        
        for _ in range(rounds):
            round_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                fn()
                end = time.perf_counter()
                round_times.append(end - start)
            all_times.extend(round_times)
        
        # Convert to milliseconds
        times_ms = [t * 1000 for t in all_times]
        
        return {
            "operation": operation_name,
            "count": len(times_ms),
            "mean_ms": statistics.mean(times_ms),
            "median_ms": statistics.median(times_ms),
            "std_dev": statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
            "p95_ms": sorted(times_ms)[int(len(times_ms) * 0.95)],
            "p99_ms": sorted(times_ms)[int(len(times_ms) * 0.99)],
        }
    
    def compute_speedup(self, reference: Dict, target: Dict) -> float:
        """Compute speedup factor."""
        return reference["mean_ms"] / target["mean_ms"] if target["mean_ms"] > 0 else 0
    
    def benchmark_dense_matrix_ops(self):
        """Benchmark dense matrix operations across different sizes."""
        print("\n" + "="*80)
        print("DENSE MATRIX OPERATIONS PERFORMANCE ANALYSIS")
        print("="*80)
        
        sizes = [50, 100, 200, 500]
        results = {}
        
        for size in sizes:
            print(f"\n📊 Size {size}x{size}:")
            print("-" * 80)
            
            A = [[random.random() for _ in range(size)] for _ in range(size)]
            B = [[random.random() for _ in range(size)] for _ in range(size)]
            
            size_key = f"{size}x{size}"
            results[size_key] = {}
            
            # VectorQuant C
            if C_AVAILABLE:
                metric = self.measure_operation(
                    f"VectorQuant C {size}x{size}",
                    lambda: self.c_backend.matrix_multiply(A, B),
                    iterations=3, rounds=5
                )
                results[size_key]["vectorquant_c"] = metric
                self._print_metric(metric)
            
            # NumPy
            if NUMPY_AVAILABLE:
                A_np = np.array(A)
                B_np = np.array(B)
                metric = self.measure_operation(
                    f"NumPy {size}x{size}",
                    lambda: np.dot(A_np, B_np),
                    iterations=3, rounds=5
                )
                results[size_key]["numpy"] = metric
                self._print_metric(metric)
            
            # Calculate speedup
            if "vectorquant_c" in results[size_key] and "numpy" in results[size_key]:
                speedup = self.compute_speedup(results[size_key]["numpy"], results[size_key]["vectorquant_c"])
                symbol = "🚀" if speedup > 1 else "🔻"
                print(f"\n{symbol} C vs NumPy: {speedup:.2f}x")
        
        self.measurements["matrix_operations"] = results
    
    def benchmark_decompositions(self):
        """Benchmark matrix decompositions."""
        print("\n" + "="*80)
        print("MATRIX DECOMPOSITIONS PERFORMANCE ANALYSIS")
        print("="*80)
        
        decompositions = [
            ("LU", self.c_backend.lu_decomposition if C_AVAILABLE else None, "scipy.linalg.lu"),
            ("QR", self.c_backend.qr_decomposition if C_AVAILABLE else None, "scipy.linalg.qr"),
            ("SVD", self.c_backend.svd if C_AVAILABLE else None, "scipy.linalg.svd"),
        ]
        
        size = 100
        results = {}
        
        for decomp_name, c_func, scipy_name in decompositions:
            print(f"\n📊 {decomp_name} Decomposition ({size}x{size}):")
            print("-" * 80)
            
            A = [[random.random() for _ in range(size)] for _ in range(size)]
            results[decomp_name] = {}
            
            # VectorQuant C
            if c_func:
                metric = self.measure_operation(
                    f"VectorQuant C {decomp_name}",
                    lambda: c_func(A),
                    iterations=5, rounds=5
                )
                results[decomp_name]["vectorquant_c"] = metric
                self._print_metric(metric)
            
            # SciPy
            if SCIPY_AVAILABLE:
                A_np = np.array(A)
                if decomp_name == "LU":
                    metric = self.measure_operation(
                        f"SciPy {decomp_name}",
                        lambda: scipy.linalg.lu(A_np),
                        iterations=5, rounds=5
                    )
                elif decomp_name == "QR":
                    metric = self.measure_operation(
                        f"SciPy {decomp_name}",
                        lambda: scipy.linalg.qr(A_np),
                        iterations=5, rounds=5
                    )
                elif decomp_name == "SVD":
                    metric = self.measure_operation(
                        f"SciPy {decomp_name}",
                        lambda: scipy.linalg.svd(A_np),
                        iterations=5, rounds=5
                    )
                results[decomp_name]["scipy"] = metric
                self._print_metric(metric)
            
            # Calculate speedup
            if "vectorquant_c" in results[decomp_name] and "scipy" in results[decomp_name]:
                speedup = self.compute_speedup(results[decomp_name]["scipy"], results[decomp_name]["vectorquant_c"])
                symbol = "🚀" if speedup > 1 else "🔻"
                print(f"\n{symbol} C vs SciPy: {speedup:.2f}x")
        
        self.measurements["decompositions"] = results
    
    def benchmark_covariance_scaling(self):
        """Benchmark covariance computation with varying dimensions."""
        print("\n" + "="*80)
        print("COVARIANCE MATRIX COMPUTATION - SCALING ANALYSIS")
        print("="*80)
        
        dimensions = [(10, 1000), (50, 2000), (100, 2000), (200, 3000)]
        results = {}
        
        for n_vars, n_obs in dimensions:
            print(f"\n📊 {n_vars} variables, {n_obs} observations:")
            print("-" * 80)
            
            data = [[random.random() for _ in range(n_obs)] for _ in range(n_vars)]
            dim_key = f"{n_vars}vars_{n_obs}obs"
            results[dim_key] = {}
            
            # VectorQuant C
            if C_AVAILABLE:
                metric = self.measure_operation(
                    f"VectorQuant C Cov",
                    lambda: self.c_backend.covariance_matrix(data),
                    iterations=3, rounds=5
                )
                results[dim_key]["vectorquant_c"] = metric
                self._print_metric(metric)
            
            # NumPy
            if NUMPY_AVAILABLE:
                data_np = np.array(data)
                metric = self.measure_operation(
                    f"NumPy Cov",
                    lambda: np.cov(data_np),
                    iterations=3, rounds=5
                )
                results[dim_key]["numpy"] = metric
                self._print_metric(metric)
            
            # Calculate speedup
            if "vectorquant_c" in results[dim_key] and "numpy" in results[dim_key]:
                speedup = self.compute_speedup(results[dim_key]["numpy"], results[dim_key]["vectorquant_c"])
                symbol = "🚀" if speedup > 1 else "🔻"
                print(f"\n{symbol} C vs NumPy: {speedup:.2f}x")
        
        self.measurements["covariance_scaling"] = results
    
    def benchmark_fft_scaling(self):
        """Benchmark FFT with varying sizes."""
        print("\n" + "="*80)
        print("FFT PERFORMANCE - SCALING ANALYSIS")
        print("="*80)
        
        sizes = [128, 256, 512, 1024, 2048]
        results = {}
        
        for size in sizes:
            print(f"\n📊 FFT Size: {size}")
            print("-" * 80)
            
            data = [complex(random.random(), random.random()) for _ in range(size)]
            results[f"fft_{size}"] = {}
            
            # VectorQuant C
            if C_AVAILABLE:
                metric = self.measure_operation(
                    f"VectorQuant C FFT {size}",
                    lambda: self.c_backend.radix2_fft(data),
                    iterations=5, rounds=5
                )
                results[f"fft_{size}"]["vectorquant_c"] = metric
                self._print_metric(metric)
            
            # NumPy
            if NUMPY_AVAILABLE:
                data_np = np.array(data)
                metric = self.measure_operation(
                    f"NumPy FFT {size}",
                    lambda: np.fft.fft(data_np),
                    iterations=5, rounds=5
                )
                results[f"fft_{size}"]["numpy"] = metric
                self._print_metric(metric)
            
            # Calculate speedup
            if "vectorquant_c" in results[f"fft_{size}"] and "numpy" in results[f"fft_{size}"]:
                speedup = self.compute_speedup(results[f"fft_{size}"]["numpy"], results[f"fft_{size}"]["vectorquant_c"])
                symbol = "🚀" if speedup > 1 else "🔻"
                print(f"\n{symbol} C vs NumPy: {speedup:.2f}x")
        
        self.measurements["fft_scaling"] = results
    
    def benchmark_throughput(self):
        """Benchmark throughput: operations per second."""
        print("\n" + "="*80)
        print("THROUGHPUT ANALYSIS (Operations Per Second)")
        print("="*80)
        
        size = 100
        A = [[random.random() for _ in range(size)] for _ in range(size)]
        B = [[random.random() for _ in range(size)] for _ in range(size)]
        
        print(f"\nMatrix Size: {size}x{size}")
        print("-" * 80)
        
        duration = 5.0  # Run for 5 seconds
        results = {}
        
        # VectorQuant C
        if C_AVAILABLE:
            count = 0
            start = time.perf_counter()
            while time.perf_counter() - start < duration:
                self.c_backend.matrix_multiply(A, B)
                count += 1
            
            ops_per_sec = count / duration
            results["vectorquant_c"] = ops_per_sec
            print(f"✓ VectorQuant C: {ops_per_sec:,.0f} ops/sec")
        
        # NumPy
        if NUMPY_AVAILABLE:
            A_np = np.array(A)
            B_np = np.array(B)
            count = 0
            start = time.perf_counter()
            while time.perf_counter() - start < duration:
                np.dot(A_np, B_np)
                count += 1
            
            ops_per_sec = count / duration
            results["numpy"] = ops_per_sec
            print(f"✓ NumPy: {ops_per_sec:,.0f} ops/sec")
        
        # Calculate speedup
        if "vectorquant_c" in results and "numpy" in results:
            speedup = results["numpy"] / results["vectorquant_c"]
            symbol = "🚀" if speedup > 1 else "🔻"
            print(f"\n{symbol} Throughput Ratio (NumPy/C): {speedup:.2f}x")
        
        self.measurements["throughput"] = results
    
    def _print_metric(self, metric: Dict):
        """Pretty print a metric."""
        print(f"  Mean: {metric['mean_ms']:10.6f} ms")
        print(f"  Median: {metric['median_ms']:10.6f} ms")
        print(f"  Std Dev: {metric['std_dev']:10.6f} ms")
        print(f"  Min: {metric['min_ms']:10.6f} ms")
        print(f"  Max: {metric['max_ms']:10.6f} ms")
        print(f"  P95: {metric['p95_ms']:10.6f} ms")
        print(f"  P99: {metric['p99_ms']:10.6f} ms")
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("="*80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"C Backend Available: {C_AVAILABLE}")
        print(f"NumPy Available: {NUMPY_AVAILABLE}")
        print(f"SciPy Available: {SCIPY_AVAILABLE}")
        print("="*80)
        
        # Save to JSON
        output = {
            "timestamp": datetime.now().isoformat(),
            "availability": {
                "c_backend": C_AVAILABLE,
                "numpy": NUMPY_AVAILABLE,
                "scipy": SCIPY_AVAILABLE
            },
            "measurements": self.measurements
        }
        
        with open("bench_performance_metrics.json", 'w') as f:
            json.dump(output, f, indent=2)
        
        print("\n✓ Performance metrics saved to: bench_performance_metrics.json")


def main():
    """Run advanced performance analysis."""
    print("\n" + "="*80)
    print("VectorQuant Advanced Performance Analysis")
    print("="*80)
    
    analyzer = PerformanceAnalyzer()
    
    if not C_AVAILABLE:
        print("\n⚠️  WARNING: C backend not available. Install: pip install vectorquant-c")
    
    # Run benchmarks
    analyzer.benchmark_dense_matrix_ops()
    analyzer.benchmark_decompositions()
    analyzer.benchmark_covariance_scaling()
    analyzer.benchmark_fft_scaling()
    analyzer.benchmark_throughput()
    
    # Generate report
    analyzer.generate_performance_report()


if __name__ == "__main__":
    main()
