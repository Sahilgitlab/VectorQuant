"""
VectorQuant Detailed Speedup Analysis
======================================

Provides detailed speedup analysis, comparative matrices, and performance insights.
Generates formatted comparison tables and detailed performance breakdown.
"""

import time
import random
import json
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

from vectorquant.core.backend import C_AVAILABLE, CBackend, PythonBackend

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import scipy.linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class DetailedSpeedupAnalyzer:
    """Detailed speedup analysis with formatted comparison tables."""
    
    def __init__(self):
        self.c_backend = CBackend() if C_AVAILABLE else None
        self.python_backend = PythonBackend()
        self.comparison_matrix = defaultdict(dict)
    
    def measure_ms(self, fn, iterations=1):
        """Measure execution time in milliseconds."""
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        return (time.perf_counter() - start) * 1000 / iterations
    
    def print_comparison_table(self, test_name: str, results: Dict[str, float]):
        """Print formatted comparison table."""
        if not results:
            return
        
        # Sort by time (fastest first)
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        
        print(f"\n{'Test: ' + test_name:^80}")
        print("┌" + "─" * 78 + "┐")
        print(f"│ {'Implementation':<25} {'Time (ms)':<15} {'Speedup vs C':<15} {'Status':<20} │")
        print("├" + "─" * 78 + "┤")
        
        baseline = results.get("vectorquant_c", float('inf'))
        
        for impl, time_ms in sorted_results:
            speedup = baseline / time_ms if impl != "vectorquant_c" else 1.0
            
            if impl == "vectorquant_c":
                status = "🚀 BASELINE"
                speedup_str = "—"
            elif speedup > 1.2:
                status = "⚡ C FASTER"
                speedup_str = f"{speedup:.2f}x"
            elif speedup > 0.8:
                status = "≈ SIMILAR"
                speedup_str = f"{speedup:.2f}x"
            else:
                status = "🐢 SLOWER"
                speedup_str = f"{speedup:.2f}x"
            
            print(f"│ {impl:<25} {time_ms:>13.6f} {speedup_str:>14} {status:>20} │")
        
        print("└" + "─" * 78 + "┘")
        
        self.comparison_matrix[test_name] = results
    
    def analyze_scaling_behavior(self):
        """Analyze how performance scales with problem size."""
        print("\n" + "="*80)
        print("SCALING BEHAVIOR ANALYSIS")
        print("="*80)
        print("How execution time scales with problem size\n")
        
        sizes = [50, 100, 150, 200]
        c_times = []
        numpy_times = []
        
        print(f"{'Size':<10} {'VectorQuant C (ms)':<20} {'NumPy (ms)':<20} {'Ratio':<10}")
        print("-" * 80)
        
        for size in sizes:
            A = [[random.random() for _ in range(size)] for _ in range(size)]
            B = [[random.random() for _ in range(size)] for _ in range(size)]
            
            # VectorQuant C
            if C_AVAILABLE:
                c_time = self.measure_ms(lambda: self.c_backend.matrix_multiply(A, B), iterations=3)
                c_times.append(c_time)
            else:
                c_time = 0
            
            # NumPy
            if NUMPY_AVAILABLE:
                A_np = np.array(A)
                B_np = np.array(B)
                np_time = self.measure_ms(lambda: np.dot(A_np, B_np), iterations=3)
                numpy_times.append(np_time)
            else:
                np_time = 0
            
            ratio = np_time / c_time if c_time > 0 else 0
            print(f"{size:<10} {c_time:<20.6f} {np_time:<20.6f} {ratio:<10.2f}x")
        
        # Analyze scaling pattern
        if len(c_times) >= 2:
            print("\nScaling Analysis (time ratio between consecutive sizes):")
            print("-" * 80)
            for i in range(1, len(c_times)):
                ratio = c_times[i] / c_times[i-1]
                size_ratio = (sizes[i] / sizes[i-1]) ** 3  # Expected for O(n³)
                print(f"Size {sizes[i-1]}→{sizes[i]}: C={ratio:.2f}x (expected {size_ratio:.2f}x for O(n³))")
    
    def analyze_algorithm_efficiency(self):
        """Analyze computational efficiency of algorithms."""
        print("\n" + "="*80)
        print("ALGORITHM EFFICIENCY ANALYSIS")
        print("="*80)
        print("Operations count vs actual execution time\n")
        
        # Matrix multiplication: n³ operations
        print("Matrix Multiplication (MatMul) - O(n³) algorithm")
        print("-" * 80)
        
        test_configs = [
            (50, "Small"),
            (100, "Medium"),
            (200, "Large")
        ]
        
        for size, label in test_configs:
            A = [[random.random() for _ in range(size)] for _ in range(size)]
            B = [[random.random() for _ in range(size)] for _ in range(size)]
            
            ops_gflops = (2 * size**3) / 1e9  # GigaFLOPs
            
            if C_AVAILABLE:
                time_ms = self.measure_ms(lambda: self.c_backend.matrix_multiply(A, B), iterations=3)
                time_sec = time_ms / 1000
                gflops = ops_gflops / time_sec if time_sec > 0 else 0
                
                print(f"  {label:8} ({size}×{size}): {ops_gflops:.2f} GFLOPs theoretical, "
                      f"{gflops:.2f} GFLOPs achieved ({gflops/ops_gflops*100:.1f}% utilization)")
    
    def analyze_cache_behavior(self):
        """Analyze cache efficiency."""
        print("\n" + "="*80)
        print("CACHE BEHAVIOR ANALYSIS")
        print("="*80)
        print("Working set size vs execution time\n")
        
        sizes = [50, 100, 200, 500]
        
        print(f"{'Size':<10} {'Memory (KB)':<15} {'C Time (ms)':<15} {'Efficiency':<15}")
        print("-" * 80)
        
        for size in sizes:
            # Memory in KB for 4 matrices (A, B, C, temp)
            memory_kb = (4 * size * size * 8) / 1024
            
            A = [[random.random() for _ in range(size)] for _ in range(size)]
            B = [[random.random() for _ in range(size)] for _ in range(size)]
            
            if C_AVAILABLE:
                time_ms = self.measure_ms(lambda: self.c_backend.matrix_multiply(A, B), iterations=1)
                # Efficiency: lower is better (less time for same memory)
                efficiency = memory_kb / time_ms if time_ms > 0 else 0
                
                print(f"{size:<10} {memory_kb:>13.1f} {time_ms:>14.6f} {efficiency:>14.2f}")
        
        print("\nNote: Higher efficiency is better for cache utilization")
    
    def comprehensive_operation_comparison(self):
        """Comprehensive comparison of all operations."""
        print("\n" + "="*80)
        print("COMPREHENSIVE OPERATION COMPARISON")
        print("="*80)
        
        operations = []
        
        # Matrix operations
        if C_AVAILABLE:
            # LU Decomposition
            A_lu = [[random.random() for _ in range(100)] for _ in range(100)]
            c_time = self.measure_ms(lambda: self.c_backend.lu_decomposition(A_lu), iterations=1)
            operations.append(("LU Decomposition (100×100)", c_time, "vectorquant_c"))
            
            if SCIPY_AVAILABLE:
                A_np = np.array(A_lu)
                np_time = self.measure_ms(lambda: scipy.linalg.lu(A_np), iterations=1)
                operations.append(("LU Decomposition (100×100)", np_time, "scipy"))
            
            # QR Decomposition
            A_qr = [[random.random() for _ in range(100)] for _ in range(100)]
            c_time = self.measure_ms(lambda: self.c_backend.qr_decomposition(A_qr), iterations=1)
            operations.append(("QR Decomposition (100×100)", c_time, "vectorquant_c"))
            
            if SCIPY_AVAILABLE:
                A_np = np.array(A_qr)
                np_time = self.measure_ms(lambda: scipy.linalg.qr(A_np), iterations=1)
                operations.append(("QR Decomposition (100×100)", np_time, "scipy"))
            
            # Covariance
            data_cov = [[random.random() for _ in range(2000)] for _ in range(100)]
            c_time = self.measure_ms(lambda: self.c_backend.covariance_matrix(data_cov), iterations=1)
            operations.append(("Covariance (100×2000)", c_time, "vectorquant_c"))
            
            if NUMPY_AVAILABLE:
                data_np = np.array(data_cov)
                np_time = self.measure_ms(lambda: np.cov(data_np), iterations=1)
                operations.append(("Covariance (100×2000)", np_time, "numpy"))
            
            # GBM
            c_time = self.measure_ms(
                lambda: self.c_backend.simulate_gbm(100.0, 0.05, 0.2, 1.0, 1/252, 50000),
                iterations=1
            )
            operations.append(("Monte Carlo GBM (50k)", c_time, "vectorquant_c"))
        
        # Print formatted table
        print(f"\n{'Operation':<35} {'Impl':<15} {'Time (ms)':<15}")
        print("-" * 80)
        
        grouped = defaultdict(list)
        for op, time, impl in operations:
            grouped[op].append((impl, time))
        
        for op, results in grouped.items():
            results_dict = {impl: time for impl, time in results}
            self.print_comparison_table(op, results_dict)
    
    def generate_insights(self):
        """Generate key performance insights."""
        print("\n" + "="*80)
        print("KEY PERFORMANCE INSIGHTS")
        print("="*80)
        
        insights = []
        
        if C_AVAILABLE:
            insights.append("✓ VectorQuant C backend is optimized for:")
            insights.append("  • Covariance computation (parallel column-wise)")
            insights.append("  • Stochastic simulation (vectorized RNG)")
            insights.append("  • Large-scale matrix operations (cache-aware blocking)")
        
        if NUMPY_AVAILABLE:
            insights.append("\n✓ NumPy excels at:")
            insights.append("  • Small to medium matrix operations (BLAS)")
            insights.append("  • Vectorized array operations")
            insights.append("  • Memory-efficient broadcasting")
        
        if SCIPY_AVAILABLE:
            insights.append("\n✓ SciPy adds:")
            insights.append("  • Specialized decomposition algorithms")
            insights.append("  • Advanced numerical methods")
            insights.append("  • Statistical distributions")
        
        insights.extend([
            "\n💡 Recommendations:",
            "  • Use VectorQuant C for high-frequency risk computations",
            "  • Use NumPy for prototyping and exploratory analysis",
            "  • Use SciPy for specialized mathematical operations",
            "  • Combine approaches: prototype in NumPy, deploy in VectorQuant C"
        ])
        
        for insight in insights:
            print(insight)
    
    def benchmark_report(self) -> Dict:
        """Generate comprehensive report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "comparison_matrix": dict(self.comparison_matrix),
            "insights": {
                "c_available": C_AVAILABLE,
                "numpy_available": NUMPY_AVAILABLE,
                "scipy_available": SCIPY_AVAILABLE,
            }
        }


def main():
    """Run detailed speedup analysis."""
    print("\n" + "="*80)
    print("VectorQuant Detailed Speedup Analysis")
    print("="*80)
    
    analyzer = DetailedSpeedupAnalyzer()
    
    if not C_AVAILABLE:
        print("\n⚠️  WARNING: C backend not available. Install: pip install vectorquant-c")
        return
    
    # Run analyses
    analyzer.analyze_scaling_behavior()
    analyzer.analyze_algorithm_efficiency()
    analyzer.analyze_cache_behavior()
    analyzer.comprehensive_operation_comparison()
    analyzer.generate_insights()
    
    # Save report
    report = analyzer.benchmark_report()
    with open("bench_speedup_analysis.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n✓ Speedup analysis saved to: bench_speedup_analysis.json")


if __name__ == "__main__":
    main()
