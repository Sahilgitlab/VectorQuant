"""
VectorQuant Benchmark Suite
===========================

Compares pure-Python implementation vs Rust-accelerated core.
"""

import time
import random
from vectorquant.core.backend import PythonBackend, RustBackend, RUST_AVAILABLE

def benchmark_matrix_multiply(backend, size=100):
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    B = [[random.random() for _ in range(size)] for _ in range(size)]
    
    start = time.perf_counter()
    backend.matrix_multiply(A, B)
    end = time.perf_counter()
    return end - start

def benchmark_covariance(backend, variables=10, observations=1000):
    data = [[random.random() for _ in range(observations)] for _ in range(variables)]
    
    start = time.perf_counter()
    backend.covariance_matrix(data)
    end = time.perf_counter()
    return end - start

def benchmark_gbm(backend, paths=10000):
    start = time.perf_counter()
    backend.simulate_gbm(100.0, 0.05, 0.2, 1.0, 1/252, paths)
    end = time.perf_counter()
    return end - start

def run_benchmarks():
    print("="*40)
    print("VectorQuant Performance Benchmark")
    print("="*40)
    
    backends = [("Python", PythonBackend())]
    if RUST_AVAILABLE:
        backends.append(("Rust", RustBackend()))
    else:
        print("Note: Rust backend not found. Testing Python only.")
    
    for name, backend in backends:
        print(f"\n--- Backend: {name} ---")
        
        # Matrix Mult
        t_mm = benchmark_matrix_multiply(backend, size=100)
        print(f"Matrix Multiply (100x100): {t_mm:.4f}s")
        
        # Covariance
        t_cov = benchmark_covariance(backend, variables=50, observations=1000)
        print(f"Covariance Matrix (50x1000): {t_cov:.4f}s")
        
        # GBM
        t_gbm = benchmark_gbm(backend, paths=10000)
        print(f"Monte Carlo GBM (10,000 paths): {t_gbm:.4f}s")

if __name__ == "__main__":
    run_benchmarks()
