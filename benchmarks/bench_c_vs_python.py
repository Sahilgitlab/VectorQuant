"""
VectorQuant Performance Benchmark: C vs Python
==============================================

Compares the high-performance C extension kernels against the pure-Python fallback.
"""

import time
import random
import cmath
from vectorquant.core.backend import PythonBackend, CBackend, C_AVAILABLE

def benchmark_matrix_multiply(backend, size=150):
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    B = [[random.random() for _ in range(size)] for _ in range(size)]
    
    start = time.perf_counter()
    backend.matrix_multiply(A, B)
    end = time.perf_counter()
    return end - start

def benchmark_covariance(backend, variables=100, observations=2000):
    data = [[random.random() for _ in range(observations)] for _ in range(variables)]
    
    start = time.perf_counter()
    backend.covariance_matrix(data)
    end = time.perf_counter()
    return end - start

def benchmark_gbm(backend, paths=50000):
    start = time.perf_counter()
    # s0, mu, sigma, t, dt, n_paths
    backend.simulate_gbm(100.0, 0.05, 0.2, 1.0, 1/252, paths)
    end = time.perf_counter()
    return end - start

def benchmark_fft(backend, size=1024):
    input_data = [complex(random.random(), random.random()) for _ in range(size)]
    start = time.perf_counter()
    backend.radix2_fft(input_data)
    end = time.perf_counter()
    return end - start

def benchmark_lu(backend, size=100):
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    start = time.perf_counter()
    backend.lu_decomposition(A)
    end = time.perf_counter()
    return end - start

def benchmark_cholesky(backend, size=100):
    # Create SPD matrix
    B = [[random.random() for _ in range(size)] for _ in range(size)]
    A = [[sum(B[i][k] * B[j][k] for k in range(size)) for j in range(size)] for i in range(size)]
    for i in range(size): A[i][i] += 5.0 # Ensure positive definite
    start = time.perf_counter()
    backend.cholesky_decomposition(A)
    end = time.perf_counter()
    return end - start

def benchmark_qr(backend, size=100):
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    start = time.perf_counter()
    backend.qr_decomposition(A)
    end = time.perf_counter()
    return end - start

def benchmark_eigen(backend, size=50):
    B = [[random.random() for _ in range(size)] for _ in range(size)]
    A = [[(B[i][j] + B[j][i]) for j in range(size)] for i in range(size)]
    start = time.perf_counter()
    backend.eigen_decomposition(A, num_simulations=100)
    end = time.perf_counter()
    return end - start

def benchmark_svd(backend, n=100, m=50):
    A = [[random.random() for _ in range(m)] for _ in range(n)]
    start = time.perf_counter()
    backend.svd(A)
    end = time.perf_counter()
    return end - start

def run_benchmarks():
    print("="*60)
    print(f"{'VectorQuant Performance Benchmark: C vs Python':^60}")
    print("="*60)
    
    if not C_AVAILABLE:
        print("ERROR: C backend not found. Please install vectorquant-c first.")
        return

    python_backend = PythonBackend()
    c_backend = CBackend()
    
    test_cases = [
        ("Matrix Multiply (150x150)", benchmark_matrix_multiply, {"size": 150}),
        ("Covariance Matrix (100x2000)", benchmark_covariance, {"variables": 100, "observations": 2000}),
        ("Monte Carlo GBM (50,000 paths)", benchmark_gbm, {"paths": 50000}),
        ("Radix-2 FFT (1024 samples)", benchmark_fft, {"size": 1024}),
        ("LU Decomposition (100x100)", benchmark_lu, {"size": 100}),
        ("Cholesky (100x100)", benchmark_cholesky, {"size": 100}),
        ("QR Decomposition (100x100)", benchmark_qr, {"size": 100}),
        ("Eigenvalue (50x50, 100 iter)", benchmark_eigen, {"size": 50}),
        ("SVD (100x50)", benchmark_svd, {"n": 100, "m": 50}),
    ]
    
    print(f"\n{'Operation':<35} | {'Python (s)':>10} | {'C (s)':>8} | {'Speedup':>8}")
    print("-" * 60)
    
    for title, func, kwargs in test_cases:
        # Dry run for JIT or warming up
        func(python_backend, **kwargs)
        func(c_backend, **kwargs)
        
        t_py = func(python_backend, **kwargs)
        t_c = func(c_backend, **kwargs)
        speedup = t_py / t_c if t_c > 1e-12 else float('inf')
        
        print(f"{title:<35} | {t_py:10.4f} | {t_c:8.4f} | {speedup:7.1f}x")

    print("\n" + "="*60)

if __name__ == "__main__":
    run_benchmarks()
