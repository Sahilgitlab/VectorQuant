"""
VectorQuant GPU Acceleration Benchmark

Tests the pure Python/JIT implementations against the CuPy GPU 
arrays (if installed) on a massive options pricing grid.
"""

import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq
from vectorquant.core.config import CUPY_AVAILABLE

def run_benchmark():
    print("="*60)
    print("  VectorQuant GPU Monte Carlo Benchmark (GBM)")
    print("  Comparing CPU Fallback vs GPU Acceleration")
    print("="*60)
    
    # Base params
    S0 = 100.0
    K = 100.0
    T = 1.0
    dt = 1/252
    r = 0.05
    sigma = 0.2
    
    paths_to_test = [10_000, 50_000, 100_000]
    if CUPY_AVAILABLE:
        print("  [✓] CuPy Detected! Adding massive GPU stress test (1 Million paths).")
        paths_to_test.append(1_000_000)
    else:
        print("  [x] CuPy missing. Benchmarking CPU only. (Install vectorquant[gpu] for CUDA speeds).")
    
    print("\n{:>10} {:>12} {:>15}".format(
        "Paths", "Tier", "Time (s)"
    ))
    print("-" * 50)
    
    for n_paths in paths_to_test:
        # --- CPU (JIT Accelerated) ---
        if n_paths <= 100_000: # Python loops start hanging above 100k
            engine_cpu = vq.stochastic.MonteCarloEngine(n_paths=n_paths, gpu=False)
            t0 = time.time()
            engine_cpu.european_call(S0, K, r, sigma, T)
            t1 = time.time()
            print("{:>10,d} {:>12} {:>15.4f}".format(n_paths, "CPU (numba)", t1 - t0))
        
        # --- GPU (CuPy array math) ---
        if CUPY_AVAILABLE:
            engine_gpu = vq.stochastic.MonteCarloEngine(n_paths=n_paths, gpu=True)
            # Warm up JIT/CUDA Kernels
            if n_paths == paths_to_test[0]:
               engine_gpu.european_call(S0, K, r, sigma, T)
               
            t0 = time.time()
            engine_gpu.european_call(S0, K, r, sigma, T)
            t1 = time.time()
            print("{:>10,d} {:>12} {:>15.4f}".format(n_paths, "GPU (cupy)", t1 - t0))
            
    print("="*60)
    
if __name__ == "__main__":
    run_benchmark()
