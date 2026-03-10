"""
VectorQuant — Covariance Benchmark
==================================

Benchmarks large matrix covariance calculation directly.

Usage:
    python benchmarks/bench_covariance.py
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq

def generate_returns(n_obs, n_assets):
    vq.prob.set_seed(42)
    return [[random.gauss(0, 0.02) for _ in range(n_assets)] for _ in range(n_obs)]

def benchmark_covariance(n_obs, n_assets, method="sample"):
    returns = generate_returns(n_obs, n_assets)
    engine = vq.QuantEngine()
    engine.load_returns(returns)
    
    start = time.perf_counter()
    engine.compute_covariance(method)
    elapsed = time.perf_counter() - start
    return elapsed

def main():
    print("=" * 60)
    print("  VectorQuant Covariance Calculation Benchmark")
    print("=" * 60)
    print()
    print(f"{'Assets':>10}  {'Obs':>8}  {'Method':>15}  {'Time (s)':>10}")
    print("-" * 50)

    for n_assets in [10, 25, 50, 100]:
        n_obs = 1000
        for method in ["sample", "ledoit_wolf"]:
            elapsed = benchmark_covariance(n_obs, n_assets, method)
            print(f"{n_assets:>10}  {n_obs:>8}  {method:>15}  {elapsed:>10.4f}")

if __name__ == "__main__":
    main()
