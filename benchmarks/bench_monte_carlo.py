"""
VectorQuant — Monte Carlo Benchmark
====================================

Benchmarks GBM simulation across different path counts.

Usage:
    python benchmarks/bench_monte_carlo.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq


def benchmark_gbm(n_paths, S0=100, mu=0.05, sigma=0.2, T=1.0, dt=1/252):
    """Run a GBM simulation and time it."""
    vq.prob.set_seed(42)
    start = time.perf_counter()
    paths = vq.stochastic.simulate_geometric_brownian_motion(
        S0=S0, mu=mu, sigma=sigma, T=T, dt=dt, n_paths=n_paths
    )
    elapsed = time.perf_counter() - start
    return elapsed, len(paths), len(paths[0])


def main():
    print("=" * 60)
    print("  VectorQuant Monte Carlo Benchmark")
    print("  GBM Simulation (252 daily steps, 1yr)")
    print("=" * 60)
    print()
    print(f"{'Paths':>10}  {'Time (s)':>10}  {'Steps':>8}  {'Paths/s':>12}")
    print("-" * 50)

    for n_paths in [100, 500, 1_000, 5_000, 10_000]:
        elapsed, actual_paths, steps = benchmark_gbm(n_paths)
        rate = n_paths / elapsed if elapsed > 0 else float('inf')
        print(f"{n_paths:>10,}  {elapsed:>10.4f}  {steps:>8}  {rate:>12,.0f}")

    print()
    print("=" * 60)
    print(f"  Version: {vq.__version__}")
    print("=" * 60)


if __name__ == "__main__":
    main()
