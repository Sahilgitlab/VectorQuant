"""
VectorQuant — Portfolio Optimization Benchmark
==============================================

Benchmarks the Max-Sharpe optimizer.

Usage:
    python benchmarks/bench_portfolio_optimization.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq

def benchmark_optimization(n_assets):
    # Create random expected returns and a diagonal/mock covariance
    vq.prob.set_seed(42)
    expected_returns = [vq.prob.rnorm(0.05, 0.02) for _ in range(n_assets)]
    cov = [[0.04 if i == j else 0.005 for j in range(n_assets)] for i in range(n_assets)]
    
    start = time.perf_counter()
    weights = vq.portfolio.optimize_max_sharpe(expected_returns, cov)
    elapsed = time.perf_counter() - start
    return elapsed, weights

def main():
    print("=" * 60)
    print("  VectorQuant Portfolio Optimization Benchmark")
    print("  Max Sharpe Ratio (Gradient Descent)")
    print("=" * 60)
    print()
    print(f"{'Assets':>10}  {'Time (s)':>10}")
    print("-" * 40)

    for n_assets in [5, 10, 15, 20]:
        elapsed, _ = benchmark_optimization(n_assets)
        print(f"{n_assets:>10}  {elapsed:>10.4f}")

if __name__ == "__main__":
    main()
