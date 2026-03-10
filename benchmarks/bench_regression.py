"""
VectorQuant — Regression Benchmark
==================================

Benchmarks Ordinary Least Squares (OLS) and Ridge Regression.

Usage:
    python benchmarks/bench_regression.py
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq

def generate_data(n_obs, n_features):
    vq.prob.set_seed(42)
    # Generate X
    X = [[random.gauss(0, 1) for _ in range(n_features)] for _ in range(n_obs)]
    # True weights
    w_true = [random.gauss(0, 5) for _ in range(n_features)]
    # Generate Y = X * w + noise
    Y = []
    for i in range(n_obs):
        y_val = sum(X[i][j] * w_true[j] for j in range(n_features)) + random.gauss(0, 0.5)
        Y.append(y_val)
    return X, Y

def benchmark_regression(n_obs, n_features, method="ols"):
    X, Y = generate_data(n_obs, n_features)
    
    start = time.perf_counter()
    if method == "ols":
        weights = vq.stats.linear_regression(X, Y)
    elif method == "ridge":
        weights = vq.stats.ridge_regression(X, Y, lmbda=1.0)
    elapsed = time.perf_counter() - start
    return elapsed

def main():
    print("=" * 60)
    print("  VectorQuant Regression Benchmark")
    print("=" * 60)
    print()
    print(f"{'Features':>10}  {'Obs':>8}  {'Method':>15}  {'Time (s)':>10}")
    print("-" * 50)

    for n_features in [5, 10, 20]:
        n_obs = 5000
        for method in ["ols", "ridge"]:
            elapsed = benchmark_regression(n_obs, n_features, method)
            print(f"{n_features:>10}  {n_obs:>8}  {method:>15}  {elapsed:>10.4f}")

if __name__ == "__main__":
    main()
