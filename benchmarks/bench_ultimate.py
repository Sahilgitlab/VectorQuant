"""
VectorQuant Ultimate Benchmark Suite: Comprehensive Comparison
==============================================================
Comparing VectorQuant against NumPy, SciPy, and QuantLib using real data.
"""

import time
import random
import math
import json
import sys
from datetime import datetime

# Industry Standards
import numpy as np
import scipy.linalg
import scipy.stats
import scipy.fft
import QuantLib as ql

# VectorQuant
from vectorquant.core.backend import CBackend, PythonBackend, C_AVAILABLE
import vectorquant.ai as vq_ai
import vectorquant.finance.risk_models as vq_risk
import vectorquant.finance.derivatives as vq_deriv
import vectorquant.stochastic.processes as vq_stoch

# Real Market Data (Last ~20 days closing prices as of March 2026)
AAPL_DATA = [260.83, 259.88, 257.46, 260.29, 262.52, 263.75, 264.72, 264.18, 272.95, 274.23, 272.14, 266.18, 264.58, 260.58, 264.35, 263.88, 255.78, 261.73]
MSFT_DATA = [404.42, 404.88, 405.76, 409.41, 408.96, 410.68, 405.20, 403.93, 398.55, 392.74, 401.72, 400.60, 389.00, 384.47, 397.23, 398.46, 399.60, 396.86, 401.32, 401.84]
SPY_DATA = [668.89, 676.33, 677.18, 678.27, 672.38, 681.31, 685.13, 680.33, 686.38, 685.99, 689.30, 693.15, 687.35, 682.39, 689.43, 684.48, 686.29, 682.85, 681.75, 681.27]

def calculate_returns(prices):
    return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

AAPL_RET = calculate_returns(AAPL_DATA)
MSFT_RET = calculate_returns(MSFT_DATA)
SPY_RET = calculate_returns(SPY_DATA)

class UltimateBenchmark:
    def __init__(self):
        self.results = {}
        self.c_backend = CBackend() if C_AVAILABLE else PythonBackend()
        print(f"Using Backend: {'C' if C_AVAILABLE else 'Python'}")

    @staticmethod
    def timer(func, *args, iterations=100, **kwargs):
        start = time.perf_counter()
        for _ in range(iterations):
            func(*args, **kwargs)
        return (time.perf_counter() - start) / iterations * 1000 # returns ms

    # --- STAGE 1: MATH KERNEL ---
    def bench_math(self):
        print("Benchmarking Stage 1: Math Kernels...")
        size = 100
        A = [[random.random() for _ in range(size)] for _ in range(size)]
        A_np = np.array(A)

        # MatMul
        vq_time = UltimateBenchmark.timer(self.c_backend.matrix_multiply, A, A, iterations=10)
        np_time = UltimateBenchmark.timer(np.dot, A_np, A_np, iterations=10)
        
        # SVD
        vq_svd = UltimateBenchmark.timer(self.c_backend.svd, A, iterations=5)
        np_svd = UltimateBenchmark.timer(scipy.linalg.svd, A_np, iterations=5)

        self.results['Stage 1: Math'] = {
            'Matrix Mul (100x100)': {'VectorQuant': vq_time, 'NumPy': np_time, 'Unit': 'ms'},
            'SVD (100x100)': {'VectorQuant': vq_svd, 'SciPy': np_svd, 'Unit': 'ms'}
        }

    # --- STAGE 2: FINANCE & REAL DATA ---
    def bench_finance(self):
        print("Benchmarking Stage 2: Finance & Real Data...")
        
        # VaR on SPY Real Data
        vq_var = UltimateBenchmark.timer(vq_risk.parametric_var, SPY_RET, 0.95, iterations=500)
        
        # NumPy equivalent VaR
        def np_var(ret, conf):
            return -np.percentile(ret, (1 - conf) * 100)
        np_var_time = UltimateBenchmark.timer(np_var, SPY_RET, 0.95, iterations=500)

        # Options Pricing (Black-Scholes) vs QuantLib
        S, K, r, sigma, T = 100.0, 105.0, 0.05, 0.2, 1.0
        vq_put = UltimateBenchmark.timer(vq_deriv.black_scholes_put, S, K, r, sigma, T, iterations=1000)

        def ql_put(S, K, r, sigma, T):
            today = ql.Date(13, 3, 2026)
            ql.Settings.instance().evaluationDate = today
            payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
            exercise = ql.EuropeanExercise(today + int(T*365))
            option = ql.VanillaOption(payoff, exercise)
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
            rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
            vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed()))
            process = ql.BlackScholesProcess(spot_handle, rate_handle, vol_handle)
            option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
            return option.NPV()
        
        ql_time = UltimateBenchmark.timer(ql_put, S, K, r, sigma, T, iterations=50) # QL is heavy on setup

        self.results['Stage 2: Finance'] = {
            'Parametric VaR (Real SPY Data)': {'VectorQuant': vq_var, 'NumPy': np_var_time, 'Unit': 'ms'},
            'Black-Scholes Put': {'VectorQuant': vq_put, 'QuantLib': ql_time, 'Unit': 'ms'}
        }

    # --- STAGE 3: STOCHASTIC SIMULATION ---
    def bench_stochastic(self):
        print("Benchmarking Stage 3: Stochastic Simulations...")
        paths = 10000
        vq_gbm = UltimateBenchmark.timer(self.c_backend.simulate_gbm, 100.0, 0.05, 0.2, 1.0, 1/252, paths, iterations=5)

        def np_gbm(paths):
            steps = 252
            dt = 1/252
            Z = np.random.standard_normal((paths, steps))
            drift = (0.05 - 0.5 * 0.2**2) * dt
            vol = 0.2 * math.sqrt(dt)
            return 100.0 * np.exp(np.cumsum(drift + vol * Z, axis=1))
        
        np_gbm_time = UltimateBenchmark.timer(np_gbm, paths, iterations=5)

        self.results['Stage 3: Stochastic'] = {
            'GBM Path Gen (10k paths)': {'VectorQuant': vq_gbm, 'NumPy': np_gbm_time, 'Unit': 'ms'}
        }

    # --- STAGE 4: AI VERIFICATION ---
    def bench_ai(self):
        print("Benchmarking Stage 4: AI Verification...")
        
        # Benchmark explainability/trace generation overhead
        vq_trace = UltimateBenchmark.timer(vq_ai.explain_var, SPY_RET, 0.95, iterations=10)
        
        # Benchmark hallucination check speed
        def mock_hallucination_check():
            return vq_ai.verify_calculation("sqrt(16) * 2", expected=8.0)
        
        vq_verify = UltimateBenchmark.timer(mock_hallucination_check, iterations=100)

        self.results['Stage 4: AI Verification'] = {
            'VaR Proof Trace Gen': {'VectorQuant': vq_trace, 'Baseline': 0.0, 'Unit': 'ms'},
            'Hallucination Check': {'VectorQuant': vq_verify, 'Baseline': 0.0, 'Unit': 'ms'}
        }

    def run_all(self):
        self.bench_math()
        self.bench_finance()
        self.bench_stochastic()
        self.bench_ai()
        
        with open('bench_ultimate_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        print("\nBenchmarks completed. Results saved to bench_ultimate_results.json")

if __name__ == "__main__":
    UltimateBenchmark().run_all()
