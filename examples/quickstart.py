"""
VectorQuant — Quickstart Example
==================================

End-to-end demonstration of the VectorQuant API.

Usage:
    python examples/quickstart.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq

print("=" * 60)
print(f"  VectorQuant v{vq.__version__} — Quickstart")
print("=" * 60)
print()

# ─── 1. Generate synthetic price data ───────────────────────────────────────
vq.prob.set_seed(42)
prices = [100.0]
for _ in range(252):
    ret = vq.prob.rnorm(mu=0.0003, sigma=0.015)
    prices.append(prices[-1] * (1 + ret))

# ─── 2. Compute returns ────────────────────────────────────────────────────
returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

print("📊 Sample Statistics")
print(f"   Mean daily return:  {vq.stats.mean(returns):.6f}")
print(f"   Std deviation:      {vq.stats.standard_deviation(returns):.6f}")
print(f"   Skewness:           {vq.stats.skewness(returns):.4f}")
print(f"   Kurtosis:           {vq.stats.kurtosis(returns):.4f}")
print()

# ─── 3. Risk measures ──────────────────────────────────────────────────────
var_95 = vq.risk.parametric_var(returns, 0.95)
var_99 = vq.risk.parametric_var(returns, 0.99)
cvar_95 = vq.risk.cvar(returns, 0.95)

print("⚠️  Risk Metrics")
print(f"   VaR (95%):   {var_95:.4%}")
print(f"   VaR (99%):   {var_99:.4%}")
print(f"   CVaR (95%):  {cvar_95:.4%}")
print()

# ─── 4. Black-Scholes pricing ──────────────────────────────────────────────
call_price = vq.derivatives.black_scholes_call(S=prices[-1], K=100, r=0.05, sigma=0.20, T=0.25)
put_price = vq.derivatives.black_scholes_put(S=prices[-1], K=100, r=0.05, sigma=0.20, T=0.25)
delta = vq.derivatives.bs_delta(S=prices[-1], K=100, r=0.05, sigma=0.20, T=0.25)

print("📈 Options Pricing (Current price → Strike 100, 3mo)")
print(f"   Current price:  ${prices[-1]:.2f}")
print(f"   Call price:     ${call_price:.2f}")
print(f"   Put price:      ${put_price:.2f}")
print(f"   Call delta:     {delta:.4f}")
print()

# ─── 5. Monte Carlo simulation ─────────────────────────────────────────────
vq.prob.set_seed(42)
mc_paths = vq.stochastic.simulate_geometric_brownian_motion(
    S0=prices[-1], mu=0.05, sigma=0.20, T=1.0, dt=1/252, n_paths=1000
)
final_prices = [path[-1] for path in mc_paths]

print("🎲 Monte Carlo Simulation (1000 paths, 1 year)")
print(f"   Mean final price:   ${vq.stats.mean(final_prices):.2f}")
print(f"   Median final price: ${vq.stats.median(final_prices):.2f}")
print(f"   Std of finals:      ${vq.stats.standard_deviation(final_prices):.2f}")
print()

# ─── 6. QuantEngine workflow ───────────────────────────────────────────────
# Create multi-asset returns
vq.prob.set_seed(42)
multi_returns = []
for _ in range(252):
    row = [vq.prob.rnorm(0.0003, 0.015),
           vq.prob.rnorm(0.0002, 0.012),
           vq.prob.rnorm(0.0001, 0.008)]
    multi_returns.append(row)

engine = vq.QuantEngine()
engine.load_returns(multi_returns)
engine.compute_covariance()
engine.compute_risk()
engine.optimize_portfolio()
summary = engine.summary()

print("🚀 QuantEngine — Optimal Portfolio")
print(f"   Weights:          {[f'{w:.4f}' for w in summary['weights']]}")
print(f"   Expected return:  {summary['expected_return']:.6f}")
print(f"   Volatility:       {summary['volatility']:.6f}")
print(f"   VaR (95%):        {summary['var']:.4%}")
print()

print("=" * 60)
print("  ✅ VectorQuant is ready!")
print("=" * 60)
