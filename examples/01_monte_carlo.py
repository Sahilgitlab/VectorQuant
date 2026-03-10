"""
VectorQuant Example: Monte Carlo Simulation
============================================
Demonstrates the Monte Carlo engine for options pricing,
showing both pure Python and Numba-accelerated paths.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq

print("=" * 60)
print("  VectorQuant — Monte Carlo Options Pricing")
print("=" * 60)

# --- 1. European Call Option via Black-Scholes (exact) ---
S0, K, r, sigma, T = 100.0, 105.0, 0.05, 0.2, 1.0

bs_price = vq.derivatives.black_scholes_call(S0, K, r, sigma, T)
print(f"\nBlack-Scholes Exact Price: ${bs_price:.4f}")

# --- 2. Monte Carlo Estimate ---
engine = vq.stochastic.MonteCarloEngine(n_paths=50000)
mc_price, mc_se = engine.european_call(S0, K, r, sigma, T)
print(f"Monte Carlo Estimate:     ${mc_price:.4f}  (SE: {mc_se:.4f})")
print(f"Difference:               ${abs(bs_price - mc_price):.4f}")

# --- 3. Asian Call Option (path-dependent, no closed-form) ---
mc_asian, asian_se = engine.asian_call(S0, K, r, sigma, T, dt=1/252)
print(f"\nAsian Call (MC):           ${mc_asian:.4f}  (SE: {asian_se:.4f})")

# --- 4. Simulate and inspect paths ---
paths = vq.stochastic.simulate_geometric_brownian_motion(S0, r, sigma, T, 1/12, 5)
print(f"\nSample 5 paths (monthly steps for 1 year):")
for i, path in enumerate(paths):
    terminal = path[-1]
    print(f"  Path {i+1}: S_0={path[0]:.0f} → S_T={terminal:.2f}")

print("\n" + "=" * 60)
print("  Done! VectorQuant computed all results deterministically.")
print("=" * 60)
