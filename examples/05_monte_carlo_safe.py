"""
Stochastic Module Example: Safe Monte Carlo Simulation

Demonstrates stochastic processes and Monte Carlo option pricing
with safe parameters that won't crash your PC.
"""

import math
import vectorquant as vq
from vectorquant.core.mc_config import get_safe_test_params

def main():
    print("=" * 70)
    print("VectorQuant Stochastic Module: Safe Monte Carlo Examples")
    print("=" * 70)
    print("\nNote: Using SAFE parameters to prevent PC hanging")
    print(f"Safe defaults: n_paths=1000, n_steps=50, dt=0.02\n")

    # Set seed for reproducibility
    vq.prob.set_seed(42)

    # ─── 1. STOCHASTIC PROCESSES ────────────────────────────────────────
    print("\n1. STOCHASTIC PROCESSES")
    print("-" * 70)

    params = get_safe_test_params()

    # 1A. Brownian Motion
    print("\n1A. Brownian Motion")
    paths_bm = vq.stochastic.simulate_brownian_motion(
        T=1.0, dt=params['dt'], n_paths=100
    )
    print(f"  Generated {len(paths_bm)} paths with {len(paths_bm[0])} steps")
    print(f"  First path (first 5 steps): {[f'{x:.4f}' for x in paths_bm[0][:5]]}")

    # 1B. Geometric Brownian Motion
    print("\n1B. Geometric Brownian Motion (No variance reduction)")
    paths_gbm = vq.stochastic.simulate_geometric_brownian_motion(
        S0=100, mu=0.05, sigma=0.2, T=1.0, dt=params['dt'], n_paths=100,
        antithetic=False
    )
    print(f"  Generated {len(paths_gbm)} paths")
    print(f"  First path (first 5 steps): {[f'{x:.2f}' for x in paths_gbm[0][:5]]}")
    final_prices = [path[-1] for path in paths_gbm]
    print(f"  Final price mean: ${vq.core.mean(final_prices):.2f}")
    print(f"  Final price std:  ${vq.core.standard_deviation(final_prices):.2f}")

    # 1C. GBM with Antithetic Variance Reduction
    print("\n1C. Geometric Brownian Motion (WITH variance reduction)")
    paths_gbm_anti = vq.stochastic.simulate_geometric_brownian_motion(
        S0=100, mu=0.05, sigma=0.2, T=1.0, dt=params['dt'], n_paths=50,
        antithetic=True  # Generates 100 paths (50 pairs)
    )
    print(f"  Generated {len(paths_gbm_anti)} paths (50 pairs with antithetic)")
    final_prices_anti = [path[-1] for path in paths_gbm_anti]
    print(f"  Final price mean: ${vq.core.mean(final_prices_anti):.2f}")
    print(f"  Final price std:  ${vq.core.standard_deviation(final_prices_anti):.2f}")
    print(f"  → Antithetic reduces variance by ~50%")

    # 1D. Ornstein-Uhlenbeck (Mean-reverting)
    print("\n1D. Ornstein-Uhlenbeck (Mean-reverting process)")
    paths_ou = vq.stochastic.simulate_ornstein_uhlenbeck(
        X0=0.0, theta=0.5, mu=0.0, sigma=0.1, T=1.0, dt=params['dt'], n_paths=100
    )
    print(f"  Generated {len(paths_ou)} paths")
    final_ou = [path[-1] for path in paths_ou]
    print(f"  Final value mean: {vq.core.mean(final_ou):.4f} (should be ≈ 0)")
    print(f"  Final value std:  {vq.core.standard_deviation(final_ou):.4f}")
    print(f"  → Mean-reverting: pulled back toward mean")

    # ─── 2. MONTE CARLO OPTION PRICING ────────────────────────────────
    print("\n2. MONTE CARLO OPTION PRICING")
    print("-" * 70)

    vq.prob.set_seed(42)
    
    # Create MC engine with safe path count
    mc_engine = vq.stochastic.MonteCarloEngine(n_paths=5000)

    # 2A. European Call Option
    print("\n2A. European Call Option (MC vs Black-Scholes)")
    S0, K, r, sigma, T = 100, 105, 0.05, 0.2, 1.0

    mc_price, se = mc_engine.european_call(S0=S0, K=K, r=r, sigma=sigma, T=T)
    bs_price = vq.derivatives.black_scholes_call(S0, K, r, sigma, T)

    print(f"  Spot: ${S0}, Strike: ${K}, Vol: {sigma:.1%}, T: {T}yr")
    print(f"  Monte Carlo:    ${mc_price:.2f} ± ${se:.2f}")
    print(f"  Black-Scholes:  ${bs_price:.2f}")
    print(f"  Difference:     ${abs(mc_price - bs_price):.2f}")

    # 2B. Asian Option (Path-dependent - MC only)
    print("\n2B. Asian Call Option (Path-dependent, MC only)")
    asian_price, se_asian = mc_engine.asian_call(S0=S0, K=K, r=r, sigma=sigma, T=T, dt=params['dt'])
    print(f"  Asian Call:     ${asian_price:.2f} ± ${se_asian:.2f}")
    print(f"  European Call:  ${mc_price:.2f}")
    print(f"  Asian < European? {asian_price < mc_price} (typical for call options)")

    # ─── 3. CONVERGENCE ANALYSIS ──────────────────────────────────────
    print("\n3. MONTE CARLO CONVERGENCE")
    print("-" * 70)
    print("How standard error decreases with more paths\n")

    bs_call_true = bs_price  # Use Black-Scholes as benchmark

    print(f"{'Paths':>8} {'MC Price':>12} {'Std Error':>12} {'Error vs BS':>12}")
    print("-" * 45)

    for n_paths_test in [100, 500, 1000, 2000, 5000]:
        mc_temp = vq.stochastic.MonteCarloEngine(n_paths=n_paths_test)
        vq.prob.set_seed(42)
        price_temp, se_temp = mc_temp.european_call(S0=S0, K=K, r=r, sigma=sigma, T=T)
        error_vs_bs = abs(price_temp - bs_call_true)
        print(f"{n_paths_test:>8} ${price_temp:>11.2f} ${se_temp:>11.2f} ${error_vs_bs:>11.2f}")

    print("\n" + "=" * 70)
    print("✓ Examples completed successfully!")
    print("✓ All tests used SAFE parameters - no PC hang!")
    print("=" * 70)

if __name__ == "__main__":
    main()
