"""
Finance Module Example: Portfolio Management

Markowitz optimization, risk parity, Black-Litterman, and risk attribution.
"""

import math
import vectorquant as vq

def main():
    print("=" * 70)
    print("VectorQuant Finance Module: Portfolio Examples")
    print("=" * 70)

    # Asset data
    expected_returns = [0.12, 0.10, 0.08]
    cov_matrix = [[0.04, 0.006, 0.002],
                  [0.006, 0.025, 0.004],
                  [0.002, 0.004, 0.01]]
    risk_free_rate = 0.02

    # ─── 1. BASIC PORTFOLIO CALCULATIONS ─────────────────────────────────
    print("\n1. BASIC PORTFOLIO CALCULATIONS")
    print("-" * 70)

    equal_weights = [1/3, 1/3, 1/3]
    port_ret = vq.portfolio.portfolio_return(equal_weights, expected_returns)
    port_vol = vq.portfolio.portfolio_volatility(equal_weights, cov_matrix)
    sharpe_eq = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

    print(f"Equal Weight Portfolio (33/33/33):")
    print(f"  Expected Return: {port_ret:.2%}")
    print(f"  Volatility:      {port_vol:.2%}")
    print(f"  Sharpe Ratio:    {sharpe_eq:.3f}")

    # ─── 2. MARKOWITZ OPTIMIZATION (Max Sharpe) ──────────────────────────
    print("\n2. MARKOWITZ OPTIMIZATION (Maximum Sharpe Ratio)")
    print("-" * 70)

    optimal_weights = vq.portfolio.optimize_max_sharpe(expected_returns, cov_matrix)
    opt_ret = vq.portfolio.portfolio_return(optimal_weights, expected_returns)
    opt_vol = vq.portfolio.portfolio_volatility(optimal_weights, cov_matrix)
    sharpe_opt = (opt_ret - risk_free_rate) / opt_vol if opt_vol > 0 else 0

    print(f"Optimal Weights: {[f'{w:.4f}' for w in optimal_weights]}")
    print(f"  Sum: {sum(optimal_weights):.6f} (should be 1.0)")
    print(f"Expected Return: {opt_ret:.2%}")
    print(f"Volatility:      {opt_vol:.2%}")
    print(f"Sharpe Ratio:    {sharpe_opt:.3f} (vs {sharpe_eq:.3f} equal weight)")

    # ─── 3. ALTERNATIVE ALLOCATION STRATEGIES ──────────────────────────────
    print("\n3. ALTERNATIVE ALLOCATION STRATEGIES")
    print("-" * 70)

    # 3A. 1/N (Equal Weight) vs Max Sharpe
    strategies = {
        'Equal Weight (1/N)': equal_weights,
        'Max Sharpe': optimal_weights,
    }

    print("\nComparing Allocation Strategies:")
    print(f"\n{'Strategy':<20} {'Allocation':>40}")
    print("-" * 60)
    for name, weights in strategies.items():
        weight_str = " / ".join([f'{w:.1%}' for w in weights])
        print(f"{name:<20} {weight_str:>40}")

    # ─── 4. PORTFOLIO COMPARISON TABLE ──────────────────────────────────
    print("\n4. PORTFOLIO COMPARISON")
    print("-" * 70)

    portfolios = {
        'Equal Weight': equal_weights,
        'Max Sharpe': optimal_weights,
    }

    print(f"{'Portfolio':<20} {'Return':>10} {'Vol':>10} {'Sharpe':>10}")
    print("-" * 50)
    for name, weights in portfolios.items():
        ret = vq.portfolio.portfolio_return(weights, expected_returns)
        vol = vq.portfolio.portfolio_volatility(weights, cov_matrix)
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
        print(f"{name:<20} {ret:>10.2%} {vol:>10.2%} {sharpe:>10.3f}")


    print("\n" + "=" * 70)
    print("✓ Examples completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
