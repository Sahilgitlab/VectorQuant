"""
VectorQuant Example: Portfolio Optimization & Risk
===================================================
Demonstrates portfolio construction, optimization, and risk analysis.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq

print("=" * 60)
print("  VectorQuant — Portfolio Optimization & Risk Analysis")
print("=" * 60)

# --- Synthetic 3-asset returns data (each inner list = one time period) ---
# Transpose: we need returns_matrix[t][asset]
returns_matrix = [
    [0.010, 0.005, 0.002],
    [0.020, -0.010, 0.008],
    [-0.005, 0.015, 0.001],
    [0.008, 0.003, 0.006],
    [0.015, -0.005, 0.004],
    [-0.010, 0.020, 0.003],
    [0.012, 0.008, 0.005],
    [0.003, -0.002, 0.007],
]

n_assets = len(returns_matrix[0])
n_periods = len(returns_matrix)

# --- 1. Compute per-asset statistics ---
asset_returns = []
for j in range(n_assets):
    col = [returns_matrix[t][j] for t in range(n_periods)]
    asset_returns.append(col)

expected_returns = [vq.stats.mean(col) for col in asset_returns]
print(f"\nExpected Returns: {[round(r, 6) for r in expected_returns]}")

# --- 2. Manual covariance matrix (3x3) ---
cov = [[0.0] * n_assets for _ in range(n_assets)]
for i in range(n_assets):
    for j in range(n_assets):
        mu_i = expected_returns[i]
        mu_j = expected_returns[j]
        cov[i][j] = sum(
            (asset_returns[i][t] - mu_i) * (asset_returns[j][t] - mu_j)
            for t in range(n_periods)
        ) / (n_periods - 1)

print("\nCovariance Matrix:")
for row in cov:
    print("  " + "  ".join(f"{v:10.7f}" for v in row))

# --- 3. Optimize for max Sharpe ---
weights = vq.portfolio.optimize_max_sharpe(expected_returns, cov)
print(f"\nOptimal Weights:  {[round(w, 4) for w in weights]}")

# --- 4. Risk analysis ---
portfolio_returns = []
for t in range(n_periods):
    pr = sum(weights[j] * returns_matrix[t][j] for j in range(n_assets))
    portfolio_returns.append(pr)

var_95 = vq.risk.parametric_var(portfolio_returns, 0.95)
cvar_95 = vq.risk.cvar(portfolio_returns, 0.95)
sharpe = vq.stats.mean(portfolio_returns) / vq.stats.standard_deviation(portfolio_returns)

print(f"\nPortfolio Metrics:")
print(f"  VaR (95%):      {var_95:.6f}")
print(f"  CVaR (95%):     {cvar_95:.6f}")
print(f"  Sharpe Ratio:   {sharpe:.4f}")

print("\n" + "=" * 60)
print("  Done! All computations are zero-dependency pure Python.")
print("=" * 60)
