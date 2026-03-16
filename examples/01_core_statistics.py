"""
Core Module Example: Statistics & Descriptive Analysis

This example demonstrates all statistical functions in vectorquant.core
without any external numerical dependencies.
"""

import vectorquant as vq

def main():
    print("=" * 70)
    print("VectorQuant Core Module: Statistics Examples")
    print("=" * 70)

    # Example data (daily stock returns)
    returns = [0.01, -0.02, 0.015, 0.02, -0.005, 0.018, -0.01, 0.025]
    returns_b = [0.02, -0.01, 0.01, 0.03, 0.002, 0.015, -0.005, 0.022]

    # ─── 1. BASIC STATISTICS ─────────────────────────────────────────────
    print("\n1. BASIC DESCRIPTIVE STATISTICS")
    print("-" * 70)

    mean_ret = vq.core.mean(returns)
    variance_ret = vq.core.variance(returns)
    std_ret = vq.core.standard_deviation(returns)
    median_ret = vq.core.median(returns)

    print(f"Mean Return:              {mean_ret:.6f}")
    print(f"Variance:                 {variance_ret:.8f}")
    print(f"Standard Deviation:       {std_ret:.6f}")
    print(f"Median Return:            {median_ret:.6f}")

    # ─── 2. SHAPE STATISTICS ─────────────────────────────────────────────
    print("\n2. DISTRIBUTION SHAPE (Skewness & Kurtosis)")
    print("-" * 70)

    skew = vq.core.skewness(returns)
    kurt = vq.core.kurtosis(returns)

    print(f"Skewness:                 {skew:.4f}")
    print(f"  → Near 0 = symmetric distribution")
    print(f"Kurtosis:                 {kurt:.4f}")
    print(f"  → >3 = fat tails (more extreme events)")

    # ─── 3. MULTIVARIATE STATISTICS ──────────────────────────────────────
    print("\n3. MULTIVARIATE STATISTICS (Covariance & Correlation)")
    print("-" * 70)

    cov_ab = vq.core.covariance(returns, returns_b)
    corr_ab = vq.core.correlation(returns, returns_b)

    print(f"Covariance (A, B):        {cov_ab:.8f}")
    print(f"Correlation (A, B):       {corr_ab:.4f}")
    print(f"  → +1.0 = perfect positive correlation")
    print(f"  → -1.0 = perfect negative correlation")
    print(f"  →  0.0 = uncorrelated")

    # ─── 4. PROBABILITY DISTRIBUTIONS ───────────────────────────────────
    print("\n4. PROBABILITY DISTRIBUTIONS")
    print("-" * 70)

    # Generate random samples from normal distribution
    vq.core.set_seed(42)
    normal_samples = [vq.core.rnorm(0, 1) for _ in range(10)]

    # Compute normal PDF/CDF
    z = 1.5  # standard deviations from mean
    pdf_val = vq.core.normal_pdf(z, mu=0, sigma=1)
    cdf_val = vq.core.normal_cdf(z, mu=0, sigma=1)

    print(f"Normal samples (seed=42):  {[f'{x:.4f}' for x in normal_samples[:5]]}...")
    print(f"Normal PDF(z={z}):        {pdf_val:.6f}")
    print(f"Normal CDF(z={z}):        {cdf_val:.6f}")
    print(f"  → CDF ≈ {cdf_val:.2%} probability below z={z}")

    # ─── 5. PRACTICAL EXAMPLE: PORTFOLIO DAILY RETURNS ────────────────────
    print("\n5. PRACTICAL: Portfolio Performance Analysis")
    print("-" * 70)

    # Simulate daily returns for a portfolio
    portfolio_daily = [0.001, -0.005, 0.002, 0.008, -0.001, 0.003, 0.001, 0.004,
                       0.006, -0.002, 0.005, 0.003, -0.001, 0.002]

    annual_return = vq.core.mean(portfolio_daily) * 252  # Annualized
    annual_vol = vq.core.standard_deviation(portfolio_daily) * (252 ** 0.5)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    print(f"Daily Return Mean:        {vq.core.mean(portfolio_daily):.6f}")
    print(f"Annualized Return:        {annual_return:.2%}")
    print(f"Annualized Volatility:    {annual_vol:.2%}")
    print(f"Sharpe Ratio (rf=0%):     {sharpe:.3f}")

    print("\n" + "=" * 70)
    print("✓ Examples completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
