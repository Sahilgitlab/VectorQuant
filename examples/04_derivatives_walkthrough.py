"""
Finance Module Example: Derivatives & Options Pricing

Black-Scholes pricing, Greeks, and put-call parity.
"""

import math
import vectorquant as vq

def main():
    print("=" * 70)
    print("VectorQuant Finance Module: Derivatives Examples")
    print("=" * 70)

    # ─── 1. BLACK-SCHOLES PRICING ───────────────────────────────────────
    print("\n1. BLACK-SCHOLES OPTION PRICING")
    print("-" * 70)

    S, K, r, sigma, T = 100, 105, 0.05, 0.2, 1.0

    call = vq.derivatives.black_scholes_call(S, K, r, sigma, T)
    put = vq.derivatives.black_scholes_put(S, K, r, sigma, T)

    print(f"Spot Price (S):       ${S:.2f}")
    print(f"Strike Price (K):     ${K:.2f}")
    print(f"Risk-free Rate (r):   {r:.2%}")
    print(f"Volatility (σ):       {sigma:.2%}")
    print(f"Time to Maturity (T): {T:.1f} years\n")
    print(f"European Call Price:  ${call:.2f}")
    print(f"European Put Price:   ${put:.2f}")

    # ─── 2. PUT-CALL PARITY CHECK ──────────────────────────────────────
    print("\n2. PUT-CALL PARITY VERIFICATION")
    print("-" * 70)
    print("Put-Call Parity: C - P = S - K*exp(-rT)\n")

    parity_lhs = call - put
    parity_rhs = S - K * math.exp(-r * T)
    parity_error = abs(parity_lhs - parity_rhs)

    print(f"C - P:                {parity_lhs:.6f}")
    print(f"S - K*exp(-rT):       {parity_rhs:.6f}")
    print(f"Error:                {parity_error:.2e}")
    print(f"✓ Parity holds!" if parity_error < 1e-6 else "✗ Parity violated!")

    # ─── 3. THE GREEKS ────────────────────────────────────────────────
    print("\n3. OPTION GREEKS (Sensitivity Measures)")
    print("-" * 70)

    delta = vq.derivatives.bs_delta(S, K, r, sigma, T, 'call')
    gamma = vq.derivatives.bs_gamma(S, K, r, sigma, T)
    vega = vq.derivatives.bs_vega(S, K, r, sigma, T)
    theta = vq.derivatives.bs_theta(S, K, r, sigma, T, 'call')
    rho = vq.derivatives.bs_rho(S, K, r, sigma, T)

    print(f"Delta (∂C/∂S):  {delta:.4f}")
    print(f"  → Call price changes ${delta:.2f} per $1 spot move")
    print(f"\nGamma (∂²C/∂S²): {gamma:.4f}")
    print(f"  → Delta changes {gamma:.4f} per $1 spot move")
    print(f"\nVega (∂C/∂σ):   {vega:.4f}")
    print(f"  → Call price changes ${vega:.2f} per 1% volatility change")
    print(f"\nTheta (∂C/∂T):  {theta:.4f}")
    print(f"  → Call price changes ${theta:.2f} per day (time decay)")
    print(f"\nRho (∂C/∂r):    {rho:.4f}")
    print(f"  → Call price changes ${rho:.2f} per 1% rate change")

    # ─── 4. MONEYNESS & GREEK BEHAVIOR ─────────────────────────────────
    print("\n4. GREEKS ACROSS MONEYNESS")
    print("-" * 70)
    print("How Greeks change as option moves ITM, ATM, OTM\n")

    strikes = [90, 100, 105, 110, 120]
    print(f"{'Strike':>7} {'Spot':>7} {'Call':>8} {'Delta':>8} {'Gamma':>8} {'Vega':>8}")
    print("-" * 47)

    for K_test in strikes:
        call_test = vq.derivatives.black_scholes_call(S, K_test, r, sigma, T)
        delta_test = vq.derivatives.bs_delta(S, K_test, r, sigma, T, 'call')
        gamma_test = vq.derivatives.bs_gamma(S, K_test, r, sigma, T)
        vega_test = vq.derivatives.bs_vega(S, K_test, r, sigma, T)
        
        moneyness = "OTM" if K_test > S else ("ATM" if abs(K_test - S) < 5 else "ITM")
        print(f"${K_test:<6} ${S:<6} ${call_test:>6.2f}  {delta_test:>8.4f}  {gamma_test:>8.5f}  {vega_test:>8.2f}")

    # ─── 5. IMPLIED VOLATILITY & VEGA ──────────────────────────────────
    print("\n5. VOLATILITY SENSITIVITY (Vega)")
    print("-" * 70)
    print("Call price changes with volatility changes\n")

    print(f"{'Volatility':>12} {'Call Price':>12} {'Vega':>10}")
    print("-" * 35)

    for sigma_test in [0.10, 0.15, 0.20, 0.25, 0.30]:
        call_test = vq.derivatives.black_scholes_call(S, K, r, sigma_test, T)
        vega_test = vq.derivatives.bs_vega(S, K, r, sigma_test, T)
        print(f"{sigma_test:>12.2%} ${call_test:>11.2f}  {vega_test:>10.2f}")

    print("\n" + "=" * 70)
    print("✓ Examples completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
