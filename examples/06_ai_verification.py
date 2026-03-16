"""
AI Module Example: Verification & Hallucination Detection

Demonstrates formula validation, hallucination detection, and computation tracing
for catching AI errors automatically.
"""

import vectorquant as vq

def main():
    print("=" * 70)
    print("VectorQuant AI Module: Verification Examples")
    print("=" * 70)

    # ─── 1. VERIFY MATHEMATICAL EXPRESSIONS ────────────────────────────
    print("\n1. MATHEMATICAL EXPRESSION VERIFICATION")
    print("-" * 70)

    expressions_to_verify = [
        ("sqrt(4) * 3", 6.0),
        ("exp(0)", 1.0),
        ("pow(2, 10)", 1024.0),
        ("sqrt(16) + 4", 8.0),
    ]

    print("\nVerifying mathematical expressions:\n")
    for expr, expected in expressions_to_verify:
        result = vq.ai.verify_calculation(expr, expected=expected)
        status = "✓" if result.verified else "✗"
        print(f"{status} {expr} = {expected}")
        print(f"   Computed: {result.computed_value}, Confidence: {result.confidence}\n")

    # ─── 2. PROBABILITY DISTRIBUTION VERIFICATION ──────────────────────
    print("\n2. PROBABILITY DISTRIBUTION VERIFICATION")
    print("-" * 70)

    print("\nVerifying probability calculations:\n")
    
    # Standard normal PDF at x=0
    result = vq.ai.verify_probability(
        "normal_pdf",
        params={"mu": 0, "sigma": 1},
        x=0,
        expected=0.3989,
        tolerance=0.001
    )
    status = "✓" if result.verified else "✗"
    print(f"{status} Standard Normal PDF at x=0")
    print(f"   Expected: 0.3989, Computed: {result.computed_value}\n")

    # Normal CDF at z=1.96 (95% confidence)
    result = vq.ai.verify_probability(
        "normal_cdf",
        params={"mu": 0, "sigma": 1},
        x=1.96,
        expected=0.975,
        tolerance=0.001
    )
    status = "✓" if result.verified else "✗"
    print(f"{status} Standard Normal CDF at x=1.96")
    print(f"   Expected: 0.975, Computed: {result.computed_value}\n")

    # ─── 3. FINANCIAL FORMULA VERIFICATION ────────────────────────────
    print("\n3. FINANCIAL FORMULA VERIFICATION")
    print("-" * 70)

    print("\nVerifying financial computations:\n")

    # Black-Scholes Call Option
    result = vq.ai.verify_finance_formula(
        "black_scholes_call",
        params={"S": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0},
        expected=10.45,
        tolerance=0.01
    )
    status = "✓" if result.verified else "✗"
    print(f"{status} Black-Scholes Call Option (ATM)")
    print(f"   Expected: $10.45, Computed: ${result.computed_value:.2f}")
    print(f"   Confidence: {result.confidence}\n")

    # Sharpe Ratio
    returns = [0.01, -0.02, 0.015, 0.02, -0.005, 0.018]
    result = vq.ai.verify_finance_formula(
        "sharpe_ratio",
        params={"returns": returns, "risk_free_rate": 0.02},
        expected=0.4,
        tolerance=0.1
    )
    status = "✓" if result.verified else "✗"
    print(f"{status} Sharpe Ratio")
    print(f"   Computed: {result.computed_value:.4f}")
    print(f"   Confidence: {result.confidence}\n")

    # ─── 4. PROOF TRACING (STEP-BY-STEP EXPLANATION) ───────────────────
    print("\n4. COMPUTATION TRACING (Step-by-Step Proof)")
    print("-" * 70)

    returns_trace = [0.02, -0.01, 0.015, 0.03]
    rf = 0.02

    print(f"\nShowing step-by-step proof for Value-at-Risk:")
    print(f"Returns: {returns_trace}\n")

    trace = vq.ai.explain_var(returns_trace, confidence=0.95)
    print(f"Method: {trace.method}")
    print(f"Formula: {trace.formula}\n")
    
    print("Steps:")
    for i, step in enumerate(trace.steps):
        print(f"  {i+1}. {step['step']:<30} = {step['value']}")
    
    print(f"\nFinal Result: {trace.result:.4f}\n")

    # ─── 5. SHARPE RATIO TRACE ──────────────────────────────────────────
    print("\n5. SHARPE RATIO PROOF TRACE")
    print("-" * 70)

    trace = vq.ai.explain_sharpe(returns_trace, risk_free_rate=rf)
    print(f"\nMethod: {trace.method}")
    print(f"Formula: {trace.formula}\n")
    
    print("Steps:")
    for i, step in enumerate(trace.steps):
        print(f"  {i+1}. {step['step']:<30} = {step['value']}")
    
    print(f"\nFinal Sharpe Ratio: {trace.result:.4f}\n")

    # ─── 6. HALLUCINATION-PROOF PIPELINE ────────────────────────────────
    print("\n6. HALLUCINATION-PROOF PIPELINE")
    print("-" * 70)

    pipeline = vq.ai.HallucinationProofPipeline()

    # Example 1: VaR computation
    print("\nProcessing intent: 'var' with returns data")
    result = pipeline.process("var", returns=returns_trace, confidence_level=0.95)
    print(f"  Intent: {result.intent}")
    print(f"  Result: {result.result:.4f}")
    print(f"  Verified: {result.verified}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Method: {result.method}\n")

    # Example 2: Sharpe ratio
    print("Processing intent: 'sharpe' with returns data")
    result = pipeline.process("sharpe", returns=returns_trace, risk_free_rate=rf)
    print(f"  Intent: {result.intent}")
    print(f"  Result: {result.result:.4f}")
    print(f"  Verified: {result.verified}")
    print(f"  Confidence: {result.confidence:.0%}\n")

    # ─── 7. LLM TOOL INTERFACE ──────────────────────────────────────────
    print("\n7. LLM TOOL INTERFACE")
    print("-" * 70)

    llm = vq.ai.LLMInterface()

    print("\nExecuting tool: 'calculate_var' via LLM interface")
    result = llm.execute("calculate_var", returns=returns_trace, confidence_level=0.95)
    print(f"  Tool: {result['tool']}")
    print(f"  Value: {result['value']:.4f}")
    print(f"  Verified: {result['verified']}")
    print(f"  Has Proof: {result['proof_trace'] is not None}\n")

    # ─── 8. AVAILABLE TOOLS REGISTRY ────────────────────────────────────
    print("\n8. AVAILABLE TOOLS (for AI Systems)")
    print("-" * 70)

    registry = vq.ai.get_tool_registry()
    print(f"\n{len(registry)} tools available:\n")
    for tool_name, tool_info in list(registry.items())[:5]:  # Show first 5
        print(f"  • {tool_name}")
        print(f"    Description: {tool_info['description']}")
        print()

    print("=" * 70)
    print("✓ AI Verification Examples completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
