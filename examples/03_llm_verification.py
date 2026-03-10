"""
VectorQuant Example: LLM Verification Pipeline
================================================
Demonstrates how an AI system uses VectorQuant as a
"deterministic truth engine" to prevent hallucinations.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq

print("=" * 60)
print("  VectorQuant — AI Hallucination Prevention Demo")
print("=" * 60)

# ─── Scenario 1: Verify a formula claim ────────────────────────────────
print("\n--- Scenario 1: Formula Verification ---")
print("AI claims: 'Sharpe ratio = mean return / variance'")

result = vq.ai.check_formula("sharpe_ratio", "mu / variance")
print(f"  Is correct:      {result.is_correct}")
print(f"  Correct formula: {result.correct_formula}")
print(f"  → AI hallucinated! The correct formula is: Sharpe = (mu - r_f) / sigma")

# ─── Scenario 2: Verify a numerical claim ──────────────────────────────
print("\n--- Scenario 2: Numerical Claim Check ---")
print("AI claims: 'A European call with S=100, K=100, r=5%, σ=20%, T=1yr costs $10.45'")

result = vq.ai.check_numerical_claim(
    "black_scholes_call",
    claimed_value=10.45,
    params={"S": 100, "K": 100, "r": 0.05, "sigma": 0.2, "T": 1.0},
    tolerance=0.01
)
print(f"  Claimed value: 10.45")
print(f"  Computed value: {result.computed_value}")
print(f"  Is correct: {result.is_correct}")

# ─── Scenario 3: Full Proof Trace ──────────────────────────────────────
print("\n--- Scenario 3: Proof Trace for VaR ---")
returns = [0.01, -0.02, 0.015, -0.005, 0.008, -0.01, 0.02, 0.005, -0.015, 0.012]

trace = vq.ai.explain_var(returns, confidence=0.95)
print(f"  Method: {trace.method}")
print(f"  Formula: {trace.formula}")
print(f"  Steps:")
for i, step in enumerate(trace.steps):
    print(f"    {i+1}. {step['step']} = {step['value']}")
print(f"  Result: {trace.result}")

# ─── Scenario 4: Hallucination-Proof Pipeline ─────────────────────────
print("\n--- Scenario 4: Full Pipeline (Intent → Compute → Verify → Explain) ---")

pipeline = vq.ai.HallucinationProofPipeline()
result = pipeline.process("var", returns=returns, confidence_level=0.95)
print(f"  {result}")
print(f"  Verified: {result.verified}")
print(f"  Confidence: {result.confidence:.0%}")

# ─── Scenario 5: LLM Tool Interface ───────────────────────────────────
print("\n--- Scenario 5: LLM Tool Execution with Verification ---")

llm = vq.ai.LLMInterface()
result = llm.execute("compute_sharpe", returns=returns)
print(f"  Tool: {result['tool']}")
print(f"  Value: {result['value']:.6f}")
print(f"  Verified: {result['verified']}")
print(f"  Has Proof: {result['proof_trace'] is not None}")

# ─── Scenario 6: OpenAI-Compatible Tool Schemas ───────────────────────
print("\n--- Scenario 6: OpenAI Tool Schemas ---")
schemas = vq.ai.get_tool_schemas()
print(f"  {len(schemas)} tools registered for OpenAI function calling:")
for s in schemas:
    print(f"    • {s['function']['name']}: {s['function']['description']}")

print("\n" + "=" * 60)
print("  VectorQuant: Deterministic reasoning engine for AI.")
print("  The LLM generates reasoning. VQ guarantees the math.")
print("=" * 60)
