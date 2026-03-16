# AI Verification

Hallucination detection and computation verification.

---

## The Problem

Large language models (LLMs) can confidently state incorrect answers.

**Example:**

LLM: "The Sharpe ratio of daily returns [0.01, 0.02, -0.01] with risk-free rate 2% is 0.707"

**Reality:**

```python
import vectorquant as vq

returns = [0.01, 0.02, -0.01]
sharpe = vq.portfolio.sharpe_ratio(returns, risk_free_rate=0.02/252)
# Returns: 0.196 (completely different!)
```

**Cost of missing this:** Wrong trading decision, portfolio blowup, regulatory failure.

---

## VectorQuant's Solution

**Verification pipeline:** Compute → Compare → Score

```python
verification = vq.ai.verify_calculation(
    expression="mean([0.01, 0.02, -0.01]) / std([0.01, 0.02, -0.01])",
    expected=0.707,
    tolerance=1e-6
)

print(f"Verified: {verification.verified}")           # False
print(f"Confidence: {verification.confidence:.1%}")   # 0% (completely wrong)
print(f"Actual: {verification.computed_value:.3f}")   # 0.196
```

---

## Use Cases

### 1. LLM-Generated Models

**Scenario:** You ask GPT to compute portfolio metrics.

```python
# Step 1: LLM thinks and responds
# "I computed the optimal weights: [0.45, 0.38, 0.17]"

# Step 2: Verify with VectorQuant
result = vq.ai.verify_calculation(
    expression="optimal_weights([returns], rf)",
    expected=[0.45, 0.38, 0.17],
    tolerance=0.01  # Within 1%
)

if result.verified:
    print("✓ Use these weights")
else:
    print("✗ Recompute with VectorQuant")
```

### 2. Research Paper Validation

**Scenario:** Academic claims specific Sharpe ratio.

```python
# Paper claims: "Our strategy achieves Sharpe of 2.3"

# Verify
result = vq.ai.verify_calculation(
    expression="calculate_sharpe(strategy_returns)",
    expected=2.3,
    tolerance=0.05
)

# Check if claim is defensible
if result.verified and result.confidence > 0.95:
    print("✓ Claim is accurate")
else:
    print("⚠ Claim needs scrutiny")
```

### 3. Model Debugging

**Scenario:** Your model produces unexpected results.

```python
# Step 1: What did your code compute?
my_sharpe = 1.5

# Step 2: Is it correct?
result = vq.ai.verify_calculation(
    expression="mean(returns) / std(returns)",
    expected=my_sharpe,
    tolerance=1e-10
)

if not result.verified:
    print(f"Bug found! Should be {result.computed_value:.3f}")
    print(f"Confidence: {result.confidence:.0%}")
```

### 4. Regulatory Compliance

**Scenario:** Auditors question your risk metrics.

```python
# Your computation: VaR = 2.5%
var_claimed = 0.025

# Verify
result = vq.ai.verify_calculation(
    expression="parametric_var(returns, 0.95)",
    expected=var_claimed,
    tolerance=1e-6
)

# Produce proof
audit_report = {
    "metric": "95% Daily VaR",
    "claimed": var_claimed,
    "computed": result.computed_value,
    "verified": result.verified,
    "confidence": result.confidence,
    "proof_trace": result.proof_trace
}
```

---

## Core Components

### 1. Computation

VectorQuant fully computes the requested metric.

```python
# Inside verify_calculation:
# 1. Parse the expression
# 2. Use VectorQuant engine (C or Python)
# 3. Get exact result
# 4. Return {computed_value, verification_result}
```

### 2. Comparison

Compare computed value against expected.

```python
error_rate = |computed - expected| / |expected|

# Error rate determines verification
if error_rate < tolerance:
    verified = True
else:
    verified = False
```

### 3. Confidence Scoring

Compute confidence in result (0-1 scale).

```python
def confidence_score(error_rate):
    """
    1.0 = exact match
    0.0 = off by 100% or more
    """
    return max(0, 1 - error_rate)

# Examples:
# error 0%    → confidence 1.0
# error 10%   → confidence 0.9
# error 50%   → confidence 0.5
# error 100%+ → confidence 0.0
```

---

## Verification API

### Basic Verification

```python
result = vq.ai.verify_calculation(
    expression="sqrt(4)",
    expected=2.0,
    tolerance=1e-10
)
```

**Returns:**

```python
VerificationResult {
    verified: bool,              # Did it match expected?
    computed_value: float,       # What we actually computed
    confidence: float,           # Score 0-1
    error_rate: float,           # Relative error
    expression: str,             # Original expression
    timestamp: datetime          # When verified
}
```

### Get Explanation

```python
result = vq.ai.explain_sharpe(
    returns=[0.01, 0.02, -0.01],
    risk_free_rate=0.02
)
```

**Returns:**

```python
ExplanationTrace {
    result: float,               # Final Sharpe ratio
    steps: list[dict],          # Step-by-step:
                                #   {step, formula, value}
    formula: str,               # Overall mathemical formula
    computation_time: float      # Seconds to compute
}
```

**Example output:**

```python
[
    {"step": "Mean return", "formula": "mean(x)", "value": 0.0067},
    {"step": "Std dev", "formula": "std(x)", "value": 0.0108},
    {"step": "Risk-free rate", "formula": "r_f", "value": 0.00008},
    {"step": "Sharpe ratio", "formula": "(mean - rf) / std", "value": 0.606}
]
```

### Full Pipeline

```python
pipeline = vq.ai.HallucinationProofPipeline()

result = pipeline.process(
    intent="sharpe",  # What to compute
    returns=[...],    # Parameters
    risk_free_rate=0.02
)
```

**Returns:**

```python
PipelineResult {
    result: float,               # Computed value
    verified: bool,              # Matches expectations?
    confidence: float,           # 0-1 score
    proof_trace: dict,          # Full step-by-step
    explanations: list,         # Multiple explanations
    warnings: list              # Any issues found
}
```

---

## Supported Operations

### Statistics

✓ mean, std, variance
✓ skewness, kurtosis
✓ correlation, covariance

### Portfolio

✓ portfolio_return
✓ portfolio_volatility
✓ sharpe_ratio (main use case)
✓ optimize_max_sharpe

### Derivatives

✓ black_scholes_call, put
✓ Greeks (delta, gamma, vega, theta, rho)

### Risk

✓ parametric_var, historical_var
✓ cvar

### Custom Math

✓ Arithmetic: +, -, *, /, ^
✓ Functions: sqrt, exp, log
✓ Lists: mean([x,y,z]), max([...])

---

## Example: LLM Trading Signal

**Scenario:** LLM suggests a trade based on metrics.

```python
# Step 1: LLM analysis
llm_analysis = """
Given AAPL returns of [0.02, -0.01, 0.03, 0.01, -0.02],
the daily Sharpe ratio is 1.52.
This indicates a strong buy signal.
Recommended position size: 25% of portfolio.
"""

# Step 2: Extract metrics
llm_sharpe = 1.52
llm_position_size = 0.25

# Step 3: Verify critical claims
returns = [0.02, -0.01, 0.03, 0.01, -0.02]
rf_daily = 0.02 / 252

sharpe_verification = vq.ai.verify_calculation(
    expression="sharpe_ratio(returns, rf)",
    expected=llm_sharpe,
    tolerance=0.05  # Within 5%
)

# Step 4: Decide
if sharpe_verification.verified and sharpe_verification.confidence > 0.9:
    print(f"✓ LLM analysis is sound. Sharpe ratio: {sharpe_verification.computed_value:.2f}")
    position_size = llm_position_size
else:
    print(f"⚠ LLM overestimated. Actually: {sharpe_verification.computed_value:.2f}")
    print(f"  Confidence: {sharpe_verification.confidence:.0%}")
    position_size = llm_position_size * 0.5  # Reduce aggression

execute_trade(position_size)
```

---

## Example: Academic Paper Verification

**Scenario:** You're reading a quant research paper and want to verify claims.

```python
# Paper claim #1:
# "Using mean-variance optimization on 10-year data
#  produced weights [0.35, 0.40, 0.25] with Sharpe 1.87"

# Reproduce the analysis
your_returns = load_10year_data()
your_weights = vq.portfolio.optimize_max_sharpe(your_returns, rf=0.02)
your_sharpe = vq.portfolio.sharpe_ratio(
    [w * r for w, r in zip(your_weights, your_returns)], 
    rf=0.02
)

# Verify
verification = vq.ai.verify_calculation(
    expression="optimize_max_sharpe(returns, rf=0.02)",
    expected=[0.35, 0.40, 0.25],
    tolerance=0.05
)

print(f"Weights match: {verification.verified}")
print(f"Actual weights: {your_weights}")

# Verify Sharpe
sharpe_verification = vq.ai.verify_calculation(
    expression="sharpe_ratio(portfolio_returns, rf)",
    expected=1.87,
    tolerance=0.05
)

print(f"Sharpe matches: {sharpe_verification.verified}")
print(f"Actual Sharpe: {your_sharpe:.2f}")

# Draw conclusion
if not verification.verified:
    print("⚠ Paper results may differ due to data differences")
    print(f"  Confidence in reproducibility: {verification.confidence:.0%}")
```

---

## Limitations

### What VectorQuant CAN Verify

✓ **Mathematical correctness**
  - Is 2+2=4? Yes
  - Is sqrt(4)=2? Yes
  - Is Sharpe ratio correct? Yes

✓ **Numerical accuracy**
  - Within expected floating point error
  - Bit-identical across platforms

✓ **Formula implementation**
  - LLM correctly stated the formula
  - Calculation follows the formula

### What VectorQuant CANNOT Verify

❌ **Logic correctness**
  - "Should we use Sharpe ratio?" (domain knowledge)
  - "Is this portfolio appropriate?" (business decision)
  - "Is the model framework sound?" (research question)

❌ **Data quality**
  - "Is this data accurate?" (source quality)
  - "Is time period representative?" (sample selection)
  - "Are outliers real or measurement errors?" (data cleaning)

❌ **Causality**
  - "Does X cause Y?" (causal inference, not computation)

### Example: Limitation

```python
# ✓ CAN verify: Is Sharpe computed correctly?
verification = vq.ai.verify_calculation(
    expression="mean(returns) / std(returns)",
    expected=llm_sharpe
)

# ❌ CANNOT verify: Should we use Sharpe ratio?
# This requires domain knowledge, not computation
```

---

## Integration with LLM Workflows

### Pattern 1: Verify-Then-Use

```python
def llm_assisted_optimization(user_question):
    # Step 1: LLM generates answer
    llm_response = llm.generate(user_question)
    
    # Step 2: Extract values
    weights = parse_weights(llm_response)
    
    # Step 3: Verify critical metrics
    verification = vq.ai.HallucinationProofPipeline().process(
        intent="portfolio_metrics",
        weights=weights,
        returns=market_data
    )
    
    # Step 4: Use verified values
    if verification.confidence > 0.9:
        use_weights = verification.result
    else:
        use_weights = recompute_with_vectorquant()
    
    return use_weights
```

### Pattern 2: Confidence Scaling

```python
def execute_with_confidence(weights, llm_confidence, verification_confidence):
    """
    Trade size depends on LLM confidence AND verification confidence.
    Both must be high to be aggressive.
    """
    combined_confidence = llm_confidence * verification_confidence
    
    # Scale position
    if combined_confidence > 0.95:
        position_size = 1.0  # Full size
    elif combined_confidence > 0.80:
        position_size = 0.5  # Half size
    elif combined_confidence > 0.50:
        position_size = 0.1  # Small position
    else:
        position_size = 0.0  # Wait for more info
    
    return execute_trade(weights, position_size)
```

### Pattern 3: Continuous Verification

```python
# As you compute more metrics, keep verifying
computed_metrics = {
    "mean_return": 0.08,
    "volatility": 0.15,
    "sharpe": 0.533,
    "var_95": 0.032,
    "cvar_95": 0.045
}

for metric_name, metric_value in computed_metrics.items():
    verification = vq.ai.verify_calculation(
        expression=f"{metric_name}(data)",
        expected=metric_value,
        tolerance=1e-10
    )
    
    if not verification.verified:
        print(f"⚠ {metric_name} differs!")
        print(f"  Your value: {metric_value}")
        print(f"  Actual: {verification.computed_value}")
```

---

## Best Practices

### 1. Always Verify Critical Decisions

```python
# ✓ Good: Verify before trading
is_valid = vq.ai.verify_calculation(...)
if is_valid.verified and is_valid.confidence > 0.95:
    place_trade()

# ❌ Bad: Trust LLM blindly
llm_says_buy = True
place_trade()  # Dangerous!
```

### 2. Use Appropriate Tolerances

```python
# For exact math (mean, sum)
vq.ai.verify_calculation(..., tolerance=1e-10)  # Strict

# For statistical estimates (Sharpe, VaR)
vq.ai.verify_calculation(..., tolerance=0.01)   # Loose
```

### 3. Log Verification Results

```python
# For audit trail
verification_log = []
for decision in trading_decisions:
    result = vq.ai.verify_calculation(decision)
    verification_log.append({
        "decision": decision,
        "verified": result.verified,
        "confidence": result.confidence,
        "timestamp": now()
    })

# Later: regulators can audit your logic
save_to_database(verification_log)
```

### 4. Understand Limitations

```python
# Verification checks computation, not judgment
verified = vq.ai.verify_calculation(...)

if verified.verified:
    # ✓ Computation is correct
    # ⚠ But is it the right thing to compute?
    # ⚠ Is the model framework sound?
    # ⚠ Are we using it correctly?
```

---

## Performance

### Verification Speed

| Operation | Time |
|-----------|------|
| Verify simple math | < 1ms |
| Verify Sharpe ratio | 5-10ms |
| Verify covariance | 50-100ms |
| Full proof pipeline | 100-500ms |

**Negligible compared to actual trading/optimization.**

---

## Summary

**VectorQuant's AI Verification:**
- ✓ Detects hallucinations in financial computations
- ✓ Provides confidence scores
- ✓ Generates proof traces
- ✓ Integrates with LLM workflows
- ✓ Supports regulatory auditing
- ⚠ Only verifies computation, not judgment

**Use it:** Every time an LLM or human generates a financial metric you'll act on.

**Don't use it:** For problems it can't solve (data quality, model design, causality).
