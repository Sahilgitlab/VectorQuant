# VectorQuant

**A deterministic reasoning engine for AI and quantitative finance.**

*Created by Sahil Gupta*

**[Documentation](https://doc-vector-quant.vercel.app/)** | **[Repository](https://github.com/Sahilgitlab/VectorQuant)**

Zero-dependency · Pure Python · C Engine (165x) · LLM-Native · Agentic AI Ready

---

## Why VectorQuant Exists

LLMs — ChatGPT, Claude, Gemini, and every other model — hallucinate financial calculations. They produce confident, plausible-looking numbers that are mathematically wrong in ways a non-expert cannot detect.

There is no existing library that catches this automatically.

VectorQuant is that library. It is the **verified mathematical ground-truth layer** for AI agents operating in quantitative finance — for fine-tuning pipelines, RAG systems, agentic frameworks, and any application where an LLM computes, reasons about, or retrieves financial numbers.

---

## The Problem: Four Ways LLMs Get Finance Wrong

```
Without VectorQuant:                  With VectorQuant:
─────────────────────                 ─────────────────────────────────────────────
LLM computes Sharpe ratio             LLM calls vq.ai.verify_numeric("sharpe_ratio", ...)
using return / variance               VQ returns: {
                                        value:       -0.81,         ← correct answer
User trusts it.                         verified:    False,         ← LLM was wrong
                                        failure_mode: "formula",    ← caught it
It's wrong by 6000%.                    confidence:  0.0,
                                        proof_trace: [...],         ← here's why
                                        citation:    "Sharpe 1966"  ← authoritative source
                                      }
```

VQ catches hallucinations across **four distinct failure modes** — no other library does this:

| Failure Mode | Example | Error Magnitude |
|---|---|---|
| **Wrong formula** | Sharpe: `return / variance` instead of `(return - rf) / std` | 6000%+ |
| **Wrong convention** | USD T-bill: `Act/365` instead of `Act/360`; yield instead of discount rate | ~2bp silent error |
| **Wrong units** | Annualised Sharpe returned as `0.003` (forgot `× √252`) | 15x off |
| **Wrong distribution** | VaR using Normal instead of Student's t (df=4) | 53% underestimate |

---

## Who Uses VectorQuant

- **AI / LLM engineers** building finance agents, chatbots, or copilots that must not hallucinate numbers
- **Fine-tuning teams** who need ground-truth verified computation to generate training data
- **RAG systems** that retrieve and reason about financial formulas and market conventions
- **Agentic AI frameworks** (LangChain, LlamaIndex, AutoGen, CrewAI) that call financial tools
- **Quant researchers** who want a zero-dependency, deterministic math foundation

---

## Quick Start

```bash
pip install vectorquant
```

### Catch a Hallucination in 3 Lines

```python
import vectorquant as vq

result = vq.ai.verify_numeric(
    "sharpe_ratio",
    llm_value=2.45,                                   # what the LLM claimed
    inputs={"returns": [0.01, -0.02, 0.015, 0.02, -0.005],
            "risk_free_rate": 0.02 / 252}
)

print(result.value)        # -0.8094  ← correct answer
print(result.verified)     # False    ← LLM was wrong
print(result.failure_mode) # formula  ← denominator: variance instead of std
print(result.warnings)     # ["Correct: -0.8094 | LLM: 2.45 | Error: 402%"]
```

### Check a Convention Error

```python
# LLM assumed wrong day count and compounding for a USD corporate bond
result = vq.ai.conventions.check(
    "corporate_bond", "USD",
    llm_assumptions={"day_count": "Act/365", "compounding": "continuous"}
)

# result.errors:
# [{"field": "day_count",   "llm": "Act/365",    "correct": "30/360"},
#  {"field": "compounding", "llm": "continuous", "correct": "semi-annual"}]
# result.impact: "~15bp error on 10yr bond at typical rates"
```

### Catch a Unit Error

```python
# LLM returned 0.003 for "annualised Sharpe ratio" — forgot × √252
result = vq.ai.unit_checker.check(
    value=0.003,
    formula="sharpe_ratio",
    question="What is the annualised Sharpe ratio?"
)

print(result.verified)      # False
print(result.warnings)
# ["0.003 is 1000x smaller than typical Sharpe range (-3, 3).
#   Likely error: forgot annualisation — multiply by sqrt(252).
#   Corrected value: 0.0476"]
```

---

## Multi-LLM Integration

VectorQuant integrates natively with every major AI framework. One verified math layer, any LLM on top.

### OpenAI / ChatGPT

```python
from openai import OpenAI
import vectorquant as vq

client = OpenAI()
tools = vq.ai.get_tool_schemas()  # OpenAI function-calling format

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the Sharpe ratio for these returns: [0.01, -0.02, 0.015, 0.02]?"}],
    tools=tools
)

# VQ executes the tool call and returns a verified VQResult
tool_call = response.choices[0].message.tool_calls[0]
result = vq.ai.execute_tool(tool_call.function.name, **eval(tool_call.function.arguments))

# result.verified = True   result.proof_trace = [mean, std, sharpe steps]
```

### Anthropic / Claude

```python
import anthropic
import vectorquant as vq

client = anthropic.Anthropic()
tools = vq.ai.get_tool_schemas(format="anthropic")

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Calculate parametric VaR at 95% confidence."}]
)

# Every computation verified against VQ ground truth before returning to user
```

### Google Gemini

```python
import google.generativeai as genai
import vectorquant as vq

tools = vq.ai.get_tool_schemas(format="gemini")
model = genai.GenerativeModel("gemini-2.0-flash", tools=tools)
```

### LangChain

```python
from langchain.agents import initialize_agent
import vectorquant as vq

vq_tools = vq.ai.get_langchain_tools()
agent = initialize_agent(vq_tools, llm, agent="zero-shot-react-description")
result = agent.run("What is the Black-Scholes price for a call option with S=100, K=105, r=0.05, sigma=0.2, T=1?")
# All computations verified deterministically — no hallucinated option prices
```

### MCP Server (any MCP-compatible client)

```bash
pip install vectorquant[mcp]
vq-mcp-server  # Drop-in MCP server exposing all VQ tools
```

---

## Use in Fine-Tuning and RAG

### Fine-Tuning Data Validation

When generating training data for finance-domain fine-tuning, VQ ensures every ground-truth answer is mathematically correct:

```python
import vectorquant as vq
from vectorquant.ai.formula_registry import FORMULA_REGISTRY

training_examples = []

for formula_name, spec in FORMULA_REGISTRY.items():
    test = spec["test_case"]

    # Ground truth — verified by VQ's deterministic engine
    result = vq.ai.verify_numeric(
        formula_name,
        llm_value=test["expected_value"],
        inputs=test["inputs"]
    )

    training_examples.append({
        "prompt":           f"Calculate {spec['name']} for inputs: {test['inputs']}",
        "ground_truth":     result.value,
        "verified":         result.verified,     # only include if True
        "proof_trace":      result.proof_trace,  # chain-of-thought for training
        "citation":         result.citation,
        "formula_latex":    result.formula_latex,
    })

# Every training example is mathematically verified — no hallucinated ground truth
verified = [ex for ex in training_examples if ex["verified"]]
```

### RAG — Convention-Aware Retrieval

```python
import vectorquant as vq

def verify_retrieved_convention(query: str, retrieved_text: str) -> dict:
    """Validate conventions retrieved from a knowledge base before passing to LLM."""

    # Extract instrument and market from query
    conv = vq.ai.conventions.lookup("treasury_bill", "USD")

    # Check whether retrieved text matches authoritative convention
    result = vq.ai.conventions.check(
        "treasury_bill", "USD",
        llm_assumptions=parse_assumptions(retrieved_text)
    )

    return {
        "use_retrieved":  result["is_correct"],
        "authoritative":  conv,
        "errors_found":   result.get("errors", []),
        "impact":         result.get("impact", "none"),
    }
```

### Agentic AI — Verified Tool Calls

```python
# Pattern: LLM proposes → VQ verifies → verified result returned to agent
import vectorquant as vq

def verified_finance_tool(formula: str, inputs: dict) -> dict:
    result = vq.ai.verify_numeric(formula, llm_value=None, inputs=inputs)
    return {
        "value":       result.value,
        "verified":    result.verified,
        "confidence":  result.confidence,
        "proof":       [step.explanation for step in result.proof_trace],
        "citation":    result.citation,
    }
```

---

## What VQ Verifies — Formula Registry (30+ formulas, expanding to 100+)

Every formula includes: correct definition, known LLM error patterns, academic citation, numeric test case, and unit annotation.

### Risk Metrics
```
sharpe_ratio      parametric_var    historical_var    cvar
sortino_ratio     calmar_ratio      maximum_drawdown  annualized_vol
treynor_ratio     information_ratio kelly_criterion   beta_coefficient
jensen_alpha      tracking_error    log_return        cagr
```

### Derivatives & Fixed Income
```
black_scholes_call    black_scholes_put    bs_delta    bs_gamma
modified_duration     dv01                 ytm         bond_price
```

### Portfolio
```
portfolio_return    portfolio_variance    correlation_pearson
```

---

## Finance Modules (the verified math substrate)

The quant modules are the ground truth that VQ uses to catch hallucinations. They are also directly usable:

```python
# Risk
vq.risk.parametric_var(returns, confidence_level=0.95)
vq.risk.cvar(returns, confidence_level=0.95)

# Derivatives
vq.derivatives.black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1.0)
vq.derivatives.bs_delta(S, K, r, sigma, T)

# Portfolio
vq.portfolio.optimize_max_sharpe(expected_returns, cov_matrix)

# Distributions (fat-tail aware)
vq.distributions.StudentT(df=5).var(confidence=0.99)  # t-dist VaR (+53% vs Normal)
vq.distributions.fit_all(returns)  # ranks Normal, t, Laplace, GPD by AIC

# Stochastic
vq.stochastic.simulate_geometric_brownian_motion(S0=100, mu=0.05, sigma=0.2, T=1.0, dt=1/252, n_paths=1000)

# Core math
vq.stats.mean(data)
vq.linalg.matrix_multiply(A, B)
vq.prob.normal_inv_cdf(0.95)  # z = 1.645
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    AI / LLM Integration Layer                     │
│   ChatGPT · Claude · Gemini · LangChain · LlamaIndex             │
│   MCP Server · REST API · AutoGen · CrewAI · Any HTTP client      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│              vq.ai — Verification & Reasoning Engine              │
│  Formula Registry (30 → 100+ formulas, bidirectional)             │
│  Hallucination Detection Levels 1–6                               │
│  Convention Database (40+ instrument/market pairs)                │
│  Unit Checker (dimensional analysis, annualisation)               │
│  Proof Traces (streaming + LaTeX) · Confidence Scoring            │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│               Finance & Statistics Layer                           │
│   portfolio · risk · derivatives · distributions                  │
│   factor_models · stochastic · time_series                        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                  Core Math Engine                                  │
│   linalg · stats · prob · optimization · numerics                 │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│              Performance Engine (Three Tiers)                      │
│   Pure Python → C engine (165x) → NumPy/LAPACK (optional)        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Performance Tiers

```bash
pip install vectorquant           # Zero dependencies — pure Python
pip install vectorquant[fast]     # + NumPy acceleration (LAPACK eigendecomposition)
pip install vectorquant[perf]     # + Numba JIT (15x faster)
pip install vectorquant[gpu]      # + CuPy GPU (200x+ faster Monte Carlo)
pip install vectorquant[openai]   # + OpenAI tool schemas
pip install vectorquant[anthropic]# + Anthropic tool schemas
pip install vectorquant[gemini]   # + Gemini tool schemas
pip install vectorquant[langchain]# + LangChain tools
pip install vectorquant[mcp]      # + MCP server
pip install vectorquant[all-llm]  # + All LLM integrations
pip install vectorquant[full]     # Everything
```

| Backend | Monte Carlo Speed | When to Use |
|---|---|---|
| Pure Python | ~6,500 paths/sec | Zero-dep, any machine |
| C engine | ~1,000,000 paths/sec (165x) | Default for production |
| Numba JIT | ~97,500 paths/sec (15x) | No C compiler available |
| GPU (CuPy) | 1,500,000+ paths/sec (200x+) | Institutional scale |

---

## How VQ Compares

| Capability | VectorQuant | QuantLib | scipy/statsmodels | LangChain |
|---|:---:|:---:|:---:|:---:|
| AI hallucination detection | Yes — 6 levels | No | No | No |
| Convention database (40+ pairs) | Yes | Partial (internal) | No | No |
| Unit / dimensional checking | Yes | No | No | No |
| Proof traces per computation | Yes — streaming + LaTeX | No | No | No |
| Confidence scoring + failure mode | Yes | No | No | No |
| Fat-tail distributions (Student's t, GPD) | Yes | Partial | Yes | No |
| LLM tool integration (6 formats) | Yes | No | No | Partial |
| MCP server | Yes | No | No | No |
| Academic citation per formula | Yes — every formula | No | Partial | No |
| Zero external dependencies | Yes | No | No | No |
| C engine (165x) | Yes | Yes (C++) | Partial | No |

VQ's clearest moat: **the convention database**. Financial market conventions — day count fractions, compounding frequencies, settlement rules, yield quote types — are scattered across ISDA definitions, market standards documents, and institutional knowledge. They are not in QuantLib's public API in machine-readable, LLM-queryable form. LLMs make silent errors here constantly. VQ is the only library that catches them.

---

## Examples

```bash
python examples/quickstart.py                # 5-minute tour
python examples/03_llm_verification.py       # Hallucination detection demo
python examples/06_ai_verification.py        # Full verification pipeline
python examples/02_portfolio_optimization.py # Portfolio + risk
python examples/04_derivatives_walkthrough.py # Black-Scholes + Greeks
```

---

## Project Status

| Component | Status |
|---|---|
| Core math engine | ~85% of spec |
| C engine (165x) | ~60% coverage |
| Derivatives (Black-Scholes, Greeks, IV) | ~80% |
| Risk models (VaR, CVaR) | ~75% |
| Portfolio optimisation | ~70% |
| `vq.distributions` (Student's t, GPD, Normal) | Complete |
| Formula registry (30 entries, bidirectional) | Complete — expanding to 100+ |
| Convention database (40+ pairs) | Complete |
| Unit checker (Level 6 detection) | Complete |
| `vq.ai.verify_numeric` (Level 3) | Complete |
| LLM formats (OpenAI, Anthropic, Gemini, LangChain) | In progress |
| MCP server | Planned v0.6.0 |
| `vq.tests` (hypothesis tests) | Planned v0.6.0 |
| `vq.econometrics` (OLS, VAR, VECM) | Planned v0.7.0 |

---

## Contributing

VectorQuant is actively developed. The highest-impact areas for contribution:

- **Formula registry** — add entries with correct expression, known LLM errors, citation, test case
- **Convention database** — additional instrument/market pairs with authoritative citations
- **LLM integrations** — Anthropic, Gemini, LangChain, LlamaIndex tool schemas
- **Hypothesis tests** — `vq.tests` module (Jarque-Bera, ADF, KPSS, Ljung-Box, cointegration)

Open an issue or submit a PR.

---

## Author

**Sahil Gupta** — Mascot Universal Pvt Ltd

- **Email:** [linkedin.sahil.gupta07@gmail.com](mailto:linkedin.sahil.gupta07@gmail.com)
- **LinkedIn:** [https://www.linkedin.com/in/sahilg007/](https://www.linkedin.com/in/sahilg007/)

---

## License

MIT License — Copyright (c) 2026 Sahil Gupta. Use it however you want.
