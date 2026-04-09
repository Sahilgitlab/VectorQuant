# VectorQuant — Claude Code Project Instructions

## Vision

**A deterministic reasoning engine for AI and quantitative finance.**

VectorQuant is the **verified mathematical ground-truth layer for AI systems operating in finance**. Its primary audience is AI engineers, not quant researchers. The quant math is the substrate. The verification is the product.

```
Primary users:
  - LLM engineers building finance agents (ChatGPT, Claude, Gemini, Llama, local models)
  - Fine-tuning teams generating verified training data for finance models
  - RAG systems retrieving and reasoning about financial formulas and conventions
  - Agentic AI frameworks (LangChain, LlamaIndex, AutoGen, CrewAI, MCP)
  - Compliance/audit teams needing traceable AI computation records (EU AI Act, SEC, FCA)
  - Local LLM deployments (hedge funds, banks) requiring air-gapped verification
```

---

## The Market Gap — Why This Library Must Exist

**Hallucination rate for structured financial calculations: 15–52% across 37 models (2026 benchmarks).**  
Zero existing libraries catch it automatically.

| Competitor | What They Guard | What They Miss |
|---|---|---|
| Guardrails AI | Prompt injection, PII, toxicity | Numerical correctness |
| NeMo Guardrails | Text safety, off-topic | Formula errors, convention errors |
| LangChain guardrails | Output format, toxicity | Whether Sharpe = return/variance is wrong |
| Cleanlab / DataIQ | Text quality scoring | Finance formula correctness in training data |
| Ragas / TruLens | RAG answer-vs-context faithfulness | Whether retrieved formula used wrong convention |

**VectorQuant is the only library in the exact space of deterministic numerical verification for finance.**

### Five Critical Gaps (all unoccupied)

1. **No deterministic verifier for finance math** — Guardrail frameworks cover text safety, zero cover numerical correctness. VQ owns this space entirely.

2. **Fine-tuning data has no financial math QA** — AI teams lose 40–60% of project time on data cleaning. Incorrect formula examples in SFT datasets propagate directly into model weights. Nobody scores this today. `vq.ai.sft_validator` is the first automated pipeline to score finance training data for mathematical correctness.

3. **Agentic AI has no math audit trail** — McKinsey 2025 agentic AI governance: "embedding monitoring hooks directly into agent workflows" is the core shift. No library generates a proof trace (formula used, citation, unit check, convention) for agentic financial computations. `VQResult.proof_trace` is structurally the answer.

4. **MCP has no finance-domain verification server** — MCP ecosystem has servers for search, code, data — zero domain-specific numerical verifiers. First mover gets organic adoption from every agent developer listing tools.

5. **Local LLMs have zero offline verifier** — Hedge funds and banks running Llama/Mistral/DeepSeek cannot call cloud APIs for compliance. VQ's zero-dependency design and C extension make it the only viable offline option.

### Regulatory Tailwind

EU AI Act (2025), FCA Consumer Duty, SEC AI guidance all require traceable, explainable AI decisions in finance. `VQResult.proof_trace` + `citation` + `convention_used` are already exactly the data regulators require. `vq.audit` converts this into enterprise revenue.

---

## The Core Thesis (Never Drift From This)

LLMs hallucinate financial calculations in four distinct failure modes:

```
1. Wrong formula      → vq.ai.formula_registry   (Sharpe: variance vs std → 6000% error)
2. Wrong convention   → vq.ai.conventions         (USD T-bill: Act/365 vs Act/360)
3. Wrong units/scale  → vq.ai.unit_checker        (0.003 returned for annualised Sharpe)
4. Wrong distribution → vq.distributions          (Normal vs Student's t → 53% VaR underestimate)
```

When writing code, reviewing features, or planning sprints — ask: "Does this make hallucination detection better or more reachable?" If not, it is secondary to the Study A script and formula registry expansion.

---

## Current State (April 2026)

**Version:** 0.5.2 (PyPI) | **Tests:** 203 passing | **Formula registry:** 30 entries | **Target:** 0.8.0

### Built

| Module | Status |
|---|---|
| Core math (`vq.linalg`, `vq.stats`, `vq.prob`, `vq.optim`) | ~85% |
| C engine (165x on covered ops) | ~60% coverage |
| `vq.finance` (risk, derivatives, portfolio, factor models) | ~75% |
| `vq.stochastic` (GBM, Heston, Monte Carlo) | ~80% |
| `vq.distributions` (StudentT, GPD, Normal, fit_all) | Complete |
| `vq.ai.formula_registry` — 30 entries, bidirectional | Complete |
| `vq.ai.numeric_verifier` — verify_numeric + check_expression | Complete |
| `vq.ai.conventions_db` — 40+ pairs | Complete |
| `vq.ai.unit_checker` — Level 6 detection | Complete |

### Not Yet Built

- Formula registry: 30 → 100+ (must reach 50 before Study A, 500 is long-term target)
- `scripts/research/study_a_hallucination_prevalence.py` — **next task**
- `vq.ai.sft_validator` — fine-tuning dataset QA pipeline
- `vq.mcp` — MCP server
- `vq.serve` — REST API with OpenAPI spec
- `vq.rag` — RAG post-retrieval verifier
- `vq.audit` — regulatory report generator
- `vq.agent.interceptor` — `@vq.verify` decorator
- Anthropic, Gemini, LangChain, LlamaIndex tool formats
- `vq.tests` (hypothesis tests)
- `vq.econometrics` (OLS, VAR, VECM) — v0.7.0

---

## Roadmap

### Immediate — Study A + Registry Expansion (now, before anything else)

The single highest-leverage move: publish FinMathBench v1 before building new features. A published benchmark makes VQ the measurement standard. Once you're the measurement standard, library adoption follows automatically.

- [ ] Formula registry: 30 → 100 entries (fixed income, credit, all Greeks, crypto)
- [ ] `scripts/research/study_a_hallucination_prevalence.py` — prompts GPT-4o + Claude + Gemini, writes CSV
- [ ] arXiv preprint — establish timestamp priority (cs.AI or q-fin.CP)
- [ ] Release `FinMathBench v1` to HuggingFace Datasets — perpetual citation source

### v0.6.0 — Research Foundation (now → June 2026)

**Priority 1 — must ship for paper and adoption:**
- [ ] `vq.mcp` — MCP server (`pip install vectorquant[mcp] && vq serve-mcp`)
- [ ] `vq.serve` — REST API with auto-generated OpenAPI spec (any language can use VQ)
- [ ] `vq.ai.sft_validator` — score finance SFT dataset rows for formula correctness
- [ ] Anthropic + Gemini + LangChain + LlamaIndex tool schemas
- [ ] `vq.tests` — Jarque-Bera, ADF, KPSS, Ljung-Box, Breusch-Pagan, cointegration
- [ ] `vq.audit` — export VQResult proof traces as EU AI Act / SEC / FCA compliant JSON/PDF
- [ ] NumPy acceleration tier (`pip install vectorquant[fast]`)

### v0.7.0 — Ecosystem Integration (June → Dec 2026)

- [ ] `vq.rag` — RAG post-retrieval verifier with Ragas/TruLens/Arize adapters
- [ ] `vq.agent.interceptor` — `@vq.verify(formula="sharpe_ratio")` decorator for LangChain/CrewAI/AutoGen
- [ ] `vq.offline` — certified air-gapped wheel, signed checksum, "works air-gapped" marketing
- [ ] `vq.benchmark` — public leaderboard web dashboard (vq.ai/leaderboard), auto-updated weekly
- [ ] `vq.econometrics` — OLS (Newey-West), VAR, VECM, Fama-MacBeth, Granger causality
- [ ] Formula registry: 100 → 300 entries (crypto funding rates, IL for AMMs, macro, actuarial)
- [ ] Submit paper to NeurIPS Datasets & Benchmarks track

### v0.8.0 — Performance + Streaming (2027)

- [ ] `vq.stream` — real-time streaming verification (SSE/WebSocket, token-by-token confidence)
- [ ] GPU acceleration (CuPy) + distributed Monte Carlo
- [ ] Formula registry: 300 → 500+ entries

### v0.9.0 — Domain Expansion (post-paper)

- [ ] `vq.medical` — NNT, sensitivity/specificity, Kaplan-Meier, NNH
- [ ] `vq.legal` — actuarial discount rates, damages calculations, mortality tables
- [ ] FinMathBench v2 — expanded to medical + legal

---

## New Module Specs (to implement)

### `vq.ai.sft_validator` — Fine-Tuning Dataset QA
```python
# Score every (prompt, response) pair in a finance SFT dataset
report = vq.ai.sft_validator.score_dataset(
    dataset,                   # HuggingFace Dataset or list of dicts
    prompt_col="prompt",
    response_col="response"
)
# Per-row: verified/unverified, failure_mode, suggested_correction, confidence
# Summary: % correct, breakdown by failure mode, rows to remove/fix
```

### `vq.audit` — Regulatory Report Generator
```python
# Export VQResult proof traces as regulator-ready reports
report = vq.audit.generate(
    result,                    # VQResult
    format="json",             # "json" | "pdf"
    standard="eu_ai_act"       # "eu_ai_act" | "sec" | "fca"
)
# Includes: formula, citation DOI, LaTeX, convention assumptions, failure mode, timestamp
```

### `vq.agent.interceptor` — Drop-In Agentic Middleware
```python
@vq.verify(formula="sharpe_ratio")
def compute_sharpe(returns, rf):
    return (mean(returns) - rf) / std(returns)

# Any call to compute_sharpe() now returns a VQResult, auto-verified
# Works as a LangChain tool wrapper, CrewAI task wrapper, AutoGen function
```

### `vq.rag` — RAG Post-Retrieval Verifier
```python
# Intercept RAG output, extract numeric claims, verify each
result = vq.rag.verify_answer(
    answer=llm_answer,
    context=retrieved_chunks,
    inputs=user_inputs
)
# Returns: inline confidence scores, which claims verified, which failed + why
```

### `vq.mcp` — MCP Server
```bash
pip install vectorquant[mcp]
vq serve-mcp  # Exposes verify_numeric, check_expression, conventions.lookup, unit_checker
              # Works with Claude, GPT-4o, Gemini, any MCP-compatible agent
```

---

## Architecture

```
vectorquant/
  core/          — linalg, stats, prob, optimization, result, backend, C bridge
  finance/       — portfolio, risk_models, derivatives, covariance, factor_models
  stochastic/    — processes, monte_carlo, copulas
  time_series/   — analysis, regime_detection
  distributions/ — student_t, gpd, normal, fit
  ai/            — formula_registry, numeric_verifier, unit_checker, conventions_db,
                   sft_validator (planned), audit (planned), rag (planned),
                   agent_interceptor (planned), verify, proof_trace, tools,
                   reasoning, pipeline, verifier, formula_validator,
                   trace_generator, expression_parser, llm, agent_interface,
                   tool_registry
  mcp/           — MCP server (planned)
  serve/         — REST API (planned)

vectorquant-c/   — C extension source + built .pyd
scripts/research/— study_a_hallucination_prevalence.py (next task)
benchmarks/      — bench_c_vs_python, bench_ai_verification, bench_statistics
examples/        — quickstart, llm_verification, ai_verification, portfolio, derivatives
tests/           — 203 passing (target: 1000+)
```

### VQResult Schema (every public function returns this)

```python
@dataclass
class VQResult:
    value: Any
    verified: bool
    confidence: float          # 0.0–1.0
    formula_used: str
    formula_latex: str
    citation: str
    proof_trace: list[ProofStep]
    backend: str               # "python" | "C" | "numpy"
    computation_time_ms: float
    failure_mode: str | None   # "formula"|"convention"|"unit"|"distribution"|None
    unit_check: UnitCheckResult | None
    convention_used: dict | None
    warnings: list[str]

    @property
    def is_correct(self) -> bool:
        return self.verified and self.failure_mode is None
```

---

## Coding Rules

1. **Zero external dependencies** for core. NumPy is `[fast]` only. Never `import numpy` at module level.
2. **Every public function returns VQResult.** No bare floats, dicts, or tuples from public APIs.
3. **Formula registry entries** must have: `correct_expression`, `latex`, `citation`, `common_errors` (≥2 with severity), `unit`, `test_case` with known-correct `expected_value` and `tolerance`.
4. **Three backends must be bit-identical** within 1e-10. Backend divergence is a critical bug.
5. **Study A runs before new features ship.** No `vq.econometrics`, HRP, GPU, or streaming until FinMathBench v1 exists.
6. **Error patterns checked before correct expression** in `check_expression` — correct expression wins on tie.
7. **AI positioning first** — every doc, example, and README leads with verification use case.
8. **No backwards-compatibility shims**, feature flags, or provisional APIs. Build right or defer.

---

## Research Paper

- **Claim:** Four LLM hallucination failure modes in finance; VQ detects all four with 15–52% baseline hallucination rate confirmed across 37 models
- **Target:** NeurIPS "Datasets and Benchmarks" or ICML under LLM reliability/evaluation
- **Title:** *"Silent Numerical Hallucination in Financial AI Agents: Taxonomy, Measurement, and Mitigation"*
- **Key deliverable:** Study A dataset = FinMathBench v1 (50+ formulas × 3 LLMs × 4 failure modes)
- **Release:** HuggingFace Datasets for perpetual citation and discovery
- **arXiv first** before formal submission — establishes priority timestamp
- **Co-author:** Finance academic (convention DB) or ML researcher (NeurIPS methodology)
