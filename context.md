# VectorQuant — Project Context & Work Log

## Project Identity

**Library:** VectorQuant  
**Tagline:** A deterministic reasoning engine for AI and quantitative finance  
**Author:** Sahil Gupta, Mascot Universal Pvt Ltd  
**PyPI version:** 0.5.2 | **Spec:** v0.6.1 | **Tests:** 203 passing | **Target:** 0.8.0

**Primary positioning:** The only library providing deterministic numerical verification for AI/LLM financial calculations — for ChatGPT, Claude, Gemini, local models, fine-tuning pipelines, RAG systems, and agentic frameworks.

---

## Market Context

**Key stat:** Hallucination rate for structured financial calculations is **15–52% across 37 models (2026 benchmarks)**. No existing library catches this automatically.

**Competitive landscape:**

| Existing Tool | Guards Against | Misses |
|---|---|---|
| Guardrails AI, NeMo Guardrails | Prompt injection, PII, toxicity | Numerical correctness |
| LangChain guardrails | Output format, toxicity | Formula/convention/unit errors |
| Cleanlab / DataIQ | Text quality | Finance formula correctness in training data |
| Ragas / TruLens / Arize | RAG answer-vs-context consistency | Formula convention/unit verification |
| QuantLib / scipy | Correct calculation (if you call right function) | LLM hallucination detection, audit trail |

**VectorQuant owns the exact intersection:** deterministic numerical verification + AI/LLM integration.

### Five Critical Gaps (all unoccupied, validated April 2026)

1. **No deterministic verifier for finance math** — text guardrails are the only tooling; zero numerical verification exists
2. **Fine-tuning data has no financial math QA** — incorrect formulas in SFT datasets propagate into model weights; nobody checks this
3. **Agentic AI has no math audit trail** — McKinsey 2025: "embedding monitoring hooks into agent workflows" is the core shift; no library does this for financial math
4. **MCP has no finance-domain verification server** — first mover gets organic adoption from every agent developer
5. **Local LLMs have zero offline verifier** — banks and hedge funds cannot call cloud APIs; VQ's zero-dep C-extension design is the only viable option

### Regulatory Tailwind
EU AI Act (2025), FCA Consumer Duty, SEC AI guidance all require traceable, explainable AI decisions in finance. `VQResult.proof_trace` + `citation` + `convention_used` already produce exactly the data regulators require.

---

## Work History

### Phase 1 — Core Library (pre-conversation, v0.1–v0.5.2)
- Core math engine: `vq.linalg`, `vq.stats`, `vq.prob`, `vq.optim` (~85%)
- C extension (`vectorquant-c`): stats, linalg, stochastic, optimization, FFT, Kalman, sparse, QMC
- Finance: risk models, derivatives, portfolio, factor models
- Stochastic: GBM, Heston, Monte Carlo, copulas
- AI layer (v0.5.x): `verify_calculation`, `proof_trace`, OpenAI tool schemas
- 89+ tests, published to PyPI at v0.5.2

### Phase 2 — v0.6 AI Verification Layer (this conversation series)

**Step 1 — VQResult schema** (`vectorquant/core/result.py`)
- `VQResult` dataclass unified across all modules: `value`, `verified`, `confidence`, `formula_used`, `formula_latex`, `citation`, `proof_trace`, `backend`, `computation_time_ms`, `failure_mode`, `unit_check`, `convention_used`, `warnings`
- `.is_correct` property: `verified and failure_mode is None`

**Step 2 — Formula Registry** (`vectorquant/ai/formula_registry.py`)
- 30 entries with `correct_expression`, `latex`, `citation`, `common_errors` (≥2), `unit`, `test_case`
- Bidirectional: `get_formula(name)`, `match_expression(expr)`, `list_formulas()`
- Token-based Jaccard similarity with error-pattern priority logic

**Step 3 — Convention Database** (`vectorquant/ai/conventions_db.py`)
- 40+ instrument/market pairs: Treasuries, govt bonds, corp bonds, IRS, equity options, FX, futures
- `lookup(instrument, market)`, `check(instrument, market, llm_assumptions)`, `day_count_fraction()`

**Step 4 — Unit Checker** (`vectorquant/ai/unit_checker.py`)
- Level 6 detection: `Scale` enum, `VQUnit`, `UNITS` dict for all 30 formulas
- Detects wrong scale, missing annualisation, percentage vs ratio confusion

**Step 5 — Numeric Verifier** (`vectorquant/ai/numeric_verifier.py`)
- Level 2: `check_expression` — Jaccard matching, error patterns checked before correct expression
- Level 3: `verify_numeric` — deterministic recomputation via `_COMPUTE_MAP` (21 formulas)
- Bug fixed: correct expression wins when `sim_correct >= best_error_sim`

**Step 6 — Distributions** (`vectorquant/distributions/`)
- `StudentT`, `GPD`, `Normal`: pdf, cdf, ppf, var, cvar, fit
- `fit_all(returns)`: ranks by AIC/BIC
- Zero dependencies — all math utils in `_math_utils.py`

### Phase 3 — Repository Cleanup (April 2026)
125 files removed — scope-creep not in v0.6 spec:
- Deleted: `infrastructure/`, `research/`, 10 finance modules, 5 AI modules, 20+ redundant benchmarks
- Fixed: all `__init__.py` imports, `monte_carlo.py` dependency, tests updated
- Result: 203 passing tests. Commit `2ed9ac7`

### Phase 4 — Positioning: AI/LLM First (April 2026)
User clarified primary identity: AI/LLM verification, not general quant library.
- README rewritten: leads with AI use cases, multi-LLM integration, fine-tuning/RAG
- CLAUDE.md updated: AI-first vision, coding rule added
- Commit `346c35a`

### Phase 5 — Strategic Analysis (April 2026, this session)
Full competitive and market analysis completed. Key outputs:

**New modules planned (added to roadmap):**
- `vq.ai.sft_validator` — fine-tuning dataset QA (Priority 1, v0.6.0)
- `vq.mcp` — MCP server (Priority 1, v0.6.0)
- `vq.serve` — REST API with OpenAPI spec (Priority 1, v0.6.0)
- `vq.audit` — regulatory report generator for EU AI Act/SEC/FCA (Priority 1, v0.6.0)
- `vq.rag` — RAG post-retrieval verifier with Ragas/TruLens adapters (Priority 2, v0.7.0)
- `vq.agent.interceptor` — `@vq.verify` decorator for LangChain/CrewAI/AutoGen (Priority 2, v0.7.0)
- `vq.offline` — certified air-gapped bundle with signed checksum (Priority 2, v0.7.0)
- `vq.benchmark` — public live leaderboard dashboard (Priority 2, v0.7.0)
- `vq.stream` — real-time streaming verification SSE/WebSocket (Priority 3, v0.8.0)
- `vq.medical`, `vq.legal` — domain expansion (Priority 3, v0.9.0)

**Formula registry expansion targets:**
- 30 (current) → 100 (pre-Study A) → 300 (v0.7.0) → 500+ (v0.8.0+)
- New domains: fixed income (duration, Z-spread, OAS), credit (PD, LGD, EAD, Basel), crypto (funding rates, IL for AMMs), macro (GDP deflator, Taylor rule), actuarial

**FinMathBench v1:**
- First dataset testing finance formula hallucination across LLMs
- Referenced by 2026 survey (arxiv 2510.06265) as open challenge
- Release to HuggingFace Datasets = perpetual citation source
- Positions VQ as the measurement standard for the domain

**Three-phase timeline:**
- Phase 1 (now → June 2026, v0.6.0): Study A + registry expansion + MCP + SFT validator + audit + REST API
- Phase 2 (June → Dec 2026, v0.7.0): RAG verifier + agent interceptor + offline bundle + leaderboard + econometrics
- Phase 3 (2027, v0.9.0+): Medical/legal domains + streaming + GPU + 500+ formulas

---

## Key Decisions

| Decision | Rationale |
|---|---|
| Study A + FinMathBench v1 before any new features | Published benchmark = measurement standard = automatic library adoption |
| `vq.audit` is Priority 1 (not deferred) | EU AI Act compliance demand converts directly to enterprise revenue |
| `vq.mcp` is Priority 1 | First mover in MCP finance verification = organic adoption from every agent dev |
| `vq.ai.sft_validator` is Priority 1 | Completely unoccupied niche; AI teams lose 40–60% time on data quality |
| Formula registry target revised to 500 (not 100) | Research moat; 500 entries = no competitor can replicate quickly |
| `vq.offline` explicitly marketed to banks/hedge funds | Zero-dep + C extension = only viable air-gapped option; they will pay |
| AI/LLM verification is the product; quant math is the substrate | Repositions vs QuantLib (calculation engine); VQ is verification layer |
| Error patterns checked before correct expression, correct wins on tie | Prevents correct formula from falsely matching error superset |

---

## Current Repository Structure

```
VectorQuant/
├── CLAUDE.md                    ← Claude Code instructions (read first every session)
├── context.md                   ← this file
├── README.md                    ← AI-first public docs
├── LICENSE, pyproject.toml
│
├── vectorquant/
│   ├── core/        result.py, linalg, stats, prob, optim, backend, C bridge
│   ├── finance/     portfolio, risk_models, derivatives, covariance, factor_models
│   ├── stochastic/  processes, monte_carlo, copulas
│   ├── time_series/ analysis, regime_detection
│   ├── distributions/ student_t, gpd, normal, fit  ← new v0.6
│   └── ai/          formula_registry, numeric_verifier, unit_checker,
│                    conventions_db, verify, proof_trace, tools, reasoning,
│                    pipeline, verifier, formula_validator, trace_generator,
│                    expression_parser, llm, agent_interface, tool_registry
│
├── vectorquant-c/   C extension (165x speedup)
├── scripts/         (empty — study_a script goes here next)
├── benchmarks/      5 files
├── examples/        7 files
└── tests/           203 passing
```

---

## Immediate Next Steps (in order)

1. **`scripts/research/study_a_hallucination_prevalence.py`** — loop FORMULA_REGISTRY × GPT-4o + Claude + Gemini → CSV
2. **Formula registry: 30 → 100 entries** — fixed income first (duration, DV01, YTM, convexity), then credit, then Greeks
3. **arXiv preprint** — establish priority timestamp
4. **FinMathBench v1 on HuggingFace Datasets**
5. **`vq.mcp`** — MCP server, first in ecosystem for financial verification
6. **`vq.ai.sft_validator`** — finance SFT dataset QA
7. **`vq.audit`** — EU AI Act / SEC / FCA report generator
