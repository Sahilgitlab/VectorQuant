# VectorQuant — Project Context & Work Log

## Project Identity

**Library:** VectorQuant  
**Tagline:** A deterministic reasoning engine for AI and quantitative finance  
**Author:** Sahil Gupta, Mascot Universal Pvt Ltd  
**PyPI version:** 0.5.2 (published)  
**Spec version:** v0.6.1  
**Tests:** 203 passing, 1 skipped  
**Target version:** 0.8.0  

**Primary positioning:** AI/LLM verification layer — for ChatGPT, Claude, Gemini, agentic AI, fine-tuning pipelines, and RAG systems in finance. The quant math is the substrate; the verification is the product.

---

## The Problem This Solves

LLMs hallucinate financial calculations in four distinct failure modes:

| Mode | Example | Magnitude |
|---|---|---|
| Wrong formula | Sharpe: `return / variance` instead of `(return - rf) / std` | 6000%+ |
| Wrong convention | USD T-bill: `Act/365` instead of `Act/360`; yield vs discount rate | ~2bp silent |
| Wrong units | Annualised Sharpe returned as `0.003` (forgot `× √252`) | 15x off |
| Wrong distribution | VaR using Normal instead of Student's t (df=4) | 53% underestimate |

VectorQuant catches all four automatically, with proof traces, confidence scores, and academic citations. No other library does this.

---

## Work History

### Phase 1 — Core Library (pre-conversation, v0.1–v0.5.2)
Built before this conversation series:
- Core math: `vq.linalg`, `vq.stats`, `vq.prob`, `vq.optim` (~85%)
- C extension (`vectorquant-c`): stats, linalg, stochastic, optimization, FFT, Kalman, sparse, QMC
- Finance: risk models (VaR, CVaR), derivatives (Black-Scholes, Greeks, IV), portfolio (max-Sharpe, Black-Litterman), factor models
- Stochastic: GBM, Heston, Monte Carlo engine, copulas
- Time series: HMM, ARIMA scaffolding, regime detection
- AI layer (v0.5.x): `verify_calculation`, `proof_trace`, OpenAI tool schemas, `LLMInterface`
- 89+ tests, published to PyPI

### Phase 2 — v0.6 AI Verification Layer
Built in this conversation series, per spec Section 10:

**Step 1 — VQResult schema** (`vectorquant/core/result.py`)
- `VQResult` dataclass: `value`, `verified`, `confidence`, `formula_used`, `formula_latex`, `citation`, `proof_trace`, `backend`, `computation_time_ms`, `failure_mode`, `unit_check`, `convention_used`, `warnings`
- `ProofStep`, `UnitCheckResult` dataclasses
- `.is_correct` property: `verified and failure_mode is None`

**Step 2 — Formula Registry** (`vectorquant/ai/formula_registry.py`)
- 30 entries: Sharpe, Sortino, VaR×2, CVaR, max drawdown, Calmar, annualised vol, CAGR, BS call/put, portfolio return/variance, beta, Jensen's alpha, Treynor, Kelly, log return, correlation Pearson, modified duration, DV01, tracking error, information ratio
- Each entry: `correct_expression`, `latex`, `citation`, `common_errors` (≥2 with severity), `unit` annotation, `test_case`
- Bidirectional lookup: `get_formula(name)`, `match_expression(expr)`, `list_formulas()`
- Token-based Jaccard similarity matcher

**Step 3 — Convention Database** (`vectorquant/ai/conventions_db.py`)
- 40+ instrument/market pairs: US Treasuries, EUR/GBP/JPY govts, corp bonds, money market, IRS (USD/EUR/GBP), equity options, FX forwards, futures
- `lookup(instrument, market)`, `check(instrument, market, llm_assumptions)`, `day_count_fraction(start, end, convention)`

**Step 4 — Unit Checker** (`vectorquant/ai/unit_checker.py`)
- Level 6 detection: `Scale` enum, `VQUnit` dataclass, `UNITS` dict for all 30 formulas
- Detects: wrong scale, missing annualisation, percentage vs ratio confusion
- Returns `VQResult` with `verified=False` and corrected value suggestion

**Step 5 — Numeric Verifier** (`vectorquant/ai/numeric_verifier.py`)
- Level 2: `check_expression` — token Jaccard matching against correct expression and known error patterns
- Level 3: `verify_numeric` — deterministic recomputation via `_COMPUTE_MAP` (21 formulas)
- **Key bug fixed:** expression checker checks error patterns before correct expression, correct expression wins when `sim_correct >= best_error_sim`

**Step 6 — Distributions** (`vectorquant/distributions/`)
- `StudentT`, `GPD`, `Normal`: pdf, cdf, ppf, var, cvar (closed-form ES), fit
- `fit_all(returns)`: ranks all distributions by AIC/BIC
- Zero external dependencies — gamma, incomplete beta, Brent's method all in `_math_utils.py`

### Phase 3 — Repository Cleanup (April 2026)
Removed 125 files — scope-creep not in v0.6 spec:

**Deleted modules:**
- `vectorquant/ai/`: asset_universe, decision_engine, explainability, rl_allocation, strategy_lifecycle, hallucination_check
- `vectorquant/finance/`: decision_theory, extreme_value_theory, macro_models, market_microstructure, network_theory, risk_monitoring, risk_attribution, risk_parity, stress_testing, volatility_surface
- `vectorquant/core/`: symbolic_math
- `vectorquant/infrastructure/`: entire folder
- `vectorquant/research/`: entire folder
- 20+ redundant benchmarks, JSON artifacts, duplicate examples, private spec docs

**Fixed after cleanup:**
- `__init__.py` files for `vectorquant/`, `finance/`, `ai/` — removed deleted imports, added v0.6 exports
- `stochastic/monte_carlo.py` — removed infrastructure dependency
- Tests updated: replaced deleted-module tests with v0.6 verification layer tests
- Expression checker logic fixed (error pattern priority)

**Result:** 203 tests passing. Commit `2ed9ac7`.

### Phase 4 — Positioning Sharpened (April 2026, this session)
User clarified: VectorQuant's primary identity is **AI/LLM verification**, not a general quant library.

- README.md rewritten: leads with AI use cases, multi-LLM integration, fine-tuning/RAG patterns, hallucination detection — quant math moved to "supporting cast" section
- CLAUDE.md updated: AI-first vision stated clearly, coding rule added (AI positioning leads in all docs/examples)
- context.md: updated to reflect sharpened positioning

---

## Key Decisions

| Decision | Rationale |
|---|---|
| AI/LLM verification is the product; quant math is the substrate | User direction. Repositions vs QuantLib (a calculation engine, not a verification layer) |
| Expression checker: error patterns before correct, correct wins on tie | Prevents correct formula from falsely matching error pattern (variance-denominator case) |
| `vq.distributions` in v0.6 not v0.7 | Student's t needed for distribution-aware VaR — load-bearing for paper thesis |
| `vq.econometrics` deferred to v0.7 | Not needed for Study A; build after convention database ships |
| `infrastructure/` fully removed | Parallel simulation not in spec; MC falls back to serial |
| Zero-dependency constraint enforced | Core must install anywhere; numpy is `[fast]` optional |
| All public functions return `VQResult` | Unified schema enables downstream verification, proof traces, and LLM tool responses |

---

## Current Repository Structure

```
VectorQuant/
├── CLAUDE.md                    ← Claude Code project instructions (read first)
├── context.md                   ← this file (work history + decisions)
├── README.md                    ← AI-first public documentation
├── LICENSE
├── pyproject.toml
├── .github/workflows/publish.yml
│
├── vectorquant/
│   ├── core/        result.py, linalg, stats, prob, optim, backend, C bridge
│   ├── finance/     portfolio, risk_models, derivatives, covariance, factor_models
│   ├── stochastic/  processes, monte_carlo, copulas
│   ├── time_series/ analysis, regime_detection
│   ├── distributions/ student_t, gpd, normal, fit    ← new v0.6
│   └── ai/          formula_registry, numeric_verifier, unit_checker,
│                    conventions_db, verify, proof_trace, tools, reasoning,
│                    pipeline, verifier, formula_validator, trace_generator,
│                    expression_parser, llm, agent_interface, tool_registry
│
├── vectorquant-c/   C extension source + built .pyd (165x speedup)
├── benchmarks/      5 files: c_vs_python, ai_verification, statistics, runner
├── examples/        7 files: quickstart, llm_verification, ai_verification,
│                    portfolio, derivatives, monte_carlo, monte_carlo_safe
└── tests/           203 passing
```

---

## Next Task

**Build `scripts/research/study_a_hallucination_prevalence.py`**

```python
# For each formula in FORMULA_REGISTRY:
#   Prompt GPT-4o, Claude Sonnet, Gemini Pro with the test_case inputs
#   Run vq.ai.verify_numeric(formula, llm_value, inputs)
#   Write row to data/study_a_results.csv
#
# Output: FinMathBench v1 — the empirical core of the research paper
```

Prerequisite: formula registry should have 50+ entries before running at scale.

---

## Research Paper

- **Thesis:** Four failure modes by which LLMs hallucinate in quantitative finance; VQ detects all four
- **Venue:** NeurIPS (Datasets & Benchmarks) or ICML
- **Title:** *"Silent Numerical Hallucination in Financial AI Agents: Taxonomy, Measurement, and Mitigation"*
- **Blocker:** Study A dataset (50+ formulas × 3 LLMs × error classification)
- **Deliverable:** `FinMathBench v1` CSV — becomes a lasting benchmark independent of the library
- **arXiv:** Post before formal submission to establish timestamp
