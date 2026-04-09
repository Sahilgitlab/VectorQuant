# VectorQuant — Claude Code Project Instructions

## Vision

**A deterministic reasoning engine for AI and quantitative finance.**

VectorQuant is the **verified mathematical ground-truth layer for AI systems operating in finance**. Its primary audience is AI engineers, not quant researchers. The quant math is the substrate. The verification layer is the product.

```
Primary users:
  - LLM engineers building finance agents (ChatGPT, Claude, Gemini, Llama)
  - Fine-tuning teams generating verified training data for finance models
  - RAG systems retrieving and reasoning about financial formulas and conventions
  - Agentic AI frameworks (LangChain, LlamaIndex, AutoGen, CrewAI, MCP)

The quant math is there because verification requires correct ground truth.
The quant math is not the product. The verification is.
```

---

## The Core Thesis (Never Drift From This)

LLMs hallucinate financial calculations in four distinct failure modes. No existing library catches them automatically. VectorQuant does.

```
Four failure modes:
1. Wrong formula      → vq.ai.formula_registry   (Sharpe: variance vs std → 6000% error)
2. Wrong convention   → vq.ai.conventions         (USD T-bill: Act/365 vs Act/360)
3. Wrong units/scale  → vq.ai.unit_checker        (0.003 returned for annualised Sharpe)
4. Wrong distribution → vq.distributions          (Normal vs Student's t → 53% VaR underestimate)
```

When writing code, reviewing PRs, or planning features — always ask: "Does this make the hallucination detection better?" If not, it is secondary.

---

## Current State (April 2026)

**Version:** 0.5.2 (PyPI) | **Spec:** v0.6.1 | **Tests:** 203 passing | **Target:** 0.8.0

### Built

| Module | Status |
|---|---|
| Core math engine (`vq.linalg`, `vq.stats`, `vq.prob`, `vq.optim`) | ~85% |
| C engine (`vectorquant-c`) — 165x on covered ops | ~60% coverage |
| `vq.finance` (risk, derivatives, portfolio, factor models, covariance) | ~75% |
| `vq.stochastic` (GBM, Heston, Monte Carlo) | ~80% |
| `vq.time_series` (HMM, ARIMA, rolling stats) | ~70% |
| `vq.distributions` (StudentT, GPD, Normal, fit_all) | Complete |
| `vq.ai.formula_registry` — 30 entries, bidirectional lookup | Complete |
| `vq.ai.numeric_verifier` — verify_numeric + check_expression (Levels 2/3) | Complete |
| `vq.ai.conventions_db` — 40+ instrument/market pairs | Complete |
| `vq.ai.unit_checker` — dimensional analysis (Level 6) | Complete |
| OpenAI tool schemas | Partial |

### Not Yet Built

- Formula registry: 30 → 100+ entries (needed before research paper)
- `vq.tests` (Jarque-Bera, ADF, KPSS, Ljung-Box, etc.) — v0.6.0
- Anthropic, Gemini, LangChain, LlamaIndex tool formats — v0.6.0
- MCP server — v0.6.0
- Hallucination Levels 4/5 (code verification, full convention integration)
- `vq.econometrics` (OLS, VAR, VECM) — v0.7.0
- **Study A script** (`scripts/research/study_a_hallucination_prevalence.py`) — next task

---

## Next Task

`scripts/research/study_a_hallucination_prevalence.py`

Loops over `FORMULA_REGISTRY`, prompts GPT-4o + Claude Sonnet + Gemini Pro, runs `vq.ai.verify_numeric()` on each LLM output, writes CSV → `FinMathBench v1`.

This is the empirical core of the research paper. Build it next.

---

## Architecture

```
vectorquant/
  core/          — linalg, stats, prob, optimization, result, backend, C bridge
  finance/       — portfolio, risk_models, derivatives, covariance, factor_models
  stochastic/    — processes, monte_carlo, copulas
  time_series/   — analysis, regime_detection
  distributions/ — student_t, gpd, normal, fit       ← new in v0.6
  ai/            — formula_registry, numeric_verifier, unit_checker,
                   conventions_db, verify, proof_trace, tools, reasoning,
                   pipeline, verifier, formula_validator, trace_generator,
                   expression_parser, llm, agent_interface, tool_registry

vectorquant-c/   — C extension source + built .pyd
benchmarks/      — 5 focused benchmark files
examples/        — 7 examples (AI verification leads)
tests/           — 203 passing
```

### Key Public API

```python
import vectorquant as vq

# Primary: AI Verification
vq.ai.verify_numeric("sharpe_ratio", llm_value=2.45, inputs={...})
vq.ai.check_expression("sharpe_ratio", "mean(r) / variance(r)")
vq.ai.check_formula("sharpe_ratio", "(mean(r) - rf) / std(r)")
vq.ai.conventions.lookup("treasury_bill", "USD")
vq.ai.conventions.check("corporate_bond", "USD", {"day_count": "Act/365"})
vq.ai.unit_checker.check(value=0.003, formula="sharpe_ratio", question="...")
vq.ai.formula_registry.get_formula("sharpe_ratio")
vq.ai.formula_registry.match_expression("mean(r) / variance(r)")

# Secondary: Verified Finance Math
vq.risk.parametric_var(returns, confidence_level=0.95)
vq.derivatives.black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1.0)
vq.portfolio.optimize_max_sharpe(expected_returns, cov_matrix)
vq.distributions.StudentT(df=5).var(confidence=0.99)
vq.distributions.fit_all(returns)
```

### VQResult Schema (every function returns this)

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

## Roadmap

### v0.5.3 — Foundation Complete (current)
- [x] VQResult unified schema
- [x] Formula registry (30 entries)
- [x] Convention database (40+ entries)
- [x] Unit checker (Level 6)
- [x] verify_numeric / check_expression (Levels 2/3)
- [x] vq.distributions (StudentT, GPD, Normal, fit_all)
- [x] Repository cleanup
- [ ] Study A script → FinMathBench v1
- [ ] Formula registry 30 → 100+ entries

### v0.6.0 — AI Verification Core (target: June 2026)
- [ ] Anthropic + Gemini tool schemas
- [ ] LangChain + LlamaIndex tool wrappers
- [ ] MCP server (`pip install vectorquant[mcp]`)
- [ ] `vq.tests` module (Jarque-Bera, ADF, KPSS, Ljung-Box, Breusch-Pagan, cointegration)
- [ ] Code verification (Level 4)
- [ ] NumPy acceleration tier (`pip install vectorquant[fast]`)
- [ ] REST API server (`vq.serve()`)

### v0.7.0 — Mathematics Expansion
- [ ] `vq.econometrics`: OLS (Newey-West), VAR, VECM, Fama-MacBeth, Granger causality
- [ ] Full fixed income: yield curve bootstrap, forward rates, Z-spread, OAS
- [ ] Advanced portfolio: HRP, robust optimisation, transaction cost optimisation
- [ ] GARCH with GED/skew-t errors, SABR, variance swaps
- [ ] Regime detection: HMM with Bai-Perron structural break tests

### v0.8.0 — Performance
- [ ] GPU acceleration (CuPy)
- [ ] Distributed Monte Carlo
- [ ] Streaming VQResult for real-time risk systems

---

## Coding Rules

1. **Zero external dependencies** for core. NumPy is `[fast]` only. Never `import numpy` at module level.
2. **Every public function returns VQResult.** No bare floats, dicts, or tuples from public APIs.
3. **Formula registry entries** must have: `correct_expression`, `latex`, `citation`, `common_errors` (≥2 with severity), `unit`, `test_case` with known-correct `expected_value` and `tolerance`.
4. **Three backends must be bit-identical** within 1e-10. Backend divergence is a critical bug.
5. **Do not build** `vq.econometrics`, `vq.tests`, HRP, GPU, or GARCH until Study A script runs on 50+ registry entries. Paper first.
6. **No backwards-compatibility shims**, feature flags, or provisional APIs. Build it right or defer.
7. **Error patterns checked before correct expression** in `check_expression` — correct expression wins on tie (see `numeric_verifier.py` logic).
8. **AI positioning first** — when writing examples, docs, or READMEs, lead with the verification use case, not the quant math.

---

## Research Paper

- **Claim:** Four distinct LLM hallucination failure modes in quantitative finance; VectorQuant detects all four automatically
- **Target:** NeurIPS "Datasets and Benchmarks" or ICML under LLM reliability/evaluation
- **Title:** *"Silent Numerical Hallucination in Financial AI Agents: Taxonomy, Measurement, and Mitigation"*
- **Key deliverable:** Study A dataset (50+ formulas × 3 LLMs → CSV) = FinMathBench v1
- **arXiv first** (cs.AI or q-fin.CP) before formal submission to establish priority
- **Co-author:** Finance academic (convention DB credibility) or ML researcher (NeurIPS methodology)
