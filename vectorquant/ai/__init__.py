"""
VectorQuant AI — Hallucination Verification Layer

Catches LLM hallucinations in quantitative finance calculations across
four failure modes: wrong formula, wrong convention, wrong units,
wrong distribution assumption.

    vq.ai.verify_numeric("sharpe_ratio", llm_value=2.45, inputs={...})
    vq.ai.check_expression("sharpe_ratio", "mean(r)/variance(r)")
    vq.ai.conventions.lookup("treasury_bill", "USD")
    vq.ai.unit_checker.check(0.003, "sharpe_ratio")
    vq.ai.formula_registry.get_formula("sharpe_ratio")
"""

# ── Core verification (proof traces, LLM tool execution) ─────────────────────

from .verify import (
    VerificationResult,
    verify_calculation,
    verify_probability,
    verify_finance_formula,
)

from .proof_trace import (
    ExplanationTrace,
    explain_var,
    explain_sharpe,
    explain_black_scholes,
    explain_monte_carlo,
)

from .tools import get_tool_registry, execute_tool, get_tool_schemas

from .reasoning import ReasoningEngine, ReasoningResult

# ── v0.6 AI Verification Layer ────────────────────────────────────────────────

# Formula registry — bidirectional lookup, 30+ entries with citations
from . import formula_registry
from .formula_registry import (
    get_formula,
    list_formulas,
    match_expression,
    FORMULA_REGISTRY,
)

# Level 2/3 — Expression + Numeric Hallucination Detection
from .numeric_verifier import (
    verify_numeric,
    check_expression,
    check_formula,
)

# Level 5 — Convention Check
from . import conventions_db as conventions

# Level 6 — Unit / Dimensional Check
from . import unit_checker

from .llm import LLMInterface

from .pipeline import HallucinationProofPipeline, PipelineResult

# ── Agent Protocol (v0.5.x) ───────────────────────────────────────────────────

from .agent_interface import (
    VectorQuantTool, ComputationResult, ComputationMetadata, ParameterValidator,
)

from .tool_registry import (
    ToolRegistry, get_registry, get_all_tools,
    get_tool_schema, execute_tool as registry_execute_tool,
    list_tools_full, search_tools,
)

# ── Expression parsing and verification pipeline ──────────────────────────────

from .expression_parser import (
    parse_expression, ExpressionParser, ExpressionTokenizer,
    ExpressionValidator, ParsedExpression, FunctionCall,
)

from .verifier import (
    VerificationPipeline, VerificationReport,
    ExpressionExtractor, StageExecutor, StageComparator,
    verify_llm_statement, get_verifier,
)

# ── Formula validation and proof generation ───────────────────────────────────

from .formula_validator import (
    FormulaValidator, ValidationResult, FormulaError,
    DimensionValidator, validate_formula, ErrorType,
)

from .trace_generator import (
    ComputationTracer, ComputationTrace, ProofTree,
    ExplainabilityReporter, trace_and_explain,
)
