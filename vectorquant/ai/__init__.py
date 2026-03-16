"""
VectorQuant AI — Autonomous Decision Intelligence

Provides strategy scoring, reinforcement learning allocation,
strategy lifecycle management, and decision explainability.
"""

from .decision_engine import (
    score_strategy, rank_strategies,
    dynamic_regime_allocation,
)

from .rl_allocation import AllocationEnv, BasicQTable

from .strategy_lifecycle import LifecycleState, StrategyLifecycle

from .explainability import explain_decision

from .asset_universe import AssetUniverse, AssetData

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

from .hallucination_check import (
    HallucinationResult,
    check_formula,
    check_numerical_claim,
    validate_prediction,
)

from .tools import get_tool_registry, execute_tool, get_tool_schemas

from .reasoning import ReasoningEngine, ReasoningResult

from .llm import LLMInterface

from .pipeline import HallucinationProofPipeline, PipelineResult

# Phase 9.1: Agent Protocol
from .agent_interface import (
    VectorQuantTool, ComputationResult, ParameterValidator
)

from .tool_registry import (
    ToolRegistry, get_registry, get_all_tools
)

# Phase 9.2: Verification Pipeline
from .expression_parser import (
    parse_expression, ExpressionParser, ExpressionTokenizer,
    ExpressionValidator, ParsedExpression, FunctionCall
)

from .verifier import (
    VerificationPipeline, VerificationReport,
    ExpressionExtractor, StageExecutor, StageComparator,
    verify_llm_statement, get_verifier
)

# Phase 9.3: Formula Validator
from .formula_validator import (
    FormulaValidator, ValidationResult, FormulaError,
    DimensionValidator, validate_formula, ErrorType
)

# Phase 9.4: Trace & Proof Generation
from .trace_generator import (
    ComputationTracer, ComputationTrace, ProofTree, ProofStep,
    ExplainabilityReporter, trace_and_explain
)

# Phase 9.1: Agent Protocol
from .agent_interface import (
    VectorQuantTool,
    ComputationResult,
    ComputationMetadata,
    ParameterValidator,
)

from .tool_registry import (
    ToolRegistry,
    get_registry,
    get_all_tools,
    get_tool_schema,
    execute_tool as registry_execute_tool,
    list_tools_full,
    search_tools,
)
