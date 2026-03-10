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
