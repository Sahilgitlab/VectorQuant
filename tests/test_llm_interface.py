"""
LLM Interface and Pipeline Tests

Tests the JSON tool schemas, LLMInterface execute method,
and the HallucinationProofPipeline.
"""

import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq

# ─── JSON Tool Schemas ──────────────────────────────────────────────────

def test_tool_schemas_exist():
    schemas = vq.ai.get_tool_schemas()
    assert isinstance(schemas, list)
    assert len(schemas) >= 7

def test_tool_schema_format():
    schemas = vq.ai.get_tool_schemas()
    for schema in schemas:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"
        assert "properties" in schema["function"]["parameters"]

def test_tool_schema_has_required_fields():
    schemas = vq.ai.get_tool_schemas()
    var_schema = [s for s in schemas if s["function"]["name"] == "calculate_var"][0]
    assert "required" in var_schema["function"]["parameters"]
    assert "returns" in var_schema["function"]["parameters"]["required"]


# ─── LLM Interface ──────────────────────────────────────────────────────

def test_llm_interface_execute_var():
    llm = vq.ai.LLMInterface()
    result = llm.execute("calculate_var", returns=[0.01, -0.02, 0.015, -0.005, 0.008])
    assert "value" in result
    assert "verified" in result
    assert "proof_trace" in result
    assert isinstance(result["value"], float)

def test_llm_interface_execute_sharpe():
    llm = vq.ai.LLMInterface()
    result = llm.execute("compute_sharpe", returns=[0.01, -0.02, 0.015, -0.005, 0.008])
    assert result["verified"] == True
    assert result["proof_trace"] is not None

def test_llm_interface_execute_bs():
    llm = vq.ai.LLMInterface()
    result = llm.execute("price_call_option", S=100, K=100, r=0.05, sigma=0.2, T=1.0)
    assert result["verified"] == True
    assert result["value"] > 0

def test_llm_interface_openai_tools():
    tools = vq.ai.LLMInterface.get_openai_tools()
    assert isinstance(tools, list)
    assert len(tools) >= 7

def test_llm_interface_langchain_tools():
    tools = vq.ai.LLMInterface.get_langchain_tools()
    assert isinstance(tools, list)
    assert len(tools) >= 7


# ─── Hallucination-Proof Pipeline ───────────────────────────────────────

def test_pipeline_var():
    pipeline = vq.ai.HallucinationProofPipeline()
    result = pipeline.process("var", returns=[0.01, -0.02, 0.015, -0.005, 0.008], confidence_level=0.95)
    assert result.verified == True
    assert result.confidence == 1.0
    assert result.result is not None

def test_pipeline_sharpe():
    pipeline = vq.ai.HallucinationProofPipeline()
    result = pipeline.process("sharpe", returns=[0.01, -0.02, 0.015, -0.005, 0.008])
    assert result.verified == True

def test_pipeline_unknown_intent():
    pipeline = vq.ai.HallucinationProofPipeline()
    result = pipeline.process("unknown_intent")
    assert result.verified == False
    assert result.error is not None

def test_pipeline_result_dict():
    pipeline = vq.ai.HallucinationProofPipeline()
    result = pipeline.process("sharpe", returns=[0.01, 0.02, 0.015])
    d = result.to_dict()
    assert "intent" in d
    assert "result" in d
    assert "verified" in d
    assert "confidence" in d
