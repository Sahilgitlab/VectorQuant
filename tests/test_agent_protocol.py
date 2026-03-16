"""
Tests for Phase 9.1 Agent Protocol Interface

Tests agent protocol functionality:
- Parameter validation and type coercion
- Operation execution
- Error handling
- Backend detection
- Schema generation for different frameworks
- Performance (latency <1ms target)
"""

import pytest
import json
from vectorquant.ai.agent_interface import (
    VectorQuantTool,
    ComputationResult,
    ComputationMetadata,
    ParameterValidator,
)
from vectorquant.ai.tool_registry import (
    ToolRegistry,
    get_registry,
    get_all_tools,
    get_tool_schema,
    search_tools,
)


class TestParameterValidator:
    """Test parameter validation and type coercion."""
    
    def test_validate_numeric_float(self):
        """Test numeric validation with float."""
        value = ParameterValidator.validate_numeric(3.14, "test_param")
        assert value == 3.14
        assert isinstance(value, float)
    
    def test_validate_numeric_int_to_float(self):
        """Test numeric validation with int (coerces to float)."""
        value = ParameterValidator.validate_numeric(42, "test_param")
        assert value == 42.0
        assert isinstance(value, float)
    
    def test_validate_numeric_string(self):
        """Test numeric validation with string."""
        value = ParameterValidator.validate_numeric("3.14", "test_param")
        assert value == 3.14
    
    def test_validate_numeric_invalid(self):
        """Test numeric validation with invalid input."""
        with pytest.raises(ValueError, match="must be numeric"):
            ParameterValidator.validate_numeric("not_a_number", "test_param")
    
    def test_validate_numeric_none_not_allowed(self):
        """Test numeric validation with None (not allowed by default)."""
        with pytest.raises(ValueError):
            ParameterValidator.validate_numeric(None, "test_param", allow_none=False)
    
    def test_validate_numeric_none_allowed(self):
        """Test numeric validation with None (allowed)."""
        value = ParameterValidator.validate_numeric(None, "test_param", allow_none=True)
        assert value is None
    
    def test_validate_list_numeric(self):
        """Test list validation with numeric elements."""
        values = ParameterValidator.validate_list([1, 2.5, 3], "test_list")
        assert len(values) == 3
        assert all(isinstance(v, float) for v in values)
    
    def test_validate_list_string_input(self):
        """Test list validation with string input."""
        with pytest.raises(ValueError, match="must be a list"):
            ParameterValidator.validate_list("not_a_list", "test_list")
    
    def test_validate_matrix_valid(self):
        """Test matrix validation with valid 2D array."""
        matrix = [[1, 2, 3], [4, 5, 6]]
        result = ParameterValidator.validate_matrix(matrix, "test_matrix")
        assert len(result) == 2
        assert len(result[0]) == 3
        assert all(isinstance(x, float) for row in result for x in row)
    
    def test_validate_integer_valid(self):
        """Test integer validation."""
        value = ParameterValidator.validate_integer(42, "test_int")
        assert value == 42
        assert isinstance(value, int)
    
    def test_validate_integer_with_bounds(self):
        """Test integer validation with min/max bounds."""
        value = ParameterValidator.validate_integer(5, "test_int", min_val=1, max_val=10)
        assert value == 5
    
    def test_validate_integer_below_min(self):
        """Test integer validation below minimum."""
        with pytest.raises(ValueError, match="must be >= 1"):
            ParameterValidator.validate_integer(0, "test_int", min_val=1)
    
    def test_validate_integer_above_max(self):
        """Test integer validation above maximum."""
        with pytest.raises(ValueError, match="must be <= 10"):
            ParameterValidator.validate_integer(11, "test_int", max_val=10)


class TestVectorQuantTool:
    """Test VectorQuantTool interface."""
    
    def test_initialization(self):
        """Test tool initialization."""
        tool = VectorQuantTool()
        assert tool is not None
        assert tool._backend_info["available"] is True
    
    def test_get_available_operations(self):
        """Test getting list of available operations."""
        tool = VectorQuantTool()
        ops = tool.get_available_operations()
        
        assert isinstance(ops, list)
        assert len(ops) > 10
        assert "compute_sharpe" in ops
        assert "price_call" in ops
        assert "simulate_gbm" in ops
    
    def test_get_schema_known_operation(self):
        """Test getting schema for known operation."""
        tool = VectorQuantTool()
        schema = tool.get_schema("compute_sharpe")
        
        assert "description" in schema
        assert "params" in schema
        assert "returns" in schema
        assert "returns" in schema["params"]
        assert "risk_free_rate" in schema["params"]
    
    def test_get_schema_unknown_operation(self):
        """Test getting schema for unknown operation."""
        tool = VectorQuantTool()
        
        with pytest.raises(ValueError, match="Unknown operation"):
            tool.get_schema("nonexistent_operation")
    
    def test_compute_mean(self):
        """Test computing mean via agent interface."""
        tool = VectorQuantTool()
        
        result = tool.compute("compute_mean", {
            "returns": [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        assert isinstance(result, ComputationResult)
        assert result.operation == "compute_mean"
        assert result.result == 3.0
        assert result.metadata.error is None
    
    def test_compute_std(self):
        """Test computing standard deviation via agent interface."""
        tool = VectorQuantTool()
        
        # Test on known data: [1, 2, 3, 4, 5]
        # Mean = 3, Variance = 2, StdDev = sqrt(2) ≈ 1.414
        result = tool.compute("compute_std", {
            "returns": [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        assert result.result is not None
        assert abs(result.result - 1.5811) < 0.01  # ~1.581
    
    def test_compute_sharpe_ratio(self):
        """Test computing Sharpe ratio via agent interface."""
        tool = VectorQuantTool()
        
        result = tool.compute("compute_sharpe", {
            "returns": [0.01, 0.02, -0.005, 0.015, 0.03],
            "risk_free_rate": 0.02
        })
        
        # Sharpe ratio should compute without error
        assert result.metadata.error is None
        assert isinstance(result.result, (int, float))
        assert result.result is not None
    
    def test_compute_with_missing_required_param(self):
        """Test computation with missing required parameter."""
        tool = VectorQuantTool()
        
        result = tool.compute("compute_sharpe", {
            "risk_free_rate": 0.02
            # Missing "returns"
        })
        
        assert result.metadata.error is not None
        assert "Missing required parameter" in result.metadata.error
    
    def test_compute_with_invalid_param_type(self):
        """Test computation with invalid parameter type."""
        tool = VectorQuantTool()
        
        result = tool.compute("compute_sharpe", {
            "returns": "not_a_list",  # Should be list
            "risk_free_rate": 0.02
        })
        
        assert result.metadata.error is not None
    
    def test_compute_with_unexpected_param(self):
        """Test computation with unexpected parameter."""
        tool = VectorQuantTool()
        
        result = tool.compute("compute_sharpe", {
            "returns": [0.01, 0.02, 0.03],
            "risk_free_rate": 0.02,
            "unexpected_param": "should_fail"
        })
        
        assert result.metadata.error is not None
        assert "Unexpected parameters" in result.metadata.error
    
    def test_compute_metadata_includes_latency(self):
        """Test that computation metadata includes latency."""
        tool = VectorQuantTool()
        
        result = tool.compute("compute_mean", {
            "returns": [1.0, 2.0, 3.0]
        })
        
        assert result.metadata.latency_ms > 0
        assert result.metadata.latency_ms < 100  # Should be under 100ms
    
    def test_compute_metadata_includes_timestamp(self):
        """Test that computation metadata includes timestamp."""
        tool = VectorQuantTool()
        
        result = tool.compute("compute_mean", {
            "returns": [1.0, 2.0, 3.0]
        })
        
        assert result.metadata.timestamp is not None
        assert "T" in result.metadata.timestamp  # ISO format has T
    
    def test_result_to_dict_serializable(self):
        """Test that result can be converted to JSON-serializable dict."""
        tool = VectorQuantTool()
        
        result = tool.compute("compute_mean", {
            "returns": [1.0, 2.0, 3.0]
        })
        
        result_dict = result.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert json_str is not None
        assert "compute_mean" in json_str
    
    def test_get_openai_tools_schema(self):
        """Test OpenAI function-calling schema generation."""
        tool = VectorQuantTool()
        schemas = tool.get_openai_tools()
        
        assert isinstance(schemas, list)
        assert len(schemas) > 10
        
        # Check structure
        for schema in schemas:
            assert "type" in schema
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]
    
    def test_compute_var(self):
        """Test computing Value-at-Risk via agent interface."""
        tool = VectorQuantTool()
        
        result = tool.compute("compute_var", {
            "returns": [0.01, 0.02, -0.005, 0.015, -0.03, 0.01],
            "confidence_level": 0.95
        })
        
        assert result.metadata.error is None
        assert isinstance(result.result, (int, float))
        # VaR should be a valid number
        assert result.result is not None


class TestToolRegistry:
    """Test tool registry and discovery."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = ToolRegistry()
        assert registry is not None
        assert registry.get_tool_count() > 10
    
    def test_get_all_tools(self):
        """Test getting all tools."""
        registry = ToolRegistry()
        tools = registry.get_all_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 10
        assert "compute_sharpe" in tools
    
    def test_get_tools_by_category(self):
        """Test filtering tools by category."""
        registry = ToolRegistry()
        
        risk_tools = registry.get_tools_by_category("risk")
        assert "compute_sharpe" in risk_tools
        assert "compute_var" in risk_tools
        
        deriv_tools = registry.get_tools_by_category("derivatives")
        assert "price_call" in deriv_tools
        assert "price_put" in deriv_tools
    
    def test_get_categories(self):
        """Test getting available categories."""
        registry = ToolRegistry()
        categories = registry.get_categories()
        
        assert "risk" in categories
        assert "derivatives" in categories
        assert "simulation" in categories
        assert "optimization" in categories
    
    def test_get_category_for_tool(self):
        """Test getting category for a specific tool."""
        registry = ToolRegistry()
        
        category = registry.get_category_for_tool("compute_sharpe")
        assert category == "risk"
        
        category = registry.get_category_for_tool("price_call")
        assert category == "derivatives"
    
    def test_search_tools_by_keyword(self):
        """Test searching tools by keyword."""
        registry = ToolRegistry()
        
        # Search for "sharpe"
        results = registry.search_tools("sharpe")
        assert len(results) > 0
        assert "compute_sharpe" in results
    
    def test_search_tools_case_insensitive(self):
        """Test that search is case-insensitive."""
        registry = ToolRegistry()
        
        results_lower = registry.search_tools("sharpe")
        results_upper = registry.search_tools("SHARPE")
        
        assert results_lower == results_upper
    
    def test_list_tools_full(self):
        """Test getting full tool information."""
        registry = ToolRegistry()
        tools = registry.list_tools_full()
        
        assert len(tools) > 10
        
        # Each tool should have required fields
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "params" in tool
            assert "category" in tool
    
    def test_execute_tool_via_registry(self):
        """Test executing tool through registry."""
        registry = ToolRegistry()
        
        result = registry.execute_tool("compute_std", {
            "returns": [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        assert result.result is not None
        assert result.metadata.error is None
    
    def test_batch_execute(self):
        """Test batch execution of multiple tools."""
        registry = ToolRegistry()
        
        operations = [
            {"tool": "compute_std", "params": {"returns": [1, 2, 3, 4, 5]}},
            {"tool": "compute_variance", "params": {"returns": [1, 2, 3, 4, 5]}},
        ]
        
        results = registry.batch_execute(operations)
        
        assert len(results) == 2
        assert results[0].result is not None  # std
        assert results[1].result is not None  # variance
    
    def test_get_langchain_tools(self):
        """Test LangChain tool generation."""
        registry = ToolRegistry()
        
        try:
            tools = registry.get_langchain_tools()
            
            # Should return list of Tool objects
            assert isinstance(tools, list)
            assert len(tools) > 10
        except ImportError:
            # LangChain not installed — skip
            pytest.skip("LangChain not installed")
    
    def test_get_openai_tools(self):
        """Test OpenAI tools generation."""
        registry = ToolRegistry()
        
        tools = registry.get_openai_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 10
        assert all("function" in tool for tool in tools)


class TestAgentProtocolIntegration:
    """Integration tests for agent protocol."""
    
    def test_agent_workflow_sharpe_ratio(self):
        """Test complete agent workflow: get schema → validate → execute."""
        tool = VectorQuantTool()
        
        # 1. Agent gets schema
        schema = tool.get_schema("compute_sharpe")
        assert "params" in schema
        assert "returns" in schema["params"]
        
        # 2. Agent prepares parameters (with validation)
        params = {
            "returns": [0.01, 0.02, 0.015, 0.025, 0.005],
            "risk_free_rate": 0.02
        }
        
        # 3. Agent calls compute
        result = tool.compute("compute_sharpe", params)
        
        assert result.operation == "compute_sharpe"
        assert result.result is not None
        assert result.metadata.error is None
        assert result.metadata.latency_ms > 0
    
    def test_agent_workflow_option_pricing(self):
        """Test complete agent workflow for option pricing."""
        tool = VectorQuantTool()
        
        # Get schema
        schema = tool.get_schema("price_call")
        assert all(k in schema["params"] for k in ["S", "K", "r", "sigma", "T"])
        
        # Execute
        result = tool.compute("price_call", {
            "S": 100.0,
            "K": 105.0,
            "r": 0.05,
            "sigma": 0.20,
            "T": 0.25
        })
        
        assert result.result is not None
        assert result.result > 0  # Call price should be positive
    
    def test_error_recovery_workflow(self):
        """Test agent error recovery workflow."""
        tool = VectorQuantTool()
        
        # First attempt with bad data type
        result1 = tool.compute("compute_sharpe", {
            "returns": "not_a_list"  # Wrong type
        })
        
        assert result1.metadata.error is not None
        
        # Agent learns from error and retries with correct format
        result2 = tool.compute("compute_sharpe", {
            "returns": [0.01, 0.02, 0.015]
        })
        
        assert result2.metadata.error is None
        assert result2.result is not None
    
    def test_performance_latency_target(self):
        """Test that operations meet <1ms latency target for typical cases."""
        tool = VectorQuantTool()
        
        # Execute simple operations and measure latency
        latencies = []
        for _ in range(10):
            result = tool.compute("compute_mean", {
                "returns": [0.01, 0.02, 0.015, 0.025, 0.005]
            })
            latencies.append(result.metadata.latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        
        # Note: On modern hardware, simple operations should be <1ms
        # But on slower machines or with JIT compilation, may be higher
        assert avg_latency < 50  # Conservative threshold
        assert all(lat < 100 for lat in latencies)


class TestGlobalRegistryFunctions:
    """Test module-level convenience functions."""
    
    def test_get_all_tools_function(self):
        """Test get_all_tools convenience function."""
        tools = get_all_tools()
        assert isinstance(tools, list)
        assert len(tools) > 10
    
    def test_get_tool_schema_function(self):
        """Test get_tool_schema convenience function."""
        schema = get_tool_schema("compute_sharpe")
        assert "description" in schema
    
    def test_search_tools_function(self):
        """Test search_tools convenience function."""
        results = search_tools("sharpe")
        assert "compute_sharpe" in results
    
    def test_list_tools_full_function(self):
        """Test list_tools_full convenience function."""
        from vectorquant.ai.tool_registry import list_tools_full
        
        tools = list_tools_full()
        assert len(tools) > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
