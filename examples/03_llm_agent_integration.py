"""
Phase 9.1 — LLM Agent Integration Examples

Demonstrates how AI agents (Claude, Gemini, LangChain, etc.) interact
with VectorQuant through the standardized agent protocol.

Three integration patterns:
1. OpenAI Function Calling (Claude, GPT)
2. LangChain Tool Integration
3. Custom Agent Implementation
"""

import json
from vectorquant.ai.agent_interface import VectorQuantTool
from vectorquant.ai.tool_registry import get_registry


# ============================================================================
# PATTERN 1: OpenAI Function Calling (Claude, GPT-4, etc.)
# ============================================================================

def example_openai_function_calling():
    """
    Example: Claude using VectorQuant tools via OpenAI function-calling protocol
    
    This demonstrates how Claude (via Anthropic API) can invoke VectorQuant tools.
    
    Usage with Anthropic SDK:
    
        import anthropic
        
        client = anthropic.Anthropic()
        tools = VectorQuantTool().get_openai_tools()
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
            messages=[{
                "role": "user",
                "content": "What is the Sharpe ratio of returns [0.01, 0.02, 0.015]?"
            }]
        )
    """
    
    print("=" * 70)
    print("PATTERN 1: OpenAI Function Calling (Claude, GPT-4)")
    print("=" * 70)
    
    tool = VectorQuantTool()
    
    # Get OpenAI-compatible tool schemas
    schemas = tool.get_openai_tools()
    
    print(f"\n[OK] Generated {len(schemas)} tools for OpenAI function calling")
    print(f"\nExample schema (compute_sharpe):")
    
    sharpe_schema = next(s for s in schemas if s["function"]["name"] == "compute_sharpe")
    print(json.dumps(sharpe_schema, indent=2))
    
    # Simulate Claude invoking a tool
    print("\n\n--- Simulated Claude -> VectorQuant Call ---")
    print("Claude invoked: compute_sharpe(returns=[0.01, 0.02, 0.015], risk_free_rate=0.02)")
    
    result = tool.compute("compute_sharpe", {
        "returns": [0.01, 0.02, 0.015, 0.025, 0.005],
        "risk_free_rate": 0.02
    })
    
    print(f"\nVectorQuant returned:")
    print(f"  Sharpe Ratio: {result.result:.4f}")
    print(f"  Latency: {result.metadata.latency_ms:.2f}ms")
    print(f"  Backend: {result.metadata.backend}")


# ============================================================================
# PATTERN 2: LangChain Tool Integration
# ============================================================================

def example_langchain_integration():
    """
    Example: LangChain agent using VectorQuant tools
    
    This demonstrates integration with LangChain agents.
    
    Usage:
    
        from langchain.tools import ToolException
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        
        from vectorquant.ai.agent_interface import VectorQuantTool
        
        # Get LangChain tools
        tools = VectorQuantTool().get_langchain_tools()
        
        # Create agent
        llm = ChatOpenAI()
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
        
        # Run agent
        result = executor.invoke({
            "input": "Calculate the Sharpe ratio of Apple returns"
        })
    """
    
    print("\n" + "=" * 70)
    print("PATTERN 2: LangChain Tool Integration")
    print("=" * 70)
    
    try:
        tool = VectorQuantTool()
        tools = tool.get_langchain_tools()
        
        print(f"\n[OK] Generated {len(tools)} LangChain Tool objects")
        
        # Show tool interface
        if tools:
            first_tool = tools[0]
            print(f"\nExample tool:")
            print(f"  Name: {first_tool.name}")
            print(f"  Description: {first_tool.description}")
            print(f"  Callable: {callable(first_tool.func)}")
        
        print("\n[OK] Ready to integrate with LangChain agents")
        print("  Example: ReAct agent, OpenAI agent, etc.")
        
    except ImportError:
        print("\n[WARN] LangChain not installed.")
        print("  Install with: pip install langchain langchain-openai")


# ============================================================================
# PATTERN 3: Custom Agent Implementation
# ============================================================================

class CustomFinancialAgent:
    """
    Custom financial agent that uses VectorQuant for computations.
    
    Demonstrates how to build a simple agent that:
    1. Accepts natural language requests
    2. Interprets the request
    3. Calls VectorQuantTool.compute()
    4. Returns results to user
    """
    
    def __init__(self):
        self.tool = VectorQuantTool()
        self.request_log = []
    
    def process_request(self, user_request: str, data: dict = None) -> dict:
        """
        Process a natural language request and execute appropriate tool.
        
        Args:
            user_request: Natural language request (e.g., "Calculate Sharpe")
            data: Data dict with "returns" or other parameters
            
        Returns:
            Result dict with tool output and metadata
        """
        
        # Simple keyword matching (in real implementation, use LLM)
        request_lower = user_request.lower()
        
        # Map keywords to operations
        operation = None
        if "sharpe" in request_lower:
            operation = "compute_sharpe"
        elif "var" in request_lower and "variance" not in request_lower:
            operation = "compute_var"
        elif "variance" in request_lower or "volatility" in request_lower:
            operation = "compute_std"
        elif "mean" in request_lower or "average" in request_lower:
            operation = "compute_mean"
        elif "cvar" in request_lower or "conds" in request_lower:
            operation = "compute_cvar"
        else:
            return {"error": f"Cannot interpret request: {user_request}"}
        
        # Execute tool
        result = self.tool.compute(operation, data or {})
        
        # Log request
        self.request_log.append({
            "request": user_request,
            "operation": operation,
            "result": result.to_dict(),
        })
        
        return result.to_dict()
    
    def get_session_log(self) -> list:
        """Get log of all requests in this session."""
        return self.request_log


def example_custom_agent():
    """
    Example: Custom financial agent
    """
    
    print("\n" + "=" * 70)
    print("PATTERN 3: Custom Agent Implementation")
    print("=" * 70)
    
    # Create custom agent
    agent = CustomFinancialAgent()
    
    # Example 1: Natural language Sharpe ratio request
    print("\n--- Request 1: Natural Language ---")
    request = "Calculate the Sharpe ratio for my portfolio"
    data = {
        "returns": [0.02, 0.015, 0.03, -0.005, 0.025],
        "risk_free_rate": 0.02
    }
    
    print(f"User: {request}")
    print(f"Data: {data}")
    
    result = agent.process_request(request, data)
    
    print(f"\nAgent Response:")
    print(f"  Operation: {result['operation']}")
    print(f"  Sharpe Ratio: {result['result']:.4f}")
    print(f"  Latency: {result['metadata']['latency_ms']:.2f}ms")
    
    # Example 2: VaR request
    print("\n--- Request 2: Value-at-Risk ---")
    request = "What's the VaR of this portfolio?"
    data = {
        "returns": [0.01, 0.02, -0.01, 0.015, -0.03, 0.02, 0.01],
        "confidence_level": 0.95
    }
    
    print(f"User: {request}")
    result = agent.process_request(request, data)
    
    print(f"\nAgent Response:")
    print(f"  Operation: {result['operation']}")
    print(f"  VaR (95%): {result['result']:.6f}")
    
    # Show session log
    print(f"\n--- Session Log ({len(agent.request_log)} requests) ---")
    for i, log in enumerate(agent.request_log, 1):
        print(f"{i}. {log['request']} -> {log['operation']}")


# ============================================================================
# PATTERN 4: Multi-Tool Workflow
# ============================================================================

def example_multi_tool_workflow():
    """
    Example: Complex workflow using multiple tools
    
    Demonstrates:
    - Chaining multiple VectorQuant operations
    - Passing results between operations
    - Building computation pipelines
    """
    
    print("\n" + "=" * 70)
    print("PATTERN 4: Multi-Tool Workflow")
    print("=" * 70)
    
    registry = get_registry()
    
    # Workflow: Analyze portfolio performance
    print("\n--- Workflow: Portfolio Performance Analysis ---")
    
    returns = [0.02, 0.015, 0.03, -0.005, 0.025, 0.01, 0.02]
    
    # Step 1: Compute basic statistics
    print("\nStep 1: Computing portfolio statistics...")
    
    mean_result = registry.execute_tool("compute_mean", {"returns": returns})
    print(f"  Mean return: {mean_result.result:.4f}")
    
    std_result = registry.execute_tool("compute_std", {"returns": returns})
    print(f"  Std deviation: {std_result.result:.4f}")
    
    # Step 2: Compute risk metrics
    print("\nStep 2: Computing risk metrics...")
    
    sharpe_result = registry.execute_tool("compute_sharpe", {
        "returns": returns,
        "risk_free_rate": 0.02
    })
    print(f"  Sharpe ratio: {sharpe_result.result:.4f}")
    
    var_result = registry.execute_tool("compute_var", {
        "returns": returns,
        "confidence_level": 0.95
    })
    print(f"  VaR (95%): {var_result.result:.6f}")
    
    # Step 3: Summary
    print("\n--- Portfolio Summary ---")
    print(f"Expected Return:  {mean_result.result:.4f}")
    print(f"Volatility:       {std_result.result:.4f}")
    print(f"Sharpe Ratio:     {sharpe_result.result:.4f}")
    print(f"Value-at-Risk:    {var_result.result:.6f}")
    
    total_latency = sum([
        mean_result.metadata.latency_ms,
        std_result.metadata.latency_ms,
        sharpe_result.metadata.latency_ms,
        var_result.metadata.latency_ms,
    ])
    
    print(f"\nTotal computation time: {total_latency:.2f}ms")
    print("All computations executed via C backend for optimal performance.")


# ============================================================================
# PATTERN 5: Error Handling & Recovery
# ============================================================================

def example_error_handling():
    """
    Example: Robust error handling in agent workflows
    
    Demonstrates:
    - Graceful error detection
    - Automatic parameter validation
    - Recovery strategies
    """
    
    print("\n" + "=" * 70)
    print("PATTERN 5: Error Handling & Recovery")
    print("=" * 70)
    
    tool = VectorQuantTool()
    
    # Attempt 1: Invalid parameter type
    print("\n--- Attempt 1: Invalid Parameter Type ---")
    print("User input: compute_sharpe(returns='0.01 0.02')")  # String instead of list
    
    result = tool.compute("compute_sharpe", {
        "returns": "0.01 0.02"  # Wrong!
    })
    
    if result.metadata.error:
        print(f"[ERROR] {result.metadata.error}")
        print("  Recommendation: Convert to numeric list")
    
    # Attempt 2: Recovery
    print("\n--- Attempt 2: Recovery ---")
    print("Agent corrects input: compute_sharpe(returns=[0.01, 0.02, 0.015])")
    
    result = tool.compute("compute_sharpe", {
        "returns": [0.01, 0.02, 0.015]
    })
    
    if result.metadata.error is None:
        print(f"[OK] Success! Sharpe ratio: {result.result:.4f}")
    
    # Attempt 3: Missing required parameter
    print("\n--- Attempt 3: Missing Required Parameter ---")
    
    result = tool.compute("compute_sharpe", {
        "risk_free_rate": 0.02  # Missing "returns"
    })
    
    if result.metadata.error:
        print(f"[ERROR] {result.metadata.error}")
        print("  Recommendation: Provide 'returns' parameter")
    
    # Attempt 4: Validation with metadata
    print("\n--- Attempt 4: Full Validation ---")
    
    schema = tool.get_schema("compute_sharpe")
    required_params = [k for k, v in schema["params"].items() 
                      if "default" not in v]
    
    print(f"Schema validation:")
    print(f"  Required parameters: {required_params}")
    print(f"  Optional parameters: {[k for k in schema['params'] if 'default' in schema['params'][k]]}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 9.1 — LLM AGENT INTEGRATION EXAMPLES")
    print("=" * 70)
    
    # Run all examples
    example_openai_function_calling()
    example_langchain_integration()
    example_custom_agent()
    example_multi_tool_workflow()
    example_error_handling()
    
    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nIntegration patterns demonstrated:")
    print("  1. [OK] OpenAI Function Calling (Claude, GPT-4)")
    print("  2. [OK] LangChain Tool Integration")
    print("  3. [OK] Custom Agent Implementation")
    print("  4. [OK] Multi-Tool Workflows")
    print("  5. [OK] Error Handling & Recovery")
    print("\nNext step: Phase 9.2 (Verification Pipeline)")
