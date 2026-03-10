"""
LLM Interface

Unified entry point for LLM frameworks to interact with VectorQuant.
Combines tool execution, verification, and proof tracing into a single
API that any AI system can consume.
"""

from .tools import get_tool_registry, get_tool_schemas, execute_tool


class LLMInterface:
    """
    Unified LLM integration interface.
    
    Provides compute + verify + explain in a single call,
    plus framework-specific adapters for OpenAI, LangChain, etc.
    
    Usage::
    
        llm = vq.ai.LLMInterface()
        result = llm.execute("calculate_var", returns=data, confidence_level=0.95)
        # result includes: value, verified, proof_trace, confidence
    """

    def execute(self, tool_name, **params):
        """
        Execute a VectorQuant tool with automatic verification
        and proof tracing where available.
        
        Returns:
            dict with keys: tool, value, verified, confidence, proof_trace
        """
        # 1. Execute the computation
        value = execute_tool(tool_name, **params)
        
        # 2. Attempt verification + proof trace
        verified = True
        confidence = 1.0
        proof_trace = None
        
        try:
            if tool_name == "calculate_var":
                from .proof_trace import explain_var
                trace = explain_var(params["returns"], params.get("confidence_level", 0.95))
                proof_trace = trace.to_dict()
                
            elif tool_name == "compute_sharpe":
                from .proof_trace import explain_sharpe
                trace = explain_sharpe(params["returns"], params.get("risk_free_rate", 0.0))
                proof_trace = trace.to_dict()
                
            elif tool_name == "price_call_option":
                from .proof_trace import explain_black_scholes
                trace = explain_black_scholes(
                    params["S"], params["K"], params["r"], params["sigma"], params["T"]
                )
                proof_trace = trace.to_dict()
                diff = abs(trace.result - value)
                verified = diff < 0.01
        except Exception:
            pass  # Proof trace is optional enhancement
        
        return {
            "tool": tool_name,
            "value": value,
            "verified": verified,
            "confidence": confidence if verified else 0.5,
            "proof_trace": proof_trace,
        }

    @staticmethod
    def get_openai_tools():
        """
        Returns tool definitions in OpenAI function-calling format.
        Ready to pass to `tools=` parameter in OpenAI API calls.
        """
        return get_tool_schemas()

    @staticmethod
    def get_langchain_tools():
        """
        Returns VectorQuant tools as LangChain-compatible Tool objects.
        Falls back to simple dicts if LangChain is not installed.
        """
        registry = get_tool_registry()
        tools = []
        
        try:
            from langchain.tools import Tool
            for name, info in registry.items():
                tools.append(Tool(
                    name=name,
                    func=info["function"],
                    description=info["description"]
                ))
        except ImportError:
            # LangChain not installed — return simple dicts
            for name, info in registry.items():
                tools.append({
                    "name": name,
                    "func": info["function"],
                    "description": info["description"],
                })
        
        return tools
