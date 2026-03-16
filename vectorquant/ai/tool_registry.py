"""
Tool Registry and Discovery (Phase 9.1)

Central registry for all VectorQuant operations available to AI agents.
Provides tool discovery, filtering, and caching.
"""

from typing import List, Dict, Any, Optional
from .agent_interface import VectorQuantTool


class ToolRegistry:
    """
    Central registry for VectorQuant tools.
    
    Provides:
    - Tool discovery and filtering
    - Schema generation for different frameworks
    - Tool categorization
    - Caching for performance
    """
    
    # Tool categories for filtering
    CATEGORIES = {
        "statistics": [
            "compute_mean",
            "compute_std",
            "compute_variance",
            "compute_covariance",
            "compute_correlation",
        ],
        "risk": [
            "compute_sharpe",
            "compute_var",
            "compute_cvar",
        ],
        "derivatives": [
            "price_call",
            "price_put",
        ],
        "simulation": [
            "simulate_gbm",
        ],
        "optimization": [
            "optimize_portfolio",
        ],
        "linear_algebra": [
            "matrix_multiply",
            "compute_determinant",
            "compute_inverse",
        ],
    }
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tool = VectorQuantTool()
        self._cache = {}
    
    def get_all_tools(self) -> List[str]:
        """Get names of all available tools."""
        return self._tool.get_available_operations()
    
    def get_tool_count(self) -> int:
        """Get total number of available tools."""
        return len(self.get_all_tools())
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """
        Get tools in a specific category.
        
        Args:
            category: Category name (e.g., "risk", "derivatives", "simulation")
            
        Returns:
            List of tool names in that category
            
        Raises:
            ValueError: If category not found
        """
        if category not in self.CATEGORIES:
            available = ", ".join(self.CATEGORIES.keys())
            raise ValueError(f"Unknown category '{category}'. "
                           f"Available: {available}")
        
        return self.CATEGORIES[category]
    
    def get_categories(self) -> List[str]:
        """Get list of available tool categories."""
        return list(self.CATEGORIES.keys())
    
    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """
        Get schema for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Schema dict with description, parameters, and return type
        """
        cache_key = f"schema:{tool_name}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self._tool.get_schema(tool_name)
        
        return self._cache[cache_key]
    
    def get_tools_by_param_type(self, param_type: str) -> List[str]:
        """
        Find tools that accept a specific parameter type.
        
        Args:
            param_type: Parameter type to search for ("numeric", "list", "matrix", etc.)
            
        Returns:
            List of tool names that accept this parameter type
        """
        matching = []
        for tool_name in self.get_all_tools():
            schema = self.get_tool_schema(tool_name)
            params = schema.get("params", {})
            
            for param_info in params.values():
                if param_info.get("type") == param_type:
                    matching.append(tool_name)
                    break
        
        return matching
    
    def get_category_for_tool(self, tool_name: str) -> Optional[str]:
        """
        Get category for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Category name if found, None otherwise
        """
        for category, tools in self.CATEGORIES.items():
            if tool_name in tools:
                return category
        
        return None
    
    def list_tools_full(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about all tools.
        
        Returns:
            List of dicts with name, category, description, and param info
        """
        result = []
        for tool_name in self.get_all_tools():
            schema = self.get_tool_schema(tool_name)
            result.append({
                "name": tool_name,
                "category": self.get_category_for_tool(tool_name),
                "description": schema.get("description", ""),
                "params": schema.get("params", {}),
                "returns": schema.get("returns", {}),
            })
        
        return result
    
    def search_tools(self, keyword: str) -> List[str]:
        """
        Search for tools by keyword in name or description.
        
        Args:
            keyword: Search term (case-insensitive)
            
        Returns:
            List of matching tool names
        """
        keyword_lower = keyword.lower()
        matching = []
        
        for tool_name in self.get_all_tools():
            # Check name
            if keyword_lower in tool_name.lower():
                matching.append(tool_name)
                continue
            
            # Check description
            schema = self.get_tool_schema(tool_name)
            description = schema.get("description", "").lower()
            if keyword_lower in description:
                matching.append(tool_name)
        
        return matching
    
    def get_langchain_tools(self):
        """Get all tools as LangChain Tool objects."""
        return self._tool.get_langchain_tools()
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Get all tools as OpenAI function-calling schemas."""
        return self._tool.get_openai_tools()
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any],
                    verify: bool = False) -> Any:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool
            params: Parameters dict
            verify: If True, verify the computation
            
        Returns:
            ComputationResult with result and metadata
        """
        return self._tool.compute(tool_name, params, verify=verify)
    
    def batch_execute(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """
        Execute multiple tools in sequence.
        
        Args:
            operations: List of {"tool": name, "params": params} dicts
            
        Returns:
            List of ComputationResult objects
        """
        results = []
        for op in operations:
            tool_name = op.get("tool")
            params = op.get("params", {})
            verify = op.get("verify", False)
            
            result = self.execute_tool(tool_name, params, verify=verify)
            results.append(result)
        
        return results


# Global registry instance
_global_registry = None


def get_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    
    return _global_registry


def get_all_tools() -> List[str]:
    """Get list of all available tools."""
    return get_registry().get_all_tools()


def get_tool_schema(tool_name: str) -> Dict[str, Any]:
    """Get schema for a tool."""
    return get_registry().get_tool_schema(tool_name)


def execute_tool(tool_name: str, params: Dict[str, Any],
                verify: bool = False) -> Any:
    """Execute a tool."""
    return get_registry().execute_tool(tool_name, params, verify=verify)


def list_tools_full() -> List[Dict[str, Any]]:
    """Get detailed info about all tools."""
    return get_registry().list_tools_full()


def search_tools(keyword: str) -> List[str]:
    """Search for tools by keyword."""
    return get_registry().search_tools(keyword)
