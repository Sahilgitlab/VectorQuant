"""
Phase 9.4: Trace & Proof Generation

Enables explainable AI by tracking computation flow and generating proofs.

Generates:
1. Computation traces (all intermediate values)
2. Proof trees (what contributed to each result)
3. Human-readable explanations
4. JSON/LaTeX exports for documentation

Example:
    tracer = ComputationTracer()
    result = tracer.trace_sharpe(
        returns=[0.01, 0.02, 0.015, 0.03],
        rf=0.03
    )
    # Result includes:
    # - mean = 0.0175
    # - std = 0.0078
    # - sharpe = (0.0175 - 0.03) / 0.0078 = -1.603
    # - proof tree showing each step
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import json


class ComputationNodeType(Enum):
    """Types of computation nodes in trace tree."""
    INPUT = "input"
    INTERMEDIATE = "intermediate"
    OUTPUT = "output"
    OPERATION = "operation"


@dataclass
class ComputationNode:
    """Single node in computation trace tree."""
    name: str
    node_type: ComputationNodeType
    value: Any = None
    operation: Optional[str] = None
    inputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.node_type.value,
            "value": self.value,
            "operation": self.operation,
            "inputs": self.inputs,
            "metadata": self.metadata
        }


@dataclass
class ComputationTrace:
    """Complete trace of a computation."""
    operation_name: str
    nodes: Dict[str, ComputationNode] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    timestamps: Dict[str, float] = field(default_factory=dict)
    total_time_ms: float = 0.0
    
    def add_node(self, node: ComputationNode):
        """Add a node to the trace."""
        self.nodes[node.name] = node
        self.execution_order.append(node.name)
    
    def add_input(self, name: str, value: Any, **metadata):
        """Add an input node."""
        node = ComputationNode(
            name=name,
            node_type=ComputationNodeType.INPUT,
            value=value,
            metadata=metadata
        )
        self.add_node(node)
    
    def add_intermediate(self, name: str, value: Any, operation: str, 
                        inputs: List[str], **metadata):
        """Add an intermediate computation node."""
        node = ComputationNode(
            name=name,
            node_type=ComputationNodeType.INTERMEDIATE,
            value=value,
            operation=operation,
            inputs=inputs,
            metadata=metadata
        )
        self.add_node(node)
    
    def add_output(self, name: str, value: Any, inputs: List[str], **metadata):
        """Add an output node."""
        node = ComputationNode(
            name=name,
            node_type=ComputationNodeType.OUTPUT,
            value=value,
            inputs=inputs,
            metadata=metadata
        )
        self.add_node(node)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "operation": self.operation_name,
            "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
            "execution_order": self.execution_order,
            "total_time_ms": self.total_time_ms
        }
    
    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class ProofStep:
    """Single step in a proof."""
    step_number: int
    operation: str
    left_operand: Any
    right_operand: Optional[Any] = None
    result: Optional[Any] = None
    formula: str = ""
    explanation: str = ""
    
    def to_text(self) -> str:
        """Convert to human-readable text."""
        parts = [f"Step {self.step_number}: {self.operation}"]
        
        if self.left_operand is not None:
            parts.append(f"  Left: {self.left_operand}")
        
        if self.right_operand is not None:
            parts.append(f"  Right: {self.right_operand}")
        
        if self.formula:
            parts.append(f"  Formula: {self.formula}")
        
        if self.result is not None:
            parts.append(f"  Result: {self.result}")
        
        if self.explanation:
            parts.append(f"  Note: {self.explanation}")
        
        return "\n".join(parts)


@dataclass
class ProofTree:
    """Tree of proof steps for a computation."""
    operation_name: str
    steps: List[ProofStep] = field(default_factory=list)
    final_result: Optional[Any] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def add_step(self, step: ProofStep):
        """Add a proof step."""
        self.steps.append(step)
    
    def to_text(self) -> str:
        """Convert to human-readable proof."""
        lines = [
            f"Computation Proof: {self.operation_name}",
            "=" * 50
        ]
        
        for step in self.steps:
            lines.append(step.to_text())
            lines.append("")
        
        if self.final_result is not None:
            lines.extend([
                "=" * 50,
                f"Final Result: {self.final_result}",
                f"Timestamp: {self.timestamp}"
            ])
        
        return "\n".join(lines)
    
    def to_latex(self) -> str:
        """Export as LaTeX."""
        lines = [
            r"\documentclass{article}",
            r"\usepackage{amsmath}",
            r"\begin{document}",
            f"\n\\section{{{self.operation_name} Computation}}\n"
        ]
        
        for i, step in enumerate(self.steps, 1):
            lines.append(f"\\subsection{{Step {i}: {step.operation}}}\n")
            
            if step.formula:
                lines.append(f"\\[{step.formula}\\]\n")
            
            if step.explanation:
                lines.append(f"{step.explanation}\n")
        
        lines.extend([
            r"\end{document}"
        ])
        
        return "\n".join(lines)


class ComputationTracer:
    """Traces computation execution for explainability."""
    
    def trace_mean(self, data: List[float]) -> Tuple[ComputationTrace, ProofTree]:
        """Trace mean computation."""
        trace = ComputationTrace("mean")
        proof = ProofTree("mean")
        
        # Input
        trace.add_input("data", data, length=len(data))
        
        # Intermediate: sum
        data_sum = sum(data)
        trace.add_intermediate(
            "sum",
            data_sum,
            "add",
            ["data"],
            description="Sum all elements"
        )
        
        proof.add_step(ProofStep(
            step_number=1,
            operation="Sum",
            left_operand="[" + ", ".join(f"{x:.4f}" for x in data) + "]",
            result=f"{data_sum:.6f}",
            formula=f"\\sum_{{i=1}}^{{n}} x_i = {data_sum:.6f}",
            explanation=f"Sum {len(data)} elements"
        ))
        
        # Output: mean
        mean_val = data_sum / len(data)
        trace.add_output(
            "mean",
            mean_val,
            ["sum"],
            formula="mean = sum / n"
        )
        
        proof.add_step(ProofStep(
            step_number=2,
            operation="Divide",
            left_operand=f"{data_sum:.6f}",
            right_operand=len(data),
            result=f"{mean_val:.6f}",
            formula=f"\\mu = \\frac{{\\sum x_i}}{{n}} = \\frac{{{data_sum:.6f}}}{{{len(data)}}} = {mean_val:.6f}",
            explanation="Divide sum by count"
        ))
        
        proof.final_result = mean_val
        
        return trace, proof
    
    def trace_std(self, data: List[float]) -> Tuple[ComputationTrace, ProofTree]:
        """Trace standard deviation computation."""
        trace = ComputationTrace("std")
        proof = ProofTree("std")
        
        n = len(data)
        
        # Step 1: Mean
        trace.add_input("data", data, length=n)
        mean_val = sum(data) / n
        trace.add_intermediate("mean", mean_val, "mean", ["data"])
        
        proof.add_step(ProofStep(
            step_number=1,
            operation="Compute Mean",
            left_operand="[...]",
            result=f"{mean_val:.6f}",
            formula=f"\\mu = {mean_val:.6f}"
        ))
        
        # Step 2: Deviations
        deviations = [(x - mean_val) for x in data]
        trace.add_intermediate("deviations", deviations, "subtract", ["data", "mean"])
        
        proof.add_step(ProofStep(
            step_number=2,
            operation="Calculate Deviations",
            left_operand=f"[...]",
            right_operand=f"{mean_val:.6f}",
            result=f"[{', '.join(f'{d:.6f}' for d in deviations[:3])}...]",
            formula=f"x_i - \\mu"
        ))
        
        # Step 3: Squared deviations
        sq_deviations = [d**2 for d in deviations]
        sum_sq_dev = sum(sq_deviations)
        trace.add_intermediate("sum_sq_dev", sum_sq_dev, "sum", ["deviations"])
        
        proof.add_step(ProofStep(
            step_number=3,
            operation="Sum Squared Deviations",
            left_operand=f"[{', '.join(f'{d**2:.6f}' for d in deviations[:3])}...]",
            result=f"{sum_sq_dev:.6f}",
            formula=f"\\sum (x_i - \\mu)^2 = {sum_sq_dev:.6f}"
        ))
        
        # Step 4: Variance
        variance = sum_sq_dev / (n - 1) if n > 1 else 0
        trace.add_intermediate("variance", variance, "divide", ["sum_sq_dev"])
        
        proof.add_step(ProofStep(
            step_number=4,
            operation="Calculate Variance",
            left_operand=f"{sum_sq_dev:.6f}",
            right_operand=n - 1,
            result=f"{variance:.6f}",
            formula=f"s^2 = \\frac{{\\sum (x_i - \\mu)^2}}{{n-1}} = {variance:.6f}",
            explanation="Using Bessel's correction (n-1)"
        ))
        
        # Step 5: Std dev
        import math
        std_val = math.sqrt(variance)
        trace.add_output("std", std_val, ["variance"])
        
        proof.add_step(ProofStep(
            step_number=5,
            operation="Take Square Root",
            left_operand=f"{variance:.6f}",
            result=f"{std_val:.6f}",
            formula=f"s = \\sqrt{{s^2}} = \\sqrt{{{variance:.6f}}} = {std_val:.6f}"
        ))
        
        proof.final_result = std_val
        
        return trace, proof
    
    def trace_sharpe(self, returns: List[float], rf: float = 0.0) -> Tuple[ComputationTrace, ProofTree]:
        """Trace Sharpe ratio computation."""
        trace = ComputationTrace("sharpe_ratio")
        proof = ProofTree("sharpe_ratio")
        
        # Step 1: Mean return
        trace.add_input("returns", returns, length=len(returns))
        trace.add_input("risk_free_rate", rf)
        
        mean_return = sum(returns) / len(returns)
        trace.add_intermediate("mean_return", mean_return, "mean", ["returns"])
        
        proof.add_step(ProofStep(
            step_number=1,
            operation="Calculate Mean Return",
            left_operand="[" + ", ".join(f"{x:.4f}" for x in returns) + "]",
            result=f"{mean_return:.6f}",
            formula=f"\\mu_r = \\frac{{1}}{{n}}\\sum r_i = {mean_return:.6f}"
        ))
        
        # Step 2: Volatility (std dev)
        deviations = [(x - mean_return) for x in returns]
        sum_sq_dev = sum(d**2 for d in deviations)
        variance = sum_sq_dev / (len(returns) - 1)
        import math
        volatility = math.sqrt(variance)
        trace.add_intermediate("volatility", volatility, "std", ["returns"])
        
        proof.add_step(ProofStep(
            step_number=2,
            operation="Calculate Volatility (Std Dev)",
            left_operand="[...]",
            result=f"{volatility:.6f}",
            formula=f"\\sigma = \\sqrt{{\\frac{{\\sum (r_i - \\mu_r)^2}}{{n-1}}}} = {volatility:.6f}"
        ))
        
        # Step 3: Excess return
        excess_return = mean_return - rf
        trace.add_intermediate("excess_return", excess_return, "subtract", 
                              ["mean_return", "risk_free_rate"])
        
        proof.add_step(ProofStep(
            step_number=3,
            operation="Calculate Excess Return",
            left_operand=f"{mean_return:.6f}",
            right_operand=f"{rf:.6f}",
            result=f"{excess_return:.6f}",
            formula=f"r_e = \\mu_r - r_f = {mean_return:.6f} - {rf:.6f} = {excess_return:.6f}"
        ))
        
        # Step 4: Sharpe ratio
        if volatility > 0:
            sharpe = excess_return / volatility
        else:
            sharpe = 0.0
        trace.add_output("sharpe", sharpe, ["excess_return", "volatility"])
        
        proof.add_step(ProofStep(
            step_number=4,
            operation="Calculate Sharpe Ratio",
            left_operand=f"{excess_return:.6f}",
            right_operand=f"{volatility:.6f}",
            result=f"{sharpe:.6f}",
            formula=f"S = \\frac{{r_e}}{{\\sigma}} = \\frac{{{excess_return:.6f}}}{{{volatility:.6f}}} = {sharpe:.6f}"
        ))
        
        proof.final_result = sharpe
        
        return trace, proof


class ExplainabilityReporter:
    """Generates human-readable explanations from traces."""
    
    @staticmethod
    def generate_report(trace: ComputationTrace, proof: ProofTree) -> str:
        """Generate a complete explainability report."""
        parts = [
            "=" * 70,
            f"COMPUTATION EXPLAINABILITY REPORT",
            "=" * 70,
            f"\nOperation: {trace.operation_name}",
            f"Total Execution Time: {trace.total_time_ms:.3f}ms\n",
            "COMPUTATION STEPS:",
            "-" * 70
        ]
        
        # Add proof steps
        parts.append(proof.to_text())
        
        # Add trace information
        parts.extend([
            "\nCOMPUTATION TRACE:",
            "-" * 70
        ])
        
        for name in trace.execution_order:
            node = trace.nodes[name]
            parts.append(f"\n{name} ({node.node_type.value}):")
            parts.append(f"  Value: {node.value}")
            if node.operation:
                parts.append(f"  Operation: {node.operation}")
            if node.inputs:
                parts.append(f"  Inputs: {', '.join(node.inputs)}")
        
        parts.append("\n" + "=" * 70)
        
        return "\n".join(parts)


def trace_and_explain(operation: str, **kwargs) -> Tuple[ComputationTrace, ProofTree, str]:
    """
    Quick interface to trace an operation and get explanation.
    
    Args:
        operation: Operation name (mean, std, sharpe, etc.)
        **kwargs: Operation parameters
    
    Returns:
        (trace, proof, human_readable_report)
    """
    tracer = ComputationTracer()
    reporter = ExplainabilityReporter()
    
    if operation == "mean":
        trace, proof = tracer.trace_mean(kwargs.get("data", []))
    elif operation == "std":
        trace, proof = tracer.trace_std(kwargs.get("data", []))
    elif operation == "sharpe":
        trace, proof = tracer.trace_sharpe(
            kwargs.get("returns", []),
            kwargs.get("rf", 0.0)
        )
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    report = reporter.generate_report(trace, proof)
    
    return trace, proof, report
