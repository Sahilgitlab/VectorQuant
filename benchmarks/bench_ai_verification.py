
import vectorquant.ai as vq_ai
from benchmarks.bench_utils import BenchmarkRunner

runner = BenchmarkRunner("AI Verification")

def run_ai_benchmarks():
    # 1. Hallucination Check Accuracy & Latency
    expression = "sqrt(256) + log(100, 10)"
    expected = 18.0
    
    runner.run(
        "Hallucination_Check",
        vq_ai.verify_calculation,
        args=(expression, expected),
        iterations=1000
    )

    # 2. Risk Trace Generation
    returns = [0.01, -0.02, 0.015, -0.005] * 25
    runner.run(
        "VaR_Trace_Gen",
        vq_ai.explain_var,
        args=(returns, 0.95),
        iterations=100
    )

if __name__ == "__main__":
    run_ai_benchmarks()
    runner.save()
    runner.print_table()
