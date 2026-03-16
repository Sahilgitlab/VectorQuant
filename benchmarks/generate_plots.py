
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_results(category):
    path = f"benchmarks/results/{category.lower()}_results.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def plot_benchmark(category, title, xlabel, ylabel, filename):
    results = load_results(category)
    if not results:
        return

    names = list(results.keys())
    means = [r["mean"] for r in results.values()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, means, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"benchmarks/plots/{filename}.png")
    plt.close()

def plot_scaling(category, test_base_name, sizes, title, filename):
    results = load_results(category)
    if not results:
        return

    vq_times = []
    base_times = []
    
    for s in sizes:
        vq_key = f"{test_base_name}_{s}"
        base_key = f"NumPy_{test_base_name}_{s}"
        if vq_key in results:
            vq_times.append(results[vq_key]["mean"])
        if base_key in results:
            base_times.append(results[base_key]["mean"])

    if not vq_times: return

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, vq_times, marker='o', label='VectorQuant')
    if base_times:
        plt.plot(sizes, base_times, marker='s', label='NumPy')
    
    plt.title(title)
    plt.xlabel("Scale (Paths / Matrix Size)")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"benchmarks/plots/{filename}.png")
    plt.close()

if __name__ == "__main__":
    os.makedirs("benchmarks/plots", exist_ok=True)
    
    # 1. Linear Algebra Scaling
    # (Simplified for now, picking MatMul as example)
    results_la = load_results("linear algebra")
    if results_la:
        sizes = [50, 200, 500]
        vq_mm = [results_la[f"MatMul_{s}"]["mean"] for s in sizes]
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, vq_mm, marker='x', label='VectorQuant MatMul')
        plt.title("VectorQuant Linear Algebra Scaling")
        plt.xlabel("Matrix Dimension")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.savefig("benchmarks/plots/linalg_scaling.png")
        plt.close()

    # 2. Monte Carlo Scaling
    plot_scaling("Monte Carlo", "GBM", [10000, 100000, 1000000], "Monte Carlo Path Generation Scaling", "mc_scaling")

    # 3. AI Verification Latency
    plot_benchmark("AI Verification", "AI Verification Latency", "Test", "Mean Time (ms)", "ai_latency")

    print("Plots generated in benchmarks/plots/")
