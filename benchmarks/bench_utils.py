
import time
import numpy as np
import json
import os
from statistics import mean, median, stdev

class BenchmarkRunner:
    def __init__(self, category_name, seed=42):
        self.category = category_name
        self.results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.results_file = os.path.join(self.results_dir, f"{category_name.lower()}_results.json")
        self.registry = {}
        np.random.seed(seed)
        import random
        random.seed(seed)

    def run(self, name, func, args=(), kwargs={}, iterations=100, warmup=10, compare_to=None):
        print(f"Running benchmark: {name}...", end="", flush=True)
        
        # Correctness check
        error = None
        if compare_to:
            try:
                vq_res = func(*args, **kwargs)
                base_res = compare_to(*args, **kwargs)
                # Convert to numpy arrays for comparison, handling potential list-of-list issues
                vq_arr = np.array(vq_res, dtype=object) if isinstance(vq_res, (list, tuple)) else np.array(vq_res)
                base_arr = np.array(base_res, dtype=object) if isinstance(base_res, (list, tuple)) else np.array(base_res)
                
                np.testing.assert_allclose(np.asanyarray(vq_res), np.asanyarray(base_res), rtol=1e-7, atol=1e-10)
                error = "PASS"
            except Exception as e:
                err_msg = str(e).replace("\n", " ")
                error = f"FAIL ({err_msg[:100]}...)" if len(err_msg) > 100 else f"FAIL ({err_msg})"

        # Warm-up
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # Measurement
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            times.append((time.perf_counter() - start) * 1000) # ms
        
        res = {
            "mean": mean(times),
            "median": median(times),
            "std_dev": stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "correctness": error
        }
        
        self.registry[name] = res
        print(f" Done ({res['mean']:.4f} ms)")
        return res

    def save(self):
        with open(self.results_file, "w") as f:
            json.dump(self.registry, f, indent=4)
        print(f"\nResults saved to {self.results_file}")

    def print_table(self):
        print(f"\n--- {self.category} Benchmark Results ---")
        print(f"{'Test':<30} | {'Mean (ms)':<10} | {'Median':<10} | {'Correctness':<10}")
        print("-" * 70)
        for name, r in self.registry.items():
            print(f"{name:<30} | {r['mean']:<10.4f} | {r['median']:<10.4f} | {r['correctness'] or 'N/A'}")
