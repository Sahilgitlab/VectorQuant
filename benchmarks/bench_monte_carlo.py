
import numpy as np
import math
from vectorquant.core.backend import get_backend
from benchmarks.bench_utils import BenchmarkRunner

backend = get_backend()
runner = BenchmarkRunner("Monte Carlo")

def run_mc_benchmarks(paths):
    print(f"\n--- Paths: {paths:,} ---")
    
    # Geometric Brownian Motion
    s0, mu, sigma, t, dt = 100.0, 0.05, 0.2, 1.0, 1/252
    
    def vq_gbm():
        if paths >= 100000:
            import vectorquant_c_core
            return vectorquant_c_core.simulate_gbm(s0, mu, sigma, t, dt, paths, int(True))
        return backend.simulate_gbm(s0, mu, sigma, t, dt, paths, antithetic=True)

    def np_gbm():
        steps = int(t/dt)
        Z = np.random.standard_normal((paths * 2, steps))
        drift = (mu - 0.5 * sigma**2) * dt
        vol = sigma * math.sqrt(dt)
        return 100 * np.exp(np.cumsum(drift + vol * Z, axis=1))

    runner.run(
        f"GBM_{paths}",
        vq_gbm,
        iterations=5,
        warmup=2
    )
    
    runner.run(
        f"NumPy_GBM_{paths}",
        np_gbm,
        iterations=5,
        warmup=2
    )

if __name__ == "__main__":
    for paths in [10000, 100000, 200000]:
        run_mc_benchmarks(paths)
    
    runner.save()
    runner.print_table()
