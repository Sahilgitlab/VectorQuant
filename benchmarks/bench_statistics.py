
import numpy as np
from vectorquant.core.backend import get_backend
from benchmarks.bench_utils import BenchmarkRunner

backend = get_backend()
runner = BenchmarkRunner("Statistics")

def run_stats_benchmarks(size):
    print(f"\n--- Vector Size: {size} ---")
    data = np.random.rand(size).tolist()
    data_np = np.array(data)

    # 1. Covariance Matrix (Requires multiple columns)
    cols = 5
    data_matrix = [np.random.rand(size).tolist() for _ in range(cols)]
    runner.run(
        f"Covariance_{size}x{cols}",
        backend.covariance_matrix,
        args=(data_matrix,),
        compare_to=lambda d: np.cov(np.array(d), ddof=1).tolist()
    )

    # 2. Incremental Mean/Var Update (Streaming)
    n, mean, m2 = 100, 0.5, 10.0
    x = 0.75
    runner.run(
        f"Incremental_MeanVar",
        backend.incremental_mean_var,
        args=(n, mean, m2, x),
        compare_to=lambda n, m, m2, x: (n+1, m + (x-m)/(n+1), m2 + (x-m)*(x - (m + (x-m)/(n+1))))
    )

    # 3. Incremental Covariance
    n, mx, my, cxy = 100, 0.5, 0.5, 5.0
    x, y = 0.8, 0.2
    runner.run(
        f"Incremental_Cov",
        backend.incremental_covariance,
        args=(n, mx, my, cxy, x, y)
    )

if __name__ == "__main__":
    for size in [1000, 10000, 100000]:
        run_stats_benchmarks(size)
    
    runner.save()
    runner.print_table()
