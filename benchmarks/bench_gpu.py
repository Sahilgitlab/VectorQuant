
import numpy as np
import time
from vectorquant.core.backend import get_backend
from benchmarks.bench_utils import BenchmarkRunner

runner = BenchmarkRunner("GPU Acceleration")

def run_gpu_benchmarks():
    try:
        import cupy as cp
        GPU_AVAILABLE = True
    except ImportError:
        GPU_AVAILABLE = False

    if not GPU_AVAILABLE:
        print("GPU (CuPy) not available. Skipping GPU benchmarks.")
        runner.run("GPU_Status", lambda: "SKIPPED")
        return

    # If CuPy is available, benchmark a 1000x1000 MatMul
    size = 1000
    A_gpu = cp.random.rand(size, size)
    B_gpu = cp.random.rand(size, size)
    
    runner.run(
        f"CuPy_MatMul_{size}",
        lambda: cp.dot(A_gpu, B_gpu),
        iterations=100,
        warmup=10
    )

    # Compare with CPU NumPy
    A_cpu = np.random.rand(size, size)
    B_cpu = np.random.rand(size, size)
    runner.run(
        f"NumPy_MatMul_{size}",
        lambda: np.dot(A_cpu, B_cpu),
        iterations=10,
        warmup=2
    )

if __name__ == "__main__":
    run_gpu_benchmarks()
    runner.save()
    runner.print_table()
