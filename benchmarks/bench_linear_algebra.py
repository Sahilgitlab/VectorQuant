
import numpy as np
import scipy.linalg
from vectorquant.core.backend import get_backend
from benchmarks.bench_utils import BenchmarkRunner

backend = get_backend()
runner = BenchmarkRunner("Linear Algebra")

def run_linalg_benchmarks(size):
    print(f"\n--- Matrix Size: {size}x{size} ---")
    A = np.random.rand(size, size).tolist()
    B = np.random.rand(size, size).tolist()
    A_np = np.array(A)
    B_np = np.array(B)

    # 1. Matrix Multiplication
    runner.run(
        f"MatMul_{size}",
        backend.matrix_multiply,
        args=(A, B),
        compare_to=lambda a, b: np.dot(np.array(a), np.array(b)).tolist()
    )

    # 2. LU Decomposition
    runner.run(
        f"LU_{size}",
        backend.lu_decomposition,
        args=(A,),
        compare_to=lambda a: scipy.linalg.lu(np.array(a))[1:3] # Returns (P, L, U) - we compare L, U
    )

    # 3. QR Decomposition
    runner.run(
        f"QR_{size}",
        backend.qr_decomposition,
        args=(A,),
        compare_to=lambda a: scipy.linalg.qr(np.array(a))
    )

    # 4. Eigen Decomposition (Small iterations for speed in benchmark)
    eigen_iters = 5 if size > 50 else 100
    runner.run(
        f"Eigen_{size}",
        backend.eigen_decomposition,
        args=(A, 20), # Fewer internal iterations for large matrices
        iterations=eigen_iters,
        compare_to=lambda a, n: (np.sort(scipy.linalg.eigvals(np.array(a))), None) # Only compare sorted eigenvalues
    )

if __name__ == "__main__":
    for size in [50, 200, 500]:
        run_linalg_benchmarks(size)
    
    runner.save()
    runner.print_table()
