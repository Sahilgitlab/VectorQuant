
import numpy as np
import scipy.sparse
from vectorquant.core.backend import get_backend
from benchmarks.bench_utils import BenchmarkRunner

backend = get_backend()
runner = BenchmarkRunner("Sparse Operations")

def run_sparse_benchmarks(size, density):
    print(f"\n--- Size: {size}x{size}, Density: {density} ---")
    
    # Create random sparse matrix in CSR format
    A_sparse = scipy.sparse.random(size, size, density=density, format='csr')
    B_dense = np.random.rand(size, 64) # Multiply by dense tall matrix
    
    data = A_sparse.data.tolist()
    indices = A_sparse.indices.tolist()
    indptr = A_sparse.indptr.tolist()
    B_list = B_dense.tolist()

    runner.run(
        f"SparseDenseMatMul_{size}_{density}",
        backend.sparse_dense_matmul,
        args=(data, indices, indptr, size, 64, size, B_list),
        compare_to=lambda *a: (A_sparse @ B_dense).tolist()
    )

if __name__ == "__main__":
    for size in [1000, 5000]:
        for density in [0.01, 0.05]:
            run_sparse_benchmarks(size, density)
    
    runner.save()
    runner.print_table()
