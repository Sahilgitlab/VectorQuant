
import time
import numpy as np
from vectorquant.core.backend import get_backend, set_backend

def benchmark_op(name, op_func, A_list, n_iter=10):
    start = time.perf_counter()
    for _ in range(n_iter):
        op_func(A_list)
    end = time.perf_counter()
    return (end - start) / n_iter

def run_benchmarks():
    set_backend("c")
    backend = get_backend()
    
    print("VectorQuant Performance Benchmark (C-Core vs NumPy)")
    print("=" * 60)
    print(f"{'Operation':<20} | {'VectorQuant (ms)':<20} | {'NumPy (ms)':<20}")
    print("-" * 60)
    
    sizes = [100, 500]
    
    for size in sizes:
        A_np = np.random.randn(size, size)
        A_list = A_np.tolist()
        
        # 1. Dot Product (Vector 1D)
        v1 = np.random.randn(size).tolist()
        v2 = np.random.randn(size).tolist()
        v1_np = np.array(v1)
        v2_np = np.array(v2)
        
        vq_dot = benchmark_op("Dot", lambda x: backend.dot(v1, v2), None) * 1000
        np_dot = benchmark_op("Dot", lambda x: np.dot(v1_np, v2_np), None) * 1000
        print(f"{f'Dot ({size})':<20} | {vq_dot:<20.4f} | {np_dot:<20.4f}")
        
        # 2. Matrix Multiply
        vq_matmul = benchmark_op("MatMul", lambda x: backend.matrix_multiply(A_list, A_list), None) * 1000
        np_matmul = benchmark_op("MatMul", lambda x: np.dot(A_np, A_np), None) * 1000
        print(f"{f'MatMul ({size}x{size})':<20} | {vq_matmul:<20.4f} | {np_matmul:<20.4f}")
        
        # 3. QR Decomposition
        vq_qr = benchmark_op("QR", lambda x: backend.qr_decomposition(A_list), None) * 1000
        np_qr = benchmark_op("QR", lambda x: np.linalg.qr(A_np), None) * 1000
        print(f"{f'QR ({size}x{size})':<20} | {vq_qr:<20.4f} | {np_qr:<20.4f}")

        # 4. SVD (only for 100x100 to keep it fast for now)
        if size <= 256:
            vq_svd = benchmark_op("SVD", lambda x: backend.svd(A_list), None) * 1000
            np_svd = benchmark_op("SVD", lambda x: np.linalg.svd(A_np), None) * 1000
            print(f"{f'SVD ({size}x{size})':<20} | {vq_svd:<20.4f} | {np_svd:<20.4f}")

    print("=" * 60)

if __name__ == "__main__":
    run_benchmarks()
