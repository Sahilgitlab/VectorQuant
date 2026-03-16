
import random
import time
from vectorquant.core.backend import get_backend, set_backend

def generate_random_matrix(n):
    return [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]

def test_batched_linalg():
    batch_size = 100
    dim = 5
    matrices = [generate_random_matrix(dim) for _ in range(batch_size)]
    
    backend = get_backend()
    
    # Test Batched LU
    print(f"Running batched LU for {batch_size} matrices of size {dim}x{dim}...")
    start = time.perf_counter()
    results_lu = backend.batched_lu(matrices)
    end = time.perf_counter()
    print(f"Batched LU took {end - start:.4f}s")
    
    assert len(results_lu) == batch_size
    for i in range(10):  # Check first 10
        if results_lu[i] is not None:
            L, U = results_lu[i]
            assert len(L) == dim
            assert len(U) == dim
            # A ~= LU check could be added here
            
    # Test Batched QR
    print(f"Running batched QR for {batch_size} matrices of size {dim}x{dim}...")
    start = time.perf_counter()
    results_qr = backend.batched_qr(matrices)
    end = time.perf_counter()
    print(f"Batched QR took {end - start:.4f}s")
    
    assert len(results_qr) == batch_size
    for i in range(10):
        if results_qr[i] is not None:
            Q, R = results_qr[i]
            assert len(Q) == dim
            assert len(R) == dim

    # Test Batched SVD
    print(f"Running batched SVD for {batch_size} matrices of size {dim}x{dim}...")
    start = time.perf_counter()
    results_svd = backend.batched_svd(matrices)
    end = time.perf_counter()
    print(f"Batched SVD took {end - start:.4f}s")
    
    assert len(results_svd) == batch_size
    for i in range(10):
        if results_svd[i] is not None:
            U, S, VT = results_svd[i]
            assert len(U) == dim
            assert len(S) == dim
            assert len(VT) == dim

    print("Batched Linalg verification passed!")

if __name__ == "__main__":
    print("Testing C Backend:")
    set_backend("c")
    test_batched_linalg()
    
    print("\nTesting Python Backend:")
    set_backend("python")
    test_batched_linalg()
