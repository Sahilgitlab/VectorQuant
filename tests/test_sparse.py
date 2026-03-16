
import math
from vectorquant.core.backend import get_backend, set_backend

def test_sparse_matmul():
    # Sparse matrix A (3x4):
    # [1, 0, 0, 2]
    # [0, 0, 3, 0]
    # [4, 0, 0, 5]
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    indices = [0, 3, 2, 0, 3]
    indptr = [0, 2, 3, 5]
    rows = 3
    k = 4
    
    # Dense matrix B (4x2):
    # [1, 2]
    # [3, 4]
    # [5, 6]
    # [7, 8]
    B = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ]
    cols = 2

    # Expected Result C = A * B (3x2):
    # [1*1 + 2*7, 1*2 + 2*8] = [15, 18]
    # [3*5, 3*6]             = [15, 18]
    # [4*1 + 5*7, 4*2 + 5*8] = [39, 48]
    expected = [
        [15.0, 18.0],
        [15.0, 18.0],
        [39.0, 48.0]
    ]

    backends = ["python", "c"]
    results = {}

    for b_name in backends:
        set_backend(b_name)
        backend = get_backend()
        
        C = backend.sparse_dense_matmul(data, indices, indptr, rows, cols, k, B)
        results[b_name] = C
        print(f"\nResults for {b_name} backend:")
        for row in C:
            print(row)

    python_C = results["python"]
    c_C = results["c"]

    for i in range(rows):
        for j in range(cols):
            assert math.isclose(python_C[i][j], expected[i][j], rel_tol=1e-9)
            assert math.isclose(c_C[i][j], expected[i][j], rel_tol=1e-9)

    print("\nSparse Matrix Multiplication verification passed!")

if __name__ == "__main__":
    test_sparse_matmul()
