import pytest
import random
import math
from vectorquant.core.backend import PythonBackend, CBackend, C_AVAILABLE
from vectorquant.core.linear_algebra import matrix_multiply

def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

@pytest.mark.skipif(not C_AVAILABLE, reason="C backend not available")
def test_cholesky_decomposition_c():
    n = 3
    # Create a symmetric positive-definite matrix A = B * B^T
    B = [[random.random() for _ in range(n)] for _ in range(n)]
    A = matrix_multiply(B, transpose(B))
    
    # Add small value to diagonal for stability
    for i in range(n):
        A[i][i] += 0.1

    cb = CBackend()
    pb = PythonBackend()
    
    L_c = cb.cholesky_decomposition(A)
    L_p = pb.cholesky_decomposition(A)
    
    # Verify reconstruction A = L * L^T
    A_rec = matrix_multiply(L_c, transpose(L_c))
    
    for i in range(n):
        for j in range(n):
            assert abs(A_rec[i][j] - A[i][j]) < 1e-10
            assert abs(L_c[i][j] - L_p[i][j]) < 1e-10

if __name__ == "__main__":
    test_cholesky_decomposition_c()
    print("Cholesky Decomposition test passed!")
