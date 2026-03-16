import pytest
import random
from vectorquant.core.backend import PythonBackend, CBackend, C_AVAILABLE
from vectorquant.core.linear_algebra import matrix_multiply

@pytest.mark.skipif(not C_AVAILABLE, reason="C backend not available")
def test_lu_decomposition_c():
    n = 4
    A = [[random.random() for _ in range(n)] for _ in range(n)]
    
    cb = CBackend()
    pb = PythonBackend()
    
    L_c, U_c = cb.lu_decomposition(A)
    L_p, U_p = pb.lu_decomposition(A)
    
    # Verify reconstruction A = L * U
    A_rec = matrix_multiply(L_c, U_c)
    
    for i in range(n):
        for j in range(n):
            assert abs(A_rec[i][j] - A[i][j]) < 1e-10
            assert abs(L_c[i][j] - L_p[i][j]) < 1e-10
            assert abs(U_c[i][j] - U_p[i][j]) < 1e-10

if __name__ == "__main__":
    test_lu_decomposition_c()
    print("LU Decomposition test passed!")
