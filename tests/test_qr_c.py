import pytest
import random
import math
from vectorquant.core.backend import PythonBackend, CBackend, C_AVAILABLE
from vectorquant.core.linear_algebra import matrix_multiply, transpose

@pytest.mark.skipif(not C_AVAILABLE, reason="C backend not available")
def test_qr_decomposition_c():
    n, m = 4, 3
    # Create a random n x m matrix
    A = [[random.random() for _ in range(m)] for _ in range(n)]

    cb = CBackend()
    pb = PythonBackend()
    
    Q_c, R_c = cb.qr_decomposition(A)
    Q_p, R_p = pb.qr_decomposition(A)
    
    # Verify reconstruction A = Q * R
    A_rec = matrix_multiply(Q_c, R_c)
    
    for i in range(n):
        for j in range(m):
            assert abs(A_rec[i][j] - A[i][j]) < 1e-10
            
    # Verify Q is orthogonal (Q^T * Q = I)
    QT_Q = matrix_multiply(transpose(Q_c), Q_c)
    for i in range(m):
        for j in range(m):
            expected = 1.0 if i == j else 0.0
            assert abs(QT_Q[i][j] - expected) < 1e-10

    # Verify R is upper triangular
    for i in range(m):
        for j in range(i):
            assert abs(R_c[i][j]) < 1e-10

    print("QR Decomposition test passed!")

if __name__ == "__main__":
    test_qr_decomposition_c()
