import pytest
import random
from vectorquant.core.backend import PythonBackend, CBackend, C_AVAILABLE
from vectorquant.core.linear_algebra import matrix_multiply, transpose

def is_orthogonal(Q, tol=1e-8):
    n = len(Q)
    m = len(Q[0])
    # Q^T * Q = I?
    QT_Q = matrix_multiply(transpose(Q), Q)
    for i in range(m):
        for j in range(m):
            expected = 1.0 if i == j else 0.0
            if abs(QT_Q[i][j] - expected) > tol:
                return False
    return True

@pytest.mark.skipif(not C_AVAILABLE, reason="C backend not available")
def test_svd_c():
    n, m = 4, 3
    # Create a random matrix
    A = [[random.random() for _ in range(m)] for _ in range(n)]

    cb = CBackend()
    U, S, VT = cb.svd(A)

    # 1. Verify Reconstruction: A = U * S_diag * VT
    # S is a list of singular values. Create S_diag (m x m if we use thin SVD logic)
    # Our C implementation returns U (n x m), S (m), VT (m x m)
    S_diag = [[0.0]*m for _ in range(m)]
    for i in range(m):
        S_diag[i][i] = S[i]

    # U * S_diag
    U_S = matrix_multiply(U, S_diag)
    # (U * S_diag) * VT
    A_rec = matrix_multiply(U_S, VT)

    for i in range(n):
        for j in range(m):
            assert abs(A_rec[i][j] - A[i][j]) < 1e-8

    # 2. Verify Orthogonality
    assert is_orthogonal(U)
    assert is_orthogonal(transpose(VT)) # V is orthogonal

    # 3. Verify Singular Values are sorted
    for i in range(len(S) - 1):
        assert S[i] >= S[i+1]

    print("SVD test passed!")

if __name__ == "__main__":
    test_svd_c()
