"""
VectorQuant — Core Module Tests

Tests for linear algebra, probability, statistics, and optimization.
"""

import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq


# ─── Linear Algebra ──────────────────────────────────────────────────────────

def test_identity_matrix():
    I = vq.linalg.identity(3)
    assert len(I) == 3
    assert I[0][0] == 1.0
    assert I[0][1] == 0.0
    assert I[1][1] == 1.0


def test_matrix_multiply():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = vq.linalg.matrix_multiply(A, B)
    assert C[0][0] == 19  # 1*5 + 2*7
    assert C[0][1] == 22  # 1*6 + 2*8
    assert C[1][0] == 43  # 3*5 + 4*7
    assert C[1][1] == 50  # 3*6 + 4*8


def test_determinant():
    A = [[1, 2], [3, 4]]
    det = vq.linalg.determinant(A)
    assert abs(det - (-2.0)) < 1e-8


def test_matrix_inverse():
    A = [[4, 7], [2, 6]]
    inv = vq.linalg.matrix_inverse(A)
    I = vq.linalg.matrix_multiply(A, inv)
    # Should approximate identity
    assert abs(I[0][0] - 1.0) < 1e-8
    assert abs(I[0][1] - 0.0) < 1e-8
    assert abs(I[1][0] - 0.0) < 1e-8
    assert abs(I[1][1] - 1.0) < 1e-8


def test_cholesky_positive_definite():
    """Cholesky should only succeed on PD matrices and produce L * L^T = A."""
    A = [[4, 2], [2, 3]]
    L = vq.linalg.cholesky_decomposition(A)
    # L * L^T should approximate A
    Lt = vq.linalg.transpose(L)
    A_reconstructed = vq.linalg.matrix_multiply(L, Lt)
    for i in range(2):
        for j in range(2):
            assert abs(A_reconstructed[i][j] - A[i][j]) < 1e-8


def test_dot_product():
    a = [1, 2, 3]
    b = [4, 5, 6]
    assert vq.linalg.dot(a, b) == 32  # 1*4 + 2*5 + 3*6


# ─── Probability ─────────────────────────────────────────────────────────────

def test_normal_pdf_at_zero():
    pdf = vq.prob.normal_pdf(0, mu=0, sigma=1)
    expected = 1.0 / math.sqrt(2 * math.pi)
    assert abs(pdf - expected) < 1e-8


def test_normal_cdf_symmetry():
    assert abs(vq.prob.normal_cdf(0) - 0.5) < 1e-4
    assert vq.prob.normal_cdf(10) > 0.999
    assert vq.prob.normal_cdf(-10) < 0.001


def test_set_seed_reproducibility():
    vq.prob.set_seed(42)
    a = vq.prob.rnorm()
    vq.prob.set_seed(42)
    b = vq.prob.rnorm()
    assert a == b


# ─── Statistics ──────────────────────────────────────────────────────────────

def test_mean():
    assert vq.stats.mean([1, 2, 3, 4, 5]) == 3.0


def test_median():
    assert vq.stats.median([1, 3, 5, 7, 9]) == 5
    assert vq.stats.median([1, 2, 3, 4]) == 2.5


def test_variance():
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    var = vq.stats.variance(data, sample=False)
    assert abs(var - 4.0) < 1e-8


def test_correlation():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    corr = vq.stats.correlation(x, y)
    assert abs(corr - 1.0) < 1e-8  # Perfect correlation


def test_covariance_matrix_symmetry():
    data = [[1, 2, 3], [4, 5, 6]]
    cov = vq.stats.covariance_matrix(data)
    assert len(cov) == 2
    assert abs(cov[0][1] - cov[1][0]) < 1e-12


# ─── Optimization ────────────────────────────────────────────────────────────

def test_gradient_descent_quadratic():
    """Minimize f(x) = x^2, should converge to 0."""
    f = lambda x: x[0]**2
    grad = lambda x: [2*x[0]]
    result = vq.optim.gradient_descent(f, grad, [5.0], lr=0.1)
    assert abs(result[0]) < 1e-4


# ─── Numerical Precision ────────────────────────────────────────────────────

def test_covariance_positive_definite_after_npd():
    """nearest_positive_definite should produce a PD matrix."""
    from vectorquant.core.numerical_stability import nearest_positive_definite
    # A matrix that is NOT positive definite
    A = [[1, 2], [2, 1]]
    A_pd = nearest_positive_definite(A)
    # Check eigenvalues are positive (check via diagonal of Cholesky)
    L = vq.linalg.cholesky_decomposition(A_pd)
    for i in range(len(L)):
        assert L[i][i] > 0
