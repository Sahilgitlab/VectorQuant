"""
VectorQuant — Numerical Stability Tests

Validates exactness, robustness against near-zero floating point 
errors, and positive-definite guarantees crucial for financial math.
"""

import sys
import os
import math
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq

def test_covariance_positive_definite():
    """Ensure covariance matrices generated are symmetric positive semi-definite."""
    # Data that might produce rank-deficient or singular matrices
    returns = [
        [0.01, 0.01, 0.01],
        [-0.01, -0.01, -0.01],
        [0.02, 0.02, 0.02]
    ]
    cov = vq.stats.covariance_matrix(returns)
    
    # 1. Check symmetry
    n = len(cov)
    for i in range(n):
        for j in range(n):
            assert abs(cov[i][j] - cov[j][i]) < 1e-12
            
    # 2. Check Positive Semi-Definite (eigenvalues >= 0)
    eigenvalues, _ = vq.linalg.eigen_decomposition(cov)
    for ev in eigenvalues:
        assert ev >= -1e-10  # Allow for small FP inaccuracies


def test_cholesky_factorization_precision():
    """Check Cholesky handles small numerical negative zeros gracefully."""
    A = [[1.0, 0.9999999], 
         [0.9999999, 1.0]]
    
    # Should not throw math domain error (sqrt of negative)
    L = vq.linalg.cholesky_decomposition(A)
    
    # Reconstruct A = L * L^T
    Lt = vq.linalg.transpose(L)
    A_rec = vq.linalg.matrix_multiply(L, Lt)
    
    for i in range(2):
        for j in range(2):
            assert abs(A_rec[i][j] - A[i][j]) < 1e-6


def test_black_scholes_precision_extreme_moneyness():
    """Check option pricing doesn't NaN or diverge at extreme moneyness."""
    # Deep in the money call
    call_itm = vq.derivatives.black_scholes_call(S=1000, K=10, r=0.0, sigma=0.1, T=1.0)
    assert abs(call_itm - 990.0) < 1.0
    
    # Deep out of the money call
    call_otm = vq.derivatives.black_scholes_call(S=10, K=1000, r=0.0, sigma=0.1, T=1.0)
    assert call_otm < 1e-10


def test_nearest_positive_definite_correction():
    """Check that a slightly non-PD matrix is corrected successfully."""
    from vectorquant.core.numerical_stability import nearest_positive_definite
    
    # Intentionally non-PD correlation-like matrix
    A = [[1.0, 0.9, 0.9],
         [0.9, 1.0, 0.1],
         [0.9, 0.1, 1.0]]
    
    A_pd = nearest_positive_definite(A)
    
    # Check PD by ensuring Cholesky works (which requires PD)
    L = vq.linalg.cholesky_decomposition(A_pd)
    assert len(L) == 3
