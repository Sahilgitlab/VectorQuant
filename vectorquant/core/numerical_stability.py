"""
Numerical Stability Safeguards and Covariance Extensions
"""
import math
from .linear_algebra import matrix_multiply, transpose, identity, svd, matrix_inverse, eigen_decomposition
from .statistics import covariance_matrix

def condition_number(A):
    """
    Computes condition number kappa(A) using SVD.
    kappa(A) = sigma_max / sigma_min
    If close to 1, stable. If very large, unstable.
    """
    try:
        _, singular_values, _ = svd(A)
        if len(singular_values) == 0: return float('inf')
        s_max = singular_values[0]
        s_min = singular_values[-1]
        
        if s_min < 1e-12:
            return float('inf')
        return s_max / s_min
    except:
        return float('inf')

def nearest_positive_definite(A, epsilon=1e-8):
    """
    Finds the nearest positive definite matrix to a symmetric matrix A.
    Uses eigenvalue decomposition and thresholds negative eigenvalues.
    B = (A + A_t) / 2
    eig_vals, eig_vecs = eig(B)
    eig_vals = max(eig_val, epsilon)
    A_pd = eig_vecs * diag(eig_vals) * eig_vecs^T
    """
    n = len(A)
    # Ensure symmetric
    B = [[(A[i][j] + A[j][i]) / 2.0 for j in range(n)] for i in range(n)]
    
    # Eigendecomposition
    eigvals, eigvecs_T = eigen_decomposition(B)
    eigvecs = transpose(eigvecs_T) # Columns are eigenvectors
    
    # Threshold eigenvalues
    new_eigvals = [max(ev, epsilon) for ev in eigvals]
    
    # Construct diagonal matrix
    Lambda = [[new_eigvals[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    # Reconstruct: V * Lambda * V^T
    temp = matrix_multiply(eigvecs, Lambda)
    A_pd = matrix_multiply(temp, eigvecs_T)
    
    # Force perfect symmetry again due to fp errors
    A_pd_sym = [[(A_pd[i][j] + A_pd[j][i]) / 2.0 for j in range(n)] for i in range(n)]
    return A_pd_sym
