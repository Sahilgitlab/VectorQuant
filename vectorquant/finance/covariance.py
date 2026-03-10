"""
Advanced Covariance Estimation Models
"""
import math
from vectorquant.core.linear_algebra import matrix_add, matrix_scale, identity, transpose, matrix_multiply
from vectorquant.core.statistics import covariance_matrix, mean
from vectorquant.core.numerical_stability import condition_number, nearest_positive_definite

def ledoit_wolf_shrinkage(returns_matrix, target_matrix=None, delta=None):
    """
    Computes shrinkage covariance matrix: Sigma* = (1-delta) * Sigma + delta * Target
    If target_matrix is None, uses a diagonal matrix with the average sample variance.
    If delta is None, it uses a simple heuristic (e.g. 0.5 for demonstration, analytical optimal
    delta usually requires complex cross-validation or Ledoit-Wolf analytic formula).
    """
    # returns_matrix columns are assets, rows are observations
    # We expect columns to be passed into covariance_matrix
    cols = transpose(returns_matrix)
    sample_cov = covariance_matrix(cols)
    n = len(sample_cov)
    
    if target_matrix is None:
        # Compute avg variance
        avg_var = sum(sample_cov[i][i] for i in range(n)) / n
        target_matrix = [[avg_var if i == j else 0.0 for j in range(n)] for i in range(n)]
        
    if delta is None:
        delta = 0.5 # Simplified heuristic
        
    delta = max(0.0, min(delta, 1.0))
    
    part1_cov = matrix_scale(sample_cov, 1.0 - delta)
    part2_target = matrix_scale(target_matrix, delta)
    
    shrunk_cov = matrix_add(part1_cov, part2_target)
    
    # Guarantee positive definiteness
    return nearest_positive_definite(shrunk_cov)

def ewma_covariance(returns_matrix, lmbda=0.94):
    """
    Computes exponentially weighted moving average covariance.
    Expects chronological returns: returns_matrix[t][asset_i]
    Sigma_t = lambda * Sigma_{t-1} + (1-lambda) * r_t * r_t^T
    Returns the final covariance matrix at time T.
    """
    if not returns_matrix: return []
    
    T = len(returns_matrix)
    N = len(returns_matrix[0])
    
    # Initialize with standard covariance of first portion (or just zeros)
    sigma = [[0.0 for _ in range(N)] for _ in range(N)]
    
    # Start iterating
    for t in range(T):
        r_t = [[returns_matrix[t][i]] for i in range(N)]
        r_t_trans = transpose(r_t)
        innov = matrix_multiply(r_t, r_t_trans)
        
        term1 = matrix_scale(sigma, lmbda)
        term2 = matrix_scale(innov, 1.0 - lmbda)
        sigma = matrix_add(term1, term2)
        
    return nearest_positive_definite(sigma)

def robust_covariance_mcd_approx(returns_matrix, trim_fraction=0.1):
    """
    Approximation of Minimum Covariance Determinant by trimming worst outliers.
    """
    T = len(returns_matrix)
    if T == 0: return []
    N = len(returns_matrix[0])
    
    # Find squared Mahalanobis-like distances roughly by distance from mean
    col_means = [sum(returns_matrix[t][i] for t in range(T)) / T for i in range(N)]
    
    distances = []
    for t in range(T):
        dist = sum((returns_matrix[t][i] - col_means[i])**2 for i in range(N))
        distances.append((dist, t))
        
    distances.sort()
    
    # Keep the best (1 - trim_fraction)
    n_keep = int(T * (1.0 - trim_fraction))
    kept_indices = [idx for _, idx in distances[:n_keep]]
    
    kept_returns = [returns_matrix[idx] for idx in kept_indices]
    cols = transpose(kept_returns)
    sample_cov = covariance_matrix(cols)
    
    return nearest_positive_definite(sample_cov)
