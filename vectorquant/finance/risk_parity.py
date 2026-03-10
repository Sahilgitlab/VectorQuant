"""
Hierarchical Risk Parity (HRP)
"""
from vectorquant.finance.network_theory import correlation_distance
from vectorquant.core.statistics import correlation_matrix

def _get_distance_matrix(corr_matrix):
    n = len(corr_matrix)
    dist = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = correlation_distance(corr_matrix[i][j])
    return dist

def hrp_recursive_bisection(cov_matrix, sort_order):
    """
    Very simplified Hierarchical Risk Parity recursive weighting.
    Requires a pre-sorted list of indices (e.g. from quasi-diagonalization of distance matrix).
    """
    n = len(sort_order)
    weights = [1.0] * n
    
    def recurse(items):
        if len(items) == 1:
            return
        
        # Bisect
        split = len(items) // 2
        left = items[:split]
        right = items[split:]
        
        # Calc cluster variance (inverse proportion)
        v_left = sum(cov_matrix[i][i] for i in left) # Naive diag approx for snippet
        v_right = sum(cov_matrix[i][i] for i in right)
        
        # Alloc factor
        alpha = 1.0 - v_left / (v_left + v_right) if v_left + v_right > 0 else 0.5
        
        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= (1.0 - alpha)
            
        recurse(left)
        recurse(right)
        
    recurse(sort_order)
    
    # Reconstruct final ordered weights
    final_weights = [0.0] * n
    for i, orig_idx in enumerate(sort_order):
        final_weights[orig_idx] = weights[i]
        
    return final_weights
