"""
Copulas
"""
import math
from vectorquant.core.probability import normal_cdf, normal_inv_cdf, rnorm
from vectorquant.core.linear_algebra import cholesky_decomposition, matrix_multiply, transpose

def generate_gaussian_copula_samples(correlation_matrix, n_samples):
    """
    Generates samples exhibiting the dependency structure of the given correlation matrix,
    using a Gaussian Copula.
    Returns a matrix of shape (n_samples, n_vars) with uniform margins in [0, 1].
    """
    n_vars = len(correlation_matrix)
    
    # 1. Decompose correlation matrix
    try:
        L = cholesky_decomposition(correlation_matrix)
    except:
        # Fallback to identity if poorly conditioned
        from .linear_algebra import identity
        L = identity(n_vars)
        
    samples = []
    for _ in range(n_samples):
        # 2. Draw independent standard normals
        z = [[rnorm(0, 1)] for _ in range(n_vars)]
        
        # 3. Apply Cholesky
        x = matrix_multiply(L, z)
        
        # 4. Map back to uniform via standard normal CDF
        u = [normal_cdf(x[i][0]) for i in range(n_vars)]
        samples.append(u)
        
    return samples
