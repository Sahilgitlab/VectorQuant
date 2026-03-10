"""
Statistics Engine
"""
import math
from .linear_algebra import matrix_multiply, transpose, matrix_inverse, dot, matrix_add, matrix_scale, identity

def mean(data):
    if not data: return 0.0
    return sum(data) / len(data)

def median(data):
    if not data: return 0.0
    s_data = sorted(data)
    n = len(s_data)
    if n % 2 == 1:
        return s_data[n // 2]
    return (s_data[n // 2 - 1] + s_data[n // 2]) / 2.0

def variance(data, sample=True):
    n = len(data)
    if n < 2: return 0.0
    m = mean(data)
    var = sum((x - m) ** 2 for x in data) / (n - 1 if sample else n)
    return var

def standard_deviation(data, sample=True):
    return math.sqrt(variance(data, sample))

def skewness(data):
    n = len(data)
    if n < 3: return 0.0
    m = mean(data)
    m2 = sum((x - m) ** 2 for x in data) / n
    m3 = sum((x - m) ** 3 for x in data) / n
    if m2 == 0: return 0.0
    return m3 / (m2 ** 1.5)

def kurtosis(data):
    n = len(data)
    if n < 4: return 0.0
    m = mean(data)
    m2 = sum((x - m) ** 2 for x in data) / n
    m4 = sum((x - m) ** 4 for x in data) / n
    if m2 == 0: return 0.0
    return m4 / (m2 ** 2) - 3.0 # Excess kurtosis

def covariance(x, y, sample=True):
    n = len(x)
    if n != len(y) or n < 2: return 0.0
    mx = mean(x)
    my = mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1 if sample else n)

def correlation(x, y):
    cov = covariance(x, y)
    sx = standard_deviation(x)
    sy = standard_deviation(y)
    if sx == 0 or sy == 0: return 0.0
    return cov / (sx * sy)

def covariance_matrix(data_matrix_cols):
    """
    Input: list of variables, where each variable is a list of observations.
    """
    n_vars = len(data_matrix_cols)
    cov_mat = [[0.0 for _ in range(n_vars)] for _ in range(n_vars)]
    for i in range(n_vars):
        for j in range(i, n_vars):
            c = covariance(data_matrix_cols[i], data_matrix_cols[j])
            cov_mat[i][j] = c
            cov_mat[j][i] = c
    return cov_mat

def correlation_matrix(data_matrix_cols):
    n_vars = len(data_matrix_cols)
    corr_mat = [[1.0 for _ in range(n_vars)] for _ in range(n_vars)]
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            c = correlation(data_matrix_cols[i], data_matrix_cols[j])
            corr_mat[i][j] = c
            corr_mat[j][i] = c
    return corr_mat

def linear_regression(X, y):
    """
    Multiple linear regression.
    X: list of lists (rows of features), should include constant term if intercept desired.
    y: list of target values.
    Returns beta coefficients.
    beta = (X^T X)^-1 X^T y
    """
    X_t = transpose(X)
    XtX = matrix_multiply(X_t, X)
    try:
        inv_XtX = matrix_inverse(XtX)
    except:
        # Fallback to pseudo-inverse logic locally
        from .linear_algebra import pseudoinverse
        inv_XtX = pseudoinverse(XtX)
        
    XtY = matrix_multiply(X_t, [[yi] for yi in y])
    beta_col = matrix_multiply(inv_XtX, XtY)
    return [b[0] for b in beta_col]

def ridge_regression(X, y, lmbda=1.0):
    """
    beta = (X^T X + lambda I)^-1 X^T y
    """
    X_t = transpose(X)
    XtX = matrix_multiply(X_t, X)
    n_cols = len(XtX)
    penalty = matrix_scale(identity(n_cols), lmbda)
    penalty[0][0] = 0.0 # Typically don't penalize intercept
    
    XtX_ridge = matrix_add(XtX, penalty)
    try:
        inv_XtX_ridge = matrix_inverse(XtX_ridge)
    except:
        from .linear_algebra import pseudoinverse
        inv_XtX_ridge = pseudoinverse(XtX_ridge)
        
    XtY = matrix_multiply(X_t, [[yi] for yi in y])
    beta_col = matrix_multiply(inv_XtX_ridge, XtY)
    return [b[0] for b in beta_col]

def bayesian_regression(X, y, alpha_prior=1.0, beta_prior=1.0):
    """
    Simplified Bayesian regression returning MAP estimate of weights (equivalent to Ridge).
    lambda = alpha_prior / beta_prior
    """
    lmbda = alpha_prior / beta_prior
    return ridge_regression(X, y, lmbda)
