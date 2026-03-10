"""
VectorQuant Core — Mathematical Kernel

Provides foundational math primitives: linear algebra, probability,
statistics, optimization, and numerical stability.
"""

from .linear_algebra import (
    zeros, identity, vector_norm, dot,
    matrix_add, matrix_subtract, matrix_scale, matrix_multiply,
    transpose, lu_decomposition, solve_lu, matrix_inverse,
    determinant, cholesky_decomposition, trace,
    qr_decomposition, eigen_decomposition, pseudoinverse, svd,
)

from .probability import (
    set_seed, runif, rnorm,
    normal_pdf, normal_cdf, normal_inv_cdf,
    lognormal_pdf, student_t_pdf,
    uniform_pdf, exponential_pdf, poisson_pmf,
)

from .statistics import (
    mean, median, variance, standard_deviation,
    skewness, kurtosis,
    covariance, correlation,
    covariance_matrix, correlation_matrix,
    linear_regression, ridge_regression, bayesian_regression,
)

from .optimization import (
    gradient_descent, newtons_method_opt,
)

from .numerical_stability import (
    condition_number, nearest_positive_definite,
)
