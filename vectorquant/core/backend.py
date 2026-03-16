"""
VectorQuant Backend Dispatcher
==============================

Handles dynamic dispatch between the pure-Python implementation 
and the high-performance C core extension.
"""

from .config import njit_fallback

class PythonBackend:
    """Pure Python implementation (with optional Numba fallback)"""
    
    @property
    def is_compiled(self):
        return False
        
    # --- Linear Algebra ---
    @staticmethod
    def dot(a, b):
        from .linear_algebra import dot
        return dot(a, b)
        
    @staticmethod
    def matrix_multiply(A, B):
        from .linear_algebra import matrix_multiply
        return matrix_multiply(A, B)
        
    # --- Statistics ---
    @staticmethod
    def covariance_matrix(data_matrix_cols):
        from .statistics import covariance_matrix
        return covariance_matrix(data_matrix_cols)

    @staticmethod
    def simulate_gbm(s0, mu, sigma, t, dt, n_paths, antithetic=False):
        from .stochastic import simulate_gbm
        return simulate_gbm(s0, mu, sigma, t, dt, n_paths, antithetic)

    @staticmethod
    def radix2_fft(input_list):
        from .linear_algebra import radix2_fft
        return radix2_fft(input_list)

    @staticmethod
    def lu_decomposition(A):
        from .linear_algebra import lu_decomposition
        return lu_decomposition(A)

    @staticmethod
    def cholesky_decomposition(A):
        from .linear_algebra import cholesky_decomposition
        return cholesky_decomposition(A)

    @staticmethod
    def qr_decomposition(A):
        from .linear_algebra import qr_decomposition
        return qr_decomposition(A)

    @staticmethod
    def eigen_decomposition(A, num_simulations=100):
        from .linear_algebra import eigen_decomposition
        return eigen_decomposition(A, num_simulations)

    @staticmethod
    def bfgs_minimize(f, grad_f, x0, tol=1e-6, max_iter=100):
        from .optimization import bfgs_minimize
        return bfgs_minimize(f, grad_f, x0, tol, max_iter)

    @staticmethod
    def sobol_sequence(n):
        from .stochastic import sobol_sequence
        return sobol_sequence(n)

    @staticmethod
    def halton_sequence(n, dim):
        from .stochastic import halton_sequence
        return halton_sequence(n, dim)

    @staticmethod
    def scrambled_sobol(n, seed=0):
        from .stochastic import scrambled_sobol
        return scrambled_sobol(n, seed)

    @staticmethod
    def svd(A):
        from .linear_algebra import svd
        return svd(A)

    @staticmethod
    def incremental_mean_var(n, mean, m2, x):
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2
        return n, mean, m2

    @staticmethod
    def incremental_covariance(n, mean_x, mean_y, c_xy, x, y):
        n += 1
        delta_x = x - mean_x
        mean_x += delta_x / n
        delta_y = y - mean_y
        mean_y += delta_y / n
        c_xy += delta_x * (y - mean_y)
        return n, mean_x, mean_y, c_xy

    @staticmethod
    def batched_lu(matrices):
        from .linear_algebra import lu_decomposition
        return [lu_decomposition(m) for m in matrices]

    @staticmethod
    def batched_qr(matrices):
        from .linear_algebra import qr_decomposition
        return [qr_decomposition(m) for m in matrices]

    @staticmethod
    def batched_svd(matrices):
        from .linear_algebra import svd
        return [svd(m) for m in matrices]

    @staticmethod
    def kalman_predict(x, P, F, Q):
        """
        Standard Kalman Predict: x = Fx, P = FPF' + Q
        """
        from .linear_algebra import matrix_multiply, transpose
        x_new = matrix_multiply(F, [[val] for val in x])
        x_new = [row[0] for row in x_new]
        
        P_new = matrix_multiply(matrix_multiply(F, P), transpose(F))
        for i in range(len(P_new)):
            for j in range(len(P_new)):
                P_new[i][j] += Q[i][j]
        return x_new, P_new

    @staticmethod
    def kalman_update(x, P, H, R, z):
        """
        Standard Kalman Update:
        y = z - Hx
        S = HPH' + R
        K = PH' S^-1
        x = x + Ky
        P = (I - KH)P
        """
        from .linear_algebra import matrix_multiply, transpose, matrix_inverse
        n = len(x)
        m = len(z)
        
        # y = z - Hx
        Hx = matrix_multiply(H, [[val] for val in x])
        y = [z[i] - Hx[i][0] for i in range(m)]
        
        # S = HPH' + R
        HT = transpose(H)
        PHT = matrix_multiply(P, HT)
        S = matrix_multiply(H, PHT)
        for i in range(m):
            for j in range(m):
                S[i][j] += R[i][j]
                
        # K = PHT * S^-1
        S_inv = matrix_inverse(S)
        K = matrix_multiply(PHT, S_inv)
        
        # x = x + Ky
        Ky = matrix_multiply(K, [[val] for val in y])
        x_new = [x[i] + Ky[i][0] for i in range(n)]
        
        # P = (I - KH)P
        KH = matrix_multiply(K, H)
        I_KH = [[(1.0 if i == j else 0.0) - KH[i][j] for j in range(n)] for i in range(n)]
        P_new = matrix_multiply(I_KH, P)
        
        return x_new, P_new

    @staticmethod
    def sparse_dense_matmul(data, indices, indptr, rows, cols, k, B):
        """
        Python fallback for CSR Sparse-Dense MatMul
        """
        C = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for p in range(indptr[i], indptr[i+1]):
                col_index = indices[p]
                val = data[p]
                for j in range(cols):
                    C[i][j] += val * B[col_index][j]
        return C

class CBackend(PythonBackend):
    """C Core Extension implementation"""
    
    def __init__(self):
        import vectorquant_c_core
        self.core = vectorquant_c_core
        
    @property
    def is_compiled(self):
        return True
        
    @staticmethod
    def dot(a, b):
        import vectorquant_c_core
        return vectorquant_c_core.dot(a, b)
        
    @staticmethod
    def matrix_multiply(A, B):
        import vectorquant_c_core
        return vectorquant_c_core.matrix_multiply(A, B)
        
    @staticmethod
    def covariance_matrix(data_matrix_cols):
        import vectorquant_c_core
        return vectorquant_c_core.covariance_matrix(data_matrix_cols)

    @staticmethod
    def simulate_gbm(s0, mu, sigma, t, dt, n_paths, antithetic=False):
        import vectorquant_c_core
        flat_list, cols = vectorquant_c_core.simulate_gbm(s0, mu, sigma, t, dt, n_paths, int(antithetic))
        # Pure Python slicing is actually faster than numpy conversion for flat lists of this size
        return [flat_list[i*cols:(i+1)*cols] for i in range(len(flat_list) // cols)]

    @staticmethod
    def linear_regression(X, y):
        # Fallback to Python until Regression is implemented in C
        from .linear_algebra import matrix_multiply
        # Basic fallback not implemented here directly; usually via other means
        return []

    @staticmethod
    def bfgs_minimize(f, grad_f, x0, tol=1e-6, max_iter=100):
        import vectorquant_c_core
        return vectorquant_c_core.bfgs_minimize(f, grad_f, x0, tol, max_iter)

    @staticmethod
    def radix2_fft(input_list):
        import vectorquant_c_core
        return vectorquant_c_core.radix2_fft(input_list)

    @staticmethod
    def lu_decomposition(A):
        import vectorquant_c_core
        return vectorquant_c_core.matrix_lu(A)

    @staticmethod
    def cholesky_decomposition(A):
        import vectorquant_c_core
        return vectorquant_c_core.matrix_cholesky(A)

    @staticmethod
    def qr_decomposition(A):
        import vectorquant_c_core
        return vectorquant_c_core.matrix_qr(A)

    @staticmethod
    def eigen_decomposition(A, num_simulations=100):
        import vectorquant_c_core
        return vectorquant_c_core.matrix_eigen(A, num_simulations)

    @staticmethod
    def sobol_sequence(n):
        import vectorquant_c_core
        return vectorquant_c_core.sobol_sequence(n)

    @staticmethod
    def halton_sequence(n, dim):
        import vectorquant_c_core
        return vectorquant_c_core.halton_sequence(n, dim)

    @staticmethod
    def scrambled_sobol(n, seed=0):
        import vectorquant_c_core
        return vectorquant_c_core.scrambled_sobol(n, seed)

    @staticmethod
    def incremental_mean_var(n, mean, m2, x):
        import vectorquant_c_core
        return vectorquant_c_core.incremental_mean_var(n, mean, m2, x)

    @staticmethod
    def incremental_covariance(n, mean_x, mean_y, c_xy, x, y):
        import vectorquant_c_core
        return vectorquant_c_core.incremental_covariance(n, mean_x, mean_y, c_xy, x, y)

    @staticmethod
    def batched_lu(matrices):
        import vectorquant_c_core
        return vectorquant_c_core.batched_matrix_lu(matrices)

    @staticmethod
    def batched_qr(matrices):
        import vectorquant_c_core
        return vectorquant_c_core.batched_matrix_qr(matrices)

    @staticmethod
    def batched_svd(matrices):
        import vectorquant_c_core
        return vectorquant_c_core.batched_matrix_svd(matrices)

    @staticmethod
    def kalman_predict(x, P, F, Q):
        import vectorquant_c_core
        return vectorquant_c_core.kalman_predict(x, P, F, Q)

    @staticmethod
    def kalman_update(x, P, H, R, z):
        import vectorquant_c_core
        return vectorquant_c_core.kalman_update(x, P, H, R, z)

    @staticmethod
    def sparse_dense_matmul(data, indices, indptr, rows, cols, k, B):
        import vectorquant_c_core
        return vectorquant_c_core.sparse_dense_matmul(data, indices, indptr, rows, cols, k, B)


# Global active backend
from ._c_backend import C_AVAILABLE

if C_AVAILABLE:
    active_backend = CBackend()
else:
    active_backend = PythonBackend()

def get_backend():
    return active_backend

def set_backend(backend_name):
    global active_backend
    if backend_name.lower() == "c":
        if C_AVAILABLE:
            active_backend = CBackend()
        else:
            raise ImportError("C backend requested but vectorquant_c_core is not installed.")
    elif backend_name.lower() == "python":
        active_backend = PythonBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
