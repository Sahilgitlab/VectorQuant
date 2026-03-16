
import unittest
import numpy as np
from scipy import linalg as scipy_linalg
from vectorquant.core.backend import get_backend, set_backend

class TestNumericalValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_backend("c")
        cls.backend = get_backend()
        print(f"\nUsing backend: {cls.backend.__class__.__name__}")

    def test_matmul_accuracy(self):
        """Test Matrix Multiplication against NumPy"""
        A = np.random.randn(10, 10).tolist()
        B = np.random.randn(10, 10).tolist()
        
        expected = np.dot(np.array(A), np.array(B))
        actual = self.backend.matrix_multiply(A, B)
        
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_lu_decomposition(self):
        """Test LU Decomposition (Basic Doolittle) against NumPy reconstruction"""
        A_np = np.random.randn(5, 5)
        A = A_np.tolist()
        
        # VectorQuant basic LU returns L, U (no pivoting for now)
        L, U = self.backend.lu_decomposition(A)
        
        # Reconstruction: A = L @ U
        reconstructed = np.dot(np.array(L), np.array(U))
        np.testing.assert_allclose(reconstructed, A_np, rtol=1e-10)

    def test_lu_ill_conditioned_stable(self):
        """Test LU on strictly diagonally dominant matrix (stable without pivot)"""
        A_np = np.array([
            [10.0, 1.0, 1.0],
            [1.0, 20.0, 1.0],
            [1.0, 1.0, 30.0]
        ])
        A = A_np.tolist()
        L, U = self.backend.lu_decomposition(A)
        reconstructed = np.dot(np.array(L), np.array(U))
        np.testing.assert_allclose(reconstructed, A_np, rtol=1e-10)

    def test_qr_decomposition(self):
        """Test QR Decomposition against NumPy"""
        A_np = np.random.randn(5, 5)
        A = A_np.tolist()
        
        Q, R = self.backend.qr_decomposition(A)
        
        # Check orthogonality
        Q_np = np.array(Q)
        np.testing.assert_allclose(np.dot(Q_np, Q_np.T), np.eye(5), atol=1e-10)
        
        # Check reconstruction
        reconstructed = np.dot(np.array(Q), np.array(R))
        np.testing.assert_allclose(reconstructed, A_np, rtol=1e-10)

    def test_svd(self):
        """Test SVD against NumPy"""
        A_np = np.random.randn(5, 3)
        A = A_np.tolist()
        
        U, S, Vt = self.backend.svd(A)
        
        # S is expected as a diagonal matrix or a list of singular values
        if isinstance(S[0], list):
            S_mat = np.array(S)
        else:
            S_mat = np.diag(S)
            
        reconstructed = np.dot(np.dot(np.array(U), S_mat), np.array(Vt))
        np.testing.assert_allclose(reconstructed, A_np, rtol=1e-10)

    def test_ill_conditioned_hilbert(self):
        """Test on Hilbert Matrix (highly ill-conditioned)"""
        n = 5
        H = scipy_linalg.hilbert(n)
        H_list = H.tolist()
        
        # Test Determinant/Inverse indirectly via LU
        L, U = self.backend.lu_decomposition(H_list)
        reconstructed = np.dot(np.array(L), np.array(U))
        
        # For Hilbert, we might need slightly looser tolerance as n increases
        np.testing.assert_allclose(reconstructed, H, rtol=1e-8)

    def test_near_singular_matrix(self):
        """Test on near-singular matrix"""
        A = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.00000001],
            [0.0, 1.0, 1.0]
        ])
        A_list = A.tolist()
        
        # QR and SVD should be stable
        Q, R = self.backend.qr_decomposition(A_list)
        reconstructed = np.dot(np.array(Q), np.array(R))
        np.testing.assert_allclose(reconstructed, A, rtol=1e-10)

    def test_large_dynamic_range(self):
        """Test with large differences in magnitude"""
        A = np.array([
            [1e10, 1.0],
            [1.0, 1e-10]
        ])
        A_list = A.tolist()
        
        U, S, Vt = self.backend.svd(A_list)
        if not isinstance(S[0], list):
            S_mat = np.diag(S)
        else:
            S_mat = np.array(S)
            
        reconstructed = np.dot(np.dot(np.array(U), S_mat), np.array(Vt))
        np.testing.assert_allclose(reconstructed, A, rtol=1e-10)

    def test_svd_ill_conditioned(self):
        """Test SVD on ill-conditioned matrix to check stability"""
        A_np = np.array([
            [1.0, 1.0],
            [1.0, 1.0 + 1e-9]
        ])
        A_list = A_np.tolist()
        
        U, S, Vt = self.backend.svd(A_list)
        if not isinstance(S[0], list):
            S_mat = np.diag(S)
        else:
            S_mat = np.array(S)
            
        reconstructed = np.dot(np.dot(np.array(U), S_mat), np.array(Vt))
        # This might fail on the current A.T @ A implementation due to squared condition number
        np.testing.assert_allclose(reconstructed, A_np, rtol=1e-8)

if __name__ == "__main__":
    unittest.main()
