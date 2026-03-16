#include "core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Forward declarations and helpers
typedef struct {
    double* data;
    int rows;
    int cols;
    int is_copied;
    Py_buffer view;
} MatrixView;

static void free_matrix_view(MatrixView* mv) {
    if (mv->is_copied) {
        if (mv->data) free(mv->data);
    } else if (mv->data != NULL) {
        PyBuffer_Release(&(mv->view));
    }
}

static int parse_matrix_view(PyObject* mat_obj, MatrixView* mv);
static double* parse_matrix(PyObject* mat_obj, int* rows, int* cols);
static PyObject* build_matrix(double* data, int rows, int cols);

// Matrix utilities
static void internal_qr(double* A, double* Q, double* R, int n, int m);
static int internal_lu(double* A, double* L, double* U, int n);
static void internal_matmul(double* A, double* B, double* C, int m, int n, int p);


// Cache-friendly matrix multiplication with OpenMP
PyObject* matrix_multiply(PyObject* self, PyObject* args) {
    PyObject *mat1_obj, *mat2_obj;
    if (!PyArg_ParseTuple(args, "OO", &mat1_obj, &mat2_obj)) return NULL;

    MatrixView mv1, mv2;
    if (parse_matrix_view(mat1_obj, &mv1) != 0) return NULL;
    if (parse_matrix_view(mat2_obj, &mv2) != 0) {
        free_matrix_view(&mv1);
        return NULL;
    }

    if (mv1.cols != mv2.rows) {
        free_matrix_view(&mv1);
        free_matrix_view(&mv2);
        PyErr_SetString(PyExc_ValueError, "Matrix dimension mismatch");
        return NULL;
    }

    int m = mv1.rows;
    int n = mv1.cols;
    int p = mv2.cols;

    if (m == 0 || n == 0 || p == 0) {
        free_matrix_view(&mv1);
        free_matrix_view(&mv2);
        return PyList_New(0);
    }

    double *C = (double*)calloc(m * p, sizeof(double));
    if (!C) {
        free_matrix_view(&mv1);
        free_matrix_view(&mv2);
        return PyErr_NoMemory();
    }

    double *A = mv1.data;
    double *B = mv2.data;

    // Multiply C = A * B
    #define BLOCK_SIZE 64
    if (m > 256 && n > 256 && p > 256) {
        // Cache-blocked ikj with OpenMP
        int i0, k0, j0, i, k, j;
        #pragma omp parallel for private(i0, k0, j0, i, k, j)
        for (i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
            for (k0 = 0; k0 < n; k0 += BLOCK_SIZE) {
                for (j0 = 0; j0 < p; j0 += BLOCK_SIZE) {
                    int i_end = (i0 + BLOCK_SIZE > m) ? m : (i0 + BLOCK_SIZE);
                    int k_end = (k0 + BLOCK_SIZE > n) ? n : (k0 + BLOCK_SIZE);
                    int j_end = (j0 + BLOCK_SIZE > p) ? p : (j0 + BLOCK_SIZE);
                    for (i = i0; i < i_end; i++) {
                        for (k = k0; k < k_end; k++) {
                            double a_ik = A[i * n + k];
                            for (j = j0; j < j_end; j++) {
                                C[i * p + j] += a_ik * B[k * p + j];
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Simple ikj with OpenMP (faster for small matrices)
        int i, k, j;
        #pragma omp parallel for private(k, j)
        for (i = 0; i < m; i++) {
            for (k = 0; k < n; k++) {
                double a_ik = A[i * n + k];
                for (j = 0; j < p; j++) {
                    C[i * p + j] += a_ik * B[k * p + j];
                }
            }
        }
    }

    PyObject* result = build_matrix(C, m, p);

    free(C);
    free_matrix_view(&mv1);
    free_matrix_view(&mv2);

    return result;
}
// Optimized matrix parsing supporting Buffer Protocol (NumPy arrays) and Fallback lists
static int parse_matrix_view(PyObject* mat_obj, MatrixView* mv) {
    mv->data = NULL;
    mv->is_copied = 0;

    // Try Buffer Protocol (NumPy, memoryview, bytes)
    if (PyObject_GetBuffer(mat_obj, &(mv->view), PyBUF_RESTR_CONTIG_RO) == 0) {
        if (mv->view.ndim == 2) {
            mv->rows = (int)mv->view.shape[0];
            mv->cols = (int)mv->view.shape[1];
            mv->data = (double*)mv->view.buf;
            mv->is_copied = 0;
            return 0;
        }
        PyBuffer_Release(&(mv->view));
    }

    // Fallback to List of Lists
    if (!PyList_Check(mat_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected matrix (buffer or list of lists)");
        return -1;
    }

    mv->rows = (int)PyList_Size(mat_obj);
    if (mv->rows == 0) { mv->cols = 0; return 0; }
    
    PyObject* first_row = PyList_GetItem(mat_obj, 0);
    if (!PyList_Check(first_row)) {
        PyErr_SetString(PyExc_TypeError, "Expected list of lists");
        return -1;
    }
    mv->cols = (int)PyList_Size(first_row);
    
    mv->data = (double*)malloc(mv->rows * mv->cols * sizeof(double));
    if (!mv->data) {
        PyErr_NoMemory();
        return -1;
    }
    mv->is_copied = 1;

    for (int i = 0; i < mv->rows; i++) {
        PyObject* row = PyList_GetItem(mat_obj, i);
        for (int j = 0; j < mv->cols; j++) {
            mv->data[i * mv->cols + j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }
    return 0;
}

// Legacy helper for simple calls (uses copy)
static double* parse_matrix(PyObject* mat_obj, int* rows, int* cols) {
    MatrixView mv;
    if (parse_matrix_view(mat_obj, &mv) != 0) return NULL;
    *rows = mv.rows;
    *cols = mv.cols;
    if (mv.is_copied) return mv.data;
    
    // If it was a buffer, we must copy it because the caller expects a freeable pointer
    double* copy = (double*)malloc(mv.rows * mv.cols * sizeof(double));
    if (copy) memcpy(copy, mv.data, mv.rows * mv.cols * sizeof(double));
    free_matrix_view(&mv);
    return copy;
}

static PyObject* build_matrix(double* data, int rows, int cols) {
    PyObject* result = PyList_New(rows);
    if (!result) return NULL;
    for (int i = 0; i < rows; i++) {
        PyObject* row = PyList_New(cols);
        if (!row) return NULL;
        for (int j = 0; j < cols; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(data[i * cols + j]));
        }
        PyList_SetItem(result, i, row);
    }
    return result;
}

// LU Decomposition (Doolittle Algorithm)
PyObject* matrix_lu(PyObject* self, PyObject* args) {
    PyObject* mat_obj;
    if (!PyArg_ParseTuple(args, "O", &mat_obj)) return NULL;

    MatrixView mv;
    if (parse_matrix_view(mat_obj, &mv) != 0) return NULL;

    if (mv.rows != mv.cols) {
        free_matrix_view(&mv);
        PyErr_SetString(PyExc_ValueError, "Matrix must be square");
        return NULL;
    }

    int n = mv.rows;
    double* A = mv.data;
    double* L = (double*)calloc(n * n, sizeof(double));
    double* U = (double*)calloc(n * n, sizeof(double));

    if (!L || !U) {
        free_matrix_view(&mv);
        if (L) free(L);
        if (U) free(U);
        return PyErr_NoMemory();
    }

    int status = internal_lu(A, L, U, n);
    if (status != 0) {
        free_matrix_view(&mv);
        free(L); free(U);
        PyErr_SetString(PyExc_ValueError, "Matrix is singular");
        return NULL;
    }

    PyObject* py_L = build_matrix(L, n, n);
    PyObject* py_U = build_matrix(U, n, n);

    free(L); free(U);
    free_matrix_view(&mv);
    return Py_BuildValue("(OO)", py_L, py_U);
}

// Cholesky Decomposition
PyObject* matrix_cholesky(PyObject* self, PyObject* args) {
    PyObject* mat_obj;
    if (!PyArg_ParseTuple(args, "O", &mat_obj)) return NULL;

    MatrixView mv;
    if (parse_matrix_view(mat_obj, &mv) != 0) return NULL;

    if (mv.rows != mv.cols) {
        free_matrix_view(&mv);
        PyErr_SetString(PyExc_ValueError, "Matrix must be square");
        return NULL;
    }

    int n = mv.rows;
    double* A = mv.data;
    double* L = (double*)calloc(n * n, sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++)
                sum += L[i * n + k] * L[j * n + k];

            if (i == j) {
                double val = A[i * n + i] - sum;
                if (val < 0) {
                    free_matrix_view(&mv); free(L);
                    PyErr_SetString(PyExc_ValueError, "Matrix is not positive-definite");
                    return NULL;
                }
                L[i * n + j] = sqrt(val);
            } else {
                L[i * n + j] = (1.0 / L[j * n + j] * (A[i * n + j] - sum));
            }
        }
    }

    PyObject* py_L = build_matrix(L, n, n);
    free_matrix_view(&mv); free(L);
    return py_L;
}

// Internal helpers for raw buffer operations
static void internal_matmul(double* A, double* B, double* C, int m, int n, int p) {
    for (int i = 0; i < m * p; i++) C[i] = 0;
    if (m > 256 && n > 256 && p > 256) {
        #ifndef BLOCK_SIZE
        #define BLOCK_SIZE 64
        #endif
        for (int i0 = 0; i0 < m; i0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < n; k0 += BLOCK_SIZE) {
                for (int j0 = 0; j0 < p; j0 += BLOCK_SIZE) {
                    int i_end = (i0 + BLOCK_SIZE > m) ? m : (i0 + BLOCK_SIZE);
                    int k_end = (k0 + BLOCK_SIZE > n) ? n : (k0 + BLOCK_SIZE);
                    int j_end = (j0 + BLOCK_SIZE > p) ? p : (j0 + BLOCK_SIZE);
                    for (int i = i0; i < i_end; i++) {
                        for (int k = k0; k < k_end; k++) {
                            double a_ik = A[i * n + k];
                            for (int j = j0; j < j_end; j++) {
                                C[i * p + j] += a_ik * B[k * p + j];
                            }
                        }
                    }
                }
            }
        }
    } else {
        int i, k, j;
        #pragma omp parallel for private(k, j)
        for (i = 0; i < m; i++) {
            for (k = 0; k < n; k++) {
                double a_ik = A[i * n + k];
                for (j = 0; j < p; j++) {
                    C[i * p + j] += a_ik * B[k * p + j];
                }
            }
        }
    }
}

static void internal_qr(double* A, double* Q, double* R, int n, int m) {
    for (int i = 0; i < n * m; i++) Q[i] = A[i];
    for (int k = 0; k < m; k++) {
        double norm = 0;
        for (int i = 0; i < n; i++) norm += Q[i * m + k] * Q[i * m + k];
        norm = sqrt(norm);
        R[k * m + k] = norm;
        if (norm > 1e-12) {
            for (int i = 0; i < n; i++) Q[i * m + k] /= norm;
        }
        for (int j = k + 1; j < m; j++) {
            double dot = 0;
            for (int i = 0; i < n; i++) dot += Q[i * m + k] * Q[i * m + j];
            R[k * m + j] = dot;
            for (int i = 0; i < n; i++) Q[i * m + j] -= dot * Q[i * m + k];
        }
    }
}

// QR Decomposition (Modified Gram-Schmidt)
PyObject* matrix_qr(PyObject* self, PyObject* args) {
    PyObject* mat_obj;
    if (!PyArg_ParseTuple(args, "O", &mat_obj)) return NULL;
    
    MatrixView mv;
    if (parse_matrix_view(mat_obj, &mv) != 0) return NULL;

    int n = mv.rows;
    int m = mv.cols;
    double* A = mv.data;
    double* Q = (double*)malloc(n * m * sizeof(double));
    double* R = (double*)calloc(m * m, sizeof(double));

    if (!Q || !R) {
        free_matrix_view(&mv);
        if (Q) free(Q);
        if (R) free(R);
        return PyErr_NoMemory();
    }

    internal_qr(A, Q, R, n, m);

    PyObject* py_Q = build_matrix(Q, n, m);
    PyObject* py_R = build_matrix(R, m, m);

    free_matrix_view(&mv); free(Q); free(R);
    return Py_BuildValue("(OO)", py_Q, py_R);
}

// Eigenvalue Decomposition (QR Algorithm)
PyObject* matrix_eigen(PyObject* self, PyObject* args) {
    PyObject* mat_obj;
    int iterations = 100;
    if (!PyArg_ParseTuple(args, "O|i", &mat_obj, &iterations)) return NULL;

    MatrixView mv;
    if (parse_matrix_view(mat_obj, &mv) != 0) return NULL;
    
    if (mv.rows != mv.cols) {
        free_matrix_view(&mv);
        return PyErr_Format(PyExc_ValueError, "Must be square matrix");
    }

    int n = mv.rows;
    double* Ak = (double*)malloc(n * n * sizeof(double));
    if (!Ak) { free_matrix_view(&mv); return PyErr_NoMemory(); }
    memcpy(Ak, mv.data, n * n * sizeof(double));

    double* Q = (double*)malloc(n * n * sizeof(double));
    double* R = (double*)malloc(n * n * sizeof(double));
    double* Q_total = (double*)calloc(n * n, sizeof(double));
    double* temp = (double*)malloc(n * n * sizeof(double));

    if (!Q || !R || !Q_total || !temp) {
        free_matrix_view(&mv); free(Ak);
        if (Q) free(Q); if (R) free(R); if (Q_total) free(Q_total); if (temp) free(temp);
        return PyErr_NoMemory();
    }

    // Initialize Q_total as Identity
    for (int i = 0; i < n; i++) Q_total[i * n + i] = 1.0;

    for (int iter = 0; iter < iterations; iter++) {
        internal_qr(Ak, Q, R, n, n);
        internal_matmul(R, Q, Ak, n, n, n);
        internal_matmul(Q_total, Q, temp, n, n, n);
        for (int i = 0; i < n * n; i++) Q_total[i] = temp[i];
    }

    PyObject* py_eigenvalues = PyList_New(n);
    for (int i = 0; i < n; i++) PyList_SetItem(py_eigenvalues, i, PyFloat_FromDouble(Ak[i * n + i]));
    
    PyObject* py_eigenvectors = build_matrix(Q_total, n, n);

    free_matrix_view(&mv); free(Ak); free(Q); free(R); free(Q_total); free(temp);
    return Py_BuildValue("(OO)", py_eigenvalues, py_eigenvectors);
}

static void internal_transpose(double* A, double* AT, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            AT[j * n + i] = A[i * m + j];
        }
    }
}

static void internal_svd(double* A, int n, int m, double* U, double* sigma, double* VT) {
    // Optimized One-Sided Jacobi SVD with OpenMP
    double* B = (double*)malloc(n * m * sizeof(double));
    double* V = (double*)calloc(m * m, sizeof(double));
    for (int i = 0; i < n * m; i++) B[i] = A[i];
    for (int i = 0; i < m; i++) V[i * m + i] = 1.0;

    int max_sweeps = 30;
    double tol = 1e-15;

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double max_converge = 0;
        for (int i = 0; i < m - 1; i++) {
            for (int j = i + 1; j < m; j++) {
                double aii = 0, ajj = 0, aij = 0;
                
                int k_dot;
                #pragma omp parallel for reduction(+:aii, ajj, aij)
                for (k_dot = 0; k_dot < n; k_dot++) {
                    double b_ki = B[k_dot * m + i];
                    double b_kj = B[k_dot * m + j];
                    aii += b_ki * b_ki;
                    ajj += b_kj * b_kj;
                    aij += b_ki * b_kj;
                }

                double iter_converge = fabs(aij) / sqrt(aii * ajj + 1e-20);
                if (iter_converge > max_converge) max_converge = iter_converge;

                if (fabs(aij) > tol) {
                    double tau = (ajj - aii) / (2.0 * aij);
                    double t = (tau >= 0) ? 1.0 / (tau + sqrt(1.0 + tau * tau)) : -1.0 / (-tau + sqrt(1.0 + tau * tau));
                    double c = 1.0 / sqrt(1.0 + t * t);
                    double s = c * t;

                    // Parallel column updates
                    int k_upd;
                    #pragma omp parallel for
                    for (k_upd = 0; k_upd < n; k_upd++) {
                        double b_ki = B[k_upd * m + i];
                        double b_kj = B[k_upd * m + j];
                        B[k_upd * m + i] = c * b_ki - s * b_kj;
                        B[k_upd * m + j] = s * b_ki + c * b_kj;
                    }

                    int k_v;
                    #pragma omp parallel for
                    for (k_v = 0; k_v < m; k_v++) {
                        double v_ki = V[k_v * m + i];
                        double v_kj = V[k_v * m + j];
                        V[k_v * m + i] = c * v_ki - s * v_kj;
                        V[k_v * m + j] = s * v_ki + c * v_kj;
                    }
                }
            }
        }
        if (max_converge < tol) break;
    }

    // Compute singular values and normalized U
    int j_svd;
    #pragma omp parallel for
    for (j_svd = 0; j_svd < m; j_svd++) {
        double norm = 0;
        int i_svd;
        for (i_svd = 0; i_svd < n; i_svd++) norm += B[i_svd * m + j_svd] * B[i_svd * m + j_svd];
        norm = sqrt(norm);
        sigma[j_svd] = norm;
        for (i_svd = 0; i_svd < n; i_svd++) {
            U[i_svd * m + j_svd] = (norm > 1e-15) ? B[i_svd * m + j_svd] / norm : 0.0;
        }
    }

    // Sort singular values (Small sort, serial is fine)
    for (int i = 0; i < m - 1; i++) {
        for (int j = 0; j < m - i - 1; j++) {
            if (sigma[j] < sigma[j + 1]) {
                double temp = sigma[j]; sigma[j] = sigma[j + 1]; sigma[j + 1] = temp;
                for (int k = 0; k < n; k++) {
                    double t = U[k * m + j]; U[k * m + j] = U[k * m + j + 1]; U[k * m + j + 1] = t;
                }
                for (int k = 0; k < m; k++) {
                    double t = V[k * m + j]; V[k * m + j] = V[k * m + j + 1]; V[k * m + j + 1] = t;
                }
            }
        }
    }

    // Transpose V to VT
    int i_tr, j_tr;
    #pragma omp parallel for collapse(2)
    for (i_tr = 0; i_tr < m; i_tr++) {
        for (j_tr = 0; j_tr < m; j_tr++) {
            VT[i_tr * m + j_tr] = V[j_tr * m + i_tr];
        }
    }

    free(B); free(V);
}

PyObject* matrix_svd(PyObject* self, PyObject* args) {
    PyObject* mat_obj;
    if (!PyArg_ParseTuple(args, "O", &mat_obj)) return NULL;

    MatrixView mv;
    if (parse_matrix_view(mat_obj, &mv) != 0) return NULL;

    int n = mv.rows;
    int m = mv.cols;
    double* A = mv.data;

    double* U = (double*)malloc(n * m * sizeof(double));
    double* sigma = (double*)malloc(m * sizeof(double));
    double* VT = (double*)malloc(m * m * sizeof(double));

    if (!U || !sigma || !VT) {
        free_matrix_view(&mv);
        if (U) free(U); if (sigma) free(sigma); if (VT) free(VT);
        return PyErr_NoMemory();
    }

    internal_svd(A, n, m, U, sigma, VT);

    PyObject* py_U = build_matrix(U, n, m);
    PyObject* py_S = PyList_New(m);
    for (int i = 0; i < m; i++) PyList_SetItem(py_S, i, PyFloat_FromDouble(sigma[i]));
    PyObject* py_VT = build_matrix(VT, m, m);

    free_matrix_view(&mv); free(U); free(sigma); free(VT);
    return Py_BuildValue("(OOO)", py_U, py_S, py_VT);
}

int internal_lu(double* A, double* L, double* U, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++)
                sum += (L[i * n + j] * U[j * n + k]);
            U[i * n + k] = A[i * n + k] - sum;
        }
        for (int k = i; k < n; k++) {
            if (i == k)
                L[i * n + i] = 1.0;
            else {
                double sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (L[k * n + j] * U[j * n + i]);
                if (U[i * n + i] == 0) return -1; // Singular
                L[k * n + i] = (A[k * n + i] - sum) / U[i * n + i];
            }
        }
    }
    return 0;
}

PyObject* batched_matrix_lu(PyObject* self, PyObject* args) {
    PyObject* batch_obj;
    Py_ssize_t batch_size;
    PyObject* result;
    double** A_batch;
    double** L_batch;
    double** U_batch;
    int* dims;
    int* status;
    int b;

    if (!PyArg_ParseTuple(args, "O", &batch_obj)) return NULL;
    if (!PyList_Check(batch_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of matrices");
        return NULL;
    }

    batch_size = PyList_Size(batch_obj);
    A_batch = (double**)malloc(batch_size * sizeof(double*));
    L_batch = (double**)malloc(batch_size * sizeof(double*));
    U_batch = (double**)malloc(batch_size * sizeof(double*));
    dims = (int*)malloc(batch_size * sizeof(int));
    status = (int*)malloc(batch_size * sizeof(int));

    // 1. Parse everything (Single-threaded)
    for (b = 0; b < (int)batch_size; b++) {
        int n, m;
        A_batch[b] = parse_matrix(PyList_GetItem(batch_obj, b), &n, &m);
        if (A_batch[b] && n == m) {
            dims[b] = n;
            L_batch[b] = (double*)calloc(n * n, sizeof(double));
            U_batch[b] = (double*)calloc(n * n, sizeof(double));
            status[b] = 0;
        } else {
            if (A_batch[b]) free(A_batch[b]);
            A_batch[b] = NULL;
            L_batch[b] = NULL;
            U_batch[b] = NULL;
            dims[b] = 0;
            status[b] = -2; // Invalid
        }
    }

    // 2. Compute in parallel (No Python API calls)
    #pragma omp parallel for private(b)
    for (b = 0; b < (int)batch_size; b++) {
        if (status[b] == 0) {
            status[b] = internal_lu(A_batch[b], L_batch[b], U_batch[b], dims[b]);
        }
    }

    // 3. Build Python result (Single-threaded)
    result = PyList_New(batch_size);
    for (b = 0; b < (int)batch_size; b++) {
        if (status[b] == 0) {
            PyObject* py_L = build_matrix(L_batch[b], dims[b], dims[b]);
            PyObject* py_U = build_matrix(U_batch[b], dims[b], dims[b]);
            PyList_SetItem(result, b, Py_BuildValue("(OO)", py_L, py_U));
        } else {
            Py_INCREF(Py_None);
            PyList_SetItem(result, b, Py_None);
        }
        if (A_batch[b]) free(A_batch[b]);
        if (L_batch[b]) free(L_batch[b]);
        if (U_batch[b]) free(U_batch[b]);
    }

    free(A_batch); free(L_batch); free(U_batch); free(dims); free(status);
    return result;
}

PyObject* batched_matrix_qr(PyObject* self, PyObject* args) {
    PyObject* batch_obj;
    Py_ssize_t batch_size;
    PyObject* result;
    double** A_batch;
    double** Q_batch;
    double** R_batch;
    int* rows_batch;
    int* cols_batch;
    int b;

    if (!PyArg_ParseTuple(args, "O", &batch_obj)) return NULL;
    if (!PyList_Check(batch_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of matrices");
        return NULL;
    }

    batch_size = PyList_Size(batch_obj);
    A_batch = (double**)malloc(batch_size * sizeof(double*));
    Q_batch = (double**)malloc(batch_size * sizeof(double*));
    R_batch = (double**)malloc(batch_size * sizeof(double*));
    rows_batch = (int*)malloc(batch_size * sizeof(int));
    cols_batch = (int*)malloc(batch_size * sizeof(int));

    // 1. Parse (Single-threaded)
    for (b = 0; b < (int)batch_size; b++) {
        int n, m;
        A_batch[b] = parse_matrix(PyList_GetItem(batch_obj, b), &n, &m);
        if (A_batch[b]) {
            rows_batch[b] = n;
            cols_batch[b] = m;
            Q_batch[b] = (double*)malloc(n * m * sizeof(double));
            R_batch[b] = (double*)calloc(m * m, sizeof(double));
        } else {
            Q_batch[b] = NULL;
            R_batch[b] = NULL;
            rows_batch[b] = 0;
            cols_batch[b] = 0;
        }
    }

    // 2. Compute in parallel
    #pragma omp parallel for private(b)
    for (b = 0; b < (int)batch_size; b++) {
        if (A_batch[b]) {
            internal_qr(A_batch[b], Q_batch[b], R_batch[b], rows_batch[b], cols_batch[b]);
        }
    }

    // 3. Build results
    result = PyList_New(batch_size);
    for (b = 0; b < (int)batch_size; b++) {
        if (A_batch[b]) {
            PyObject* py_Q = build_matrix(Q_batch[b], rows_batch[b], cols_batch[b]);
            PyObject* py_R = build_matrix(R_batch[b], cols_batch[b], cols_batch[b]);
            PyList_SetItem(result, b, Py_BuildValue("(OO)", py_Q, py_R));
        } else {
            Py_INCREF(Py_None);
            PyList_SetItem(result, b, Py_None);
        }
        if (A_batch[b]) free(A_batch[b]);
        if (Q_batch[b]) free(Q_batch[b]);
        if (R_batch[b]) free(R_batch[b]);
    }

    free(A_batch); free(Q_batch); free(R_batch); free(rows_batch); free(cols_batch);
    return result;
}
PyObject* batched_matrix_svd(PyObject* self, PyObject* args) {
    PyObject* batch_obj;
    Py_ssize_t batch_size;
    PyObject* result;
    double** A_batch;
    double** U_batch;
    double** S_batch;
    double** VT_batch;
    int* rows_batch;
    int* cols_batch;
    int b;

    if (!PyArg_ParseTuple(args, "O", &batch_obj)) return NULL;
    if (!PyList_Check(batch_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of matrices");
        return NULL;
    }

    batch_size = PyList_Size(batch_obj);
    A_batch = (double**)malloc(batch_size * sizeof(double*));
    U_batch = (double**)malloc(batch_size * sizeof(double*));
    S_batch = (double**)malloc(batch_size * sizeof(double*));
    VT_batch = (double**)malloc(batch_size * sizeof(double*));
    rows_batch = (int*)malloc(batch_size * sizeof(int));
    cols_batch = (int*)malloc(batch_size * sizeof(int));

    for (b = 0; b < (int)batch_size; b++) {
        int n, m;
        A_batch[b] = parse_matrix(PyList_GetItem(batch_obj, b), &n, &m);
        if (A_batch[b]) {
            rows_batch[b] = n;
            cols_batch[b] = m;
            U_batch[b] = (double*)malloc(n * m * sizeof(double));
            S_batch[b] = (double*)malloc(m * sizeof(double));
            VT_batch[b] = (double*)malloc(m * m * sizeof(double));
        } else {
            rows_batch[b] = 0; cols_batch[b] = 0;
            U_batch[b] = NULL; S_batch[b] = NULL; VT_batch[b] = NULL;
        }
    }

    #pragma omp parallel for private(b)
    for (b = 0; b < (int)batch_size; b++) {
        if (A_batch[b]) {
            internal_svd(A_batch[b], rows_batch[b], cols_batch[b], U_batch[b], S_batch[b], VT_batch[b]);
        }
    }

    result = PyList_New(batch_size);
    for (b = 0; b < (int)batch_size; b++) {
        if (A_batch[b]) {
            PyObject* py_U = build_matrix(U_batch[b], rows_batch[b], cols_batch[b]);
            PyObject* py_S = PyList_New(cols_batch[b]);
            for (int i = 0; i < cols_batch[b]; i++) PyList_SetItem(py_S, i, PyFloat_FromDouble(S_batch[b][i]));
            PyObject* py_VT = build_matrix(VT_batch[b], cols_batch[b], cols_batch[b]);
            PyList_SetItem(result, b, Py_BuildValue("(OOO)", py_U, py_S, py_VT));
        } else {
            Py_INCREF(Py_None);
            PyList_SetItem(result, b, Py_None);
        }
        if (A_batch[b]) free(A_batch[b]);
        if (U_batch[b]) free(U_batch[b]);
        if (S_batch[b]) free(S_batch[b]);
        if (VT_batch[b]) free(VT_batch[b]);
    }

    free(A_batch); free(U_batch); free(S_batch); free(VT_batch); free(rows_batch); free(cols_batch);
    return result;
}

void internal_solve_lu(double* L, double* U, double* b, double* x, int n) {
    double* y = (double*)malloc(n * sizeof(double));
    // Forward substitution Ly = b
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) sum += L[i * n + j] * y[j];
        y[i] = b[i] - sum;
    }
    // Backward substitution Ux = y
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++) sum += U[i * n + j] * x[j];
        x[i] = (y[i] - sum) / U[i * n + i];
    }
    free(y);
}

int internal_invert_lu(double* A, double* Inv, int n) {
    double* L = (double*)calloc(n * n, sizeof(double));
    double* U = (double*)calloc(n * n, sizeof(double));
    if (internal_lu(A, L, U, n) != 0) {
        free(L); free(U);
        return -1;
    }
    double* b = (double*)calloc(n, sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) b[i] = (i == j) ? 1.0 : 0.0;
        internal_solve_lu(L, U, b, x, n);
        for (int i = 0; i < n; i++) Inv[i * n + j] = x[i];
    }
    free(L); free(U); free(b); free(x);
    return 0;
}
