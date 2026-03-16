#ifndef VQ_CORE_H
#define VQ_CORE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Linalg exports
PyObject* dot(PyObject* self, PyObject* args);
PyObject* matrix_multiply(PyObject* self, PyObject* args);
PyObject* matrix_lu(PyObject* self, PyObject* args);
PyObject* matrix_cholesky(PyObject* self, PyObject* args);
PyObject* matrix_qr(PyObject* self, PyObject* args);
PyObject* matrix_eigen(PyObject* self, PyObject* args);
PyObject* matrix_svd(PyObject* self, PyObject* args);
PyObject* batched_matrix_lu(PyObject* self, PyObject* args);
PyObject* batched_matrix_qr(PyObject* self, PyObject* args);
PyObject* batched_matrix_svd(PyObject* self, PyObject* args);

// Stats exports
PyObject* covariance_matrix(PyObject* self, PyObject* args);
PyObject* incremental_mean_var(PyObject* self, PyObject* args);
PyObject* incremental_covariance(PyObject* self, PyObject* args);

// Stochastic exports
PyObject* simulate_gbm(PyObject* self, PyObject* args);
PyObject* kalman_predict(PyObject* self, PyObject* args);
PyObject* kalman_update(PyObject* self, PyObject* args);

// Optimization exports
PyObject* bfgs_minimize(PyObject* self, PyObject* args);

// Signal Processing exports
PyObject* radix2_fft(PyObject* self, PyObject* args);

// QMC exports
PyObject* sobol_sequence(PyObject* self, PyObject* args);
PyObject* halton_sequence(PyObject* self, PyObject* args);
PyObject* scrambled_sobol(PyObject* self, PyObject* args);

// Autodiff exports
PyObject* dual_op(PyObject* self, PyObject* args);

// Internal C-to-C helpers
int internal_lu(double* A, double* L, double* U, int n);
void internal_solve_lu(double* L, double* U, double* b, double* x, int n);
int internal_invert_lu(double* A, double* Inv, int n);

// Sparse exports
PyObject* sparse_dense_matmul(PyObject* self, PyObject* args);

#endif // VQ_CORE_H
