#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <immintrin.h>
#include "core.h"

/* 
 * VectorQuant C-Core Extension
 * Performance-critical numerical kernels.
 */

static PyObject* dot_product(PyObject* self, PyObject* args) {
    PyObject *obj1, *obj2;
    if (!PyArg_ParseTuple(args, "OO", &obj1, &obj2)) return NULL;

    Py_buffer view1, view2;
    int has_view1 = (PyObject_GetBuffer(obj1, &view1, PyBUF_SIMPLE) == 0);
    int has_view2 = (PyObject_GetBuffer(obj2, &view2, PyBUF_SIMPLE) == 0);

    if (has_view1 && has_view2) {
        if (view1.len != view2.len) {
            PyBuffer_Release(&view1); PyBuffer_Release(&view2);
            PyErr_SetString(PyExc_ValueError, "Vectors must be of same length");
            return NULL;
        }
        double result = 0.0;
        double *d1 = (double*)view1.buf;
        double *d2 = (double*)view2.buf;
        int n = (int)(view1.len / sizeof(double));
        int i_dot = 0;

        Py_BEGIN_ALLOW_THREADS
        __m256d sum_v = _mm256_setzero_pd();
        for (i_dot = 0; i_dot <= n - 4; i_dot += 4) {
            __m256d v1 = _mm256_loadu_pd(&d1[i_dot]);
            __m256d v2 = _mm256_loadu_pd(&d2[i_dot]);
            sum_v = _mm256_fmadd_pd(v1, v2, sum_v);
        }
        
        double tmp[4];
        _mm256_storeu_pd(tmp, sum_v);
        result = tmp[0] + tmp[1] + tmp[2] + tmp[3];

        for (; i_dot < n; i_dot++) result += d1[i_dot] * d2[i_dot];
        Py_END_ALLOW_THREADS

        PyBuffer_Release(&view1); PyBuffer_Release(&view2);
        return PyFloat_FromDouble(result);
    }

    if (has_view1) PyBuffer_Release(&view1);
    if (has_view2) PyBuffer_Release(&view2);

    // Fallback to List and sequential calculation for lists
    if (!PyList_Check(obj1) || !PyList_Check(obj2)) {
        PyErr_SetString(PyExc_TypeError, "Expected lists or buffers");
        return NULL;
    }

    Py_ssize_t n = PyList_Size(obj1);
    if (n != PyList_Size(obj2)) {
        PyErr_SetString(PyExc_ValueError, "Vectors must be of same length");
        return NULL;
    }

    double result = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        double v1 = PyFloat_AsDouble(PyList_GetItem(obj1, i));
        double v2 = PyFloat_AsDouble(PyList_GetItem(obj2, i));
        result += v1 * v2;
    }

    return PyFloat_FromDouble(result);
}

static PyMethodDef CoreMethods[] = {
    {"dot", dot_product, METH_VARARGS, "Compute dot product of two vectors."},
    {"matrix_multiply", matrix_multiply, METH_VARARGS, "Compute matrix multiplication of two matrices."},
    {"matrix_lu", matrix_lu, METH_VARARGS, "Compute LU decomposition of a square matrix."},
    {"matrix_cholesky", matrix_cholesky, METH_VARARGS, "Compute Cholesky decomposition of a symmetric positive-definite matrix."},
    {"matrix_qr", matrix_qr, METH_VARARGS, "Compute QR decomposition of a matrix."},
    {"matrix_eigen", matrix_eigen, METH_VARARGS, "Compute eigenvalues and eigenvectors using QR algorithm."},
    {"matrix_svd", matrix_svd, METH_VARARGS, "Compute Singular Value Decomposition (SVD)."},
    {"batched_matrix_lu", batched_matrix_lu, METH_VARARGS, "Batched LU decomposition."},
    {"batched_matrix_qr", batched_matrix_qr, METH_VARARGS, "Batched QR decomposition."},
    {"batched_matrix_svd", batched_matrix_svd, METH_VARARGS, "Batched SVD decomposition."},
    {"covariance_matrix", covariance_matrix, METH_VARARGS, "Compute covariance matrix of column datasets."},
    {"incremental_mean_var", incremental_mean_var, METH_VARARGS, "Update mean and M2 using Welford's algorithm."},
    {"incremental_covariance", incremental_covariance, METH_VARARGS, "Update covariance incrementally for two variables."},
    {"simulate_gbm", simulate_gbm, METH_VARARGS, "Simulate GBM paths."},
    {"kalman_predict", kalman_predict, METH_VARARGS, "Kalman filter prediction step."},
    {"kalman_update", kalman_update, METH_VARARGS, "Kalman filter update step."},
    {"sparse_dense_matmul", sparse_dense_matmul, METH_VARARGS, "Sparse-Dense matrix multiplication (CSR)."},
    {"bfgs_minimize", bfgs_minimize, METH_VARARGS, "Minimize function using BFGS."},
    {"radix2_fft", radix2_fft, METH_VARARGS, "Compute Radix-2 FFT."},
    {"sobol_sequence", sobol_sequence, METH_VARARGS, "Generate 1D Sobol sequence."},
    {"halton_sequence", halton_sequence, METH_VARARGS, "Generate multidimensional Halton sequence."},
    {"scrambled_sobol", scrambled_sobol, METH_VARARGS, "Generate scrambled 1D Sobol sequence."},
    {"dual_op", dual_op, METH_VARARGS, "Compute dual number operation for autodiff."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef coremodule = {
    PyModuleDef_HEAD_INIT,
    "vectorquant_c_core",
    "VectorQuant C-Core numerical kernels",
    -1,
    CoreMethods
};

PyMODINIT_FUNC PyInit_vectorquant_c_core(void) {
    return PyModule_Create(&coremodule);
}
