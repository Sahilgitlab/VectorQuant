#include "core.h"
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Covariance matrix from list of column lists
PyObject* covariance_matrix(PyObject* self, PyObject* args) {
    PyObject* data_matrix_cols;
    if (!PyArg_ParseTuple(args, "O", &data_matrix_cols)) return NULL;

    Py_ssize_t n_cols = PyList_Size(data_matrix_cols);
    if (n_cols == 0) return PyList_New(0);

    PyObject* first_col = PyList_GetItem(data_matrix_cols, 0);
    Py_ssize_t n_rows = PyList_Size(first_col);

    if (n_rows < 2) {
        PyErr_SetString(PyExc_ValueError, "At least 2 observations required for covariance");
        return NULL;
    }

    // Allocate flat buffer
    double* data = (double*)malloc(n_cols * n_rows * sizeof(double));
    double* means = (double*)malloc(n_cols * sizeof(double));
    double* cov = (double*)calloc(n_cols * n_cols, sizeof(double));

    if (!data || !means || !cov) {
        free(data); free(means); free(cov);
        return PyErr_NoMemory();
    }

    // Unpack data and compute means
    for (Py_ssize_t j = 0; j < n_cols; j++) {
        PyObject* col = PyList_GetItem(data_matrix_cols, j);
        double sum = 0.0;
        for (Py_ssize_t i = 0; i < n_rows; i++) {
            double val = PyFloat_AsDouble(PyList_GetItem(col, i));
            data[j * n_rows + i] = val;
            sum += val;
        }
        means[j] = sum / (double)n_rows;
    }

    // Compute covariance matrix with OpenMP
    double den = (double)(n_rows - 1);
    int i, j, k;
    #pragma omp parallel for private(j, k)
    for (i = 0; i < (int)n_cols; i++) {
        for (j = i; j < (int)n_cols; j++) {
            double sum = 0.0;
            for (k = 0; k < (int)n_rows; k++) {
                double diff_i = data[i * n_rows + k] - means[i];
                double diff_j = data[j * n_rows + k] - means[j];
                sum += diff_i * diff_j;
            }
            double val = sum / den;
            cov[i * n_cols + j] = val;
            if (i != j) {
                cov[j * n_cols + i] = val; // Symmetric
            }
        }
    }

    // Pack result
    PyObject* result = PyList_New(n_cols);
    for (Py_ssize_t i = 0; i < n_cols; i++) {
        PyObject* row = PyList_New(n_cols);
        for (Py_ssize_t j = 0; j < n_cols; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(cov[i * n_cols + j]));
        }
        PyList_SetItem(result, i, row);
    }

    free(data);
    free(means);
    free(cov);

    return result;
}

// Incremental mean and variance update (Welford's algorithm)
PyObject* incremental_mean_var(PyObject* self, PyObject* args) {
    double n, mean, M2, x;
    if (!PyArg_ParseTuple(args, "dddd", &n, &mean, &M2, &x)) return NULL;

    n += 1.0;
    double delta = x - mean;
    mean += delta / n;
    double delta2 = x - mean;
    M2 += delta * delta2;

    return Py_BuildValue("(ddd)", n, mean, M2);
}

// Incremental covariance update
PyObject* incremental_covariance(PyObject* self, PyObject* args) {
    double n, mean_x, mean_y, C_xy, x, y;
    if (!PyArg_ParseTuple(args, "dddddd", &n, &mean_x, &mean_y, &C_xy, &x, &y)) return NULL;

    n += 1.0;
    double delta_x = x - mean_x;
    mean_x += delta_x / n;
    
    // For covariance, we need the new mean of y but the old mean of x (or vice-versa)
    // Co-moments update: C_xy = C_xy + (x_new - mean_x_old) * (y_new - mean_y_new)
    double delta_y = y - mean_y;
    mean_y += delta_y / n;
    
    C_xy += delta_x * (y - mean_y);

    return Py_BuildValue("(dddd)", n, mean_x, mean_y, C_xy);
}
