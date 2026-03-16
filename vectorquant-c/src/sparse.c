
#include "core.h"
#include <stdlib.h>

/*
 * Compressed Sparse Row (CSR) implementation for VectorQuant
 * - data: values of non-zero elements
 * - indices: column indices of non-zero elements
 * - indptr: row pointers (where each row starts in data/indices)
 */

typedef struct {
    double* data;
    int* indices;
    int* indptr;
    int nnz;
    int rows;
    int cols;
} CSRMatrix;

// Sparse-Dense Matrix Multiplication: C = A * B
// A is sparse (rows x k), B is dense (k x cols), C is dense (rows x cols)
PyObject* sparse_dense_matmul(PyObject* self, PyObject* args) {
    PyObject *data_obj, *indices_obj, *indptr_obj, *B_obj;
    int rows, cols, k;
    
    if (!PyArg_ParseTuple(args, "OOOiiiO", &data_obj, &indices_obj, &indptr_obj, 
                         &rows, &cols, &k, &B_obj)) return NULL;

    int nnz = (int)PyList_Size(data_obj);
    double* data = (double*)malloc(nnz * sizeof(double));
    int* indices = (int*)malloc(nnz * sizeof(int));
    int* indptr = (int*)malloc((rows + 1) * sizeof(int));

    for (int i = 0; i < nnz; i++) {
        data[i] = PyFloat_AsDouble(PyList_GetItem(data_obj, i));
        indices[i] = (int)PyLong_AsLong(PyList_GetItem(indices_obj, i));
    }
    for (int i = 0; i <= rows; i++) {
        indptr[i] = (int)PyLong_AsLong(PyList_GetItem(indptr_obj, i));
    }

    // B is dense k x cols. Let's parse it.
    int b_rows, b_cols;
    // Assuming B is passed as a flat list or nested list? 
    // Let's use internal_parse_dense for consistency if we had one, 
    // but we'll manually unpack here for B.
    double* B = (double*)malloc(k * cols * sizeof(double));
    for (int i = 0; i < k; i++) {
        PyObject* row = PyList_GetItem(B_obj, i);
        for (int j = 0; j < cols; j++) {
            B[i * cols + j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    double* C = (double*)calloc(rows * cols, sizeof(double));

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < rows; i++) {
        for (int p = indptr[i]; p < indptr[i+1]; p++) {
            int col_index = indices[p];
            double val = data[p];
            for (int j = 0; j < cols; j++) {
                C[i * cols + j] += val * B[col_index * cols + j];
            }
        }
    }

    // Pack C
    PyObject* res = PyList_New(rows);
    for (int i = 0; i < rows; i++) {
        PyObject* row = PyList_New(cols);
        for (int j = 0; j < cols; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(C[i * cols + j]));
        }
        PyList_SetItem(res, i, row);
    }

    free(data); free(indices); free(indptr); free(B); free(C);
    return res;
}
