#include "core.h"
#include <stdlib.h>
#include <math.h>

/*
 * Kalman Filter Implementation
 * x = state vector
 * P = covariance matrix
 * F = state transition matrix
 * H = observation matrix
 * Q = process noise covariance
 * R = observation noise covariance
 * z = observation vector
 */

// Helper: Matrix-Vector multiplication (y = M * x)
static void mat_vec_mul(double* M, double* x, double* y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        y[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            y[i] += M[i * cols + j] * x[j];
        }
    }
}

// Helper: Matrix-Matrix multiplication (C = A * B)
static void mat_mat_mul(double* A, double* B, double* C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

// Helper: Matrix transpose
static void mat_transpose(double* A, double* AT, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            AT[j * rows + i] = A[i * cols + j];
        }
    }
}

// Kalman Predict: x = Fx, P = FPF' + Q
PyObject* kalman_predict(PyObject* self, PyObject* args) {
    PyObject *x_obj, *P_obj, *F_obj, *Q_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &x_obj, &P_obj, &F_obj, &Q_obj)) return NULL;

    int n = (int)PyList_Size(x_obj);
    double* x = (double*)malloc(n * sizeof(double));
    double* P = (double*)malloc(n * n * sizeof(double));
    double* F = (double*)malloc(n * n * sizeof(double));
    double* Q = (double*)malloc(n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        x[i] = PyFloat_AsDouble(PyList_GetItem(x_obj, i));
        PyObject* P_row = PyList_GetItem(P_obj, i);
        PyObject* F_row = PyList_GetItem(F_obj, i);
        PyObject* Q_row = PyList_GetItem(Q_obj, i);
        for (int j = 0; j < n; j++) {
            P[i * n + j] = PyFloat_AsDouble(PyList_GetItem(P_row, j));
            F[i * n + j] = PyFloat_AsDouble(PyList_GetItem(F_row, j));
            Q[i * n + j] = PyFloat_AsDouble(PyList_GetItem(Q_row, j));
        }
    }

    double* x_new = (double*)malloc(n * sizeof(double));
    double* P_temp = (double*)malloc(n * n * sizeof(double));
    double* P_new = (double*)malloc(n * n * sizeof(double));
    double* FT = (double*)malloc(n * n * sizeof(double));

    // x = F * x
    mat_vec_mul(F, x, x_new, n, n);

    // P = F * P * F' + Q
    mat_transpose(F, FT, n, n);
    mat_mat_mul(F, P, P_temp, n, n, n);
    mat_mat_mul(P_temp, FT, P_new, n, n, n);

    for (int i = 0; i < n * n; i++) P_new[i] += Q[i];

    // Build return values
    PyObject* py_x = PyList_New(n);
    PyObject* py_P = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyList_SetItem(py_x, i, PyFloat_FromDouble(x_new[i]));
        PyObject* row = PyList_New(n);
        for (int j = 0; j < n; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(P_new[i * n + j]));
        }
        PyList_SetItem(py_P, i, row);
    }

    free(x); free(P); free(F); free(Q);
    free(x_new); free(P_temp); free(P_new); free(FT);

    return Py_BuildValue("(OO)", py_x, py_P);
}

// Kalman Update Step
PyObject* kalman_update(PyObject* self, PyObject* args) {
    PyObject *x_obj, *P_obj, *H_obj, *R_obj, *z_obj;
    if (!PyArg_ParseTuple(args, "OOOOO", &x_obj, &P_obj, &H_obj, &R_obj, &z_obj)) return NULL;

    int n = (int)PyList_Size(x_obj);
    int m = (int)PyList_Size(z_obj);

    double* x = (double*)malloc(n * sizeof(double));
    double* P = (double*)malloc(n * n * sizeof(double));
    double* H = (double*)malloc(m * n * sizeof(double));
    double* R = (double*)malloc(m * m * sizeof(double));
    double* z = (double*)malloc(m * sizeof(double));

    for (int i = 0; i < n; i++) {
        x[i] = PyFloat_AsDouble(PyList_GetItem(x_obj, i));
        PyObject* P_row = PyList_GetItem(P_obj, i);
        for (int j = 0; j < n; j++) P[i * n + j] = PyFloat_AsDouble(PyList_GetItem(P_row, j));
    }
    for (int i = 0; i < m; i++) {
        z[i] = PyFloat_AsDouble(PyList_GetItem(z_obj, i));
        PyObject* H_row = PyList_GetItem(H_obj, i);
        PyObject* R_row = PyList_GetItem(R_obj, i);
        for (int j = 0; j < n; j++) H[i * n + j] = PyFloat_AsDouble(PyList_GetItem(H_row, j));
        for (int j = 0; j < m; j++) R[i * m + j] = PyFloat_AsDouble(PyList_GetItem(R_row, j));
    }

    // y = z - Hx
    double* Hx = (double*)malloc(m * sizeof(double));
    mat_vec_mul(H, x, Hx, m, n);
    double* y = (double*)malloc(m * sizeof(double));
    for (int i = 0; i < m; i++) y[i] = z[i] - Hx[i];

    // S = HPH' + R
    double* HT = (double*)malloc(n * m * sizeof(double));
    mat_transpose(H, HT, m, n);
    double* PH_T = (double*)malloc(n * m * sizeof(double));
    mat_mat_mul(P, HT, PH_T, n, n, m);
    double* S = (double*)malloc(m * m * sizeof(double));
    mat_mat_mul(H, PH_T, S, m, n, m);
    for (int i = 0; i < m * m; i++) S[i] += R[i];

    // K = PH' S^-1
    double* S_inv = (double*)malloc(m * m * sizeof(double));
    if (internal_invert_lu(S, S_inv, m) != 0) {
        free(x); free(P); free(H); free(R); free(z); free(Hx); free(y); free(HT); free(PH_T); free(S); free(S_inv);
        PyErr_SetString(PyExc_ValueError, "Innovation covariance S is singular");
        return NULL;
    }
    double* K = (double*)malloc(n * m * sizeof(double));
    mat_mat_mul(PH_T, S_inv, K, n, m, m);

    // x = x + Ky
    double* Ky = (double*)malloc(n * sizeof(double));
    mat_vec_mul(K, y, Ky, n, m);
    double* x_new = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) x_new[i] = x[i] + Ky[i];

    // P = (I - KH)P
    double* KH = (double*)malloc(n * n * sizeof(double));
    mat_mat_mul(K, H, KH, n, m, n);
    double* I_KH = (double*)calloc(n * n, sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            I_KH[i * n + j] = (i == j ? 1.0 : 0.0) - KH[i * n + j];
        }
    }
    double* P_new = (double*)malloc(n * n * sizeof(double));
    mat_mat_mul(I_KH, P, P_new, n, n, n);

    // Build return values
    PyObject* py_x = PyList_New(n);
    PyObject* py_P = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyList_SetItem(py_x, i, PyFloat_FromDouble(x_new[i]));
        PyObject* row = PyList_New(n);
        for (int j = 0; j < n; j++) PyList_SetItem(row, j, PyFloat_FromDouble(P_new[i * n + j]));
        PyList_SetItem(py_P, i, row);
    }

    free(x); free(P); free(H); free(R); free(z); free(Hx); free(y); free(HT); free(PH_T); free(S); free(S_inv); free(K); free(Ky); free(x_new); free(KH); free(I_KH); free(P_new);

    return Py_BuildValue("(OO)", py_x, py_P);
}
