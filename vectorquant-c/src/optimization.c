#include "core.h"
#include <stdlib.h>
#include <math.h>

// Simple backtracking line search
static double backtracking_line_search(PyObject* f, PyObject* x_py, double* x, double* d, int n, double f_val, double* grad) {
    double alpha = 1.0;
    double rho = 0.5;
    double c = 1e-4;
    double dot_grad_d = 0.0;
    for (int i = 0; i < n; i++) dot_grad_d += grad[i] * d[i];

    for (int iter = 0; iter < 20; iter++) {
        PyObject* x_new_py = PyList_New(n);
        for (int i = 0; i < n; i++) {
            PyList_SetItem(x_new_py, i, PyFloat_FromDouble(x[i] + alpha * d[i]));
        }

        PyObject* args = PyTuple_Pack(1, x_new_py);
        PyObject* result = PyObject_CallObject(f, args);
        double f_new = PyFloat_AsDouble(result);
        
        Py_DECREF(args);
        Py_DECREF(result);
        Py_DECREF(x_new_py);

        if (f_new <= f_val + c * alpha * dot_grad_d) {
            return alpha;
        }
        alpha *= rho;
    }
    return alpha;
}

PyObject* bfgs_minimize(PyObject* self, PyObject* args) {
    PyObject *f, *grad_f, *x0_py;
    double tol = 1e-6;
    int max_iter = 100;

    if (!PyArg_ParseTuple(args, "OOO|di", &f, &grad_f, &x0_py, &tol, &max_iter)) {
        return NULL;
    }

    int n = (int)PyList_Size(x0_py);
    double* x = (double*)malloc(n * sizeof(double));
    double* grad = (double*)malloc(n * sizeof(double));
    double* H = (double*)calloc(n * n, sizeof(double)); // Inverse Hessian

    if (!x || !grad || !H) {
        free(x); free(grad); free(H);
        return PyErr_NoMemory();
    }

    // Initialize x and H = Identity
    for (int i = 0; i < n; i++) {
        x[i] = PyFloat_AsDouble(PyList_GetItem(x0_py, i));
        H[i * n + i] = 1.0;
    }

    double* s = (double*)malloc(n * sizeof(double));
    double* y = (double*)malloc(n * sizeof(double));
    double* Hy = (double*)malloc(n * sizeof(double));
    double* d = (double*)malloc(n * sizeof(double));

    for (int iter = 0; iter < max_iter; iter++) {
        // Compute gradient
        PyObject* x_py = PyList_New(n);
        for (int i = 0; i < n; i++) PyList_SetItem(x_py, i, PyFloat_FromDouble(x[i]));
        
        PyObject* g_args = PyTuple_Pack(1, x_py);
        PyObject* g_res = PyObject_CallObject(grad_f, g_args);
        
        double grad_norm = 0.0;
        for (int i = 0; i < n; i++) {
            grad[i] = PyFloat_AsDouble(PyList_GetItem(g_res, i));
            grad_norm += grad[i] * grad[i];
        }
        grad_norm = sqrt(grad_norm);

        if (grad_norm < tol) {
            Py_DECREF(x_py); Py_DECREF(g_args); Py_DECREF(g_res);
            break;
        }

        // Search direction d = -H * grad
        for (int i = 0; i < n; i++) {
            d[i] = 0.0;
            for (int j = 0; j < n; j++) {
                d[i] -= H[i * n + j] * grad[j];
            }
        }

        // Line search
        PyObject* f_args = PyTuple_Pack(1, x_py);
        PyObject* f_res = PyObject_CallObject(f, f_args);
        double f_val = PyFloat_AsDouble(f_res);
        
        double alpha = backtracking_line_search(f, x_py, x, d, n, f_val, grad);

        // Update x, s = x_new - x
        for (int i = 0; i < n; i++) {
            double x_old = x[i];
            x[i] += alpha * d[i];
            s[i] = x[i] - x_old;
        }

        // Compute new gradient and y = grad_new - grad
        PyObject* x_new_py = PyList_New(n);
        for (int i = 0; i < n; i++) PyList_SetItem(x_new_py, i, PyFloat_FromDouble(x[i]));
        PyObject* gn_args = PyTuple_Pack(1, x_new_py);
        PyObject* gn_res = PyObject_CallObject(grad_f, gn_args);
        
        double ys = 0.0;
        for (int i = 0; i < n; i++) {
            double grad_new = PyFloat_AsDouble(PyList_GetItem(gn_res, i));
            y[i] = grad_new - grad[i];
            ys += y[i] * s[i];
        }

        if (ys > 1e-10) {
            // BFGS Update
            // H = (I - rho*s*y^T) H (I - rho*y*s^T) + rho*s*s^T
            double rho = 1.0 / ys;
            
            // Hy = H * y
            for (int i = 0; i < n; i++) {
                Hy[i] = 0.0;
                for (int j = 0; j < n; j++) Hy[i] += H[i * n + j] * y[j];
            }
            
            double yHy = 0.0;
            for (int i = 0; i < n; i++) yHy += y[i] * Hy[i];
            
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    H[i * n + j] += (1.0 + rho * yHy) * rho * s[i] * s[j] -
                                     rho * (s[i] * Hy[j] + Hy[i] * s[j]);
                }
            }
        }

        Py_DECREF(x_py); Py_DECREF(g_args); Py_DECREF(g_res);
        Py_DECREF(f_args); Py_DECREF(f_res);
        Py_DECREF(x_new_py); Py_DECREF(gn_args); Py_DECREF(gn_res);
    }

    PyObject* result = PyList_New(n);
    for (int i = 0; i < n; i++) PyList_SetItem(result, i, PyFloat_FromDouble(x[i]));

    free(x); free(grad); free(H);
    free(s); free(y); free(Hy); free(d);

    return result;
}
