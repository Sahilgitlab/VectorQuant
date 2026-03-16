#include "core.h"
#include <stdlib.h>
#include <math.h>

typedef struct {
    double val;
    double der;
} dual;

PyObject* dual_op(PyObject* self, PyObject* args) {
    double v1, d1, v2, d2;
    const char* op;
    if (!PyArg_ParseTuple(args, "dddds", &v1, &d1, &v2, &d2, &op)) return NULL;

    dual res = {0, 0};
    if (strcmp(op, "add") == 0) {
        res.val = v1 + v2;
        res.der = d1 + d2;
    } else if (strcmp(op, "mul") == 0) {
        res.val = v1 * v2;
        res.der = v1 * d2 + v2 * d1;
    } else if (strcmp(op, "div") == 0) {
        if (v2 == 0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "Division by zero in dual op");
            return NULL;
        }
        res.val = v1 / v2;
        res.der = (d1 * v2 - v1 * d2) / (v2 * v2);
    } else if (strcmp(op, "sin") == 0) {
        res.val = sin(v1);
        res.der = cos(v1) * d1;
    } else if (strcmp(op, "cos") == 0) {
        res.val = cos(v1);
        res.der = -sin(v1) * d1;
    } else if (strcmp(op, "exp") == 0) {
        res.val = exp(v1);
        res.der = res.val * d1;
    } else {
        PyErr_SetString(PyExc_ValueError, "Unsupported dual operation");
        return NULL;
    }

    return Py_BuildValue("dd", res.val, res.der);
}
