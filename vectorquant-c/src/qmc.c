#include "core.h"
#include <stdlib.h>
#include <stdint.h>

// Van der Corput sequence (1D Sobol)
static double van_der_corput(uint64_t n, uint32_t base) {
    double q = 0, bk = 1.0 / base;
    while (n > 0) {
        q += (n % base) * bk;
        n /= base;
        bk /= base;
    }
    return q;
}

static const uint32_t primes[] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113
};

PyObject* halton_sequence(PyObject* self, PyObject* args) {
    int n, dim;
    if (!PyArg_ParseTuple(args, "ii", &n, &dim)) return NULL;
    if (dim > 30) return PyErr_Format(PyExc_ValueError, "Maximum dimension is 30");

    PyObject* result = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyObject* point = PyList_New(dim);
        for (int d = 0; d < dim; d++) {
            PyList_SetItem(point, d, PyFloat_FromDouble(van_der_corput(i + 1, primes[d])));
        }
        PyList_SetItem(result, i, point);
    }
    return result;
}

// Scrambled Sobol (1D for now, using Digital Shift)
PyObject* scrambled_sobol(PyObject* self, PyObject* args) {
    int n;
    uint32_t seed = 0;
    if (!PyArg_ParseTuple(args, "i|I", &n, &seed)) return NULL;

    PyObject* result = PyList_New(n);
    // Simple XOR scrambling (Digital Shift)
    // In a real Sobol implementation, we'd use Gray codes and direction numbers.
    // For this 'advanced' phase, we'll implement a robust 1D scrambled sequence.
    for (int i = 0; i < n; i++) {
        uint64_t val = 0;
        uint64_t k = i + 1;
        double vdc = van_der_corput(k, 2);
        
        // Convert to fixed point for bitwise scrambling
        uint32_t bits = (uint32_t)(vdc * 4294967296.0);
        bits ^= seed;
        double scrambled = (double)bits / 4294967296.0;
        
        PyList_SetItem(result, i, PyFloat_FromDouble(scrambled));
    }
    return result;
}
PyObject* sobol_sequence(PyObject* self, PyObject* args) {
    int n;
    if (!PyArg_ParseTuple(args, "i", &n)) return NULL;
    if (n < 0) return PyErr_Format(PyExc_ValueError, "n must be non-negative");

    PyObject* result = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyList_SetItem(result, i, PyFloat_FromDouble(van_der_corput(i + 1, 2)));
    }
    return result;
}
