#include "core.h"
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    double real;
    double imag;
} vq_complex;

// Recursive radix-2 Cooley-Tukey FFT implementation (iterative version)
PyObject* radix2_fft(PyObject* self, PyObject* args) {
    PyObject* input_list;
    if (!PyArg_ParseTuple(args, "O", &input_list)) return NULL;

    Py_ssize_t n = PyList_Size(input_list);
    
    // Check if n is power of 2
    if ((n & (n - 1)) != 0 || n == 0) {
        PyErr_SetString(PyExc_ValueError, "Input size must be a power of 2");
        return NULL;
    }

    vq_complex* data = (vq_complex*)malloc(n * sizeof(vq_complex));
    if (!data) return PyErr_NoMemory();

    // Unpack input
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PyList_GetItem(input_list, i);
        if (PyComplex_Check(item)) {
            data[i].real = PyComplex_RealAsDouble(item);
            data[i].imag = PyComplex_ImagAsDouble(item);
        } else {
            data[i].real = PyFloat_AsDouble(item);
            data[i].imag = 0.0;
        }
    }

    // Bit-reversal permutation
    for (Py_ssize_t i = 1, j = 0; i < n; i++) {
        Py_ssize_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            vq_complex temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }

    // Butterfly operations
    for (Py_ssize_t len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / len;
        vq_complex wlen = {cos(ang), -sin(ang)};
        for (Py_ssize_t i = 0; i < n; i += len) {
            vq_complex w = {1.0, 0.0};
            for (Py_ssize_t j = 0; j < len / 2; j++) {
                vq_complex u = data[i + j];
                // v = data[i + j + len / 2] * w
                vq_complex v_orig = data[i + j + len / 2];
                vq_complex v = {
                    v_orig.real * w.real - v_orig.imag * w.imag,
                    v_orig.real * w.imag + v_orig.imag * w.real
                };
                
                data[i + j].real = u.real + v.real;
                data[i + j].imag = u.imag + v.imag;
                data[i + j + len / 2].real = u.real - v.real;
                data[i + j + len / 2].imag = u.imag - v.imag;
                
                // w *= wlen
                double next_w_real = w.real * wlen.real - w.imag * wlen.imag;
                w.imag = w.real * wlen.imag + w.imag * wlen.real;
                w.real = next_w_real;
            }
        }
    }

    // Pack output
    PyObject* result = PyList_New(n);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyList_SetItem(result, i, PyComplex_FromDoubles(data[i].real, data[i].imag));
    }

    free(data);
    return result;
}
