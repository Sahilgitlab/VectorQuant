#include "core.h"
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Fast PRNG: Xoroshiro128+
static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t next_rng(uint64_t s[2]) {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
    s[1] = rotl(s1, 37); // c

    return result;
}

// Box-Muller transform for standard normal variables
static inline double rand_normal(uint64_t s[2]) {
    double u1, u2;
    do {
        u1 = (double)next_rng(s) / (double)UINT64_MAX;
    } while (u1 <= 1e-15); // avoid log(0)
    u2 = (double)next_rng(s) / (double)UINT64_MAX;
    
    // Pi = 3.14159265358979323846
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
    return z;
}


// Simulate Geometric Brownian Motion (GBM) paths
// Returns a 2D matrix of shape (n_paths, n_steps+1)
PyObject* simulate_gbm(PyObject* self, PyObject* args) {
    double s0, mu, sigma, t, dt;
    int n_paths, antithetic;
    
    if (!PyArg_ParseTuple(args, "dddddii", &s0, &mu, &sigma, &t, &dt, &n_paths, &antithetic)) {
        return NULL;
    }

    if (dt <= 0 || n_paths <= 0 || t <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid parameters for GBM");
        return NULL;
    }

    int n_steps = (int)(t / dt);
    if (n_steps == 0) n_steps = 1;

    // Output dimension is n_paths x (n_steps + 1)
    int actual_paths = antithetic ? n_paths * 2 : n_paths;
    int cols = n_steps + 1;
    
    double* paths = (double*)malloc(actual_paths * cols * sizeof(double));
    if (!paths) return PyErr_NoMemory();
    
    // Precompute drift and diffusion
    double drift = (mu - 0.5 * sigma * sigma) * dt;
    double diffusion = sigma * sqrt(dt);

    int i;
    #pragma omp parallel for
    for (i = 0; i < n_paths; i++) {
        uint64_t state[2];
        // simple seed initialization based on path index and thread id
        state[0] = 0x8A5CB74B92C81DF4ULL ^ ((uint64_t)i * 0x12345678ULL);
        state[1] = 0x5D2CA94B1E91CB43ULL ^ ((uint64_t)i * 0x87654321ULL);
        
        // Advance state a few times to discard initial correlation
        next_rng(state); next_rng(state);
        
        int row1 = i * cols;
        paths[row1] = s0;
        
        int row2 = -1;
        if (antithetic) {
            row2 = (i + n_paths) * cols;
            paths[row2] = s0;
        }

        double current_s1 = log(s0);
        double current_s2 = log(s0);

        for (int j = 1; j <= n_steps; j+=2) {
            // Box-Muller provides two normals at once
            double u1, u2;
            do { u1 = (double)next_rng(state) / (double)UINT64_MAX; } while (u1 <= 1e-15);
            u2 = (double)next_rng(state) / (double)UINT64_MAX;
            
            double mag = sqrt(-2.0 * log(u1));
            double z1 = mag * cos(2.0 * 3.14159265358979323846 * u2);
            double z2 = mag * sin(2.0 * 3.14159265358979323846 * u2);
            
            // Step 1
            double shock1 = diffusion * z1;
            current_s1 += drift + shock1;
            paths[row1 + j] = exp(current_s1);
            if (antithetic) {
                current_s2 += drift - shock1;
                paths[row2 + j] = exp(current_s2);
            }
            
            // Step 2 (if not over limits)
            if (j + 1 <= n_steps) {
                double shock2 = diffusion * z2;
                current_s1 += drift + shock2;
                paths[row1 + j + 1] = exp(current_s1);
                if (antithetic) {
                    current_s2 += drift - shock2;
                    paths[row2 + j + 1] = exp(current_s2);
                }
            }
        }
    }

    // Return a flat list of paths to minimize list-of-lists overhead
    PyObject* result = PyList_New(actual_paths * cols);
    for (int i = 0; i < actual_paths * cols; i++) {
        PyList_SetItem(result, i, PyFloat_FromDouble(paths[i]));
    }

    free(paths);
    return Py_BuildValue("(Oi)", result, cols);
}
