# CLAUDE.md — Implementation Rules for AI Coding Agents

This document defines implementation rules for AI coding agents contributing to VectorQuant.

**Agents modifying this repository must follow the rules below.**

---

## System Architecture

VectorQuant consists of three layers.

### Python API Layer

```
vectorquant/
├── finance/
├── stochastic/
├── research/
└── ai/
```

### Dispatch Layer

```
vectorquant/core/backend.py
vectorquant/core/config.py
```

### Native Performance Engine

```
vectorquant-c/src/
```

**Key principle:** The Python layer should not perform heavy numerical computation.

All performance critical operations must run inside the C engine.

---

## Implementation Rules

### Rule 1: Zero Dependency Policy

Do not introduce numerical dependencies.

**Forbidden libraries:**
- numpy
- scipy
- torch
- jax
- tensorflow

All numerical algorithms must exist inside the VectorQuant engine.

### Rule 2: Kernel Reuse

Before implementing new algorithms:

1. Inspect existing kernels inside `vectorquant-c/src`
2. Extend internal kernels when possible

**High priority kernels:**
- Matrix multiplication
- Covariance
- Random number generation
- Linear system solvers

Avoid algorithm duplication.

### Rule 3: Backend Dispatch

Backend detection occurs only at import time.

**Correct pattern:**

```python
active_backend = CBackend()
```

Avoid runtime branching inside tight loops.

### Rule 4: Memory Layout

All numerical kernels must assume:

- Contiguous memory
- Predictable strides
- SIMD friendly alignment

Avoid Python object containers.
Use flat numerical buffers.

### Rule 5: Parallelization Strategy

Parallelization must occur at the outer computation level.

**Examples:**

- **Monte Carlo simulations** → parallelize across simulation paths
- **Covariance** → parallelize across columns
- **Matrix multiplication** → parallelize across rows

Avoid nested parallel loops.

### Rule 6: Deterministic Random Numbers

All stochastic models must use the internal random number generator.

**Default generator:**
- Xoroshiro128+

Random seeds must be configurable.

### Rule 7: SIMD Optimization

Critical kernels should support SIMD vectorization.

**Preferred implementation:**
- AVX2 intrinsics
- SSE intrinsics

Assembly may be used only for extremely small critical loops.

---

## AI Verification Layer

VectorQuant supports verification of AI generated numerical results.

**Example flow:**

```
LLM reasoning
    ↓
extract numeric expression
    ↓
VectorQuant recomputation
    ↓
verification result
```

**Primary module:**
`vectorquant/ai/verifier.py`
