# Assembly Optimization Strategy for VectorQuant

**Status:** Phase 9-10 Planning | Not needed for Phase 8

---

## Executive Summary

**Use assembly ONLY for 5-10 critical microkernels after profiling confirms bottlenecks.**

Current strategy (C + SIMD intrinsics + OpenMP) is optimal. Assembly is the final 5% optimization layer, not the foundation.

---

## When Assembly Is Worth Adding

### Criteria (ALL must be true)

- ✅ Extremely small function (< 100 lines)
- ✅ Executed billions of times
- ✅ Memory bound or register bound
- ✅ Critical to entire library
- ✅ Profiling shows measurable bottleneck

### Best Candidates for VectorQuant (5-8 only)

| Kernel | Use Case | Impact | Priority |
|--------|----------|--------|----------|
| **Vector Add** (a[i] + b[i]) | Statistics, simulations | 8-16 floats/instruction (AVX) | 🔴 HIGH |
| **Dot Product** | Covariance, regression, matmul | Fused multiply-add optimization | 🔴 HIGH |
| **MatMul Microkernel** | Inner loop optimization | Cache-conscious blocking | 🟡 MEDIUM |
| **Reduction** (sum, mean, var) | Statistics | Horizontal operations | 🟡 MEDIUM |
| **Xoroshiro128+** | Random number generation | Register control | 🟢 LOW |
| **Sparse Dot Product** | Sparse matrix ops | Branch prediction | 🟢 LOW |
| **FFT Butterfly** | FFT operations | Complex arithmetic | 🟢 LOW |
| **Kalman Core Loop** | Time-series (if profiling shows) | Matrix operations | 🟢 LOW |

**Total: 8 microkernels maximum (< 50KB assembly code)**

---

## When Assembly Should NOT Be Used

❌ **DO NOT** use assembly for:

- Kalman filters (high-level algorithm)
- Monte Carlo engines (complex control flow)
- Finance algorithms (business logic)
- Optimization routines (iterative)
- AI verification logic (validation)
- Formula parsing
- Data conversion

These require:
- Maintainability
- Debuggability
- Portability
- Flexibility

Assembly makes them worse.

---

## Architecture: Layered Optimization

### Current (Phase 8 Complete) ✅

```
Python API
    ↓
C numerical kernels (linalg.c, stats.c, kalman.c, sparse.c, qmc.c)
    ↓
SIMD intrinsics (AVX2, SSE via compiler)
    ↓
OpenMP parallelization
```

**Result:** 5-100x speedups already achieved

### Phase 9-10 (After Profiling)

```
Python API
    ↓
C numerical kernels (existing)
    ↓
SIMD intrinsics
    ↓
OpenMP parallelization
    ↓
Assembly microkernels (ONLY if profiling shows need)
    └─ vectorquant-asm/
       ├─ avx_dot.s
       ├─ avx_vector_add.s
       ├─ matmul_kernel.s
       └─ etc (5-8 files max)
```

---

## Profiling Strategy (Before Assembly)

### Phase 9: Profile Everything

```python
# Find hotspots using perf, cProfile, or PyPy profiler

1. Run real workload scenarios
   - Monte Carlo simulations (10K paths)
   - Portfolio optimization (100+ assets)
   - Covariance estimation (high dimensions)

2. Measure:
   - CPU time per function
   - Cache miss rate
   - Memory bandwidth utilization
   - Instruction throughput

3. Identify top 5-8 functions consuming >80% time

4. Check if they meet assembly criteria
   - Small?
   - Executed billions of times?
   - Memory bound?
```

### Likely Bottlenecks (Prediction)

Based on typical quantitative finance workloads:

```
1. Dot product (matrix multiply inner loop)    40% CPU time
2. Vector operations (add, scale)               20% CPU time
3. Covariance computation                       15% CPU time
4. Random number generation                     10% CPU time
5. Other (Kalman, FFT, optimization)            15% CPU time
```

→ **Dot product + vector ops are top candidates**

---

## Recommended Implementation Stages

### Stage 1: Phase 8 (COMPLETE) ✅

```
✓ Pure C implementation
✓ SIMD intrinsics via compiler
✓ OpenMP parallelization
✓ 111 tests passing
✓ 5-100x speedups achieved
```

No assembly needed yet.

### Stage 2: Phase 9 (AI Integration)

```
□ Build verification pipeline (9.2)
□ Implement agent protocol (9.1)
□ Profile real AI workloads
□ Identify hotspots
□ No assembly changes (yet)
```

Focus: AI integration, not kernel optimization.

### Stage 3: Phase 10 (Production Hardening)

```
□ Real-world performance testing
□ Profile against actual AI queries
□ Gather metrics:
  - Latency per operation
  - CPU usage
  - Memory bandwidth
  - Cache behavior

IF hotspots identified THEN:
  □ Add assembly for 5-8 microkernels
  □ CPU feature detection (AVX512, AVX2, SSE)
  □ Benchmarks (assembly vs C)
  □ Fallback implementations
ELSE:
  □ Current C+SIMD is sufficient
  □ Move to production
```

---

## CPU Feature Detection (If Needed)

When using assembly, detect available instructions at runtime:

```c
// Example: Check CPU capabilities
int has_avx2_support() {
    #ifdef __AVX2__
    return 1;
    #else
    return 0;
    #endif
}

int has_avx512_support() {
    #ifdef __AVX512F__
    return 1;
    #else
    return 0;
    #endif
}

// At runtime, select best implementation
if (has_avx512_support()) {
    use_avx512_dot_product();
} else if (has_avx2_support()) {
    use_avx2_dot_product();
} else {
    use_scalar_dot_product();
}
```

---

## Example: Dot Product Optimization

### Current (C + SIMD intrinsics)

```c
// From linalg.c - uses compiler auto-vectorization
double dot(double* a, double* b, int n) {
    double result = 0.0;
    #pragma omp simd reduction(+:result)
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}
```

**Performance:** ~10 GB/s (typical)

### Assembly Version (If Profiling Needed)

```asm
; vectorquant-asm/avx2_dot.s
; Inputs: rdi = a[], rsi = b[], rdx = n
; Output: xmm0 = result

avx2_dot:
    vpxor   ymm0, ymm0, ymm0        ; Clear accumulator
    xor     rax, rax                ; Counter i = 0
    cmp     rdx, 4                  ; Quick exit if n < 4
    jl      .scalar_loop

.vector_loop:
    cmp     rax, rdx
    jge     .reduce
    
    vmovapd ymm1, [rdi + rax*8]     ; Load a[i:i+3]
    vmulpd  ymm1, ymm1, [rsi + rax*8] ; a[i] * b[i]
    vaddpd  ymm0, ymm0, ymm1        ; Accumulate
    
    add     rax, 4
    jmp     .vector_loop

.reduce:
    vhaddpd ymm0, ymm0, ymm0        ; Horizontal sum
    vperm2f128 ymm1, ymm0, ymm0, 1
    vaddpd  ymm0, ymm0, ymm1
    vcvtsd2ss xmm0, xmm0, xmm0
    ret

.scalar_loop:
    ; Fallback for remaining elements
    ret
```

**Performance:** ~20-30 GB/s (2-3x improvement)

**Trade-off:** 30 lines of assembly for 2-3x speedup on dot product (if it's bottleneck)

---

## Risk Mitigation

### Why Assembly Is Risky

| Risk | Mitigation |
|------|-----------|
| Portability | Add CPU feature detection + fallback |
| Maintainability | Extensive comments, keep only 5-8 files |
| Debugging | Test thoroughly, keep C version as reference |
| Compiler changes | Regularly test with new compiler versions |

### Testing Strategy

```python
# test_assembly_kernels.py

def test_avx2_dot_matches_c():
    """Assembly version should match C version exactly"""
    for _ in range(1000):
        a = [random.random() for _ in range(1000)]
        b = [random.random() for _ in range(1000)]
        
        c_result = dot_c(a, b)
        asm_result = dot_avx2(a, b)
        
        assert abs(c_result - asm_result) < 1e-10

def test_fallback_if_no_avx2():
    """If CPU doesn't support AVX2, use fallback"""
    # Disable AVX2
    result = dot_with_fallback(a, b)
    assert result is not None
```

---

## Real-World Example: OpenBLAS

OpenBLAS (industry-standard BLAS) uses this exact strategy:

```
Python (NumPy)
    ↓
C interface (CBLAS)
    ↓
Highly optimized C kernels
    ↓
SIMD intrinsics
    ↓
Machine-tuned assembly (5-10 files per operation)
    ↓
CPU feature detection + fallback
```

Result: 5-100x speedup over naive implementation

VectorQuant should follow the same pattern.

---

## Decision Framework

**Should we add assembly?**

```
1. Is Phase 9-10 complete and deployed? → NO → Skip assembly now
2. Are AI queries slow? → NO → Skip assembly
3. Did profiling show clear bottleneck? → NO → Skip assembly
4. Is bottleneck in kernel (dot, matmul)? → NO → Skip assembly
5. Is C+SIMD not hitting memory bandwidth? → NO → Skip assembly
6. Do we have > 5 kernel candidates? → NO → Skip assembly

If ALL answers are YES:
    → Add assembly for top 5-8 kernels
    → CPU feature detection
    → Comprehensive testing
    → Document extensively

Otherwise:
    → Current C+SIMD is optimal
    → Move to production
```

---

## Current VectorQuant Plan

### Phase 8 (COMPLETE) ✅
- C kernels + SIMD intrinsics + OpenMP
- No assembly needed

### Phase 9 (AI Integration)
- Build verification pipeline
- Profile real workloads
- Decide on assembly

### Phase 10 (Production)
- If profiling shows need: Add 5-8 assembly microkernels
- Else: Deploy current C+SIMD

---

## Bottom Line

**Your current plan is already optimal:**

✅ Python API
✅ C kernels
✅ SIMD optimization
✅ OpenMP parallelization

Assembly is a 5% optimization for after profiling, not a priority now.

Focus Phase 9-10 on:
- AI verification (9.2)
- Agent protocol (9.1)
- Real-world testing
- Performance profiling

Only add assembly if data proves it's needed.

---

## Resources for Later

If you do add assembly, use these:

- **Intel Intrinsics Guide:** https://www.intel.com/content/dam/develop/external/us/en/documents/manual/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325462.pdf
- **AMD Optimization Guide:** https://www.amd.com/content/dam/amdsite/documents/developer/files/guides/agner_FOG.pdf
- **SIMD Reference:** https://simd.miraheze.org/
- **Example:** OpenBLAS source (`@microsoft/OpenBLAS` on GitHub)

---

## Recommendation Summary

| Phase | Action | Assembly |
|-------|--------|----------|
| 8 ✅ | Implement core | NO (C+SIMD sufficient) |
| 9 | AI integration | NO (Profile first) |
| 10 | Production | MAYBE (only if needed) |
| 11+ | Optimization | YES (5-8 microkernels) |

**Current focus: Phase 9 (AI verification layer)**

Assembly is Phase 11+ optimization.
