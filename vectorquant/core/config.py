"""
VectorQuant Performance Configuration
=====================================

Provides optional JIT acceleration via Numba if available.
VectorQuant aims to be zero-dependency by default.
"""

import functools
import logging

logger = logging.getLogger(__name__)

NUMBA_AVAILABLE = False
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    pass

def njit_fallback(func):
    """
    Optional Numba JIT decorator.
    
    If Numba is installed (e.g., via `pip install vectorquant[perf]`),
    it compiles the function for 10x-50x speedups.
    Otherwise, it acts as a transparent pass-through decorator.
    """
    if NUMBA_AVAILABLE:
        # We use cache=True to avoid recompiling across runs
        # We use fastmath=True for maximum speed (relaxes strict IEEE 754 compliance)
        return numba.njit(cache=True, fastmath=True)(func)
    else:
        # Just return the original function
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper


# --- GPU CONFIGURATION (EXTRA) ---
# When users install vectorquant[gpu], cupy is available.
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    # Optional dependency not installed
    CUPY_AVAILABLE = False
    cp = None
    
def get_array_module(use_gpu=False):
    """
    Returns the appropriate array module (cupy if GPU is requested and available,
    otherwise raises an error or falls back to standard math if applicable).
    Because our core is list-of-lists pure Python, this is strictly used
    for the isolated GPU array operations in monte_carlo.
    """
    if use_gpu:
        if not CUPY_AVAILABLE:
            raise ImportError(
                "GPU acceleration requested but CuPy is not installed. "
                "Install via: pip install vectorquant[gpu]"
            )
        return cp
    # If GPU is not requested, or if CuPy is not available and we're not using GPU,
    # we should return a CPU-based array module. For now, we'll assume numpy
    # is the default if not using GPU. If numpy is not imported here,
    # it implies the core logic uses standard Python lists/math.
    # Given the context "Because our core is list-of-lists pure Python",
    # returning None or raising an error for non-GPU path might be appropriate
    # if there's no explicit CPU array module fallback.
    # For now, let's return None, implying pure Python or a different module
    # is used when GPU is not active.
    return None
