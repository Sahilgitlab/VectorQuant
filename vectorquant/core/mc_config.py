"""
Monte Carlo Configuration Module

Centralizes all Monte Carlo simulation parameters to prevent hanging tests
and ensure consistent behavior across the library.
"""

# ─── Safe Test Defaults ───────────────────────────────────────────────────────
# Used in unit tests to ensure fast, reliable test execution
SAFE_TEST_N_PATHS = 1000          # Down from 50k for test speed
SAFE_TEST_N_STEPS = 50            # Down from 252 for test speed
SAFE_TEST_DT = 0.02               # 1.0 / 50

# ─── Performance Test Defaults ───────────────────────────────────────────────
# Used only in @pytest.mark.slow benchmarking tests
PERFORMANCE_TEST_N_PATHS = 50000  # Large path count for realistic benchmarks
PERFORMANCE_TEST_N_STEPS = 252    # Full year with daily steps
PERFORMANCE_TEST_DT = 0.00397      # 1.0 / 252

# ─── Production Defaults ─────────────────────────────────────────────────────
# Recommended for production use
PRODUCTION_N_PATHS = 10000
PRODUCTION_N_STEPS = 252
PRODUCTION_DT = 0.00397

# ─── Thresholds for Automatic Backend Selection ──────────────────────────────
# When n_paths exceeds this, automatically use C backend
C_BACKEND_THRESHOLD_PATHS = 5000

# When n_steps exceeds this, use C backend
C_BACKEND_THRESHOLD_STEPS = 100

def get_safe_test_params(n_paths_override=None, n_steps_override=None):
    """
    Get safe test parameters for unit tests.
    
    Args:
        n_paths_override: Override default safe n_paths
        n_steps_override: Override default safe n_steps
        
    Returns:
        dict: {n_paths, n_steps, dt}
    """
    n_paths = n_paths_override or SAFE_TEST_N_PATHS
    n_steps = n_steps_override or SAFE_TEST_N_STEPS
    return {
        'n_paths': n_paths,
        'n_steps': n_steps,
        'dt': 1.0 / n_steps
    }

def get_performance_test_params(n_paths_override=None, n_steps_override=None):
    """
    Get performance test parameters for benchmarking.
    
    Args:
        n_paths_override: Override default performance n_paths
        n_steps_override: Override default performance n_steps
        
    Returns:
        dict: {n_paths, n_steps, dt}
    """
    n_paths = n_paths_override or PERFORMANCE_TEST_N_PATHS
    n_steps = n_steps_override or PERFORMANCE_TEST_N_STEPS
    return {
        'n_paths': n_paths,
        'n_steps': n_steps,
        'dt': 1.0 / n_steps
    }

def should_use_c_backend(n_paths, n_steps):
    """
    Determine if C backend should be used based on problem size.
    
    Args:
        n_paths: Number of simulation paths
        n_steps: Number of time steps
        
    Returns:
        bool: True if should use C backend
    """
    return (n_paths >= C_BACKEND_THRESHOLD_PATHS or 
            n_steps >= C_BACKEND_THRESHOLD_STEPS)
