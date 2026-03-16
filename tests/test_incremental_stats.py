
import random
import statistics
from vectorquant.core.backend import get_backend, set_backend

def test_incremental_stats():
    data = [random.uniform(-10, 10) for _ in range(1000)]
    
    # Batch reference
    mean_ref = statistics.mean(data)
    var_ref = statistics.variance(data)
    
    # Incremental
    backend = get_backend()
    n, mean, m2 = 0, 0.0, 0.0
    for x in data:
        n, mean, m2 = backend.incremental_mean_var(n, mean, m2, x)
    
    var_inc = m2 / (n - 1)
    
    print(f"Mean Reference: {mean_ref:.8f}, Incremental: {mean:.8f}")
    print(f"Var Reference: {var_ref:.8f}, Incremental: {var_inc:.8f}")
    
    assert abs(mean - mean_ref) < 1e-10
    assert abs(var_inc - var_ref) < 1e-10
    print("Incremental Mean/Var test passed!")

def test_incremental_covariance():
    x_data = [random.uniform(-10, 10) for _ in range(1000)]
    y_data = [random.uniform(-10, 10) for _ in range(1000)]
    
    # Batch reference
    mean_x_ref = statistics.mean(x_data)
    mean_y_ref = statistics.mean(y_data)
    cov_ref = sum((x - mean_x_ref) * (y - mean_y_ref) for x, y in zip(x_data, y_data)) / (len(x_data) - 1)
    
    # Incremental
    backend = get_backend()
    n, mx, my, cxy = 0, 0.0, 0.0, 0.0
    for x, y in zip(x_data, y_data):
        n, mx, my, cxy = backend.incremental_covariance(n, mx, my, cxy, x, y)
    
    cov_inc = cxy / (n - 1)
    
    print(f"Cov Reference: {cov_ref:.8f}, Incremental: {cov_inc:.8f}")
    assert abs(cov_inc - cov_ref) < 1e-10
    print("Incremental Covariance test passed!")

if __name__ == "__main__":
    print("Testing C Backend:")
    set_backend("c")
    test_incremental_stats()
    test_incremental_covariance()
    
    print("\nTesting Python Backend:")
    set_backend("python")
    test_incremental_stats()
    test_incremental_covariance()
