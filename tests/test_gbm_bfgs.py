
import math
from vectorquant.core.backend import get_backend, set_backend

def test_gbm_bfgs():
    # Test GBM
    s0, mu, sigma, t, dt, n_paths = 100.0, 0.05, 0.2, 1.0, 0.1, 5
    
    print("Testing GBM simulation...")
    for b_name in ["python", "c"]:
        set_backend(b_name)
        backend = get_backend()
        paths = backend.simulate_gbm(s0, mu, sigma, t, dt, n_paths, antithetic=True)
        print(f"{b_name} backend: generated {len(paths)} paths.")
        assert len(paths) == n_paths * 2
        assert len(paths[0]) == int(t/dt) + 1

    # Test BFGS minimize
    # Minimize f(x) = (x-3)^2 + (y+2)^2
    def f(v):
        return (v[0]-3)**2 + (v[1]+2)**2
    def grad_f(v):
        return [2*(v[0]-3), 2*(v[1]+2)]
        
    x0 = [0.0, 0.0]
    print("\nTesting BFGS minimization...")
    for b_name in ["python", "c"]:
        set_backend(b_name)
        backend = get_backend()
        res = backend.bfgs_minimize(f, grad_f, x0)
        print(f"{b_name} backend result: {res}")
        assert math.isclose(res[0], 3.0, abs_tol=1e-2)
        assert math.isclose(res[1], -2.0, abs_tol=1e-2)

    print("\nGBM and BFGS tests passed!")

if __name__ == "__main__":
    test_gbm_bfgs()
