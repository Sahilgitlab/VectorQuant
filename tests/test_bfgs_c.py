import vectorquant.core.backend as backend
import math

# Simple Quadratic Function: f(x, y) = (x-2)^2 + (y-3)^2
def f(x):
    return (x[0] - 2.0)**2 + (x[1] - 3.0)**2

def grad_f(x):
    return [2.0 * (x[0] - 2.0), 2.0 * (x[1] - 3.0)]

def test_bfgs():
    b = backend.get_backend()
    print(f"Using backend: {type(b).__name__}")
    
    x0 = [0.0, 0.0]
    result = b.bfgs_minimize(f, grad_f, x0)
    
    print(f"Initial: {x0}")
    print(f"Result: {result}")
    
    expected = [2.0, 3.0]
    for i in range(len(result)):
        assert abs(result[i] - expected[i]) < 1e-4
    print("BFGS Test Passed!")

if __name__ == "__main__":
    test_bfgs()
