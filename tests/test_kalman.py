
import math
from vectorquant.core.backend import get_backend, set_backend

def test_kalman():
    # Setup a simple 2D constant velocity model
    x = [0.0, 1.0]  # [position, velocity]
    P = [[1.0, 0.0], [0.0, 1.0]]
    F = [[1.0, 1.0], [0.0, 1.0]]
    Q = [[0.01, 0.0], [0.0, 0.01]]
    H = [[1.0, 0.0]]  # Observe position only
    R = [[0.1]]
    z = [1.1] # Observation at t=1

    backends = ["python", "c"]
    results = {}

    for b_name in backends:
        set_backend(b_name)
        backend = get_backend()
        
        # Predict
        x_p, P_p = backend.kalman_predict(x, P, F, Q)
        # Update
        x_u, P_u = backend.kalman_update(x_p, P_p, H, R, z)
        
        results[b_name] = (x_u, P_u)
        print(f"\nResults for {b_name} backend:")
        print(f"State: {x_u}")
        print(f"Covariance: {P_u}")

    # Verify consistency
    python_x, python_P = results["python"]
    c_x, c_P = results["c"]

    for i in range(len(python_x)):
        assert math.isclose(python_x[i], c_x[i], rel_tol=1e-9)
        for j in range(len(python_x)):
            assert math.isclose(python_P[i][j], c_P[i][j], rel_tol=1e-9)

    print("\nKalman Filter verification passed! (C and Python match)")

if __name__ == "__main__":
    test_kalman()
