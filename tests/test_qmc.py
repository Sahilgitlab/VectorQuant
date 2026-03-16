
import math
from vectorquant.core.backend import get_backend, set_backend

def test_qmc():
    n = 10
    dim = 3
    seed = 12345

    backends = ["python", "c"]
    results = {}

    for b_name in backends:
        set_backend(b_name)
        backend = get_backend()
        
        # Test Sobol (1D)
        s1 = backend.sobol_sequence(n)
        # Test Halton (MD)
        h = backend.halton_sequence(n, dim)
        # Test Scrambled Sobol (1D)
        ss = backend.scrambled_sobol(n, seed)
        
        results[b_name] = (s1, h, ss)
        print(f"\nResults for {b_name} backend:")
        print(f"Sobol (first 5): {s1[:5]}")
        print(f"Halton (first 2): {h[:2]}")
        print(f"Scrambled (first 5): {ss[:5]}")

    # Verify consistency
    python_s1, python_h, python_ss = results["python"]
    c_s1, c_h, c_ss = results["c"]

    for i in range(n):
        assert math.isclose(python_s1[i], c_s1[i], rel_tol=1e-9)
        assert math.isclose(python_ss[i], c_ss[i], rel_tol=1e-9)
        for d in range(dim):
            assert math.isclose(python_h[i][d], c_h[i][d], rel_tol=1e-9)

    print("\nQMC sequences verification passed! (C and Python match)")

if __name__ == "__main__":
    test_qmc()
