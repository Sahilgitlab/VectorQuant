"""
Core Module Example: Optimization Algorithms

Demonstrates gradient descent, BFGS, and constrained optimization.
"""

import math
import vectorquant as vq

def main():
    print("=" * 70)
    print("VectorQuant Core Module: Optimization Examples")
    print("=" * 70)

    # ─── 1. SIMPLE UNCONSTRAINED OPTIMIZATION ────────────────────────────
    print("\n1. UNCONSTRAINED OPTIMIZATION")
    print("-" * 70)
    print("Minimize: f(x, y) = (x - 3)² + (y + 2)²")
    print("Expected minimum: x=3, y=-2\n")

    def objective(v):
        """Objective function"""
        return (v[0] - 3)**2 + (v[1] + 2)**2

    def gradient(v):
        """Gradients: df/dx and df/dy"""
        return [2*(v[0] - 3), 2*(v[1] + 2)]

    x0 = [0.0, 0.0]

    # Gradient Descent
    print("Method 1: Gradient Descent (lr=0.01)")
    result_gd = vq.core.gradient_descent(
        objective, gradient, x0, 
        lr=0.01, 
        max_iter=1000
    )
    print(f"  Result:  x = {result_gd[0]:.6f}, y = {result_gd[1]:.6f}")
    print(f"  Error:   {abs(result_gd[0] - 3.0) + abs(result_gd[1] + 2.0):.6f}")

    # Newton's Method (alternative to BFGS)
    print("\nMethod 2: Gradient Descent with More Iterations")
    result_gd2 = vq.core.gradient_descent(
        objective, gradient, x0, 
        lr=0.05, 
        max_iter=500
    )
    print(f"  Result:  x = {result_gd2[0]:.6f}, y = {result_gd2[1]:.6f}")
    print(f"  Error:   {abs(result_gd2[0] - 3.0) + abs(result_gd2[1] + 2.0):.6f}")

    # ─── 2. ROSENBROCK FUNCTION (Challenging) ────────────────────────────
    print("\n2. ROSENBROCK FUNCTION (Non-convex)")
    print("-" * 70)
    print("Minimize: f(x, y) = 100*(y - x²)² + (1 - x)²")
    print("Expected minimum: x=1, y=1\n")

    def rosenbrock(v):
        x, y = v[0], v[1]
        return 100 * (y - x**2)**2 + (1 - x)**2

    def rosenbrock_grad(v):
        x, y = v[0], v[1]
        df_dx = -400 * x * (y - x**2) - 2 * (1 - x)
        df_dy = 200 * (y - x**2)
        return [df_dx, df_dy]

    x0_rb = [-1.2, 1.0]
    result_rb = vq.core.gradient_descent(rosenbrock, rosenbrock_grad, x0_rb, lr=0.0001, max_iter=5000)
    print(f"Result: x = {result_rb[0]:.6f}, y = {result_rb[1]:.6f}")
    print(f"Error:  {abs(result_rb[0] - 1.0) + abs(result_rb[1] - 1.0):.6f}")

    # ─── 3. PORTFOLIO VARIANCE MINIMIZATION ──────────────────────────────
    print("\n3. PORTFOLIO VARIANCE MINIMIZATION")
    print("-" * 70)
    print("Global Minimum Variance Portfolio\n")

    # Covariance matrix
    cov = [[0.04, 0.006, 0.002],
           [0.006, 0.025, 0.004],
           [0.002, 0.004, 0.01]]

    def portfolio_variance(weights):
        """w^T * Cov * w"""
        total = 0.0
        for i in range(len(weights)):
            for j in range(len(weights)):
                total += weights[i] * cov[i][j] * weights[j]
        return total

    def portfolio_variance_grad(weights):
        """∇(w^T * Cov * w) = 2 * Cov * w"""
        n = len(weights)
        grad = [0.0] * n
        for i in range(n):
            for j in range(n):
                grad[i] += 2 * cov[i][j] * weights[j]
        return grad

    w0 = [1/3, 1/3, 1/3]  # Equal weight starting point
    result_pmv = vq.core.gradient_descent(
        portfolio_variance, 
        portfolio_variance_grad, 
        w0,
        lr=0.01,
        max_iter=1000
    )
    
    # Normalize to sum to 1
    w_norm = [w / sum(result_pmv) for w in result_pmv]
    var_min = portfolio_variance(w_norm)

    print(f"Weights:        {[f'{w:.4f}' for w in w_norm]}")
    print(f"Min Variance:   {var_min:.6f}")
    print(f"Min Volatility: {math.sqrt(var_min):.6f} (std dev)")

    print("\n" + "=" * 70)
    print("✓ Examples completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
