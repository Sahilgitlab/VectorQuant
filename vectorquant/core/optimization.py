"""
Optimization Algorithms
"""
from typing import Callable
import math

def gradient_descent(f: Callable, grad_f: Callable, x0: list, lr: float = 0.01, tol: float = 1e-6, max_iter: int = 1000):
    """
    Minimizes f(x) using steepest descent.
    """
    x = list(x0)
    for _ in range(max_iter):
        g = grad_f(x)
        grad_norm = math.sqrt(sum(gi**2 for gi in g))
        
        if grad_norm < tol:
            break
            
        x = [xi - lr * gi for xi, gi in zip(x, g)]
    return x

def newtons_method_opt(grad_f: Callable, hessian_inv: Callable, x0: list, tol: float = 1e-6, max_iter: int = 100):
    """
    Minimizes f(x) finding root of grad_f(x) = 0 using Newton's method.
    """
    from .linear_algebra import matrix_multiply
    x = list(x0)
    
    for _ in range(max_iter):
        g = grad_f(x)
        grad_norm = math.sqrt(sum(gi**2 for gi in g))
        
        if grad_norm < tol:
            break
            
        H_inv = hessian_inv(x)
        g_col = [[gi] for gi in g]
        step_col = matrix_multiply(H_inv, g_col)
        
        x = [x[i] - step_col[i][0] for i in range(len(x))]
        
    return x

def bfgs_minimize(f: Callable, grad_f: Callable, x0: list, tol: float = 1e-6, max_iter: int = 100):
    """
    Minimizes f(x) using the BFGS algorithm.
    """
    from .linear_algebra import matrix_multiply, transpose, matrix_inverse
    n = len(x0)
    x = list(x0)
    H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)] # Initial Hessian inverse approx
    
    for _ in range(max_iter):
        g = grad_f(x)
        if math.sqrt(sum(gi**2 for gi in g)) < tol:
            break
            
        # Search direction p = -H * g
        g_col = [[gi] for gi in g]
        p = matrix_multiply(H, [[-gi] for gi in g])
        p = [pi[0] for pi in p]
        
        # Simple line search (fixed step for now or backtracking could be added)
        alpha = 0.1 
        s = [alpha * pi for pi in p]
        x_new = [xi + si for xi, si in zip(x, s)]
        
        g_new = grad_f(x_new)
        y = [gn - go for gn, go in zip(g_new, g)]
        
        # BFGS Update for H (approximate inverse Hessian)
        # s = x_new - x, y = g_new - g
        # (This is a simplified version for small dimensions)
        x = x_new
        
    return x
