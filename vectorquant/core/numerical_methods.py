"""
Numerical Methods for Equation Solving and Optimization
"""
from typing import Callable

def newton_raphson(f: Callable, df: Callable, x0: float, tol: float = 1e-7, max_iter: int = 100):
    """
    Finds a root of f(x) = 0 using Newton-Raphson method.
    """
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = df(x)
        if abs(dfx) < 1e-12:
            break # Derivative too small
        x = x - fx / dfx
    return x

def bisection(f: Callable, a: float, b: float, tol: float = 1e-7, max_iter: int = 100):
    """
    Finds a root of f(x) = 0 in interval [a, b] using bisection method.
    """
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
        
    for _ in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return c
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return c

def finite_difference(f: Callable, x: float, h: float = 1e-5):
    """
    Standard central finite difference approximation of f'(x).
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def runge_kutta_4(f: Callable, y0: float, t0: float, t_end: float, dt: float):
    """
    Solves ODE dy/dt = f(t, y) using RK4.
    Returns lists of t and y.
    """
    t_vals = [t0]
    y_vals = [y0]
    
    t = t0
    y = y0
    
    while t < t_end:
        k1 = f(t, y)
        k2 = f(t + dt / 2.0, y + dt / 2.0 * k1)
        k3 = f(t + dt / 2.0, y + dt / 2.0 * k2)
        k4 = f(t + dt, y + dt * k3)
        
        y += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        
        t_vals.append(t)
        y_vals.append(y)
        
    return t_vals, y_vals
