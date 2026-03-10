"""
Stochastic Processes
"""
import math
import random
from vectorquant.core.config import njit_fallback

@njit_fallback
def simulate_brownian_motion(T, dt, n_paths):
    """
    Standard Brownian Motion W(t)
    dW = sqrt(dt) * Z
    """
    n_steps = int(T / dt)
    paths = []
    for _ in range(n_paths):
        path = [0.0]
        for _ in range(n_steps):
            dW = math.sqrt(dt) * random.gauss(0.0, 1.0)
            path.append(path[-1] + dW)
        paths.append(path)
    return paths

@njit_fallback
def simulate_geometric_brownian_motion(S0, mu, sigma, T, dt, n_paths, antithetic=False):
    """
    dS = mu * S * dt + sigma * S * dW
    S(t+dt) = S(t) * exp((mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z)
    """
    n_steps = int(T / dt)
    paths = []
    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    
    # If antithetic, we generate pairs of paths (Z and -Z)
    paths_to_generate = n_paths // 2 if antithetic else n_paths
    
    for _ in range(paths_to_generate):
        path = [S0]
        if antithetic:
            path_anti = [S0]
            
        for _ in range(n_steps):
            Z = random.gauss(0.0, 1.0)
            path.append(path[-1] * math.exp(drift + vol * Z))
            if antithetic:
                path_anti.append(path_anti[-1] * math.exp(drift + vol * (-Z)))
                
        paths.append(path)
        if antithetic:
            paths.append(path_anti)
            
    # Handle odd number if necessary
    if antithetic and len(paths) < n_paths:
        extra_path = [S0]
        for _ in range(n_steps):
            Z = random.gauss(0.0, 1.0)
            extra_path.append(extra_path[-1] * math.exp(drift + vol * Z))
        paths.append(extra_path)
        
    return paths

@njit_fallback
def simulate_ornstein_uhlenbeck(X0, theta, mu, sigma, T, dt, n_paths):
    """
    dX = theta * (mu - X) * dt + sigma * dW
    """
    n_steps = int(T / dt)
    paths = []
    
    for _ in range(n_paths):
        path = [X0]
        for _ in range(n_steps):
            dW = math.sqrt(dt) * random.gauss(0.0, 1.0)
            dX = theta * (mu - path[-1]) * dt + sigma * dW
            path.append(path[-1] + dX)
        paths.append(path)
    return paths

@njit_fallback
def simulate_heston(S0, v0, mu, kappa, theta, sigma_v, rho, T, dt, n_paths):
    """
    Heston Model for Stochastic Volatility
    dS = mu * S * dt + sqrt(v) * S * dW1
    dv = kappa * (theta - v) * dt + sigma_v * sqrt(v) * dW2
    corr(dW1, dW2) = rho
    """
    n_steps = int(T / dt)
    s_paths = []
    v_paths = []
    
    for _ in range(n_paths):
        s_path = [S0]
        v_path = [v0]
        for _ in range(n_steps):
            Z1 = random.gauss(0.0, 1.0)
            Z2 = random.gauss(0.0, 1.0)
            # Correlated Brownian motions
            W1 = Z1
            W2 = rho * Z1 + math.sqrt(1 - rho**2) * Z2
            
            S_t = s_path[-1]
            v_t = max(v_path[-1], 0.0) # Ensure variance is non-negative
            
            dS = mu * S_t * dt + math.sqrt(v_t) * S_t * math.sqrt(dt) * W1
            dv = kappa * (theta - v_t) * dt + sigma_v * math.sqrt(v_t) * math.sqrt(dt) * W2
            
            s_path.append(S_t + dS)
            v_path.append(v_t + dv)
            
        s_paths.append(s_path)
        v_paths.append(v_path)
        
    return s_paths, v_paths

@njit_fallback
def simulate_vasicek_model(r0, a, b, sigma, T, dt, n_paths):
    """
    dr = a * (b - r) * dt + sigma * dW
    Interest rate model with mean reversion.
    """
    # Vasicek is mathematically equivalent to the Ornstein-Uhlenbeck process.
    return simulate_ornstein_uhlenbeck(r0, a, b, sigma, T, dt, n_paths)

@njit_fallback
def simulate_cir_model(r0, a, b, sigma, T, dt, n_paths):
    """
    Cox-Ingersoll-Ross model
    dr = a * (b - r) * dt + sigma * sqrt(r) * dW
    Prevents rates from going negative (if 2ab >= sigma^2).
    """
    n_steps = int(T / dt)
    paths = []
    
    for _ in range(n_paths):
        path = [r0]
        for _ in range(n_steps):
            Z = random.gauss(0.0, 1.0)
            r_t = max(path[-1], 0.0) # Ensure no sqrt of negative number
            dr = a * (b - r_t) * dt + sigma * math.sqrt(r_t) * math.sqrt(dt) * Z
            path.append(r_t + dr)
        paths.append(path)
    return paths

def simulate_gbm_gpu(S0, mu, sigma, T, dt, n_paths):
    """
    Pure GPU acceleration using CuPy.
    Calculates thousands of paths simultaneously without loops.
    
    Returns:
        List of lists (transferred back to CPU for API consistency)
        or a raw CuPy array if integrated deeper.
    """
    from vectorquant.core.config import get_array_module
    cp = get_array_module(use_gpu=True)
    
    n_steps = int(T / dt)
    
    # 1. Allocate a memory matrix on the GPU: shape (n_paths, n_steps + 1)
    paths = cp.zeros((n_paths, n_steps + 1), dtype=cp.float32)
    paths[:, 0] = S0
    
    # 2. Generate all normal random variables at once on the GPU
    #    shape (n_paths, n_steps)
    Z = cp.random.standard_normal((n_paths, n_steps), dtype=cp.float32)
    
    # 3. Compute the step multipliers
    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    
    # 4. Multipliers = exp(drift + vol * Z)
    multipliers = cp.exp(drift + vol * Z)
    
    # 5. Cumulative product along the steps axis
    # paths[:, 1:] = S0 * cumprod(multipliers)
    paths[:, 1:] = S0 * cp.cumprod(multipliers, axis=1)
    
    # Return as standard Python lists to seamlessly replace the CPU simulator
    return paths.tolist()
