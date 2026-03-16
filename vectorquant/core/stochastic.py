
import math
import random
from .probability import rnorm

def simulate_gbm(s0, mu, sigma, t, dt, n_paths, antithetic=False):
    """
    Geometric Brownian Motion simulation:
    dS = mu*S*dt + sigma*S*dW
    """
    actual_paths = n_paths * 2 if antithetic else n_paths
    n_steps = int(t / dt)
    if n_steps == 0: n_steps = 1
    cols = n_steps + 1

    paths = []
    for _ in range(n_paths):
        path1 = [s0]
        path2 = [s0] if antithetic else None
        
        curr1 = math.log(s0)
        curr2 = math.log(s0) if antithetic else 0
        
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * math.sqrt(dt)
        
        for _ in range(n_steps):
            shock = diffusion * random.gauss(0, 1)
            curr1 += drift + shock
            path1.append(math.exp(curr1))
            if antithetic:
                curr2 += drift - shock
                path2.append(math.exp(curr2))
        
        paths.append(path1)
        if antithetic:
            paths.append(path2)
            
    # Return as list of lists for Python backend consistency
    return paths

def sobol_sequence(n):
    """
    1D Sobol sequence (Van der Corput) in base 2
    """
    def van_der_corput(n, base=2):
        q, bk = 0, 1.0 / base
        while n > 0:
            q += (n % base) * bk
            n //= base
            bk /= base
        return q
        
    return [van_der_corput(i + 1) for i in range(n)]

def halton_sequence(n, dim):
    """
    Multidimensional Halton sequence
    """
    def van_der_corput(n, base):
        q, bk = 0, 1.0 / base
        while n > 0:
            q += (n % base) * bk
            n //= base
            bk /= base
        return q
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 
              59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    if dim > len(primes):
        raise ValueError(f"Max supported dimension is {len(primes)}")
        
    result = []
    for i in range(n):
        point = [van_der_corput(i + 1, primes[d]) for d in range(dim)]
        result.append(point)
    return result

def scrambled_sobol(n, seed=0):
    """
    Scrambled 1D Sobol (Digital Shift)
    """
    def van_der_corput(n, base=2):
        q, bk = 0, 1.0 / base
        while n > 0:
            q += (n % base) * bk
            n //= base
            bk /= base
        return q
        
    result = []
    for i in range(n):
        vdc = van_der_corput(i + 1)
        bits = int(vdc * 4294967296)
        bits ^= seed
        scrambled = bits / 4294967296.0
        result.append(scrambled)
    return result
