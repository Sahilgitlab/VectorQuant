"""
Probability Engine
"""
import math
import time
import random

# Simple Linear Congruential Generator
class LCG:
    def __init__(self, seed=None):
        if seed is None:
            self.state = int(time.time() * 1000)
        else:
            self.state = seed
        self.a = 1664525
        self.c = 1013904223
        self.m = 2**32

    def uniform(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

_default_rng = LCG()

def set_seed(seed):
    global _default_rng
    _default_rng = LCG(seed)
    random.seed(seed)

def runif():
    return _default_rng.uniform()

def normal_pdf(x, mu=0.0, sigma=1.0):
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

def normal_cdf(x, mu=0.0, sigma=1.0):
    # Abramowitz and Stegun approximation
    z = (x - mu) / sigma
    if z < 0:
        return 1.0 - normal_cdf(-x, -mu, sigma)
    
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    
    t = 1.0 / (1.0 + p * z)
    poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    return 1.0 - normal_pdf(x, mu, sigma) * sigma * poly

def normal_inv_cdf(p, mu=0.0, sigma=1.0):
    # Beasley-Springer-Moro approximation
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    
    a0, a1, a2, a3 = 2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637
    b1, b2, b3, b4 = -8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833
    c0, c1, c2, c3 = 0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863
    c4, c5, c6, c7, c8 = 0.0038405729373609, 0.0003951896511919, 0.0000321767881768, 0.0000002888167364, 0.00000003960315187
    
    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        z = y * (((a3 * r + a2) * r + a1) * r + a0) / ((((b4 * r + b3) * r + b2) * r + b1) * r + 1)
    else:
        r = p
        if y > 0:
            r = 1 - p
        r = math.log(-math.log(r))
        z = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r * (c7 + r * c8)))))))
        if y < 0:
            z = -z
            
    return mu + z * sigma

# Box-Muller transform
def rnorm(mu=0.0, sigma=1.0):
    u1 = runif()
    u2 = runif()
    # avoid log(0)
    while u1 < 1e-15:
        u1 = runif()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mu + z0 * sigma

def lognormal_pdf(x, mu=0.0, sigma=1.0):
    if x <= 0: return 0.0
    return (1.0 / (x * sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((math.log(x) - mu) / sigma) ** 2)

def student_t_pdf(x, df):
    # Uses simplified gamma approximations for small integer df or approximations for large
    # A full exact implementation requires gamma function
    # Providing a basic approximation here
    return (1.0 / math.sqrt(df * math.pi)) * math.exp(-0.5 * (df + 1) * math.log(1 + (x * x) / df))

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def exponential_pdf(x, lmbda):
    if x < 0: return 0.0
    return lmbda * math.exp(-lmbda * x)

def poisson_pmf(k, lmbda):
    return (math.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)
