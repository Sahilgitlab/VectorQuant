"""
Monte Carlo Simulation Engine
"""
from vectorquant.stochastic.processes import simulate_geometric_brownian_motion
from vectorquant.core.statistics import mean, standard_deviation
import math

class MonteCarloEngine:
    def __init__(self, n_paths=10000, parallel=False, n_jobs=None, gpu=False):
        self.n_paths = n_paths
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.gpu = gpu

    def path_dependent_option(self, S0, mu, sigma, T, dt, payoff_func, discount_rate):
        """
        Simulate asset paths and apply a payoff function to path
        payoff_func: callable taking a list (price path) and returning payoff
        """
        if self.gpu:
            from vectorquant.stochastic.processes import simulate_gbm_gpu
            paths = simulate_gbm_gpu(S0, mu, sigma, T, dt, self.n_paths)
        else:
            paths = simulate_geometric_brownian_motion(S0, mu, sigma, T, dt, self.n_paths)
        payoffs = [payoff_func(path) for path in paths]
        
        expected_payoff = mean(payoffs)
        price = math.exp(-discount_rate * T) * expected_payoff
        
        # Standard error of MC
        se = standard_deviation(payoffs) / math.sqrt(self.n_paths)
        
        return price, se

    def european_call(self, S0, K, r, sigma, T):
        # Payoff: max(S_T - K, 0)
        payoff_func = lambda path: max(path[-1] - K, 0.0)
        return self.path_dependent_option(S0, r, sigma, T, T, payoff_func, r)
        
    def asian_call(self, S0, K, r, sigma, T, dt):
        # Payoff: max(mean(S) - K, 0)
        payoff_func = lambda path: max(mean(path) - K, 0.0)
        return self.path_dependent_option(S0, r, sigma, T, dt, payoff_func, r)
