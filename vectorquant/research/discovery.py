"""
Research Discovery Engine
"""
from vectorquant.research.pipeline import StrategyPipeline
import random

class DiscoveryEngine:
    def __init__(self, data_matrix):
        self.data = data_matrix
        self.hall_of_fame = [] # Top strategies
        
    def discover_signals(self, iterations=100):
        """
        Randomly generates signal functions and validates them via Pipeline.
        """
        for _ in range(iterations):
            # Define a random signal selector (e.g. random relative weights)
            def random_signal_func(panel):
                n_assets = len(panel[0])
                # Random weights as signal
                return [random.uniform(-1, 1) for _ in range(n_assets)]
            
            # Helper for portfolio: proportional to signal
            def signal_to_weight(sig):
                s = sum(abs(x) for x in sig)
                if s == 0: return [0.0] * len(sig)
                return [x / s for x in sig]
                
            pipeline = StrategyPipeline(self.data)
            try:
                metrics = (pipeline.sanitize()
                          .generate_signals(random_signal_func)
                          .construct_portfolio(signal_to_weight)
                          .evaluate())
                
                if metrics['bootstrap_sharpe'] > 0.5:
                    self.hall_of_fame.append({
                        "id": random.getrandbits(32),
                        "metrics": metrics
                    })
            except Exception:
                continue
                
        # Sort by Sharpe
        self.hall_of_fame.sort(key=lambda x: x['metrics']['bootstrap_sharpe'], reverse=True)
        return self.hall_of_fame
