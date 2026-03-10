"""
Research Framework Pipeline
"""
from vectorquant.infrastructure.data_quality import forward_fill_missing, outlier_detection_zscore
from vectorquant.research.feature_engineering import cross_sectional_zscore
from vectorquant.research.model_validation import bootstrap_performance
from vectorquant.core.statistics import mean, standard_deviation
import copy

class StrategyPipeline:
    def __init__(self, data_matrix):
        """
        data_matrix: List of dicts/lists representing the panel data
        Example: [[100, 50, 20], [101, 51, 19], ...]
        """
        self.raw_data = data_matrix
        self.cleaned_data = None
        self.signals = None
        self.weights = None
        
    def sanitize(self, fill=True, remove_outliers=False):
        """ Runs data quality checks """
        data = copy.deepcopy(self.raw_data)
        if fill:
            for i in range(len(data[0])):
                # Extract col
                col = [row[i] for row in data]
                clean_col = forward_fill_missing(col)
                for r in range(len(data)):
                    data[r][i] = clean_col[r]
                    
        self.cleaned_data = data
        return self
        
    def generate_signals(self, signal_func):
        """
        signal_func takes historical panel matrix up to T and returns vector of signals for T.
        Applies feature engineering.
        """
        self.signals = []
        for t in range(len(self.cleaned_data)):
            # In a real pipeline, only pass data up to t to prevent lookahead bias
            sig = signal_func(self.cleaned_data[:t+1])
            # Default cross-sectional Z-score norm
            from vectorquant.research.feature_engineering import cross_sectional_zscore
            sig_norm = cross_sectional_zscore(sig)
            self.signals.append(sig_norm)
            
        return self
        
    def construct_portfolio(self, portfolio_func):
        """
        portfolio_func takes standardized signals and returns asset weights.
        """
        self.weights = []
        for sig in self.signals:
            w = portfolio_func(sig)
            self.weights.append(w)
        return self
        
    def evaluate(self):
        """
        Runs simple backtest holding weights and computes bootstrap metrics.
        Returns dict of metrics.
        """
        returns = []
        T = len(self.cleaned_data)
        N = len(self.cleaned_data[0]) if T > 0 else 0
        
        for t in range(1, T):
            w = self.weights[t-1]
            # Simple return approx
            pt = self.cleaned_data[t]
            pt_1 = self.cleaned_data[t-1]
            
            day_ret = 0.0
            for i in range(N):
                if pt_1[i] != 0:
                    r_i = (pt[i] - pt_1[i]) / pt_1[i]
                    day_ret += w[i] * r_i
            returns.append(day_ret)
            
        m, s = bootstrap_performance(returns, n_bootstraps=500)
        sharpe = m / s if s > 0 else 0.0
        
        return {
            "mean_return_daily": m,
            "volatility_daily": s,
            "bootstrap_sharpe": sharpe
        }
