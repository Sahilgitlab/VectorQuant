"""
Dynamic & Real-Time Risk Control
"""
class RiskMonitor:
    def __init__(self, max_drawdown_limit=0.15, var_limit=0.05, max_correlation=0.8):
        self.max_drawdown_limit = max_drawdown_limit # e.g. 15%
        self.var_limit = var_limit # 5% Value at Risk
        self.max_correlation = max_correlation
        self.alerts = []
        
    def check_drawdown(self, portfolio_history):
        from vectorquant.finance.financial_math import max_drawdown
        mdd = max_drawdown(portfolio_history)
        if mdd > self.max_drawdown_limit:
            self._trigger_alert("DRAWDOWN_BREACH", f"Max drawdown {mdd:.2%} exceeded limit {self.max_drawdown_limit:.2%}")
        return mdd
        
    def check_var(self, returns, confidence_level=0.95):
        from vectorquant.finance.risk_models import historical_var
        var = historical_var(returns, confidence_level)
        if var > self.var_limit:
            self._trigger_alert("VAR_BREACH", f"VaR {var:.2%} exceeded limit {self.var_limit:.2%}")
        return var
        
    def check_correlation(self, corr_matrix):
        if not corr_matrix: return 0.0
        n = len(corr_matrix)
        max_c = 0.0
        for i in range(n):
            for j in range(i+1, n):
                if corr_matrix[i][j] > max_c:
                    max_c = corr_matrix[i][j]
        
        if max_c > self.max_correlation:
            self._trigger_alert("CORRELATION_SPIKE", f"Pairwise correlation {max_c:.2f} exceeded limit {self.max_correlation:.2f}")
        return max_c
        
    def _trigger_alert(self, alert_type, message):
        self.alerts.append({"type": alert_type, "message": message, "action": "REDUCE_EXPOSURE"})
        
    def get_active_alerts(self):
        return self.alerts
        
    def clear_alerts(self):
        self.alerts = []
