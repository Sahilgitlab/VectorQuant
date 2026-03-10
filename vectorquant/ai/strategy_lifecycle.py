"""
Strategy Lifecycle Management
"""
from enum import Enum
from datetime import datetime

class LifecycleState(Enum):
    RESEARCH = "RESEARCH"
    PAPER_TRADING = "PAPER_TRADING"
    LIVE_TRADING = "LIVE_TRADING"
    RETIRED = "RETIRED"

class StrategyLifecycle:
    def __init__(self, strategy_id, name):
        self.strategy_id = strategy_id
        self.name = name
        self.state = LifecycleState.RESEARCH
        self.history = []
        self._record_transition("Created in RESEARCH")
        
    def _record_transition(self, reason):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "state": self.state.value,
            "reason": reason
        })
        
    def evaluate_promotion(self, sharpe_ratio, periods_active, max_drawdown):
        """
        Agent rules for strategy state transitions.
        """
        if self.state == LifecycleState.RESEARCH:
            if sharpe_ratio > 1.0 and max_drawdown < 0.20:
                self.state = LifecycleState.PAPER_TRADING
                self._record_transition(f"Promoted to PAPER: Sharpe {sharpe_ratio:.2f} > 1.0")
                
        elif self.state == LifecycleState.PAPER_TRADING:
            if sharpe_ratio > 1.5 and periods_active >= 60: # Assume e.g. 60 days
                self.state = LifecycleState.LIVE_TRADING
                self._record_transition(f"Promoted to LIVE: Sharpe {sharpe_ratio:.2f} > 1.5 over {periods_active} periods")
            elif sharpe_ratio < 0.5:
                self.state = LifecycleState.RETIRED
                self._record_transition(f"Demoted to RETIRED: Poor paper performance (Sharpe {sharpe_ratio:.2f})")
                
        elif self.state == LifecycleState.LIVE_TRADING:
            if max_drawdown > 0.15 or sharpe_ratio < 0.8:
                self.state = LifecycleState.RETIRED
                self._record_transition(f"Emergency RETIRED: Drawdown {max_drawdown:.2%} or Sharpe {sharpe_ratio:.2f} breached live limits")
                
        return self.state
