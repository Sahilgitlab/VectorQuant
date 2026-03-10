"""
Reinforcement Learning for Allocation (Framework)
"""

class AllocationEnv:
    """
    Simplified MDP Environment for Portfolio Control.
    State: [Volatility, Momentum, Regime, CurrentExposure]
    Action: [DecreaseRisk, Hold, IncreaseRisk]
    """
    def __init__(self, initial_state):
        self.state = initial_state # [vol, mom, regime, exp]
        self.history = []
        
    def step(self, action, next_market_data, realized_return):
        """
        Executes an action and transition to next state.
        action: -1 (De-risk), 0 (Hold), 1 (Risk-on)
        reward: risk-adjusted return
        """
        # 1. Update Exposure
        prev_exp = self.state[3]
        new_exp = max(0.0, min(2.0, prev_exp + action * 0.1)) # 10% increments, cap at 2x leverage
        
        # 2. Transition State
        vol, mom, regime = next_market_data
        self.state = [vol, mom, regime, new_exp]
        
        # 3. Compute Reward: Sharpe-like local reward
        # (Realized Return * Exposure) / Vol
        reward = (realized_return * new_exp) / (vol if vol > 0.01 else 0.01)
        
        return self.state, reward

class BasicQTable:
    """
    Minimalistic placeholder for a Q-Learning agent logic.
    """
    def __init__(self):
        self.q_values = {} # (state_bucket, action) -> value
        
    def discretize_state(self, state):
        # vol, mom, regime, exp
        return (round(state[0], 1), round(state[1], 1), int(state[2]), round(state[3], 1))
        
    def get_action(self, state, epsilon=0.1):
        import random
        if random.random() < epsilon:
            return random.choice([-1, 0, 1])
        
        s = self.discretize_state(state)
        best_a = 0
        max_q = float('-inf')
        for a in [-1, 0, 1]:
            q = self.q_values.get((s, a), 0.0)
            if q > max_q:
                max_q = q
                best_a = a
        return best_a

    def update(self, state, action, reward, next_state, alpha=0.1, gamma=0.9):
        s = self.discretize_state(state)
        ns = self.discretize_state(next_state)
        
        current_q = self.q_values.get((s, action), 0.0)
        max_next_q = max(self.q_values.get((ns, a), 0.0) for a in [-1, 0, 1])
        
        # Q-Learning update rule
        self.q_values[(s, action)] = current_q + alpha * (reward + gamma * max_next_q - current_q)
