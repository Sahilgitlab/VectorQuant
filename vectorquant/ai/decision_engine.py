"""
Strategy Selection & Allocation Intelligence
"""
def score_strategy(sharpe, stability_prob, liquidity_score):
    """
    Score = Sharpe * Stability * Liquidity
    stability_prob: P(SR > benchmark) from Probabilistic Sharpe Ratio
    liquidity_score: 0.0 to 1.0 normalized market depth penalty
    """
    if sharpe < 0: return 0.0
    return sharpe * stability_prob * liquidity_score

def rank_strategies(strategy_metrics_list):
    """
    Takes a list of dicts: [{'id': 1, 'sharpe': 1.5, 'stability': 0.9, 'liquidity': 1.0}]
    Returns sorted array by composite score.
    """
    scored = []
    for s in strategy_metrics_list:
        score = score_strategy(s.get('sharpe', 0), s.get('stability', 0), s.get('liquidity', 1.0))
        scored.append((score, s))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for score, s in scored]

def dynamic_regime_allocation(base_weights, current_volatility, vol_threshold=0.20):
    """
    Agent rule: if high volatility regime, reduce exposure by holding cash.
    """
    if current_volatility > vol_threshold:
        # Scale down exposure proportionally
        scale_factor = vol_threshold / current_volatility
        return [w * scale_factor for w in base_weights]
    return base_weights
