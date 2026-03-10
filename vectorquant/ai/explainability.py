"""
AI Explainability Engine
"""

def explain_decision(prev_state, new_state, alerts):
    """
    Generates a structured explanation object for the LLM.
    prev_state/new_state: dictionaries containing {'vol': v, 'exp': e, 'sharpe': s}
    """
    explanations = []
    
    # 1. Analyze exposure change
    exp_delta = new_state.get('exp', 0) - prev_state.get('exp', 0)
    if abs(exp_delta) > 0.01:
        direction = "increased" if exp_delta > 0 else "reduced"
        explanations.append(f"Portfolio exposure was {direction} by {abs(exp_delta):.1%}.")
        
    # 2. Check for alerts
    for alert in alerts:
        explanations.append(f"Triggered {alert['type']}: {alert['message']}. Action taken: {alert['action']}.")
        
    # 3. Volatility check
    vol_new = new_state.get('vol', 0)
    vol_prev = prev_state.get('vol', 0)
    if vol_new > vol_prev * 1.2:
        explanations.append(f"Market volatility has increased significantly ({vol_prev:.2%} -> {vol_new:.2%}), prompting risk-off behavior.")
        
    return {
        "summary": " ".join(explanations),
        "data_points": {
            "exposure_delta": exp_delta,
            "volatility_current": vol_new,
            "active_alerts_count": len(alerts)
        }
    }
