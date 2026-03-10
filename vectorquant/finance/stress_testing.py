"""
Stress Testing & Scenario Engine
"""
from vectorquant.finance.portfolio import portfolio_return, portfolio_variance
import math

def historical_stress_test(weights, historical_returns_matrix, scenario_start_idx, scenario_end_idx):
    """
    Computes portfolio performance during a specific historical window.
    weights: list of final portfolio weights.
    historical_returns_matrix: chronological list of lists.
    scenario dates: indices representing the scenario (e.g., 2008 crash).
    Returns total cumulative return during that scenario.
    """
    scenario_returns = historical_returns_matrix[scenario_start_idx:scenario_end_idx]
    cum_return = 1.0
    
    for row in scenario_returns:
        dt_ret = portfolio_return(weights, row)
        cum_return *= (1.0 + dt_ret)
        
    return cum_return - 1.0

def hypothetical_scenario(weights, asset_betas, factor_shock):
    """
    Calculates expected portfolio impact given asset sensitivities to a factor
    and a discrete factor shock.
    asset_betas: list of betas for each asset to the underlying factor.
    factor_shock: scalar (e.g., -0.30 for a 30% market crash)
    Returns expected portfolio return.
    """
    expected_asset_returns = [beta * factor_shock for beta in asset_betas]
    return portfolio_return(weights, expected_asset_returns)

def reverse_stress_test(weights, cov_matrix, max_acceptable_loss, confidence_level=0.99):
    """
    Finds the market shock multiplier required to hit the max_acceptable_loss.
    Using parametric VaR assumption: Loss = Z * Port_Vol * Multiplier.
    Multiplier = Max_Loss / (Z * Port_Vol)
    Returns how many standard deviations the market needs to move to cause this loss.
    """
    from vectorquant.core.probability import normal_inv_cdf
    port_vol = portfolio_variance(weights, cov_matrix) ** 0.5
    if port_vol == 0: return float('inf')
    
    z_score = normal_inv_cdf(confidence_level)
    worst_case_expected = z_score * port_vol
    
    multiplier = max_acceptable_loss / worst_case_expected
    return multiplier
