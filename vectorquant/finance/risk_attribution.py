"""
Risk Attribution Engine
"""
import math
from vectorquant.finance.portfolio import portfolio_variance
from vectorquant.core.linear_algebra import matrix_multiply

def marginal_contribution_to_risk(weights, cov_matrix):
    """
    Computes MCR for each asset.
    MCR_i = d(PortVol)/d(w_i) = (Sigma * w)_i / PortVol
    """
    port_vol = portfolio_variance(weights, cov_matrix) ** 0.5
    if port_vol == 0: return [0.0] * len(weights)
    
    w_col = [[w] for w in weights]
    sigma_w = matrix_multiply(cov_matrix, w_col)
    
    mcr = [sigma_w[i][0] / port_vol for i in range(len(weights))]
    return mcr

def risk_contribution(weights, cov_matrix):
    """
    Computes Total Risk Contribution of each asset.
    RC_i = w_i * MCR_i
    Sum of RC_i equals Portfolio Volatility.
    """
    mcr = marginal_contribution_to_risk(weights, cov_matrix)
    return [weights[i] * mcr[i] for i in range(len(weights))]

def relative_risk_contribution(weights, cov_matrix):
    """
    Computes % contribution of each asset to total variance.
    %RC_i = RC_i / PortVol
    """
    rc = risk_contribution(weights, cov_matrix)
    port_vol = sum(rc) # Euler's theorem
    if port_vol == 0: return [0.0] * len(weights)
    return [r / port_vol for r in rc]

def factor_risk_attribution(portfolio_betas, factor_cov_matrix):
    """
    Computes risk contribution from orthogonal/systematic factors.
    RC_factor_k = beta_k * (Sigma_F * beta)_k / PortVol_factors
    factor_cov_matrix: covariance matrix of the factors themselves.
    """
    factor_vol_squared = portfolio_variance(portfolio_betas, factor_cov_matrix)
    if factor_vol_squared <= 0: return [0.0] * len(portfolio_betas)
    port_vol = factor_vol_squared ** 0.5
    
    beta_col = [[b] for b in portfolio_betas]
    sigma_beta = matrix_multiply(factor_cov_matrix, beta_col)
    
    mcr = [sigma_beta[i][0] / port_vol for i in range(len(portfolio_betas))]
    return [portfolio_betas[i] * mcr[i] for i in range(len(portfolio_betas))]
