"""
Factor Models

Implements fundamental and statistical factor models for expected returns
and risk attribution. 

Models include:
- CAPM (Capital Asset Pricing Model)
- Fama-French 3-Factor Model
- Fama-French 5-Factor Model
- Rolling OLS Beta Estimation
"""

from vectorquant.core.statistics import linear_regression
from vectorquant.core.linear_algebra import transpose

def capm_expected_return(risk_free_rate, beta, market_return):
    """
    Computes expected return using the Capital Asset Pricing Model (CAPM).
    E(R) = R_f + beta * (E(R_m) - R_f)
    """
    return risk_free_rate + beta * (market_return - risk_free_rate)


def fama_french_3_factor(risk_free_rate, beta_mkt, expected_mkt_premium,
                         beta_smb, expected_smb_premium,
                         beta_hml, expected_hml_premium):
    """
    Computes expected return using the Fama-French 3-Factor Model.
    
    Factors:
    - MKT: Market Risk Premium (R_m - R_f)
    - SMB: Size Premium (Small Minus Big)
    - HML: Value Premium (High Minus Low Book-to-Market)
    """
    expected_return = risk_free_rate + (
        beta_mkt * expected_mkt_premium +
        beta_smb * expected_smb_premium +
        beta_hml * expected_hml_premium
    )
    return expected_return


def fama_french_5_factor(risk_free_rate, beta_mkt, expected_mkt_premium,
                         beta_smb, expected_smb_premium,
                         beta_hml, expected_hml_premium,
                         beta_rmw, expected_rmw_premium,
                         beta_cma, expected_cma_premium):
    """
    Computes expected return using the Fama-French 5-Factor Model.
    
    Extends FF3 with:
    - RMW: Profitability Premium (Robust Minus Weak)
    - CMA: Investment Premium (Conservative Minus Aggressive)
    """
    expected_return = risk_free_rate + (
        beta_mkt * expected_mkt_premium +
        beta_smb * expected_smb_premium +
        beta_hml * expected_hml_premium +
        beta_rmw * expected_rmw_premium +
        beta_cma * expected_cma_premium
    )
    return expected_return


def estimate_factor_betas(asset_returns, factor_returns, add_intercept=True):
    """
    Estimates the factor betas (sensitivities) of an asset to a set of factors using OLS.
    
    Args:
        asset_returns: List of asset returns [y_1, y_2, ... y_T]
        factor_returns: List of lists containing factor returns. 
                        Shape should be (n_factors, T) OR (T, n_factors).
                        The function auto-detects orientation.
        add_intercept: If True, adds an alpha (intercept) term to the regression.
        
    Returns:
        betas: List of beta coefficients matching the factors. If add_intercept=True, 
               the intercept (alpha) is the *last* element in the list.
    """
    if not asset_returns or not factor_returns:
        raise ValueError("Returns arrays cannot be empty.")
        
    T = len(asset_returns)
    
    # Determine orientation of factor_returns
    if len(factor_returns) == T and isinstance(factor_returns[0], list):
        # Format is (T, n_factors) - exactly what we need for X
        X = [list(row) for row in factor_returns] # copy
    elif len(factor_returns[0]) == T:
        # Format is (n_factors, T) - need to transpose
        X = transpose(factor_returns)
    else:
        raise ValueError("Dimension mismatch between asset_returns and factor_returns.")
        
    # Add intercept column (alpha) to the END
    if add_intercept:
        for i in range(T):
            X[i].append(1.0)
            
    # Run OLS: returns [beta_1, beta_2, ... beta_k, alpha]
    coefficients = linear_regression(X, asset_returns)
    
    return coefficients
