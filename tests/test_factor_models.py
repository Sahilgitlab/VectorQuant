"""
Factor Models Validation

Verifies the mathematical correctness of CAPM, Fama-French models,
and Multivariate OLS sensitivity estimation.
"""

import sys
import os
import math
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq

def test_capm():
    rf = 0.02
    beta = 1.2
    rm = 0.10
    
    # E(R) = 0.02 + 1.2 * (0.10 - 0.02) = 0.02 + 0.096 = 0.116
    expected = vq.finance.capm_expected_return(rf, beta, rm)
    assert abs(expected - 0.116) < 1e-6

def test_fama_french_3_factor():
    rf = 0.01
    
    # Betas
    b_mkt = 1.1
    b_smb = 0.5
    b_hml = -0.2
    
    # Expected premiums
    ep_mkt = 0.06
    ep_smb = 0.02
    ep_hml = 0.03
    
    # 0.01 + (1.1*0.06) + (0.5*0.02) + (-0.2*0.03)
    # 0.01 + 0.066 + 0.010 - 0.006 = 0.080
    expected = vq.finance.fama_french_3_factor(
        rf, 
        b_mkt, ep_mkt,
        b_smb, ep_smb,
        b_hml, ep_hml
    )
    assert abs(expected - 0.080) < 1e-6

def test_fama_french_5_factor():
    rf = 0.02
    # FF3 components: 1.0 * 0.05 + 0.2 * 0.02 + 0.3 * 0.02 = 0.05 + 0.004 + 0.006 = 0.06
    # FF5 extensions: 0.1 * 0.01 + (-0.1) * 0.02 = 0.001 - 0.002 = -0.001
    # Total = 0.02 + 0.06 - 0.001 = 0.079
    expected = vq.finance.fama_french_5_factor(
        0.02,
        1.0, 0.05,
        0.2, 0.02,
        0.3, 0.02,
        0.1, 0.01,
        -0.1, 0.02
    )
    assert abs(expected - 0.079) < 1e-6

def test_estimate_factor_betas():
    # Synthetic asset returns completely driven by a market factor (beta=1.5) + alpha(0.01)
    # y = alpha + 1.5 * MKT
    MKT = [0.01, -0.02, 0.03, -0.01, 0.05]
    y = [0.01 + 1.5 * rm for rm in MKT]
    
    # Run OLS sensitivity
    # factor_returns shape: (T, 1) to pass to OLS X
    factor_returns = [[rm] for rm in MKT]
    
    betas = vq.finance.estimate_factor_betas(y, factor_returns, add_intercept=True)
    
    # We expect beta[0] (MKT) to be ~1.5 and beta[1] (alpha) to be ~0.01
    assert abs(betas[0] - 1.5) < 1e-5
    assert abs(betas[1] - 0.01) < 1e-5
    
def test_estimate_factor_betas_multivariate():
    # Test auto-orientation detection (passing factor returns as n_factors x T)
    F1 = [0.01, 0.02, -0.01, 0.03, 0.00]
    F2 = [0.00,  0.01, -0.02, 0.01, 0.02]
    
    # y = 0.05 + 2.0*F1 - 1.0*F2
    y = [0.05 + 2.0*f1 - 1.0*f2 for f1, f2 in zip(F1, F2)]
    
    factors = [F1, F2] # Shape is (n_factors, T) which is 2x5
    
    betas = vq.finance.estimate_factor_betas(y, factors, add_intercept=True)
    
    assert abs(betas[0] - 2.0) < 1e-5
    assert abs(betas[1] - -1.0) < 1e-5
    assert abs(betas[2] - 0.05) < 1e-5 # Alpha is always appended last
