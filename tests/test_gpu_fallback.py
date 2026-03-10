"""
GPU Monte Carlo Fallback Test

Ensures that calling gpu=True fails gracefully if CuPy
is not installed on the system (which is default behavior).
"""

import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorquant as vq
from vectorquant.core.config import CUPY_AVAILABLE, get_array_module

def test_cupy_unavailable_raises_importerror_gracefully(monkeypatch):
    """
    Forces CUPY_AVAILABLE to False to mock an environment without CuPy.
    Verifies that calling the engine with gpu=True gracefully raises 
    ImportError instructing the user to install the [gpu] extra.
    """
    import vectorquant.core.config
    monkeypatch.setattr(vectorquant.core.config, 'CUPY_AVAILABLE', False)
    
    with pytest.raises(ImportError) as excinfo:
        vectorquant.core.config.get_array_module(use_gpu=True)
        
    assert "Install via: pip install vectorquant[gpu]" in str(excinfo.value)
    
def test_montecarlo_engine_gpu_flag_raises_error(monkeypatch):
    import vectorquant.core.config
    monkeypatch.setattr(vectorquant.core.config, 'CUPY_AVAILABLE', False)
    

    # Mocking standard options pricing variables
    S0 = 100.0
    K = 105.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    # Init MonteCarlo with gpu
    mc_engine = vq.stochastic.MonteCarloEngine(n_paths=100, gpu=True)
    
    with pytest.raises(ImportError):
        # Trigger the engine
        mc_engine.european_call(S0, K, r, sigma, T)
