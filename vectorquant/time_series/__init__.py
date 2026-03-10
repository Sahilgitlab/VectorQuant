"""
VectorQuant Time Series — Analysis & Regime Detection

Provides moving averages, volatility estimators, AR models,
Hidden Markov Model regime detection, and CUSUM change-point detection.
"""

from .analysis import (
    sma, ema, wma,
    rolling_volatility, ewma_volatility,
    ar_1_model,
)

from .regime_detection import (
    forward_algorithm_hmm,
    viterbi_algorithm_hmm,
    cusum,
)
