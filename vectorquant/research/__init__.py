"""
VectorQuant Research — Experimentation & Backtesting

Provides strategy backtesting, research pipelines, experiment tracking,
automated signal discovery, model validation, and feature engineering.
"""

from .backtesting import (
    apply_transaction_costs,
    rolling_window_backtest,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
)

from .pipeline import StrategyPipeline

from .experiment_tracker import ExperimentTracker, display_leaderboard

from .discovery import DiscoveryEngine
