"""
VectorQuant Infrastructure — Engineering Utilities

Provides parallel computation engines and data quality tools.
"""

from .parallel_engine import parallel_simulate_paths

from .data_quality import (
    outlier_detection_zscore,
    forward_fill_missing,
)
