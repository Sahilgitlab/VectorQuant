"""
VectorQuant — Universal Result Schema

Every VectorQuant function returns a VQResult. This provides:
- The computed value
- Verification status and confidence
- Step-by-step proof trace with LaTeX
- Academic citation
- Unit and convention metadata
- Backend information

VQResult supports __float__ and comparison operators for backward
compatibility with code that expects raw numeric returns.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProofStep:
    """A single step in a proof trace."""
    name: str                    # e.g. "compute_mean"
    formula: str                 # e.g. "sum(r) / n"
    formula_latex: str           # e.g. r"\bar{r} = \frac{\sum r_i}{n}"
    value: Any                   # the computed value at this step
    explanation: str             # one sentence, plain English


@dataclass
class UnitCheckResult:
    """Result of dimensional / unit validation."""
    passed: bool
    expected_scale: str          # e.g. "ratio (-3 to 3)"
    actual_interpretation: str   # e.g. "0.003 is consistent with daily return"
    likely_error: Optional[str] = None     # e.g. "forgot sqrt(252) annualisation"
    corrected_value: Optional[float] = None


@dataclass
class VQResult:
    """Universal result container for all VectorQuant computations.

    Supports backward-compatible numeric operations via __float__,
    __int__, and rich comparison operators. Code that previously
    expected a raw float will continue to work::

        var = vq.risk.parametric_var(returns)
        assert var > 0          # works — compares var.value
        x = var + 1.0           # works — uses float(var.value)
    """
    # Core — always present
    value: Any                              # the answer
    verified: bool = True                   # was this computation verified?
    confidence: float = 1.0                 # 0.0 to 1.0
    formula_used: str = ""                  # human-readable
    formula_latex: str = ""                 # LaTeX
    citation: str = ""                      # academic source
    proof_trace: list = field(default_factory=list)   # list[ProofStep]
    backend: str = "python"                 # "python" | "C" | "numpy"
    computation_time_ms: float = 0.0        # performance

    # Hallucination detection fields — None when not applicable
    failure_mode: Optional[str] = None      # "formula"|"convention"|"unit"|"distribution"|None
    unit_check: Optional[UnitCheckResult] = None
    convention_used: Optional[dict] = None

    # Metadata
    warnings: list = field(default_factory=list)

    # ── Convenience properties ──────────────────────────────────────────

    @property
    def is_correct(self) -> bool:
        """True if the computation is verified and no failure mode detected."""
        return self.verified and self.failure_mode is None

    # ── Backward-compatible numeric operations ──────────────────────────

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __abs__(self):
        return abs(self.value)

    def __neg__(self):
        return -self.value

    def __pos__(self):
        return +self.value

    def __round__(self, n=None):
        return round(self.value, n)

    # Arithmetic
    def __add__(self, other):
        other_val = other.value if isinstance(other, VQResult) else other
        return self.value + other_val

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        other_val = other.value if isinstance(other, VQResult) else other
        return self.value - other_val

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        other_val = other.value if isinstance(other, VQResult) else other
        return self.value * other_val

    def __rmul__(self, other):
        return other * self.value

    def __truediv__(self, other):
        other_val = other.value if isinstance(other, VQResult) else other
        return self.value / other_val

    def __rtruediv__(self, other):
        return other / self.value

    # Comparisons
    def __lt__(self, other):
        other_val = other.value if isinstance(other, VQResult) else other
        return self.value < other_val

    def __le__(self, other):
        other_val = other.value if isinstance(other, VQResult) else other
        return self.value <= other_val

    def __gt__(self, other):
        other_val = other.value if isinstance(other, VQResult) else other
        return self.value > other_val

    def __ge__(self, other):
        other_val = other.value if isinstance(other, VQResult) else other
        return self.value >= other_val

    def __eq__(self, other):
        if isinstance(other, VQResult):
            return self.value == other.value
        return self.value == other

    def __ne__(self, other):
        if isinstance(other, VQResult):
            return self.value != other.value
        return self.value != other

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        v = f"{self.value:.6f}" if isinstance(self.value, float) else repr(self.value)
        return f"VQResult(value={v}, verified={self.verified})"

    def __bool__(self):
        return bool(self.value)


def _timed_result(func):
    """Decorator to automatically populate computation_time_ms."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        if isinstance(result, VQResult):
            result.computation_time_ms = elapsed
        return result
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
