"""Preprocessing utilities for DDC estimation.

This module provides transparent preprocessing functions for:
- State discretization (continuous to discrete bins)
- Panel data validation
- Next-state computation
- Running normalization for feature standardization
"""

from econirl.preprocessing.discretization import (
    discretize_state,
    discretize_mileage,
)
from econirl.preprocessing.validation import (
    check_panel_structure,
    compute_next_states,
    PanelValidationResult,
)
from econirl.preprocessing.running_norm import RunningNorm
from econirl.preprocessing.diagnostics import feature_diagnostics

__all__ = [
    "discretize_state",
    "discretize_mileage",
    "check_panel_structure",
    "compute_next_states",
    "PanelValidationResult",
    "RunningNorm",
    "feature_diagnostics",
]
