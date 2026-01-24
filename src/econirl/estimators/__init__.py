"""Sklearn-style estimators for dynamic discrete choice models.

This module provides high-level estimators with a scikit-learn style API:
- NFXP: Nested Fixed Point estimator (Rust 1987, 1988)

Example:
    >>> from econirl.estimators import NFXP
    >>> import pandas as pd
    >>>
    >>> # Load your data
    >>> df = pd.read_csv("bus_data.csv")
    >>>
    >>> # Create and fit the estimator
    >>> model = NFXP(n_states=90, discount=0.9999)
    >>> model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> # Access results
    >>> print(model.params_)
    >>> print(model.summary())
"""

from econirl.estimators.nfxp import NFXP

__all__ = ["NFXP"]
