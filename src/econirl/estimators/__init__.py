"""Sklearn-style estimators for dynamic discrete choice models.

This module provides high-level estimators with a scikit-learn style API:
- NFXP: Nested Fixed Point estimator (Rust 1987, 1988)
- CCP: Conditional Choice Probability estimator (Hotz-Miller 1993, NPL)

Example:
    >>> from econirl.estimators import NFXP, CCP
    >>> import pandas as pd
    >>>
    >>> # Load your data
    >>> df = pd.read_csv("bus_data.csv")
    >>>
    >>> # Create and fit the NFXP estimator
    >>> model = NFXP(n_states=90, discount=0.9999)
    >>> model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> # Or use the faster CCP estimator (Hotz-Miller)
    >>> model_ccp = CCP(n_states=90, discount=0.9999)
    >>> model_ccp.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> # Access results (same interface)
    >>> print(model.params_)
    >>> print(model.summary())
"""

from econirl.estimators.ccp import CCP
from econirl.estimators.nfxp import NFXP

__all__ = ["NFXP", "CCP"]
