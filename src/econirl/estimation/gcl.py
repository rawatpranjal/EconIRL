"""Compatibility imports for Guided Cost Learning.

The implementation lives in :mod:`econirl.contrib.gcl`. This module keeps the
older ``econirl.estimation.gcl`` import path working for users and tests.
"""

from econirl.contrib.gcl import GCLConfig, GCLEstimator

__all__ = ["GCLConfig", "GCLEstimator"]
