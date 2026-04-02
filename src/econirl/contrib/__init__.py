"""Contributed estimators — functional but not production-tier.

These estimators have been moved out of the main API because they either
overlap significantly with a production estimator or have not been
validated to production standards. They remain importable for research
and backward compatibility.

Usage:
    >>> from econirl.contrib import MaxEntIRLEstimator
    >>> from econirl.contrib import GAILEstimator, GAILConfig
"""

from econirl.contrib.maxent_irl import MaxEntIRLEstimator
from econirl.contrib.deep_maxent_irl import DeepMaxEntIRLEstimator
from econirl.contrib.max_margin_irl import MaxMarginIRLEstimator
from econirl.contrib.max_margin_planning import MaxMarginPlanningEstimator, MMPConfig
from econirl.contrib.gcl import GCLEstimator, GCLConfig
from econirl.contrib.bayesian_irl import BayesianIRLEstimator
# IQ-Learn promoted to production (econirl.estimation.iq_learn)
from econirl.estimation.iq_learn import IQLearnEstimator, IQLearnConfig
from econirl.contrib.gail import GAILEstimator, GAILConfig

__all__ = [
    "MaxEntIRLEstimator",
    "DeepMaxEntIRLEstimator",
    "MaxMarginIRLEstimator",
    "MaxMarginPlanningEstimator",
    "MMPConfig",
    "GCLEstimator",
    "GCLConfig",
    "BayesianIRLEstimator",
    "IQLearnEstimator",
    "IQLearnConfig",
    "GAILEstimator",
    "GAILConfig",
]
