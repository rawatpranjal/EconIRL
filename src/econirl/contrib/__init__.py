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
    "GAILEstimator",
    "GAILConfig",
    # Estimators demoted from the core API, re-exported here (see __getattr__)
    "TDCCP",
    "MPEC",
    "NNES",
    "SEES",
    "UFXP",
    "NeuralUFXP",
    "NeuralMPEC",
    "RHIP",
    "IQLearn",
    "FIRL",
    "BC",
    "GenPQR",
    "MaxEntIRL",
    "MaxMarginIRL",
]

# Demoted estimators are imported lazily so contrib can re-export the sklearn
# wrappers (econirl.estimators.*) without a circular import: those wrapper modules
# import back into econirl.contrib at module load, so eager imports here would
# deadlock. Attribute name equals the class name; each maps to its current home.
_DEMOTED = {
    "TDCCP": ("econirl.estimators.tdccp", "TDCCP"),
    "MPEC": ("econirl.estimation", "MPEC"),
    "NNES": ("econirl.estimators.nnes", "NNES"),
    "SEES": ("econirl.estimators.sees", "SEES"),
    "UFXP": ("econirl.estimators.ufxp", "UFXP"),
    "NeuralUFXP": ("econirl.estimators.ufxp_neural", "NeuralUFXP"),
    "NeuralMPEC": ("econirl.estimation", "NeuralMPEC"),
    "RHIP": ("econirl.estimators.rhip", "RHIP"),
    "IQLearn": ("econirl.estimation", "IQLearn"),
    "FIRL": ("econirl.estimation", "FIRL"),
    "BC": ("econirl.estimation", "BC"),
    "GenPQR": ("econirl.estimators.genpqr", "GenPQR"),
    "MaxEntIRL": ("econirl.estimators.maxent_irl", "MaxEntIRL"),
    "MaxMarginIRL": ("econirl.estimators.max_margin_irl", "MaxMarginIRL"),
}


def __getattr__(name: str):
    if name in _DEMOTED:
        import importlib

        module_path, attr = _DEMOTED[name]
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module 'econirl.contrib' has no attribute {name!r}")
