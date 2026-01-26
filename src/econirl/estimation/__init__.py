"""Estimation algorithms for dynamic discrete choice models."""

from econirl.estimation.base import Estimator, EstimationResult
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.maxent_irl import MaxEntIRLEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.estimation.max_margin_planning import MaxMarginPlanningEstimator, MMPConfig
from econirl.estimation.gcl import GCLEstimator, GCLConfig
from econirl.estimation.transitions import (
    estimate_transition_probs,
    estimate_transition_probs_by_group,
)

# Adversarial methods
from econirl.estimation.adversarial import (
    GAILEstimator,
    GAILConfig,
    AIRLEstimator,
    AIRLConfig,
    TabularDiscriminator,
    LinearDiscriminator,
)

__all__ = [
    # Base
    "Estimator",
    "EstimationResult",
    # Forward estimation
    "NFXPEstimator",
    "CCPEstimator",
    # IRL methods
    "MaxEntIRLEstimator",
    "MCEIRLEstimator",
    "MCEIRLConfig",
    "MaxMarginPlanningEstimator",
    "MMPConfig",
    "GCLEstimator",
    "GCLConfig",
    # Adversarial methods
    "GAILEstimator",
    "GAILConfig",
    "AIRLEstimator",
    "AIRLConfig",
    "TabularDiscriminator",
    "LinearDiscriminator",
    # Utilities
    "estimate_transition_probs",
    "estimate_transition_probs_by_group",
]
