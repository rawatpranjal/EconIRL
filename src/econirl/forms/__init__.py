"""Synthetic-DGP forms framework (internal).

Known-truth dynamic-choice problems in standard topologies, plus the estimator
capability registry that says which forms each estimator can run. This subpackage
is not yet part of the public ``econirl`` API surface; the public exports and the
factories/loader are added in later chunks.
"""

from econirl.forms.base import REWARD_FORMS, TOPOLOGIES, Form, FormSpec
from econirl.forms.capabilities import CAPABILITIES, EstimatorCapability

__all__ = [
    "Form",
    "FormSpec",
    "TOPOLOGIES",
    "REWARD_FORMS",
    "EstimatorCapability",
    "CAPABILITIES",
]
