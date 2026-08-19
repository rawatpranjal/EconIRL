"""Adversarial IRL methods for discrete choice models.

Production:
    AIRL (Adversarial IRL) - Fu et al. 2018
    AIRL2 (anchored AIRL with heterogeneity) - Lee, Sudhir & Wang 2026
Contrib: GAIL moved to econirl.contrib.gail
"""

import warnings

from econirl.estimation.adversarial.airl import AIRLConfig, AIRLEstimator
from econirl.estimation.adversarial.airl2 import AIRL2Config, AIRL2Estimator
from econirl.estimation.adversarial.discriminator import (
    LinearDiscriminator,
    TabularDiscriminator,
)

__all__ = [
    "TabularDiscriminator",
    "LinearDiscriminator",
    "AIRLEstimator",
    "AIRLConfig",
    "AIRL2Estimator",
    "AIRL2Config",
    "AIRLHetEstimator",
    "AIRLHetConfig",
]

_DEPRECATED_AIRL_HET = {
    "AIRLHetEstimator": AIRL2Estimator,
    "AIRLHetConfig": AIRL2Config,
}


def __getattr__(name: str):
    replacement = _DEPRECATED_AIRL_HET.get(name)
    if replacement is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"{name} is deprecated; use {replacement.__name__}. The AIRLHet alias "
        "will be removed after the 0.1.x series.",
        DeprecationWarning,
        stacklevel=2,
    )
    return replacement
