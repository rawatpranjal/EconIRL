"""Adversarial IRL methods for discrete choice models.

Production:
    AIRL - unified identified AIRL facade:
        Fu et al. 2018 state-only mode
        Lee, Sudhir & Wang 2026 anchored and heterogeneous modes
Contrib: GAIL moved to econirl.contrib.gail
"""

from econirl.estimation.adversarial.discriminator import (
    TabularDiscriminator,
    LinearDiscriminator,
)
from econirl.estimation.adversarial.airl import AIRL, AIRLEstimator, AIRLConfig
from econirl.estimation.adversarial.airl_het import AIRLHetEstimator, AIRLHetConfig

__all__ = [
    "TabularDiscriminator",
    "LinearDiscriminator",
    "AIRL",
    "AIRLEstimator",
    "AIRLConfig",
    "AIRLHetEstimator",
    "AIRLHetConfig",
]
