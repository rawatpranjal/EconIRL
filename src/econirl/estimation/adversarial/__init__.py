"""Adversarial IRL methods for discrete choice models.

Production:
    AIRL (Adversarial IRL) - Fu et al. 2018
    AIRL-Het (AIRL with heterogeneity) - Lee, Sudhir & Wang 2026
Contrib: GAIL moved to econirl.contrib.gail
"""

from econirl.estimation.adversarial.discriminator import (
    TabularDiscriminator,
    LinearDiscriminator,
)
from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig
from econirl.estimation.adversarial.airl_het import AIRLHetEstimator, AIRLHetConfig

__all__ = [
    "TabularDiscriminator",
    "LinearDiscriminator",
    "AIRLEstimator",
    "AIRLConfig",
    "AIRLHetEstimator",
    "AIRLHetConfig",
]
