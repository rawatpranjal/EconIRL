"""Adversarial IRL methods for discrete choice models.

Production: AIRL (Adversarial IRL) - Fu et al. 2018
Contrib: GAIL moved to econirl.contrib.gail
"""

from econirl.estimation.adversarial.discriminator import (
    TabularDiscriminator,
    LinearDiscriminator,
)
from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig

__all__ = [
    "TabularDiscriminator",
    "LinearDiscriminator",
    "AIRLEstimator",
    "AIRLConfig",
]
