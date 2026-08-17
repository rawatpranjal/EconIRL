"""Compatibility import for the qualified public AIRL estimator."""

from econirl.estimators.airl import AIRL

NeuralAIRL = AIRL

__all__ = ["AIRL", "NeuralAIRL"]
