"""Listing 4: NNES on the Keane-Wolpin occupational choice panel. Demonstrates
that the unified workflow scales to four actions and over five thousand
reachable states.

Reproduces: Listing 4.
"""
from __future__ import annotations

import numpy as np

SEED = 42
np.random.seed(SEED)

from econirl.datasets import load_keane_wolpin
from econirl.estimation.nnes import NNESEstimator

panel = load_keane_wolpin(as_panel=True)
est = NNESEstimator(
    hidden_dim=64,
    num_layers=2,
    v_epochs=200,
    outer_max_iter=100,
    n_outer_iterations=3,
)
result = est.estimate(panel=panel)
print(result.summary())
