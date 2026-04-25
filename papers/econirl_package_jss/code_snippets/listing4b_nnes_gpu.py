"""Listing 4b: same NNES fit as Listing 4 but pinned to a graphics processing
unit. Reports the wall-clock time. The same code runs on CPU when no GPU is
available.

Reproduces: follow-up listing in Section 4.3.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "gpu")

import numpy as np

SEED = 42
np.random.seed(SEED)

import jax

from econirl.datasets import load_keane_wolpin
from econirl.estimation.nnes import NNESEstimator

print(f"JAX backend: {jax.default_backend()}")
panel = load_keane_wolpin(as_panel=True)
est = NNESEstimator(hidden_dim=64, num_layers=2, v_epochs=200, outer_max_iter=100)

t0 = time.time()
result = est.estimate(panel=panel)
elapsed = time.time() - t0
print(f"NNES on Keane-Wolpin finished in {elapsed:.1f} seconds on {jax.default_backend()}.")
