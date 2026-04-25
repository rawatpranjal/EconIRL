"""Listing 1: introduction teaser. Fits NFXP on the Rust bus panel and prints
a short summary. Output is the eight-line console block reproduced in Listing 1
of the paper.

Reproduces: Listing 1.
Run from repo root: python papers/econirl_package_jss/code_snippets/teaser_nfxp_rust.py
"""
from __future__ import annotations

import numpy as np

SEED = 42
np.random.seed(SEED)

from econirl.datasets import load_rust_bus
from econirl.estimation import NFXP

panel = load_rust_bus(as_panel=True)
result = NFXP(discount_factor=0.9999).estimate(panel)
print(result.summary())
print(result.identification.condition_number)
print(result.confidence_interval(level=0.95))
