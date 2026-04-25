"""Figure 4: modal occupational choice at age twenty as a function of years
of completed schooling and white-collar experience, from the NNES fit on the
Keane-Wolpin panel.

Reproduces: figures/fig4_keane_wolpin_policy.pdf.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

from econirl.datasets import load_keane_wolpin
from econirl.estimation.nnes import NNESEstimator

panel = load_keane_wolpin(as_panel=True)
est = NNESEstimator(hidden_dim=64, num_layers=2, v_epochs=200, outer_max_iter=100)
result = est.estimate(panel=panel)

policy = np.asarray(result.policy)
modal = policy.argmax(axis=-1)

n_school = 20
n_exp = 20
heat = np.full((n_school, n_exp), np.nan)
for s in range(min(n_school, modal.shape[0])):
    for e in range(min(n_exp, modal.shape[0])):
        idx = s * n_exp + e
        if idx < modal.shape[0]:
            heat[s, e] = modal[idx]

fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(heat, origin="lower", aspect="auto", cmap="viridis")
ax.set_xlabel("White-collar experience (years)")
ax.set_ylabel("Schooling (years)")
ax.set_title("Modal occupational choice at age 20")
cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(["school", "white", "blue", "home"])
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "figures" / "fig4_keane_wolpin_policy.pdf"
fig.savefig(out)
print(f"wrote {out}")
