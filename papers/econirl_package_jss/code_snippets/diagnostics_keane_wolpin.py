"""Pre-estimation diagnostics on the Keane-Wolpin panel.

Reproduces: row D2 of diagnostics.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SEED = 42
np.random.seed(SEED)

from econirl.datasets import load_keane_wolpin

panel = load_keane_wolpin(as_panel=True)
features = getattr(panel, "feature_matrix", None)
if features is None:
    print("Keane-Wolpin panel does not expose feature_matrix; running coverage only.")
    rank = -1
    cond = float("inf")
    n_features = -1
else:
    phi = np.asarray(features)
    flat = phi.reshape(-1, phi.shape[-1])
    nonzero = flat[np.linalg.norm(flat, axis=1) > 0]
    rank = int(np.linalg.matrix_rank(flat))
    cond = float(np.linalg.cond(nonzero)) if nonzero.shape[0] >= nonzero.shape[1] else float("inf")
    n_features = phi.shape[-1]

n_states = getattr(panel, "num_states", None) or max((max(t.states) + 1) for t in panel.trajectories)
states_observed = set()
state_actions = {}
for traj in panel.trajectories:
    for s, a in zip(traj.states, traj.actions):
        states_observed.add(int(s))
        state_actions.setdefault(int(s), set()).add(int(a))
single_action_states = sum(1 for v in state_actions.values() if len(v) == 1)
coverage = len(states_observed) / n_states

row = {
    "dataset": "keane_wolpin",
    "feature_rank": rank,
    "n_features": n_features,
    "condition_number": f"{cond:.2e}" if cond != float("inf") else "inf",
    "state_coverage": f"{coverage:.4f}",
    "single_action_states": single_action_states,
}

out = Path(__file__).resolve().parents[1] / "diagnostics.csv"
write_header = not out.exists()
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if write_header:
        w.writeheader()
    w.writerow(row)
print(row)
