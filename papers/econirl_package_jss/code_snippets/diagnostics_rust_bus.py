"""Pre-estimation diagnostics on the Rust bus panel per BENCHMARK_PROTOCOL.md.
Reports feature matrix rank, condition number, state coverage, and
single-action state count.

Reproduces: row D1 of diagnostics.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SEED = 42
np.random.seed(SEED)

from econirl.datasets import load_rust_bus
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility

env = RustBusEnvironment(num_mileage_bins=90, discount_factor=0.9999)
panel = load_rust_bus(as_panel=True)
utility = LinearUtility.from_environment(env)

phi = np.asarray(utility.feature_matrix)
flat = phi.reshape(-1, phi.shape[-1])
nonzero = flat[np.linalg.norm(flat, axis=1) > 0]

rank = int(np.linalg.matrix_rank(flat))
cond = float(np.linalg.cond(nonzero)) if nonzero.shape[0] >= nonzero.shape[1] else float("inf")

states_observed = set()
single_action_states = 0
state_actions = {}
for traj in panel.trajectories:
    for s, a in zip(traj.states, traj.actions):
        states_observed.add(int(s))
        state_actions.setdefault(int(s), set()).add(int(a))
single_action_states = sum(1 for v in state_actions.values() if len(v) == 1)
coverage = len(states_observed) / env.num_mileage_bins

row = {
    "dataset": "rust_bus",
    "feature_rank": rank,
    "n_features": phi.shape[-1],
    "condition_number": f"{cond:.2e}",
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
