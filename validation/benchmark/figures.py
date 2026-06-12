"""Data-generating-process figures for the simulation-study pages.

One small 1x2 figure per experiment cell: simulated state trajectories on the
left, the optimal (true-parameter) value function on the right. Everything is
deterministic given the cell's seeds, so the figure regenerates exactly with
the run.
"""

from __future__ import annotations

import numpy as np


def dgp_figure(env, panel, oracle_value, out_path: str, n_traj: int = 12) -> None:
    """Render the 1x2 DGP figure (trajectories, optimal value) to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = int(env.num_states)
    V = np.asarray(oracle_value, dtype=np.float64).reshape(-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.2))

    for traj in list(panel.trajectories)[:n_traj]:
        states = np.asarray(traj.states, dtype=np.int64)
        ax1.plot(np.arange(states.shape[0]), states, lw=0.8, alpha=0.6)
    ax1.set_xlabel("period $t$")
    ax1.set_ylabel("state index $s_t$")
    ax1.set_title(f"simulated trajectories (first {min(n_traj, len(panel.trajectories))} agents)")

    idx = np.arange(S)
    if S > 200:
        ax2.plot(idx, V, lw=0.0, marker=".", ms=2.0, alpha=0.6)
    else:
        ax2.plot(idx, V, lw=1.2)
    ax2.set_xlabel("state index $s$")
    ax2.set_ylabel("$V^*(s)$")
    ax2.set_title("optimal value function (true parameters)")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
