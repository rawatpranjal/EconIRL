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


# ---------------------------------------------------------------------------
# Comparison figures: pure functions of the raw records (no env, no re-run)
# ---------------------------------------------------------------------------

_FAMILY_COLOR = {"structural": "#3b6ea5", "behavioral": "#c1654a"}


def _cell_meta(data: dict, cell_id: str) -> dict:
    return next(cm for cm in data["meta"]["cells"] if cm["cell_id"] == cell_id)


def _mean_metric(data: dict, cell_id: str, name: str, key: str):
    """Mean of a record metric over the successful reps of one estimator."""
    vals = [r[key] for r in data["records"]
            if r["cell"] == cell_id and r["estimator"] == name
            and r["error"] is None and r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def results_figure(data: dict, cell_id: str, out_path: str) -> None:
    """Performance scorecard: policy-TV per estimator, lower is better.

    One horizontal bar per roster estimator for the given cell, sorted best
    first, colored by family. The single metric every estimator shares, so the
    whole roster is comparable on one axis. A pure function of the records.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cm = _cell_meta(data, cell_id)
    fam = {s["name"]: s["family"] for s in cm["roster"]}
    scored = [(s["name"], _mean_metric(data, cell_id, s["name"], "policy_tv"))
              for s in cm["roster"]]
    scored = [(n, v) for n, v in scored if v is not None]
    scored.sort(key=lambda nv: nv[1], reverse=True)  # best (lowest) ends on top

    names = [n for n, _ in scored]
    vals = [v for _, v in scored]
    colors = [_FAMILY_COLOR.get(fam.get(n), "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(names) + 1.2))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("policy total variation vs the truth (lower is better)")
    ax.set_title("Performance: distance from the true choice probabilities")
    xmax = max(vals) if vals else 1.0
    for yi, v in zip(y, vals):
        ax.text(v + 0.01 * xmax, yi, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, xmax * 1.18)
    families = [f for f in ("structural", "behavioral") if f in fam.values()]
    # Bars sort worst-last (longest at the bottom), so the upper-right corner is
    # the clear space; lower-right would sit under the longest bar.
    ax.legend(handles=[Patch(color=_FAMILY_COLOR[f], label=f) for f in families],
              loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def scaling_figure(data: dict, out_path: str) -> None:
    """Scaling: fit time and policy TV against problem size.

    One line per estimator across the study's cells, x-axis the number of
    states. Left panel is wall-clock fit time, right panel is policy TV, both
    averaged over reps. A pure function of the records across all cells.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted(data["meta"]["cells"], key=lambda cm: cm["num_states"])
    xs = [cm["num_states"] for cm in cells]
    names = [s["name"] for s in cells[0]["roster"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for name in names:
        times = [_mean_metric(data, cm["cell_id"], name, "runtime") for cm in cells]
        tvs = [_mean_metric(data, cm["cell_id"], name, "policy_tv") for cm in cells]
        ax1.plot(xs, [np.nan if t is None else t for t in times],
                 marker="o", lw=1.3, ms=4, label=name)
        ax2.plot(xs, [np.nan if v is None else v for v in tvs],
                 marker="o", lw=1.3, ms=4, label=name)
    ax1.set_xlabel("number of states")
    ax1.set_ylabel("fit time (s)")
    ax1.set_title("compute vs problem size")
    ax2.set_xlabel("number of states")
    ax2.set_ylabel("policy total variation")
    ax2.set_title("accuracy vs problem size")
    ax2.legend(loc="best", fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        ax.set_xticks(xs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
