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


# ---------------------------------------------------------------------------
# Tailored, per-study figures. Each is a pure function of the saved records
# and/or the environment: no estimator is re-run. The recovered reward is the
# linear utility R(s,a) = phi(s,a) . theta_hat, read off the best structural
# estimator's mean parameters via env.compute_utility_matrix(theta_hat). The
# true reward is env.compute_utility_matrix() at the true theta. The same env
# constructor that built the run rebuilds the geometry on --page, so these
# regenerate exactly like the scorecard and scaling figures.
# ---------------------------------------------------------------------------


def _structural_mean_params(data: dict, cell_id: str):
    """Mean recovered theta of the best structural estimator on a cell.

    Picks the structural estimator with the lowest parameter RMSE against the
    cell's true theta (so the heatmap shows the family's best recovery, not an
    arbitrary roster order). Returns ``(name, mean_theta)`` or ``(None, None)``
    when no structural estimator returned a theta on the truth's scale.
    """
    cm = _cell_meta(data, cell_id)
    true_theta = cm.get("true_theta")
    K = None if true_theta is None else len(true_theta)
    best = (None, None, np.inf)
    for spec in cm["roster"]:
        if spec.get("family") != "structural":
            continue
        plists = [r["params"] for r in data["records"]
                  if r["cell"] == cell_id and r["estimator"] == spec["name"]
                  and r["error"] is None and r["params"] is not None
                  and (K is None or len(r["params"]) == K)]
        if not plists:
            continue
        mean_p = np.mean(np.asarray(plists, dtype=np.float64), axis=0)
        if true_theta is not None:
            rmse = float(np.sqrt(np.mean((mean_p - np.asarray(true_theta)) ** 2)))
        else:
            rmse = 0.0
        if rmse < best[2]:
            best = (spec["name"], mean_p, rmse)
    return best[0], best[1]


def reward_heatmap(env, params, out_path: str, title: str | None = None) -> None:
    """True reward beside the recovered reward, as state x action heatmaps.

    ``params`` is the recovered theta (a K-vector). Both panels share one color
    scale so the recovered reward reads against the truth at a glance. A pure
    function of the env and the passed parameters.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R_true = np.asarray(env.compute_utility_matrix(), dtype=np.float64)
    R_hat = np.asarray(env.compute_utility_matrix(np.asarray(params, dtype=np.float64)),
                       dtype=np.float64)
    A = R_true.shape[1]
    vmin = float(min(R_true.min(), R_hat.min()))
    vmax = float(max(R_true.max(), R_hat.max()))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    for ax, R, sub in ((axes[0], R_true, "true reward"),
                       (axes[1], R_hat, "recovered reward")):
        im = ax.imshow(R, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel("action $a$")
        ax.set_ylabel("state $s$")
        ax.set_title(sub)
        ax.set_xticks(range(A))
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("$R(s, a)$")
    if title:
        fig.suptitle(title, y=0.99)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def network_plot(env, out_path: str, seed: int = 0) -> None:
    """The road network in 2D: nodes at their positions, edges, value coloring.

    Node positions are the seeded draw the env was built from (the first
    ``rng.uniform`` in ``road_network``), reproduced here from ``seed``. Edges
    are read off the transition kernel: a non-self-loop move under any action is
    an edge. Nodes are colored by the optimal value $V^*(s)$ at the true
    parameters. A pure function of the env and the construction seed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from validation.benchmark.runner import _oracle

    N = int(env.num_states)
    coords = np.random.default_rng(seed).uniform(size=(N, 2))
    T = np.asarray(env.transition_matrices, dtype=np.float64)
    _, V = _oracle(env)
    V = np.asarray(V, dtype=np.float64).reshape(-1)

    edges = set()
    for s in range(N):
        for a in range(T.shape[0]):
            s_next = int(np.argmax(T[a, s]))
            if s_next != s:
                edges.add((min(s, s_next), max(s, s_next)))

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for s0, s1 in edges:
        ax.plot([coords[s0, 0], coords[s1, 0]], [coords[s0, 1], coords[s1, 1]],
                color="#bbbbbb", lw=0.7, zorder=1)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=V, cmap="viridis", s=70,
                    edgecolors="white", linewidths=0.6, zorder=2)
    goal = int(np.argmax(V))
    ax.scatter(coords[goal, 0], coords[goal, 1], marker="*", s=260,
               c="#c1654a", edgecolors="white", linewidths=0.8, zorder=3,
               label="highest value")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$V^*(s)$ at the true parameters")
    ax.set_xlabel("position $x$")
    ax.set_ylabel("position $y$")
    ax.set_title(f"Road network ({N} nodes): spatial layout and optimal value")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def grid_maps(env, out_path: str) -> None:
    """Two NxN heatmaps on the actual grid: per-state reward and optimal value.

    The per-state reward is the best move's reward at each cell,
    $\\max_a R(s, a)$ at the true parameters. The value is $V^*(s)$. Both are
    laid out on the real grid via ``state_to_grid_position``. A pure function of
    the env.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from validation.benchmark.runner import _oracle

    N = int(env.grid_size)
    R = np.asarray(env.compute_utility_matrix(), dtype=np.float64)
    r_best = R.max(axis=1)  # best move's reward at each state
    _, V = _oracle(env)
    V = np.asarray(V, dtype=np.float64).reshape(-1)

    R_grid = np.full((N, N), np.nan)
    V_grid = np.full((N, N), np.nan)
    for s in range(int(env.num_states)):
        row, col = env.state_to_grid_position(s)
        R_grid[row, col] = r_best[s]
        V_grid[row, col] = V[s]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    for ax, grid, sub, lab in (
        (axes[0], R_grid, "per-state reward $\\max_a R(s, a)$", "$R$"),
        (axes[1], V_grid, "optimal value $V^*(s)$", "$V^*$"),
    ):
        im = ax.imshow(grid, origin="upper", cmap="viridis")
        ax.set_title(sub)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)
    fig.suptitle("Gridworld: true reward and optimal value on the grid", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def reward_curve(env, out_path: str, params=None, action_labels=None,
                 state_label: str = "state index $s$", x=None) -> None:
    """Reward against the ordered state index, one line per action, plus $V^*$.

    For the 1D-natural environments (mileage, age, inventory, aggregate
    mileage) the state index is the natural ordering, so a line per action
    shows the structural shape: a downward-sloping keep cost crossing a flat
    replacement line. When ``params`` is given the recovered reward is drawn
    dashed over the true reward, so recovery is visible on the same axes.
    Optimal value $V^*(s)$ shares a twin axis.

    ``x`` overrides the horizontal axis with a per-state ordering variable (for
    a factored state where the raw index is not monotone, e.g. fleet aggregate
    mileage). States are then plotted sorted by ``x`` so the lines read
    cleanly. A pure function of the env and the passed parameters.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from validation.benchmark.runner import _oracle

    R_true = np.asarray(env.compute_utility_matrix(), dtype=np.float64)
    S, A = R_true.shape
    _, V = _oracle(env)
    V = np.asarray(V, dtype=np.float64).reshape(-1)
    R_hat = (None if params is None
             else np.asarray(env.compute_utility_matrix(np.asarray(params, dtype=np.float64)),
                             dtype=np.float64))
    if x is None:
        s_idx = np.arange(S)
    else:
        s_idx = np.asarray(x, dtype=np.float64).reshape(-1)
        order = np.argsort(s_idx, kind="stable")
        s_idx = s_idx[order]
        R_true = R_true[order]
        V = V[order]
        if R_hat is not None:
            R_hat = R_hat[order]
    labels = action_labels or [f"action {a}" for a in range(A)]
    colors = plt.cm.tab10(np.linspace(0, 1, max(A, 2)))
    marker = "." if x is not None else None

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for a in range(A):
        ax.plot(s_idx, R_true[:, a], lw=1.6, color=colors[a], marker=marker,
                ms=4, label=f"{labels[a]} (true)")
        if R_hat is not None:
            ax.plot(s_idx, R_hat[:, a], lw=1.3, ls="--", color=colors[a],
                    alpha=0.9, label=f"{labels[a]} (recovered)")
    ax.set_xlabel(state_label)
    ax.set_ylabel("reward $R(s, a)$")
    ax.set_title("Reward shape and optimal value")
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(s_idx, V, lw=1.4, color="#555555", alpha=0.7, label="$V^*(s)$")
    ax2.set_ylabel("$V^*(s)$", color="#555555")
    ax2.tick_params(axis="y", labelcolor="#555555")
    ax2.spines["top"].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def horizon_frontier(data: dict, cell_id: str, out_path: str) -> None:
    """RHIP's signature figure: policy TV against the planning horizon $H$.

    Reads the RHIP-H0 / -H1 / -H3 / -Hinf records on a cell. The horizon is the
    single knob that spans a family of methods: $H=0$ is the Max-Margin-Planning
    end, $H=\\infty$ matches Max Causal Entropy IRL. Accuracy improves smoothly
    along the horizon. A pure function of the records.

    Wall-clock fit time is deliberately not plotted here: the $H=\\infty$ path
    reuses the optimized MCE-IRL soft value iteration, so it runs faster than the
    finite-horizon path, which does not reflect the planning cost that grows with
    $H$. Showing it would invert the real tradeoff.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [("RHIP-H0", 0.0, "H=0"), ("RHIP-H1", 1.0, "H=1"),
             ("RHIP-H3", 3.0, "H=3"), ("RHIP-Hinf", 4.0, "H=inf")]
    xs, tvs, ticks, names = [], [], [], []
    for name, x, lab in order:
        tv = _mean_metric(data, cell_id, name, "policy_tv")
        if tv is None:
            continue
        xs.append(x)
        tvs.append(tv)
        ticks.append(lab)
        names.append(name)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    tv_color = "#3b6ea5"
    ax.plot(xs, tvs, marker="o", lw=1.6, ms=6, color=tv_color)
    ax.set_xlabel("planning horizon $H$")
    ax.set_ylabel("policy total variation vs the truth")
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks)
    ax.set_title("RHIP: accuracy across the planning horizon", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Anchor the two endpoints to their classic-method names.
    if xs:
        y0 = tvs[0]
        ax.annotate("H=0 = Max-Margin-Planning end", xy=(xs[0], y0),
                    xytext=(6, -14), textcoords="offset points", fontsize=8,
                    color=tv_color)
        if names[-1] == "RHIP-Hinf":
            ax.annotate("H=inf = MCE-IRL", xy=(xs[-1], tvs[-1]),
                        xytext=(-92, 10), textcoords="offset points", fontsize=8,
                        color=tv_color)

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
