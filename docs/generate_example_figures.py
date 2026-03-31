#!/usr/bin/env python3
"""Generate static figures for RTD example pages.

Produces:
    docs/_static/taxi_gridworld_policy.png  — 5x5 optimal policy grid with value heatmap
    docs/_static/rust_bus_counterfactual.png — 2x2 counterfactual: RC and theta_c sensitivity
    docs/_static/shanghai_network.png        — spatial road network with sample trajectories

Usage:
    python docs/generate_example_figures.py
    python docs/generate_example_figures.py --only gridworld
    python docs/generate_example_figures.py --only rust_bus
    python docs/generate_example_figures.py --only shanghai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import torch

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

OUT_DIR = Path(__file__).resolve().parent / "_static"


# =====================================================================
# 1. Taxi Gridworld: 5x5 Optimal Policy with Value Heatmap
# =====================================================================

def generate_gridworld_policy_figure():
    """Generate a 5x5 grid showing optimal policy arrows and value heatmap."""
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import hybrid_iteration
    from econirl.core.types import DDCProblem
    from econirl.environments.gridworld import GridworldEnvironment
    from econirl.preferences.action_reward import ActionDependentReward

    grid_size = 5
    discount = 0.95
    env = GridworldEnvironment(grid_size=grid_size, discount_factor=discount)
    transitions = env.transition_matrices

    # Build action-dependent features (same as examples/taxi_gridworld.py)
    n_states = grid_size * grid_size
    goal_r, goal_c = grid_size - 1, grid_size - 1
    deltas = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]  # L, R, U, D, Stay
    features = torch.zeros(n_states, 5, 4)

    for s in range(n_states):
        r, c = s // grid_size, s % grid_size
        d = abs(r - goal_r) + abs(c - goal_c)
        for a, (dr, dc) in enumerate(deltas):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                nr, nc = r, c
            ns = nr * grid_size + nc
            nd = abs(nr - goal_r) + abs(nc - goal_c)
            features[s, a, 0] = -1.0 if ns != s else 0.0
            if ns != s:
                features[s, a, 1] = 1.0 if nd < d else -1.0
            if a == 2:
                features[s, a, 2] = 1.0
            elif a == 3:
                features[s, a, 2] = -1.0
            if a == 1:
                features[s, a, 3] = 1.0
            elif a == 0:
                features[s, a, 3] = -1.0

    true_params = torch.tensor([-0.5, 2.0, 0.1, 0.1])
    utility = ActionDependentReward(
        feature_matrix=features,
        parameter_names=["move_cost", "goal_approach", "northward", "eastward"],
    )
    reward_matrix = utility.compute(true_params)
    problem = DDCProblem(num_states=n_states, num_actions=5, discount_factor=discount)
    operator = SoftBellmanOperator(problem, transitions)
    result = hybrid_iteration(operator, reward_matrix, tol=1e-10)

    V = result.V.numpy()
    policy = result.policy.numpy()

    # Reshape value to grid
    V_grid = V.reshape(grid_size, grid_size)

    # Arrow directions for each action: (dx, dy) in plot coords
    # Actions: 0=Left, 1=Right, 2=Up, 3=Down, 4=Stay
    # In plot coords: x=col (right), y=row (up because imshow inverts)
    arrow_dx = [-0.3, 0.3, 0.0, 0.0, 0.0]
    arrow_dy = [0.0, 0.0, -0.3, 0.3, 0.0]  # imshow has y increasing downward

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(V_grid, cmap="YlOrRd", interpolation="nearest", aspect="equal")

    for s in range(n_states):
        r, c = s // grid_size, s % grid_size
        greedy_action = policy[s].argmax()

        if s == n_states - 1:
            # Goal state: draw star
            ax.plot(c, r, marker="*", markersize=20, color="white",
                    markeredgecolor="black", markeredgewidth=1.0, zorder=5)
        elif greedy_action == 4:
            # Stay action: draw dot
            ax.plot(c, r, marker="o", markersize=8, color="white",
                    markeredgecolor="black", markeredgewidth=1.0, zorder=5)
        else:
            dx = arrow_dx[greedy_action]
            dy = arrow_dy[greedy_action]
            ax.annotate(
                "", xy=(c + dx, r + dy), xytext=(c - dx * 0.2, r - dy * 0.2),
                arrowprops=dict(
                    arrowstyle="-|>", color="white", lw=2.5,
                    mutation_scale=18,
                ),
                zorder=5,
            )

    # Grid lines
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5, alpha=0.5)
        ax.axvline(i - 0.5, color="white", linewidth=0.5, alpha=0.5)

    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xticklabels([str(i) for i in range(grid_size)])
    ax.set_yticklabels([str(i) for i in range(grid_size)])
    ax.set_xlabel("Column", fontsize=11)
    ax.set_ylabel("Row", fontsize=11)
    ax.set_title("Optimal Policy and Value Function (5x5 Grid)", fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("V(s)", fontsize=11)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker=">", color="white", markeredgecolor="black",
                    markersize=10, label="Greedy action", linestyle="None"),
        plt.Line2D([0], [0], marker="*", color="white", markeredgecolor="black",
                    markersize=12, label="Goal (4,4)", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              framealpha=0.8, facecolor="white")

    plt.tight_layout()
    out_path = OUT_DIR / "taxi_gridworld_policy.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# =====================================================================
# 2. Rust Bus: RC and theta_c Counterfactual Analysis (2x2)
# =====================================================================

def generate_rust_bus_counterfactual_figure():
    """Generate 2x2 counterfactual figure: RC and theta_c sensitivity."""
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import hybrid_iteration
    from econirl.environments.rust_bus import RustBusEnvironment
    from econirl.preferences.linear import LinearUtility

    env = RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        num_mileage_bins=90,
        discount_factor=0.9999,
    )
    transitions = env.transition_matrices
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    operator = SoftBellmanOperator(problem, transitions)

    def solve_for_params(theta_c, rc):
        """Solve for policy and value given parameters."""
        params = torch.tensor([theta_c, rc], dtype=torch.float32)
        reward = utility.compute(params)
        sol = hybrid_iteration(operator, reward, tol=1e-10)
        return sol.policy.numpy(), sol.V.numpy()

    baseline_theta_c = 0.001
    baseline_rc = 3.0
    mileage = np.arange(90) * 5  # Each bin is ~5000 miles, show in thousands

    # --- RC scenarios ---
    rc_scenarios = [
        (1.5, "RC = 1.5 (halved)", "#2ca02c"),
        (3.0, "RC = 3.0 (baseline)", "#1f77b4"),
        (6.0, "RC = 6.0 (doubled)", "#ff7f0e"),
        (9.0, "RC = 9.0 (tripled)", "#d62728"),
    ]
    rc_policies = []
    rc_values = []
    for rc_val, _, _ in rc_scenarios:
        pol, val = solve_for_params(baseline_theta_c, rc_val)
        rc_policies.append(pol)
        rc_values.append(val)

    # --- theta_c scenarios ---
    tc_scenarios = [
        (0.0005, r"$\theta_c$ = 0.0005 (halved)", "#2ca02c"),
        (0.001, r"$\theta_c$ = 0.001 (baseline)", "#1f77b4"),
        (0.002, r"$\theta_c$ = 0.002 (doubled)", "#ff7f0e"),
    ]
    tc_policies = []
    tc_values = []
    for tc_val, _, _ in tc_scenarios:
        pol, val = solve_for_params(tc_val, baseline_rc)
        tc_policies.append(pol)
        tc_values.append(val)

    # --- Plot 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Replacement probability under RC scenarios
    ax = axes[0, 0]
    for i, (_, label, color) in enumerate(rc_scenarios):
        ax.plot(mileage, rc_policies[i][:, 1], label=label, color=color, linewidth=2)
    ax.set_xlabel("Mileage (thousands of miles)", fontsize=10)
    ax.set_ylabel("P(replace)", fontsize=10)
    ax.set_title("Replacement Probability: Varying RC", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # Top-right: Replacement probability under theta_c scenarios
    ax = axes[0, 1]
    for i, (_, label, color) in enumerate(tc_scenarios):
        ax.plot(mileage, tc_policies[i][:, 1], label=label, color=color, linewidth=2)
    ax.set_xlabel("Mileage (thousands of miles)", fontsize=10)
    ax.set_ylabel("P(replace)", fontsize=10)
    ax.set_title(r"Replacement Probability: Varying $\theta_c$", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Relative value V(s) - V(0) for RC scenarios
    ax = axes[1, 0]
    for i, (_, label, color) in enumerate(rc_scenarios):
        v_rel = rc_values[i] - rc_values[i][0]
        ax.plot(mileage, v_rel, label=label, color=color, linewidth=2)
    ax.set_xlabel("Mileage (thousands of miles)", fontsize=10)
    ax.set_ylabel("V(s) - V(0)", fontsize=10)
    ax.set_title("Relative Value Function: Varying RC", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Relative value V(s) - V(0) for theta_c scenarios
    ax = axes[1, 1]
    for i, (_, label, color) in enumerate(tc_scenarios):
        v_rel = tc_values[i] - tc_values[i][0]
        ax.plot(mileage, v_rel, label=label, color=color, linewidth=2)
    ax.set_xlabel("Mileage (thousands of miles)", fontsize=10)
    ax.set_ylabel("V(s) - V(0)", fontsize=10)
    ax.set_title(r"Relative Value Function: Varying $\theta_c$", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    out_path = OUT_DIR / "rust_bus_counterfactual.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# =====================================================================
# 3. Shanghai Route Choice: Spatial Network Map
# =====================================================================

def generate_shanghai_network_figure():
    """Generate spatial road network map with sample trajectories."""
    import pandas as pd

    data_dir = Path("/Volumes/Expansion/datasets/shanghai_taxi_rcm_airl/data")
    if not data_dir.exists():
        print("  SKIPPED: Shanghai data not available at", data_dir)
        return

    nodes = pd.read_csv(data_dir / "node.txt")
    nodes = nodes.rename(columns={"y": "lat", "x": "lon"})
    edges = pd.read_csv(data_dir / "edge.txt")
    transit = np.load(data_dir / "transit.npy")
    train_df = pd.read_csv(data_dir / "cross_validation" / "train_CV0_size1000.csv")

    # Build osmid -> (lon, lat) lookup for plotting (lon=x, lat=y)
    node_coords = {}
    for _, row in nodes.iterrows():
        node_coords[int(row["osmid"])] = (row["lon"], row["lat"])

    # Map each edge to its start/end coordinates
    edge_segments = {}  # n_id -> ((x0,y0), (x1,y1))
    edge_types = {}     # n_id -> highway type
    for _, row in edges.iterrows():
        nid = int(row["n_id"])
        u_node = int(row["u"])
        v_node = int(row["v"])
        if u_node in node_coords and v_node in node_coords:
            edge_segments[nid] = (node_coords[u_node], node_coords[v_node])
        hw = str(row["highway"]).strip()
        if "primary" in hw:
            edge_types[nid] = "primary"
        elif "secondary" in hw:
            edge_types[nid] = "secondary"
        elif "tertiary" in hw:
            edge_types[nid] = "tertiary"
        elif "living_street" in hw:
            edge_types[nid] = "living_street"
        elif "residential" in hw:
            edge_types[nid] = "residential"
        else:
            edge_types[nid] = "unclassified"

    # Color and width by road type
    type_style = {
        "primary":      ("#d62728", 1.8, 1.0),
        "secondary":    ("#ff7f0e", 1.4, 0.9),
        "tertiary":     ("#bcbd22", 1.0, 0.8),
        "residential":  ("#aec7e8", 0.6, 0.5),
        "living_street": ("#c7c7c7", 0.5, 0.4),
        "unclassified": ("#c7c7c7", 0.5, 0.4),
    }

    # Compute edge visit frequencies from training trajectories
    edge_visit_count = np.zeros(714)
    paths = []
    for _, row in train_df.iterrows():
        path = [int(x) for x in str(row["path"]).split("_")]
        paths.append(path)
        for nid in path:
            edge_visit_count[nid] += 1

    # Select 5 sample trajectories of varying length
    path_lengths = [len(p) for p in paths]
    sorted_indices = np.argsort(path_lengths)
    # Pick short, medium, and long trajectories
    n_paths = len(paths)
    sample_indices = [
        sorted_indices[n_paths // 10],
        sorted_indices[n_paths // 4],
        sorted_indices[n_paths // 2],
        sorted_indices[3 * n_paths // 4],
        sorted_indices[9 * n_paths // 10],
    ]
    sample_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#e377c2", "#17becf"]

    # --- Figure: 2 panels ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Helper: draw all edges
    def draw_network(ax, edge_colors_fn, edge_width_fn, edge_alpha_fn):
        for nid in range(714):
            if nid not in edge_segments:
                continue
            (x0, y0), (x1, y1) = edge_segments[nid]
            color, width, alpha = edge_colors_fn(nid)
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=width,
                    alpha=alpha, solid_capstyle="round", zorder=1)

    # Left panel: Road network by type + sample trajectories
    draw_network(ax1,
                 lambda nid: type_style.get(edge_types.get(nid, "unclassified"),
                                            ("#c7c7c7", 0.5, 0.4)),
                 lambda nid: None, lambda nid: None)

    # Overlay sample trajectories
    for idx, (si, color) in enumerate(zip(sample_indices, sample_colors)):
        path = paths[si]
        xs, ys = [], []
        for nid in path:
            if nid in edge_segments:
                (x0, y0), (x1, y1) = edge_segments[nid]
                if not xs:
                    xs.append(x0)
                    ys.append(y0)
                xs.append(x1)
                ys.append(y1)
        ax1.plot(xs, ys, color=color, linewidth=2.5, alpha=0.85, zorder=3,
                 label=f"Route {idx+1} ({len(path)} segments)")

    # Legend for road types
    type_patches = [
        mpatches.Patch(color="#d62728", label="Primary"),
        mpatches.Patch(color="#ff7f0e", label="Secondary"),
        mpatches.Patch(color="#bcbd22", label="Tertiary"),
        mpatches.Patch(color="#aec7e8", label="Residential"),
    ]
    leg1 = ax1.legend(handles=type_patches, loc="upper left", fontsize=7,
                      title="Road Type", title_fontsize=8, framealpha=0.9)
    ax1.add_artist(leg1)
    ax1.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax1.set_xlabel("Longitude", fontsize=10)
    ax1.set_ylabel("Latitude", fontsize=10)
    ax1.set_title("Road Network and Sample Taxi Routes", fontsize=12)
    ax1.set_aspect("equal")
    ax1.tick_params(labelsize=8)

    # Right panel: Edge popularity heatmap (visit frequency)
    # Draw all edges as light gray background first
    for nid in range(714):
        if nid not in edge_segments:
            continue
        (x0, y0), (x1, y1) = edge_segments[nid]
        ax2.plot([x0, x1], [y0, y1], color="#e0e0e0", linewidth=0.4,
                 alpha=0.5, solid_capstyle="round", zorder=0)

    # Overlay visited edges with heat colors
    visit_norm = edge_visit_count / (edge_visit_count.max() + 1e-10)
    cmap = plt.cm.hot_r
    for nid in range(714):
        if nid not in edge_segments or edge_visit_count[nid] == 0:
            continue
        (x0, y0), (x1, y1) = edge_segments[nid]
        freq = visit_norm[nid]
        color = cmap(freq)
        width = 0.5 + 3.0 * freq
        alpha = 0.4 + 0.6 * freq
        ax2.plot([x0, x1], [y0, y1], color=color, linewidth=width,
                 alpha=alpha, solid_capstyle="round", zorder=1)

    # Colorbar for visit frequency
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(0, edge_visit_count.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, shrink=0.7, pad=0.02)
    cbar.set_label("Visit count (1000 training routes)", fontsize=9)

    ax2.set_xlabel("Longitude", fontsize=10)
    ax2.set_ylabel("Latitude", fontsize=10)
    ax2.set_title("Edge Popularity from Training Data", fontsize=12)
    ax2.set_aspect("equal")
    ax2.tick_params(labelsize=8)

    # Fix scientific notation on axes
    for ax in (ax1, ax2):
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))

    plt.tight_layout(pad=2.0)
    out_path = OUT_DIR / "shanghai_network.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RTD example figures")
    parser.add_argument(
        "--only", type=str, default=None,
        choices=["gridworld", "rust_bus", "shanghai"],
        help="Generate only one figure",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generators = {
        "gridworld": ("Gridworld policy figure", generate_gridworld_policy_figure),
        "rust_bus": ("Rust bus counterfactual figure", generate_rust_bus_counterfactual_figure),
        "shanghai": ("Shanghai network figure", generate_shanghai_network_figure),
    }

    targets = [args.only] if args.only else list(generators.keys())

    for name in targets:
        label, fn = generators[name]
        print(f"\nGenerating {label}...")
        fn()

    print("\nDone!")
