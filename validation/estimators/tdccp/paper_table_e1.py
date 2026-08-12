#!/usr/bin/env python3
"""Fit EconIRL TD-CCP to one exported official Table E.1 panel.

The panel must be produced by the authors' DGP with
``docs/research/papers/tdccp/zenodo_16184777/export_table_e1_panel.R``.
This script is an exact-panel bridge, not a replacement Monte Carlo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from econirl.core.types import DDCProblem, TrajectoryPanel
from econirl.estimation.td_ccp import TDCCPConfig, TDCCPEstimator
from econirl.preferences.action_reward import ActionDependentReward

N_X_STATES = 2_000
N_TYPES = 2
N_STATES = N_X_STATES * N_TYPES
PUBLISHED_K2 = {
    "nonrobust": np.array([1.866762590079, -0.149942707847, 1.073773905596]),
    "robust": np.array([1.863682573405, -0.149894947175, 1.075142962925]),
}


def _state_features() -> np.ndarray:
    states = np.arange(N_STATES)
    x = (states // N_TYPES).astype(float)
    kind = (states % N_TYPES).astype(float)
    return np.column_stack([kind, x, x**2, kind * x, kind * x**2, x**3, kind * x**3])


def _reward() -> ActionDependentReward:
    states = np.arange(N_STATES)
    x = (states // N_TYPES).astype(float)
    kind = (states % N_TYPES + 1).astype(float)
    features = np.zeros((N_STATES, 2, 3), dtype=float)
    features[:, 1, 0] = 1.0
    features[:, 1, 1] = x
    features[:, 1, 2] = kind
    return ActionDependentReward(
        jnp.asarray(features),
        ["theta_0", "theta_1", "theta_2"],
    )


def _load_panel(path: Path) -> TrajectoryPanel:
    frame = pd.read_csv(path)
    required = {"individual_id", "period", "x", "permanent_type", "action"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"panel is missing columns: {sorted(missing)}")
    sort_columns = (
        ["official_fold", "individual_id", "period"]
        if "official_fold" in frame
        else ["individual_id", "period"]
    )
    frame = frame.sort_values(sort_columns).reset_index(drop=True)
    ordered_ids = frame["individual_id"].drop_duplicates().tolist()
    id_order = {individual_id: index for index, individual_id in enumerate(ordered_ids)}
    frame["panel_id"] = frame["individual_id"].map(id_order)
    frame["state"] = (
        frame["x"].to_numpy(dtype=np.int64) * N_TYPES
        + frame["permanent_type"].to_numpy(dtype=np.int64)
        - 1
    )
    return TrajectoryPanel.from_dataframe(
        frame,
        state="state",
        action="action",
        id="panel_id",
    )


def fit_panel(path: Path, *, robust: bool) -> dict[str, object]:
    panel = _load_panel(path)
    encoded = _state_features()

    def state_encoder(states: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(encoded)[jnp.asarray(states, dtype=jnp.int32)]

    problem = DDCProblem(
        num_states=N_STATES,
        num_actions=2,
        discount_factor=0.9,
        scale_parameter=1.0,
        state_dim=7,
        state_encoder=state_encoder,
    )
    config = TDCCPConfig(
        method="semigradient",
        basis_type="encoded",
        basis_dim=1,
        basis_ridge=0.0,
        basis_action_coding="reference",
        ccp_method="logit",
        ccp_poly_degree=1,
        ccp_use_encoder=True,
        cross_fitting=robust,
        cross_fit_shuffle=False,
        cross_fit_ccp=False,
        robust_se=robust,
        n_policy_iterations=1,
        outer_max_iter=1_000,
        outer_tol=1e-10,
        compute_se=robust,
        compute_policy=False,
    )
    estimator = TDCCPEstimator(config=config, se_method="asymptotic", seed=0)
    result = estimator.estimate(
        panel=panel,
        utility=_reward(),
        problem=problem,
        transitions=jnp.ones((1, 1, 1)),
        initial_params=jnp.array([0.5, 0.5, 0.5]),
        transition_source="not used by the TD parameter stage",
    )
    mode = "robust" if robust else "nonrobust"
    estimates = np.asarray(result.parameters, dtype=float)
    target = PUBLISHED_K2[mode]
    absolute_error = np.abs(estimates - target)
    relative_error = absolute_error / np.maximum(np.abs(target), 1e-12)
    paper_inference = result.metadata.get("paper_inference") or {}
    return {
        "schema_version": 1,
        "claim": "official_panel_exact_fit_attempt",
        "mode": mode,
        "panel": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_individuals": panel.num_individuals,
            "n_observations": panel.num_observations,
            "official_simulation_k": 2,
        },
        "published_estimates": target.tolist(),
        "econirl_estimates": estimates.tolist(),
        "absolute_error": absolute_error.tolist(),
        "relative_error": relative_error.tolist(),
        "matches_four_significant_figures": bool(
            np.allclose(estimates, target, rtol=5e-5, atol=5e-7)
        ),
        "converged": bool(result.converged),
        "standard_errors": [
            float(value) if np.isfinite(value) else None
            for value in np.asarray(result.standard_errors, dtype=float)
        ],
        "metadata": {
            "method": result.metadata.get("method"),
            "basis_type": result.metadata.get("basis_type"),
            "cross_fitting": result.metadata.get("cross_fitting"),
            "se_method_detail": result.metadata.get("se_method_detail"),
            "folds": [
                {
                    "name": fold.get("name"),
                    "tilde_theta": np.asarray(fold.get("tilde_theta"), dtype=float).tolist(),
                    "theta": np.asarray(fold.get("theta"), dtype=float).tolist(),
                }
                for fold in paper_inference.get("folds", [])
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel", type=Path)
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = fit_panel(args.panel, robust=args.robust)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
