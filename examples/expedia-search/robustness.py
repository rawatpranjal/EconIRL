#!/usr/bin/env python3
"""
Expedia Hotel Search -- Robustness Checks
==========================================

Data loaded once; specs subsample from the pre-loaded dataframe.
Specs vary: discount factor, session window, subsampling, action framing,
and interaction terms. Uses NPL K=3 for speed.

Specs:
    baseline      -- beta=0.95, first 30K sessions
    beta_090      -- beta=0.90
    beta_099      -- beta=0.99
    window_10k    -- 10K sessions
    window_50k    -- 50K sessions
    subsample_b   -- sessions 30K-60K by srch_id order
    action_2      -- 2-action: scroll vs engage (click+book)
    interactions  -- adds position×quality, position×price on click/book

Usage:
    python examples/expedia-search/robustness.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.ccp import CCPEstimator
from econirl.preferences.linear import LinearUtility

DATA_DIR = Path("data/raw/expedia")

POSITION_LABELS = 5
PRICE_BINS = 3
QUALITY_BINS = 2
NUM_STATES = POSITION_LABELS * PRICE_BINS * QUALITY_BINS  # 30

NEEDED_COLS = [
    "srch_id", "position", "price_usd", "prop_starrating",
    "prop_review_score", "prop_brand_bool", "promotion_flag",
    "click_bool", "booking_bool",
]

SPECS = {
    "baseline":     {"discount_factor": 0.95, "max_sessions": 30_000, "offset": 0,      "n_actions": 3, "interactions": False},
    "beta_090":     {"discount_factor": 0.90, "max_sessions": 30_000, "offset": 0,      "n_actions": 3, "interactions": False},
    "beta_099":     {"discount_factor": 0.99, "max_sessions": 30_000, "offset": 0,      "n_actions": 3, "interactions": False},
    "window_10k":   {"discount_factor": 0.95, "max_sessions": 10_000, "offset": 0,      "n_actions": 3, "interactions": False},
    "window_50k":   {"discount_factor": 0.95, "max_sessions": 50_000, "offset": 0,      "n_actions": 3, "interactions": False},
    "subsample_b":  {"discount_factor": 0.95, "max_sessions": 30_000, "offset": 30_000, "n_actions": 3, "interactions": False},
    "action_2":     {"discount_factor": 0.95, "max_sessions": 30_000, "offset": 0,      "n_actions": 2, "interactions": False},
    "interactions": {"discount_factor": 0.95, "max_sessions": 30_000, "offset": 0,      "n_actions": 3, "interactions": True},
}

POS_MIDPOINTS   = np.array([4.5, 12.5, 20.5, 28.5, 36.5]) / 40.0
PRICE_VALUES    = np.array([0.0, -0.5, -1.0])   # 0=cheap 1=mid 2=expensive


def encode_state(pos_bin, price_bin, quality_bin):
    return pos_bin * (PRICE_BINS * QUALITY_BINS) + price_bin * QUALITY_BINS + quality_bin


def decode_state(s):
    qb = s % QUALITY_BINS
    rem = s // QUALITY_BINS
    pb = rem % PRICE_BINS
    pos = rem // PRICE_BINS
    return pos, pb, qb


# ---------------------------------------------------------------------------
# Load full dataset once
# ---------------------------------------------------------------------------

def load_all_sessions() -> tuple[pl.DataFrame, list]:
    print("Loading Expedia train.csv (once)...")
    df = (
        pl.scan_csv(str(DATA_DIR / "train.csv"), null_values=["NULL"])
        .select(NEEDED_COLS)
        .collect()
    )
    print(f"  Loaded {df.shape[0]:,} rows")

    # Per-session price normalization
    df = df.with_columns([
        pl.col("price_usd").mean().over("srch_id").alias("price_mean"),
        pl.col("price_usd").std().over("srch_id").alias("price_std"),
    ])
    df = df.with_columns(
        ((pl.col("price_usd") - pl.col("price_mean")) /
         (pl.col("price_std") + 1e-8)).alias("price_norm")
    )

    # Quality
    df = df.with_columns(
        pl.col("prop_review_score").fill_null(
            pl.col("prop_review_score").median().over("srch_id")
        ).alias("review_filled")
    )
    df = df.with_columns(
        (pl.col("prop_starrating") / 5.0 + pl.col("review_filled") / 5.0).alias("quality_raw")
    )
    df = df.with_columns(
        pl.col("quality_raw").median().over("srch_id").alias("quality_median")
    )
    df = df.with_columns(
        (pl.col("quality_raw") >= pl.col("quality_median")).cast(pl.Int32).alias("quality_bin")
    )

    # Price bin
    df = df.with_columns([
        pl.col("price_norm").rank(method="ordinal").over("srch_id").alias("price_rank"),
        pl.col("srch_id").count().over("srch_id").alias("session_size"),
    ])
    df = df.with_columns(
        pl.when(pl.col("price_rank") <= pl.col("session_size") / 3).then(0)
          .when(pl.col("price_rank") <= 2 * pl.col("session_size") / 3).then(1)
          .otherwise(2).cast(pl.Int32).alias("price_bin")
    )

    # Position bin
    df = df.with_columns(
        pl.when(pl.col("position") < 9).then(0)
          .when(pl.col("position") < 17).then(1)
          .when(pl.col("position") < 25).then(2)
          .when(pl.col("position") < 33).then(3)
          .otherwise(4).cast(pl.Int32).alias("pos_bin")
    )

    df = df.with_columns(
        (pl.col("pos_bin") * (PRICE_BINS * QUALITY_BINS)
         + pl.col("price_bin") * QUALITY_BINS
         + pl.col("quality_bin")).alias("state")
    )

    # Fill any nulls in state (can arise from null starrating/review propagating through quality_raw)
    df = df.with_columns(pl.col("state").fill_null(0))
    df = df.sort(["srch_id", "position"])
    df = df.with_columns(
        pl.col("state").shift(-1).over("srch_id").fill_null(0).alias("next_state")
    )

    all_session_ids = df["srch_id"].unique().sort().to_list()
    print(f"  Total sessions: {len(all_session_ids):,}")
    return df, all_session_ids


# ---------------------------------------------------------------------------
# Build Panel from pre-loaded df
# ---------------------------------------------------------------------------

def build_panel(df: pl.DataFrame, all_session_ids: list,
                max_sessions: int, offset: int, n_actions: int) -> Panel:
    window = all_session_ids[offset: offset + max_sessions]
    sub = df.filter(pl.col("srch_id").is_in(window))

    if n_actions == 3:
        sub = sub.with_columns(
            pl.when(pl.col("booking_bool") == 1).then(2)
              .when(pl.col("click_bool") == 1).then(1)
              .otherwise(0).cast(pl.Int32).alias("action")
        )
    else:
        sub = sub.with_columns(
            pl.when(
                (pl.col("click_bool") == 1) | (pl.col("booking_bool") == 1)
            ).then(1).otherwise(0).cast(pl.Int32).alias("action")
        )

    grouped = (
        sub
        .group_by("srch_id", maintain_order=True)
        .agg([
            pl.col("state").alias("states"),
            pl.col("action").alias("actions"),
            pl.col("next_state").alias("next_states"),
        ])
    )

    terminal = 2 if n_actions == 3 else 1
    trajectories = []
    for row in grouped.iter_rows(named=True):
        states = np.array(row["states"], dtype=np.int32)
        actions = np.array(row["actions"], dtype=np.int32)
        next_states = np.array(row["next_states"], dtype=np.int32)
        term_idx = np.where(actions == terminal)[0]
        if len(term_idx) > 0:
            end = term_idx[0] + 1
            states, actions, next_states = states[:end], actions[:end], next_states[:end]
        if len(states) >= 2:
            trajectories.append(Trajectory(
                states=jnp.array(states),
                actions=jnp.array(actions),
                next_states=jnp.array(next_states),
                individual_id=int(row["srch_id"]),
            ))

    return Panel(trajectories=trajectories)


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_features(n_actions: int, interactions: bool) -> tuple[np.ndarray, list[str]]:
    if n_actions == 2:
        param_names = ["theta_position", "theta_price", "theta_quality", "engage_value"]
        features = np.zeros((NUM_STATES, 2, 4))
        for s in range(NUM_STATES):
            pos, pb, qb = decode_state(s)
            features[s, 0, 0] = POS_MIDPOINTS[pos]
            features[s, 1, 1] = PRICE_VALUES[pb]
            features[s, 1, 2] = float(qb)
            features[s, 1, 3] = 1.0
        return features, param_names

    if not interactions:
        param_names = ["theta_position", "theta_price", "theta_quality",
                       "click_cost", "book_value"]
        features = np.zeros((NUM_STATES, 3, 5))
        for s in range(NUM_STATES):
            pos, pb, qb = decode_state(s)
            features[s, 0, 0] = POS_MIDPOINTS[pos]
            for a in [1, 2]:
                features[s, a, 1] = PRICE_VALUES[pb]
                features[s, a, 2] = float(qb)
            features[s, 1, 3] = 1.0
            features[s, 2, 4] = 1.0
        return features, param_names

    # Interactions: add position×quality and position×price on click+book
    # position×quality: does quality matter more at early positions?
    # position×price: does price sensitivity increase with fatigue?
    param_names = ["theta_position", "theta_price", "theta_quality",
                   "click_cost", "book_value",
                   "pos_x_quality", "pos_x_price"]
    features = np.zeros((NUM_STATES, 3, 7))
    for s in range(NUM_STATES):
        pos, pb, qb = decode_state(s)
        pm = POS_MIDPOINTS[pos]
        pv = PRICE_VALUES[pb]
        features[s, 0, 0] = pm
        for a in [1, 2]:
            features[s, a, 1] = pv
            features[s, a, 2] = float(qb)
            features[s, a, 5] = pm * float(qb)   # position × quality
            features[s, a, 6] = pm * abs(pv)      # position × |price| (fatigue × price)
        features[s, 1, 3] = 1.0
        features[s, 2, 4] = 1.0
    return features, param_names


# ---------------------------------------------------------------------------
# Transition estimation
# ---------------------------------------------------------------------------

def estimate_transitions(panel: Panel, n_actions: int) -> np.ndarray:
    counts = [np.zeros((NUM_STATES, NUM_STATES)) for _ in range(n_actions)]
    for traj in panel.trajectories:
        sa, aa, na = np.array(traj.states), np.array(traj.actions), np.array(traj.next_states)
        for t in range(len(sa) - 1):
            s, sp, a = int(sa[t]), int(na[t]), int(aa[t])
            if 0 <= s < NUM_STATES and 0 <= sp < NUM_STATES and 0 <= a < n_actions:
                counts[a][s, sp] += 1
    transitions = np.zeros((n_actions, NUM_STATES, NUM_STATES))
    for a in range(n_actions):
        rs = counts[a].sum(axis=1, keepdims=True)
        transitions[a] = np.where(rs > 0, counts[a] / np.maximum(rs, 1), np.eye(NUM_STATES))
    terminal = n_actions - 1
    transitions[terminal] = np.zeros((NUM_STATES, NUM_STATES))
    transitions[terminal, :, 0] = 1.0
    return transitions


# ---------------------------------------------------------------------------
# Run one spec
# ---------------------------------------------------------------------------

def run_spec(spec_name: str, cfg: dict,
             df: pl.DataFrame, all_session_ids: list) -> dict:
    print(f"\n{'='*60}")
    print(f"Spec: {spec_name}  |  beta={cfg['discount_factor']}, "
          f"sessions={cfg['max_sessions']:,}, offset={cfg['offset']}, "
          f"actions={cfg['n_actions']}, interactions={cfg['interactions']}")

    t0 = time.time()
    panel = build_panel(df, all_session_ids, cfg["max_sessions"], cfg["offset"], cfg["n_actions"])
    n_obs = sum(len(t.states) for t in panel.trajectories)
    print(f"  Panel: {len(panel.trajectories):,} sessions, {n_obs:,} obs")

    features, param_names = build_features(cfg["n_actions"], cfg["interactions"])

    F = features.reshape(-1, features.shape[-1])
    rank = np.linalg.matrix_rank(F)
    n_feat = features.shape[-1]
    if rank < n_feat:
        print(f"  WARNING: rank {rank} < {n_feat}")
    else:
        cond = np.linalg.cond(F[F.any(axis=1)])
        print(f"  Rank {rank}/{n_feat}, cond={cond:.1f}")

    transitions = estimate_transitions(panel, cfg["n_actions"])
    utility = LinearUtility(feature_matrix=features, parameter_names=param_names)
    problem = DDCProblem(num_states=NUM_STATES, num_actions=cfg["n_actions"],
                         discount_factor=cfg["discount_factor"])

    npl = CCPEstimator(num_policy_iterations=3, compute_hessian=True, verbose=False)
    result = npl.estimate(panel, utility, problem, transitions)
    elapsed = time.time() - t0

    params = {n: float(result.parameters[i]) for i, n in enumerate(param_names)}
    ses = {}
    if result.standard_errors is not None:
        ses = {n: float(result.standard_errors[i]) for i, n in enumerate(param_names)}

    print(f"  LL={result.log_likelihood:.1f}  time={elapsed:.0f}s")
    for name in param_names:
        se_str = f" ± {ses[name]:.4f}" if name in ses else ""
        print(f"  {name:20s} = {params[name]:+.4f}{se_str}")

    return {
        "spec": spec_name, "config": cfg,
        "n_sessions": len(panel.trajectories), "n_obs": n_obs,
        "log_likelihood": float(result.log_likelihood),
        "elapsed_s": round(elapsed, 1),
        "parameters": params, "standard_errors": ses,
        "param_names": param_names,
    }


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_table(results: dict):
    core = ["theta_position", "theta_price", "theta_quality"]
    interaction_params = ["pos_x_quality", "pos_x_price"]
    specs = list(results.keys())
    w = 13

    def row(param):
        line = f"{param:22s}"
        for s in specs:
            val = results[s]["parameters"].get(param, float("nan"))
            line += f"{val:>{w}.4f}"
        return line

    header = f"{'param':22s}" + "".join(f"{s:>{w}s}" for s in specs)
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("Expedia Robustness — parameter comparison")
    print(sep)
    print(header)
    print("-" * len(header))
    for p in core:
        print(row(p))
    print("-" * len(header))
    print("Interactions (last spec only):")
    for p in interaction_params:
        print(row(p))
    print("-" * len(header))
    print(f"{'log_likelihood':22s}" + "".join(
        f"{results[s]['log_likelihood']:>{w}.1f}" for s in specs))
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 72)
    print("Expedia Hotel Search -- Robustness Checks")
    print("=" * 72)

    df, all_session_ids = load_all_sessions()

    all_results = {}
    for spec_name, cfg in SPECS.items():
        all_results[spec_name] = run_spec(spec_name, cfg, df, all_session_ids)

    print_table(all_results)

    out_path = Path("examples/expedia-search/robustness_results.json")
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
