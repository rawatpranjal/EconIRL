#!/usr/bin/env python3
"""
KKBOX Churn -- Robustness Checks
==================================

Data loaded once; specs subsample from the pre-loaded dataframe.
Specs vary: discount factor, sample size, random seed, action framing,
and interaction terms. Uses NPL K=3 for speed.

Specs:
    baseline      -- beta=0.95, 20K users, seed=42
    beta_090      -- beta=0.90
    beta_099      -- beta=0.99
    window_5k     -- 5K users
    window_50k    -- 50K users (all available)
    subsample_b   -- 20K users, seed=99
    subsample_c   -- 20K users, seed=123
    action_3      -- 3 actions: renew / pause (auto_renew flip) / cancel
    interactions  -- adds tenure×price, tenure×auto_renew, discount×auto_renew

Usage:
    python examples/kkbox-churn/robustness.py
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

DATA_DIR = Path("data/raw/kkbox/raw")
MIN_TRANSACTIONS = 3

TENURE_LABELS = 6
PRICE_LABELS = 3
AUTO_RENEW_LEVELS = 2
NUM_STATES = TENURE_LABELS * PRICE_LABELS * AUTO_RENEW_LEVELS  # 36

SPECS = {
    "baseline":     {"discount_factor": 0.95, "max_users": 20_000, "seed": 42,  "n_actions": 2, "interactions": False},
    "beta_090":     {"discount_factor": 0.90, "max_users": 20_000, "seed": 42,  "n_actions": 2, "interactions": False},
    "beta_099":     {"discount_factor": 0.99, "max_users": 20_000, "seed": 42,  "n_actions": 2, "interactions": False},
    "window_5k":    {"discount_factor": 0.95, "max_users":  5_000, "seed": 42,  "n_actions": 2, "interactions": False},
    "window_50k":   {"discount_factor": 0.95, "max_users": 50_000, "seed": 42,  "n_actions": 2, "interactions": False},
    "subsample_b":  {"discount_factor": 0.95, "max_users": 20_000, "seed": 99,  "n_actions": 2, "interactions": False},
    "subsample_c":  {"discount_factor": 0.95, "max_users": 20_000, "seed": 123, "n_actions": 2, "interactions": False},
    "action_3":     {"discount_factor": 0.95, "max_users": 20_000, "seed": 42,  "n_actions": 3, "interactions": False},
    "interactions": {"discount_factor": 0.95, "max_users": 20_000, "seed": 42,  "n_actions": 2, "interactions": True},
}

TENURE_MIDPOINTS = np.array([1, 2, 3.5, 6.5, 12, 25]) / 25.0
PRICE_MIDPOINTS  = np.array([50, 125, 200]) / 200.0


def encode_state(t_bin, p_tier, ar):
    return t_bin * (PRICE_LABELS * AUTO_RENEW_LEVELS) + p_tier * AUTO_RENEW_LEVELS + ar


def decode_state(s):
    ar = s % AUTO_RENEW_LEVELS
    rem = s // AUTO_RENEW_LEVELS
    p_tier = rem % PRICE_LABELS
    t_bin = rem // PRICE_LABELS
    return t_bin, p_tier, ar


# ---------------------------------------------------------------------------
# Load full dataset once
# ---------------------------------------------------------------------------

def load_all_users() -> pl.DataFrame:
    """Scan transactions, filter to users with enough history, return sorted df."""
    print("Loading KKBOX transactions (once)...")
    txn_lazy = pl.scan_csv(str(DATA_DIR / "transactions.csv"))

    user_counts = (
        txn_lazy
        .group_by("msno")
        .agg(pl.len().alias("n_txn"))
        .filter(pl.col("n_txn") >= MIN_TRANSACTIONS)
        .collect()
    )
    all_users = set(user_counts["msno"].to_list())
    print(f"  Eligible users: {len(all_users):,}")

    txn = (
        txn_lazy
        .filter(pl.col("msno").is_in(list(all_users)))
        .collect()
        .sort(["msno", "transaction_date"])
    )

    txn = txn.with_columns(
        pl.col("msno").cum_count().over("msno").alias("tenure"),
        (pl.col("actual_amount_paid") < pl.col("plan_list_price")).cast(pl.Int32).alias("discount"),
        pl.col("is_auto_renew").cast(pl.Int32).alias("auto_renew_int"),
    )
    txn = txn.with_columns([
        pl.when(pl.col("tenure") < 2).then(0)
          .when(pl.col("tenure") < 3).then(1)
          .when(pl.col("tenure") < 5).then(2)
          .when(pl.col("tenure") < 9).then(3)
          .when(pl.col("tenure") < 17).then(4)
          .otherwise(5).cast(pl.Int32).alias("tenure_bin"),
        pl.when(pl.col("plan_list_price") < 100).then(0)
          .when(pl.col("plan_list_price") < 150).then(1)
          .otherwise(2).cast(pl.Int32).alias("price_tier"),
    ])
    txn = txn.with_columns(
        (pl.col("tenure_bin") * (PRICE_LABELS * AUTO_RENEW_LEVELS)
         + pl.col("price_tier") * AUTO_RENEW_LEVELS
         + pl.col("auto_renew_int")).alias("state")
    )
    # Pre-compute next_auto_renew for 3-action spec
    txn = txn.with_columns(
        pl.col("auto_renew_int").shift(-1).over("msno").alias("next_auto_renew"),
        pl.col("state").shift(-1).over("msno").fill_null(0).alias("next_state"),
    )
    print(f"  Total transactions: {len(txn):,}")
    return txn, list(user_counts["msno"].to_list())


# ---------------------------------------------------------------------------
# Build Panel from pre-loaded df
# ---------------------------------------------------------------------------

def build_panel(txn: pl.DataFrame, all_user_ids: list,
                max_users: int, seed: int, n_actions: int) -> Panel:
    rng = np.random.default_rng(seed)
    if len(all_user_ids) > max_users:
        chosen = rng.choice(all_user_ids, size=max_users, replace=False).tolist()
    else:
        chosen = all_user_ids

    sub = txn.filter(pl.col("msno").is_in(chosen))

    if n_actions == 2:
        sub = sub.with_columns(pl.col("is_cancel").cast(pl.Int32).alias("action"))
    else:
        sub = sub.with_columns(
            pl.when(pl.col("is_cancel") == 1).then(2)
              .when(
                  (pl.col("is_cancel") == 0) &
                  (pl.col("next_auto_renew").is_not_null()) &
                  (pl.col("auto_renew_int") != pl.col("next_auto_renew"))
              ).then(1)
              .otherwise(0)
              .cast(pl.Int32).alias("action")
        )

    grouped = (
        sub
        .group_by("msno", maintain_order=True)
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
                individual_id=hash(row["msno"]) % (2**31),
            ))

    return Panel(trajectories=trajectories)


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_features(n_actions: int, interactions: bool) -> tuple[np.ndarray, list[str]]:
    if n_actions == 3:
        # 3-action: renew=0, pause=1, cancel=2
        param_names = ["theta_tenure", "theta_price", "theta_auto_renew",
                       "theta_discount", "pause_cost", "cancel_cost"]
        features = np.zeros((NUM_STATES, 3, 6))
        for s in range(NUM_STATES):
            t_bin, p_tier, ar = decode_state(s)
            features[s, 0, 0] = TENURE_MIDPOINTS[t_bin]
            features[s, 0, 1] = PRICE_MIDPOINTS[p_tier]
            features[s, 0, 2] = float(ar)
            features[s, 0, 3] = 1.0 if p_tier == 0 else 0.0
            features[s, 1, 4] = 1.0
            features[s, 2, 5] = 1.0
        return features, param_names

    if not interactions:
        # Baseline 2-action
        param_names = ["theta_tenure", "theta_price", "theta_auto_renew",
                       "theta_discount", "constant"]
        features = np.zeros((NUM_STATES, 2, 5))
        for s in range(NUM_STATES):
            t_bin, p_tier, ar = decode_state(s)
            features[s, 0, 0] = TENURE_MIDPOINTS[t_bin]
            features[s, 0, 1] = PRICE_MIDPOINTS[p_tier]
            features[s, 0, 2] = float(ar)
            features[s, 0, 3] = 1.0 if p_tier == 0 else 0.0
            features[s, 1, 4] = 1.0
        return features, param_names

    # Interactions: baseline + 3 cross-terms on the renew action
    # tenure×price: does loyalty reduce price sensitivity?
    # tenure×auto_renew: does inertia compound with tenure?
    # discount×auto_renew: are discounts less effective for auto-renew users?
    param_names = ["theta_tenure", "theta_price", "theta_auto_renew", "theta_discount",
                   "constant",
                   "tenure_x_price", "tenure_x_ar", "discount_x_ar"]
    features = np.zeros((NUM_STATES, 2, 8))
    for s in range(NUM_STATES):
        t_bin, p_tier, ar = decode_state(s)
        tm = TENURE_MIDPOINTS[t_bin]
        pm = PRICE_MIDPOINTS[p_tier]
        disc = 1.0 if p_tier == 0 else 0.0
        # Main effects on renew (a=0)
        features[s, 0, 0] = tm
        features[s, 0, 1] = pm
        features[s, 0, 2] = float(ar)
        features[s, 0, 3] = disc
        # Cancel constant
        features[s, 1, 4] = 1.0
        # Interactions on renew (a=0)
        features[s, 0, 5] = tm * pm          # tenure × price
        features[s, 0, 6] = tm * float(ar)   # tenure × auto_renew
        features[s, 0, 7] = disc * float(ar) # discount × auto_renew
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
        transitions[a] = np.where(rs > 0, counts[a] / np.maximum(rs, 1),
                                  np.ones((NUM_STATES, NUM_STATES)) / NUM_STATES)
    terminal = n_actions - 1
    transitions[terminal] = np.zeros((NUM_STATES, NUM_STATES))
    transitions[terminal, :, 0] = 1.0
    return transitions


# ---------------------------------------------------------------------------
# Run one spec
# ---------------------------------------------------------------------------

def run_spec(spec_name: str, cfg: dict,
             txn: pl.DataFrame, all_user_ids: list) -> dict:
    print(f"\n{'='*60}")
    print(f"Spec: {spec_name}  |  beta={cfg['discount_factor']}, "
          f"N={cfg['max_users']:,}, seed={cfg['seed']}, "
          f"actions={cfg['n_actions']}, interactions={cfg['interactions']}")

    t0 = time.time()
    panel = build_panel(txn, all_user_ids, cfg["max_users"], cfg["seed"], cfg["n_actions"])
    n_obs = sum(len(t.states) for t in panel.trajectories)
    print(f"  Panel: {len(panel.trajectories):,} users, {n_obs:,} obs")

    features, param_names = build_features(cfg["n_actions"], cfg["interactions"])

    F = features.reshape(-1, features.shape[-1])
    rank = np.linalg.matrix_rank(F)
    n_feat = features.shape[-1]
    if rank < n_feat:
        print(f"  WARNING: rank {rank} < {n_feat} — collinear features")
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
        print(f"  {name:25s} = {params[name]:+.4f}{se_str}")

    return {
        "spec": spec_name, "config": cfg,
        "n_users": len(panel.trajectories), "n_obs": n_obs,
        "log_likelihood": float(result.log_likelihood),
        "elapsed_s": round(elapsed, 1),
        "parameters": params, "standard_errors": ses,
        "param_names": param_names,
    }


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_table(results: dict):
    core = ["theta_tenure", "theta_price", "theta_auto_renew", "theta_discount"]
    interaction_params = ["tenure_x_price", "tenure_x_ar", "discount_x_ar"]
    specs = list(results.keys())
    w = 14

    def row(param):
        line = f"{param:25s}"
        for s in specs:
            val = results[s]["parameters"].get(param, float("nan"))
            line += f"{val:>{w}.4f}"
        return line

    header = f"{'param':25s}" + "".join(f"{s:>{w}s}" for s in specs)
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("KKBOX Robustness — parameter comparison")
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
    print(f"{'log_likelihood':25s}" + "".join(
        f"{results[s]['log_likelihood']:>{w}.1f}" for s in specs))
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 72)
    print("KKBOX Churn -- Robustness Checks")
    print("=" * 72)

    txn, all_user_ids = load_all_users()

    all_results = {}
    for spec_name, cfg in SPECS.items():
        all_results[spec_name] = run_spec(spec_name, cfg, txn, all_user_ids)

    print_table(all_results)

    out_path = Path("examples/kkbox-churn/robustness_results.json")
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
