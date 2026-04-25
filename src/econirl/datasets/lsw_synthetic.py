"""Lee-Sudhir-Wang serialized fiction reading dataset, semi-synthetic.

This module ships a documented semi-synthetic mirror of the empirical
setting in Lee, Sudhir, and Wang (2026), who study sequential reading
decisions on a serialized fiction platform. The original platform data
is not redistributable so the loader generates a panel from a
parametric data-generating process whose state space, action set, and
latent-type mixture mirror the source paper.

State representation (simplified to a finite integer index):
    chapter_index in {0, ..., n_chapters_per_book - 1}
    wait_bin     in {0, 1, 2, 3, 4}      (discretized days since release)
    pricing      in {0, 1}                (0 = discount window, 1 = full price)
    prev_paid    in {0, 1}                (1 if user paid for the previous chapter)

Action set:
    a = 0  pay-and-read
    a = 1  wait-and-read
    a = 2  exit the current book

Latent types (two dominant segments from Lee-Sudhir-Wang 2026):
    z = 0  high-patience, monetization-focused (mixture weight 0.4)
    z = 1  budget-conscious, patient            (mixture weight 0.6)

Identification anchor:
    Action a = 2 (exit) carries utility zero per type. Type-specific
    utilities for pay-and-read and wait-and-read are identified
    relative to exit. This mirrors the anchor assumption in the
    source paper.

Reference:
    Lee, Y.-J., Sudhir, K., and Wang, Y. (2026). "Adversarial Inverse
    Reinforcement Learning with Unobserved Heterogeneity in Sequential
    Content Consumption." Working paper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parametric data-generating process
# ---------------------------------------------------------------------------

# Per-type utility coefficients on (pay, wait, exit). Exit is the anchor
# at zero. The remaining two coefficients are intercepts per action per
# type. Calibrated below to yield a marginal purchase rate near six
# percent under the default mixture weight and price draws.
_DEFAULT_THETA = {
    0: {"alpha_pay": -1.7, "alpha_wait": -3.0},  # high-patience, pays
    1: {"alpha_pay": -5.0, "alpha_wait": -2.0},  # budget-conscious, waits
}

# Shared coefficients on observed state covariates.
_BETA_PRICE = 0.5  # additional disutility on pay when pricing == 1
_BETA_WAIT = 0.3   # additional disutility on wait per wait_bin

# Calibration tolerance for the marginal purchase rate. Generation
# raises if the simulated rate falls outside [target - tol, target + tol].
_PURCHASE_RATE_TARGET = 0.06
_PURCHASE_RATE_TOL = 0.03

# State-space dimensions.
_N_WAIT_BINS = 5
_N_PRICING = 2
_N_PREV_PAID = 2

# Action constants.
_A_PAY = 0
_A_WAIT = 1
_A_EXIT = 2


def _encode_state(
    chapter_index: int, wait_bin: int, pricing: int, prev_paid: int,
    n_chapters_per_book: int,
) -> int:
    """Pack the four state components into a single integer index."""
    state = (
        chapter_index * (_N_WAIT_BINS * _N_PRICING * _N_PREV_PAID)
        + wait_bin * (_N_PRICING * _N_PREV_PAID)
        + pricing * _N_PREV_PAID
        + prev_paid
    )
    return state


def _utility(
    action: int,
    pricing: int,
    wait_bin: int,
    type_z: int,
    theta: dict,
) -> float:
    """Per-action utility under type z and the current state covariates."""
    if action == _A_EXIT:
        return 0.0
    if action == _A_PAY:
        return theta[type_z]["alpha_pay"] - _BETA_PRICE * pricing
    if action == _A_WAIT:
        return theta[type_z]["alpha_wait"] - _BETA_WAIT * wait_bin
    raise ValueError(f"Unknown action: {action}")


def _generate_lsw_synthetic(
    n_users: int,
    n_books: int,
    n_chapters_per_book: int,
    pi_pay_segment: float,
    seed: int,
    theta: dict,
) -> tuple[pd.DataFrame, dict]:
    """Generate the bundled CSV and metadata for the dataset.

    Returns the trajectory DataFrame and a metadata dictionary that
    declares the latent type weights, the type-specific utility
    coefficients, and the calibrated marginal purchase rate.
    """
    rng = np.random.default_rng(seed)
    n_states = (
        n_chapters_per_book * _N_WAIT_BINS * _N_PRICING * _N_PREV_PAID
    )

    records = []
    for user_id in range(n_users):
        # Draw latent type once per user.
        type_z = int(rng.random() < (1.0 - pi_pay_segment))
        # type_z = 0 with probability pi_pay_segment (pay segment)
        # type_z = 1 otherwise (wait segment)

        # Draw a content embedding once per user (carried as features
        # but not part of the integer state encoding).
        embedding = rng.standard_normal(size=4).astype(np.float32)

        # Each user reads up to n_books books. They may exit early.
        for book_id in range(n_books):
            prev_paid = 0
            wait_bin = int(rng.integers(0, _N_WAIT_BINS))
            for chapter_index in range(n_chapters_per_book):
                pricing = int(rng.integers(0, _N_PRICING))
                state = _encode_state(
                    chapter_index, wait_bin, pricing, prev_paid,
                    n_chapters_per_book,
                )

                # Compute type-conditional choice probabilities.
                utilities = np.array(
                    [
                        _utility(_A_PAY, pricing, wait_bin, type_z, theta),
                        _utility(_A_WAIT, pricing, wait_bin, type_z, theta),
                        _utility(_A_EXIT, pricing, wait_bin, type_z, theta),
                    ]
                )
                shifted = utilities - np.max(utilities)
                probs = np.exp(shifted) / np.sum(np.exp(shifted))
                action = int(rng.choice(3, p=probs))

                # Update state covariates for the next period.
                if action == _A_EXIT:
                    next_chapter = 0
                    next_wait = 0
                    next_pricing = 0
                    next_prev_paid = 0
                else:
                    next_chapter = min(
                        chapter_index + 1, n_chapters_per_book - 1
                    )
                    next_wait = (
                        max(0, wait_bin - 1) if action == _A_WAIT
                        else int(rng.integers(0, _N_WAIT_BINS))
                    )
                    next_pricing = int(rng.integers(0, _N_PRICING))
                    next_prev_paid = 1 if action == _A_PAY else 0

                next_state = _encode_state(
                    next_chapter, next_wait, next_pricing, next_prev_paid,
                    n_chapters_per_book,
                )

                records.append(
                    {
                        "user_id": user_id,
                        "book_id": book_id,
                        "chapter_index": chapter_index,
                        "wait_bin": wait_bin,
                        "pricing": pricing,
                        "prev_paid": prev_paid,
                        "state": state,
                        "action": action,
                        "next_state": next_state,
                        "latent_type": type_z,
                        "embedding_0": float(embedding[0]),
                        "embedding_1": float(embedding[1]),
                        "embedding_2": float(embedding[2]),
                        "embedding_3": float(embedding[3]),
                    }
                )

                if action == _A_EXIT:
                    break

                wait_bin = next_wait
                pricing = next_pricing
                prev_paid = next_prev_paid

    df = pd.DataFrame(records)
    purchase_rate = float((df["action"] == _A_PAY).mean())

    if abs(purchase_rate - _PURCHASE_RATE_TARGET) > _PURCHASE_RATE_TOL:
        raise RuntimeError(
            f"Calibration check failed: simulated purchase rate "
            f"{purchase_rate:.4f} is outside the target window "
            f"[{_PURCHASE_RATE_TARGET - _PURCHASE_RATE_TOL:.4f}, "
            f"{_PURCHASE_RATE_TARGET + _PURCHASE_RATE_TOL:.4f}]. "
            "Adjust _DEFAULT_THETA in lsw_synthetic.py."
        )

    metadata = {
        "n_users": n_users,
        "n_books": n_books,
        "n_chapters_per_book": n_chapters_per_book,
        "pi_pay_segment": pi_pay_segment,
        "pi_wait_segment": 1.0 - pi_pay_segment,
        "n_states": n_states,
        "n_actions": 3,
        "action_labels": {"0": "pay", "1": "wait", "2": "exit"},
        "anchor_action": _A_EXIT,
        "type_theta": {
            "0": dict(theta[0]),
            "1": dict(theta[1]),
        },
        "beta_price": _BETA_PRICE,
        "beta_wait": _BETA_WAIT,
        "n_wait_bins": _N_WAIT_BINS,
        "n_pricing": _N_PRICING,
        "n_prev_paid": _N_PREV_PAID,
        "simulated_purchase_rate": purchase_rate,
        "purchase_rate_target": _PURCHASE_RATE_TARGET,
        "purchase_rate_tol": _PURCHASE_RATE_TOL,
        "seed": seed,
        "discount_factor": 0.95,
    }
    return df, metadata


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_lsw_synthetic(
    n_users: int = 5000,
    n_books: int = 50,
    n_chapters_per_book: int = 30,
    pi_pay_segment: float = 0.4,
    seed: int = 42,
    as_panel: bool = False,
) -> pd.DataFrame:
    """Load or generate the LSW serialized-content semi-synthetic panel.

    Args:
        n_users: Number of simulated users in the panel.
        n_books: Maximum number of books each user encounters before
            running out of content. Users may exit a book early.
        n_chapters_per_book: Number of chapters per book.
        pi_pay_segment: Population mixture weight on the high-patience
            pay-and-read latent type. Default 0.4 matches the source
            paper's reported relative segment shares.
        seed: Random seed for reproducibility.
        as_panel: If True, return a Panel object whose metadata field
            carries the data-generating process parameters. If False,
            return a pandas DataFrame.

    Returns:
        DataFrame with one row per chapter decision, including the
        integer state encoding, the chosen action, the next state,
        the latent type, and a four-dimensional content embedding.
        If `as_panel=True`, a Panel object whose metadata declares
        the type-specific reward coefficients.
    """
    if not 0.0 <= pi_pay_segment <= 1.0:
        raise ValueError(
            f"pi_pay_segment must be in [0, 1], got {pi_pay_segment}"
        )

    csv_path = Path(__file__).parent / "lsw_synthetic_data.csv"
    meta_path = Path(__file__).parent / "lsw_synthetic_metadata.json"

    use_cache = (
        csv_path.exists()
        and meta_path.exists()
        and n_users == 5000
        and n_books == 50
        and n_chapters_per_book == 30
        and pi_pay_segment == 0.4
        and seed == 42
    )

    if use_cache:
        df = pd.read_csv(csv_path)
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        df, metadata = _generate_lsw_synthetic(
            n_users=n_users,
            n_books=n_books,
            n_chapters_per_book=n_chapters_per_book,
            pi_pay_segment=pi_pay_segment,
            seed=seed,
            theta=_DEFAULT_THETA,
        )

    if as_panel:
        from econirl.core.types import Panel, Trajectory
        import jax.numpy as jnp

        trajectories = []
        # Each (user_id, book_id) pair is one trajectory.
        for (uid, bid), group in df.groupby(["user_id", "book_id"]):
            group = group.sort_values("chapter_index")
            states = jnp.array(group["state"].values, dtype=jnp.int32)
            actions = jnp.array(group["action"].values, dtype=jnp.int32)
            next_states = jnp.array(
                group["next_state"].values, dtype=jnp.int32
            )
            trajectories.append(
                Trajectory(
                    states=states,
                    actions=actions,
                    next_states=next_states,
                    individual_id=int(uid * 10_000 + bid),
                    metadata={
                        "user_id": int(uid),
                        "book_id": int(bid),
                        "latent_type": int(group["latent_type"].iloc[0]),
                    },
                )
            )

        return Panel(trajectories=trajectories, metadata=metadata)

    return df


def get_lsw_synthetic_info() -> dict:
    """Return metadata about the LSW serialized-content dataset.

    Returns the type-specific reward coefficients, the population
    mixture weights, the discount factor, and the action labels used
    by the AIRL-Het EM loop to validate recovered parameters.
    """
    meta_path = Path(__file__).parent / "lsw_synthetic_metadata.json"
    static = {
        "name": (
            "Lee-Sudhir-Wang serialized fiction reading "
            "(semi-synthetic mirror)"
        ),
        "n_actions": 3,
        "action_labels": {"0": "pay", "1": "wait", "2": "exit"},
        "anchor_action": _A_EXIT,
        "ground_truth": True,
        "use_case": (
            "AIRL-Het with two latent types under a calibrated "
            "marginal purchase rate"
        ),
        "reference": (
            "Lee, Y.-J., Sudhir, K., and Wang, Y. (2026). Adversarial "
            "Inverse Reinforcement Learning with Unobserved Heterogeneity "
            "in Sequential Content Consumption. Working paper."
        ),
    }
    if meta_path.exists():
        with open(meta_path) as f:
            dynamic = json.load(f)
        static.update(dynamic)
    return static
