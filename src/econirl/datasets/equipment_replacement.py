"""
Equipment Replacement Variants Dataset.

This module provides synthetic datasets for equipment replacement problems,
with multiple variants to test estimators on different action/state configurations.

Variants:
- "binary": 2 actions (keep, replace) - similar to Rust (1987)
- "ternary": 3 actions (keep, minor_repair, major_repair)
- "continuous_state": More states (200 instead of 90)

These variants are useful for testing estimator robustness across different
problem configurations.
"""

import numpy as np
import pandas as pd


def load_equipment_replacement(
    variant: str = "binary",
    n_machines: int = 100,
    n_periods: int = 100,
    as_panel: bool = False,
    seed: int = 1987,
) -> pd.DataFrame:
    """
    Load synthetic equipment replacement data with configurable variants.

    This dataset represents machines making maintenance/replacement decisions
    over time. The state represents equipment wear level, and actions vary
    by variant (binary replacement, ternary with repairs, or continuous state).

    Args:
        variant: Problem variant to generate:
            - "binary": 2 actions (keep, replace), 90 states - like Rust
            - "ternary": 3 actions (keep, minor_repair, major_repair), 90 states
            - "continuous_state": 2 actions (keep, replace), 200 states
        n_machines: Number of machines to simulate (default: 100)
        n_periods: Number of time periods per machine (default: 100)
        as_panel: If True, return data structured as a Panel object
            compatible with econirl estimators. If False (default),
            return as a pandas DataFrame.
        seed: Random seed for reproducibility (default: 1987)

    Returns:
        DataFrame with columns:
            - id: Machine identifier
            - period: Time period (0-indexed)
            - state: Discretized wear state index
            - action: Chosen action (varies by variant)
            - wear_level: Continuous wear level (underlying state)
            - variant: The variant name used to generate this data

    Raises:
        ValueError: If variant is not one of "binary", "ternary", "continuous_state"

    Example:
        >>> from econirl.datasets import load_equipment_replacement
        >>> df = load_equipment_replacement(variant="binary")
        >>> print(f"Observations: {len(df):,}")
        >>> print(f"Machines: {df['id'].nunique()}")
        >>> print(f"States: {df['state'].nunique()}")

        >>> # Test with ternary actions
        >>> df_ternary = load_equipment_replacement(variant="ternary")
        >>> print(f"Actions: {df_ternary['action'].unique()}")

        >>> # Get as Panel for estimation
        >>> panel = load_equipment_replacement(as_panel=True)
        >>> print(f"Panel with {panel.num_individuals} machines")

    Notes:
        Action interpretation by variant:
        - binary: 0=keep, 1=replace
        - ternary: 0=keep, 1=minor_repair, 2=major_repair
        - continuous_state: 0=keep, 1=replace (but with finer state grid)

        State encoding:
        - binary: 90 states (wear bins, like Rust's mileage bins)
        - ternary: 90 states with different transition dynamics
        - continuous_state: 200 states for finer granularity
    """
    valid_variants = {"binary", "ternary", "continuous_state"}
    if variant not in valid_variants:
        raise ValueError(
            f"variant must be one of {valid_variants}, got '{variant}'"
        )

    df = _generate_equipment_replacement_data(
        variant=variant,
        n_machines=n_machines,
        n_periods=n_periods,
        seed=seed,
    )

    if as_panel:
        from econirl.core.types import Panel, Trajectory
        import torch

        # Convert to Panel format
        machine_ids = df["id"].unique()
        trajectories = []

        for machine_id in machine_ids:
            machine_data = df[df["id"] == machine_id].sort_values("period")
            states = torch.tensor(machine_data["state"].values, dtype=torch.long)
            actions = torch.tensor(machine_data["action"].values, dtype=torch.long)
            # Compute next_states (shift states by 1, use 0 for last period)
            next_states = torch.cat([states[1:], torch.tensor([0])])

            traj = Trajectory(
                states=states,
                actions=actions,
                next_states=next_states,
                individual_id=int(machine_id),
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    return df


def _generate_equipment_replacement_data(
    variant: str,
    n_machines: int,
    n_periods: int,
    seed: int,
) -> pd.DataFrame:
    """
    Generate synthetic equipment replacement data.

    Creates a dataset with realistic choice patterns based on a dynamic
    discrete choice model of equipment maintenance decisions.
    """
    np.random.seed(seed)

    # Get variant-specific configuration
    config = _get_variant_config(variant)
    num_states = config["num_states"]
    num_actions = config["num_actions"]

    records = []

    for machine_id in range(n_machines):
        # Initial state: new equipment with no wear
        wear_state = 0
        wear_level = 0.0

        for period in range(n_periods):
            # Compute choice probabilities based on variant
            logits = _compute_action_logits(wear_state, variant, config)

            # Convert to probabilities using softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            # Draw action
            action = np.random.choice(num_actions, p=probs)

            # Record observation
            records.append({
                "id": machine_id,
                "period": period,
                "state": wear_state,
                "action": action,
                "wear_level": wear_level,
                "variant": variant,
            })

            # State transition based on action and variant
            wear_state, wear_level = _transition(
                wear_state, wear_level, action, variant, config
            )

    return pd.DataFrame(records)


def _get_variant_config(variant: str) -> dict:
    """Get configuration parameters for each variant."""
    if variant == "binary":
        return {
            "num_states": 90,
            "num_actions": 2,
            "action_names": {0: "keep", 1: "replace"},
            # Cost parameters (in utility units)
            "theta_c": 0.001,  # Operating cost per state
            "RC": 3.0,  # Replacement cost
            # Transition probabilities (stay, +1, +2 bins)
            "p_transition": np.array([0.3919, 0.5953, 0.0128]),
            "wear_per_bin": 5.0,  # Wear units per state bin
        }
    elif variant == "ternary":
        return {
            "num_states": 90,
            "num_actions": 3,
            "action_names": {0: "keep", 1: "minor_repair", 2: "major_repair"},
            # Cost parameters
            "theta_c": 0.001,
            "RC_minor": 1.0,  # Minor repair cost
            "RC_major": 3.0,  # Major repair cost (full replacement)
            # Transition probabilities
            "p_transition": np.array([0.3919, 0.5953, 0.0128]),
            "wear_per_bin": 5.0,
        }
    else:  # continuous_state
        return {
            "num_states": 200,
            "num_actions": 2,
            "action_names": {0: "keep", 1: "replace"},
            "theta_c": 0.0005,  # Lower cost per state (more states)
            "RC": 3.0,
            # Finer transition probabilities
            "p_transition": np.array([0.25, 0.45, 0.20, 0.08, 0.02]),
            "wear_per_bin": 2.25,  # Finer granularity
        }


def _compute_action_logits(
    wear_state: int,
    variant: str,
    config: dict,
) -> np.ndarray:
    """
    Compute action logits based on current wear state and variant.

    Returns logits for each action in the variant's action space.
    """
    num_actions = config["num_actions"]
    logits = np.zeros(num_actions)

    if variant == "binary":
        # Binary: keep vs replace
        # V(keep) ~ -theta_c * wear_state
        # V(replace) ~ -RC
        logits[0] = -config["theta_c"] * wear_state  # keep
        logits[1] = -config["RC"]  # replace

    elif variant == "ternary":
        # Ternary: keep vs minor_repair vs major_repair
        theta_c = config["theta_c"]
        # V(keep) ~ -theta_c * wear_state
        logits[0] = -theta_c * wear_state  # keep
        # V(minor_repair) ~ -RC_minor - theta_c * (wear_state/2)
        # Minor repair reduces wear but doesn't eliminate it
        logits[1] = -config["RC_minor"] - theta_c * (wear_state / 2)  # minor_repair
        # V(major_repair) ~ -RC_major (like full replacement)
        logits[2] = -config["RC_major"]  # major_repair

    else:  # continuous_state
        # Same as binary, but with finer state granularity
        logits[0] = -config["theta_c"] * wear_state  # keep
        logits[1] = -config["RC"]  # replace

    return logits


def _transition(
    wear_state: int,
    wear_level: float,
    action: int,
    variant: str,
    config: dict,
) -> tuple[int, float]:
    """
    Compute state transition based on action.

    Returns the new (wear_state, wear_level) tuple.
    """
    num_states = config["num_states"]
    p_transition = config["p_transition"]
    wear_per_bin = config["wear_per_bin"]

    if variant == "binary" or variant == "continuous_state":
        if action == 1:  # Replace
            return 0, 0.0
        else:  # Keep
            # Stochastic wear increment
            increment = np.random.choice(len(p_transition), p=p_transition)
            new_state = min(wear_state + increment, num_states - 1)
            new_level = wear_level + increment * wear_per_bin
            return new_state, new_level

    else:  # ternary
        if action == 2:  # Major repair (full replacement)
            return 0, 0.0
        elif action == 1:  # Minor repair (reduce wear by half)
            new_state = max(0, wear_state // 2)
            new_level = max(0.0, wear_level / 2)
            return new_state, new_level
        else:  # Keep
            increment = np.random.choice(len(p_transition), p=p_transition)
            new_state = min(wear_state + increment, num_states - 1)
            new_level = wear_level + increment * wear_per_bin
            return new_state, new_level


def get_equipment_replacement_info(variant: str = "binary") -> dict:
    """
    Get metadata about the equipment replacement dataset.

    Args:
        variant: Which variant to get info for ("binary", "ternary", "continuous_state")

    Returns:
        Dictionary with dataset information including number of states,
        actions, and description of the state/action spaces.
    """
    config = _get_variant_config(variant)

    return {
        "name": f"Equipment Replacement ({variant})",
        "variant": variant,
        "num_states": config["num_states"],
        "num_actions": config["num_actions"],
        "action_names": config["action_names"],
        "state_description": "Discretized equipment wear level",
        "description": {
            "binary": "Binary replacement decision (keep/replace), 90 states - similar to Rust (1987)",
            "ternary": "Three maintenance options (keep/minor_repair/major_repair), 90 states",
            "continuous_state": "Binary replacement with finer state granularity, 200 states",
        }[variant],
    }
