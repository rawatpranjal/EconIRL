"""Ground truth definitions and validation for AutoLab experiments.

Defines per-estimator, per-difficulty success criteria and provides
evaluation functions to check experiment results against thresholds.
"""

# ---------------------------------------------------------------------------
# Ground truth table: estimator -> difficulty -> criteria thresholds
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    # --- Tier 1: Must be perfect on toy, near-perfect on standard ---
    "NFXP": {
        "toy":      {"pct_optimal": 99.5, "param_rmse": 0.15, "policy_rmse": 0.02, "converged": True},
        "standard": {"pct_optimal": 99.0, "param_rmse": 0.20, "policy_rmse": 0.05, "converged": True},
        "hard":     {"pct_optimal": 95.0, "param_rmse": 0.50},
    },
    "CCP": {
        "toy":      {"pct_optimal": 99.5, "param_rmse": 0.15, "policy_rmse": 0.02, "converged": True},
        "standard": {"pct_optimal": 99.0, "param_rmse": 0.20, "policy_rmse": 0.05, "converged": True},
        "hard":     {"pct_optimal": 95.0, "param_rmse": 0.50},
    },
    "MCE IRL": {
        "toy":      {"pct_optimal": 99.0, "param_rmse": 0.20, "converged": True},
        "standard": {"pct_optimal": 98.0, "param_rmse": 0.30},
        "hard":     {"pct_optimal": 90.0},
    },
    # --- Tier 2: Near-perfect on toy, solid on standard ---
    "Max Margin": {
        "toy":      {"pct_optimal": 95.0, "policy_rmse": 0.05},
        "standard": {"pct_optimal": 90.0, "policy_rmse": 0.10},
        "hard":     {"pct_optimal": 80.0},
    },
    "TD-CCP": {
        "toy":      {"pct_optimal": 98.0, "param_rmse": 0.25},
        "standard": {"pct_optimal": 95.0, "param_rmse": 0.40},
        "hard":     {"pct_optimal": 85.0},
    },
    "GLADIUS": {
        "toy":      {"pct_optimal": 98.0, "param_rmse": 0.25},
        "standard": {"pct_optimal": 95.0, "param_rmse": 0.40},
        "hard":     {"pct_optimal": 85.0},
    },
    # --- Tier 3-4: Partial recovery acceptable, but toy should work ---
    "AIRL": {
        "toy":      {"pct_optimal": 70.0},
        "standard": {"pct_optimal": 50.0},
        "hard":     {"pct_optimal": 30.0},
    },
    "MaxEnt IRL": {
        "toy":      {"pct_optimal": 60.0},
        "standard": {"pct_optimal": 40.0},
        "hard":     {"pct_optimal": 20.0},
    },
    "GAIL": {
        "toy":      {"pct_optimal": 50.0},
        "standard": {"pct_optimal": 30.0},
        "hard":     {"pct_optimal": 10.0},
    },
    "GCL": {
        "toy":      {"pct_optimal": 50.0},
        "standard": {"pct_optimal": 30.0},
        "hard":     {"pct_optimal": 10.0},
    },
}

# ---------------------------------------------------------------------------
# Difficulty presets (DGP parameters)
# ---------------------------------------------------------------------------

DIFFICULTY_DGPS = {
    "toy":      {"n_states": 5,  "discount_factor": 0.95, "n_agents": 100, "n_periods": 50},
    "standard": {"n_states": 20, "discount_factor": 0.99, "n_agents": 200, "n_periods": 100},
    "hard":     {"n_states": 50, "discount_factor": 0.99, "n_agents": 200, "n_periods": 100},
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def classify_difficulty(dgp_config: dict) -> str:
    """Classify a DGP config as 'toy', 'standard', or 'hard' by n_states."""
    n_states = dgp_config.get("n_states", 20)
    if n_states <= 10:
        return "toy"
    elif n_states <= 30:
        return "standard"
    else:
        return "hard"


def get_ground_truth(estimator: str, difficulty: str) -> dict:
    """Return criteria thresholds for an estimator at a given difficulty.

    Returns empty dict if estimator or difficulty is unknown.
    """
    return GROUND_TRUTH.get(estimator, {}).get(difficulty, {})


def evaluate_against_ground_truth(
    result: dict, estimator: str, difficulty: str
) -> dict:
    """Evaluate experiment results against ground truth criteria.

    Returns:
        {
            "criteria": {
                name: {"threshold": X, "actual": Y, "passed": bool},
                ...
            },
            "all_passed": bool,
        }
    """
    thresholds = get_ground_truth(estimator, difficulty)
    if not thresholds:
        return {"criteria": {}, "all_passed": True}

    criteria = {}
    for name, threshold in thresholds.items():
        actual = result.get(name)

        if name == "converged":
            # Boolean criterion
            passed = actual is True
            criteria[name] = {"threshold": threshold, "actual": actual, "passed": passed}
        elif actual is None:
            # Missing metric — cannot pass
            criteria[name] = {"threshold": threshold, "actual": None, "passed": False}
        elif name in ("pct_optimal",):
            # Higher is better — actual must be >= threshold
            passed = actual >= threshold
            criteria[name] = {"threshold": threshold, "actual": round(actual, 2), "passed": passed}
        else:
            # Lower is better (rmse metrics) — actual must be <= threshold
            passed = actual <= threshold
            criteria[name] = {"threshold": threshold, "actual": round(actual, 4), "passed": passed}

    all_passed = all(c["passed"] for c in criteria.values())
    return {"criteria": criteria, "all_passed": all_passed}
