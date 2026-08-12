#!/usr/bin/env python3
"""Verify TD-CCP pickle parity in a fresh Python process."""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from econirl import TDCCP
from econirl.core.reward_spec import RewardSpec
from econirl.environments import ArrayMDP
from econirl.simulation.synthetic import simulate_panel

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "tdccp_serialization.json"


def build_workflow() -> tuple[ArrayMDP, np.ndarray, RewardSpec, np.ndarray]:
    """Build a compact identified replacement problem."""
    n_states = 12
    transitions = np.zeros((2, n_states, n_states), dtype=np.float64)
    for state in range(n_states):
        transitions[0, state, state] = 0.65
        transitions[0, state, min(state + 1, n_states - 1)] += 0.35
    transitions[1, :, 0] = 1.0
    condition = np.linspace(0.0, 1.0, n_states)
    features = np.zeros((n_states, 2, 2), dtype=np.float64)
    features[:, 0, 0] = -condition
    features[:, 1, 1] = -1.0
    names = ["condition_cost", "replacement_cost"]
    reward = RewardSpec(features, names=names)
    state_features = np.column_stack([condition, condition**2])
    env = ArrayMDP(
        transitions,
        features,
        theta=np.array([1.5, 2.2]),
        discount_factor=0.95,
        scale_parameter=1.0,
        parameter_names=names,
        seed=142_001,
    )
    return env, transitions, reward, state_features


def fit_model() -> TDCCP:
    """Fit the public estimator on a deterministic synthetic panel."""
    env, transitions, reward, state_features = build_workflow()
    panel = simulate_panel(env, n_individuals=120, n_periods=30, seed=142_001)
    model = TDCCP(
        n_states=env.num_states,
        n_actions=env.num_actions,
        discount=float(env.problem_spec.discount_factor),
        utility=reward,
        se_method="robust",
        seed=142_001,
        method="semigradient",
        basis_type="encoded",
        basis_dim=2,
        basis_ridge=1e-7,
        ccp_method="logit",
        ccp_poly_degree=2,
        ccp_use_encoder=True,
        state_features=state_features,
        cross_fitting=True,
        robust_se=True,
        outer_max_iter=500,
        outer_tol=1e-7,
    )
    model.fit(panel, transitions=transitions)
    return model


def snapshot(model: TDCCP) -> dict[str, Any]:
    """Capture every supported output graded after deserialization."""
    states = np.arange(model.n_states, dtype=np.int64)
    counterfactual = model.counterfactual(
        replacement_cost=float(model.params_["replacement_cost"] - 0.25)
    )
    return {
        "coef": np.asarray(model.coef_, dtype=float).tolist(),
        "predict_proba": model.predict_proba(states).tolist(),
        "ev_features": np.asarray(model.ev_features_, dtype=float).tolist(),
        "summary": model.summary(),
        "counterfactual_policy": np.asarray(
            counterfactual.counterfactual_policy, dtype=float
        ).tolist(),
        "counterfactual_value": np.asarray(
            counterfactual.counterfactual_value, dtype=float
        ).tolist(),
    }


def _module_path() -> Path:
    import econirl

    return Path(econirl.__file__).resolve()


def _git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _verify(
    model_path: Path,
    expected_path: Path,
    output: Path,
    *,
    require_outside_checkout: bool,
) -> int:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    with model_path.open("rb") as handle:
        restored = pickle.load(handle)  # noqa: S301 - this script owns the pickle
    actual = snapshot(restored)
    array_fields = (
        "coef",
        "predict_proba",
        "ev_features",
        "counterfactual_policy",
        "counterfactual_value",
    )
    gaps = {
        field: float(
            np.max(
                np.abs(
                    np.asarray(actual[field], dtype=float)
                    - np.asarray(expected[field], dtype=float)
                )
            )
        )
        for field in array_fields
    }
    summary_equal = actual["summary"] == expected["summary"]
    module_path = _module_path()
    module_outside_checkout = not module_path.is_relative_to(ROOT)
    passed = (
        summary_equal
        and all(gap <= 1e-12 for gap in gaps.values())
        and (module_outside_checkout or not require_outside_checkout)
    )
    receipt = {
        "estimator": "TD-CCP",
        "status": "ready" if passed else "not_ready",
        "fresh_process": True,
        "summary_equal": summary_equal,
        "maximum_absolute_gaps": gaps,
        "threshold": 1e-12,
        "econirl_version": version("econirl"),
        "econirl_module": str(module_path),
        "module_outside_checkout": module_outside_checkout,
        "wheel_origin_required": require_outside_checkout,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expect-wheel", action="store_true")
    parser.add_argument("--verify", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--require-outside-checkout", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify:
        if args.model is None or args.expected is None:
            parser.error("--verify requires --model and --expected")
        return _verify(
            args.model,
            args.expected,
            args.output,
            require_outside_checkout=args.require_outside_checkout,
        )
    module_path = _module_path()
    if args.expect_wheel and module_path.is_relative_to(ROOT):
        raise RuntimeError(f"expected an installed wheel, imported {module_path}")
    with tempfile.TemporaryDirectory(prefix="econirl-tdccp-pickle-") as temp_name:
        temp = Path(temp_name)
        model_path = temp / "tdccp.pkl"
        expected_path = temp / "expected.json"
        model = fit_model()
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        expected_path.write_text(json.dumps(snapshot(model), allow_nan=False), encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--verify",
            "--model",
            str(model_path),
            "--expected",
            str(expected_path),
            "--output",
            str(args.output),
        ]
        if args.expect_wheel:
            command.append("--require-outside-checkout")
        completed = subprocess.run(command, check=False)
    if completed.returncode == 0:
        print(f"wrote {args.output}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
