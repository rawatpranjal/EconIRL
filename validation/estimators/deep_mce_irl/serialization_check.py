#!/usr/bin/env python3
"""Verify Neural MCE-IRL pickle parity in a fresh Python process."""

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

import jax.numpy as jnp
import numpy as np

from econirl import MCEIRLNeural
from econirl.core.types import Panel, Trajectory

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "deep_mce_irl_serialization.json"


def build_panel(seed: int = 901, n_individuals: int = 160) -> Panel:
    """Return a compact two-state panel with heterogeneous choices."""
    rng = np.random.default_rng(seed)
    trajectories = []
    for individual in range(n_individuals):
        state = individual % 2
        action = int(rng.binomial(1, 0.72 if state == 0 else 0.31))
        trajectories.append(
            Trajectory(
                states=jnp.array([state]),
                actions=jnp.array([action]),
                next_states=jnp.array([state]),
                individual_id=individual,
            )
        )
    return Panel(trajectories)


def transition_tensor() -> np.ndarray:
    """Return P(s'|s,a) in (actions, states, next_states) orientation."""
    transitions = np.zeros((2, 2, 2), dtype=np.float32)
    transitions[:, 0, 0] = 1.0
    transitions[:, 1, 1] = 1.0
    return transitions


def fit_model() -> MCEIRLNeural:
    """Fit the public estimator with a bounded bootstrap."""
    return MCEIRLNeural(
        n_states=2,
        n_actions=2,
        discount=0.9,
        reward_hidden_dim=8,
        reward_num_layers=1,
        max_epochs=150,
        lr=0.05,
        occupancy_tol=0.04,
        patience=50,
        se_method="bootstrap",
        n_bootstrap=3,
        se_seed=47,
        seed=11,
    ).fit(build_panel(), transitions=transition_tensor())


def snapshot(model: MCEIRLNeural) -> dict[str, Any]:
    """Capture every supported output graded after deserialization."""
    states = np.array([0, 1])
    reward_delta = np.zeros((2, 2), dtype=float)
    reward_delta[:, 1] = 0.2
    counterfactual = model.counterfactual(reward_delta=reward_delta)
    assert model.bootstrap_ is not None
    return {
        "reward": np.asarray(model.reward_matrix_, dtype=float).tolist(),
        "predict_proba": model.predict_proba(states).tolist(),
        "summary": model.summary(),
        "confidence_intervals": {name: list(bounds) for name, bounds in model.conf_int().items()},
        "bootstrap_estimates": model.bootstrap_.estimates.tolist(),
        "counterfactual_policy": np.asarray(
            counterfactual.counterfactual_policy, dtype=float
        ).tolist(),
        "counterfactual_value": np.asarray(
            counterfactual.counterfactual_value, dtype=float
        ).tolist(),
        "counterfactual_intervals": json.loads(
            json.dumps(counterfactual.metadata["bootstrap_intervals"])
        ),
    }


def _module_path() -> Path:
    import econirl

    return Path(econirl.__file__).resolve()


def _git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _maximum_gap(actual: Any, expected: Any) -> float:
    """Return the maximum absolute numerical gap for one nested array."""
    return float(
        np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)))
    )


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
        "reward",
        "predict_proba",
        "bootstrap_estimates",
        "counterfactual_policy",
        "counterfactual_value",
    )
    gaps = {field: _maximum_gap(actual[field], expected[field]) for field in array_fields}
    summary_equal = actual["summary"] == expected["summary"]
    intervals_equal = actual["confidence_intervals"] == expected["confidence_intervals"]
    counterfactual_intervals_equal = (
        actual["counterfactual_intervals"] == expected["counterfactual_intervals"]
    )
    module_path = _module_path()
    module_outside_checkout = not module_path.is_relative_to(ROOT)
    passed = (
        summary_equal
        and intervals_equal
        and counterfactual_intervals_equal
        and all(gap <= 1e-12 for gap in gaps.values())
        and (module_outside_checkout or not require_outside_checkout)
    )
    receipt = {
        "estimator": "MCEIRLNeural",
        "status": "passed" if passed else "failed",
        "fresh_process": True,
        "summary_equal": summary_equal,
        "confidence_intervals_equal": intervals_equal,
        "counterfactual_intervals_equal": counterfactual_intervals_equal,
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

    with tempfile.TemporaryDirectory(prefix="econirl-neural-mce-pickle-") as temp_name:
        temp = Path(temp_name)
        model_path = temp / "mceirl_neural.pkl"
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
