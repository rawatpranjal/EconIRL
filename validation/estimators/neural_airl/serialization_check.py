#!/usr/bin/env python3
"""Verify NeuralAIRL pickle parity in a fresh installed-wheel process."""

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

from econirl import NeuralAIRL
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "neural_airl_serialization.json"


def case() -> tuple[np.ndarray, np.ndarray, Panel]:
    """Build the bounded nonlinear workflow without checkout-only imports."""
    x = np.linspace(-1.0, 1.0, 9)
    reward = 1.5 * np.cos(np.pi * x) - 0.35 * x
    transitions = np.zeros((2, 9, 9), dtype=np.float64)
    for state in range(9):
        transitions[0, state, (state + 1) % 9] = 0.9
        transitions[0, state, state] = 0.1
        transitions[1, state, (state - 1) % 9] = 0.9
        transitions[1, state, state] = 0.1
    problem = DDCProblem(9, 2, 0.9, 1.0)
    oracle = value_iteration(
        SoftBellmanOperator(problem, jnp.asarray(transitions)),
        jnp.repeat(jnp.asarray(reward)[:, None], 2, axis=1),
        tol=1e-10,
        max_iter=5_000,
    )
    rng = np.random.default_rng(27_001)
    trajectories: list[Trajectory] = []
    for individual in range(80):
        current = int(rng.integers(9))
        states: list[int] = []
        actions: list[int] = []
        next_states: list[int] = []
        for _ in range(20):
            chosen = int(rng.choice(2, p=np.asarray(oracle.policy[current])))
            following = int(rng.choice(9, p=transitions[chosen, current]))
            states.append(current)
            actions.append(chosen)
            next_states.append(following)
            current = following
        trajectories.append(
            Trajectory(
                states=jnp.asarray(states),
                actions=jnp.asarray(actions),
                next_states=jnp.asarray(next_states),
                individual_id=individual,
            )
        )
    return x[:, None], transitions, Panel(trajectories=trajectories)


def fit_model() -> NeuralAIRL:
    inputs, transitions, panel = case()
    return NeuralAIRL(
        n_states=9,
        n_actions=2,
        discount=0.9,
        feature_matrix=inputs,
        reward_hidden_dim=32,
        reward_num_layers=2,
        shaping_hidden_dim=32,
        policy_hidden_dim=32,
        policy_steps=15,
        discriminator_steps=3,
        max_rounds=160,
        min_rounds=70,
        policy_step_size=0.1,
        compute_se=True,
        n_bootstrap=3,
        seed=27_002,
        se_seed=27_003,
    ).fit(panel, transitions=transitions)


def snapshot(model: NeuralAIRL) -> dict[str, Any]:
    _, transitions, _ = case()
    changed = transitions.copy()
    for state in range(9):
        changed[0, state] = 0.0
        changed[0, state, (state + 1) % 9] = 0.7
        changed[0, state, state] = 0.3
    counterfactual = model.counterfactual(transitions=changed)
    assert model.bootstrap_ is not None
    return {
        "reward": np.asarray(model.reward_, dtype=float).tolist(),
        "predict_proba": model.predict_proba(np.arange(9)).tolist(),
        "summary": model.summary(),
        "confidence_intervals": {name: list(bounds) for name, bounds in model.conf_int().items()},
        "bootstrap_estimates": model.bootstrap_.estimates.tolist(),
        "counterfactual_policy": np.asarray(
            counterfactual.counterfactual_policy,
            dtype=float,
        ).tolist(),
        "counterfactual_value": np.asarray(
            counterfactual.counterfactual_value,
            dtype=float,
        ).tolist(),
    }


def module_path() -> Path:
    import econirl

    return Path(econirl.__file__).resolve()


def maximum_gap(actual: Any, expected: Any) -> float:
    return float(
        np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)))
    )


def verify(
    model_path: Path,
    expected_path: Path,
    output: Path,
    *,
    require_outside_checkout: bool,
) -> int:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    with model_path.open("rb") as handle:
        restored = pickle.load(handle)  # noqa: S301 - this program owns the pickle
    actual = snapshot(restored)
    fields = (
        "reward",
        "predict_proba",
        "bootstrap_estimates",
        "counterfactual_policy",
        "counterfactual_value",
    )
    gaps = {field: maximum_gap(actual[field], expected[field]) for field in fields}
    summary_equal = actual["summary"] == expected["summary"]
    intervals_equal = actual["confidence_intervals"] == expected["confidence_intervals"]
    imported = module_path()
    outside = not imported.is_relative_to(ROOT)
    passed = (
        summary_equal
        and intervals_equal
        and all(gap <= 1e-12 for gap in gaps.values())
        and (outside or not require_outside_checkout)
    )
    receipt = {
        "estimator": "NeuralAIRL",
        "status": "passed" if passed else "failed",
        "fresh_process": True,
        "summary_equal": summary_equal,
        "confidence_intervals_equal": intervals_equal,
        "maximum_absolute_gaps": gaps,
        "threshold": 1e-12,
        "econirl_version": version("econirl"),
        "econirl_module": str(imported),
        "module_outside_checkout": outside,
        "wheel_origin_required": require_outside_checkout,
        "git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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
        return verify(
            args.model,
            args.expected,
            args.output,
            require_outside_checkout=args.require_outside_checkout,
        )
    imported = module_path()
    if args.expect_wheel and imported.is_relative_to(ROOT):
        raise RuntimeError(f"expected an installed wheel, imported {imported}")
    with tempfile.TemporaryDirectory(prefix="econirl-neural-airl-pickle-") as temp_name:
        temp = Path(temp_name)
        model_path = temp / "neural_airl.pkl"
        expected_path = temp / "expected.json"
        model = fit_model()
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)
        expected_path.write_text(
            json.dumps(snapshot(model), allow_nan=False),
            encoding="utf-8",
        )
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
