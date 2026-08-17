#!/usr/bin/env python3
"""Verify public AIRL pickle parity in a fresh Python process."""

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

from econirl import AIRL, RewardSpec

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "airl_serialization.json"


def fit_model() -> AIRL:
    """Fit a bounded public model with functional bootstrap output."""
    from validation.estimators.airl.bootstrap_calibration import build_problem
    from validation.known_truth import SimulationConfig, simulate_known_truth_panel

    dgp = build_problem()
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=160, n_periods=40, seed=71_001),
    )
    reward = RewardSpec.state_dependent(
        dgp.feature_matrix[:, 0, :],
        dgp.parameter_names,
        dgp.problem.num_actions,
    )
    return AIRL(
        n_states=dgp.problem.num_states,
        n_actions=dgp.problem.num_actions,
        discount=dgp.problem.discount_factor,
        reward_lr=0.1,
        max_rounds=45,
        min_rounds=45,
        discriminator_steps=3,
        compute_se=True,
        n_bootstrap=3,
        seed=71_002,
        se_seed=71_003,
    ).fit(panel, transitions=np.asarray(dgp.transitions), reward=reward)


def snapshot(model: AIRL) -> dict[str, Any]:
    """Capture every supported public result graded after deserialization."""
    from validation.estimators.airl.bootstrap_calibration import build_problem

    dgp = build_problem()
    changed = np.asarray(dgp.transitions)[[1, 0]].copy()
    counterfactual = model.counterfactual(transitions=changed)
    assert model.bootstrap_ is not None
    return {
        "reward": np.asarray(model.reward_matrix_, dtype=float).tolist(),
        "predict_proba": model.predict_proba(np.arange(model.n_states)).tolist(),
        "summary": model.summary(),
        "confidence_intervals": {name: list(bounds) for name, bounds in model.conf_int().items()},
        "bootstrap_estimates": model.bootstrap_.estimates.tolist(),
        "counterfactual_policy": np.asarray(
            counterfactual.counterfactual_policy, dtype=float
        ).tolist(),
        "counterfactual_value": np.asarray(
            counterfactual.counterfactual_value, dtype=float
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
        restored = pickle.load(handle)  # noqa: S301 - this script owns the pickle
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
    imported_path = module_path()
    outside = not imported_path.is_relative_to(ROOT)
    passed = (
        summary_equal
        and intervals_equal
        and all(gap <= 1e-12 for gap in gaps.values())
        and (outside or not require_outside_checkout)
    )
    receipt = {
        "estimator": "AIRL",
        "status": "passed" if passed else "failed",
        "fresh_process": True,
        "summary_equal": summary_equal,
        "confidence_intervals_equal": intervals_equal,
        "maximum_absolute_gaps": gaps,
        "threshold": 1e-12,
        "econirl_version": version("econirl"),
        "econirl_module": str(imported_path),
        "module_outside_checkout": outside,
        "wheel_origin_required": require_outside_checkout,
        "git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
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
    imported_path = module_path()
    if args.expect_wheel and imported_path.is_relative_to(ROOT):
        raise RuntimeError(f"expected an installed wheel, imported {imported_path}")
    with tempfile.TemporaryDirectory(prefix="econirl-airl-pickle-") as temp_name:
        temp = Path(temp_name)
        model_path = temp / "airl.pkl"
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
