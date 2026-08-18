#!/usr/bin/env python3
"""Verify public GLADIUS pickle parity in a fresh Python process."""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import subprocess
import sys
import tempfile
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from econirl import GLADIUS

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "validation" / "results" / "gladius_serialization.json"


def fit_model() -> GLADIUS:
    """Fit a compact anchored model with functional bootstrap output."""
    from validation.estimators.gladius.bootstrap_calibration import _controlled_case

    panel, features, transitions, _, _ = _controlled_case(91_001)
    model = GLADIUS(
        n_actions=2,
        discount=0.95,
        q_hidden_dim=8,
        q_num_layers=1,
        ev_hidden_dim=8,
        ev_num_layers=1,
        batch_size=64,
        max_epochs=8,
        patience=9,
        anchor_action=0,
        anchor_rewards=(0.0, 0.0, 0.0, 0.0),
        compute_se=True,
        n_bootstrap=2,
        seed=91_002,
        se_seed=91_003,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        model.fit(panel, features=features, transitions=transitions)
    return model


def snapshot(model: GLADIUS) -> dict[str, Any]:
    """Capture every supported public result graded after deserialization."""
    delta = np.zeros((4, 2), dtype=float)
    delta[:, 1] = 0.25
    counterfactual = model.counterfactual(reward_delta=delta)
    assert model.bootstrap_ is not None
    return {
        "coef": np.asarray(model.coef_, dtype=float).tolist(),
        "q": np.asarray(model.q_, dtype=float).tolist(),
        "continuation_value": np.asarray(model.continuation_value_, dtype=float).tolist(),
        "reward": np.asarray(model.reward_, dtype=float).tolist(),
        "policy": np.asarray(model.policy_, dtype=float).tolist(),
        "predict_proba": model.predict_proba(np.arange(4)).tolist(),
        "summary": model.summary(),
        "confidence_intervals": {name: list(bounds) for name, bounds in model.conf_int().items()},
        "bootstrap_estimates": np.asarray(model.bootstrap_.estimates, dtype=float).tolist(),
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
        "coef",
        "q",
        "continuation_value",
        "reward",
        "policy",
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
        "estimator": "GLADIUS",
        "status": "ready" if passed else "not_ready",
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
    with tempfile.TemporaryDirectory(prefix="econirl-gladius-pickle-") as temp_name:
        temp = Path(temp_name)
        model_path = temp / "gladius.pkl"
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
