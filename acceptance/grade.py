"""Grade Joe's recovered parameters against the held-out truth.

Orchestrator-side. Joe never sees this or truth.json. Structural recovery is
scored by parameter bias (metric segmentation: parameter bias is for the
finite-theta structural family only). Behavioral metrics (policy TV) are added
per chunk when an IRL estimator needs them.

Joe's findings file must contain his recovered coefficients as:
  {"recovered_theta": {"theta_0": 1.2, "theta_1": -0.4, ...}, ...}

Usage:
  python grade.py --truth problems/nfxp/truth.json --joe reports/nfxp_findings.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Recovery bar for a messy medium problem. Confirmed per estimator with the user;
# this default is the straw man. Reward is identified only up to scale for some
# methods, so compare DIRECTION + relative size, not raw equality.
REL_TOL = 0.25  # max acceptable relative bias per parameter


def grade(truth: dict, joe: dict) -> dict:
    names = truth["parameter_names"]
    true_theta = dict(zip(names, truth["theta"]))
    got = joe.get("recovered_theta") or {}

    rows, worst = [], 0.0
    for n in names:
        t = true_theta[n]
        g = got.get(n)
        if g is None:
            rows.append({"param": n, "true": t, "recovered": None, "rel_bias": None,
                         "note": "not reported"})
            worst = float("inf")
            continue
        denom = abs(t) if abs(t) > 1e-9 else 1.0
        rel = abs(g - t) / denom
        worst = max(worst, rel)
        rows.append({"param": n, "true": round(t, 4), "recovered": round(g, 4),
                     "rel_bias": round(rel, 3)})

    passed = worst <= REL_TOL
    return {"passed": bool(passed), "worst_rel_bias": (None if worst == float("inf")
            else round(worst, 3)), "rel_tol": REL_TOL, "params": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--joe", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    truth = json.loads(Path(a.truth).read_text())
    joe = json.loads(Path(a.joe).read_text())
    result = grade(truth, joe)
    text = json.dumps(result, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


# ponytail: self-check, fails loudly if the bias math regresses.
def _selfcheck() -> None:
    truth = {"parameter_names": ["a", "b"], "theta": [2.0, -1.0]}
    assert grade(truth, {"recovered_theta": {"a": 2.1, "b": -0.95}})["passed"] is True
    assert grade(truth, {"recovered_theta": {"a": 5.0, "b": -1.0}})["passed"] is False
    assert grade(truth, {"recovered_theta": {"a": 2.0}})["passed"] is False  # missing param
    print("grade selfcheck ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _selfcheck()
    else:
        main()
