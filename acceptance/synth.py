"""Generate a matched, medium-scale, messy problem for one estimator.

Orchestrator-side. Runs against econirl's own generators, keeps the answer key
(true theta) here, and writes Joe only the observable inputs:

  problem.json   dims, discount, scale, parameter names, file pointers (NO theta)
  panel.csv      the observed id/period/state/action/next_state panel
  features.npy   the feature map phi(s,a,k) the analyst's model uses

  truth.json     HELD BACK (true theta + dims) for grade.py, never enters Joe's container

The problem is set at a genuine medium scale with realistic warts (a finite,
short panel so state coverage is sparse and some states are single-action
in-sample). It is identified, so recovery is possible. It is NOT diluted later:
if an estimator misses on it, that is a finding, not a cue to shrink it.

Usage:
  PYTHONPATH=../src python synth.py --estimator nfxp --out problems/nfxp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from econirl.environments import random_mdp
from econirl.simulation import simulate_panel

# Per-estimator matched problem. Members beyond nfxp are stubs until each chunk
# opens (the user confirms the matched problem + recovery bar per estimator).
# ponytail: one dict entry per estimator; scale is matched to the method, fixed once set.
PROBLEMS = {
    "nfxp": dict(
        num_states=200, num_actions=2, num_features=3, branching=5,
        discount_factor=0.95, scale_parameter=1.0, seed=7,
        n_individuals=300, n_periods=25,  # short panel -> sparse, messy coverage
    ),
}


def build(cfg: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    env = random_mdp(
        num_states=cfg["num_states"], num_actions=cfg["num_actions"],
        num_features=cfg["num_features"], branching=cfg["branching"],
        discount_factor=cfg["discount_factor"], scale_parameter=cfg["scale_parameter"],
        seed=cfg["seed"],
    )
    panel = simulate_panel(
        env, n_individuals=cfg["n_individuals"], n_periods=cfg["n_periods"],
        seed=cfg["seed"], use_optimal_policy=True,
    )
    df = panel.to_dataframe()

    names = list(env.parameter_names)
    theta = [float(env.true_parameters[n]) for n in names]
    features = np.asarray(env.feature_matrix)  # (S, A, K)

    # messiness report (honest, not injected): how sparse / single-action is it
    seen = df.groupby("state")["action"].nunique()
    n_single = int((seen < cfg["num_actions"]).sum())
    coverage = float(df["state"].nunique()) / cfg["num_states"]

    # ---- inputs Joe receives (no theta) ----
    df.to_csv(out / "panel.csv", index=False)
    np.save(out / "features.npy", features)
    (out / "problem.json").write_text(json.dumps({
        "num_states": cfg["num_states"], "num_actions": cfg["num_actions"],
        "num_features": cfg["num_features"], "discount_factor": cfg["discount_factor"],
        "scale_parameter": cfg["scale_parameter"], "parameter_names": names,
        "panel_file": "panel.csv", "features_file": "features.npy",
        "note": "Linear utility u(s,a)=theta . phi(s,a). Recover theta from the panel.",
    }, indent=2))

    # ---- answer key, held back ----
    (out / "truth.json").write_text(json.dumps({
        "parameter_names": names, "theta": theta,
        "num_states": cfg["num_states"], "num_actions": cfg["num_actions"],
        "discount_factor": cfg["discount_factor"], "scale_parameter": cfg["scale_parameter"],
        "messiness": {"obs": int(len(df)), "state_coverage": round(coverage, 3),
                      "single_action_states": n_single},
    }, indent=2))

    print(f"built {out.name}: {len(df)} obs, coverage {coverage:.2f}, "
          f"{n_single} single-action states. theta held in truth.json.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="nfxp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.estimator not in PROBLEMS:
        raise SystemExit(f"no matched problem for {a.estimator!r}; "
                         f"add it to PROBLEMS (one per chunk, confirmed with user)")
    build(PROBLEMS[a.estimator], Path(a.out))


if __name__ == "__main__":
    main()
