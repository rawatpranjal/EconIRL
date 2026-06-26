"""Generate a matched, medium-scale, messy, IDENTIFIED problem for one estimator.

Orchestrator-side. Keeps the answer key (true theta) here, writes Joe only the
observable inputs:

  problem.json   dims, discount, scale, parameter names, file pointers (NO theta)
  panel.csv      the observed id/period/state/action/next_state panel
  features.npy   the feature map phi(s,a,k) the analyst's model uses

  truth.json     HELD BACK (true theta + dims + diagnostics) for grade.py, never enters Joe

The problem is medium-scale with realistic warts (a short panel so coverage is
sparse and some states are single-action in-sample) but it is IDENTIFIED: the
action-contrast design is full rank, well-conditioned, and not near-collinear.
An identification gate enforces this, so we never silently ship an unrecoverable
problem and then blame the estimator. The problem is messy, not diluted: if an
estimator misses on it, that is a finding.

Usage:
  PYTHONPATH=../src python synth.py --estimator nfxp --out problems/nfxp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from econirl.environments import ArrayMDP
from econirl.simulation import simulate_panel

# Per-estimator matched problem config. One entry per estimator; scale is matched
# to the method and fixed once set (the user confirms the matched problem per chunk).
# ponytail: the builder per estimator encodes the transition structure that method handles.
PROBLEMS = {
    "nfxp": dict(
        num_states=200, num_features=3, discount_factor=0.95, scale_parameter=1.0,
        seed=7, n_individuals=300, n_periods=25,  # short panel -> sparse, messy coverage
        theta=[1.0, -0.8, 0.6],                    # comparable O(1) magnitudes, identified
    ),
}

# Identification gate thresholds (action-contrast design must clear these).
MIN_CONTRAST_STD = 1e-3      # a feature whose contrast barely varies is an intercept
MAX_CONDITION = 1e3          # near-singular contrast design = weakly identified
MAX_PAIRWISE_CORR = 0.9      # near-collinear contrasts = not separately identified


def _contrast_diagnostics(features: np.ndarray) -> dict:
    """Diagnostics on the action-contrast design phi(s,a)-phi(s,0), stacked over a>0.

    Identification of choice parameters comes from the action-contrast, not the
    raw features (Train 2009; Magnac-Thesmar 2002). Rank alone is not enough: a
    constant contrast (an intercept) or near-collinear contrasts pass a rank
    check yet leave parameters weakly identified (the CLAUDE.md learned rule:
    check magnitudes and contrast, not just raw design rank).
    """
    S, A, K = features.shape
    contrast = np.concatenate(
        [features[:, a, :] - features[:, 0, :] for a in range(1, A)], axis=0
    )  # (S*(A-1), K)
    stds = contrast.std(axis=0)
    rank = int(np.linalg.matrix_rank(contrast))
    cond = float(np.linalg.cond(contrast)) if rank == K else float("inf")
    # max abs off-diagonal correlation among contrast columns
    max_corr = 0.0
    if K > 1 and (stds > 0).all():
        c = np.corrcoef(contrast.T)
        max_corr = float(np.max(np.abs(c - np.eye(K))))
    return {"contrast_std": [round(float(x), 4) for x in stds], "rank": rank,
            "num_features": K, "condition_number": round(cond, 1),
            "max_pairwise_corr": round(max_corr, 3)}


def assert_identified(features: np.ndarray) -> dict:
    """Raise if the action-contrast design is degenerate. Returns the diagnostics."""
    d = _contrast_diagnostics(features)
    problems = []
    if d["rank"] < d["num_features"]:
        problems.append(f"contrast rank {d['rank']} < K={d['num_features']} (rank-deficient)")
    if min(d["contrast_std"]) < MIN_CONTRAST_STD:
        problems.append(f"a feature contrast barely varies (std {min(d['contrast_std'])}, "
                        f"floor {MIN_CONTRAST_STD}); it acts as an intercept, not identified")
    if d["condition_number"] > MAX_CONDITION:
        problems.append(f"contrast condition {d['condition_number']} > {MAX_CONDITION} "
                        f"(weakly identified)")
    if d["max_pairwise_corr"] > MAX_PAIRWISE_CORR:
        problems.append(f"contrast collinearity {d['max_pairwise_corr']} > {MAX_PAIRWISE_CORR} "
                        f"(features not separately identified)")
    if problems:
        raise ValueError(
            "matched problem is NOT identified, refusing to ship it:\n  - "
            + "\n  - ".join(problems)
            + f"\n  diagnostics: {d}"
        )
    return d


def _make_nfxp_env(cfg: dict, rng: np.random.Generator) -> ArrayMDP:
    """A Rust-style, identified, messy problem matched to NFXP.

    Transitions are the structure NFXP's default represents exactly: action 0
    (keep) drifts the state up by 0/1/2, action 1 (replace) resets to state 0.
    Features are an identified, well-conditioned action-contrast: action 0 is the
    outside option (zeros), action 1 carries standardized near-orthogonal random
    features, so each theta is separately identified.
    """
    S, K = cfg["num_states"], cfg["num_features"]
    A = 2
    T = np.zeros((A, S, S))
    drift = np.array([0.6, 0.3, 0.1])  # P(+0), P(+1), P(+2) under keep
    for s in range(S):
        for j, p in enumerate(drift):
            T[0, s, min(s + j, S - 1)] += p
    T[1, :, :] = 0.0
    T[1, :, 0] = 1.0  # replace resets to state 0

    feats = np.zeros((S, A, K))
    col = rng.normal(size=(S, K))
    col = (col - col.mean(0)) / col.std(0)  # standardized -> near-orthogonal contrast
    feats[:, 1, :] = col

    names = [f"theta_{i}" for i in range(K)]
    return ArrayMDP(T, feats, theta=np.asarray(cfg["theta"][:K], dtype=float),
                    discount_factor=cfg["discount_factor"],
                    scale_parameter=cfg["scale_parameter"],
                    parameter_names=names, seed=cfg["seed"])


BUILDERS = {"nfxp": _make_nfxp_env}


def build(estimator: str, cfg: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    env = BUILDERS[estimator](cfg, rng)

    features = np.asarray(env.feature_matrix)  # (S, A, K)
    diag = assert_identified(features)         # gate: refuse to ship unidentified

    panel = simulate_panel(env, n_individuals=cfg["n_individuals"],
                           n_periods=cfg["n_periods"], seed=cfg["seed"],
                           use_optimal_policy=True)
    df = panel.to_dataframe()

    names = list(env.parameter_names)
    theta = [float(env.true_parameters[n]) for n in names]

    seen = df.groupby("state")["action"].nunique()
    n_single = int((seen < env.num_actions).sum())
    coverage = float(df["state"].nunique()) / cfg["num_states"]

    # ---- inputs Joe receives (no theta) ----
    df.to_csv(out / "panel.csv", index=False)
    np.save(out / "features.npy", features)
    (out / "problem.json").write_text(json.dumps({
        "num_states": cfg["num_states"], "num_actions": int(env.num_actions),
        "num_features": cfg["num_features"], "discount_factor": cfg["discount_factor"],
        "scale_parameter": cfg["scale_parameter"], "parameter_names": names,
        "panel_file": "panel.csv", "features_file": "features.npy",
        "note": "Linear utility u(s,a)=theta . phi(s,a). Recover theta from the panel.",
    }, indent=2))

    # ---- answer key, held back ----
    (out / "truth.json").write_text(json.dumps({
        "parameter_names": names, "theta": theta,
        "num_states": cfg["num_states"], "num_actions": int(env.num_actions),
        "discount_factor": cfg["discount_factor"], "scale_parameter": cfg["scale_parameter"],
        "identification": diag,
        "messiness": {"obs": int(len(df)), "state_coverage": round(coverage, 3),
                      "single_action_states": n_single},
    }, indent=2))

    print(f"built {out.name}: {len(df)} obs, coverage {coverage:.2f}, "
          f"{n_single} single-action states. identified: rank {diag['rank']}/{diag['num_features']}, "
          f"cond {diag['condition_number']}, max-corr {diag['max_pairwise_corr']}. theta held back.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="nfxp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.estimator not in PROBLEMS or a.estimator not in BUILDERS:
        raise SystemExit(f"no matched problem for {a.estimator!r}; "
                         f"add it to PROBLEMS + BUILDERS (one per chunk, confirmed with user)")
    build(a.estimator, PROBLEMS[a.estimator], Path(a.out))


if __name__ == "__main__":
    main()
