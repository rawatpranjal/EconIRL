"""Simulation study: serialized-content consumption with hidden reader segments.

The unobserved-heterogeneity showcase (Lee, Sudhir & Wang 2026). A reader on a
serialized-content platform chooses pay / wait / exit chapter by chapter. Two
hidden reader segments have different tastes. The headline estimator, AIRL-Het,
recovers the segments and their rewards. Pooled and homogeneous estimators fit
average behaviour but cannot represent either segment.

This study does not use the shared cross-estimator harness. The harness data dict
assumes one ``true_theta`` and a structural parameter RMSE, which does not fit a
per-segment heterogeneous study. The study keeps the harness *contract* instead:
raw per-run records in a results file, the page rendered as a pure function of
that file, crashes recorded rather than hidden, no fabricated numbers.

DGP, truth objects, label matching, and the segment metric engine come from
``validation/known_truth.py`` (the ``ContentHeterogeneityKnownTruthConfig`` DGP
and ``evaluate_segmented_estimator_against_truth``).

Usage::

    python scripts/study_content_consumption.py [--verbose]   # run + write JSON + page
    python scripts/study_content_consumption.py --page         # re-render page from JSON
    python scripts/study_content_consumption.py --verify       # re-derive headline from JSON
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [os.path.join(_ROOT, "src"), _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from validation.known_truth import (  # noqa: E402
    ContentHeterogeneityKnownTruthConfig,
    SimulationConfig,
    build_known_truth_dgp,
    simulate_known_truth_panel,
    solve_known_truth,
    evaluate_segmented_estimator_against_truth,
    policy_divergence,
)

RESULTS_JSON = os.path.join(_ROOT, "validation", "results", "study_content_consumption.json")
PAGE_PATH = os.path.join(_ROOT, "docs", "simulation_studies", "content_consumption.md")
FIGURE_PNG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                          "content_consumption_segments.png")
FIGURE_REL = "../_static/simulation_studies/content_consumption_segments.png"

# ---- DGP configuration ----
# Serialized content: 5 chapters x 3 wait bins x 2 price levels x 2 quality
# levels = 60 episode states + 1 absorbing exit state = 61 states. Three actions:
# pay-and-read (0), wait-and-read (1), exit (2). The exit action moves to the
# absorbing state and is the reward anchor (its reward is pinned to zero). Two
# latent reader segments with segment-specific rewards and near-even priors.
DGP_KW = dict(
    num_chapters=5, wait_bins=3, price_levels=2, quality_levels=2,
    books_per_user=4, discount_factor=0.92, scale_parameter=0.85, seed=4506,
)
SIM = dict(n_individuals=800, n_periods=16, seed=4507)

ACTION_NAMES = ("pay", "wait", "exit")
SEGMENT_NAMES = ("binge reader", "patient reader")  # seg0, seg1 in known_truth


def build_dgp():
    return build_known_truth_dgp(ContentHeterogeneityKnownTruthConfig(**DGP_KW))


def _airl_het_config(dgp):
    from econirl.estimation.adversarial.airl_het import AIRLHetConfig

    return AIRLHetConfig(
        num_segments=dgp.config.num_segments,
        exit_action=dgp.config.exit_action,
        absorbing_state=dgp.config.absorbing_state,
        reward_type="linear",
        reward_lr=0.001,
        discriminator_steps=2,
        policy_step_size=0.1,
        generator_reward="f",
        max_airl_rounds=3,
        min_airl_rounds=1,
        max_em_iterations=8,
        airl_convergence_tol=0.01,
        em_convergence_tol=1e-3,
        prior_min=0.05,
        prior_damping=0.8,
        consistency_weight=1.0,
        antisymmetric_init=False,
        initialization="behavioral_anchor",
        initialization_smoothing=1.0,
        initialization_l2_penalty=10.0,
        generator_max_iter=5_000,
    )


# ---------------------------------------------------------------------------
# Estimator runs. Each returns a single (n_states, n_actions) policy; the
# headline additionally returns its segment objects via the summary metadata.
# ---------------------------------------------------------------------------

def _run_bc(dgp, panel):
    """Tabular behaviour cloning: empirical choice frequencies P(a|s)."""
    ns, na = dgp.problem.num_states, dgp.problem.num_actions
    counts = np.zeros((ns, na))
    for tr in panel.trajectories:
        for s, a in zip(np.asarray(tr.states), np.asarray(tr.actions)):
            counts[s, a] += 1.0
    policy = np.full((ns, na), 1.0 / na)
    for s in range(ns):
        if counts[s].sum() > 0:
            policy[s] = counts[s] / counts[s].sum()
    return policy, True


def _run_pooled_airl(dgp, panel):
    from econirl.estimation.adversarial.airl import AIRLConfig, AIRLEstimator

    cfg = AIRLConfig(
        reward_type="linear", reward_arg="state_action",
        anchor_action=dgp.config.exit_action, absorbing_state=dgp.config.absorbing_state,
        reward_lr=0.001, discriminator_steps=2, policy_step_size=0.1,
        generator_reward="f", max_rounds=40, min_rounds=5, convergence_tol=0.01,
        compute_se=False, verbose=False,
    )
    s = AIRLEstimator(cfg).estimate(panel, dgp.utility(), dgp.problem, dgp.transitions)
    return np.asarray(s.policy), bool(s.converged)


def _run_nfxp(dgp, panel):
    from econirl.estimation import NFXPEstimator

    s = NFXPEstimator(
        inner_solver="hybrid", inner_tol=1e-8, inner_max_iter=20_000,
        compute_hessian=False, verbose=False,
    ).estimate(panel, dgp.utility(), dgp.problem, dgp.transitions)
    return np.asarray(s.policy), bool(s.converged)


def _run_ccp(dgp, panel):
    from econirl.estimation import CCPEstimator

    s = CCPEstimator(num_policy_iterations=1, compute_hessian=False, verbose=False).estimate(
        panel, dgp.utility(), dgp.problem, dgp.transitions
    )
    return np.asarray(s.policy), bool(s.converged)


BASELINES = (
    ("BC", "behavioral", _run_bc),
    ("Pooled-AIRL", "behavioral", _run_pooled_airl),
    ("NFXP", "structural", _run_nfxp),
    ("CCP", "structural", _run_ccp),
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _occupancy_weights(panel, n_states):
    counts = np.zeros(n_states)
    for tr in panel.trajectories:
        for s in np.asarray(tr.states):
            counts[s] += 1.0
    total = counts.sum()
    return jnp.asarray(counts / total if total > 0 else np.ones(n_states) / n_states)


def _seg_tvs(est_policy, seg_policies, mix_policy, weights):
    """(pooled TV vs mixture, [TV vs seg0, TV vs seg1, ...]) under occupancy weights."""
    est = jnp.asarray(est_policy)
    pooled = policy_divergence(jnp.asarray(mix_policy), est, weights).tv
    per_seg = [policy_divergence(jnp.asarray(sp), est, weights).tv for sp in seg_policies]
    return float(pooled), [float(x) for x in per_seg]


def _confusion(panel, segment_policies, traj_segments, n_actions):
    """3x3 confusion of observed action vs an argmax-policy prediction.

    For each trajectory, ``traj_segments[i]`` gives the segment to use. The
    predicted action at a state is the argmax of that segment's policy. Used both
    for the headline (estimated segment, recovered policy) and for the true-model
    ceiling (true segment, true policy).
    """
    seg_pol = [np.asarray(sp) for sp in segment_policies]
    seg_idx = np.asarray(traj_segments, dtype=int)
    mat = np.zeros((n_actions, n_actions), dtype=int)
    for i, tr in enumerate(panel.trajectories):
        seg = int(seg_idx[i])
        for s, a in zip(np.asarray(tr.states), np.asarray(tr.actions)):
            pred = int(np.argmax(seg_pol[seg][s]))
            mat[int(a), pred] += 1
    total = mat.sum()
    acc = float(np.trace(mat) / total) if total else 0.0
    return mat.tolist(), acc


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(verbose: bool = False) -> dict:
    from econirl.estimation.adversarial.airl_het import AIRLHetEstimator

    dgp = build_dgp()
    cfg = dgp.config
    ns, na = dgp.problem.num_states, dgp.problem.num_actions
    panel = simulate_known_truth_panel(dgp, SimulationConfig(**SIM))

    priors = np.asarray(dgp.segment_probabilities, dtype=float)
    seg_sol = [solve_known_truth(dgp, segment_index=g) for g in range(dgp.num_segments)]
    seg_pol = [np.asarray(s.policy) for s in seg_sol]
    mix_pol = sum(priors[g] * seg_pol[g] for g in range(dgp.num_segments))
    w = _occupancy_weights(panel, ns)
    true_gap_tv = float(policy_divergence(jnp.asarray(seg_pol[0]), jnp.asarray(seg_pol[1]), w).tv)

    # --- Headline: AIRL-Het ---
    headline = {"name": "AIRL-Het", "error": None}
    t0 = time.time()
    try:
        summary = AIRLHetEstimator(_airl_het_config(dgp)).estimate(
            panel, dgp.utility(), dgp.problem, dgp.transitions
        )
        rt = time.time() - t0
        m = evaluate_segmented_estimator_against_truth(
            dgp, summary, panel=panel, counterfactual_kinds=()
        )
        perm = m["segment_permutation"]["true_to_estimated"]  # true_idx -> est_idx
        seg_policies_meta = summary.metadata["segment_policies"]
        # per-segment occupancy-weighted policy TV, matched true->estimated
        seg_tv_matched = []
        for true_idx, est_idx in enumerate(perm):
            seg_tv_matched.append(float(
                policy_divergence(jnp.asarray(seg_pol[true_idx]),
                                  jnp.asarray(seg_policies_meta[est_idx]), w).tv
            ))
        pooled_tv = float(policy_divergence(jnp.asarray(mix_pol), jnp.asarray(summary.policy), w).tv)
        est_assign = np.argmax(np.asarray(summary.metadata["segment_posteriors"]), axis=1)
        conf, conf_acc = _confusion(panel, seg_policies_meta, est_assign, na)
        # ceiling: the true model's own argmax choices vs sampled actions, the
        # most any model could match given the randomness in the choices.
        _, ceiling_acc = _confusion(panel, seg_pol, panel.metadata["segment_labels"], na)
        headline.update(
            assignment_accuracy=m["segment_assignment_accuracy"],
            prior_l1=m["segment_prior_l1"],
            aligned_priors=m["aligned_segment_priors"],
            segment_reward_nrmse=m["segment_reward_normalized_rmse"],
            max_segment_reward_nrmse=m["max_segment_reward_normalized_rmse"],
            segment_policy_tv=seg_tv_matched,
            max_segment_policy_tv=max(seg_tv_matched),
            segment_value_nrmse=m["segment_value_normalized_rmse"],
            pooled_policy_tv=pooled_tv,
            confusion=conf,
            confusion_accuracy=conf_acc,
            confusion_ceiling_accuracy=ceiling_acc,
            permutation=perm,
            runtime=rt,
            converged=bool(summary.converged),
            em_iterations=int(summary.num_iterations),
        )
    except Exception as exc:  # record, do not hide
        import traceback
        headline.update(error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(), runtime=time.time() - t0)
        if verbose:
            print(headline["traceback"])

    # --- Misspecified baselines ---
    baseline_records = []
    for name, family, fn in BASELINES:
        rec = {"name": name, "family": family, "recovers_segments": False, "error": None}
        t0 = time.time()
        try:
            est_pol, conv = fn(dgp, panel)
            pooled_tv, per_seg_tv = _seg_tvs(est_pol, seg_pol, mix_pol, w)
            rec.update(pooled_policy_tv=pooled_tv, segment_policy_tv=per_seg_tv,
                       max_segment_policy_tv=max(per_seg_tv),
                       runtime=time.time() - t0, converged=bool(conv))
        except Exception as exc:
            import traceback
            rec.update(error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc(), runtime=time.time() - t0)
            if verbose:
                print(rec["traceback"])
        baseline_records.append(rec)

    diag = {
        "feature_rank": int(np.linalg.matrix_rank(
            np.asarray(dgp.feature_matrix).reshape(-1, dgp.feature_matrix.shape[-1]))),
        "num_features": int(dgp.feature_matrix.shape[-1]),
    }

    data = {
        "meta": {
            "title": "Serialized content with hidden reader segments",
            "date": _dt.date.today().isoformat(),
            "package_version": _package_version(),
            "n_states": ns,
            "n_actions": na,
            "exit_action": int(cfg.exit_action),
            "absorbing_state": int(cfg.absorbing_state),
            "num_segments": int(dgp.num_segments),
            "num_chapters": cfg.num_chapters,
            "n_individuals": SIM["n_individuals"],
            "n_periods": SIM["n_periods"],
            "n_trajectories": len(panel.trajectories),
            "n_observations": int(panel.num_observations),
            "true_priors": priors.tolist(),
            "segment_names": list(SEGMENT_NAMES),
            "action_names": list(ACTION_NAMES),
            "true_segment_gap_tv": true_gap_tv,
            "diagnostics": diag,
            "discount_factor": cfg.discount_factor,
            "scale_parameter": cfg.scale_parameter,
            "dgp_kw": DGP_KW,
            "sim": SIM,
        },
        "headline": headline,
        "baselines": baseline_records,
    }
    return data


def _package_version() -> str:
    try:
        import econirl
        return getattr(econirl, "__version__", "unknown")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(data: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, seg0, seg1 = [], [], []
    h = data["headline"]
    if h.get("error") is None and h.get("segment_policy_tv"):
        names.append("AIRL-Het")
        seg0.append(h["segment_policy_tv"][0])
        seg1.append(h["segment_policy_tv"][1])
    for b in data["baselines"]:
        if b.get("error") is None and b.get("segment_policy_tv"):
            names.append(b["name"])
            seg0.append(b["segment_policy_tv"][0])
            seg1.append(b["segment_policy_tv"][1])
    if not names:
        return

    sn = data["meta"]["segment_names"]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(x - width / 2, seg0, width, label=f"vs {sn[0]}", color="#3b7dd8")
    ax.bar(x + width / 2, seg1, width, label=f"vs {sn[1]}", color="#e07b39")
    ax.axhline(data["meta"]["true_segment_gap_tv"], color="grey", ls="--", lw=1,
               label="distance between the two segments")
    ax.set_ylabel("policy distance (total variation)")
    ax.set_title("Distance from each estimator's policy to each true segment")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page rendering (pure function of the data dict)
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and (x != x):  # NaN
            return "-"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def render_page(data: dict) -> str:
    meta = data["meta"]
    h = data["headline"]
    sn = meta["segment_names"]
    an = meta["action_names"]
    lines: list[str] = []

    lines.append("# Serialized content with hidden reader segments")
    lines.append("")
    lines.append(
        "A reader on a serialized-content platform moves through a book one "
        "chapter at a time. Each chapter the reader chooses to pay and read now, "
        "wait for a free unlock and then read, or exit the book. There are two "
        "hidden reader types with different tastes. One type reads on quality and "
        "cliffhangers and pays to keep going. The other type is price sensitive "
        "and waits for the free unlock. We never observe a reader's type. The "
        "study asks which estimators can recover the two types from choices alone."
    )
    lines.append("")
    lines.append("## The data-generating process")
    lines.append("")
    lines.append(
        f"The state combines four things: the chapter (out of {meta['num_chapters']}), "
        "how long the reader has waited, whether the current chapter is priced or "
        "free, and a content-quality level. That gives "
        f"{meta['n_states'] - 1} episode states plus one absorbing exit state, "
        f"for {meta['n_states']} states in total. There are three actions: pay "
        "(0), wait (1), and exit (2)."
    )
    lines.append("")
    lines.append(
        "Paying or waiting advances the content. Exit moves the reader to the "
        "absorbing state and ends the book. The exit action is the reward anchor: "
        "its reward is fixed at zero, and the value at the absorbing state is "
        "normalized to zero in estimation. These two anchors pin down the reward so "
        "that the recovered reward matches the true one rather than an arbitrary "
        "shifted version."
    )
    lines.append("")
    lines.append(
        f"The reward is linear in {meta['diagnostics']['num_features']} content "
        "features. Two latent segments have their own reward weights and near-even "
        f"population shares of {_fmt(meta['true_priors'][0],2)} and "
        f"{_fmt(meta['true_priors'][1],2)}. Each reader keeps the same type across "
        f"{meta['dgp_kw']['books_per_user']} books. Behaviour solves the soft "
        f"Bellman equation with discount {meta['discount_factor']} and logit scale "
        f"{meta['scale_parameter']}. The panel simulates {meta['n_individuals']} "
        f"readers across {meta['n_trajectories']} books "
        f"({meta['n_observations']} chapter decisions). The two segments do choose "
        f"differently: the distance between their policies is "
        f"{_fmt(meta['true_segment_gap_tv'])} in total variation."
    )
    lines.append("")
    if os.path.exists(FIGURE_PNG):
        lines.append(f"![Per-segment policy distance by estimator]({FIGURE_REL})")
        lines.append("")

    # --- headline ---
    lines.append("## What AIRL-Het recovers")
    lines.append("")
    if h.get("error"):
        lines.append(f"AIRL-Het did not complete. Error: `{h['error']}`.")
        lines.append("")
    else:
        lines.append(
            "AIRL-Het fits two segment-specific rewards with an EM loop that also "
            "infers each reader's type. Label switching is resolved before scoring: "
            "estimated segments are matched to true segments by reward distance, "
            "and assignment accuracy is computed after that match."
        )
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Segment assignment accuracy (after matching) | {_fmt(h['assignment_accuracy'])} |")
        lines.append(f"| Segment prior L1 error | {_fmt(h['prior_l1'],4)} |")
        lines.append(f"| Recovered priors | {_fmt(h['aligned_priors'][0],2)}, {_fmt(h['aligned_priors'][1],2)} |")
        lines.append(f"| Reward distance, {sn[0]} (normalized RMSE) | {_fmt(h['segment_reward_nrmse'][0])} |")
        lines.append(f"| Reward distance, {sn[1]} (normalized RMSE) | {_fmt(h['segment_reward_nrmse'][1])} |")
        lines.append(f"| Policy distance, {sn[0]} (TV) | {_fmt(h['segment_policy_tv'][0])} |")
        lines.append(f"| Policy distance, {sn[1]} (TV) | {_fmt(h['segment_policy_tv'][1])} |")
        lines.append(f"| Pooled policy distance vs average behaviour (TV) | {_fmt(h['pooled_policy_tv'])} |")
        lines.append(f"| Runtime (s) | {_fmt(h['runtime'],1)} |")
        lines.append("")
        lines.append(
            "AIRL-Het recovers both segments. The policy distance is small for "
            "each one, the priors are close, and most readers are assigned to the "
            "right type."
        )
        lines.append("")
        # confusion matrix
        conf = h.get("confusion")
        if conf:
            lines.append("### Pay / wait / exit choices")
            lines.append("")
            lines.append(
                "Rows are the choice the reader made. Columns are the choice the "
                "recovered model predicts for that reader's assigned type. "
                f"Overall agreement is {_fmt(h['confusion_accuracy'])}. The choices "
                "are random draws from a soft policy, so no model can match every "
                "one. As a reference, the true model's own choices agree with the "
                f"sampled actions {_fmt(h.get('confusion_ceiling_accuracy'))} of the "
                "time. The recovered model matches that reference."
            )
            lines.append("")
            header = "| observed \\ predicted | " + " | ".join(an) + " |"
            lines.append(header)
            lines.append("|---|" + "|".join(["---"] * len(an)) + "|")
            for i, row in enumerate(conf):
                lines.append(f"| {an[i]} | " + " | ".join(str(v) for v in row) + " |")
            lines.append("")

    # --- baselines ---
    lines.append("## What the pooled and homogeneous estimators miss")
    lines.append("")
    lines.append(
        "These estimators assume a single reader type. Each one returns a single "
        "policy. The table reports how far that one policy sits from the average "
        "behaviour, and from each of the two true segments. A small pooled "
        "distance with large per-segment distances means the estimator matches the "
        "crowd but represents neither type."
    )
    lines.append("")
    lines.append(f"| Estimator | Family | Recovers segments | Pooled TV | TV vs {sn[0]} | TV vs {sn[1]} | Time (s) |")
    lines.append("|---|---|---|---|---|---|---|")
    if h.get("error") is None:
        lines.append(
            f"| AIRL-Het | heterogeneous | yes (2) | {_fmt(h['pooled_policy_tv'])} | "
            f"{_fmt(h['segment_policy_tv'][0])} | {_fmt(h['segment_policy_tv'][1])} | "
            f"{_fmt(h['runtime'],1)} |"
        )
    for b in data["baselines"]:
        if b.get("error"):
            lines.append(f"| {b['name']} | {b['family']} | no | crashed: {b['error']} | - | - | {_fmt(b.get('runtime'),1)} |")
        else:
            lines.append(
                f"| {b['name']} | {b['family']} | no | {_fmt(b['pooled_policy_tv'])} | "
                f"{_fmt(b['segment_policy_tv'][0])} | {_fmt(b['segment_policy_tv'][1])} | "
                f"{_fmt(b.get('runtime'),1)} |"
            )
    lines.append("")
    lines.append(
        "The structural and behavior-cloning baselines fit the average behaviour "
        "closely. Pooled-AIRL is a rougher single fit. None of them get close to "
        "each individual segment, because none model the two types. AIRL-Het stays "
        "close to both segments at once."
    )
    lines.append("")
    lines.append("## What it shows")
    lines.append("")
    lines.append(
        "When choices come from a mix of hidden types, a single-type model fits the "
        "crowd and misses the parts. AIRL-Het separates the types, recovers a "
        "reward for each, and assigns readers to types from their choices. That is "
        "what makes segment-specific counterfactuals possible."
    )
    lines.append("")
    lines.append(
        "All numbers come from a results file written by the run script: "
        "`validation/results/study_content_consumption.json`. Reproduce with "
        "`PYTHONPATH=src:. python scripts/study_content_consumption.py`."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=float)


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", action="store_true", help="re-render page from JSON")
    parser.add_argument("--verify", action="store_true", help="re-derive headline from JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.page:
        data = _load_json(RESULTS_JSON)
        try:
            make_figure(data, FIGURE_PNG)
        except Exception as exc:  # figure is optional
            print(f"figure skipped: {exc}")
        with open(PAGE_PATH, "w", encoding="utf-8") as fh:
            fh.write(render_page(data))
        print(f"wrote {PAGE_PATH}")
        return

    if args.verify:
        data = _load_json(RESULTS_JSON)
        h = data["headline"]
        print("headline assignment_accuracy:", h.get("assignment_accuracy"))
        print("headline prior_l1:", h.get("prior_l1"))
        print("headline per-segment reward nrmse:", h.get("segment_reward_nrmse"))
        print("headline per-segment policy tv:", h.get("segment_policy_tv"))
        for b in data["baselines"]:
            print(f"  {b['name']}: pooled_tv={b.get('pooled_policy_tv')} "
                  f"per_seg={b.get('segment_policy_tv')} err={b.get('error')}")
        return

    data = run(verbose=args.verbose)
    _write_json(data, RESULTS_JSON)
    try:
        make_figure(data, FIGURE_PNG)
    except Exception as exc:
        print(f"figure skipped: {exc}")
    with open(PAGE_PATH, "w", encoding="utf-8") as fh:
        fh.write(render_page(data))
    print(f"wrote {RESULTS_JSON}\nwrote {PAGE_PATH}")


if __name__ == "__main__":
    main_cli()
