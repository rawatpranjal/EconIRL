"""Simulation study: Rust (1987) bus engine replacement.

Generates ``validation/results/sim_rust_bus.json`` and renders
``docs/simulation_studies/rust_bus.md`` from it via the shared harness
(``validation/benchmark/harness.py``): raw per-replication facts, page as a pure
function of the JSON, crashes first class, no fabricated metrics.

Usage:
    python scripts/sim_rust_bus.py [--verbose]      # run + write JSON
    python scripts/sim_rust_bus.py --page           # regenerate the page
    python scripts/sim_rust_bus.py --verify         # re-derive table from JSON
    python scripts/sim_rust_bus.py --only-estimator NAME   # retry one, merge
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from econirl.environments import RustBusEnvironment  # noqa: E402
from validation.benchmark.harness import Cell, RosterEntry, main_cli  # noqa: E402
from validation.benchmark.runner import _action_reward, _linear_utility  # noqa: E402

RESULTS_JSON = os.path.join(_ROOT, "validation", "results", "sim_rust_bus.json")
PAGE_PATH = os.path.join(_ROOT, "docs", "simulation_studies", "rust_bus.md")
FIGURE_PNG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                          "rust_bus_dgp.png")
RESULTS_FIG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                           "rust_bus_results.png")
SCALING_FIG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                           "rust_bus_scaling.png")
CURVE_FIG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                         "rust_bus_reward_curve.png")
REWARD_FIG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                          "rust_bus_reward.png")

# The canonical recoverable cell: identical to the prior benchmark ladder's
# simple_binary configuration, so this page genuinely re-homes that cell.
# The headline cell is 20 mileage bins. The scaling sweep reruns a trimmed
# roster at 15 and 30 bins to trace compute and accuracy against problem size.
HEADLINE_BINS = 20
SCALING_BINS = (15, 30)
ENV = dict(num_mileage_bins=HEADLINE_BINS, operating_cost=0.01,
           replacement_cost=2.0, discount_factor=0.95)


def _env(num_mileage_bins=HEADLINE_BINS):
    return RustBusEnvironment(**{**ENV, "num_mileage_bins": num_mileage_bins})


# ---------------------------------------------------------------------------
# Roster (uniform .estimate(panel, spec, problem, transitions) path)
# ---------------------------------------------------------------------------


def _run_nfxp(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(inner_solver="hybrid", inner_tol=1e-10,
                        inner_max_iter=100000, compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_ccp(env, panel):
    from econirl.estimation import CCPEstimator

    est = CCPEstimator(num_policy_iterations=5, compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_mpec(env, panel):
    from econirl.estimation.mpec import MPECConfig, MPECEstimator

    est = MPECEstimator(config=MPECConfig(solver="sqp", outer_max_iter=200, tol=1e-8, constraint_tol=1e-6),
                        compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_nnes(env, panel):
    from econirl.estimation.nnes import NNESEstimator

    est = NNESEstimator(hidden_dim=64, v_epochs=800, n_outer_iterations=5,
                        compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_sees(env, panel):
    from econirl.estimation.sees import SEESEstimator

    # Basis must span the value function: bspline basis_dim >= num_states (20).
    # The cost coefficients live on very different scales here (per-bin
    # operating cost vs a flat replacement cost), which stretches the
    # optimization landscape: the default 500 L-BFGS iterations stop
    # mid-descent. More iterations plus a data-driven extra start reach the
    # optimum the basis already represents exactly.
    est = SEESEstimator(basis_type="bspline", basis_dim=20, warm_start_value=True,
                        penalty_weight=10.0, max_iter=3000, num_theta_starts=3,
                        compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_tdccp(env, panel):
    from econirl.estimation import TDCCPConfig, TDCCPEstimator

    est = TDCCPEstimator(config=TDCCPConfig(hidden_dim=64, avi_iterations=15,
                                            epochs_per_avi=15, compute_se=False, verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_ufxp(env, panel):
    from econirl.estimation import UFXPEstimator

    # Bray's unnested fixed point with the paper's optimal weighting (OUFXP):
    # closed form for linear utility, MLE-efficient, with standard errors from
    # the efficient moment variance.
    est = UFXPEstimator(weights="optimal", verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_mce_irl(env, panel):
    from econirl.estimation import MCEIRLConfig, MCEIRLEstimator

    est = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.05, outer_max_iter=100,
                                              inner_max_iter=2000, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_maxent_irl(env, panel):
    from econirl.contrib.maxent_irl import MaxEntIRLEstimator

    # Feed the action-dependent features: a state-only reward cannot represent
    # the action contrast that drives the keep/replace choice. learning_rate
    # drives the Adam step (a fixed scalar step overshoots the mileage-cost
    # coordinate, whose feature column is ~19x the replacement indicator's).
    est = MaxEntIRLEstimator(inner_tol=1e-8, inner_max_iter=5000, outer_max_iter=500,
                             learning_rate=0.05, compute_hessian=False, verbose=False)
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_iq_learn(env, panel):
    from econirl.estimation.iq_learn import IQLearnConfig, IQLearnEstimator

    # q_type="linear" uses the feature structure; a tabular Q does not
    # propagate to unvisited states.
    est = IQLearnEstimator(config=IQLearnConfig(q_type="linear", divergence="chi2",
                                                alpha=3.0, max_iter=2000, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_gladius(env, panel):
    from econirl.estimation import GLADIUSConfig, GLADIUSEstimator

    est = GLADIUSEstimator(config=GLADIUSConfig(max_epochs=300, q_hidden_dim=128,
                                                v_hidden_dim=128, q_lr=1e-4, v_lr=1e-4,
                                                patience=60, compute_se=False, verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_airl(env, panel):
    from econirl.estimation import AIRLConfig, AIRLEstimator

    # reward_arg="state_action": the default "state" marginalizes the reward
    # across actions. The recovered reward is in its own parameterization by
    # design, so policy TV is the right scorecard.
    est = AIRLEstimator(config=AIRLConfig(reward_type="linear", reward_arg="state_action",
                                          reward_lr=0.01, discriminator_steps=10,
                                          max_rounds=300, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_firl(env, panel):
    from econirl.estimation.f_irl import FIRLEstimator

    # fkl is the estimator's validated divergence for state-action cells: its
    # log density-ratio gradient is bounded, where the chi2 ratio gradient is
    # unbounded and saturates the reward clip on near-deterministic experts.
    # reward_clip=10 matches the natural cost scale (the estimator default).
    est = FIRLEstimator(f_divergence="fkl", lr=0.2, max_iter=400, reward_clip=10.0,
                        verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_bc(env, panel):
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator

    est = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_deep_mce_irl(env, panel):
    from types import SimpleNamespace

    from econirl.estimators.mceirl_neural import MCEIRLNeural

    # sklearn-style .fit interface; adapted to the uniform result shape. coef_
    # is the neural reward projected onto the linear features, so the regret
    # transfer uses that projection, not the raw network.
    m = MCEIRLNeural(n_states=int(env.num_states), n_actions=int(env.num_actions),
                     discount=float(env.problem_spec.discount_factor),
                     max_epochs=200, verbose=False)
    m.fit(panel, features=np.asarray(env.feature_matrix),
          transitions=np.asarray(env.transition_matrices))
    return SimpleNamespace(parameters=m.coef_, standard_errors=None, policy=m.policy_,
                           value_function=m.value_, converged=bool(m.converged_))


def _run_max_margin(env, panel):
    from econirl.contrib.max_margin_irl import MaxMarginIRLEstimator

    # Requires a reward spec (LinearReward/ActionDependentReward), not the
    # structural LinearUtility wrapper.
    est = MaxMarginIRLEstimator(max_iterations=50, compute_hessian=False, verbose=False)
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


ROSTER = (
    RosterEntry("NFXP", "structural", _run_nfxp, uses_transitions=True),
    RosterEntry("CCP", "structural", _run_ccp, uses_transitions=True),
    RosterEntry("TD-CCP", "structural", _run_tdccp, uses_transitions=True),
    RosterEntry("MCE-IRL", "behavioral", _run_mce_irl, uses_transitions=True),
    RosterEntry("GLADIUS", "behavioral", _run_gladius, uses_transitions=True),
)

# Trimmed roster for the scaling sweep: five representative lines, a strict
# subset of ROSTER (same _run_* and uses_transitions). Thirteen lines on one
# plot is unreadable and slow.
SCALING_ROSTER = (
    RosterEntry("NFXP", "structural", _run_nfxp, uses_transitions=True),
    RosterEntry("CCP", "structural", _run_ccp, uses_transitions=True),
    RosterEntry("MCE-IRL", "behavioral", _run_mce_irl, uses_transitions=True),
    RosterEntry("GLADIUS", "behavioral", _run_gladius, uses_transitions=True),
)


DIAGNOSES = {
    "UFXP": "Unnested fixed point (Bray; Oguz and Bray 2026) with the paper's "
            "optimal weighting. The value function is eliminated before any "
            "parameter search, so the linear case is closed form and as "
            "efficient as maximum likelihood.",
    "MCE-IRL": "Its convergence indicator reports whether the gradient norm crossed "
               "the tolerance. The objective often plateaus first, so it "
               "can read False while the policy is essentially exact.",
    "MaxEnt-IRL": "It trails MCE-IRL because trajectory-entropy matching is "
                  "not the causal choice model that generated the data.",
    "IQ-Learn": "Uses the linear feature structure. A tabular Q-table would "
                "not propagate to unvisited states.",
}

EXCLUDED = [
    {"name": "MPEC, NNES, SEES, UFXP", "reason": "Other-tier structural "
     "estimators; this study shows the core structural family (NFXP, CCP, "
     "TD-CCP). They ran and their records remain in the results file"},
    {"name": "AIRL, Deep-MCE-IRL", "reason": "core IRL methods shown on the "
     "gridworld study, whose deterministic state-only reward suits them"},
    {"name": "MaxEnt-IRL, IQ-Learn", "reason": "trajectory MaxEnt and inverse "
     "soft-Q; not part of the core roster"},
    {"name": "AIRL2 / AIRL2", "reason": "designed for latent-type "
     "heterogeneity; this panel has one agent type"},
    {"name": "MMP, GAIL", "reason": "too slow for this page's per-fit budget"},
    {"name": "GCL, DeepMaxEnt-IRL, Bayesian-IRL", "reason": "research code, "
     "not benchmarked"},
    {"name": "MaxMargin-IRL", "reason": "its unit-norm reward direction has "
     "no link to the choice model's noise scale, so it is not a "
     "like-for-like baseline on this problem; it ran 3/3 and its raw records "
     "remain in the results file"},
    {"name": "f-IRL, BC", "reason": "they recover a tabular reward and a "
     "choice-probability table, objects in a different parameterization; "
     "their raw records remain in the results file"},
]

_HEADLINE_CELL = Cell(
    cell_id="rust_bus",
    label="Bus engine (20 mileage bins)",
    description=(
        "Harold Zurcher's bus-engine replacement problem (Rust 1987): a "
        "binary keep-or-replace choice over a discretized mileage state, "
        "with linear operating and replacement costs. "
        f"`RustBusEnvironment(num_mileage_bins={ENV['num_mileage_bins']}, "
        f"operating_cost={ENV['operating_cost']}, "
        f"replacement_cost={ENV['replacement_cost']}, "
        f"discount_factor={ENV['discount_factor']})`."
    ),
    env_factory=_env,
    roster=ROSTER,
    n_individuals=500,
    n_periods=80,
    seed=42,
    n_replications=30,
    fit_timeout=600,
    figure=FIGURE_PNG,
    results_figure=RESULTS_FIG,
)


def _scaling_cell(num_mileage_bins):
    """One scaling-only cell at a given mileage-grid size.

    Runs the trimmed SCALING_ROSTER to feed the scaling figure. Renders no
    tables or figures of its own.
    """
    return Cell(
        cell_id=f"rust_bus_{num_mileage_bins}",
        label=f"Bus engine ({num_mileage_bins} mileage bins)",
        description=(
            "Bus-engine replacement at a smaller or larger mileage grid. "
            f"`RustBusEnvironment(num_mileage_bins={num_mileage_bins}, "
            f"operating_cost={ENV['operating_cost']}, "
            f"replacement_cost={ENV['replacement_cost']}, "
            f"discount_factor={ENV['discount_factor']})`."
        ),
        env_factory=(lambda n=num_mileage_bins: _env(n)),
        roster=SCALING_ROSTER,
        n_individuals=500,
        n_periods=80,
        seed=42,
        n_replications=5,
        fit_timeout=600,
        scaling_only=True,
    )


# Headline first so single-cell runs (--only-cell rust_bus) still work.
CELLS = (
    _HEADLINE_CELL,
    *(_scaling_cell(n) for n in SCALING_BINS),
)

NARRATIVE = {
    "title": "Bus engine replacement",
    "intro": (
        "The canonical structural benchmark (Rust 1987). A single agent decides "
        "each period whether to keep paying a mileage-dependent operating cost "
        "or pay a fixed replacement cost to reset the bus engine. The "
        "data-generating process is fully known, so the table reports the exact "
        "recovered cost parameters, the distance between each estimator's "
        "policy and the true one, and the welfare lost when the recovered model "
        "is carried into three counterfactual worlds.\n"
        "\n"
        "![A mechanic rebuilding a bus engine on a workbench in a 1943 garage, "
        "with parts bins lining the wall behind him.]"
        "(../_static/simulation_studies/rust_bus_garage_1943.jpg)\n"
        "\n"
        "*Rebuilding a bus engine at the Tennessee Coach Company garage in "
        "Knoxville, September 1943. Photograph by Esther Bubley, "
        "[Library of Congress FSA/OWI collection]"
        "(https://commons.wikimedia.org/wiki/File:Tennessee-coach-company-mechanic-knox-1943-tn1.jpg), "
        "digital ID fsa.8d33516, public domain. Rust (1987) studied the same "
        "decision at the Madison, Wisconsin bus fleet.*\n"
        "\n"
        "## The data-generating process\n"
        "\n"
        "Mileage sits on a discrete grid $s \\in \\{0, \\ldots, S-1\\}$. Keeping "
        "the engine (action $0$) pays a per-bin operating cost and lets mileage "
        "drift up by $\\Delta s \\in \\{0, 1, 2\\}$. Replacing it (action $1$) "
        "pays a flat cost and resets the engine:\n"
        "\n"
        "$$\n"
        "u_\\theta(s, a) =\n"
        "\\begin{cases}\n"
        "-\\theta_{\\mathrm{oc}}\\, s & a = 0 \\ (\\text{keep}) \\\\\n"
        "-\\theta_{\\mathrm{rc}} & a = 1 \\ (\\text{replace}),\n"
        "\\end{cases}\n"
        "\\qquad\n"
        "P(s' \\mid s, 1) = p_{\\Delta s'},\\ s' \\in \\{0, 1, 2\\},\n"
        "$$\n"
        "\n"
        "where replacement resets the engine and the same one-period drift "
        "$p = (p_0, p_1, p_2)$ then applies from zero, so the post-replacement "
        "state lands on $\\{0, 1, 2\\}$ rather than exactly on zero.\n"
        "\n"
        "The true parameters are $\\theta_{\\mathrm{oc}} = 0.01$ and "
        "$\\theta_{\\mathrm{rc}} = 2.0$. The agent discounts at $\\beta$ and "
        "faces i.i.d. logit taste shocks (scale $\\sigma = 1$), so behavior "
        "solves the soft Bellman equation\n"
        "\n"
        "$$\n"
        "V(s) = \\log \\sum_{a} \\exp\\Bigl(u_\\theta(s,a) + "
        "\\beta\\, \\mathbb{E}\\bigl[V(s') \\mid s,a\\bigr]\\Bigr),\n"
        "\\qquad \\pi^*(a \\mid s) \\propto \\exp\\Bigl(u_\\theta(s,a) + "
        "\\beta\\, \\mathbb{E}\\bigl[V(s') \\mid s,a\\bigr]\\Bigr),\n"
        "$$\n"
        "\n"
        "and the panel simulates $N$ buses for $T$ periods from $\\pi^*$. The "
        "figure shows the sawtooth mileage paths (rising drift, replacement "
        "resets) and the declining value of holding higher mileage. Every "
        "estimator below sees the same panels."
    ),
    "cells": {
        "rust_bus": {
            "after": (
                "The structural family (NFXP, CCP, MPEC, NNES, SEES, TD-CCP, "
                "UFXP) recovers the cost parameters on the same scale as the "
                "truth, so Param RMSE applies to it alone. MCE-IRL uses the same "
                "linear cost features and recovers the same scale, but the IRL "
                "family is scored on behavior and regret because reward is only "
                "partially identified from behavior in general. Estimators that "
                "recover a transferable reward adapt under the interventions. "
                "Policy-only methods keep their old policy, which is why "
                "their Type C regret is large."
            ),
        },
    },
    "scaling_intro": (
        "The same study at three mileage-grid sizes (15, 20, 30 bins). Each "
        "line is one estimator: fit time on the left, policy total variation "
        "on the right. The structural methods stay cheap and accurate across "
        "sizes. The neural and IRL methods cost more and are less accurate. "
        "These are small fits, so the compute lines reflect fixed overhead as "
        "much as problem size, and the trend is not a clean monotone curve."
    ),
    "script": "scripts/sim_rust_bus.py",
    "results_rel": "validation/results/sim_rust_bus.json",
    "extra_sections": (
        "## Reward and structure\n"
        "\n"
        "Mileage is the natural ordering, so reward plots cleanly against the "
        "state index. The keep cost slopes down with mileage. The replacement "
        "cost is flat. Where they cross is near the replacement threshold. The "
        "recovered reward (dashed) tracks the true reward, and the optimal value "
        "falls as mileage rises.\n"
        "\n"
        "![Reward against mileage, keep versus replace, with optimal value]"
        "(../_static/simulation_studies/rust_bus_reward_curve.png)\n"
        "\n"
        "The same reward as a state-by-action heatmap puts the true and recovered "
        "rewards side by side on one color scale.\n"
        "\n"
        "![True and recovered reward heatmaps]"
        "(../_static/simulation_studies/rust_bus_reward.png)\n"
    ),
}

_ACTIONS = ["keep", "replace"]


def _make_curve_fig(data):
    from validation.benchmark.figures import (_structural_mean_params,
                                              reward_curve)

    _, theta = _structural_mean_params(data, "rust_bus")
    reward_curve(_env(), CURVE_FIG, params=theta, action_labels=_ACTIONS,
                 state_label="mileage bin $s$")


def _make_reward_fig(data):
    from validation.benchmark.figures import (_structural_mean_params,
                                              reward_heatmap)

    name, theta = _structural_mean_params(data, "rust_bus")
    if theta is None:
        return
    reward_heatmap(_env(), theta, REWARD_FIG,
                   title=f"Recovered reward from {name}")


EXTRA_FIGURES = [(CURVE_FIG, _make_curve_fig), (REWARD_FIG, _make_reward_fig)]


if __name__ == "__main__":
    main_cli(cells=CELLS, title="Simulation study: bus engine replacement",
             narrative=NARRATIVE, diagnoses=DIAGNOSES, excluded=EXCLUDED,
             results_json=RESULTS_JSON, page_path=PAGE_PATH,
             scaling_figure=SCALING_FIG, extra_figures=EXTRA_FIGURES)
