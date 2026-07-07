"""Simulation study: optimal replacement of vehicles (RDW scrappage).

Generates ``validation/results/study_vehicle_scrappage.json`` and renders
``docs/simulation_studies/vehicle_scrappage.md`` from it via the shared harness
(``validation/benchmark/harness.py``), inheriting the honesty contract: raw
per-replication facts, page as a pure function of the JSON, crashes first class,
no fabricated metrics.

Usage::

    python scripts/study_vehicle_scrappage.py [--verbose]      # run + write JSON
    python scripts/study_vehicle_scrappage.py --page           # regenerate the page
    python scripts/study_vehicle_scrappage.py --verify         # re-derive table from JSON
    python scripts/study_vehicle_scrappage.py --only-estimator NAME   # retry one, merge
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from econirl.environments.rdw_scrappage import RDWScrapageEnvironment  # noqa: E402
from validation.benchmark.harness import Cell, RosterEntry, main_cli  # noqa: E402
from validation.benchmark.runner import _action_reward, _linear_utility  # noqa: E402

RESULTS_JSON = os.path.join(_ROOT, "validation", "results", "study_vehicle_scrappage.json")
PAGE_PATH = os.path.join(_ROOT, "docs", "simulation_studies", "vehicle_scrappage.md")
FIGURE_PNG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                          "vehicle_scrappage_dgp.png")
RESULTS_FIG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                           "vehicle_scrappage_results.png")
CURVE_FIG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                         "vehicle_scrappage_reward_curve.png")
REWARD_FIG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                          "vehicle_scrappage_reward.png")

# ---- DGP configuration ----
# 25 age bins x 3 defect levels = 75 states, 2 actions (keep/scrap).
# Four reward parameters: age_cost, minor_defect_cost, major_defect_cost,
# replacement_cost. Action-contrast rank is 4/4: the scrap action has a
# constant feature vector [-replacement_cost, 0, 0, 0] while the keep action
# varies with age and defect level across all 75 states, so all four parameters
# are identified from observed choices.
ENV = dict(discount_factor=0.95, seed=0)


def _env():
    return RDWScrapageEnvironment(**ENV)


# ---------------------------------------------------------------------------
# Roster: _run_*(env, panel) -> result with .parameters, .standard_errors,
#         .policy, .value_function, .converged
# ---------------------------------------------------------------------------


def _run_nfxp(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(
        inner_solver="hybrid", inner_tol=1e-10,
        inner_max_iter=100_000, compute_hessian=True, verbose=False,
    )
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_ccp(env, panel):
    from econirl.estimation import CCPEstimator

    est = CCPEstimator(num_policy_iterations=5, compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_mpec(env, panel):
    from econirl.estimation.mpec import MPECConfig, MPECEstimator

    # solver="sqp" is the real constrained MLE; "slsqp" is a legacy alias for
    # the augmented-Lagrangian penalty solver and does not check optimality.
    est = MPECEstimator(
        config=MPECConfig(solver="sqp", outer_max_iter=200, tol=1e-8, constraint_tol=1e-6),
        compute_hessian=True, verbose=False,
    )
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_ufxp(env, panel):
    from econirl.estimation import UFXPEstimator

    # Bray's unnested fixed point with the paper's optimal weighting (OUFXP):
    # closed form for linear utility, MLE-efficient, with standard errors from
    # the efficient moment variance.
    est = UFXPEstimator(weights="optimal", verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_nnes(env, panel):
    from econirl.estimation.nnes import NNESEstimator

    est = NNESEstimator(hidden_dim=64, v_epochs=800, n_outer_iterations=5,
                        compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_mce_irl(env, panel):
    from econirl.estimation import MCEIRLConfig, MCEIRLEstimator

    est = MCEIRLEstimator(config=MCEIRLConfig(
        learning_rate=0.05, outer_max_iter=80, inner_max_iter=1000,
        compute_se=False, verbose=False,
    ))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


ROSTER = (
    RosterEntry("NFXP",   "structural", _run_nfxp, uses_transitions=True),
    RosterEntry("CCP",    "structural", _run_ccp, uses_transitions=True),
    RosterEntry("MPEC",   "structural", _run_mpec,  timeout=180,
                uses_transitions=True),
    RosterEntry("UFXP",   "structural", _run_ufxp,  timeout=180,
                uses_transitions=True),
    RosterEntry("NNES",   "structural", _run_nnes,  timeout=300,
                uses_transitions=True),
    RosterEntry("MCE-IRL","behavioral", _run_mce_irl, uses_transitions=True),
)

# ---------------------------------------------------------------------------
# Diagnoses, excluded, cell, narrative
# ---------------------------------------------------------------------------

DIAGNOSES = {
    "NFXP": (
        "Full-solution MLE with a nested Bellman fixed-point inner loop. "
        "Quadratic convergence near the optimum. "
        "All four parameters are identified because the scrap action has a "
        "constant feature vector while the keep action varies with age and "
        "defect level, so the action-contrast covers all four reward dimensions."
    ),
    "CCP": (
        "CCP uses a first-step nonparametric policy estimate to avoid the "
        "inner Bellman loop. One policy-iteration step corrects the bias from "
        "the nonparametric first stage. The replacement state is absorbing in "
        "reverse: scrapping always resets to age 0, so the stationary "
        "distribution has good coverage near the reset state but sparser "
        "coverage at very high ages."
    ),
    "MPEC": (
        "Mathematical programming with equilibrium constraints. "
        "MPEC is not in the CAPABILITIES registry (so run_form does not "
        "surface it automatically) but runs correctly via the direct "
        ".estimate() path. Uses solver='sqp' for real constrained MLE; "
        "the legacy 'slsqp' alias checks only Bellman feasibility."
    ),
    "UFXP": (
        "Bray's unnested fixed point with optimal GMM weighting (OUFXP). "
        "Closed form for linear utility: no inner Bellman solve and no outer "
        "gradient iteration. MLE-efficient for the linear reward class. "
        "Standard errors come from the efficient moment variance."
    ),
    "NNES": (
        "Semi-parametric structural estimator that approximates the value "
        "function with a neural network (NPL-based, Nguyen 2025). "
        "Phase 1 trains a V-network on the NPL value target; Phase 2 "
        "maximizes the profiled pseudo-likelihood through the network. "
        "The zero Jacobian property of the NPL mapping makes first-order "
        "errors in the V-network orthogonal to the score."
    ),
    "MCE-IRL": (
        "Its convergence indicator mirrors the inner optimizer's success "
        "status. The optimizer can stop short while the recovered policy is "
        "already accurate, so the flag can read False on an accurate fit."
    ),
}

EXCLUDED = [
    {
        "name": "IQ-Learn, f-IRL",
        "reason": (
            "not separately identified from choices on this problem; "
            "reward is only partially identified from behavior"
        ),
    },
    {
        "name": "NeuralGLADIUS, SEES, TD-CCP",
        "reason": (
            "correct behavioral or structural estimators but slower on "
            "the 75-state space; the roster already covers the structural "
            "and behavioral families"
        ),
    },
    {
        "name": "MaxEnt-IRL, MaxMargin-IRL, NeuralAIRL, Deep-MCE-IRL",
        "reason": (
            "trajectory-entropy and max-margin objectives are not the "
            "choice model that generated the data; neural AIRL adds "
            "compute without new information here"
        ),
    },
]

CELLS = (
    Cell(
        cell_id="vehicle_scrappage",
        label="Vehicle scrappage (75 states, 2 actions)",
        description=(
            "Vehicle scrappage decision based on Dutch RDW inspection data. "
            "A vehicle owner observes the car's age bin "
            "$a \\in \\{0, \\ldots, 24\\}$ and the APK defect severity "
            "$d \\in \\{\\text{pass}, \\text{minor}, \\text{major}\\}$ from "
            "the annual inspection and decides whether to keep or scrap the "
            "vehicle. The flat state index is $s = 3a + d$, giving 75 states "
            "and 2 actions. Reward is linear in four features:\n\n"
            "$U(s, \\text{keep}) = -\\theta_{\\text{age}}\\,a "
            "- \\theta_{\\text{minor}}\\,\\mathbf{1}\\{d=1\\} "
            "- \\theta_{\\text{major}}\\,\\mathbf{1}\\{d=2\\}$; "
            "$U(s, \\text{scrap}) = -\\theta_{\\text{rc}}$.\n\n"
            "True parameters: $\\theta = [0.15,\\;0.5,\\;1.5,\\;3.0]$. "
            "If keep: age increments by 1, defect level transitions "
            "stochastically with age-dependent probabilities. "
            "If scrap: state resets to $(a=0,\\;d=0)$. "
            f"``RDWScrapageEnvironment(discount_factor={ENV['discount_factor']}, "
            f"seed={ENV['seed']})``."
        ),
        env_factory=_env,
        roster=ROSTER,
        n_individuals=200,
        n_periods=35,
        seed=42,
        n_replications=30,
        fit_timeout=240,
        param_block=True,
        figure=FIGURE_PNG,
        results_figure=RESULTS_FIG,
    ),
)

NARRATIVE = {
    "title": "Optimal replacement: vehicle scrappage (RDW)",
    "intro": (
        "A vehicle owner decides each year whether to keep running a car or "
        "scrap it and buy a new one. The decision depends on the car's age and "
        "how well it passed the mandatory Dutch APK roadworthiness inspection. "
        "The model is an optimal stopping problem in the spirit of Rust (1987), "
        "applied to vehicle scrappage.\n"
        "\n"
        "## The data-generating process\n"
        "\n"
        "The state is a pair: vehicle age bin $a$ and APK defect level $d$. "
        "Age runs from 0 to 24 years. The defect level is pass (0), minor "
        "defects (1), or major defects / rejection (2). The flat state index "
        "is $s = 3a + d$, giving 75 states.\n"
        "\n"
        "Each period the owner chooses to keep or scrap the vehicle. "
        "If keeping: age increments by one year and the defect level "
        "transitions stochastically. Older cars face higher probabilities "
        "of moving to a worse defect level. "
        "If scrapping: the state resets to a new car at $(a = 0,\\;d = 0)$.\n"
        "\n"
        "The reward is linear in four features:\n"
        "\n"
        "$$\n"
        "u_\\theta(s, a) = \\begin{cases}\n"
        "-\\theta_{\\text{age}}\\,a "
        "- \\theta_{\\text{minor}}\\,\\mathbf{1}\\{d=1\\} "
        "- \\theta_{\\text{major}}\\,\\mathbf{1}\\{d=2\\} "
        "& a = \\text{keep} \\\\\n"
        "-\\theta_{\\text{rc}} & a = \\text{scrap}\n"
        "\\end{cases}\n"
        "$$\n"
        "\n"
        "where $\\theta_{\\text{age}}$ is the per-year operating cost, "
        "$\\theta_{\\text{minor}}$ is the penalty for minor defects, "
        "$\\theta_{\\text{major}}$ is the penalty for major defects, and "
        "$\\theta_{\\text{rc}}$ is the replacement cost. "
        "The true parameters are "
        "$\\theta = [0.15,\\;0.5,\\;1.5,\\;3.0]$.\n"
        "\n"
        "Agents discount future payoffs at $\\beta = 0.95$ and face "
        "i.i.d. logit taste shocks (scale $\\sigma = 1$). Their behaviour "
        "solves the soft Bellman equation. The action-contrast feature vector "
        "is $[-a,\\;-\\mathbf{1}\\{d=1\\},\\;-\\mathbf{1}\\{d=2\\},\\;1]$ for "
        "the keep-minus-scrap difference at each state. Because age $a$ and "
        "both defect indicators vary independently across the 75 states, the "
        "contrast feature matrix has rank 4 and all four parameters are "
        "identified from observed choices. "
        "The optimal policy scraps at high ages and after major defect "
        "findings: only part of the state space lies on the equilibrium path, "
        "because vehicles that reach old age with clean inspections are rare. "
        "The panel simulates $N$ agents for $T$ periods from the true optimal "
        "policy. The figure shows simulated age-defect paths and the optimal "
        "value function."
    ),
    "cells": {
        "vehicle_scrappage": {
            "after": (
                "The structural family (NFXP, CCP, MPEC, UFXP, NNES) recovers "
                "all four parameters on the same scale as the truth, so Param RMSE "
                "applies to them alone. MCE-IRL here uses the same linear features "
                "and recovers the same values, but its weights stay out of the "
                "recovery table because an IRL reward is only partially identified "
                "in general. Policy TV and regret are the right scorecards for the "
                "behavioral family."
            ),
        },
    },
    "script": "scripts/study_vehicle_scrappage.py",
    "results_rel": "validation/results/study_vehicle_scrappage.json",
    "extra_sections": (
        "## Reward and structure\n"
        "\n"
        "Reward plots against vehicle age. Each age carries three defect levels, "
        "so the keep line spreads into a band as defects raise the running cost. "
        "The scrap line is flat. The recovered reward (dashed) tracks the truth, "
        "and the optimal value falls as the car ages.\n"
        "\n"
        "![Reward against vehicle age, keep versus scrap, with optimal value]"
        "(../_static/simulation_studies/vehicle_scrappage_reward_curve.png)\n"
        "\n"
        "The same reward as a state-by-action heatmap puts the true and recovered "
        "rewards side by side on one color scale.\n"
        "\n"
        "![True and recovered reward heatmaps]"
        "(../_static/simulation_studies/vehicle_scrappage_reward.png)\n"
    ),
}

_ACTIONS = ["keep", "scrap"]


def _ages(env):
    """Per-state age in years: s = 3*age + defect, so age = s // 3."""
    return np.arange(int(env.num_states)) // 3


def _make_curve_fig(data):
    from validation.benchmark.figures import (_structural_mean_params,
                                              reward_curve)

    _, theta = _structural_mean_params(data, "vehicle_scrappage")
    env = _env()
    reward_curve(env, CURVE_FIG, params=theta, action_labels=_ACTIONS,
                 state_label="vehicle age (years)", x=_ages(env))


def _make_reward_fig(data):
    from validation.benchmark.figures import (_structural_mean_params,
                                              reward_heatmap)

    name, theta = _structural_mean_params(data, "vehicle_scrappage")
    if theta is None:
        return
    reward_heatmap(_env(), theta, REWARD_FIG,
                   title=f"Recovered reward from {name}")


EXTRA_FIGURES = [(CURVE_FIG, _make_curve_fig), (REWARD_FIG, _make_reward_fig)]


if __name__ == "__main__":
    main_cli(
        cells=CELLS,
        title="Simulation study: optimal replacement of vehicles (RDW scrappage)",
        narrative=NARRATIVE,
        diagnoses=DIAGNOSES,
        excluded=EXCLUDED,
        results_json=RESULTS_JSON,
        page_path=PAGE_PATH,
        extra_figures=EXTRA_FIGURES,
    )
