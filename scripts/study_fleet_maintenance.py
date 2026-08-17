"""Simulation study: fleet maintenance (multi-component bus engine replacement).

Generates ``validation/results/study_fleet_maintenance.json`` and renders
``docs/simulation_studies/fleet_maintenance.md`` from it via the shared harness
(``validation/benchmark/harness.py``), inheriting the honesty contract: raw
per-replication facts, page as a pure function of the JSON, crashes first class,
no fabricated metrics.

Usage::

    python scripts/study_fleet_maintenance.py [--verbose]      # run + write JSON
    python scripts/study_fleet_maintenance.py --page           # regenerate the page
    python scripts/study_fleet_maintenance.py --verify         # re-derive table from JSON
    python scripts/study_fleet_maintenance.py --only-estimator NAME   # retry one, merge
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from econirl.environments.multi_component_bus import MultiComponentBusEnvironment  # noqa: E402
from validation.benchmark.harness import Cell, RosterEntry, main_cli  # noqa: E402
from validation.benchmark.runner import _action_reward, _linear_utility  # noqa: E402

RESULTS_JSON = os.path.join(_ROOT, "validation", "results", "study_fleet_maintenance.json")
PAGE_PATH = os.path.join(_ROOT, "docs", "simulation_studies", "fleet_maintenance.md")
_STATIC = os.path.join(_ROOT, "docs", "_static", "simulation_studies")
FIGURE_PNG = os.path.join(_STATIC, "fleet_maintenance_dgp.png")
RESULTS_FIG = os.path.join(_STATIC, "fleet_maintenance_results.png")
REWARD_FIG = os.path.join(_STATIC, "fleet_maintenance_reward.png")
CURVE_FIG = os.path.join(_STATIC, "fleet_maintenance_reward_curve.png")

# ---- DGP configuration ----
# K=3 components x M=6 mileage bins = 216 states, 2 actions (keep/replace).
# Materially larger than the single-component siblings (route_choice 25,
# stockpiling 20). Full-roster fits (5 estimators x 2 reps) run in minutes.
# The action-contrast rank is 3/3: all three reward parameters are identified
# from choice data — the replacement indicator and the two mileage terms
# differentiate across actions.
ENV = dict(
    K=3,
    M=6,
    seed=0,
    discount_factor=0.95,
    operating_cost=1.0,
    quadratic_cost=0.5,
    replacement_cost=3.0,
)


def _env():
    return MultiComponentBusEnvironment(**ENV)


# ---------------------------------------------------------------------------
# Roster: _run_*(env, panel) -> result with .parameters, .standard_errors,
#         .policy, .value_function, .converged
# ---------------------------------------------------------------------------


def _run_nfxp(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-10,
        inner_max_iter=100_000,
        compute_hessian=True,
        verbose=False,
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
        compute_hessian=True,
        verbose=False,
    )
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_mce_irl(env, panel):
    from econirl.estimation import MCEIRLConfig, MCEIRLEstimator

    est = MCEIRLEstimator(
        config=MCEIRLConfig(
            learning_rate=0.05,
            outer_max_iter=80,
            inner_max_iter=1000,
            compute_se=False,
            verbose=False,
        )
    )
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


ROSTER = (
    RosterEntry("NFXP", "structural", _run_nfxp, uses_transitions=True),
    RosterEntry("CCP", "structural", _run_ccp, uses_transitions=True),
    RosterEntry("MCE-IRL", "behavioral", _run_mce_irl, uses_transitions=True),
)

# ---------------------------------------------------------------------------
# Diagnoses, excluded, cell, narrative
# ---------------------------------------------------------------------------

DIAGNOSES = {
    "NFXP": (
        "Full-solution MLE with a nested Bellman fixed-point inner loop. "
        "Quadratic convergence near the optimum. "
        "All three parameters are identified from the action-dependent "
        "replacement indicator: the keep action has features [0, -x, -x^2] "
        "and the replace action has feature [-1, 0, 0], so the action-contrast "
        "varies across states and all three coordinates are recoverable."
    ),
    "CCP": (
        "CCP uses a first-step nonparametric policy estimate to avoid the "
        "inner Bellman loop. One policy-iteration step corrects the bias from "
        "the nonparametric first stage. Fast on the 216-state factored space "
        "because it avoids repeated value-iteration inner solves."
    ),
    "MPEC": (
        "Mathematical programming with equilibrium constraints. "
        "MPEC is not in the CAPABILITIES registry (so run_form does not "
        "surface it automatically) but runs correctly via the direct "
        ".estimate() path. Uses solver='sqp' for real constrained MLE; "
        "the legacy 'slsqp' alias checks only Bellman feasibility."
    ),
    "MCE-IRL": (
        "Its convergence indicator mirrors the inner optimizer's success "
        "status. The optimizer can stop short while the recovered policy is "
        "already accurate, so it can read False on an accurate fit."
    ),
}

EXCLUDED = [
    {
        "name": "MPEC",
        "reason": (
            "an Other-tier constrained-optimization form of the same MLE; "
            "NFXP and CCP carry the structural recovery here"
        ),
    },
    {
        "name": "IQ-Learn, f-IRL",
        "reason": (
            "not separately identified from choices on this problem; "
            "reward is only partially identified from behavior"
        ),
    },
    {
        "name": "NNES, SEES, TD-CCP, UFXP",
        "reason": (
            "correct structural estimators but slower on a 216-state factored "
            "space; NFXP and CCP already cover the structural family"
        ),
    },
    {
        "name": "MaxEnt-IRL, MaxMargin-IRL, AIRL, Deep-MCE-IRL",
        "reason": (
            "trajectory-entropy and max-margin objectives are not the "
            "choice model that generated the data; AIRL also requires a "
            "state-only reward and cannot represent the action contrast"
        ),
    },
    {
        "name": "NeuralGLADIUS",
        "reason": (
            "a model-free neural policy learner; on this 216-state factored "
            "problem it reduces to behavioral cloning on the choice likelihood "
            "with no structural edge. Its arena is scalability and predictive "
            "likelihood on large state spaces where full-solution methods become "
            "intractable, a regime outside these policy-recovery studies"
        ),
    },
]

CELLS = (
    Cell(
        cell_id="fleet_maintenance",
        label="Fleet maintenance (216 states, 2 actions)",
        description=(
            "Multi-component bus engine replacement. A fleet operator maintains "
            "$K = 3$ independent engine components, each tracked over $M = 6$ "
            "mileage bins. The joint state $s$ is encoded via mixed-radix: "
            "$s = m_0 + M m_1 + M^2 m_2$ where $m_k \\in \\{0, \\ldots, M-1\\}$ "
            "is the mileage bin of component $k$. This gives $M^K = 216$ states. "
            "Each period the operator chooses to keep all components or replace "
            "all components at once. Reward is linear in three features, with "
            "aggregate mileage $x(s) = \\sum_k m_k / M$: "
            "keep action has features $[0,\\,-x(s),\\,-x(s)^2]$ "
            "and replace action has features $[-1,\\,0,\\,0]$. "
            f"``MultiComponentBusEnvironment(K={ENV['K']}, M={ENV['M']}, "
            f"discount_factor={ENV['discount_factor']}, "
            f"operating_cost={ENV['operating_cost']}, "
            f"quadratic_cost={ENV['quadratic_cost']}, "
            f"replacement_cost={ENV['replacement_cost']}, "
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
    "title": "Fleet maintenance (multi-component bus engine replacement)",
    "intro": (
        "A fleet operator maintains a bus with $K$ independent engine components. "
        "Each period, the operator chooses to keep all components running or replace "
        "them all at once. Replacing is costly upfront but avoids the growing "
        "operating cost that accumulates as mileage rises. The problem is a "
        "multi-component extension of Rust (1987).\n"
        "\n"
        "## The data-generating process\n"
        "\n"
        "Each of the $K$ engine components ages through $M$ mileage bins "
        "$m_k \\in \\{0, \\ldots, M-1\\}$. The joint state is encoded via "
        "mixed-radix as a single flat index $s \\in \\{0, \\ldots, M^K - 1\\}$. "
        "This study uses $K = 3$ components and $M = 6$ bins, giving "
        "$M^K = 216$ states and 2 actions.\n"
        "\n"
        "Each component mileage advances stochastically each period by 0, 1, "
        "or 2 bins with probabilities $(0.3919,\\;0.5953,\\;0.0128)$, "
        "capped at the last bin. The components evolve independently, "
        "so the joint transition matrix is a Kronecker product of the "
        "per-component matrices. Replacing resets all components to bin 0 "
        "and applies one mileage draw.\n"
        "\n"
        "The reward is linear in three features. Let "
        "$x(s) = \\sum_k m_k / M$ be the aggregate normalized mileage:\n"
        "\n"
        "$$\n"
        "u_\\theta(s, a) = \\begin{cases}\n"
        "-\\theta_{\\mathrm{op}}\\,x(s) - "
        "\\theta_{\\mathrm{q}}\\,x(s)^2 & a = \\text{keep} \\\\\n"
        "-\\theta_{\\mathrm{rc}} & a = \\text{replace}\n"
        "\\end{cases}\n"
        "$$\n"
        "\n"
        "where $\\theta_{\\mathrm{rc}}$ is the replacement cost, "
        "$\\theta_{\\mathrm{op}}$ is the linear operating cost, and "
        "$\\theta_{\\mathrm{q}}$ is the quadratic wear cost. "
        "The true parameters are "
        "$\\theta = [\\theta_{\\mathrm{rc}},\\;\\theta_{\\mathrm{op}},"
        "\\;\\theta_{\\mathrm{q}}] = [3.0,\\;1.0,\\;0.5]$.\n"
        "\n"
        "Agents discount future payoffs at $\\beta = 0.95$ and face i.i.d. logit "
        "taste shocks (scale $\\sigma = 1$). Their behaviour solves the soft Bellman "
        "equation. The action-contrast feature vector is "
        "$[-1,\\;x(s),\\;x(s)^2]$ for each state $s$. "
        "Because $x(s)$ takes many distinct values across the 216 states, "
        "the contrast feature matrix has rank 3 and all three parameters are "
        "identified from observed choices. "
        "The operator replaces when aggregate mileage exceeds a threshold "
        "(approximately $x(s) > 1.65$ at the true parameters). "
        "The panel simulates $N$ agents for $T$ periods from the true optimal policy."
    ),
    "cells": {
        "fleet_maintenance": {
            "after": (
                "The structural family (NFXP, CCP) recovers all three "
                "parameters on the same scale as the truth, so Param RMSE applies "
                "to them alone. MCE-IRL here uses the same linear features and "
                "recovers the same values, but its weights stay out of the "
                "recovery table because an IRL reward is only partially identified "
                "in general. Policy TV and regret are the behavioral scorecards."
            ),
        },
    },
    "script": "scripts/study_fleet_maintenance.py",
    "results_rel": "validation/results/study_fleet_maintenance.json",
    "extra_sections": (
        "## Reward and structure\n"
        "\n"
        "The true and recovered rewards sit side by side as state-by-action "
        "heatmaps on one color scale. The 216 states are the factored mileage "
        "combinations. The replace action is flat. The keep action darkens as "
        "aggregate mileage rises.\n"
        "\n"
        "![True and recovered reward heatmaps]"
        "(../_static/simulation_studies/fleet_maintenance_reward.png)\n"
        "\n"
        "The raw state index is not ordered, because the state is a factored "
        "combination of three component mileages. Plotting reward against "
        "aggregate mileage $x(s) = \\sum_k m_k / M$ recovers the structural "
        "shape: the keep cost falls with mileage, the replace cost is flat, and "
        "the recovered reward (dashed) tracks the truth.\n"
        "\n"
        "![Reward against aggregate mileage, keep versus replace]"
        "(../_static/simulation_studies/fleet_maintenance_reward_curve.png)\n"
    ),
}

_ACTIONS = ["keep", "replace"]


def _aggregate_mileage(env):
    """Per-state aggregate mileage x(s), read off the keep-action feature.

    The keep action's operating-cost feature is -x(s) (feature column 1), so
    x(s) = -feature_matrix[s, keep, 1]. Pure from the env's features.
    """
    fm = np.asarray(env.feature_matrix)
    return -fm[:, 0, 1]


def _make_reward_fig(data):
    from validation.benchmark.figures import _structural_mean_params, reward_heatmap

    name, theta = _structural_mean_params(data, "fleet_maintenance")
    if theta is None:
        return
    reward_heatmap(_env(), theta, REWARD_FIG, title=f"Recovered reward from {name}")


def _make_curve_fig(data):
    from validation.benchmark.figures import _structural_mean_params, reward_curve

    _, theta = _structural_mean_params(data, "fleet_maintenance")
    env = _env()
    reward_curve(
        env,
        CURVE_FIG,
        params=theta,
        action_labels=_ACTIONS,
        state_label="aggregate mileage $x(s)$",
        x=_aggregate_mileage(env),
    )


EXTRA_FIGURES = [(REWARD_FIG, _make_reward_fig), (CURVE_FIG, _make_curve_fig)]


if __name__ == "__main__":
    main_cli(
        cells=CELLS,
        title="Simulation study: fleet maintenance (multi-component bus engine replacement)",
        narrative=NARRATIVE,
        diagnoses=DIAGNOSES,
        excluded=EXCLUDED,
        results_json=RESULTS_JSON,
        page_path=PAGE_PATH,
        extra_figures=EXTRA_FIGURES,
    )
