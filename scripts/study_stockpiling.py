"""Simulation study: consumer stockpiling of a storable good (Hendel-Nevo).

Generates ``validation/results/study_stockpiling.json`` and renders
``docs/simulation_studies/stockpiling.md`` from it via the shared harness
(``validation/benchmark/harness.py``), inheriting the honesty contract: raw
per-replication facts, page as a pure function of the JSON, crashes first class,
no fabricated metrics.

Usage::

    python scripts/study_stockpiling.py [--verbose]      # run + write JSON
    python scripts/study_stockpiling.py --page           # regenerate the page
    python scripts/study_stockpiling.py --verify         # re-derive table from JSON
    python scripts/study_stockpiling.py --only-estimator NAME   # retry one, merge
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from econirl.environments.storable_goods import storable_goods  # noqa: E402
from validation.benchmark.harness import Cell, RosterEntry, main_cli  # noqa: E402
from validation.benchmark.runner import _action_reward, _linear_utility  # noqa: E402

RESULTS_JSON = os.path.join(_ROOT, "validation", "results", "study_stockpiling.json")
PAGE_PATH = os.path.join(_ROOT, "docs", "simulation_studies", "stockpiling.md")
FIGURE_PNG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                          "stockpiling_dgp.png")
RESULTS_FIG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                           "stockpiling_results.png")
CURVE_FIG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                         "stockpiling_reward_curve.png")
REWARD_FIG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                          "stockpiling_reward.png")

# ---- DGP configuration ----
# 10 inventory levels x 2 price regimes = 20 states, 2 actions. Small enough for
# sub-minute per-estimator fits; the action-dependent stockout feature makes all
# three reward parameters identified from choices (contrast rank 3).
ENV = dict(max_inventory=9, pack_size=3, discount_factor=0.95, seed=0)


def _env():
    return storable_goods(**ENV)


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


def _run_mce_irl(env, panel):
    from econirl.estimation import MCEIRLConfig, MCEIRLEstimator

    est = MCEIRLEstimator(config=MCEIRLConfig(
        learning_rate=0.05, outer_max_iter=80, inner_max_iter=1000,
        compute_se=False, verbose=False,
    ))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


ROSTER = (
    RosterEntry("NFXP",         "structural", _run_nfxp, uses_transitions=True),
    RosterEntry("CCP",          "structural", _run_ccp, uses_transitions=True),
    RosterEntry("MCE-IRL",      "behavioral", _run_mce_irl, uses_transitions=True),
)

# ---------------------------------------------------------------------------
# Diagnoses, excluded, cell, narrative
# ---------------------------------------------------------------------------

DIAGNOSES = {
    "NFXP": (
        "Full-solution MLE with a nested Bellman fixed-point inner loop. "
        "Quadratic convergence near the optimum; all three parameters are "
        "identified from the action-dependent stockout feature and the "
        "price-varying spending feature."
    ),
    "CCP": (
        "CCP uses a first-step nonparametric policy estimate to avoid the "
        "inner Bellman loop. One policy-iteration step corrects the bias from "
        "the nonparametric first stage."
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
            "correct structural estimators but slower here; "
            "NFXP and CCP already cover the structural family"
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
    {
        "name": "NeuralGLADIUS",
        "reason": (
            "a model-free neural policy learner; on a 20-state tabular problem "
            "it reduces to behavioral cloning on the choice likelihood with no "
            "structural edge, and it trailed every method shown. Its decisive "
            "setting is scale, where the full-solution methods time out; see "
            "the fleet-maintenance scale study"
        ),
    },
]

CELLS = (
    Cell(
        cell_id="stockpiling",
        label="Stockpiling (20 states, 2 actions)",
        description=(
            "Consumer stockpiling of a storable good. A household holds "
            "inventory $i \\in \\{0, \\dots, 9\\}$ and faces a price regime "
            "$p \\in \\{\\text{sale}, \\text{regular}\\}$ that follows an "
            "exogenous two-state Markov chain. State $s = 2i + p$ gives 20 "
            "states. Each period the household consumes one unit and chooses "
            "whether to buy a pack of 3 units. Reward is linear in three "
            "features: spending on the purchase, holding cost on carried "
            "inventory, and a stockout penalty when no unit is on hand. "
            f"``storable_goods(max_inventory={ENV['max_inventory']}, "
            f"pack_size={ENV['pack_size']}, "
            f"discount_factor={ENV['discount_factor']}, "
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
    "title": "Consumer stockpiling of a storable good",
    "intro": (
        "A household buys a storable good over time. It consumes one unit each "
        "period. The shelf price swings between a low sale price and a high "
        "regular price. The household can buy a pack now to avoid paying the "
        "high price later. The cost of doing so is the holding cost on the "
        "inventory it carries.\n"
        "\n"
        "## The data-generating process\n"
        "\n"
        "The state is a pair: inventory $i$ and price regime $p$. Inventory "
        "runs from 0 to 9 units. The price regime is sale or regular and "
        "follows a two-state Markov chain. Sales are short. Regular spells are "
        "longer. The flat state index is $s = 2i + p$, giving 20 states.\n"
        "\n"
        "Each period the household chooses to buy a pack of 3 units or not. "
        "Then it consumes one unit if any is on hand. Inventory carried to the "
        "next period is the post-purchase stock minus one, capped at 9.\n"
        "\n"
        "The reward is linear in three features:\n"
        "\n"
        "$$\n"
        "u_\\theta(s, a) = "
        "\\theta_{\\mathrm{spend}}\\,(-c_p B a) + "
        "\\theta_{\\mathrm{hold}}\\,(-i') + "
        "\\theta_{\\mathrm{stock}}\\,(-\\mathbf{1}\\{i + B a = 0\\})\n"
        "$$\n"
        "\n"
        "where $c_p$ is the per-unit price (1 on sale, 2 regular), $B = 3$ is "
        "the pack size, $a \\in \\{0, 1\\}$ is the buy decision, $i'$ is the "
        "inventory carried forward, and the last term is a stockout penalty "
        "paid when the household wanted a unit but had none. The true "
        "parameters are $\\theta = [1.0,\\;0.2,\\;3.0]$.\n"
        "\n"
        "Agents discount future payoffs at $\\beta$ and face i.i.d. logit taste "
        "shocks (scale $\\sigma = 1$). Their behaviour solves the soft Bellman "
        "equation. The stockout feature is action-dependent, because buying when "
        "inventory is empty avoids the penalty, so all three parameters are "
        "identified from observed purchases. The optimal policy stockpiles: it "
        "buys more often on sale than at the regular price, at every inventory "
        "level, and buys less as inventory rises. The panel simulates $N$ agents "
        "for $T$ periods from the true optimal policy. The figure shows simulated "
        "inventory-price paths and the optimal value function."
    ),
    "cells": {
        "stockpiling": {
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
    "script": "scripts/study_stockpiling.py",
    "results_rel": "validation/results/study_stockpiling.json",
    "extra_sections": (
        "## Reward and structure\n"
        "\n"
        "The state index is $s = 2i + p$: inventory $i$ rises in steps of two, "
        "and the sale and regular price regimes interleave. The reward against "
        "$s$ shows the buy and no-buy lines, with the recovered reward dashed "
        "over the true reward. The zigzag is the price regime alternating. The "
        "optimal value falls as inventory and holding cost rise.\n"
        "\n"
        "![Reward against the inventory-price state index, buy versus no-buy]"
        "(../_static/simulation_studies/stockpiling_reward_curve.png)\n"
        "\n"
        "The same reward as a state-by-action heatmap puts the true and recovered "
        "rewards side by side on one color scale.\n"
        "\n"
        "![True and recovered reward heatmaps]"
        "(../_static/simulation_studies/stockpiling_reward.png)\n"
    ),
}

_ACTIONS = ["no buy", "buy"]


def _make_curve_fig(data):
    from validation.benchmark.figures import (_structural_mean_params,
                                              reward_curve)

    _, theta = _structural_mean_params(data, "stockpiling")
    reward_curve(_env(), CURVE_FIG, params=theta, action_labels=_ACTIONS,
                 state_label="state index $s = 2i + p$")


def _make_reward_fig(data):
    from validation.benchmark.figures import (_structural_mean_params,
                                              reward_heatmap)

    name, theta = _structural_mean_params(data, "stockpiling")
    if theta is None:
        return
    reward_heatmap(_env(), theta, REWARD_FIG,
                   title=f"Recovered reward from {name}")


EXTRA_FIGURES = [(CURVE_FIG, _make_curve_fig), (REWARD_FIG, _make_reward_fig)]


if __name__ == "__main__":
    main_cli(
        cells=CELLS,
        title="Simulation study: consumer stockpiling of a storable good",
        narrative=NARRATIVE,
        diagnoses=DIAGNOSES,
        excluded=EXCLUDED,
        results_json=RESULTS_JSON,
        page_path=PAGE_PATH,
        extra_figures=EXTRA_FIGURES,
    )
