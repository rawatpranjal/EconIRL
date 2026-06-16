"""Simulation study: route choice on a synthetic road network.

Generates ``validation/results/study_route_choice.json`` and renders
``docs/simulation_studies/route_choice.md`` from it via the shared harness
(``validation/benchmark/harness.py``), inheriting the honesty contract: raw
per-replication facts, page as a pure function of the JSON, crashes first class,
no fabricated metrics.

Usage::

    python scripts/study_route_choice.py [--verbose]      # run + write JSON
    python scripts/study_route_choice.py --page           # regenerate the page
    python scripts/study_route_choice.py --verify         # re-derive table from JSON
    python scripts/study_route_choice.py --only-estimator NAME   # retry one, merge
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from econirl.environments.road_network import road_network  # noqa: E402
from validation.benchmark.harness import Cell, RosterEntry, main_cli  # noqa: E402
from validation.benchmark.runner import _action_reward, _linear_utility  # noqa: E402

RESULTS_JSON = os.path.join(_ROOT, "validation", "results", "study_route_choice.json")
PAGE_PATH = os.path.join(_ROOT, "docs", "simulation_studies", "route_choice.md")
FIGURE_PNG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                          "route_choice_dgp.png")

# ---- DGP configuration ----
# 25 nodes, 4 actions. Small enough for sub-minute per-estimator fits while
# giving the structural family enough variation to identify all three parameters.
ENV = dict(num_nodes=25, num_actions=4, seed=0, discount_factor=0.95)


def _env():
    return road_network(**ENV)


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

    est = CCPEstimator(num_policy_iterations=1, compute_hessian=True, verbose=False)
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


def _run_neural_gladius(env, panel):
    from types import SimpleNamespace

    from econirl.estimators import NeuralGLADIUS

    ps = env.problem_spec
    m = NeuralGLADIUS(
        n_actions=int(env.num_actions),
        discount=float(ps.discount_factor),
        max_epochs=200,
        verbose=False,
    )
    m.fit(
        panel,
        features=np.asarray(env.feature_matrix),
        transitions=np.asarray(env.transition_matrices),
    )
    return SimpleNamespace(
        parameters=None,
        standard_errors=None,
        policy=m.policy_,
        value_function=getattr(m, "value_", None),
        converged=bool(getattr(m, "converged_", True)),
    )


ROSTER = (
    RosterEntry("NFXP",         "structural", _run_nfxp),
    RosterEntry("CCP",          "structural", _run_ccp),
    RosterEntry("MPEC",         "structural", _run_mpec, timeout=120),
    RosterEntry("MCE-IRL",      "behavioral", _run_mce_irl),
    RosterEntry("NeuralGLADIUS","behavioral", _run_neural_gladius),
)

# ---------------------------------------------------------------------------
# Diagnoses, excluded, cell, narrative
# ---------------------------------------------------------------------------

DIAGNOSES = {
    "NFXP": (
        "Full-solution MLE with a nested Bellman fixed-point inner loop. "
        "Quadratic convergence near the optimum; all three parameters are "
        "identified from the edge-feature contrast."
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
        "Its convergence indicator reports whether the gradient norm crossed "
        "the tolerance. The objective often plateaus before that, so it can "
        "read False while the recovered policy is accurate."
    ),
    "NeuralGLADIUS": (
        "Model-free neural policy learner. Uses only the feature matrix and "
        "the observed panel; it does not use transition matrices. "
        "Capped at 200 epochs here for short compute."
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
        "name": "NNES, SEES, TD-CCP, UFXP",
        "reason": (
            "correct structural estimators but slower on a 25-node graph; "
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
]

CELLS = (
    Cell(
        cell_id="route_choice",
        label="Route choice (25 nodes, 4 actions)",
        description=(
            "Synthetic route-choice problem on a random geometric graph. "
            "25 nodes placed uniformly in the unit square; edges within "
            "Euclidean radius 0.25, plus a spanning tree for connectivity. "
            "An agent at node $s$ picks among 4 nearest neighbours; actions "
            "beyond node degree self-loop. Reward is linear in three edge "
            "features: negative edge length, destination amenity, and negative "
            "shortest-path distance to a fixed goal. "
            f"``road_network(num_nodes={ENV['num_nodes']}, "
            f"num_actions={ENV['num_actions']}, "
            f"discount_factor={ENV['discount_factor']}, "
            f"seed={ENV['seed']})``."
        ),
        env_factory=_env,
        roster=ROSTER,
        n_individuals=200,
        n_periods=35,
        seed=42,
        n_replications=2,
        fit_timeout=240,
        param_block=True,
        figure=FIGURE_PNG,
    ),
)

NARRATIVE = {
    "title": "Route choice on a synthetic road network",
    "intro": (
        "A traveller moves through a road network one step at a time. "
        "Each period, the agent chooses among the nearest neighbours of the current node. "
        "The utility depends on the edge: how long it is, how attractive the destination "
        "is, and how close the destination sits to a fixed goal node.\n"
        "\n"
        "## The data-generating process\n"
        "\n"
        "Nodes are scattered uniformly at random in the unit square. "
        "Edges connect pairs within a fixed Euclidean radius. "
        "A spanning tree is overlaid to keep the graph connected. "
        "The agent at node $s$ picks among $A$ nearest neighbours sorted by distance. "
        "Actions beyond the node degree self-loop.\n"
        "\n"
        "The reward for traversing edge $(s, a) \\to s'$ is linear in three features:\n"
        "\n"
        "$$\n"
        "u_\\theta(s, a) = "
        "\\theta_{\\mathrm{cost}}\\,(-d_{ss'}) + "
        "\\theta_{\\mathrm{am}}\\,\\mathrm{am}(s') + "
        "\\theta_{\\mathrm{goal}}\\,(-\\ell_{s'})\n"
        "$$\n"
        "\n"
        "where $d_{ss'}$ is the Euclidean edge length, $\\mathrm{am}(s')$ is a "
        "node-level amenity draw, and $\\ell_{s'}$ is the shortest-path distance "
        "from $s'$ to a fixed destination node. "
        "The true parameters are $\\theta = [1.0,\\;0.5,\\;1.0]$.\n"
        "\n"
        "Agents discount future payoffs at $\\beta$ and face i.i.d. logit taste "
        "shocks (scale $\\sigma = 1$). Their behaviour solves the soft Bellman equation. "
        "All three parameters are identified from observed route choices because the "
        "features vary across edges, not just states. "
        "The panel simulates $N$ agents for $T$ periods from the true optimal policy. "
        "The figure shows simulated paths and the optimal value function "
        "(lower at nodes far from the goal)."
    ),
    "cells": {
        "route_choice": {
            "after": (
                "The structural family (NFXP, CCP, MPEC) recovers all three "
                "parameters on the same scale as the truth, so Param RMSE applies "
                "to them alone. MCE-IRL and NeuralGLADIUS recover a reward in "
                "their own parameterization: reward is only partially identified "
                "from behaviour, so comparing their internal weights to the truth "
                "is not meaningful. Policy TV and regret are the right scorecards "
                "for the behavioral family."
            ),
        },
    },
    "script": "scripts/study_route_choice.py",
    "results_rel": "validation/results/study_route_choice.json",
}


if __name__ == "__main__":
    main_cli(
        cells=CELLS,
        title="Simulation study: route choice on a road network",
        narrative=NARRATIVE,
        diagnoses=DIAGNOSES,
        excluded=EXCLUDED,
        results_json=RESULTS_JSON,
        page_path=PAGE_PATH,
    )
