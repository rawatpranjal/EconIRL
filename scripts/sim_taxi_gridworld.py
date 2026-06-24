"""Simulation study: gridworld navigation (the MaxEnt IRL tradition).

Generates ``validation/results/sim_taxi_gridworld.json`` and renders
``docs/simulation_studies/taxi_gridworld.md`` via the shared harness. See
``validation/benchmark/harness.py`` for the honesty contract.

Usage:
    python scripts/sim_taxi_gridworld.py [--verbose]
    python scripts/sim_taxi_gridworld.py --page
    python scripts/sim_taxi_gridworld.py --verify
    python scripts/sim_taxi_gridworld.py --only-estimator NAME
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from econirl.environments import GridworldEnvironment  # noqa: E402
from validation.benchmark.harness import Cell, RosterEntry, main_cli  # noqa: E402
from validation.benchmark.runner import _action_reward, _linear_utility  # noqa: E402

RESULTS_JSON = os.path.join(_ROOT, "validation", "results", "sim_taxi_gridworld.json")
PAGE_PATH = os.path.join(_ROOT, "docs", "simulation_studies", "taxi_gridworld.md")
_STATIC = os.path.join(_ROOT, "docs", "_static", "simulation_studies")
FIGURE_PNG = os.path.join(_STATIC, "taxi_gridworld_dgp.png")
RESULTS_FIG = os.path.join(_STATIC, "taxi_gridworld_results.png")
SCALING_FIG = os.path.join(_STATIC, "taxi_gridworld_scaling.png")

# 8x8 grid, 64 states, 5 actions. discount 0.95 (not the 0.99 default) keeps
# the inner solves cheap on a page meant to be light; the economics is the same.
# The scaling sweep reruns a trimmed roster at 6x6 (36 states) and 10x10 (100
# states) to trace compute and accuracy against problem size.
ENV = dict(grid_size=8, step_penalty=-0.1, terminal_reward=10.0,
           distance_weight=0.1, discount_factor=0.95)
HEADLINE_SIZE = 8
SCALING_SIZES = (6, 10)


def _env(grid_size=HEADLINE_SIZE):
    cfg = dict(ENV)
    cfg["grid_size"] = grid_size
    return GridworldEnvironment(**cfg)


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------


def _run_nfxp(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(inner_solver="hybrid", inner_tol=1e-10,
                        inner_max_iter=100000, compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_ccp(env, panel):
    from econirl.estimation import CCPEstimator

    est = CCPEstimator(num_policy_iterations=1, compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_mpec(env, panel):
    from econirl.estimation.mpec import MPECConfig, MPECEstimator

    est = MPECEstimator(config=MPECConfig(solver="sqp", outer_max_iter=200, tol=1e-8, constraint_tol=1e-6),
                        compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_maxent_irl(env, panel):
    from econirl.contrib.maxent_irl import MaxEntIRLEstimator

    # Action-dependent features: the reward here depends on where the action
    # leads (terminal indicator), not on the state alone. Adaptive per-
    # parameter steps (Adam) handle the mixed feature scales.
    est = MaxEntIRLEstimator(inner_tol=1e-8, inner_max_iter=5000, outer_max_iter=500,
                             learning_rate=0.05, compute_hessian=False, verbose=False)
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_mce_irl(env, panel):
    from econirl.estimation import MCEIRLConfig, MCEIRLEstimator

    est = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.05, outer_max_iter=100,
                                              inner_max_iter=2000, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


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


def _run_airl(env, panel):
    from econirl.estimation import AIRLConfig, AIRLEstimator

    est = AIRLEstimator(config=AIRLConfig(reward_type="linear", reward_arg="state_action",
                                          reward_lr=0.01, discriminator_steps=10,
                                          max_rounds=300, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_iq_learn(env, panel):
    from econirl.estimation.iq_learn import IQLearnConfig, IQLearnEstimator

    est = IQLearnEstimator(config=IQLearnConfig(q_type="linear", divergence="chi2",
                                                alpha=3.0, max_iter=2000, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_firl(env, panel):
    from econirl.estimation.f_irl import FIRLEstimator

    # fkl (bounded gradient) with the estimator's default reward clip; the
    # chi2 ratio gradient is unbounded on near-deterministic experts.
    est = FIRLEstimator(f_divergence="fkl", lr=0.2, max_iter=400, reward_clip=10.0,
                        verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_gladius(env, panel):
    from econirl.estimation import GLADIUSConfig, GLADIUSEstimator

    est = GLADIUSEstimator(config=GLADIUSConfig(max_epochs=300, q_hidden_dim=128,
                                                v_hidden_dim=128, q_lr=1e-4, v_lr=1e-4,
                                                patience=60, compute_se=False, verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_bc(env, panel):
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator

    est = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_ufxp(env, panel):
    from econirl.estimation import UFXPEstimator

    # Bray's unnested fixed point with optimal weighting (OUFXP). Conditions
    # are scored only at visited states and the optimal weights downweight
    # thin states by their sample share, which is the interesting behavior on
    # this concentrated-coverage grid.
    est = UFXPEstimator(weights="optimal", verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


ROSTER = (
    RosterEntry("MaxEnt-IRL", "behavioral", _run_maxent_irl, uses_transitions=True),
    RosterEntry("MCE-IRL", "behavioral", _run_mce_irl, uses_transitions=True),
    RosterEntry("Deep-MCE-IRL", "behavioral", _run_deep_mce_irl, uses_transitions=True),
    RosterEntry("AIRL", "behavioral", _run_airl, uses_transitions=True),
    RosterEntry("IQ-Learn", "behavioral", _run_iq_learn, uses_transitions=True),
    RosterEntry("f-IRL", "behavioral", _run_firl, uses_transitions=True),
    RosterEntry("GLADIUS", "behavioral", _run_gladius, uses_transitions=True),
    RosterEntry("BC", "behavioral", _run_bc, uses_transitions=True),
    RosterEntry("NFXP", "structural", _run_nfxp, uses_transitions=True),
    RosterEntry("CCP", "structural", _run_ccp, uses_transitions=True),
    RosterEntry("MPEC", "structural", _run_mpec, uses_transitions=True),
    RosterEntry("UFXP", "structural", _run_ufxp, uses_transitions=True),
)

# A trimmed, representative subset of ROSTER for the scaling sweep: five lines
# (two structural, the structural-plus-MPEC trio, one IRL) keep the scaling
# figure readable and the sweep fast. Same _run_* and uses_transitions as above.
SCALING_ROSTER = (
    RosterEntry("NFXP", "structural", _run_nfxp, uses_transitions=True),
    RosterEntry("CCP", "structural", _run_ccp, uses_transitions=True),
    RosterEntry("MPEC", "structural", _run_mpec, uses_transitions=True),
    RosterEntry("MCE-IRL", "behavioral", _run_mce_irl, uses_transitions=True),
    RosterEntry("GLADIUS", "behavioral", _run_gladius, uses_transitions=True),
)


DIAGNOSES = {
    "MCE-IRL": "Two of its three reward directions, the state-only step and "
               "distance features, are unidentified here. Its gradient "
               "ascent can drift along them, and in one replication of "
               "three the policy collapsed. Read the per-rep records, not "
               "just the mean.",
    "f-IRL": "The strongest behavioral score on this page. It recovers a "
             "tabular reward, one value per state-action pair, which does "
             "not depend on the deficient feature basis at all.",
}

EXCLUDED = [
    {"name": "SEES", "reason": "its spline value basis is built for an ordered "
     "1-D state index. A 2-D grid breaks that geometry, so running it here "
     "would be misspecification by construction"},
    {"name": "NNES, TD-CCP", "reason": "the structural contrast is carried by "
     "NFXP/CCP/MPEC/UFXP here. The full structural roster runs on the bus "
     "engine and abstract MDP pages"},
    {"name": "MMP, GAIL, GCL, DeepMaxEnt-IRL, Bayesian-IRL", "reason": "research "
     "code or too slow; not benchmarked in this study"},
]

def _scaling_cell(grid_size):
    """One scaling-only gridworld cell at a given size.

    Runs the trimmed SCALING_ROSTER to feed the scaling figure (fit time and
    policy TV vs problem size). Renders no per-cell tables of its own. The
    smallest size is a scaling cell, so the scaling figure's line set comes
    from SCALING_ROSTER, not the full headline roster.
    """
    n_states = grid_size * grid_size
    return Cell(
        cell_id=f"taxi_gridworld_{grid_size}",
        label=f"Gridworld {grid_size}x{grid_size}",
        description=(
            f"`GridworldEnvironment(grid_size={grid_size}, "
            f"step_penalty={ENV['step_penalty']}, "
            f"terminal_reward={ENV['terminal_reward']}, "
            f"distance_weight={ENV['distance_weight']}, "
            f"discount_factor={ENV['discount_factor']})`. Transitions are "
            f"deterministic; {n_states} states, 5 actions."
        ),
        env_factory=(lambda n=grid_size: _env(n)),
        roster=SCALING_ROSTER,
        n_individuals=500,
        n_periods=20,
        seed=7,
        n_replications=2,
        fit_timeout=900,
        scaling_only=True,
    )


# Headline first so single-cell runs (--only-cell gridworld) still work.
CELLS = (
    Cell(
        cell_id="gridworld",
        label="Gridworld 8x8",
        description=(
            f"`GridworldEnvironment(grid_size={HEADLINE_SIZE}, "
            f"step_penalty={ENV['step_penalty']}, "
            f"terminal_reward={ENV['terminal_reward']}, "
            f"distance_weight={ENV['distance_weight']}, "
            f"discount_factor={ENV['discount_factor']})`. Transitions are "
            "deterministic; 64 states, 5 actions (left, right, up, down, stay)."
        ),
        env_factory=(lambda n=HEADLINE_SIZE: _env(n)),
        roster=ROSTER,
        n_individuals=500,
        n_periods=20,
        seed=7,
        n_replications=3,
        fit_timeout=900,
        figure=FIGURE_PNG,
        results_figure=RESULTS_FIG,
        # Parameters are not separately identified here (action-contrast rank
        # 1), so the parameter and regret-transfer columns would print
        # arbitrary ridge points and meaningless transfers.
        show_params=False,
        show_regret=False,
    ),
    *(_scaling_cell(n) for n in SCALING_SIZES),
)

NARRATIVE = {
    "title": "Gridworld navigation",
    "intro": (
        "Gridworld navigation is the home turf of the maximum-entropy IRL "
        "tradition of Ziebart's MaxEnt and its descendants, so this page "
        "weights the roster toward IRL methods. NFXP, CCP, MPEC, and UFXP run "
        "as the structural contrast. The environment also supplies a stress "
        "the bus engine does not. Every trajectory starts at the same corner "
        "and walks toward the goal, so states off that path are visited "
        "rarely or never. Methods that invert state-by-state choice "
        "frequencies feel that thinness. Methods that share strength through "
        "features or networks do not.\n"
        "\n"
        "## The data-generating process\n"
        "\n"
        "States are cells of an $N \\times N$ grid indexed $s = "
        "\\mathrm{row} \\cdot N + \\mathrm{col}$, with five actions (left, "
        "right, up, down, stay), deterministic moves, and an absorbing goal at "
        "the bottom-right corner. The reward has three parts: a per-step "
        "penalty, a terminal bonus when the chosen move reaches the goal, and "
        "a shaping term in the Manhattan distance $d(s)$ to the goal:\n"
        "\n"
        "$$\n"
        "u_\\theta(s, a) = \\theta_{\\mathrm{step}}\\, "
        "\\mathbf{1}\\{s \\neq s_{\\mathrm{goal}}\\}\n"
        "+ \\theta_{\\mathrm{goal}}\\, \\mathbf{1}\\{s'(s, a) = "
        "s_{\\mathrm{goal}}\\}\n"
        "- \\theta_{\\mathrm{dist}}\\, \\frac{d(s)}{2N},\n"
        "$$\n"
        "\n"
        "with $\\theta_{\\mathrm{step}} = -0.1$, $\\theta_{\\mathrm{goal}} = "
        "10$, $\\theta_{\\mathrm{dist}} = 0.1$. The agent discounts at "
        "$\\beta$ and faces i.i.d. logit taste shocks (scale $\\sigma = 1$), "
        "so behavior solves the soft Bellman equation\n"
        "\n"
        "$$\n"
        "V(s) = \\log \\sum_{a} \\exp\\Bigl(u_\\theta(s,a) + "
        "\\beta\\, \\mathbb{E}\\bigl[V(s') \\mid s,a\\bigr]\\Bigr),\n"
        "\\qquad \\pi^*(a \\mid s) \\propto \\exp\\Bigl(u_\\theta(s,a) + "
        "\\beta\\, \\mathbb{E}\\bigl[V(s') \\mid s,a\\bigr]\\Bigr),\n"
        "$$\n"
        "\n"
        "and every trajectory starts at the top-left corner (state 0). The "
        "figure shows the resulting paths climbing the state index toward the "
        "absorbing goal and the value function rising with proximity to it. "
        "The horizon is deliberately short (20 periods) because the goal is "
        "absorbing: once there, an agent generates no further information."
    ),
    "cells": {
        "gridworld": {
            "after": (
                "The structural rows match behavior almost perfectly, but "
                "their parameters are not separately identified here, so "
                "the parameter columns are omitted. The raw feature design "
                "has full rank, while the action-contrast design has rank "
                "1. The step-penalty and distance features take the same "
                "value for every action at a state, so they cancel out of "
                "every choice probability. The regret columns are omitted "
                "for the same reason, because transferring an unidentified "
                "reward is not a meaningful exercise. The lesson is to check "
                "the rank of the action-differenced features before "
                "estimating."
            ),
        },
    },
    "scaling_intro": (
        "The same study at three grid sizes (36, 64, 100 states). Each line is "
        "one estimator: fit time on the left, policy total variation on the "
        "right. The structural rows track behavior closely across sizes. The "
        "compute lines reflect fixed overhead as much as problem size at this "
        "scale, so the time curves need not rise cleanly with the state count. "
        "Policy total variation is the right scorecard here because the reward "
        "parameters are not separately identified on this grid."
    ),
    "script": "scripts/sim_taxi_gridworld.py",
    "results_rel": "validation/results/sim_taxi_gridworld.json",
}


if __name__ == "__main__":
    main_cli(cells=CELLS, title="Simulation study: gridworld navigation",
             narrative=NARRATIVE, diagnoses=DIAGNOSES, excluded=EXCLUDED,
             results_json=RESULTS_JSON, page_path=PAGE_PATH,
             scaling_figure=SCALING_FIG)
