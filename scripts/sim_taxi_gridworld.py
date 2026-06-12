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
FIGURE_PNG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                          "taxi_gridworld_dgp.png")

# 8x8 grid, 64 states, 5 actions. discount 0.95 (not the 0.99 default) keeps
# the inner solves cheap on a page meant to be light; the economics is the same.
ENV = dict(grid_size=8, step_penalty=-0.1, terminal_reward=10.0,
           distance_weight=0.1, discount_factor=0.95)


def _env():
    return GridworldEnvironment(**ENV)


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

    est = MPECEstimator(config=MPECConfig(solver="slsqp", max_iter=200, constraint_tol=1e-6),
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
    RosterEntry("MaxEnt-IRL", "behavioral", _run_maxent_irl),
    RosterEntry("MCE-IRL", "behavioral", _run_mce_irl),
    RosterEntry("Deep-MCE-IRL", "behavioral", _run_deep_mce_irl),
    RosterEntry("AIRL", "behavioral", _run_airl),
    RosterEntry("IQ-Learn", "behavioral", _run_iq_learn),
    RosterEntry("f-IRL", "behavioral", _run_firl),
    RosterEntry("GLADIUS", "behavioral", _run_gladius),
    RosterEntry("BC", "behavioral", _run_bc),
    RosterEntry("NFXP", "structural", _run_nfxp),
    RosterEntry("CCP", "structural", _run_ccp),
    RosterEntry("MPEC", "structural", _run_mpec),
    RosterEntry("UFXP", "structural", _run_ufxp),
)


DIAGNOSES = {
    "MaxEnt-IRL": "The Ziebart tradition this environment comes from. Matches "
                  "discounted feature counts; with two of three features "
                  "state-only, most of its objective is insensitive to the "
                  "choice contrasts the data carry.",
    "MCE-IRL": "Causal maximum-entropy IRL. Two of its three reward directions "
               "are unidentified here (state-only features), so its gradient "
               "ascent can drift far along them; in one replication of three "
               "the resulting policy collapsed outright. Read the per-rep "
               "records, not just the mean.",
    "Deep-MCE-IRL": "Neural-reward MCE-IRL via its sklearn-style fit interface; "
                    "parameters are the neural reward projected onto the linear "
                    "features.",
    "AIRL": "reward_arg='state_action'; recovered parameters stay gauge/shaping-"
            "unidentified by design, so policy TV is the right scorecard, and "
            "here even behavior is hard: the discriminator sees mostly "
            "corridor states.",
    "IQ-Learn": "q_type='linear' ties its Q function to the same feature basis, "
                "so it inherits the contrast-rank problem on top of thin "
                "coverage.",
    "f-IRL": "Recovers a tabular reward, one value per state-action pair, which "
             "does not depend on the deficient feature basis at all - the "
             "strongest behavioral score on this page.",
    "GLADIUS": "Neural Q and expected-value networks; tracks behavior where "
               "data exists.",
    "BC": "Behavioral cloning; matches observed choices where data exists, "
          "falls back to uniform where it does not, and recovers no reward.",
    "NFXP": "Structural contrast: exact MLE. It reproduces the true policy "
            "almost perfectly while reporting parameters far from the truth - "
            "the likelihood is flat along the two state-only feature "
            "directions, so the parameter numbers are arbitrary points on a "
            "ridge, not estimation error.",
    "CCP": "Structural contrast. Same flat likelihood as NFXP, plus inverted "
           "choice probabilities estimated from a concentrated state "
           "distribution.",
    "MPEC": "Structural contrast: constrained MLE on the same ridge.",
    "UFXP": "Unnested fixed point (Bray; Oguz and Bray 2026) with optimal "
            "weighting (OUFXP). Its moment system is built from exactly the "
            "action-contrast features, so the rank-1 contrast design leaves "
            "two directions to the minimum-norm solution; behavior stays "
            "close, parameters are pinned only in the identified direction.",
}

EXCLUDED = [
    {"name": "SEES", "reason": "its spline value basis is built for an ordered "
     "1-D state index; a 2-D grid breaks that geometry, so running it here would "
     "be a misspecification by construction"},
    {"name": "NNES, TD-CCP", "reason": "the structural contrast is carried by "
     "NFXP/CCP/MPEC/UFXP here; the full structural roster runs on the bus "
     "engine and abstract MDP pages"},
    {"name": "MMP, GAIL, GCL, DeepMaxEnt-IRL, Bayesian-IRL", "reason": "dropped "
     "from the study's rosters by scope decision (MMP and GAIL also failed a "
     "20-30 minute single-fit budget on the bus engine page)"},
]

CELLS = (
    Cell(
        cell_id="gridworld",
        label="Gridworld 8x8",
        description=(
            "An agent starts at the top-left corner of an 8x8 grid and walks "
            "toward an absorbing goal at the bottom-right, with a per-step "
            "penalty, a terminal reward, and a distance shaping term. "
            f"`GridworldEnvironment(grid_size={ENV['grid_size']}, "
            f"step_penalty={ENV['step_penalty']}, "
            f"terminal_reward={ENV['terminal_reward']}, "
            f"distance_weight={ENV['distance_weight']}, "
            f"discount_factor={ENV['discount_factor']})`. Transitions are "
            "deterministic; 64 states, 5 actions (left, right, up, down, stay)."
        ),
        env_factory=_env,
        roster=ROSTER,
        n_individuals=500,
        n_periods=20,
        seed=7,
        n_replications=3,
        fit_timeout=900,
        figure=FIGURE_PNG,
    ),
)

NARRATIVE = {
    "title": "Gridworld navigation",
    "intro": (
        "Gridworld navigation is the home turf of the maximum-entropy IRL "
        "tradition (Ziebart's MaxEnt and its descendants), so this page weights "
        "the roster toward IRL methods, with NFXP, CCP, MPEC, and UFXP as the "
        "structural contrast. The environment also supplies a stress the bus "
        "engine does not: every trajectory starts at the same corner and walks "
        "toward the goal, so states far from the start-to-goal path are visited "
        "rarely or never. Methods that rely on inverting state-by-state choice "
        "frequencies feel that thinness; methods that share strength through "
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
                "The headline reading is the gap between the Policy TV and "
                "Param RMSE columns for the structural family: near-perfect "
                "behavior with parameter estimates orders of magnitude from "
                "the truth. That is not estimation error, it is structural "
                "non-identification, and the design line above the table "
                "says why: the raw feature design has full rank, but the "
                "action-contrast design - the only thing choice data can "
                "ever identify - has rank 1, because the step-penalty and "
                "distance features take the same value for every action at a "
                "state and difference out of every choice probability. Any "
                "parameter vector on that two-dimensional ridge produces the "
                "same behavior, so each estimator reports an arbitrary ridge "
                "point. The practical lesson: check the rank of the "
                "action-differenced features before estimating, not just the "
                "raw design.\n"
                "\n"
                "One regret caveat specific to this page: transitions here are "
                "deterministic, so the Type B intervention (replace the dynamics "
                "with a random sparse world) is a stark change rather than a "
                "perturbation; read Type B as a stress test of reward "
                "transferability under completely new dynamics, not as a local "
                "robustness check."
            ),
        },
    },
    "script": "scripts/sim_taxi_gridworld.py",
    "results_rel": "validation/results/sim_taxi_gridworld.json",
}


if __name__ == "__main__":
    main_cli(cells=CELLS, title="Simulation study: gridworld navigation",
             narrative=NARRATIVE, diagnoses=DIAGNOSES, excluded=EXCLUDED,
             results_json=RESULTS_JSON, page_path=PAGE_PATH)
