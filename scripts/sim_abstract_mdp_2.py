"""Simulation study: harder abstract MDP (scale, near-unit discount, collinearity).

Three cells in one experiment, re-homing the old benchmark ladder's
``large_sparse``, ``high_beta``, and ``rank_deficient`` stories:

- ``harder_300``: 300-state Garnet MDP at discount 0.95. The vanilla-vs-refined
  NFXP runtime contrast is measured here, never asserted.
- ``high_discount``: the same MDP at discount 0.99, structural family with
  standard errors, 10 replications. Carries the bias / variance / coverage
  block, including the SE-availability rate per estimator.
- ``rank_deficient``: deliberately collinear features (24 states). The
  identification-failure story: the identified combination is recovered, the
  collinear direction is not.

Generates ``validation/results/sim_abstract_mdp_2.json`` and renders
``docs/simulation_studies/abstract_mdp_2_harder.md``. Honesty contract in
``validation/benchmark/harness.py``.

Usage:
    python scripts/sim_abstract_mdp_2.py [--verbose] [--only-cell ID]
    python scripts/sim_abstract_mdp_2.py --page
    python scripts/sim_abstract_mdp_2.py --verify
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from econirl.environments import random_mdp  # noqa: E402
from validation.benchmark.cells import _rank_deficient  # noqa: E402
from validation.benchmark.harness import Cell, RosterEntry, main_cli  # noqa: E402
from validation.benchmark.runner import _action_reward, _linear_utility  # noqa: E402

RESULTS_JSON = os.path.join(_ROOT, "validation", "results", "sim_abstract_mdp_2.json")
PAGE_PATH = os.path.join(_ROOT, "docs", "simulation_studies", "abstract_mdp_2_harder.md")
FIGURE_PNG = os.path.join(_ROOT, "docs", "_static", "simulation_studies",
                          "abstract_mdp_2_dgp.png")

MDP = dict(num_states=300, num_actions=2, num_features=3, branching=5, seed=505)


def _env_095():
    return random_mdp(discount_factor=0.95, **MDP)


def _env_099():
    return random_mdp(discount_factor=0.99, **MDP)


# ---------------------------------------------------------------------------
# Roster. NFXP appears twice on purpose: "sa" is Rust's original successive
# approximation (pure contraction), "polyalgorithm" is the Iskhakov et al
# SA-then-Newton-Kantorovich refinement. Same likelihood, same answer; the
# runtime column carries the story.
# ---------------------------------------------------------------------------


def _run_nfxp_sa(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(inner_solver="sa", inner_tol=1e-10,
                        inner_max_iter=100000, compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_nfxp_nk(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(inner_solver="polyalgorithm", inner_tol=1e-10,
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


def _run_nnes(env, panel):
    from econirl.estimation.nnes import NNESEstimator

    est = NNESEstimator(hidden_dim=64, v_epochs=800, n_outer_iterations=5,
                        compute_se=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_sees(env, panel):
    from econirl.estimation.sees import SEESEstimator

    # Basis must span the value function: bspline basis_dim >= num_states.
    est = SEESEstimator(basis_type="bspline", basis_dim=int(env.num_states),
                        warm_start_value=True, penalty_weight=10.0,
                        compute_se=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_tdccp(env, panel):
    from econirl.estimation import TDCCPConfig, TDCCPEstimator

    est = TDCCPEstimator(config=TDCCPConfig(hidden_dim=64, avi_iterations=15,
                                            epochs_per_avi=15, compute_se=True,
                                            verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_ufxp(env, panel):
    from econirl.estimation import UFXPEstimator

    # Bray's unnested fixed point: duals computed once before the parameter
    # search; the linear-utility case is closed-form least squares.
    est = UFXPEstimator(num_projections=64, seed=0, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_mce_irl(env, panel):
    from econirl.estimation import MCEIRLConfig, MCEIRLEstimator

    est = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.05, outer_max_iter=100,
                                              inner_max_iter=2000, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


_STRUCTURAL = (
    RosterEntry("NFXP-SA", "structural", _run_nfxp_sa),
    RosterEntry("NFXP-NK", "structural", _run_nfxp_nk),
    RosterEntry("CCP", "structural", _run_ccp),
    RosterEntry("MPEC", "structural", _run_mpec),
    RosterEntry("NNES", "structural", _run_nnes),
    RosterEntry("SEES", "structural", _run_sees),
    RosterEntry("TD-CCP", "structural", _run_tdccp),
    RosterEntry("UFXP", "structural", _run_ufxp),
)

ROSTER_A = _STRUCTURAL + (RosterEntry("MCE-IRL", "behavioral", _run_mce_irl),)
ROSTER_B = _STRUCTURAL
ROSTER_C = (
    RosterEntry("NFXP-NK", "structural", _run_nfxp_nk),
    RosterEntry("CCP", "structural", _run_ccp),
    RosterEntry("MPEC", "structural", _run_mpec),
    RosterEntry("UFXP", "structural", _run_ufxp),
)


DIAGNOSES = {
    "NFXP-SA": "Rust's original inner loop: successive approximation, a pure "
               "contraction with rate equal to the discount factor. It reaches "
               "the same maximum-likelihood answer; what changes with scale and "
               "discount is how long it takes.",
    "NFXP-NK": "The Iskhakov et al refinement: successive approximation to get "
               "near the fixed point, then Newton-Kantorovich steps. Same "
               "estimate, different bill.",
    "CCP": "Hotz-Miller inversion: estimate choice probabilities, invert once, no "
           "fixed point inside the optimizer. Its standard errors come from the "
           "outer Hessian and can fail to be finite even when the point estimate "
           "is fine; the SE-availability column makes that visible.",
    "MPEC": "Constrained MLE: the Bellman equation enters as constraints for the "
            "SLSQP solver, with one variable per state plus the parameters.",
    "NNES": "Neural value network plus structural MLE.",
    "SEES": "Sieve value function: bspline basis with basis_dim = num_states so "
            "the basis can span the value function.",
    "TD-CCP": "Neural CCP with approximate value iteration and cross-fitted "
              "standard errors.",
    "UFXP": "Unnested fixed point (Bray; Oguz and Bray 2026). The Bellman "
            "first-order conditions are scored directly; the value function is "
            "eliminated by duals computed once before the search, so no fixed "
            "point is ever solved inside the optimizer and the linear case is "
            "closed-form. Random-projection weights (not the efficient OUFXP "
            "step), no standard errors.",
    "MCE-IRL": "Behavioral reference on the harder cell. Its converged flag is "
               "conservative (gradient-norm tolerance); read it next to Policy TV.",
}

EXCLUDED = [
    {"name": "MaxEnt-IRL, IQ-Learn, AIRL, f-IRL, GLADIUS, Deep-MCE-IRL, BC",
     "reason": "this page's question is structural: parameter recovery, inference "
     "quality, and identification as the problem hardens. The IRL family is "
     "compared on the Rust bus and gridworld pages; MCE-IRL stays here as the "
     "behavioral reference"},
    {"name": "GAIL, GCL, DeepMaxEnt-IRL, Bayesian-IRL",
     "reason": "known slow; their single-run showing is on the Rust bus page"},
]

CELLS = (
    Cell(
        cell_id="harder_300",
        label="300 states, discount 0.95",
        description=(
            "A 300-state Garnet MDP with stochastic sparse transitions "
            "(branching 5) and a 3-feature linear reward: "
            "`random_mdp(num_states=300, num_actions=2, num_features=3, "
            "branching=5, discount_factor=0.95, seed=505)`."
        ),
        env_factory=_env_095,
        roster=ROSTER_A,
        n_individuals=500,
        n_periods=60,
        seed=505,
        n_replications=3,
        figure=FIGURE_PNG,
    ),
    Cell(
        cell_id="high_discount",
        label="Same MDP, discount 0.99",
        description=(
            "The identical 300-state MDP with the discount factor moved from "
            "0.95 to 0.99, where continuation values dominate flow payoffs and "
            "the inner fixed point becomes a slow contraction. Structural "
            "family only, 10 replications, standard errors requested from "
            "every estimator."
        ),
        env_factory=_env_099,
        roster=ROSTER_B,
        n_individuals=500,
        n_periods=60,
        seed=505,
        n_replications=10,
        param_block=True,
    ),
    Cell(
        cell_id="rank_deficient",
        label="Collinear features (24 states)",
        description=(
            "A small MDP whose third reward feature is exactly twice the "
            "second (design rank 2 of 3). The likelihood identifies only the "
            "combination theta_1 + 2 theta_2; no estimator can recover the "
            "individual coordinates, and the page checks what each one does "
            "about it."
        ),
        env_factory=_rank_deficient,
        roster=ROSTER_C,
        n_individuals=500,
        n_periods=80,
        seed=606,
        n_replications=20,
        param_block=True,
    ),
)

NARRATIVE = {
    "title": "Abstract MDP 2: scale, discount, and identification",
    "intro": (
        "The sanity-check page showed every estimator recovering an easy "
        "problem. This page hardens the problem along three separate axes and "
        "watches the structural family specifically, because the questions are "
        "structural: what happens to runtime as the state space grows, what "
        "happens to inference as the discount factor approaches one, and what "
        "happens to the parameters when the reward features are collinear. "
        "Each axis gets its own cell, run on the same engine and reported from "
        "the same raw records as every other page.\n"
        "\n"
        "## The data-generating process\n"
        "\n"
        "The first two cells draw one Garnet-style MDP from the seed and hold "
        "it fixed. Each state-action pair reaches a uniform random subset of "
        "$b$ states with Dirichlet weights, mixed with a small self-loop mass "
        "$\\ell$:\n"
        "\n"
        "$$\n"
        "P(s' \\mid s, a) \\;=\\; (1-\\ell)\\, D_{s,a}(s') \\;+\\; "
        "\\ell\\, \\mathbf{1}\\{s'=s\\},\n"
        "\\qquad D_{s,a} \\sim \\mathrm{Dirichlet}(\\mathbf{1}_b),\\quad "
        "b = 5,\\ \\ell = 0.05 .\n"
        "$$\n"
        "\n"
        "The reward is linear in polynomial features of the normalized state "
        "index $x_s = s/(S-1)$. Action $0$ is a zeroed outside option (the "
        "identification anchor); for $a \\geq 1$,\n"
        "\n"
        "$$\n"
        "u_\\theta(s,a) = \\theta^\\top \\varphi(s,a),\n"
        "\\qquad \\varphi(s,a) = \\bigl(1,\\ x_s,\\ x_s^{2} + a\\bigr),\n"
        "\\qquad \\theta \\sim \\mathcal{N}(0,\\ 0.25\\, I_3).\n"
        "$$\n"
        "\n"
        "The agent discounts at $\\beta$ and faces i.i.d. logit taste shocks "
        "(scale $\\sigma = 1$), so behavior solves the soft Bellman equation\n"
        "\n"
        "$$\n"
        "V(s) = \\log \\sum_{a} \\exp\\Bigl(u_\\theta(s,a) + "
        "\\beta\\, \\mathbb{E}\\bigl[V(s') \\mid s,a\\bigr]\\Bigr),\n"
        "\\qquad \\pi^*(a \\mid s) \\propto \\exp\\Bigl(u_\\theta(s,a) + "
        "\\beta\\, \\mathbb{E}\\bigl[V(s') \\mid s,a\\bigr]\\Bigr),\n"
        "$$\n"
        "\n"
        "and the data are $N$ independent agents simulated for $T$ periods "
        "from $\\pi^*$ and $P$. The third cell swaps in a small handcrafted "
        "MDP whose features are deliberately collinear; its construction is "
        "described in that cell."
    ),
    "cells": {
        "harder_300": {
            "before": (
                "The first cell is about cost at scale. All estimators face the "
                "same 300-state problem; the runtime column is the result. The "
                "two NFXP rows are the same estimator with two inner solvers, "
                "Rust's original successive approximation against the "
                "Newton-Kantorovich polyalgorithm, so the refinement's value is "
                "measured on this page rather than asserted."
            ),
        },
        "high_discount": {
            "before": (
                "The second cell moves the discount factor to 0.99 and asks a "
                "harder question than point recovery: is the reported "
                "uncertainty usable? The parameter table reports bias, the "
                "spread of estimates across replications, RMSE, and the share "
                "of nominal 95% intervals that actually cover the truth, "
                "together with how often each estimator produced finite "
                "standard errors at all."
            ),
        },
        "rank_deficient": {
            "before": (
                "The last cell breaks identification on purpose. With the third "
                "feature exactly twice the second, the design matrix has rank 2 "
                "and the individual coordinates theta_1 and theta_2 are not "
                "separately identified, only their combination. The interesting "
                "output is not the (meaningless) per-coordinate bias but the "
                "design diagnostics above the table and how each estimator's "
                "intervals behave when the question has no answer."
            ),
        },
    },
    "script": "scripts/sim_abstract_mdp_2.py",
    "results_rel": "validation/results/sim_abstract_mdp_2.json",
}


if __name__ == "__main__":
    main_cli(cells=CELLS, title="Simulation study: harder abstract MDP",
             narrative=NARRATIVE, diagnoses=DIAGNOSES, excluded=EXCLUDED,
             results_json=RESULTS_JSON, page_path=PAGE_PATH)
