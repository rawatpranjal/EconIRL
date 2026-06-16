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

import dataclasses
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

    est = MPECEstimator(config=MPECConfig(solver="sqp", outer_max_iter=200, tol=1e-8, constraint_tol=1e-6),
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

    # Bray's unnested fixed point with optimal weighting (OUFXP): closed form
    # for linear utility, MLE-efficient, standard errors from the efficient
    # moment variance.
    est = UFXPEstimator(weights="optimal", verbose=False)
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
# Cell B: NFXP-SA's slow contraction at discount 0.99 is a runtime story, not
# an inference story; two measured fits carry it (the same MLE's coverage is
# carried by NFXP-NK at the full replication count). Each fit gets a hard
# 30-minute budget, recorded as a timeout if exceeded.
ROSTER_B = tuple(
    dataclasses.replace(e, max_reps=2, timeout=1800) if e.name == "NFXP-SA" else e
    for e in _STRUCTURAL
)
ROSTER_C = (
    RosterEntry("NFXP-NK", "structural", _run_nfxp_nk),
    RosterEntry("CCP", "structural", _run_ccp),
    RosterEntry("MPEC", "structural", _run_mpec),
    RosterEntry("UFXP", "structural", _run_ufxp),
)


DIAGNOSES = {
    "NFXP-SA": "Rust's original successive-approximation inner loop. Same "
               "maximum-likelihood answer as NFXP-NK, different runtime.",
    "CCP": "Its standard errors come from the outer Hessian and can fail to "
           "be finite even when the point estimate is fine. The SE avail "
           "column makes that visible.",
    "UFXP": "Unnested fixed point (Bray; Oguz and Bray 2026) with optimal "
            "weighting. As efficient as maximum likelihood, with standard "
            "errors, so it enters the coverage table on equal terms.",
    "MCE-IRL": "Behavioral reference. Its convergence indicator is a conservative "
               "gradient-norm check, so read it next to Policy TV.",
}

EXCLUDED = [
    {"name": "MaxEnt-IRL, IQ-Learn, AIRL, f-IRL, GLADIUS, Deep-MCE-IRL, BC",
     "reason": "this page's question is structural. Parameter recovery, "
     "inference quality, and identification as the problem hardens. The IRL "
     "family is compared on the bus engine and gridworld pages. MCE-IRL "
     "stays here as the behavioral reference"},
    {"name": "GAIL, GCL, DeepMaxEnt-IRL, Bayesian-IRL",
     "reason": "research code or too slow; not benchmarked in this study"},
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
        fit_timeout=1800,
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
        fit_timeout=1800,
    ),
    Cell(
        cell_id="rank_deficient",
        label="Collinear features (24 states)",
        description=(
            "A small MDP whose third reward feature is exactly twice the "
            "second (design rank 2 of 3). The likelihood identifies only the "
            "combination theta_1 + 2 theta_2, so no estimator can recover "
            "the individual coordinates."
        ),
        env_factory=_rank_deficient,
        roster=ROSTER_C,
        n_individuals=500,
        n_periods=80,
        seed=606,
        n_replications=20,
        # Individual coordinates are not identified, so per-coordinate
        # parameter tables would print arbitrary ridge points.
        param_block=False,
        show_params=False,
        fit_timeout=600,
    ),
)

NARRATIVE = {
    "title": "Abstract MDP 2",
    "intro": (
        "The sanity-check page showed every estimator recovering an easy "
        "problem. This page hardens the problem along three separate axes and "
        "watches the structural family specifically. What happens to runtime "
        "as the state space grows? What happens to inference as the discount "
        "factor approaches one? What happens to the parameters when the "
        "reward features are collinear? Each axis gets its own cell, run on "
        "the same engine and reported from the same raw records as every "
        "other page.\n"
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
        "index $x_s = s/(S-1)$. Action $0$ is a zeroed outside option, the "
        "identification anchor. For $a \\geq 1$,\n"
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
        "MDP whose features are deliberately collinear. Its construction is "
        "described in that cell."
    ),
    "cells": {
        "harder_300": {
            "before": (
                "The first cell is about cost at scale. All estimators face "
                "the same 300-state problem, and the runtime column is the "
                "result. The two NFXP rows are the same estimator with two "
                "inner solvers, Rust's original successive approximation "
                "against the Newton-Kantorovich polyalgorithm. The "
                "refinement's value is measured on this page rather than "
                "asserted."
            ),
            "after": (
                "The two NFXP rows land within a second of each other, so "
                "the textbook solver gap does not bite at 300 states. The "
                "high-dimension page is where it starts to. The "
                "approximation-based members (SEES, TD-CCP) give up some "
                "parameter precision relative to the exact family while "
                "staying close on behavior."
            ),
        },
        "high_discount": {
            "before": (
                "The second cell moves the discount factor to 0.99 and asks "
                "whether the reported uncertainty is usable. The parameter "
                "table shows bias, the spread across replications, RMSE, "
                "coverage of the nominal 95% intervals, and how often each "
                "estimator produced finite standard errors. NFXP-SA runs 2 "
                "of 10 replications as a runtime spot-check. Its inference "
                "is the same MLE as NFXP-NK, which runs all 10."
            ),
            "after": (
                "The SE avail column is the headline. One estimator "
                "routinely fails to deliver usable standard errors here "
                "while recovering good point estimates. Without that column, "
                "the blank coverage entries would read as a formatting gap "
                "rather than an inference failure."
            ),
        },
        "rank_deficient": {
            "before": (
                "The last cell breaks identification on purpose. The third "
                "feature is exactly twice the second, so only the "
                "combination theta_1 + 2 theta_2 is identified, and the "
                "design diagnostics above flag it. The parameter columns "
                "are omitted. Every estimator still matches behavior, which "
                "is what partial identification looks like in practice."
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
