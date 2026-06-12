"""Simulation study: Rust (1987) bus engine replacement.

Generates ``validation/results/sim_rust_bus.json`` and renders
``docs/simulation_studies/rust_bus.md`` from it via the shared harness
(``validation/benchmark/harness.py``), inheriting the honesty contract of
``scripts/quick_all_estimators.py``: raw per-replication facts, page as a pure
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

# The canonical recoverable cell: identical to the prior benchmark ladder's
# simple_binary configuration, so this page genuinely re-homes that cell.
ENV = dict(num_mileage_bins=20, operating_cost=0.01, replacement_cost=2.0,
           discount_factor=0.95)


def _env():
    return RustBusEnvironment(**ENV)


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
                        compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_sees(env, panel):
    from econirl.estimation.sees import SEESEstimator

    # Basis must span the value function: bspline basis_dim >= num_states (20).
    est = SEESEstimator(basis_type="bspline", basis_dim=20, warm_start_value=True,
                        penalty_weight=10.0, compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_tdccp(env, panel):
    from econirl.estimation import TDCCPConfig, TDCCPEstimator

    est = TDCCPEstimator(config=TDCCPConfig(hidden_dim=64, avi_iterations=15,
                                            epochs_per_avi=15, compute_se=False, verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_ufxp(env, panel):
    from econirl.estimation import UFXPEstimator

    # Bray's unnested fixed point: the value-function dependence of the Bellman
    # first-order conditions is removed by duals computed once before the
    # search, so the linear-utility case is closed-form least squares.
    est = UFXPEstimator(num_projections=64, seed=0, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_mce_irl(env, panel):
    from econirl.estimation import MCEIRLConfig, MCEIRLEstimator

    est = MCEIRLEstimator(config=MCEIRLConfig(learning_rate=0.05, outer_max_iter=100,
                                              inner_max_iter=2000, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_maxent_irl(env, panel):
    from econirl.contrib.maxent_irl import MaxEntIRLEstimator

    # Feed the action-dependent features: a state-only reward cannot represent
    # the action contrast that drives the keep/replace choice.
    est = MaxEntIRLEstimator(inner_tol=1e-8, inner_max_iter=5000, outer_max_iter=500,
                             compute_hessian=False, verbose=False)
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
    # across actions. Recovered parameters stay gauge/shaping-unidentified by
    # design, so policy TV is the right scorecard.
    est = AIRLEstimator(config=AIRLConfig(reward_type="linear", reward_arg="state_action",
                                          reward_lr=0.01, discriminator_steps=10,
                                          max_rounds=300, compute_se=False, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_firl(env, panel):
    from econirl.estimation.f_irl import FIRLEstimator

    est = FIRLEstimator(f_divergence="chi2", lr=0.5, max_iter=400, reward_clip=100.0,
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


def _run_mmp(env, panel):
    from econirl.contrib.max_margin_planning import MaxMarginPlanningEstimator

    est = MaxMarginPlanningEstimator(verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_gcl(env, panel):
    from econirl.contrib.gcl import GCLConfig, GCLEstimator

    est = GCLEstimator(config=GCLConfig(hidden_dims=[32, 32], cost_lr=0.001,
                                        max_iterations=100, n_sample_trajectories=50,
                                        verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_gail(env, panel):
    from econirl.contrib.gail import GAILConfig, GAILEstimator

    # Requires a reward spec (LinearReward/ActionDependentReward), not the
    # structural LinearUtility wrapper.
    est = GAILEstimator(GAILConfig(discriminator_type="tabular", discriminator_lr=0.05,
                                   discriminator_steps=3, policy_step_size=0.3,
                                   max_rounds=300, reward_transform="softplus",
                                   convergence_tol=1e-6, verbose=False))
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_deep_maxent(env, panel):
    from econirl.contrib.deep_maxent_irl import DeepMaxEntIRLEstimator

    est = DeepMaxEntIRLEstimator(hidden_dims=[32, 32], lr=1e-3, max_epochs=300,
                                 compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_bayesian_irl(env, panel):
    from econirl.contrib.bayesian_irl import BayesianIRLEstimator

    est = BayesianIRLEstimator(n_samples=2000, burnin=500, compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


ROSTER = (
    RosterEntry("NFXP", "structural", _run_nfxp),
    RosterEntry("CCP", "structural", _run_ccp),
    RosterEntry("MPEC", "structural", _run_mpec),
    RosterEntry("NNES", "structural", _run_nnes),
    RosterEntry("SEES", "structural", _run_sees),
    RosterEntry("TD-CCP", "structural", _run_tdccp),
    RosterEntry("UFXP", "structural", _run_ufxp),
    RosterEntry("MCE-IRL", "behavioral", _run_mce_irl),
    RosterEntry("MaxEnt-IRL", "behavioral", _run_maxent_irl),
    RosterEntry("IQ-Learn", "behavioral", _run_iq_learn),
    RosterEntry("GLADIUS", "behavioral", _run_gladius),
    RosterEntry("AIRL", "behavioral", _run_airl),
    RosterEntry("f-IRL", "behavioral", _run_firl),
    RosterEntry("Deep-MCE-IRL", "behavioral", _run_deep_mce_irl),
    RosterEntry("MaxMargin-IRL", "behavioral", _run_max_margin),
    RosterEntry("MMP", "behavioral", _run_mmp),
    RosterEntry("BC", "behavioral", _run_bc),
    # Known-slow estimators run once each, visibly (Ran column shows 1/1),
    # rather than being excluded.
    RosterEntry("GCL", "behavioral", _run_gcl, max_reps=1),
    RosterEntry("GAIL", "behavioral", _run_gail, max_reps=1),
    RosterEntry("DeepMaxEnt-IRL", "behavioral", _run_deep_maxent, max_reps=1),
    RosterEntry("Bayesian-IRL", "behavioral", _run_bayesian_irl, max_reps=1),
)


DIAGNOSES = {
    "NFXP": "Reference structural estimator; recovers cleanly.",
    "CCP": "Hotz-Miller conditional choice probabilities; recovers cleanly.",
    "MPEC": "Constrained MLE; recovers cleanly.",
    "NNES": "Neural value network plus structural MLE.",
    "SEES": "bspline basis with basis_dim >= num_states (20); a smaller basis "
            "underfits the value function and biases theta.",
    "TD-CCP": "Neural CCP with approximate value iteration.",
    "UFXP": "Unnested fixed point (Bray; Oguz and Bray 2026). Scores projected "
            "Bellman first-order conditions; the value function is eliminated by "
            "duals computed once before the parameter search, making the linear "
            "case closed-form. Uses random-projection weights (not the efficient "
            "OUFXP second step) and reports no standard errors, so expect "
            "consistency with some efficiency loss against the MLE family.",
    "MCE-IRL": "Causal maximum-entropy IRL. Its converged flag reports whether the "
               "gradient norm crossed the tolerance; the objective often plateaus "
               "first, so the flag can read False while the recovered policy is "
               "essentially exact. Read it next to Policy TV.",
    "MaxEnt-IRL": "Fed action-dependent features; a state-only reward is broadcast "
                  "equally across actions and cannot represent the action contrast.",
    "IQ-Learn": "q_type='linear' uses the feature structure; a tabular Q-table does "
                "not propagate to unvisited states.",
    "GLADIUS": "Neural Q and expected-value networks; tracks behavior.",
    "AIRL": "reward_arg='state_action'; recovered parameters stay gauge/shaping-"
            "unidentified by design, so policy TV is the right scorecard.",
    "f-IRL": "f-divergence IRL; tracks behavior.",
    "Deep-MCE-IRL": "Neural-reward MCE-IRL via its sklearn-style fit interface; "
                    "parameters are the neural reward projected onto the linear "
                    "features.",
    "MaxMargin-IRL": "Margin-based reward recovery (Ng-Russell tradition); no "
                     "probabilistic choice model, so no standard errors.",
    "MMP": "Max-margin planning (Ratliff et al); structured-margin variant.",
    "BC": "Behavioral cloning; matches observed choices but recovers no reward, so "
          "it cannot transfer to a counterfactual world.",
    "GCL": "Guided cost learning; sampling-based, slow, run once.",
    "GAIL": "Adversarial imitation; matches occupancy, recovers no transferable "
            "reward. Slow, run once.",
    "DeepMaxEnt-IRL": "Neural-network reward via feature matching. Slow, run once.",
    "Bayesian-IRL": "MCMC over rewards; posterior mean reward. Slow, run once.",
}

EXCLUDED = [
    {"name": "AIRL-Het / AAIRL", "reason": "designed for latent-type heterogeneity; "
     "this panel has a single agent type"},
]

CELLS = (
    Cell(
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
        n_replications=3,
    ),
)

NARRATIVE = {
    "title": "Bus engine replacement",
    "intro": (
        "The canonical structural benchmark. A single agent decides each period "
        "whether to keep paying a mileage-dependent operating cost or pay a fixed "
        "replacement cost to reset the bus engine. The data-generating process is "
        "fully known, so the table reports the exact recovered cost parameters, "
        "the distance between each estimator's policy and the true one, and the "
        "welfare lost when the recovered model is carried into three "
        "counterfactual worlds. Every estimator that accepts the package's "
        "uniform estimate interface is run; slow ones run once instead of being "
        "dropped."
    ),
    "cells": {
        "rust_bus": {
            "after": (
                "Reading the table: the structural family (NFXP, CCP, MPEC, NNES, "
                "SEES, TD-CCP) recovers the cost parameters in the same gauge as "
                "the truth, so Param RMSE applies to it alone. The IRL family is "
                "scored on behavior and regret; reward parameters from these "
                "methods live in a different gauge (reward is only partially "
                "identified from behavior), so parameter-level comparisons "
                "across the divide would be meaningless. Estimators that recover "
                "a reward in the linear feature gauge adapt under the Type A/B/C "
                "interventions; policy-only methods keep their frozen policy, "
                "which is exactly why their Type C regret is large."
            ),
        },
    },
    "script": "scripts/sim_rust_bus.py",
    "results_rel": "validation/results/sim_rust_bus.json",
}


if __name__ == "__main__":
    main_cli(cells=CELLS, title="Simulation study: bus engine replacement",
             narrative=NARRATIVE, diagnoses=DIAGNOSES, excluded=EXCLUDED,
             results_json=RESULTS_JSON, page_path=PAGE_PATH)
