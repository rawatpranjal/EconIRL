"""Simulation study: high-dimensional abstract MDP (machine-learning estimators).

The page's claim is that classical structural solvers stop being practical at a
few thousand states while the machine-learning estimators keep running. That
claim is demonstrated, not asserted: ``--probe`` runs a single fit of every
candidate (classical and ML) in its own subprocess with a hard timeout, and the
measured outcomes (seconds, timeout, or the verbatim crash) are stored in the
results JSON and rendered as a first-class page section. The main cell then
runs the estimators that the probes showed to be feasible.

Generates ``validation/results/sim_abstract_mdp_3.json`` (and
``..._probes.json``) and renders ``docs/simulation_studies/abstract_mdp_3_highdim.md``.
Honesty contract in ``validation/benchmark/harness.py``.

Usage:
    python scripts/sim_abstract_mdp_3.py --probe          # feasibility probes first
    python scripts/sim_abstract_mdp_3.py [--verbose]      # main run (embeds probes)
    python scripts/sim_abstract_mdp_3.py --page
    python scripts/sim_abstract_mdp_3.py --verify
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from econirl.environments import random_mdp  # noqa: E402
from validation.benchmark.harness import Cell, RosterEntry, main_cli  # noqa: E402
from validation.benchmark.runner import _action_reward, _linear_utility  # noqa: E402

RESULTS_JSON = os.path.join(_ROOT, "validation", "results", "sim_abstract_mdp_3.json")
PROBES_JSON = os.path.join(_ROOT, "validation", "results", "sim_abstract_mdp_3_probes.json")
PAGE_PATH = os.path.join(_ROOT, "docs", "simulation_studies", "abstract_mdp_3_highdim.md")

# 3000 states: the dense (2, S, S) transition tensor every tabular method
# consumes is ~144 MB here and grows quadratically; this is the regime the
# probes interrogate.
N_STATES = 3000
MDP = dict(num_states=N_STATES, num_actions=2, num_features=3, branching=8,
           discount_factor=0.95, seed=707)
N_INDIVIDUALS = 500
N_PERIODS = 60
PROBE_TIMEOUT = 900  # seconds per probe fit


def _env():
    return random_mdp(**MDP)


# ---------------------------------------------------------------------------
# Estimators. Classical solvers appear only in the probes; the main roster is
# the ML family plus behavioral cloning as the cheap frozen baseline.
# Hyperparameters for the ML family follow the package's scaling presets
# (econirl.evaluation.benchmark.get_scaling_estimator_specs).
# ---------------------------------------------------------------------------


def _run_nfxp_sa(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(inner_solver="sa", inner_tol=1e-10,
                        inner_max_iter=100000, compute_hessian=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_nfxp_nk(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(inner_solver="polyalgorithm", inner_tol=1e-10,
                        inner_max_iter=100000, compute_hessian=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_ccp(env, panel):
    from econirl.estimation import CCPEstimator

    est = CCPEstimator(num_policy_iterations=1, compute_hessian=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_mpec(env, panel):
    from econirl.estimation.mpec import MPECConfig, MPECEstimator

    # One SLSQP variable per state plus the parameters: 3003 variables here.
    est = MPECEstimator(config=MPECConfig(solver="slsqp", max_iter=200, constraint_tol=1e-6),
                        compute_hessian=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_tdccp(env, panel):
    from econirl.estimation import TDCCPConfig, TDCCPEstimator

    # Scaling preset: wider updates, sampled batches, no dense fixed point.
    est = TDCCPEstimator(config=TDCCPConfig(hidden_dim=64, avi_iterations=30,
                                            epochs_per_avi=20, learning_rate=5e-4,
                                            batch_size=512, compute_se=False,
                                            verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_nnes(env, panel):
    from econirl.estimation.nnes import NNESEstimator

    # Scaling preset: v_epochs grows with the state count.
    est = NNESEstimator(hidden_dim=64, v_epochs=1000, outer_max_iter=200,
                        compute_se=False, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_gladius(env, panel):
    from econirl.estimation import GLADIUSConfig, GLADIUSEstimator

    # Scaling preset: sampled batches, Bellman penalty, weight decay.
    est = GLADIUSEstimator(config=GLADIUSConfig(max_epochs=500, q_hidden_dim=64,
                                                v_hidden_dim=64, q_num_layers=2,
                                                v_num_layers=2, batch_size=256,
                                                bellman_penalty_weight=0.1,
                                                weight_decay=1e-3,
                                                compute_se=False, verbose=False))
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_deep_mce_irl(env, panel):
    from types import SimpleNamespace

    from econirl.estimators.mceirl_neural import MCEIRLNeural

    # sklearn-style .fit interface; each epoch re-solves soft value iteration
    # on the dense tensor, so the probe decides whether this is feasible here.
    m = MCEIRLNeural(n_states=int(env.num_states), n_actions=int(env.num_actions),
                     discount=float(env.problem_spec.discount_factor),
                     max_epochs=200, verbose=False)
    m.fit(panel, features=np.asarray(env.feature_matrix),
          transitions=np.asarray(env.transition_matrices))
    return SimpleNamespace(parameters=m.coef_, standard_errors=None, policy=m.policy_,
                           value_function=m.value_, converged=bool(m.converged_))


def _run_bc(env, panel):
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator

    est = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


_ALL_RUNNERS = {
    "NFXP-SA": _run_nfxp_sa,
    "NFXP-NK": _run_nfxp_nk,
    "CCP": _run_ccp,
    "MPEC": _run_mpec,
    "TD-CCP": _run_tdccp,
    "NNES": _run_nnes,
    "GLADIUS": _run_gladius,
    "Deep-MCE-IRL": _run_deep_mce_irl,
    "BC": _run_bc,
}

PROBE_ORDER = ["NFXP-SA", "NFXP-NK", "CCP", "MPEC",
               "TD-CCP", "NNES", "GLADIUS", "Deep-MCE-IRL"]

ROSTER = (
    RosterEntry("TD-CCP", "structural", _run_tdccp),
    RosterEntry("NNES", "structural", _run_nnes),
    RosterEntry("GLADIUS", "behavioral", _run_gladius),
    RosterEntry("Deep-MCE-IRL", "behavioral", _run_deep_mce_irl),
    RosterEntry("BC", "behavioral", _run_bc),
)


DIAGNOSES = {
    "TD-CCP": "Neural CCP with approximate value iteration on sampled batches; "
              "never materializes a dense fixed point.",
    "NNES": "Neural value network plus structural MLE; the network replaces the "
            "inner solve.",
    "GLADIUS": "Neural Q and expected-value networks trained on sampled batches "
               "with a Bellman penalty.",
    "Deep-MCE-IRL": "Neural-reward MCE-IRL; each epoch still re-solves soft value "
                    "iteration on the dense tensor, so it sits between the "
                    "classical and sampled families. Parameters are the neural "
                    "reward projected onto the linear features.",
    "BC": "Behavioral cloning, the cheap frozen baseline: counts choices where "
          "data exists, knows nothing anywhere else, recovers no reward.",
}

EXCLUDED = [
    {"name": "NFXP, CCP, MPEC", "reason": "shown in the feasibility probes above "
     "with their measured outcomes rather than asserted away"},
    {"name": "MCE-IRL, MaxEnt-IRL, AIRL, IQ-Learn, f-IRL and the other IRL "
     "methods", "reason": "their exact inner solvers face the same dense-tensor "
     "cost the probes document for the classical family; the IRL comparison "
     "lives on the Rust bus and gridworld pages"},
]

CELLS = (
    Cell(
        cell_id="highdim_3000",
        label=f"{N_STATES} states",
        description=(
            f"A {N_STATES}-state Garnet MDP with stochastic sparse transitions "
            "(branching 8) and a 3-feature linear reward: "
            f"`random_mdp(num_states={N_STATES}, num_actions=2, num_features=3, "
            f"branching=8, discount_factor=0.95, seed=707)`. The panel "
            f"({N_INDIVIDUALS} individuals x {N_PERIODS} periods) covers at most "
            f"{N_INDIVIDUALS * N_PERIODS} state visits, so most states are "
            "rarely or never observed; estimators reach them only through the "
            "reward features, not through memorized choices."
        ),
        env_factory=_env,
        roster=ROSTER,
        n_individuals=N_INDIVIDUALS,
        n_periods=N_PERIODS,
        seed=707,
        n_replications=3,
    ),
)

NARRATIVE = {
    "title": "Abstract MDP 3: high dimension",
    "intro": (
        "The last rung of the ladder asks what survives when the state space "
        "reaches a few thousand states. Every tabular structural solver "
        "consumes a dense transition tensor whose memory and per-iteration cost "
        "grow with the square of the state count, and an optimizer like MPEC "
        "additionally carries one variable per state. Rather than assert where "
        "that breaks, the feasibility probes below run every candidate once "
        "under a hard time budget and report what actually happened; the main "
        "table then benchmarks the estimators that remain practical."
    ),
    "cells": {
        "highdim_3000": {
            "after": (
                "Behavioral cloning is the control group: it is nearly free and "
                "matches the data where the data exists, but it carries no "
                "reward, so it cannot say anything at unvisited states or under "
                "the counterfactual interventions. The gap between its regret "
                "and the reward-recovering estimators' regret is the value of "
                "estimating structure at this scale."
            ),
        },
    },
    "script": "scripts/sim_abstract_mdp_3.py",
    "results_rel": "validation/results/sim_abstract_mdp_3.json",
}


# ---------------------------------------------------------------------------
# Feasibility probes
# ---------------------------------------------------------------------------


def _probe_one(name: str) -> None:
    """Child-process entry: one fit, result as JSON on the last stdout line."""
    env = _env()
    panel_seed = MDP["seed"] + 1000  # same panel as replication 0 of the main run
    from econirl.simulation.synthetic import simulate_panel

    panel = simulate_panel(env, n_individuals=N_INDIVIDUALS,
                           n_periods=N_PERIODS, seed=panel_seed)
    t0 = time.time()
    try:
        _ALL_RUNNERS[name](env, panel)
        out = {"estimator": name, "n_states": N_STATES, "outcome": "completed",
               "seconds": time.time() - t0, "error": None}
    except Exception as exc:  # noqa: BLE001 - the failure IS the result
        out = {"estimator": name, "n_states": N_STATES, "outcome": "crashed",
               "seconds": time.time() - t0,
               "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(out))


def _run_probes() -> None:
    """Run every probe serially, each in a subprocess with a hard timeout."""
    results = []
    for name in PROBE_ORDER:
        print(f"probe {name} at {N_STATES} states (budget {PROBE_TIMEOUT}s)...",
              flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--probe-one", name],
                capture_output=True, text=True, timeout=PROBE_TIMEOUT)
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                rec = {"estimator": name, "n_states": N_STATES, "outcome": "crashed",
                       "seconds": time.time() - t0,
                       "error": (proc.stderr.strip().splitlines() or ["no output"])[-1][:200]}
        except subprocess.TimeoutExpired:
            rec = {"estimator": name, "n_states": N_STATES, "outcome": "timeout",
                   "seconds": None,
                   "error": f"killed at the {PROBE_TIMEOUT:.0f}s budget"}
        results.append(rec)
        print(f"  -> {rec['outcome']}"
              + (f" in {rec['seconds']:.1f}s" if rec.get("seconds") else "")
              + (f" ({rec['error']})" if rec.get("error") else ""), flush=True)

    payload = {
        "description": (
            "Single fits at the main cell's exact configuration (same "
            "environment, same replication-0 panel), one subprocess per "
            "estimator, run before the main benchmark to decide the roster "
            "empirically."
        ),
        "timeout_seconds": PROBE_TIMEOUT,
        "probes": results,
    }
    with open(PROBES_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {PROBES_JSON}")


def _extra_meta() -> dict | None:
    if os.path.exists(PROBES_JSON):
        return {"feasibility_probes": json.load(open(PROBES_JSON))}
    return None


if __name__ == "__main__":
    if "--probe-one" in sys.argv:
        _probe_one(sys.argv[sys.argv.index("--probe-one") + 1])
    elif "--probe" in sys.argv:
        _run_probes()
    else:
        main_cli(cells=CELLS, title="Simulation study: high-dimensional abstract MDP",
                 narrative=NARRATIVE, diagnoses=DIAGNOSES, excluded=EXCLUDED,
                 results_json=RESULTS_JSON, page_path=PAGE_PATH,
                 extra_meta=_extra_meta())
