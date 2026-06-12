"""Smoke tests for the per-experiment simulation-study harness.

Fast: a tiny cell, the cheapest estimator, one replication. Verifies the raw
JSON schema, that a crashing estimator is recorded (verbatim message) rather
than dropped, that rendering is a pure function of the JSON (byte-identical on
a second call), and that the regret fallback works on environments without a
``true_reward_matrix`` property (RustBus).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np

from econirl.environments import RustBusEnvironment, random_mdp
from validation.benchmark.harness import (
    Cell,
    RosterEntry,
    merge_estimator,
    render_console,
    render_page,
    run_experiment,
)
from validation.benchmark.regret import estimator_regret
from validation.benchmark.runner import _oracle


def _run_bc(env, panel):
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
    from validation.benchmark.runner import _linear_utility

    est = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec,
                        env.transition_matrices)


def _run_crash(env, panel):
    raise RuntimeError("deliberate test crash")


def _tiny_cell(roster) -> Cell:
    return Cell(
        cell_id="tiny",
        label="Tiny",
        description="Tiny abstract MDP for the smoke test.",
        env_factory=lambda: random_mdp(num_states=6, num_actions=2, num_features=2,
                                       branching=3, discount_factor=0.9, seed=11),
        roster=roster,
        n_individuals=80,
        n_periods=30,
        seed=1,
        n_replications=1,
    )


_NARRATIVE = {
    "title": "Test page",
    "intro": "Intro prose.",
    "script": "scripts/test.py",
    "results_rel": "validation/results/test.json",
}


def test_records_schema_and_render_idempotence() -> None:
    cell = _tiny_cell((RosterEntry("BC", "behavioral", _run_bc),))
    data = run_experiment((cell,), title="t", diagnoses={}, excluded=[])

    assert len(data["records"]) == 1
    rec = data["records"][0]
    for key in ("estimator", "family", "cell", "rep", "params", "standard_errors",
                "policy_tv", "value_rmse", "regret", "runtime", "converged", "error"):
        assert key in rec
    assert rec["error"] is None
    assert rec["policy_tv"] is not None
    assert rec["regret"] is not None and rec["regret"]["transferred"] is False
    assert data["meta"]["cells"][0]["true_theta"] is not None
    json.dumps(data)  # JSON-serializable

    page1 = render_page(data, _NARRATIVE)
    page2 = render_page(data, _NARRATIVE)
    assert page1 == page2
    assert "| BC |" in page1
    render_console(data)


def test_crash_is_recorded_not_dropped() -> None:
    cell = _tiny_cell((RosterEntry("Crasher", "behavioral", _run_crash),))
    data = run_experiment((cell,), title="t", diagnoses={}, excluded=[])
    rec = data["records"][0]
    assert rec["error"] == "RuntimeError: deliberate test crash"
    page = render_page(data, _NARRATIVE)
    assert "crashed 1/1" in page
    assert "deliberate test crash" in page


def test_max_reps_caps_slow_estimators() -> None:
    roster = (RosterEntry("BC", "behavioral", _run_bc),
              RosterEntry("BC-slow", "behavioral", _run_bc, max_reps=1))
    cell = dataclasses.replace(_tiny_cell(roster), n_replications=2)
    data = run_experiment((cell,), title="t", diagnoses={}, excluded=[])
    counts = {}
    for r in data["records"]:
        counts[r["estimator"]] = counts.get(r["estimator"], 0) + 1
    assert counts == {"BC": 2, "BC-slow": 1}


def test_merge_estimator_replaces_only_target() -> None:
    roster = (RosterEntry("BC", "behavioral", _run_bc),
              RosterEntry("Crasher", "behavioral", _run_crash))
    cell = _tiny_cell(roster)
    old = run_experiment((cell,), title="t", diagnoses={"Crasher": "old"}, excluded=[])

    fixed = _tiny_cell((RosterEntry("Crasher", "behavioral", _run_bc),))
    new = run_experiment((fixed,), title="t", diagnoses={"Crasher": "fixed"}, excluded=[])
    merged = merge_estimator(old, new, "Crasher")

    by_est = {r["estimator"]: r for r in merged["records"]}
    assert by_est["Crasher"]["error"] is None  # replaced by the fixed run
    assert by_est["BC"]["error"] is None  # untouched
    assert merged["meta"]["diagnoses"]["Crasher"] == "fixed"


def test_regret_fallback_on_rust_bus() -> None:
    # RustBus has no true_reward_matrix property; the fallback uses
    # compute_utility_matrix(). True parameters must give ~zero regret with
    # structural transfer; a policy-only estimator must be scored frozen.
    env = RustBusEnvironment(num_mileage_bins=10, operating_cost=0.01,
                             replacement_cost=2.0, discount_factor=0.9)
    oracle_policy, _ = _oracle(env)
    true_theta = np.asarray(env.get_true_parameter_vector())

    row = estimator_regret(env, true_theta, oracle_policy)
    assert row.transferred is True
    for v in (row.baseline, row.type_a, row.type_b, row.type_c):
        assert np.isfinite(v) and abs(v) < 1e-3

    frozen = estimator_regret(env, None, oracle_policy)
    assert frozen.transferred is False
    assert np.isfinite(frozen.type_c)
