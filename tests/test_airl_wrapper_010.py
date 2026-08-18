"""Load-bearing tests for the public AIRL workflow."""

from __future__ import annotations

import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.reward_spec import RewardSpec
from econirl.core.solvers import value_iteration
from validation.known_truth import (
    SimulationConfig,
    build_known_truth_dgp,
    get_cell,
    normalized_rmse,
    simulate_known_truth_panel,
)


@pytest.fixture(scope="module")
def airl_recovery_case():
    """Fit once on the paper-side state-only identification surface."""
    from econirl import AIRL

    cell = get_cell("airl_paper_identification")
    dgp = build_known_truth_dgp(cell.dgp_config)
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=300, n_periods=80, seed=1711),
    )
    reward = RewardSpec.state_dependent(
        dgp.feature_matrix[:, 0, :],
        names=dgp.parameter_names,
        n_actions=dgp.problem.num_actions,
    )
    model = AIRL(
        n_states=dgp.problem.num_states,
        n_actions=dgp.problem.num_actions,
        discount=dgp.problem.discount_factor,
        max_rounds=200,
        min_rounds=150,
        discriminator_steps=5,
        policy_step_size=0.1,
        generator_reward="f",
        compute_se=False,
        seed=1710,
    ).fit(panel, transitions=np.asarray(dgp.transitions), reward=reward)
    return dgp, model


@pytest.mark.slow
def test_airl_recovers_state_reward_and_transfers_it(airl_recovery_case):
    """The public estimator must recover behavior that survives new dynamics."""
    dgp, model = airl_recovery_case
    learned_reward = np.asarray(model.reward_, dtype=float)
    true_reward = np.asarray(dgp.homogeneous_reward[:, 0], dtype=float)

    changed = np.asarray(dgp.transitions)[[1, 2, 3, 0]].copy()
    transferred = model.counterfactual(transitions=changed)
    truth = value_iteration(
        SoftBellmanOperator(dgp.problem, jnp.asarray(changed)),
        dgp.homogeneous_reward,
        tol=1e-10,
        max_iter=5_000,
    )
    transfer_tv = 0.5 * np.mean(
        np.abs(np.asarray(transferred.counterfactual_policy) - np.asarray(truth.policy)).sum(axis=1)
    )

    assert normalized_rmse(learned_reward, true_reward) <= 0.20
    assert transfer_tv <= 0.08


def test_airl_refuses_airl_het_inputs_before_training(monkeypatch):
    """AIRL must fail closed when asked to estimate heterogeneous rewards."""
    from econirl import AIRL
    from econirl.estimation.adversarial.airl import AIRLEstimator

    monkeypatch.setattr(
        AIRLEstimator,
        "estimate",
        lambda *args, **kwargs: pytest.fail("optimizer ran before scope validation"),
    )
    model = AIRL(n_states=3, n_actions=2, compute_se=False)
    action_features = np.zeros((3, 2, 1), dtype=float)
    action_features[:, 1, 0] = 1.0
    unsupported = RewardSpec.state_action_dependent(
        jnp.asarray(action_features),
        names=["action"],
    )

    with pytest.raises(ValueError, match="AIRLHet"):
        model.fit(None, transitions=np.zeros((2, 3, 3)), reward=unsupported)
    with pytest.raises(NotImplementedError, match="AIRLHet"):
        model.fit(None, transitions=np.zeros((2, 3, 3)), context=np.zeros(1))


def test_airl_serialization_preserves_transfer_decision(airl_recovery_case):
    """A persisted public fit must make the same changed-dynamics decision."""

    dgp, model = airl_recovery_case
    changed = np.asarray(dgp.transitions)[[1, 2, 3, 0]].copy()
    before = model.counterfactual(transitions=changed).counterfactual_policy
    restored = pickle.loads(pickle.dumps(model))
    after = restored.counterfactual(transitions=changed).counterfactual_policy

    np.testing.assert_allclose(after, before, atol=1e-8)
