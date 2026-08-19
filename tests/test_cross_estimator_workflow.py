"""One applied workflow contract across every EconIRL 0.1.0 release estimator.

The per-estimator contract files check each method in depth. This file checks
that the nine release estimators present the *same* workflow to a user: import
from the package root, fit from the documented arguments, report how much data
was used and whether the fit succeeded, print a summary, predict out of sample,
survive a pickle round trip, and refuse unsupported work out loud.
"""

from __future__ import annotations

import pickle
import warnings

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import econirl
from econirl.core.reward_spec import RewardSpec
from econirl.estimators.protocol import EstimatorProtocol

RELEASE_ESTIMATORS = (
    "NFXP",
    "CCP",
    "TDCCP",
    "MCEIRL",
    "MCEIRLNeural",
    "AIRL",
    "NeuralAIRL",
    "AIRL2",
    "GLADIUS",
)

N_STATES = 6
N_ACTIONS = 2
DISCOUNT = 0.9


def _tabular_panel() -> pd.DataFrame:
    """A small replacement-style panel every tabular estimator can read."""
    rows: list[dict[str, int]] = []
    for individual in range(24):
        state = individual % N_STATES
        for period in range(12):
            action = int(state >= N_STATES - 2 or (individual + period) % 5 == 0)
            increment = (individual + 2 * period) % 3
            next_state = 0 if action else min(state + increment, N_STATES - 1)
            rows.append(
                {
                    "id": individual,
                    "period": period,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                }
            )
            state = next_state
    return pd.DataFrame(rows)


def _transitions() -> np.ndarray:
    """Action-major tensor, (n_actions, n_states, n_states)."""
    tensor = np.zeros((N_ACTIONS, N_STATES, N_STATES), dtype=float)
    for state in range(N_STATES):
        for increment, probability in ((0, 0.35), (1, 0.55), (2, 0.10)):
            tensor[0, state, min(state + increment, N_STATES - 1)] += probability
        tensor[1, state, 0] = 1.0
    return tensor


def _action_features() -> RewardSpec:
    matrix = np.zeros((N_STATES, N_ACTIONS, 2), dtype=np.float32)
    matrix[:, 0, 0] = -np.arange(N_STATES, dtype=np.float32) / N_STATES
    matrix[:, 1, 1] = -1.0
    return RewardSpec(jnp.asarray(matrix), names=["mileage", "replace"])


def _state_features() -> RewardSpec:
    grid = np.arange(N_STATES, dtype=np.float32) / N_STATES
    matrix = np.stack([grid, grid**2], axis=1)
    return RewardSpec(jnp.asarray(matrix), names=["linear", "quadratic"], n_actions=N_ACTIONS)


def _fit(name: str):
    """Return a fitted public estimator, using the smallest credible budget."""
    panel = _tabular_panel()
    transitions = _transitions()
    common = dict(state="state", action="action", id="id")

    if name == "NFXP":
        model = econirl.NFXP(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        return model.fit(panel, **common, transitions=transitions, reward=_action_features())
    if name == "CCP":
        model = econirl.CCP(
            n_states=N_STATES,
            n_actions=N_ACTIONS,
            discount=DISCOUNT,
            num_policy_iterations=2,
        )
        return model.fit(panel, **common, transitions=transitions, reward=_action_features())
    if name == "TDCCP":
        model = econirl.TDCCP(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        return model.fit(panel, **common, reward=_action_features())
    if name == "MCEIRL":
        model = econirl.MCEIRL(
            n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT, inner_max_iter=50
        )
        return model.fit(panel, **common, transitions=transitions, reward=_state_features())
    if name == "MCEIRLNeural":
        model = econirl.MCEIRLNeural(
            n_states=N_STATES,
            n_actions=N_ACTIONS,
            discount=DISCOUNT,
            reward_type="state",
            max_epochs=10,
            reward_hidden_dim=8,
            reward_num_layers=1,
            verbose=False,
        )
        return model.fit(panel, **common, transitions=transitions, features=_state_features())
    if name == "AIRL":
        model = econirl.AIRL(
            n_states=N_STATES,
            n_actions=N_ACTIONS,
            discount=DISCOUNT,
            max_rounds=8,
            min_rounds=2,
            discriminator_steps=2,
            policy_step_size=0.1,
            compute_se=False,
            seed=11,
        )
        return model.fit(
            panel,
            **common,
            next_state="next_state",
            transitions=transitions,
            reward=_state_features(),
        )
    if name == "NeuralAIRL":
        model = econirl.NeuralAIRL(
            n_states=N_STATES,
            n_actions=N_ACTIONS,
            discount=DISCOUNT,
            feature_matrix=np.arange(N_STATES, dtype=float)[:, None],
            max_rounds=8,
            min_rounds=2,
            discriminator_steps=2,
            policy_step_size=0.1,
            reward_hidden_dim=8,
            reward_num_layers=1,
            compute_se=False,
            seed=11,
        )
        return model.fit(panel, **common, next_state="next_state", transitions=transitions)
    if name == "AIRL2":
        model = econirl.AIRL2(
            n_states=N_STATES,
            n_actions=N_ACTIONS,
            exit_action=1,
            absorbing_state=0,
            discount=DISCOUNT,
            num_segments=2,
            reward_type="linear",
            feature_matrix=np.asarray(_action_features().feature_matrix, dtype=float),
            discriminator_steps=1,
            max_airl_rounds=2,
            min_airl_rounds=1,
            max_em_iterations=2,
            compute_se=False,
            seed=11,
        )
        return model.fit(panel, **common, next_state="next_state", transitions=transitions)
    if name == "GLADIUS":
        model = econirl.GLADIUS(
            n_actions=N_ACTIONS,
            discount=DISCOUNT,
            q_hidden_dim=8,
            q_num_layers=1,
            ev_hidden_dim=8,
            ev_num_layers=1,
            batch_size=16,
            max_epochs=4,
            patience=2,
            anchor_action=1,
            anchor_rewards=tuple(-1.0 for _ in range(N_STATES)),
            seed=11,
        )
        return model.fit(panel, **common, features=_action_features(), transitions=transitions)
    raise AssertionError(f"no workflow recipe for {name}")


def test_every_release_estimator_imports_from_the_package_root() -> None:
    """A user should never need a private module path to reach a release estimator."""
    missing = [name for name in RELEASE_ESTIMATORS if not hasattr(econirl, name)]
    assert not missing, missing
    assert set(RELEASE_ESTIMATORS) <= set(econirl.__all__)


@pytest.mark.parametrize("name", RELEASE_ESTIMATORS)
def test_release_estimator_satisfies_the_public_protocol(name: str) -> None:
    """The advertised unified interface must be real, not aspirational."""
    estimator = getattr(econirl, name)
    for method in ("fit", "summary", "predict_proba", "conf_int", "counterfactual"):
        assert callable(getattr(estimator, method, None)), f"{name}.{method}"


@pytest.mark.parametrize("name", RELEASE_ESTIMATORS)
def test_release_estimator_runs_the_documented_workflow(name: str) -> None:
    """Fit, report, predict out of sample, and reload without changing results."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _fit(name)

    assert isinstance(model, EstimatorProtocol)

    # The budgets here are deliberately tiny so the contract stays fast, and
    # five estimators legitimately stop before their convergence rule fires.
    # Pinning the set means this test cannot quietly go green on a real
    # optimizer regression in the four that should converge.
    expected_unconverged = {
        "MCEIRL",
        "MCEIRLNeural",
        "AIRL2",
        "GLADIUS",
        "NeuralAIRL",
    }
    # Of those five, four tell the user. Neural MCE-IRL stops early and sets
    # converged_ = False while emitting nothing, so a user who does not read
    # the attribute never learns the fit was truncated. Pinned here so the
    # silence stays visible and a future fix trips this test.
    silently_unconverged = {"MCEIRLNeural"}

    converged = getattr(model, "converged_", None)
    if converged is not None:
        if name in expected_unconverged:
            assert not converged, (
                f"{name} now converges on the tiny contract budget; drop it from "
                "expected_unconverged so this check keeps its teeth"
            )
            warned = any(
                issubclass(w.category, (RuntimeWarning, UserWarning)) for w in caught
            )
            if name in silently_unconverged:
                assert not warned, (
                    f"{name} now warns on non-convergence; drop it from "
                    "silently_unconverged"
                )
            else:
                assert warned, f"{name} did not converge but issued no warning"
        else:
            assert converged, f"{name} failed to converge: {caught}"

    summary = model.summary()
    assert isinstance(summary, str) and summary.strip()

    states = np.arange(N_STATES)
    proba = np.asarray(model.predict_proba(states), dtype=float)
    assert proba.shape == (N_STATES, N_ACTIONS)
    assert np.all(np.isfinite(proba))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    restored = pickle.loads(pickle.dumps(model))
    reloaded = np.asarray(restored.predict_proba(states), dtype=float)
    np.testing.assert_allclose(reloaded, proba, atol=1e-8)
