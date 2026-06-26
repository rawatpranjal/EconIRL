"""Wrapper-level recovery and guardrail tests for MCEIRL on a general MDP.

These tests drive the PUBLIC ``MCEIRL`` sklearn wrapper (not the low-level
``MCEIRLEstimator``) on a genuine action-dependent MDP that is NOT the Rust bus.
The existing wrapper tests only exercise 2-action, bus-shaped problems, so they
never touched the wrapper's transition-inference and next-state-synthesis paths.
That gap let four Rust-bus-specific defaults survive:

  * ``_dataframe_to_panel`` ignored an observed ``next_state`` column,
  * ``transitions=None`` silently built a bus "reset to state 0" kernel for a>=1,
  * the same bus-reset fired silently for any 2D transition input,
  * ``reward_`` reported only the action-0 column.

The MDP here is built with ``ArrayMDP`` (explicit action-dependent transitions
and full-rank action-contrast features), so a correct estimator must recover the
reward; a Rust-bus shortcut cannot.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from econirl.environments.array_mdp import ArrayMDP
from econirl.estimators.mce_irl import MCEIRL
from econirl.simulation.synthetic import simulate_panel

S, A, K = 8, 3, 3
THETA = np.array([1.0, -1.0, 0.5])
NAMES = ["f0", "f1", "f2"]


def _general_mdp(seed: int = 0) -> ArrayMDP:
    """A genuine action-dependent MDP (not bus-reset), full action-contrast rank."""
    rng = np.random.default_rng(seed)
    transitions = np.stack(
        [rng.dirichlet(np.full(S, 0.4), size=S) for _ in range(A)]
    )  # (A, S, S), action-dependent, full support
    features = rng.normal(size=(S, A, K))
    return ArrayMDP(
        transitions,
        features,
        theta=dict(zip(NAMES, THETA)),
        discount_factor=0.95,
        scale_parameter=1.0,
        seed=seed,
    )


def _panel_to_df(panel) -> pd.DataFrame:
    rows = []
    for traj in panel.trajectories:
        states = np.asarray(traj.states)
        actions = np.asarray(traj.actions)
        next_states = np.asarray(traj.next_states)
        for t in range(len(states)):
            rows.append(
                {
                    "id": int(traj.individual_id),
                    "period": t,
                    "state": int(states[t]),
                    "action": int(actions[t]),
                    "next_state": int(next_states[t]),
                }
            )
    return pd.DataFrame(rows)


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@pytest.fixture(scope="module")
def general_mdp():
    return _general_mdp()


@pytest.fixture(scope="module")
def general_panel(general_mdp):
    return simulate_panel(general_mdp, n_individuals=1500, n_periods=40, seed=1)


def _fit(panel_or_df, env, **fit_kwargs) -> MCEIRL:
    model = MCEIRL(
        n_states=S,
        n_actions=A,
        discount=0.95,
        feature_matrix=np.asarray(env.feature_matrix),
        feature_names=NAMES,
        se_method="hessian",
        n_bootstrap=0,
        inner_max_iter=2000,
    )
    return model.fit(panel_or_df, **fit_kwargs)


def test_recovers_with_explicit_transitions(general_mdp, general_panel):
    """Given correct (A,S,S) transitions, the wrapper recovers the reward.

    This passes on the current code and guards the math against regression.
    """
    model = _fit(general_panel, general_mdp, transitions=np.asarray(general_mdp.transition_matrices))
    feat_diff = float(model._result.metadata["feature_difference"])
    assert feat_diff < 0.05, f"feature residual too large: {feat_diff}"
    assert _cosine(model.coef_, THETA) > 0.9


def test_dataframe_next_state_column_is_used(general_mdp, general_panel):
    """fit() must consume an observed next_state column, not synthesize it."""
    df = _panel_to_df(general_panel)
    model = _fit(
        df,
        general_mdp,
        state="state",
        action="action",
        id="id",
        next_state="next_state",
        transitions=np.asarray(general_mdp.transition_matrices),
    )
    # The wrapper's internal panel must reflect the observed next-states exactly.
    by_id = {int(t.individual_id): t for t in model._panel.trajectories}
    for _id, group in df.groupby("id"):
        observed = group.sort_values("period")["next_state"].to_numpy()
        got = np.asarray(by_id[int(_id)].next_states)
        np.testing.assert_array_equal(got, observed)
    # And recovery still holds through the DataFrame path.
    assert _cosine(model.coef_, THETA) > 0.9


def test_no_transitions_multi_action_raises(general_mdp, general_panel):
    """transitions=None for a >2-action MDP must fail loud, not bus-reset."""
    model = MCEIRL(
        n_states=S,
        n_actions=A,
        discount=0.95,
        feature_matrix=np.asarray(general_mdp.feature_matrix),
        feature_names=NAMES,
        se_method="hessian",
        n_bootstrap=0,
    )
    with pytest.raises(ValueError, match="transitions"):
        model.fit(general_panel)


def test_bus_reset_emits_warning():
    """A 2D transition input triggers the bus-reset, which must announce itself."""
    rng = np.random.default_rng(2)
    s2 = 6
    transitions = np.stack([rng.dirichlet(np.full(s2, 0.5), size=s2) for _ in range(2)])
    features = rng.normal(size=(s2, 2, 2))
    env = ArrayMDP(transitions, features, theta=[1.0, -0.5], discount_factor=0.95, seed=2)
    panel = simulate_panel(env, n_individuals=200, n_periods=20, seed=3)

    keep_2d = np.asarray(transitions[0])  # 2D -> triggers the bus-reset branch
    model = MCEIRL(
        n_states=s2,
        n_actions=2,
        discount=0.95,
        feature_matrix=np.asarray(features),
        feature_names=["a", "b"],
        se_method="hessian",
        n_bootstrap=0,
        inner_max_iter=500,
    )
    with pytest.warns(UserWarning, match="bus|replacement|reset"):
        model.fit(panel, transitions=keep_2d)


def test_reward_is_policy_weighted(general_mdp, general_panel):
    """reward_ must be the policy-weighted reward, not the action-0 column."""
    model = _fit(general_panel, general_mdp, transitions=np.asarray(general_mdp.transition_matrices))
    expected = (model.policy_ * model.reward_matrix_).sum(axis=1)
    np.testing.assert_allclose(model.reward_, expected, atol=1e-5)
    # For this action-dependent reward the old action-0 column is genuinely different.
    assert not np.allclose(model.reward_, model.reward_matrix_[:, 0], atol=1e-3)
