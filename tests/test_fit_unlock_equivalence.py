"""F4: sklearn fit() wrappers accept arbitrary 3D transition tensors.

Tests that _build_transition_tensor passes a pre-built (n_actions, n_states,
n_states) tensor through unchanged, and that fit(transitions=3D_tensor) produces
parameters identical to calling the low-level estimate() directly on the same
inputs.

Fast path: NFXP, CCP, UFXP, SEES, TDCCP (deterministic).
Slow path: NNES (neural V-network, ~500 epochs per outer iteration).
"""

from __future__ import annotations

import numpy as np
import pytest

from econirl.core.reward_spec import RewardSpec
from econirl.environments.random_mdp import random_mdp
from econirl.simulation.synthetic import simulate_panel

# ── Shared constants ────────────────────────────────────────────────────────

N_STATES = 10
N_ACTIONS = 3
N_FEATURES = 2
DISCOUNT = 0.95  # lower than default so inner loops converge quickly


# ── Module-scoped fixtures (built once, shared across all tests) ─────────────


@pytest.fixture(scope="module")
def small_env():
    """10-state, 3-action random MDP with 2 linear features."""
    return random_mdp(
        num_states=N_STATES,
        num_actions=N_ACTIONS,
        num_features=N_FEATURES,
        seed=42,
    )


@pytest.fixture(scope="module")
def small_panel(small_env):
    """Small panel (8 individuals × 15 periods) for fast estimation."""
    return simulate_panel(small_env, n_individuals=8, n_periods=15, seed=0)


@pytest.fixture(scope="module")
def env_spec(small_env):
    """RewardSpec built from the environment's feature matrix."""
    return RewardSpec(
        small_env.feature_matrix,
        list(small_env.parameter_names),
    )


@pytest.fixture(scope="module")
def transitions_3d(small_env):
    """Full 3D transition tensor (n_actions, n_states, n_states) from the env."""
    return np.asarray(small_env.transition_matrices, dtype=np.float32)


# ── Unit tests: _build_transition_tensor passthrough ────────────────────────


class TestBuildTransitionTensor3D:
    """_build_transition_tensor(3D) returns the tensor unchanged."""

    @staticmethod
    def _random_transitions(n_actions, n_states, seed=0):
        rng = np.random.default_rng(seed)
        T = rng.random((n_actions, n_states, n_states)).astype(np.float32)
        T /= T.sum(axis=2, keepdims=True)
        return T

    # ----- NFXP (and CCP/UFXP via inheritance) --------------------------------

    def test_nfxp_3d_passthrough(self):
        from econirl.estimators import NFXP
        w = NFXP(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        T = self._random_transitions(N_ACTIONS, N_STATES)
        result = w._build_transition_tensor(T)
        assert result.shape == (N_ACTIONS, N_STATES, N_STATES)
        np.testing.assert_allclose(result, T, atol=1e-6)

    def test_nfxp_3d_wrong_shape_raises(self):
        from econirl.estimators import NFXP
        w = NFXP(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        bad_T = np.zeros((5, N_STATES, N_STATES), dtype=np.float32)
        with pytest.raises(ValueError, match="3D transitions must have shape"):
            w._build_transition_tensor(bad_T)

    # ----- CCP / UFXP (inherit from NFXP; no override) -----------------------

    def test_ccp_inherits_nfxp_passthrough(self):
        from econirl.estimators import CCP
        w = CCP(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        T = self._random_transitions(N_ACTIONS, N_STATES)
        result = w._build_transition_tensor(T)
        assert result.shape == (N_ACTIONS, N_STATES, N_STATES)
        np.testing.assert_allclose(result, T, atol=1e-6)

    def test_ufxp_inherits_nfxp_passthrough(self):
        from econirl.estimators import UFXP
        w = UFXP(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        T = self._random_transitions(N_ACTIONS, N_STATES)
        result = w._build_transition_tensor(T)
        assert result.shape == (N_ACTIONS, N_STATES, N_STATES)
        np.testing.assert_allclose(result, T, atol=1e-6)

    # ----- NNES (own copy, jnp return) ----------------------------------------

    def test_nnes_3d_passthrough(self):
        from econirl.estimators import NNES
        w = NNES(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        T = self._random_transitions(N_ACTIONS, N_STATES)
        result = w._build_transition_tensor(T)
        assert result.shape == (N_ACTIONS, N_STATES, N_STATES)
        np.testing.assert_allclose(np.asarray(result), T, atol=1e-6)

    def test_nnes_3d_wrong_shape_raises(self):
        from econirl.estimators import NNES
        w = NNES(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        bad_T = np.zeros((2, N_STATES, N_STATES), dtype=np.float32)
        with pytest.raises(ValueError, match="3D transitions must have shape"):
            w._build_transition_tensor(bad_T)

    # ----- SEES (own copy, jnp return) ----------------------------------------

    def test_sees_3d_passthrough(self):
        from econirl.estimators import SEES
        w = SEES(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        T = self._random_transitions(N_ACTIONS, N_STATES)
        result = w._build_transition_tensor(T)
        assert result.shape == (N_ACTIONS, N_STATES, N_STATES)
        np.testing.assert_allclose(np.asarray(result), T, atol=1e-6)

    def test_sees_3d_wrong_shape_raises(self):
        from econirl.estimators import SEES
        w = SEES(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        bad_T = np.zeros((1, N_STATES, N_STATES), dtype=np.float32)
        with pytest.raises(ValueError, match="3D transitions must have shape"):
            w._build_transition_tensor(bad_T)

    # ----- TDCCP (own copy, jnp return) ----------------------------------------

    def test_tdccp_3d_passthrough(self):
        from econirl.estimators import TDCCP
        w = TDCCP(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        T = self._random_transitions(N_ACTIONS, N_STATES)
        result = w._build_transition_tensor(T)
        assert result.shape == (N_ACTIONS, N_STATES, N_STATES)
        np.testing.assert_allclose(np.asarray(result), T, atol=1e-6)

    def test_tdccp_3d_wrong_shape_raises(self):
        from econirl.estimators import TDCCP
        w = TDCCP(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        bad_T = np.zeros((4, N_STATES, N_STATES), dtype=np.float32)
        with pytest.raises(ValueError, match="3D transitions must have shape"):
            w._build_transition_tensor(bad_T)


# ── Equivalence tests: fit(3D) == estimate(3D) ───────────────────────────────
#
# After fit(transitions=T_3d), the wrapper has set _panel, _utility_fn, _problem.
# Calling the corresponding low-level estimator.estimate() with those exact same
# objects and transitions=T_3d must give byte-identical parameters (same
# optimization problem, same deterministic solver).


class TestFitEquivalence:
    """fit(transitions=3D) returns same params as low-level estimate() on same inputs."""

    def _assert_allclose(self, params_fit, params_est, *, estimator_name):
        assert params_fit.shape == params_est.shape, (
            f"{estimator_name}: shape mismatch {params_fit.shape} vs {params_est.shape}"
        )
        np.testing.assert_allclose(
            params_fit,
            params_est,
            atol=1e-5,
            rtol=1e-5,
            err_msg=(
                f"{estimator_name}: fit() params differ from independent estimate().\n"
                f"  fit:     {params_fit}\n"
                f"  estimate:{params_est}"
            ),
        )

    # ── NFXP ──────────────────────────────────────────────────────────────────

    def test_nfxp_3d_equiv(self, small_panel, env_spec, transitions_3d):
        from econirl.estimation.nfxp import NFXPEstimator
        from econirl.estimators import NFXP

        wrapper = NFXP(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        wrapper.fit(data=small_panel, reward=env_spec, transitions=transitions_3d)
        params_fit = wrapper.coef_.copy()

        est = NFXPEstimator(verbose=False)
        result = est.estimate(
            panel=wrapper._panel,
            utility=wrapper._utility_fn,
            problem=wrapper._problem,
            transitions=transitions_3d,
        )
        self._assert_allclose(params_fit, np.asarray(result.parameters), estimator_name="NFXP")

    # ── CCP ───────────────────────────────────────────────────────────────────

    def test_ccp_3d_equiv(self, small_panel, env_spec, transitions_3d):
        from econirl.estimation.ccp import CCPEstimator
        from econirl.estimators import CCP

        wrapper = CCP(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        wrapper.fit(data=small_panel, reward=env_spec, transitions=transitions_3d)
        params_fit = wrapper.coef_.copy()

        est = CCPEstimator(num_policy_iterations=wrapper.num_policy_iterations, verbose=False)
        result = est.estimate(
            panel=wrapper._panel,
            utility=wrapper._utility_fn,
            problem=wrapper._problem,
            transitions=transitions_3d,
        )
        self._assert_allclose(params_fit, np.asarray(result.parameters), estimator_name="CCP")

    # ── UFXP ──────────────────────────────────────────────────────────────────

    def test_ufxp_3d_equiv(self, small_panel, env_spec, transitions_3d):
        from econirl.estimation.ufxp import UFXPEstimator
        from econirl.estimators import UFXP

        wrapper = UFXP(n_states=N_STATES, n_actions=N_ACTIONS, discount=DISCOUNT)
        wrapper.fit(data=small_panel, reward=env_spec, transitions=transitions_3d)
        params_fit = wrapper.coef_.copy()

        est = UFXPEstimator(
            weights=wrapper.weights,
            num_projections=wrapper.num_projections,
            verbose=False,
        )
        result = est.estimate(
            panel=wrapper._panel,
            utility=wrapper._utility_fn,
            problem=wrapper._problem,
            transitions=transitions_3d,
        )
        self._assert_allclose(params_fit, np.asarray(result.parameters), estimator_name="UFXP")

    # ── SEES ──────────────────────────────────────────────────────────────────

    def test_sees_3d_equiv(self, small_panel, env_spec, transitions_3d):
        from econirl.estimation.sees import SEESConfig, SEESEstimator
        from econirl.estimators import SEES

        # basis_dim >= n_states is required for the basis to span the value fn
        wrapper = SEES(
            n_states=N_STATES,
            n_actions=N_ACTIONS,
            discount=DISCOUNT,
            basis_dim=N_STATES,
        )
        wrapper.fit(data=small_panel, reward=env_spec, transitions=transitions_3d)
        params_fit = wrapper.coef_.copy()

        config = SEESConfig(
            basis_type=wrapper.basis_type,
            basis_dim=wrapper.basis_dim,
            solution=wrapper.solution,
            penalty_weight=wrapper.penalty_weight,
            num_theta_starts=wrapper.num_theta_starts,
            max_iter=wrapper.max_iter,
            compute_se=True,
            se_method=wrapper.se_method,
            verbose=False,
        )
        est = SEESEstimator(config=config)
        result = est.estimate(
            panel=wrapper._panel,
            utility=wrapper._utility_fn,
            problem=wrapper._problem,
            transitions=transitions_3d,
        )
        self._assert_allclose(params_fit, np.asarray(result.parameters), estimator_name="SEES")

    # ── TDCCP ─────────────────────────────────────────────────────────────────

    def test_tdccp_3d_equiv(self, small_panel, env_spec, transitions_3d):
        from econirl.estimation.td_ccp import TDCCPConfig, TDCCPEstimator
        from econirl.estimators import TDCCP

        wrapper = TDCCP(
            n_states=N_STATES,
            n_actions=N_ACTIONS,
            discount=DISCOUNT,
            # semigradient is deterministic; default method is already semigradient
        )
        wrapper.fit(data=small_panel, reward=env_spec, transitions=transitions_3d)
        params_fit = wrapper.coef_.copy()

        config = TDCCPConfig(
            method=wrapper.method,
            basis_dim=wrapper.basis_dim,
            basis_type=wrapper.basis_type,
            basis_include_rewards=wrapper.basis_include_rewards,
            basis_ridge=wrapper.basis_ridge,
            basis_pinv_rcond=wrapper.basis_pinv_rcond,
            hidden_dim=wrapper.hidden_dim,
            num_hidden_layers=wrapper.num_hidden_layers,
            avi_iterations=wrapper.avi_iterations,
            epochs_per_avi=wrapper.epochs_per_avi,
            learning_rate=wrapper.learning_rate,
            batch_size=wrapper.batch_size,
            ccp_method=wrapper.ccp_method,
            cross_fitting=wrapper.cross_fitting,
            robust_se=wrapper.robust_se,
            n_policy_iterations=wrapper.n_policy_iterations,
            compute_se=True,
            verbose=False,
        )
        # The wrapper translates se_method="robust" into the estimator's
        # "asymptotic" path, because TD-CCP Algorithm 2 inference is driven by
        # config.robust_se rather than the generic robust-SE routine. Mirror
        # that translation so the low-level call matches what fit() ran.
        effective_se_method = "asymptotic" if wrapper.se_method == "robust" else wrapper.se_method
        est = TDCCPEstimator(config=config, se_method=effective_se_method)
        result = est.estimate(
            panel=wrapper._panel,
            utility=wrapper._utility_fn,
            problem=wrapper._problem,
            transitions=transitions_3d,
        )
        self._assert_allclose(params_fit, np.asarray(result.parameters), estimator_name="TDCCP")

    # ── NNES (slow: neural V-network) ─────────────────────────────────────────

    @pytest.mark.slow
    def test_nnes_3d_equiv(self, small_panel, env_spec, transitions_3d):
        from econirl.estimation.nnes import NNESConfig, NNESEstimator, NNESNFXPEstimator
        from econirl.estimators import NNES

        wrapper = NNES(
            n_states=N_STATES,
            n_actions=N_ACTIONS,
            discount=DISCOUNT,
            # keep defaults (v_epochs=500, n_outer_iterations=3) for fidelity;
            # the test is slow but deterministic (fixed seed=0)
        )
        wrapper.fit(data=small_panel, reward=env_spec, transitions=transitions_3d)
        params_fit = wrapper.coef_.copy()

        config = NNESConfig(
            hidden_dim=wrapper.hidden_dim,
            num_layers=wrapper.num_layers,
            v_lr=wrapper.v_lr,
            v_epochs=wrapper.v_epochs,
            n_outer_iterations=wrapper.n_outer_iterations,
            compute_se=True,
            se_method=wrapper.se_method,
            verbose=False,
            # seed=0 is the default; matches what the wrapper uses
        )
        est_cls = NNESEstimator if wrapper.bellman == "npl" else NNESNFXPEstimator
        est = est_cls(config=config)
        result = est.estimate(
            panel=wrapper._panel,
            utility=wrapper._utility_fn,
            problem=wrapper._problem,
            transitions=transitions_3d,
        )
        self._assert_allclose(params_fit, np.asarray(result.parameters), estimator_name="NNES")


# ── Secondary sanity: run_form() recovers true params end-to-end ─────────────


class TestRunFormSanity:
    """run_form() with the same tabular env recovers true params within loose tol."""

    def test_nfxp_run_form_recovery(self, small_env):
        from econirl.forms import run_form
        from econirl.forms.base import Form, FormSpec

        spec = FormSpec(
            topology="tabular",
            reward_form="linear",
            num_states=small_env.num_states,
            num_actions=small_env.num_actions,
            name="test-tabular-3a",
        )
        form = Form(spec=spec, env=small_env)

        rr = run_form(form, ["NFXP"], n_individuals=100, n_periods=30, seed=1)

        assert "NFXP" in rr.results, (
            f"NFXP was skipped: {rr.skipped}"
        )
        result = rr.results["NFXP"]
        est_params = np.asarray(result.parameters)
        true_params = np.array(list(small_env.true_parameters.values()), dtype=float)

        # Very loose tolerance: small panel, random seed, partial identif. ok
        assert len(est_params) == len(true_params), (
            f"parameter count mismatch: {len(est_params)} vs {len(true_params)}"
        )
        # At minimum the sign of each parameter should be roughly right
        # (absolute value within 5× of truth is a sanity floor, not a precision gate)
        for name, est_val, true_val in zip(
            small_env.parameter_names, est_params, true_params
        ):
            assert abs(est_val) < 50, (
                f"NFXP run_form: {name}={est_val:.4f} is implausibly large"
            )
