"""NeuralGLADIUS: Context-aware Q-learning with Bellman consistency penalty.

Learns Q(s,a,ctx) and EV(s,a,ctx) via mini-batch training, then extracts
structural parameters by projecting implied rewards onto features.

No transition matrix is needed. Supports context conditioning through
pluggable state and context encoders.

Reference:
    Kang, M., et al. (2025). DDC IRL with neural networks.
"""

from __future__ import annotations

import warnings
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from statistics import NormalDist
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Literal, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pandas as pd

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.reward_spec import RewardSpec
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory, TrajectoryPanel
from econirl.estimation.gladius import GLADIUSConfig, GLADIUSEstimator
from econirl.estimators.neural_base import NeuralEstimatorMixin
from econirl.inference.results import FunctionalBootstrapResult
from econirl.simulation.counterfactual import CounterfactualResult, CounterfactualType


def _to_numpy(values: object) -> np.ndarray:
    return np.asarray(values)


def _to_jax_float(values: object) -> jax.Array:
    return jnp.asarray(values, dtype=jnp.float32)


def _to_jax_int(values: object) -> jax.Array:
    return jnp.asarray(values, dtype=jnp.int32)


def _scaled_column(values: object, *, maximum: float) -> jax.Array:
    """Pickle-safe default scalar encoder."""
    return (_to_jax_float(values) / maximum).reshape(-1, 1)


def _encoded_float_column(
    values: object,
    *,
    encoder: Callable[[object], object],
) -> jax.Array:
    """Pickle-safe adapter for user-provided encoders when the encoder is serializable."""
    return _to_jax_float(encoder(values))


def _normal_intervals(
    point: np.ndarray,
    standard_errors: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Return point-centered normal intervals using bootstrap standard errors."""
    critical_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    margin = critical_value * np.asarray(standard_errors, dtype=float)
    center = np.asarray(point, dtype=float)
    return np.column_stack((center - margin, center + margin))


class _MLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        n_hidden = max(num_layers, 0)
        keys = jr.split(key, n_hidden + 1)
        layers: list[eqx.nn.Linear] = []
        current_dim = in_dim
        for idx in range(n_hidden):
            layers.append(eqx.nn.Linear(current_dim, hidden_dim, key=keys[idx]))
            current_dim = hidden_dim
        self.layers = tuple(layers)
        self.output_layer = eqx.nn.Linear(current_dim, out_dim, key=keys[-1])

    def _forward_single(self, x: jax.Array) -> jax.Array:
        h = x
        for layer in self.layers:
            h = jax.nn.relu(layer(h))
        return self.output_layer(h)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.ndim == 1:
            return self._forward_single(x)
        return jax.vmap(self._forward_single)(x)

    def eval(self) -> _MLP:
        return self


class _ContextQNetwork(eqx.Module):
    n_actions: int = eqx.field(static=True)
    value_scale: float = eqx.field(static=True)
    net: _MLP

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        n_actions: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
        value_scale: float = 1.0,
    ):
        self.n_actions = n_actions
        self.value_scale = value_scale
        self.net = _MLP(
            state_dim + context_dim + n_actions,
            1,
            hidden_dim,
            num_layers,
            key=key,
        )

    def __call__(
        self,
        state_feat: object,
        ctx_feat: object,
        action_onehot: object,
    ) -> object:
        sf = _to_jax_float(state_feat)
        cf = _to_jax_float(ctx_feat)
        ao = _to_jax_float(action_onehot)
        x = jnp.concatenate([sf, cf, ao], axis=-1)
        out = jnp.squeeze(self.net(x), axis=-1)
        return out * self.value_scale

    def all_actions(
        self,
        state_feat: object,
        ctx_feat: object,
        n_actions: int,
    ) -> object:
        sf = _to_jax_float(state_feat)
        cf = _to_jax_float(ctx_feat)
        actions = jnp.eye(n_actions, dtype=jnp.float32)
        sf_exp = jnp.repeat(sf[:, None, :], n_actions, axis=1)
        cf_exp = jnp.repeat(cf[:, None, :], n_actions, axis=1)
        a_exp = jnp.repeat(actions[None, :, :], sf.shape[0], axis=0)
        x = jnp.concatenate([sf_exp, cf_exp, a_exp], axis=-1)
        out = jnp.squeeze(jax.vmap(self.net)(x), axis=-1)
        return out * self.value_scale

    def eval(self) -> _ContextQNetwork:
        return self


class NeuralGLADIUS(NeuralEstimatorMixin):
    """Context-aware GLADIUS estimator with sklearn-style API.

    ``GLADIUS`` (exported from :mod:`econirl.estimators`) is an alias for this
    class, so ``GLADIUS`` and ``NeuralGLADIUS`` are the same estimator.

    Scale of the recovered reward is identified by the anchor: set
    ``anchor_action`` and pass a per-state ``anchor_rewards`` vector for that
    action. Without ``anchor_rewards`` the estimator recovers reward direction
    but understates the magnitude.
    """

    def __init__(
        self,
        n_actions: int = 8,
        discount: float = 0.95,
        scale: float = 1.0,
        q_hidden_dim: int = 128,
        q_num_layers: int = 3,
        ev_hidden_dim: int = 128,
        ev_num_layers: int = 3,
        batch_size: int = 512,
        max_epochs: int = 500,
        lr: float = 1e-3,
        bellman_weight: float = 0.1,
        gradient_clip: float = 1.0,
        gradient_clip_mode: Literal["global_norm", "value"] = "value",
        patience: int = 50,
        alternating_updates: bool = True,
        lr_decay_rate: float = 0.001,
        tikhonov_annealing: bool = False,
        tikhonov_initial_weight: float = 100.0,
        anchor_action: int | None = None,
        anchor_rewards: Sequence[float] | None = None,
        value_scale: float | None = None,
        output_bias_init: float | None = 0.0,
        state_encoder: Callable[[object], object] | None = None,
        context_encoder: Callable[[object], object] | None = None,
        state_dim: int | None = None,
        context_dim: int = 0,
        feature_names: list[str] | None = None,
        objective: Literal["paper_minimax", "anchor_moment"] = "paper_minimax",
        network_mode: Literal["shared_trunk", "separate"] = "shared_trunk",
        compute_se: bool = False,
        n_bootstrap: int = 100,
        seed: int = 42,
        se_seed: int | None = None,
        verbose: bool = False,
        _ablate: dict | None = None,
    ):
        self.n_actions = n_actions
        self.discount = discount
        self.scale = scale
        self.q_hidden_dim = q_hidden_dim
        self.q_num_layers = q_num_layers
        self.ev_hidden_dim = ev_hidden_dim
        self.ev_num_layers = ev_num_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.bellman_weight = bellman_weight
        self.gradient_clip = gradient_clip
        if gradient_clip_mode not in {"global_norm", "value"}:
            raise ValueError("gradient_clip_mode must be 'global_norm' or 'value'")
        self.gradient_clip_mode = gradient_clip_mode
        self.patience = patience
        self.alternating_updates = alternating_updates
        self.lr_decay_rate = lr_decay_rate
        self.tikhonov_annealing = tikhonov_annealing
        self.tikhonov_initial_weight = tikhonov_initial_weight
        self.anchor_action = anchor_action
        self.anchor_rewards = anchor_rewards
        self.value_scale = value_scale
        self.output_bias_init = output_bias_init
        self.state_encoder = state_encoder
        self.context_encoder = context_encoder
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.feature_names = feature_names
        if objective not in {"paper_minimax", "anchor_moment"}:
            raise ValueError("objective must be 'paper_minimax' or 'anchor_moment'")
        if network_mode not in {"shared_trunk", "separate"}:
            raise ValueError("network_mode must be 'shared_trunk' or 'separate'")
        self.objective = objective
        self.network_mode = network_mode
        if n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2")
        self.compute_se = compute_se
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.se_seed = seed if se_seed is None else se_seed
        self.verbose = verbose
        # Research-only ablation switches (default off -> shipped behavior).
        # Keys: "class_weighting" (bool), "weight_decay" (float), "q_init_bias" (float).
        self._ablate = dict(_ablate or {})

        try:
            self.econirl_version_ = version("econirl")
        except PackageNotFoundError:
            self.econirl_version_ = "0+unknown"
        self._capability_details = {
            name: {
                "status": "supported",
                "reason": None,
                "substitute": None,
            }
            for name in (
                "inference",
                "prediction",
                "simulation",
                "counterfactual",
                "serialization",
            )
        }

        self.params_: dict[str, float] | None = None
        self.se_: dict[str, float] | None = None
        self.pvalues_: dict[str, float] | None = None
        self.coef_: np.ndarray | None = None
        self.policy_: np.ndarray | None = None
        self.value_: np.ndarray | None = None
        self.projection_r2_: float | None = None
        self.converged_: bool | None = None
        self.n_epochs_: int | None = None

        self._q_net: _ContextQNetwork | None = None
        self._ev_net: _ContextQNetwork | None = None
        self._state_encoder: Callable[[object], jax.Array] | None = None
        self._context_encoder: Callable[[object], jax.Array] | None = None
        self._state_dim: int | None = None
        self._context_dim: int | None = None
        self._n_states: int | None = None
        self._n_obs: int | None = None
        self._use_anchor: bool = False
        self._anchor_r: jax.Array | None = None
        self._paper_estimator: GLADIUSEstimator | None = None
        self._level_shift: float = 0.0
        self.q_: np.ndarray | None = None
        self.continuation_value_: np.ndarray | None = None
        self.reward_: np.ndarray | None = None
        self.objective_: str | None = None
        self.bootstrap_: FunctionalBootstrapResult | None = None
        self.transitions_: np.ndarray | None = None
        self.diagnostics_: dict[str, Any] | None = None
        self.termination_reason_: str | None = None
        self.failure_reason_: str | None = None
        self.n_observations_: int | None = None
        self.n_iter_: int | None = None
        self.fit_time_: float | None = None
        self.result_: Any | None = None
        self.is_fitted_: bool = False
        self._panel: Panel | None = None
        self._features: RewardSpec | object | None = None

    @property
    def capabilities_(self) -> MappingProxyType:
        """Read-only capability map shared by public estimator workflows."""
        nested = {
            name: MappingProxyType(details) for name, details in self._capability_details.items()
        }
        return MappingProxyType(nested)

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        *,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        context: str | object | None = None,
        features: RewardSpec | object | None = None,
        transitions: object = None,
    ) -> NeuralGLADIUS:
        fit_started = perf_counter()
        self.is_fitted_ = False
        self.converged_ = None
        self.termination_reason_ = None
        self.failure_reason_ = None
        self.n_iter_ = None
        self.fit_time_ = None
        self.result_ = None
        self.bootstrap_ = None
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required when data is a DataFrame"
                )
            missing_columns = [name for name in (state, action, id) if name not in data.columns]
            if missing_columns:
                raise ValueError(f"missing required data columns: {missing_columns}")
            if data[[state, action, id]].isna().any().any():
                raise ValueError("state, action, and id columns must not contain missing values")
            for name in (state, action):
                values = np.asarray(data[name])
                try:
                    integer_values = values.astype(np.int64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} must contain integer codes") from exc
                if not np.array_equal(values, integer_values):
                    raise ValueError(f"{name} must contain integer codes")
        if transitions is not None:
            warnings.warn(
                "GLADIUS does not use a transition matrix during fitting; the "
                "supplied transitions are stored only for post-estimation planning.",
                stacklevel=2,
            )
            transition_array = np.asarray(transitions, dtype=float)
            if transition_array.ndim != 3:
                raise ValueError("transitions must have shape (n_actions, n_states, n_states)")
            if transition_array.shape[0] != self.n_actions:
                raise ValueError("transitions action axis must match n_actions")
            if transition_array.shape[1] != transition_array.shape[2]:
                raise ValueError("transitions state axes must be square")
            if np.any(transition_array < 0) or not np.isfinite(transition_array).all():
                raise ValueError("transitions must be finite and nonnegative")
            if np.max(np.abs(transition_array.sum(axis=2) - 1.0)) > 1e-6:
                raise ValueError("transition rows must sum to one")
            self.transitions_ = transition_array
        else:
            self.transitions_ = None

        all_states, all_actions, all_next, all_contexts = self._extract_data(
            data, state, action, id, context
        )

        state_array = np.asarray(all_states, dtype=np.int64)
        action_array = np.asarray(all_actions, dtype=np.int64)
        next_array = np.asarray(all_next, dtype=np.int64)
        if not len(state_array):
            raise ValueError("GLADIUS requires at least one state-action observation")
        if np.any(state_array < 0) or np.any(next_array < 0):
            raise ValueError("state and next-state codes must be nonnegative")
        if np.any(action_array < 0) or np.any(action_array >= self.n_actions):
            raise ValueError(f"action codes must lie in [0, {self.n_actions})")

        observed_n_states = int(np.asarray(all_states).max()) + 1
        declared_state_sizes: dict[str, int] = {}
        if features is not None:
            feature_array = np.asarray(
                features.feature_matrix if isinstance(features, RewardSpec) else features
            )
            if feature_array.ndim != 3 or feature_array.shape[1] != self.n_actions:
                raise ValueError("features must have shape (n_states, n_actions, n_features)")
            declared_state_sizes["features"] = int(feature_array.shape[0])
        if self.transitions_ is not None:
            declared_state_sizes["transitions"] = int(self.transitions_.shape[1])
        unique_declared_sizes = set(declared_state_sizes.values())
        if len(unique_declared_sizes) > 1:
            detail = ", ".join(
                f"{name}={size}" for name, size in sorted(declared_state_sizes.items())
            )
            raise ValueError(f"declared state dimensions disagree: {detail}")
        if unique_declared_sizes:
            n_states = next(iter(unique_declared_sizes))
        elif self.anchor_rewards is not None:
            n_states = max(observed_n_states, len(self.anchor_rewards))
        else:
            n_states = observed_n_states
        if observed_n_states > n_states:
            raise ValueError(
                f"observed state codes require {observed_n_states} states, but "
                f"declared inputs provide {n_states}"
            )
        if int(next_array.max()) >= n_states:
            raise ValueError(f"next-state codes must lie in [0, {n_states})")
        self._n_states = n_states
        # Number of (s, a) observations in the panel, for an honest summary count.
        self._n_obs = int(np.asarray(all_states).shape[0])
        self.n_observations_ = self._n_obs
        self._features = features
        self._panel = (
            TrajectoryPanel.from_dataframe(data, state=state, action=action, id=id)
            if isinstance(data, pd.DataFrame)
            else data
        )
        self._build_encoders(all_states, all_contexts, n_states)
        self._build_anchor(n_states)
        self._build_diagnostics(all_states, all_actions, features)
        if self.compute_se and not self._use_anchor:
            raise ValueError(
                "GLADIUS reward uncertainty requires anchor_action and "
                "anchor_rewards so reward levels are identified"
            )

        if self.objective == "paper_minimax" and context is None:
            self._fit_paper_reference(data, state, action, id, features)
            if self.compute_se:
                self._fit_functional_bootstrap()
            self._finalize_common_fit(fit_started)
            return self

        # Predict in per-period utility units and multiply by value_scale, so the
        # MLP works in a well-conditioned range even at high discount factors
        # (true Q-values are order 1/(1-beta)). Mirrors the paper-API estimator.
        value_scale = (
            self.value_scale if self.value_scale is not None else 1.0 / (1.0 - self.discount)
        )

        key = jr.PRNGKey(np.random.randint(0, 2**31 - 1))
        q_key, ev_key = jr.split(key, 2)
        self._q_net = _ContextQNetwork(
            self._state_dim,
            self._context_dim,
            self.n_actions,
            self.q_hidden_dim,
            self.q_num_layers,
            key=q_key,
            value_scale=value_scale,
        )
        self._ev_net = _ContextQNetwork(
            self._state_dim,
            self._context_dim,
            self.n_actions,
            self.ev_hidden_dim,
            self.ev_num_layers,
            key=ev_key,
            value_scale=value_scale,
        )

        q_init_bias = self._ablate.get("q_init_bias")
        if q_init_bias is not None:
            # Start Q/zeta near the value level (output = mlp * value_scale), so the
            # anchor need not drag Q up from ~0. Sets the output-layer bias.
            b = float(q_init_bias) / value_scale

            def _set_bias(net):
                return eqx.tree_at(
                    lambda m: m.net.output_layer.bias,
                    net,
                    jnp.full_like(net.net.output_layer.bias, b),
                )

            self._q_net = _set_bias(self._q_net)
            self._ev_net = _set_bias(self._ev_net)

        self._train(all_states, all_actions, all_next, all_contexts)
        self._extract_policy_and_value(all_states, all_contexts, n_states)

        if features is not None:
            self._project_onto_features(features, all_states, all_actions, all_contexts)
        else:
            self.params_ = None
            self.se_ = None
            self.pvalues_ = None
            self.projection_r2_ = None
            self.coef_ = None

        self.objective_ = "anchor_moment"
        self.reward_ = self.reward_matrix_
        self.termination_reason_ = "converged" if self.converged_ else "max_epochs"
        self.failure_reason_ = None if self.converged_ else self.termination_reason_
        self.is_fitted_ = True
        if self.compute_se:
            self._fit_functional_bootstrap()
        self._finalize_common_fit(fit_started)
        return self

    def _finalize_common_fit(self, fit_started: float) -> None:
        """Populate the fitted-state fields shared by all public estimators."""
        self.n_iter_ = int(self.n_epochs_ or 0)
        self.fit_time_ = float(perf_counter() - fit_started)
        if self.result_ is None:
            self.result_ = {
                "objective": self.objective_,
                "q": self.q_,
                "continuation_value": self.continuation_value_,
                "reward": self.reward_matrix_,
                "policy": self.policy_,
                "value": self.value_,
            }

    def _fit_paper_reference(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None,
        action: str | None,
        id: str | None,
        features: RewardSpec | object | None,
    ) -> NeuralGLADIUS:
        """Fit through the shared author-reference GLADIUS implementation."""
        if isinstance(data, pd.DataFrame):
            assert state is not None and action is not None and id is not None
            panel: Panel = TrajectoryPanel.from_dataframe(
                data,
                state=state,
                action=action,
                id=id,
            )
        else:
            panel = data

        if isinstance(features, RewardSpec):
            reward_spec = features
        elif features is not None:
            feature_array = jnp.asarray(features, dtype=jnp.float32)
            names = self.feature_names or [f"f{index}" for index in range(feature_array.shape[-1])]
            reward_spec = RewardSpec(feature_array, names=names)
        else:
            placeholder = jnp.zeros(
                (int(self._n_states), self.n_actions, 1),
                dtype=jnp.float32,
            )
            reward_spec = RewardSpec(placeholder, names=["_unprojected"])

        config = GLADIUSConfig(
            q_hidden_dim=self.q_hidden_dim,
            q_num_layers=self.q_num_layers,
            v_hidden_dim=self.ev_hidden_dim,
            v_num_layers=self.ev_num_layers,
            q_lr=self.lr,
            v_lr=self.lr,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            bellman_penalty_weight=self.bellman_weight,
            gradient_clip=self.gradient_clip,
            gradient_clip_mode=self.gradient_clip_mode,
            patience=self.patience,
            alternating_updates=self.alternating_updates,
            lr_decay_rate=self.lr_decay_rate,
            tikhonov_annealing=self.tikhonov_annealing,
            tikhonov_initial_weight=self.tikhonov_initial_weight,
            anchor_action=self.anchor_action,
            anchor_rewards=(
                None
                if self.anchor_rewards is None
                else tuple(float(value) for value in self.anchor_rewards)
            ),
            anchor_bellman_mode="paper_minimax",
            value_scale=self.value_scale,
            output_bias_init=self.output_bias_init,
            network_mode=self.network_mode,
            compute_se=False,
            seed=self.seed,
            verbose=self.verbose,
        )
        problem = DDCProblem(
            num_states=int(self._n_states),
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=self.scale,
            state_dim=self._state_dim,
            state_encoder=self._state_encoder,
        )
        estimator = GLADIUSEstimator(config=config)
        try:
            result = estimator.estimate(
                panel=panel,
                utility=reward_spec,
                problem=problem,
                transitions=jnp.zeros(
                    (self.n_actions, int(self._n_states), int(self._n_states)),
                    dtype=jnp.float32,
                ),
            )
        except Exception as exc:
            self.termination_reason_ = "execution_failure"
            self.failure_reason_ = f"{type(exc).__name__}: {exc}"
            if self.diagnostics_ is not None:
                self.diagnostics_["optimization"] = {
                    "converged": False,
                    "termination_reason": self.termination_reason_,
                    "failure_reason": self.failure_reason_,
                }
            raise RuntimeError("GLADIUS estimation failed during optimization") from exc

        metadata = result.metadata
        self.result_ = result
        self._paper_estimator = estimator
        self.q_ = np.asarray(metadata["q_table"], dtype=float)
        self.continuation_value_ = np.asarray(metadata["ev_table"], dtype=float)
        self.reward_ = np.asarray(metadata["reward_table"], dtype=float)
        self.policy_ = np.asarray(result.policy, dtype=float)
        self.value_ = np.asarray(result.value_function, dtype=float)
        self.objective_ = str(metadata["anchor_bellman_mode"])
        self._level_shift = float(metadata.get("level_shift", 0.0))
        self.converged_ = bool(result.converged)
        self.n_epochs_ = int(result.num_iterations)
        self.termination_reason_ = (
            "converged" if self.converged_ else str(result.convergence_message or "not_converged")
        )
        self.failure_reason_ = None if self.converged_ else self.termination_reason_
        self.is_fitted_ = True
        if self.diagnostics_ is not None:
            loss_history = list(metadata.get("loss_history", []))
            self.diagnostics_["optimization"] = {
                "converged": self.converged_,
                "termination_reason": self.termination_reason_,
                "failure_reason": self.failure_reason_,
                "epochs": self.n_epochs_,
                "final_loss": float(loss_history[-1]) if loss_history else None,
            }
        if not self.converged_:
            warnings.warn(
                "GLADIUS reached max_epochs before its stopping rule; inspect "
                "diagnostics_ before using the fitted reward.",
                RuntimeWarning,
                stacklevel=2,
            )
        if features is None:
            self.params_ = None
            self.se_ = None
            self.pvalues_ = None
            self.projection_r2_ = None
            self.coef_ = None
        else:
            self._project_paper_reward(reward_spec)
        return self

    def _project_paper_reward(self, reward_spec: RewardSpec) -> None:
        """Populate descriptive feature-projection fields for compatibility.

        These regression diagnostics describe how closely the recovered reward
        lies in the supplied feature span.  They are not sampling uncertainty;
        :meth:`conf_int` therefore remains gated on the trajectory bootstrap.
        """
        assert self.reward_ is not None
        feature_matrix = np.asarray(reward_spec.feature_matrix, dtype=np.float32)
        reward = np.asarray(self.reward_, dtype=np.float32)
        reward_contrasts = np.concatenate(
            [reward[:, action] - reward[:, 0] for action in range(1, self.n_actions)]
        )
        feature_contrasts = np.concatenate(
            [
                feature_matrix[:, action, :] - feature_matrix[:, 0, :]
                for action in range(1, self.n_actions)
            ],
            axis=0,
        )
        theta, descriptive_se, r2 = self._project_parameters(
            feature_contrasts,
            reward_contrasts,
        )
        names = reward_spec.parameter_names
        self.params_ = {name: float(value) for name, value in zip(names, theta)}
        self.se_ = {name: float(value) for name, value in zip(names, descriptive_se)}
        self.pvalues_ = self._compute_pvalues(self.params_, self.se_)
        self.projection_r2_ = float(r2)
        self.coef_ = np.asarray(theta)

    def _extract_data(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None,
        action: str | None,
        id: str | None,
        context: str | object | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required when data is a DataFrame"
                )
            panel = TrajectoryPanel.from_dataframe(data, state=state, action=action, id=id)
            all_states = jnp.asarray(panel.all_states, dtype=jnp.int32)
            all_actions = jnp.asarray(panel.all_actions, dtype=jnp.int32)
            all_next = jnp.asarray(panel.all_next_states, dtype=jnp.int32)

            if isinstance(context, str):
                all_contexts = self._extract_context_from_df(data, id, context, panel)
            elif context is not None:
                all_contexts = _to_jax_int(context)
            else:
                all_contexts = jnp.zeros(len(all_states), dtype=jnp.int32)
        elif isinstance(data, (Panel, TrajectoryPanel)):
            all_states = jnp.asarray(data.get_all_states(), dtype=jnp.int32)
            all_actions = jnp.asarray(data.get_all_actions(), dtype=jnp.int32)
            all_next = jnp.asarray(data.get_all_next_states(), dtype=jnp.int32)
            if context is not None:
                all_contexts = _to_jax_int(context)
            else:
                all_contexts = jnp.zeros(len(all_states), dtype=jnp.int32)
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, got {type(data)}"
            )

        return all_states, all_actions, all_next, all_contexts

    def _extract_context_from_df(
        self,
        df: pd.DataFrame,
        id_col: str,
        context_col: str,
        panel: TrajectoryPanel,
    ) -> jax.Array:
        contexts: list[int] = []
        for _, group in df.groupby(id_col, sort=True):
            group = group.sort_index()
            contexts.extend(group[context_col].values.tolist())
        return jnp.asarray(contexts, dtype=jnp.int32)

    def _build_encoders(
        self,
        all_states: jax.Array,
        all_contexts: jax.Array,
        n_states: int,
    ) -> None:
        if self.state_encoder is not None:
            self._state_encoder = partial(
                _encoded_float_column,
                encoder=self.state_encoder,
            )
            self._state_dim = self.state_dim or 1
        else:
            max_s = max(n_states - 1, 1)
            self._state_encoder = partial(
                _scaled_column,
                maximum=float(max_s),
            )
            self._state_dim = 1

        if self.context_encoder is not None:
            self._context_encoder = partial(
                _encoded_float_column,
                encoder=self.context_encoder,
            )
            self._context_dim = self.context_dim or 1
        else:
            n_ctx = max(int(np.asarray(all_contexts).max()), 1) if len(all_contexts) else 1
            self._context_encoder = partial(
                _scaled_column,
                maximum=float(n_ctx),
            )
            self._context_dim = 1

    def _build_diagnostics(
        self,
        states: jax.Array,
        actions: jax.Array,
        features: RewardSpec | object | None,
    ) -> None:
        """Record coverage, identification, and feature checks before fitting."""
        state_array = np.asarray(states, dtype=np.int64)
        action_array = np.asarray(actions, dtype=np.int64)
        n_states = int(self._n_states)
        counts = np.zeros((n_states, self.n_actions), dtype=np.int64)
        np.add.at(counts, (state_array, action_array), 1)
        data_block: dict[str, Any] = {
            "n_observations": int(len(state_array)),
            "state_coverage": float(np.mean(counts.sum(axis=1) > 0)),
            "state_action_coverage": float(np.mean(counts > 0)),
            "single_action_states": int(np.sum((counts > 0).sum(axis=1) == 1)),
        }
        identification: dict[str, Any] = {
            "target": "anchored reward, Q, continuation value, and induced policy",
            "anchor_available": bool(self._use_anchor),
            "normalization": (
                f"known reward for action {self.anchor_action} in every state"
                if self._use_anchor
                else "policy only; absolute reward level is not identified"
            ),
        }
        if features is not None:
            matrix = np.asarray(
                features.feature_matrix if isinstance(features, RewardSpec) else features,
                dtype=float,
            )
            contrasts = np.concatenate(
                [matrix[:, action, :] - matrix[:, 0, :] for action in range(1, self.n_actions)],
                axis=0,
            )
            identification.update(
                {
                    "num_features": int(matrix.shape[-1]),
                    "contrast_rank": int(np.linalg.matrix_rank(contrasts)),
                    "contrast_condition_number": float(np.linalg.cond(contrasts)),
                }
            )
        self.diagnostics_ = {
            "data": data_block,
            "identification": identification,
            "optimization": None,
        }

    def _build_anchor(self, n_states: int) -> None:
        """Set up the anchor-action Bellman identification.

        When ``anchor_action`` and ``anchor_rewards`` are both supplied, the Q
        objective gains a Bellman term on the anchor action that pins the reward
        level (the paper's Assumption 3). Without ``anchor_rewards`` the
        ``anchor_action`` parameter has no effect.
        """
        self._use_anchor = False
        self._anchor_r = None
        if self.anchor_rewards is None:
            if self.anchor_action is not None:
                warnings.warn(
                    "anchor_action is set but anchor_rewards is None, so the "
                    "anchor has no effect; the recovered reward direction is "
                    "identified but its scale is not. Pass anchor_rewards (the "
                    "known reward for anchor_action in each state) to identify "
                    "the scale.",
                    stacklevel=3,
                )
            return
        if self.anchor_action is None:
            raise ValueError(
                "anchor_rewards was supplied but anchor_action is None; set "
                "anchor_action to the action index the rewards correspond to."
            )
        anchor_r = np.asarray(self.anchor_rewards, dtype=np.float32)
        if anchor_r.shape != (n_states,):
            raise ValueError(
                "anchor_rewards must contain one known reward per state; "
                f"expected shape ({n_states},), got {anchor_r.shape}."
            )
        self._anchor_r = jnp.asarray(anchor_r, dtype=jnp.float32)
        self._use_anchor = True

    def _train(
        self,
        states: jax.Array,
        actions: jax.Array,
        next_states: jax.Array,
        contexts: jax.Array,
    ) -> None:
        use_anchor = self._use_anchor
        anchor_action = self.anchor_action
        anchor_r = self._anchor_r

        def lr_schedule(step: jax.Array) -> jax.Array:
            return self.lr / (1.0 + self.lr_decay_rate * step)

        q_transforms = []
        ev_transforms = []
        if self.gradient_clip > 0:
            q_transforms.append(optax.clip_by_global_norm(self.gradient_clip))
            ev_transforms.append(optax.clip_by_global_norm(self.gradient_clip))
        _wd = float(self._ablate.get("weight_decay", 0.0))
        _core = (
            (lambda: optax.adamw(lr_schedule, weight_decay=_wd))
            if _wd > 0
            else (lambda: optax.adam(lr_schedule))
        )
        q_transforms.append(_core())
        ev_transforms.append(_core())

        q_optimizer = optax.chain(*q_transforms)
        ev_optimizer = optax.chain(*ev_transforms)
        q_net = self._q_net
        ev_net = self._ev_net
        q_opt_state = q_optimizer.init(eqx.filter(q_net, eqx.is_inexact_array))
        ev_opt_state = ev_optimizer.init(eqx.filter(ev_net, eqx.is_inexact_array))

        N = len(states)
        best_loss = float("inf")
        patience_counter = 0

        # Unweighted NLL (plain conditional MLE). Inverse-frequency class weighting
        # biases the fitted choice probabilities away from the empirical
        # frequencies, which corrupts the implied reward r = Q - beta*zeta and
        # collapses structural recovery on near-identified problems (ablation:
        # class weighting -> parameter cosine ~0.4; unweighted -> ~0.999 on
        # ss-spine). Set _ablate={"class_weighting": True} to restore the old
        # behavior for research only.
        if self._ablate.get("class_weighting", False):
            action_counts = np.bincount(np.asarray(actions), minlength=self.n_actions).astype(
                np.float32
            )
            action_counts = np.clip(action_counts, a_min=1.0, a_max=None)
            class_weights = jnp.asarray(N / (self.n_actions * action_counts), dtype=jnp.float32)
        else:
            class_weights = jnp.ones(self.n_actions, dtype=jnp.float32)

        def q_all(net: _ContextQNetwork, s_feat: jax.Array, ctx_feat: jax.Array) -> jax.Array:
            return jnp.asarray(net.all_actions(s_feat, ctx_feat, self.n_actions), dtype=jnp.float32)

        @eqx.filter_value_and_grad
        def ev_loss_fn(
            ev_model: _ContextQNetwork,
            q_model: _ContextQNetwork,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            actions_j: jax.Array,
            ns_feat: jax.Array,
        ) -> jax.Array:
            a_oh = jax.nn.one_hot(actions_j, self.n_actions, dtype=jnp.float32)
            zeta_sa = jnp.asarray(ev_model(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
            q_next_all = q_all(q_model, ns_feat, ctx_feat)
            v_next = self.scale * jax.nn.logsumexp(q_next_all / self.scale, axis=1)
            return jnp.mean((zeta_sa - jax.lax.stop_gradient(v_next)) ** 2)

        @eqx.filter_value_and_grad
        def q_nll_loss_fn(
            q_model: _ContextQNetwork,
            ev_model: _ContextQNetwork,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            actions_j: jax.Array,
            anchor_r_batch: jax.Array,
            ce_weight: float,
        ) -> jax.Array:
            qvals = q_all(q_model, s_feat, ctx_feat)
            log_probs = jax.nn.log_softmax(qvals / self.scale, axis=1)
            per_obs_nll = -log_probs[jnp.arange(actions_j.shape[0]), actions_j]
            weights = class_weights[actions_j]
            nll = jnp.mean(per_obs_nll * weights)
            loss = ce_weight * nll
            if use_anchor:
                # Anchor-action Bellman term pins the reward level (Assumption 3):
                # r_anchor = Q(s, a0) - beta * EV(s, a0) = anchor_r. EV is frozen
                # here so the level pressure lands on Q.
                a_oh = jax.nn.one_hot(actions_j, self.n_actions, dtype=jnp.float32)
                q_sa = jnp.sum(qvals * a_oh, axis=1)
                ev_sa = jax.lax.stop_gradient(
                    jnp.asarray(ev_model(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
                )
                anchor_td = anchor_r_batch + self.discount * ev_sa - q_sa
                mask = (actions_j == anchor_action).astype(jnp.float32)
                anchor_loss = jnp.sum(mask * anchor_td**2) / jnp.maximum(mask.sum(), 1.0)
                loss = loss + self.bellman_weight * anchor_loss
            return loss

        @eqx.filter_value_and_grad
        def joint_loss_fn(
            q_model: _ContextQNetwork,
            ev_model: _ContextQNetwork,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            actions_j: jax.Array,
            ns_feat: jax.Array,
            anchor_r_batch: jax.Array,
            ce_weight: float,
        ) -> jax.Array:
            qvals = q_all(q_model, s_feat, ctx_feat)
            log_probs = jax.nn.log_softmax(qvals / self.scale, axis=1)
            per_obs_nll = -log_probs[jnp.arange(actions_j.shape[0]), actions_j]
            weights = class_weights[actions_j]
            nll = jnp.mean(per_obs_nll * weights)
            a_oh = jax.nn.one_hot(actions_j, self.n_actions, dtype=jnp.float32)
            ev_sa = jnp.asarray(ev_model(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
            q_next_all = q_all(q_model, ns_feat, ctx_feat)
            v_next = self.scale * jax.nn.logsumexp(q_next_all / self.scale, axis=1)
            bellman = jnp.mean((ev_sa - jax.lax.stop_gradient(v_next)) ** 2)
            loss = ce_weight * nll + self.bellman_weight * bellman
            if use_anchor:
                q_sa = jnp.sum(qvals * a_oh, axis=1)
                anchor_td = anchor_r_batch + self.discount * ev_sa - q_sa
                mask = (actions_j == anchor_action).astype(jnp.float32)
                anchor_loss = jnp.sum(mask * anchor_td**2) / jnp.maximum(mask.sum(), 1.0)
                loss = loss + self.bellman_weight * anchor_loss
            return loss

        @eqx.filter_jit
        def ev_step(
            ev_model: _ContextQNetwork,
            ev_state: optax.OptState,
            q_model: _ContextQNetwork,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            actions_j: jax.Array,
            ns_feat: jax.Array,
        ) -> tuple[_ContextQNetwork, optax.OptState, jax.Array]:
            loss, grads = ev_loss_fn(ev_model, q_model, s_feat, ctx_feat, actions_j, ns_feat)
            updates, ev_state = ev_optimizer.update(grads, ev_state, ev_model)
            ev_model = eqx.apply_updates(ev_model, updates)
            return ev_model, ev_state, loss

        @eqx.filter_jit
        def q_step(
            q_model: _ContextQNetwork,
            q_state: optax.OptState,
            ev_model: _ContextQNetwork,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            actions_j: jax.Array,
            anchor_r_batch: jax.Array,
            ce_weight: float,
        ) -> tuple[_ContextQNetwork, optax.OptState, jax.Array]:
            loss, grads = q_nll_loss_fn(
                q_model, ev_model, s_feat, ctx_feat, actions_j, anchor_r_batch, ce_weight
            )
            updates, q_state = q_optimizer.update(grads, q_state, q_model)
            q_model = eqx.apply_updates(q_model, updates)
            return q_model, q_state, loss

        @eqx.filter_jit
        def joint_step(
            q_model: _ContextQNetwork,
            q_state: optax.OptState,
            ev_model: _ContextQNetwork,
            ev_state: optax.OptState,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            actions_j: jax.Array,
            ns_feat: jax.Array,
            anchor_r_batch: jax.Array,
            ce_weight: float,
        ) -> tuple[_ContextQNetwork, optax.OptState, _ContextQNetwork, optax.OptState, jax.Array]:
            loss, (q_grads, ev_grads) = eqx.filter_value_and_grad(joint_loss_fn, arg=(0, 1))(
                q_model, ev_model, s_feat, ctx_feat, actions_j, ns_feat, anchor_r_batch, ce_weight
            )
            q_updates, q_state = q_optimizer.update(q_grads, q_state, q_model)
            ev_updates, ev_state = ev_optimizer.update(ev_grads, ev_state, ev_model)
            q_model = eqx.apply_updates(q_model, q_updates)
            ev_model = eqx.apply_updates(ev_model, ev_updates)
            return q_model, q_state, ev_model, ev_state, loss

        best_q = q_net
        best_ev = ev_net

        for epoch in range(self.max_epochs):
            perm = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches = 0
            batch_idx = 0
            ce_weight = (
                self.tikhonov_initial_weight / (1.0 + epoch) if self.tikhonov_annealing else 1.0
            )

            for start in range(0, N, self.batch_size):
                idx = perm[start : start + self.batch_size]
                s = states[idx]
                a = actions[idx]
                ns = next_states[idx]
                ctx = contexts[idx]

                s_feat = self._state_encoder(s)
                ns_feat = self._state_encoder(ns)
                ctx_feat = self._context_encoder(ctx)
                if use_anchor:
                    anchor_r_batch = anchor_r[s]
                else:
                    anchor_r_batch = jnp.zeros(a.shape[0], dtype=jnp.float32)

                if self.alternating_updates and batch_idx % 2 == 0:
                    ev_net, ev_opt_state, loss = ev_step(
                        ev_net, ev_opt_state, q_net, s_feat, ctx_feat, a, ns_feat
                    )
                elif self.alternating_updates and batch_idx % 2 == 1:
                    q_net, q_opt_state, loss = q_step(
                        q_net, q_opt_state, ev_net, s_feat, ctx_feat, a, anchor_r_batch, ce_weight
                    )
                else:
                    q_net, q_opt_state, ev_net, ev_opt_state, loss = joint_step(
                        q_net,
                        q_opt_state,
                        ev_net,
                        ev_opt_state,
                        s_feat,
                        ctx_feat,
                        a,
                        ns_feat,
                        anchor_r_batch,
                        ce_weight,
                    )

                epoch_loss += float(loss)
                n_batches += 1
                batch_idx += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}")

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                best_q = q_net
                best_ev = ev_net
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        self._q_net = best_q
        self._ev_net = best_ev
        # Converged means early stopping fired; exhausting max_epochs is not
        # convergence.
        self.converged_ = patience_counter >= self.patience
        self.n_epochs_ = epoch + 1

    def _extract_policy_and_value(
        self,
        all_states: jax.Array,
        all_contexts: jax.Array,
        n_states: int,
    ) -> None:
        unique_states = jnp.arange(n_states, dtype=jnp.int32)
        ctx_default = jnp.zeros(n_states, dtype=jnp.int32)
        s_feat = self._state_encoder(unique_states)
        ctx_feat = self._context_encoder(ctx_default)
        qvals = jnp.asarray(
            self._q_net.all_actions(s_feat, ctx_feat, self.n_actions), dtype=jnp.float32
        )
        policy = jax.nn.softmax(qvals / self.scale, axis=1)
        value = self.scale * jax.nn.logsumexp(qvals / self.scale, axis=1)
        self.policy_ = np.asarray(policy)
        self.value_ = np.asarray(value)

    def _project_onto_features(
        self,
        features: RewardSpec | object,
        states: jax.Array,
        actions: jax.Array,
        contexts: jax.Array,
    ) -> None:
        if isinstance(features, RewardSpec):
            feat_matrix = features.feature_matrix
            names = features.parameter_names
        else:
            feat_matrix = features
            names = self.feature_names or [f"f{i}" for i in range(np.asarray(features).shape[-1])]

        n_s = self._n_states
        unique_states = jnp.arange(n_s, dtype=jnp.int32)
        unique_ctx = jnp.zeros(n_s, dtype=jnp.int32)
        s_feat = self._state_encoder(unique_states)
        ctx_feat = self._context_encoder(unique_ctx)
        q_all = jnp.asarray(
            self._q_net.all_actions(s_feat, ctx_feat, self.n_actions), dtype=jnp.float32
        )
        action_ids = jnp.arange(self.n_actions, dtype=jnp.int32)
        action_oh = jax.nn.one_hot(action_ids, self.n_actions, dtype=jnp.float32)

        def reward_for_action(a_oh_single: jax.Array) -> jax.Array:
            tiled = jnp.repeat(a_oh_single[None, :], n_s, axis=0)
            ev_a = jnp.asarray(self._ev_net(s_feat, ctx_feat, tiled), dtype=jnp.float32)
            return ev_a

        ev_all = jax.vmap(reward_for_action)(action_oh).T
        r_all = q_all - self.discount * ev_all

        feat_np = _to_numpy(feat_matrix)
        dr_list = []
        dphi_list = []
        for a_idx in range(1, self.n_actions):
            dr_list.append(np.asarray(r_all[:, a_idx] - r_all[:, 0]))
            dphi_list.append(feat_np[:n_s, a_idx, :] - feat_np[:n_s, 0, :])

        rewards = np.concatenate(dr_list, axis=0).astype(np.float32)
        phi = np.concatenate(dphi_list, axis=0).astype(np.float32)

        theta, se, r2 = self._project_parameters(phi, rewards)
        self.params_ = {n: float(v) for n, v in zip(names, theta)}
        self.se_ = {n: float(v) for n, v in zip(names, se)}
        self.pvalues_ = self._compute_pvalues(self.params_, self.se_)
        self.projection_r2_ = r2
        self.coef_ = np.asarray(theta)

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        if self.reward_ is not None:
            return np.asarray(self.reward_)
        if self._q_net is None or self._ev_net is None or self._n_states is None:
            return None
        n_s = self._n_states
        unique_states = jnp.arange(n_s, dtype=jnp.int32)
        ctx_default = jnp.zeros(n_s, dtype=jnp.int32)
        s_feat = self._state_encoder(unique_states)
        ctx_feat = self._context_encoder(ctx_default)
        q_all = jnp.asarray(
            self._q_net.all_actions(s_feat, ctx_feat, self.n_actions), dtype=jnp.float32
        )
        action_ids = jnp.arange(self.n_actions, dtype=jnp.int32)
        action_oh = jax.nn.one_hot(action_ids, self.n_actions, dtype=jnp.float32)

        def ev_for_action(a_oh_single: jax.Array) -> jax.Array:
            tiled = jnp.repeat(a_oh_single[None, :], n_s, axis=0)
            return jnp.asarray(self._ev_net(s_feat, ctx_feat, tiled), dtype=jnp.float32)

        ev_all = jax.vmap(ev_for_action)(action_oh).T
        return np.asarray(q_all - self.discount * ev_all)

    @property
    def value_function_(self) -> np.ndarray | None:
        """Compatibility alias for the common ``value_`` fitted field."""
        return self.value_

    def predict_proba(self, states: np.ndarray, context: object | None = None) -> np.ndarray:
        """Action probabilities for the given states.

        Parameters
        ----------
        states : array of state indices.
        context : optional context index or per-state context array. When None
            (default), returns the stored policy, which is computed at context 0.
            Pass a scalar to score all states at one context, or a per-state
            array to vary context across states.
        """
        if self.policy_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        states = np.asarray(states, dtype=np.int64)
        if states.ndim != 1:
            raise ValueError("states must be a one-dimensional array of integer codes")
        if self._n_states is None or np.any(states < 0) or np.any(states >= self._n_states):
            raise ValueError(f"states must lie in [0, {int(self._n_states or 0)})")
        if context is None:
            return self.policy_[states]
        if self._paper_estimator is not None:
            context_array = np.asarray(context)
            if np.any(context_array != 0):
                raise NotImplementedError(
                    "the paper-reference GLADIUS path is not context-conditioned; "
                    "pass context during fit to use the context-aware objective"
                )
            return self.policy_[states]
        states_j = _to_jax_int(states)
        contexts_j = _to_jax_int(context)
        if contexts_j.ndim == 0:
            contexts_j = jnp.broadcast_to(contexts_j, states_j.shape)
        s_feat = self._state_encoder(states_j)
        ctx_feat = self._context_encoder(contexts_j)
        qvals = jnp.asarray(
            self._q_net.all_actions(s_feat, ctx_feat, self.n_actions),
            dtype=jnp.float32,
        )
        probs = jax.nn.softmax(qvals / self.scale, axis=1)
        return np.asarray(probs)

    def predict_q_from_features(
        self,
        state_features: object,
        contexts: object | None = None,
    ) -> np.ndarray:
        """Q values for already-encoded state-feature vectors.

        ``state_features`` must be in the STATE-ENCODER space, of width
        ``self.state_dim`` (the output of the fitted state encoder), not a raw
        reward-feature vector ``phi(s, a)``. To score by state index, encode
        first or use :meth:`predict_proba` / :meth:`predict_reward`, which run
        the encoder for you.
        """
        if self._paper_estimator is not None:
            state_features_array = _to_jax_float(state_features)
            if state_features_array.ndim == 1:
                state_features_array = state_features_array[None, :]
            if state_features_array.shape[1] != self._state_dim:
                raise ValueError(
                    "state_features must be in the encoder space of width "
                    f"state_dim={self._state_dim}, got width "
                    f"{state_features_array.shape[1]}. Pass encoded features, "
                    "not the raw reward-feature matrix."
                )
            assert self._paper_estimator.q_net_ is not None
            return np.asarray(
                self._paper_estimator.q_net_.forward_all_actions(state_features_array)
                + self._level_shift
            )
        if self._q_net is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        s_feat = _to_jax_float(state_features)
        if s_feat.ndim == 1:
            s_feat = s_feat[None, :]
        if s_feat.shape[1] != self._state_dim:
            raise ValueError(
                f"state_features must be in the encoder space of width "
                f"state_dim={self._state_dim}, got width {s_feat.shape[1]}. "
                f"Pass encoded features (the output of the state encoder), not "
                f"the raw reward-feature matrix."
            )
        if contexts is None:
            contexts = jnp.zeros(s_feat.shape[0], dtype=jnp.int32)
        ctx_feat = self._context_encoder(contexts)
        qvals = self._q_net.all_actions(s_feat, ctx_feat, self.n_actions)
        return np.asarray(qvals)

    def predict_reward_from_features(
        self,
        state_features: object,
        actions: object,
        contexts: object | None = None,
    ) -> np.ndarray:
        """Reward for already-encoded state-feature vectors.

        ``state_features`` must be in the STATE-ENCODER space, of width
        ``self.state_dim`` (the output of the fitted state encoder), not a raw
        reward-feature vector ``phi(s, a)``. To score by state index, use
        :meth:`predict_reward`, which runs the encoder for you.
        """
        if self._paper_estimator is not None:
            state_features_array = _to_jax_float(state_features)
            if state_features_array.ndim == 1:
                state_features_array = state_features_array[None, :]
            if state_features_array.shape[1] != self._state_dim:
                raise ValueError(
                    "state_features must be in the encoder space of width "
                    f"state_dim={self._state_dim}, got width "
                    f"{state_features_array.shape[1]}. Pass encoded features, "
                    "or use predict_reward(states, actions)."
                )
            actions_array = _to_jax_int(actions)
            if actions_array.ndim == 0:
                actions_array = actions_array[None]
            if actions_array.shape[0] != state_features_array.shape[0]:
                raise ValueError("actions must contain one entry per state feature row")
            assert self._paper_estimator.q_net_ is not None
            assert self._paper_estimator.ev_net_ is not None
            action_onehot = jax.nn.one_hot(
                actions_array,
                self.n_actions,
                dtype=jnp.float32,
            )
            q_values = self._paper_estimator.q_net_.forward(
                state_features_array,
                action_onehot,
            )
            if self.network_mode == "shared_trunk":
                continuation_all = self._paper_estimator.ev_net_.zeta_all_actions(
                    state_features_array
                )
                continuation = continuation_all[
                    jnp.arange(actions_array.shape[0]),
                    actions_array,
                ]
            else:
                continuation = self._paper_estimator.ev_net_.forward(
                    state_features_array,
                    action_onehot,
                )
            return np.asarray(
                q_values - self.discount * continuation + (1.0 - self.discount) * self._level_shift
            )
        if self._q_net is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        s_feat = _to_jax_float(state_features)
        if s_feat.ndim == 1:
            s_feat = s_feat[None, :]
        if s_feat.shape[1] != self._state_dim:
            raise ValueError(
                f"state_features must be in the encoder space of width "
                f"state_dim={self._state_dim}, got width {s_feat.shape[1]}. "
                f"Pass encoded features, or use predict_reward(states, actions) "
                f"to score by state index."
            )
        actions_j = _to_jax_int(actions)
        if actions_j.ndim == 0:
            actions_j = actions_j[None]
        if contexts is None:
            contexts = jnp.zeros(s_feat.shape[0], dtype=jnp.int32)
        ctx_feat = self._context_encoder(contexts)
        a_oh = jax.nn.one_hot(actions_j, self.n_actions, dtype=jnp.float32)
        q_vals = jnp.asarray(self._q_net(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
        ev_vals = jnp.asarray(self._ev_net(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
        return np.asarray(q_vals - self.discount * ev_vals)

    def predict_reward(
        self,
        states: object,
        actions: object,
        contexts: object | None = None,
    ) -> object:
        if self.reward_ is not None:
            if contexts is not None and np.any(np.asarray(contexts) != 0):
                raise NotImplementedError(
                    "the paper-reference GLADIUS path is not context-conditioned; "
                    "pass context during fit to use the context-aware objective"
                )
            states_array = np.asarray(states, dtype=np.int64)
            actions_array = np.asarray(actions, dtype=np.int64)
            return np.asarray(self.reward_)[states_array, actions_array]
        if self._q_net is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        states_j = _to_jax_int(states)
        actions_j = _to_jax_int(actions)
        if contexts is None:
            contexts_j = jnp.zeros(states_j.shape[0], dtype=jnp.int32)
        else:
            contexts_j = _to_jax_int(contexts)
        s_feat = self._state_encoder(states_j)
        ctx_feat = self._context_encoder(contexts_j)
        a_oh = jax.nn.one_hot(actions_j, self.n_actions, dtype=jnp.float32)
        q_vals = jnp.asarray(self._q_net(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
        ev_vals = jnp.asarray(self._ev_net(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
        rewards = q_vals - self.discount * ev_vals
        return rewards

    def _fit_functional_bootstrap(self) -> None:
        """Refit whole sampled trajectories and retain reward/policy draws."""
        if self._panel is None:
            raise RuntimeError("bootstrap requires a fitted trajectory panel")
        rng = np.random.default_rng(self.se_seed)
        trajectories = list(self._panel.trajectories)
        reward_draws: list[np.ndarray] = []
        policy_draws: list[np.ndarray] = []
        failures: list[str] = []
        for draw in range(self.n_bootstrap):
            indices = rng.integers(0, len(trajectories), size=len(trajectories))
            sampled = Panel(trajectories=[trajectories[int(index)] for index in indices])
            clone = NeuralGLADIUS(
                n_actions=self.n_actions,
                discount=self.discount,
                scale=self.scale,
                q_hidden_dim=self.q_hidden_dim,
                q_num_layers=self.q_num_layers,
                ev_hidden_dim=self.ev_hidden_dim,
                ev_num_layers=self.ev_num_layers,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                lr=self.lr,
                bellman_weight=self.bellman_weight,
                gradient_clip=self.gradient_clip,
                gradient_clip_mode=self.gradient_clip_mode,
                patience=self.patience,
                alternating_updates=self.alternating_updates,
                lr_decay_rate=self.lr_decay_rate,
                tikhonov_annealing=self.tikhonov_annealing,
                tikhonov_initial_weight=self.tikhonov_initial_weight,
                anchor_action=self.anchor_action,
                anchor_rewards=self.anchor_rewards,
                value_scale=self.value_scale,
                output_bias_init=self.output_bias_init,
                state_encoder=self.state_encoder,
                state_dim=self.state_dim,
                feature_names=self.feature_names,
                objective=self.objective,
                network_mode=self.network_mode,
                compute_se=False,
                seed=self.seed + draw + 1,
                verbose=False,
            )
            try:
                clone.fit(sampled, features=self._features)
                if not clone.converged_:
                    raise RuntimeError(f"nonconverged bootstrap refit: {clone.termination_reason_}")
                reward = np.asarray(clone.reward_matrix_, dtype=float)
                policy = np.asarray(clone.policy_, dtype=float)
                if not np.isfinite(reward).all() or not np.isfinite(policy).all():
                    raise RuntimeError("nonfinite fitted functionals")
                reward_draws.append(reward)
                policy_draws.append(policy)
            except Exception as exc:
                failures.append(f"draw {draw}: {type(exc).__name__}: {exc}")

        if len(reward_draws) < 2:
            raise RuntimeError(
                "GLADIUS trajectory bootstrap produced fewer than two successful draws"
            )
        rewards = np.stack(reward_draws)
        policies = np.stack(policy_draws)
        draws = np.concatenate(
            [rewards.reshape(len(rewards), -1), policies.reshape(len(policies), -1)],
            axis=1,
        )
        point = np.concatenate(
            [
                np.asarray(self.reward_, dtype=float).reshape(-1),
                np.asarray(self.policy_, dtype=float).reshape(-1),
            ]
        )
        standard_errors = draws.std(axis=0, ddof=1)
        names = tuple(
            [
                f"reward[{state},{action}]"
                for state in range(int(self._n_states))
                for action in range(self.n_actions)
            ]
            + [
                f"policy[{state},{action}]"
                for state in range(int(self._n_states))
                for action in range(self.n_actions)
            ]
        )
        self.bootstrap_ = FunctionalBootstrapResult(
            method="pairs_cluster_normal",
            unit="individual_trajectory",
            n_requested=self.n_bootstrap,
            n_successful=len(rewards),
            seed=self.se_seed,
            estimand_names=names,
            estimates=draws,
            standard_errors=standard_errors,
            intervals=_normal_intervals(point, standard_errors, alpha=0.05),
            reward_draws=rewards,
            policy_draws=policies,
            failures=tuple(failures),
        )

    def simulate(
        self,
        n_trajectories: int,
        *,
        n_periods: int | None = None,
        seed: int | None = None,
    ) -> TrajectoryPanel:
        """Simulate fitted-policy trajectories through stored planning dynamics."""
        if not self.is_fitted_ or self.policy_ is None or self._panel is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self.transitions_ is None:
            raise NotImplementedError(
                "GLADIUS simulation requires a transition tensor supplied during fit; "
                "GLADIUS estimates behavior without transitions, so supply validated "
                "planning dynamics or use predict_proba() for behavioral predictions"
            )
        if n_trajectories < 1:
            raise ValueError("n_trajectories must be positive")
        if n_periods is None:
            n_periods = max(len(trajectory.states) for trajectory in self._panel.trajectories)
        if n_periods < 1:
            raise ValueError("n_periods must be positive")

        rng = np.random.default_rng(seed)
        initial_states = np.asarray(
            [int(trajectory.states[0]) for trajectory in self._panel.trajectories],
            dtype=np.int64,
        )
        transitions = np.asarray(self.transitions_, dtype=float)
        trajectories = []
        for individual in range(n_trajectories):
            state = int(rng.choice(initial_states))
            states = []
            actions = []
            next_states = []
            for _ in range(n_periods):
                action = int(rng.choice(self.n_actions, p=self.policy_[state]))
                next_state = int(rng.choice(int(self._n_states), p=transitions[action, state]))
                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                state = next_state
            trajectories.append(
                Trajectory(
                    states=jnp.asarray(states, dtype=jnp.int32),
                    actions=jnp.asarray(actions, dtype=jnp.int32),
                    next_states=jnp.asarray(next_states, dtype=jnp.int32),
                    individual_id=individual,
                )
            )
        return TrajectoryPanel(trajectories=trajectories)

    def _solve_reward_system(
        self,
        reward: np.ndarray,
        transitions: np.ndarray,
    ) -> Any:
        problem = DDCProblem(
            num_states=reward.shape[0],
            num_actions=reward.shape[1],
            discount_factor=self.discount,
            scale_parameter=self.scale,
        )
        result = value_iteration(
            SoftBellmanOperator(problem, jnp.asarray(transitions)),
            jnp.asarray(reward),
            tol=1e-9,
            max_iter=10_000,
        )
        if not result.converged:
            raise RuntimeError("GLADIUS counterfactual Bellman solve did not converge")
        return result

    def counterfactual(
        self,
        *,
        reward_delta: np.ndarray | None = None,
        transitions: np.ndarray | None = None,
        description: str | None = None,
    ) -> CounterfactualResult:
        """Re-solve an anchored reward or supplied-transition intervention."""
        if not self.is_fitted_ or self.reward_matrix_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if not self._use_anchor:
            raise NotImplementedError(
                "GLADIUS structural counterfactuals are unsupported without "
                "anchor_action and anchor_rewards because reward levels are not "
                "identified; use predict_proba() for behavioral comparisons"
            )
        if self.transitions_ is None:
            raise NotImplementedError(
                "GLADIUS counterfactual planning requires a baseline transition "
                "tensor supplied during fit; the estimator does not infer dynamics, "
                "so provide validated planning transitions or report policy predictions"
            )
        if (reward_delta is None) == (transitions is None):
            raise ValueError("supply exactly one of reward_delta or transitions")

        baseline_reward = np.asarray(self.reward_matrix_, dtype=float)
        baseline_transitions = np.asarray(self.transitions_, dtype=float)
        changed_reward = baseline_reward.copy()
        changed_transitions = baseline_transitions.copy()
        counterfactual_type = CounterfactualType.ENVIRONMENT_CHANGE
        changed_primitive = "transitions"
        if reward_delta is not None:
            delta = np.asarray(reward_delta, dtype=float)
            if delta.shape != baseline_reward.shape:
                raise ValueError(f"reward_delta must have shape {baseline_reward.shape}")
            if not np.isfinite(delta).all():
                raise ValueError("reward_delta must be finite")
            changed_reward += delta
            counterfactual_type = CounterfactualType.REWARD_CHANGE
            changed_primitive = "reward"
        else:
            candidate = np.asarray(transitions, dtype=float)
            if candidate.shape != baseline_transitions.shape:
                raise ValueError(f"transitions must have shape {baseline_transitions.shape}")
            if np.any(candidate < 0) or not np.isfinite(candidate).all():
                raise ValueError("transitions must be finite and nonnegative")
            if np.max(np.abs(candidate.sum(axis=2) - 1.0)) > 1e-6:
                raise ValueError("transition rows must sum to one")
            changed_transitions = candidate

        baseline = self._solve_reward_system(baseline_reward, baseline_transitions)
        changed = self._solve_reward_system(changed_reward, changed_transitions)
        value_change = changed.V - baseline.V
        return CounterfactualResult(
            baseline_policy=baseline.policy,
            counterfactual_policy=changed.policy,
            baseline_value=baseline.V,
            counterfactual_value=changed.V,
            policy_change=changed.policy - baseline.policy,
            value_change=value_change,
            welfare_change=float(jnp.mean(value_change)),
            counterfactual_type=counterfactual_type,
            description=description or f"GLADIUS {changed_primitive} counterfactual",
            metadata={
                "estimator": "GLADIUS",
                "changed_primitive": changed_primitive,
                "objective": self.objective_,
                "anchor_action": self.anchor_action,
            },
            transitions=jnp.asarray(baseline_transitions),
            counterfactual_transitions=jnp.asarray(changed_transitions),
            params=dict(self.params_ or {}),
        )

    def conf_int(self, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be finite and lie strictly between 0 and 1")
        if self.bootstrap_ is None:
            raise NotImplementedError(
                "GLADIUS sampling intervals require compute_se=True; descriptive "
                "projection standard errors are not sampling uncertainty"
            )
        point = np.concatenate(
            [
                np.asarray(self.reward_, dtype=float).reshape(-1),
                np.asarray(self.policy_, dtype=float).reshape(-1),
            ],
        )
        intervals = _normal_intervals(
            point,
            self.bootstrap_.standard_errors,
            alpha=alpha,
        )
        return {
            name: (float(lower), float(upper))
            for name, (lower, upper) in zip(self.bootstrap_.estimand_names, intervals)
        }

    def summary(self) -> str:
        if self.policy_ is None:
            return "NeuralGLADIUS: Not fitted yet. Call fit() first."
        diagnostics = self.diagnostics_ or {}
        data_checks = diagnostics.get("data", {})
        identification = diagnostics.get("identification", {})
        outcome = (
            "No feature projection; reward and policy functionals are available."
            if self.params_ is None
            else ", ".join(f"{name}={value:.4f}" for name, value in self.params_.items())
        )
        uncertainty = (
            "Whole-trajectory bootstrap: "
            f"{self.bootstrap_.n_successful}/{self.bootstrap_.n_requested} successful draws."
            if self.bootstrap_ is not None
            else "Not requested. Set compute_se=True for whole-trajectory bootstrap intervals."
        )
        projection = (
            "None"
            if self.projection_r2_ is None
            else f"R2={self.projection_r2_:.4f}; projection SEs are descriptive, not sampling SEs"
        )
        return "\n".join(
            [
                "Estimator",
                "NeuralGLADIUS (public GLADIUS paper-reference path)",
                "",
                "Data",
                f"Observations:    {int(self.n_observations_ or 0)}",
                f"States declared: {int(self._n_states or 0)}",
                "",
                "Model",
                f"Objective: {self.objective_ or self.objective}",
                f"Network mode: {self.network_mode}; discount={self.discount}; scale={self.scale}",
                "",
                "Pre-estimation checks",
                (
                    f"State coverage={data_checks.get('state_coverage', float('nan')):.3f}; "
                    f"state-action coverage="
                    f"{data_checks.get('state_action_coverage', float('nan')):.3f}"
                ),
                (
                    f"Anchor available={identification.get('anchor_available', False)}; "
                    f"contrast rank={identification.get('contrast_rank', 'not supplied')}"
                ),
                "",
                "Fit",
                f"Converged: {self.converged_}; stopping reason: {self.termination_reason_}",
                (
                    f"Iterations: {int(self.n_iter_ or 0)}; "
                    f"fit time: {float(self.fit_time_ or 0.0):.3f}s"
                ),
                "",
                "Outcome",
                f"Projected action contrasts: {outcome}",
                f"Projection diagnostic: {projection}",
                "",
                "Uncertainty",
                uncertainty,
                "",
                "Limitations",
                (
                    "Structural reward levels and counterfactual welfare require a credible "
                    "known-reward anchor in every declared state."
                ),
                (
                    "Simulation and counterfactual planning require a validated transition "
                    "tensor supplied during fit; transitions are not used for estimation."
                ),
                "The public fit never uses the simulation-only oracle epoch selector.",
            ]
        )

    def __repr__(self) -> str:
        fitted = self.policy_ is not None
        return (
            f"NeuralGLADIUS(n_actions={self.n_actions}, discount={self.discount}, fitted={fitted})"
        )
