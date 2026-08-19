"""Standalone neural AIRL for finite state dynamic choice problems.

The estimator keeps AIRL's state-only identification boundary while replacing
the linear reward, shaping potential, and generator policy with neural
function approximators.  The public tabular :class:`econirl.AIRL` estimator is
not used as an implementation alias and remains unchanged.
"""

from __future__ import annotations

import gc
import time
import warnings
from statistics import NormalDist
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.reward_spec import RewardSpec
from econirl.core.solvers import hybrid_iteration, value_iteration
from econirl.core.transition_models import DeterministicTransitions
from econirl.core.types import DDCProblem, Panel, TrajectoryPanel
from econirl.estimators.airl import AIRL
from econirl.inference.results import FunctionalBootstrapResult
from econirl.simulation.counterfactual import CounterfactualResult, CounterfactualType


class _MLP(eqx.Module):
    """Small pickle-safe MLP used by the three AIRL function classes."""

    hidden: tuple[eqx.nn.Linear, ...]
    output: eqx.nn.Linear

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ) -> None:
        keys = jr.split(key, num_layers + 1)
        layers: list[eqx.nn.Linear] = []
        width = in_dim
        for index in range(num_layers):
            layers.append(eqx.nn.Linear(width, hidden_dim, key=keys[index]))
            width = hidden_dim
        self.hidden = tuple(layers)
        self.output = eqx.nn.Linear(width, out_dim, key=keys[-1])

    def _one(self, x: jax.Array) -> jax.Array:
        value = x
        for layer in self.hidden:
            value = jax.nn.tanh(layer(value))
        return self.output(value)

    def __call__(self, x: jax.Array) -> jax.Array:
        values = jnp.asarray(x, dtype=jnp.float32)
        if values.ndim == 1:
            return self._one(values)
        return jax.vmap(self._one)(values)


class _Discriminator(eqx.Module):
    reward: _MLP
    shaping: _MLP


class NeuralAIRL(AIRL):
    """Neural AIRL with state-only rewards and no heterogeneity.

    The reward network maps supplied state inputs to ``g(s)``.  A second
    network learns the potential ``h(s)`` in AIRL's discriminator, and a policy
    network distils the soft-optimal generator policy after every adversarial
    update.  The fitted reward is normalized to mean zero because its level is
    not identified.

    Parameters are the public AIRL parameters plus the neural widths, depth,
    policy learning rate, and mini-batch controls below.  ``feature_matrix`` is
    interpreted as neural state input, not as a linear reward basis.
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.99,
        feature_matrix: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        *,
        reward_hidden_dim: int = 64,
        reward_num_layers: int = 2,
        shaping_hidden_dim: int = 64,
        shaping_num_layers: int = 2,
        policy_hidden_dim: int = 64,
        policy_num_layers: int = 2,
        reward_lr: float = 1e-2,
        policy_lr: float = 2e-3,
        discriminator_steps: int = 5,
        policy_steps: int = 20,
        batch_size: int = 512,
        generator_tol: float = 1e-8,
        generator_max_iter: int = 5_000,
        policy_step_size: float = 0.25,
        max_rounds: int = 200,
        min_rounds: int = 50,
        convergence_tol: float = 1e-3,
        shaping_l2_penalty: float = 1e-6,
        compute_se: bool = False,
        n_bootstrap: int = 100,
        seed: int = 42,
        se_seed: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        # Accept the two AIRL solver knobs used by shared construction paths.
        generator_solver = kwargs.pop("generator_solver", "hybrid")
        kwargs.pop("generator_reward", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected NeuralAIRL arguments: {unknown}")
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            discount=discount,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            reward_lr=reward_lr,
            discriminator_steps=discriminator_steps,
            generator_solver=generator_solver,
            generator_tol=generator_tol,
            generator_max_iter=generator_max_iter,
            policy_step_size=policy_step_size,
            max_rounds=max_rounds,
            min_rounds=min_rounds,
            convergence_tol=convergence_tol,
            shaping_l2_penalty=shaping_l2_penalty,
            compute_se=compute_se,
            n_bootstrap=n_bootstrap,
            seed=seed,
            se_seed=se_seed,
            verbose=verbose,
        )
        for label, value in (
            ("reward_hidden_dim", reward_hidden_dim),
            ("shaping_hidden_dim", shaping_hidden_dim),
            ("policy_hidden_dim", policy_hidden_dim),
            ("batch_size", batch_size),
            ("policy_steps", policy_steps),
        ):
            if value < 1:
                raise ValueError(f"{label} must be positive")
        for label, value in (
            ("reward_num_layers", reward_num_layers),
            ("shaping_num_layers", shaping_num_layers),
            ("policy_num_layers", policy_num_layers),
        ):
            if value < 0:
                raise ValueError(f"{label} must be nonnegative")
        if not np.isfinite(policy_lr) or policy_lr <= 0:
            raise ValueError("policy_lr must be finite and positive")

        self.reward_hidden_dim = reward_hidden_dim
        self.reward_num_layers = reward_num_layers
        self.shaping_hidden_dim = shaping_hidden_dim
        self.shaping_num_layers = shaping_num_layers
        self.policy_hidden_dim = policy_hidden_dim
        self.policy_num_layers = policy_num_layers
        self.policy_lr = policy_lr
        self.policy_steps = policy_steps
        self.batch_size = batch_size

        self._discriminator: _Discriminator | None = None
        self._policy_network: _MLP | None = None
        self.state_inputs_: np.ndarray | None = None
        self.input_mean_: np.ndarray | None = None
        self.input_scale_: np.ndarray | None = None
        self.shaping_: np.ndarray | None = None
        self.policy_network_: np.ndarray | None = None
        self.reward_scale_: float | None = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel | None,
        *,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        next_state: str | None = None,
        transitions: np.ndarray | DeterministicTransitions | None = None,
        reward: RewardSpec | None = None,
        features: RewardSpec | np.ndarray | None = None,
        context: Any | None = None,
        tasks: Any | None = None,
        task: str | None = None,
    ) -> "NeuralAIRL":
        """Fit the three neural AIRL functions to complete trajectories."""
        if context is not None:
            raise NotImplementedError(
                "NeuralAIRL v1 does not accept context. Use AIRL2 for heterogeneity."
            )
        if tasks is not None or task is not None:
            raise NotImplementedError("NeuralAIRL does not support task-specific fits")
        if reward is not None and features is not None:
            raise ValueError("supply at most one of reward or features")
        if data is None:
            raise TypeError("data must be a DataFrame, Panel, or TrajectoryPanel")
        if transitions is None:
            raise ValueError(
                "NeuralAIRL requires transitions with shape (n_actions, n_states, n_states)"
            )
        if isinstance(transitions, DeterministicTransitions):
            raise ValueError("NeuralAIRL requires a dense transition tensor")

        started = time.perf_counter()
        self._reset_fit_state()
        self.bootstrap_ = None
        panel = self._coerce_panel(data, state, action, id, next_state)
        tensor = self._validate_dense_transitions(np.asarray(transitions, dtype=np.float64))
        state_inputs = self._resolve_state_inputs(reward if reward is not None else features)
        self._validate_training_support(panel, state_inputs)

        self._panel = cast(Any, panel)
        self.transitions_ = tensor.copy()
        self.transition_tensor_ = tensor.copy()
        self.transition_source_ = "supplied action-specific tensor"
        self.state_inputs_ = state_inputs
        mean = state_inputs.mean(axis=0)
        scale = state_inputs.std(axis=0)
        self.input_mean_ = mean
        self.input_scale_ = np.where(scale > 1e-8, scale, 1.0)
        normalized = (state_inputs - self.input_mean_) / self.input_scale_

        states = np.asarray(panel.get_all_states(), dtype=np.int32)
        actions = np.asarray(panel.get_all_actions(), dtype=np.int32)
        next_states = np.asarray(panel.get_all_next_states(), dtype=np.int32)
        initial_states = np.asarray([int(t.states[0]) for t in panel.trajectories])
        lengths = np.asarray([len(t) for t in panel.trajectories], dtype=np.int32)

        training = self._train_networks(
            normalized,
            tensor,
            states,
            actions,
            next_states,
            initial_states,
            lengths,
        )
        self._discriminator = training["discriminator"]
        self._policy_network = training["policy_network"]
        reward_values = np.array(training["reward"], dtype=np.float64, copy=True)
        reward_values = reward_values - reward_values.mean()
        self.reward_scale_ = self._calibrate_reward_scale(
            reward_values,
            tensor,
            states,
            actions,
        )
        reward_values = self.reward_scale_ * reward_values
        solution = self._solve(reward_values, tensor)

        self.reward_ = reward_values
        self.shaping_ = np.asarray(training["shaping"], dtype=np.float64)
        self.policy_ = np.asarray(solution.policy, dtype=np.float64)
        self.policy_network_ = np.asarray(training["policy"], dtype=np.float64)
        self.value_ = np.asarray(solution.V, dtype=np.float64)
        self.value_function_ = self.value_
        self.params_ = {}
        self.coef_ = np.empty(0, dtype=np.float64)
        self.se_ = None
        self.pvalues_ = None
        self.log_likelihood_ = float(
            np.log(np.clip(self.policy_[states, actions], 1e-12, 1.0)).sum()
        )
        self.converged_ = bool(training["converged"])
        self.termination_reason_ = "converged" if self.converged_ else "maximum_adversarial_rounds"
        self.failure_reason_ = None if self.converged_ else self.termination_reason_
        self.n_iter_ = int(training["rounds"])
        self.n_observations_ = panel.num_observations
        self.fit_time_ = time.perf_counter() - started
        self.is_fitted_ = True
        self.result_ = cast(
            Any,
            {
                "reward": self.reward_.copy(),
                "policy": self.policy_.copy(),
                "shaping": self.shaping_.copy(),
            },
        )
        self.diagnostics_ = {
            "data": {
                "n_observations": panel.num_observations,
                "n_individuals": panel.num_individuals,
                "state_coverage": float(np.unique(states).size / self.n_states),
            },
            "identification": {
                "target": "state-only nonlinear reward up to an additive constant",
                "normalization": "reward centered over states; logit shock scale fixed at 1.0",
                "input_rank": int(np.linalg.matrix_rank(state_inputs)),
                "num_inputs": int(state_inputs.shape[1]),
                "neural_weights_interpretable": False,
                "reward_scale_calibration": "profile choice likelihood",
            },
            "optimization": {
                "converged": self.converged_,
                "termination_reason": self.termination_reason_,
                "rounds": self.n_iter_,
                "final_discriminator_loss": float(training["disc_loss"]),
                "policy_distillation_tv": float(
                    0.5 * np.abs(self.policy_network_ - self.policy_).sum(axis=1).mean()
                ),
                "reward_scale": self.reward_scale_,
            },
        }
        self._run_fit_self_check(states, actions)
        if self.compute_se:
            self._fit_functional_bootstrap()
        if not self.converged_:
            warnings.warn(
                "NeuralAIRL reached max_rounds before its policy-change stopping rule; "
                "inspect diagnostics_ before using the fitted reward.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    def _coerce_panel(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None,
        action: str | None,
        id: str | None,
        next_state: str | None,
    ) -> Panel | TrajectoryPanel:
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None or next_state is None:
                raise ValueError(
                    "state, action, id, and next_state column names are required "
                    "for NeuralAIRL DataFrame input"
                )
            self._validate_dataframe(data, state, action, id, next_state, None)
            return self._dataframe_to_panel(data, state, action, id, next_state, None)
        if isinstance(data, (Panel, TrajectoryPanel)):
            return data
        raise TypeError(f"data must be a DataFrame, Panel, or TrajectoryPanel, got {type(data)}")

    def _validate_dense_transitions(self, transitions: np.ndarray) -> np.ndarray:
        expected = (self.n_actions, self.n_states, self.n_states)
        if transitions.shape != expected:
            raise ValueError(f"transitions must have shape {expected}")
        if not np.isfinite(transitions).all() or np.any(transitions < 0):
            raise ValueError("transitions must be finite and nonnegative")
        if np.max(np.abs(transitions.sum(axis=2) - 1.0)) > 1e-6:
            raise ValueError("transition rows must sum to one")
        return transitions

    def _resolve_state_inputs(
        self,
        supplied: RewardSpec | np.ndarray | None,
    ) -> np.ndarray:
        candidate: Any = supplied if supplied is not None else self.feature_matrix
        if isinstance(candidate, RewardSpec):
            if not candidate.is_state_only:
                raise ValueError(
                    "NeuralAIRL requires state-only inputs. Use AIRL2 for "
                    "action-dependent or heterogeneous rewards."
                )
            candidate = np.asarray(candidate.feature_matrix)[:, 0, :]
        if candidate is None:
            candidate = np.linspace(-1.0, 1.0, self.n_states)[:, None]
        values = np.asarray(candidate, dtype=np.float64)
        if values.ndim == 3:
            if values.shape[:2] != (self.n_states, self.n_actions):
                raise ValueError("3D features must have shape (n_states, n_actions, n_inputs)")
            if not np.allclose(values, values[:, :1, :]):
                raise ValueError(
                    "NeuralAIRL requires action-invariant state inputs. Use AIRL2 for "
                    "action-dependent rewards."
                )
            values = values[:, 0, :]
        if values.ndim != 2 or values.shape[0] != self.n_states:
            raise ValueError("state inputs must have shape (n_states, n_inputs)")
        if not np.isfinite(values).all():
            raise ValueError("state inputs must be finite")
        if np.linalg.matrix_rank(values) < values.shape[1]:
            raise ValueError("state input matrix is rank deficient")
        return values

    def _validate_training_support(self, panel: Panel, inputs: np.ndarray) -> None:
        del inputs
        states = np.asarray(panel.get_all_states(), dtype=np.int64)
        actions = np.asarray(panel.get_all_actions(), dtype=np.int64)
        next_states = np.asarray(panel.get_all_next_states(), dtype=np.int64)
        if states.size == 0:
            raise ValueError("NeuralAIRL requires at least one observed transition")
        if np.any(states < 0) or np.any(states >= self.n_states):
            raise ValueError(f"states must lie in [0, {self.n_states})")
        if np.any(next_states < 0) or np.any(next_states >= self.n_states):
            raise ValueError(f"next states must lie in [0, {self.n_states})")
        if np.any(actions < 0) or np.any(actions >= self.n_actions):
            raise ValueError(f"actions must lie in [0, {self.n_actions})")
        missing = sorted(set(range(self.n_states)) - set(states.tolist()))
        if missing:
            raise ValueError(
                f"NeuralAIRL requires observed state coverage; missing states {missing}"
            )

    def _train_networks(
        self,
        inputs: np.ndarray,
        transitions: np.ndarray,
        expert_states: np.ndarray,
        expert_actions: np.ndarray,
        expert_next: np.ndarray,
        initial_states: np.ndarray,
        lengths: np.ndarray,
    ) -> dict[str, Any]:
        key = jr.PRNGKey(self.seed)
        reward_key, shaping_key, policy_key = jr.split(key, 3)
        discriminator = _Discriminator(
            reward=_MLP(
                inputs.shape[1],
                1,
                self.reward_hidden_dim,
                self.reward_num_layers,
                key=reward_key,
            ),
            shaping=_MLP(
                inputs.shape[1],
                1,
                self.shaping_hidden_dim,
                self.shaping_num_layers,
                key=shaping_key,
            ),
        )
        policy_network = _MLP(
            inputs.shape[1],
            self.n_actions,
            self.policy_hidden_dim,
            self.policy_num_layers,
            key=policy_key,
        )
        disc_optimizer = optax.chain(
            optax.clip_by_global_norm(5.0),
            optax.adam(self.reward_lr),
        )
        policy_optimizer = optax.chain(
            optax.clip_by_global_norm(5.0),
            optax.adam(self.policy_lr),
        )
        disc_state = disc_optimizer.init(eqx.filter(discriminator, eqx.is_inexact_array))
        policy_state = policy_optimizer.init(eqx.filter(policy_network, eqx.is_inexact_array))
        input_jax = jnp.asarray(inputs, dtype=jnp.float32)

        @eqx.filter_value_and_grad
        def disc_loss(
            model: _Discriminator,
            es: jax.Array,
            ea: jax.Array,
            ens: jax.Array,
            gs: jax.Array,
            ga: jax.Array,
            gns: jax.Array,
            log_policy: jax.Array,
        ) -> jax.Array:
            reward = jnp.squeeze(model.reward(input_jax), axis=-1)
            shaping = jnp.squeeze(model.shaping(input_jax), axis=-1)
            expert_f = reward[es] + self.discount * shaping[ens] - shaping[es]
            generated_f = reward[gs] + self.discount * shaping[gns] - shaping[gs]
            expert_logits = expert_f - log_policy[es, ea]
            generated_logits = generated_f - log_policy[gs, ga]
            classification = jax.nn.softplus(-expert_logits).mean()
            classification += jax.nn.softplus(generated_logits).mean()
            penalty = self.shaping_l2_penalty * (jnp.mean(reward**2) + jnp.mean(shaping**2))
            return classification + penalty

        @eqx.filter_value_and_grad
        def policy_loss(model: _MLP, target: jax.Array) -> jax.Array:
            logits = model(input_jax)
            return cast(jax.Array, optax.softmax_cross_entropy(logits, target).mean())

        rng = np.random.default_rng(self.seed)
        policy = np.full((self.n_states, self.n_actions), 1.0 / self.n_actions)
        last_loss = float("inf")
        converged = False
        rounds = self.max_rounds
        for round_index in range(self.max_rounds):
            generated = self._simulate_generator(
                policy,
                transitions,
                initial_states,
                lengths,
                rng,
            )
            log_policy = np.log(np.clip(policy, 1e-8, 1.0))
            for _ in range(self.discriminator_steps):
                expert_index = rng.integers(
                    0,
                    expert_states.size,
                    size=min(self.batch_size, expert_states.size),
                )
                generated_index = rng.integers(
                    0,
                    generated[0].size,
                    size=min(self.batch_size, generated[0].size),
                )
                loss, gradients = disc_loss(
                    discriminator,
                    jnp.asarray(expert_states[expert_index]),
                    jnp.asarray(expert_actions[expert_index]),
                    jnp.asarray(expert_next[expert_index]),
                    jnp.asarray(generated[0][generated_index]),
                    jnp.asarray(generated[1][generated_index]),
                    jnp.asarray(generated[2][generated_index]),
                    jnp.asarray(log_policy),
                )
                updates, disc_state = disc_optimizer.update(
                    gradients,
                    disc_state,
                    discriminator,
                )
                discriminator = eqx.apply_updates(discriminator, updates)
                last_loss = float(loss)

            reward_values = np.array(
                jnp.squeeze(discriminator.reward(input_jax), axis=-1),
                dtype=np.float64,
                copy=True,
            )
            reward_values = reward_values - reward_values.mean()
            shaping_values = np.asarray(
                jnp.squeeze(discriminator.shaping(input_jax), axis=-1),
                dtype=np.float64,
            )
            shaped_reward = reward_values[:, None]
            shaped_reward = shaped_reward + self.discount * np.einsum(
                "ast,t->sa",
                transitions,
                shaping_values,
            )
            shaped_reward = shaped_reward - shaping_values[:, None]
            target = np.asarray(self._solve(shaped_reward, transitions).policy, dtype=np.float64)
            for _ in range(self.policy_steps):
                _, gradients = policy_loss(policy_network, jnp.asarray(target))
                updates, policy_state = policy_optimizer.update(
                    gradients,
                    policy_state,
                    policy_network,
                )
                policy_network = eqx.apply_updates(policy_network, updates)
            network_policy = np.asarray(
                jax.nn.softmax(policy_network(input_jax), axis=-1),
                dtype=np.float64,
            )
            updated = (1.0 - self.policy_step_size) * policy
            updated += self.policy_step_size * network_policy
            updated /= updated.sum(axis=1, keepdims=True)
            policy_change = float(0.5 * np.abs(updated - policy).sum(axis=1).mean())
            policy = updated

            if self.verbose and (round_index + 1) % 25 == 0:
                print(
                    f"round {round_index + 1}: discriminator={last_loss:.6f}, "
                    f"policy_tv={policy_change:.6f}"
                )
            if round_index + 1 >= self.min_rounds and policy_change <= self.convergence_tol:
                converged = True
                rounds = round_index + 1
                break

        reward_values = np.asarray(
            jnp.squeeze(discriminator.reward(input_jax), axis=-1),
            dtype=np.float64,
        )
        shaping_values = np.asarray(
            jnp.squeeze(discriminator.shaping(input_jax), axis=-1),
            dtype=np.float64,
        )
        return {
            "discriminator": discriminator,
            "policy_network": policy_network,
            "reward": reward_values,
            "shaping": shaping_values,
            "policy": policy,
            "converged": converged,
            "rounds": rounds,
            "disc_loss": last_loss,
        }

    def _simulate_generator(
        self,
        policy: np.ndarray,
        transitions: np.ndarray,
        initial_states: np.ndarray,
        lengths: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        states: list[int] = []
        actions: list[int] = []
        next_states: list[int] = []
        for initial, length in zip(initial_states, lengths):
            current = int(initial)
            for _ in range(int(length)):
                chosen = int(rng.choice(self.n_actions, p=policy[current]))
                following = int(rng.choice(self.n_states, p=transitions[chosen, current]))
                states.append(current)
                actions.append(chosen)
                next_states.append(following)
                current = following
        return (
            np.asarray(states, dtype=np.int32),
            np.asarray(actions, dtype=np.int32),
            np.asarray(next_states, dtype=np.int32),
        )

    def _solve(self, reward: np.ndarray, transitions: np.ndarray) -> Any:
        problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )
        operator = SoftBellmanOperator(problem, jnp.asarray(transitions))
        solver = hybrid_iteration if self.generator_solver == "hybrid" else value_iteration
        reward_values = np.asarray(reward, dtype=np.float64)
        if reward_values.shape == (self.n_states,):
            reward_values = np.repeat(reward_values[:, None], self.n_actions, axis=1)
        if reward_values.shape != (self.n_states, self.n_actions):
            raise ValueError("reward must have shape (n_states,) or (n_states, n_actions)")
        result = solver(
            operator,
            jnp.asarray(reward_values),
            tol=self.generator_tol,
            max_iter=self.generator_max_iter,
        )
        if not result.converged:
            raise RuntimeError("NeuralAIRL policy solve did not converge")
        return result

    def _calibrate_reward_scale(
        self,
        reward: np.ndarray,
        transitions: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> float:
        """Profile the fixed logit scale after adversarial shape recovery."""

        def negative_log_likelihood(log_scale: float) -> float:
            scale = float(np.exp(log_scale))
            policy = np.asarray(self._solve(scale * reward, transitions).policy)
            return float(-np.log(np.clip(policy[states, actions], 1e-12, 1.0)).sum())

        result = minimize_scalar(
            negative_log_likelihood,
            bounds=(-3.0, 3.0),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 80},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("NeuralAIRL reward-scale profile failed")
        return float(np.exp(result.x))

    def _run_fit_self_check(self, states: np.ndarray, actions: np.ndarray) -> None:
        """Fail the fit if its public behavioral surface is numerically false."""
        assert self.policy_ is not None
        assert self.reward_ is not None
        if not np.isfinite(self.reward_).all() or not np.isfinite(self.policy_).all():
            raise RuntimeError("NeuralAIRL self-check found nonfinite reward or policy values")
        if np.max(np.abs(self.policy_.sum(axis=1) - 1.0)) > 1e-6:
            raise RuntimeError("NeuralAIRL self-check found invalid policy probabilities")
        counts = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        np.add.at(counts, (states, actions), 1.0)
        totals = counts.sum(axis=1, keepdims=True)
        empirical = np.divide(
            counts,
            totals,
            out=np.full_like(counts, 1.0 / self.n_actions),
            where=totals > 0,
        )
        empirical_effect = float(0.5 * np.abs(empirical - 1.0 / self.n_actions).sum(axis=1).mean())
        fitted_effect = float(0.5 * np.abs(self.policy_ - 1.0 / self.n_actions).sum(axis=1).mean())
        if empirical_effect >= 0.05 and fitted_effect < 0.01:
            raise RuntimeError(
                "NeuralAIRL self-check rejected a near-uniform fitted policy for "
                "materially non-uniform demonstrations"
            )

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        """Centered neural state reward repeated across actions."""
        if self.reward_ is None:
            return None
        return np.repeat(np.asarray(self.reward_)[:, None], self.n_actions, axis=1)

    def predict_reward(self, states: np.ndarray) -> np.ndarray:
        """Return the fitted centered neural reward for integer state codes."""
        if self.reward_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        indices = np.asarray(states)
        if indices.ndim == 0:
            indices = indices[None]
        if not np.issubdtype(indices.dtype, np.integer):
            numeric = np.asarray(indices, dtype=np.float64)
            if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
                raise ValueError("states must contain finite integer codes")
        integer = indices.astype(np.int64)
        if np.any(integer < 0) or np.any(integer >= self.n_states):
            raise ValueError(f"states must lie in [0, {self.n_states})")
        return np.asarray(self.reward_)[integer]

    def predict_proba(
        self,
        states: np.ndarray,
        *,
        task_id: Any | None = None,
        period: int = 0,
    ) -> np.ndarray:
        """Return fitted policy probabilities for integer state codes."""
        if task_id is not None:
            raise NotImplementedError("NeuralAIRL does not support task-specific prediction")
        if period < 0:
            raise ValueError("period must be nonnegative")
        if self.policy_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        indices = np.asarray(states)
        if indices.ndim != 1:
            raise ValueError("states must be one-dimensional")
        integer = indices.astype(np.int64)
        if np.any(integer < 0) or np.any(integer >= self.n_states):
            raise ValueError(f"states must lie in [0, {self.n_states})")
        return np.asarray(self.policy_)[integer]

    def _fit_functional_bootstrap(self) -> None:
        """Whole-trajectory bootstrap for normalized reward and policy cells."""
        assert self._panel is not None
        assert self.transitions_ is not None
        rng = np.random.default_rng(self.se_seed if self.se_seed is not None else self.seed)
        trajectories = list(self._panel.trajectories)
        if len(trajectories) < 2:
            raise ValueError("bootstrap inference requires at least two trajectories")
        reward_draws: list[np.ndarray] = []
        policy_draws: list[np.ndarray] = []
        failures: list[str] = []
        for draw in range(self.n_bootstrap):
            indices = rng.integers(0, len(trajectories), size=len(trajectories))
            sample = TrajectoryPanel(trajectories=[trajectories[int(i)] for i in indices])
            clone = NeuralAIRL(
                n_states=self.n_states,
                n_actions=self.n_actions,
                discount=self.discount,
                feature_matrix=np.asarray(self.state_inputs_),
                feature_names=self.feature_names,
                reward_hidden_dim=self.reward_hidden_dim,
                reward_num_layers=self.reward_num_layers,
                shaping_hidden_dim=self.shaping_hidden_dim,
                shaping_num_layers=self.shaping_num_layers,
                policy_hidden_dim=self.policy_hidden_dim,
                policy_num_layers=self.policy_num_layers,
                reward_lr=self.reward_lr,
                policy_lr=self.policy_lr,
                discriminator_steps=self.discriminator_steps,
                policy_steps=self.policy_steps,
                batch_size=self.batch_size,
                generator_solver=self.generator_solver,
                generator_tol=self.generator_tol,
                generator_max_iter=self.generator_max_iter,
                policy_step_size=self.policy_step_size,
                max_rounds=self.max_rounds,
                min_rounds=self.min_rounds,
                convergence_tol=self.convergence_tol,
                shaping_l2_penalty=self.shaping_l2_penalty,
                compute_se=False,
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                verbose=False,
            )
            try:
                clone.fit(sample, transitions=np.asarray(self.transitions_))
                if not clone.converged_:
                    raise RuntimeError(str(clone.termination_reason_))
                reward_draws.append(np.asarray(clone.reward_))
                policy_draws.append(np.asarray(clone.policy_))
            except Exception as exc:
                failures.append(f"draw {draw}: {type(exc).__name__}: {exc}")
            finally:
                if (draw + 1) % 10 == 0:
                    jax.clear_caches()
                    gc.collect()
        jax.clear_caches()
        gc.collect()
        if len(reward_draws) < 2:
            raise RuntimeError("NeuralAIRL bootstrap produced fewer than two successful draws")
        rewards = np.stack(reward_draws)
        policies = np.stack(policy_draws)
        estimates = np.concatenate([np.asarray(self.reward_), np.asarray(self.policy_).ravel()])
        draws = np.concatenate([rewards, policies.reshape(len(policies), -1)], axis=1)
        standard_errors = draws.std(axis=0, ddof=1)
        names = tuple(
            [f"reward[{state}]" for state in range(self.n_states)]
            + [
                f"policy[{state},{action}]"
                for state in range(self.n_states)
                for action in range(self.n_actions)
            ]
        )
        self.bootstrap_ = FunctionalBootstrapResult(
            method="pairs_cluster",
            unit="individual_trajectory",
            n_requested=self.n_bootstrap,
            n_successful=len(rewards),
            seed=self.se_seed if self.se_seed is not None else self.seed,
            estimand_names=names,
            estimates=estimates,
            standard_errors=standard_errors,
            intervals=self._normal_intervals(estimates, standard_errors, alpha=0.05),
            reward_draws=rewards,
            policy_draws=policies,
            failures=tuple(failures),
        )

    def _normal_intervals(
        self,
        estimates: np.ndarray,
        standard_errors: np.ndarray,
        *,
        alpha: float,
    ) -> np.ndarray:
        """Return normal bootstrap intervals, respecting probability bounds."""
        critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        intervals = np.column_stack(
            [estimates - critical * standard_errors, estimates + critical * standard_errors]
        )
        intervals[self.n_states :, 0] = np.clip(intervals[self.n_states :, 0], 0.0, 1.0)
        intervals[self.n_states :, 1] = np.clip(intervals[self.n_states :, 1], 0.0, 1.0)
        return intervals

    def conf_int(self, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        """Return normal bootstrap intervals for reward and policy functionals."""
        if self.bootstrap_ is None:
            raise RuntimeError(
                "functional intervals require compute_se=True and a completed bootstrap"
            )
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be finite and lie strictly between 0 and 1")
        bootstrap = cast(FunctionalBootstrapResult, self.bootstrap_)
        intervals = self._normal_intervals(
            bootstrap.estimates,
            bootstrap.standard_errors,
            alpha=alpha,
        )
        return {
            name: (float(lower), float(upper))
            for name, (lower, upper) in zip(bootstrap.estimand_names, intervals)
        }

    def counterfactual(
        self,
        *,
        params: dict[str, float] | np.ndarray | None = None,
        transitions: np.ndarray | DeterministicTransitions | None = None,
        description: str | None = None,
        reward_delta: np.ndarray | None = None,
    ) -> CounterfactualResult:
        """Re-solve a transition or state-reward change using the neural reward."""
        if params is not None:
            raise NotImplementedError(
                "NeuralAIRL weights are not structural parameters. Use reward_delta instead."
            )
        if isinstance(transitions, DeterministicTransitions):
            raise ValueError("NeuralAIRL counterfactuals require a dense transition tensor")
        if self.reward_ is None or self.policy_ is None or self.value_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if (transitions is None) == (reward_delta is None):
            raise ValueError("supply exactly one of transitions or reward_delta")
        changed_transitions = np.asarray(
            self.transitions_ if transitions is None else transitions,
            dtype=np.float64,
        )
        changed_transitions = self._validate_dense_transitions(changed_transitions)
        changed_reward = np.asarray(self.reward_, dtype=np.float64).copy()
        cf_type = CounterfactualType.ENVIRONMENT_CHANGE
        changed = "transitions"
        if reward_delta is not None:
            delta = np.asarray(reward_delta, dtype=np.float64)
            if delta.shape != changed_reward.shape or not np.isfinite(delta).all():
                raise ValueError(f"reward_delta must be finite with shape {changed_reward.shape}")
            changed_reward += delta
            changed_reward -= changed_reward.mean()
            cf_type = CounterfactualType.REWARD_CHANGE
            changed = "reward"
        solution = self._solve(changed_reward, changed_transitions)
        policy = np.asarray(solution.policy)
        value = np.asarray(solution.V)
        metadata: dict[str, Any] = {
            "estimator": "NeuralAIRL",
            "reward_scope": "state_only_nonlinear",
            "neural_weights_interpretable": False,
        }
        if isinstance(self.bootstrap_, FunctionalBootstrapResult):
            policy_shifts: list[float] = []
            value_changes: list[float] = []
            for reward_draw in self.bootstrap_.reward_draws:
                baseline_draw = self._solve(
                    np.asarray(reward_draw),
                    np.asarray(self.transitions_),
                )
                changed_draw = np.asarray(reward_draw, dtype=np.float64).copy()
                if reward_delta is not None:
                    changed_draw += delta
                    changed_draw -= changed_draw.mean()
                changed_solution = self._solve(changed_draw, changed_transitions)
                policy_shifts.append(
                    float(
                        0.5
                        * np.abs(
                            np.asarray(changed_solution.policy) - np.asarray(baseline_draw.policy)
                        )
                        .sum(axis=1)
                        .mean()
                    )
                )
                value_changes.append(
                    float(np.mean(np.asarray(changed_solution.V - baseline_draw.V)))
                )
            metadata["bootstrap_intervals"] = {
                "method": self.bootstrap_.method,
                "n_successful": self.bootstrap_.n_successful,
                "mean_policy_tv": tuple(
                    float(value) for value in np.quantile(policy_shifts, [0.025, 0.975])
                ),
                "mean_value_change": tuple(
                    float(value) for value in np.quantile(value_changes, [0.025, 0.975])
                ),
            }
        return CounterfactualResult(
            baseline_policy=jnp.asarray(self.policy_),
            counterfactual_policy=jnp.asarray(policy),
            baseline_value=jnp.asarray(self.value_),
            counterfactual_value=jnp.asarray(value),
            policy_change=jnp.asarray(policy - self.policy_),
            value_change=jnp.asarray(value - self.value_),
            welfare_change=float(np.mean(value - self.value_)),
            counterfactual_type=cf_type,
            description=description or f"NeuralAIRL {changed} counterfactual",
            metadata=metadata,
            transitions=jnp.asarray(self.transitions_),
            counterfactual_transitions=jnp.asarray(changed_transitions),
            params={},
        )

    def summary(self, alpha: float = 0.05) -> str:
        """Return a compact report without structural-weight claims."""
        del alpha
        if self.policy_ is None:
            return "Estimator\nNeuralAIRL\n\nNot fitted. Call fit() first."
        optimization = (self.diagnostics_ or {}).get("optimization", {})
        uncertainty = (
            "Not computed for this fit"
            if self.bootstrap_ is None
            else f"Trajectory bootstrap: {self.bootstrap_.n_successful}/{self.n_bootstrap} draws"
        )
        return "\n".join(
            [
                "Estimator",
                "NeuralAIRL (state-only neural adversarial IRL)",
                "",
                "Data",
                f"Observations: {self.n_observations_}",
                "",
                "Model",
                f"States: {self.n_states}",
                f"Actions: {self.n_actions}",
                f"Discount: {self.discount:.6g}",
                "Reward: nonlinear state-only network",
                "",
                "Fit",
                f"Converged: {self.converged_}",
                f"Rounds: {self.n_iter_}",
                "Final discriminator loss: "
                f"{optimization.get('final_discriminator_loss', float('nan')):.6g}",
                "",
                "Outcome",
                f"Log likelihood: {self.log_likelihood_:.6f}",
                "Recovered object: centered state reward and induced policy",
                "",
                "Uncertainty",
                uncertainty,
                "",
                "Limitations",
                "Neural weights are not structural coefficients.",
                "Context and latent heterogeneity belong to AIRL2.",
            ]
        )


__all__ = ["NeuralAIRL"]
