"""Public sklearn-style wrapper for tabular AIRL."""

from __future__ import annotations

import warnings
from typing import Any, Literal, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from econirl.core.reward_spec import RewardSpec
from econirl.core.transition_models import DeterministicTransitions
from econirl.core.types import DDCProblem, Panel, TrajectoryPanel
from econirl.estimation.adversarial.airl import AIRLConfig, AIRLEstimator
from econirl.estimators.mce_irl import MCEIRL, MCEIRLTask
from econirl.inference.results import FunctionalBootstrapResult
from econirl.preferences.reward import LinearReward
from econirl.simulation.counterfactual import CounterfactualResult


class AIRL(MCEIRL):
    """Adversarial IRL for state-only rewards in finite tabular MDPs.

    The public class intentionally enforces the setting in which AIRL's
    disentangled reward has a dynamics-transfer interpretation. Use
    :class:`econirl.AIRL2` for observed or latent context and
    action-dependent rewards.
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.99,
        feature_matrix: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        *,
        reward_lr: float = 0.02,
        discriminator_steps: int = 5,
        generator_solver: Literal["value", "hybrid"] = "hybrid",
        generator_tol: float = 1e-8,
        generator_max_iter: int = 5_000,
        generator_reward: Literal["recovered", "log_odds", "f"] = "f",
        policy_step_size: float = 0.1,
        max_rounds: int = 200,
        min_rounds: int = 150,
        convergence_tol: float = 0.01,
        shaping_l2_penalty: float = 1e-8,
        compute_se: bool = False,
        n_bootstrap: int = 100,
        seed: int = 42,
        se_seed: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            discount=discount,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            se_method="bootstrap",
            n_bootstrap=n_bootstrap,
            compute_se=compute_se,
            inner_max_iter=generator_max_iter,
            se_seed=se_seed,
            verbose=verbose,
        )
        if not np.isfinite(reward_lr) or reward_lr <= 0:
            raise ValueError("reward_lr must be finite and positive")
        if discriminator_steps < 1:
            raise ValueError("discriminator_steps must be positive")
        if not 0.0 < policy_step_size <= 1.0:
            raise ValueError("policy_step_size must lie in (0, 1]")
        if max_rounds < 1 or min_rounds < 1 or min_rounds > max_rounds:
            raise ValueError("require 1 <= min_rounds <= max_rounds")
        if generator_reward not in {"recovered", "log_odds", "f"}:
            raise ValueError("generator_reward must be 'recovered', 'log_odds', or 'f'")

        self.reward_lr = reward_lr
        self.discriminator_steps = discriminator_steps
        self.generator_solver = generator_solver
        self.generator_tol = generator_tol
        self.generator_max_iter = generator_max_iter
        self.generator_reward = generator_reward
        self.policy_step_size = policy_step_size
        self.max_rounds = max_rounds
        self.min_rounds = min_rounds
        self.convergence_tol = convergence_tol
        self.shaping_l2_penalty = shaping_l2_penalty
        self.seed = seed

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
        tasks: list[MCEIRLTask] | None = None,
        task: str | None = None,
        features: RewardSpec | np.ndarray | None = None,
        context: Any | None = None,
    ) -> "AIRL":
        """Fit AIRL after validating its state-only identification boundary."""
        if context is not None:
            raise NotImplementedError(
                "AIRL does not accept context. Use AIRL2 for heterogeneous rewards."
            )
        if tasks is not None or task is not None:
            raise NotImplementedError("AIRL does not support task-specific fits")
        if reward is not None and features is not None:
            raise ValueError("supply at most one of reward or features")
        if features is not None and not isinstance(features, RewardSpec):
            feature_array = np.asarray(features)
            if feature_array.ndim == 3:
                if feature_array.shape[:2] != (self.n_states, self.n_actions):
                    raise ValueError(
                        "3D features must have shape "
                        f"({self.n_states}, {self.n_actions}, n_features)"
                    )
                if not np.allclose(feature_array, feature_array[:, :1, :]):
                    raise ValueError(
                        "AIRL requires state-only reward features. Use AIRL2 for "
                        "action-dependent or heterogeneous rewards."
                    )
                feature_array = feature_array[:, 0, :]
            if feature_array.ndim != 2 or feature_array.shape[0] != self.n_states:
                raise ValueError(
                    "features must have shape (n_states, n_features), or be "
                    "action-invariant with shape (n_states, n_actions, n_features)"
                )
            names = self.feature_names or [f"f{index}" for index in range(feature_array.shape[1])]
            features = RewardSpec.state_dependent(jnp.asarray(feature_array), names, self.n_actions)
        reward_spec = cast(RewardSpec | None, reward if reward is not None else features)
        if reward_spec is not None and not reward_spec.is_state_only:
            raise ValueError(
                "AIRL requires state-only reward features. Use AIRL2 for "
                "action-dependent or heterogeneous rewards."
            )
        if data is None:
            raise TypeError("data must be a DataFrame, Panel, or TrajectoryPanel")
        if transitions is None:
            raise ValueError("AIRL requires transitions with shape (n_actions, n_states, n_states)")
        if isinstance(transitions, DeterministicTransitions):
            raise ValueError(
                "AIRL requires a dense transition tensor with shape (n_actions, n_states, n_states)"
            )

        self._reset_fit_state()
        panel: Panel | TrajectoryPanel
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None or next_state is None:
                raise ValueError(
                    "state, action, id, and next_state column names are required "
                    "for AIRL DataFrame input"
                )
            self._validate_dataframe(data, state, action, id, next_state, None)
            panel = self._dataframe_to_panel(data, state, action, id, next_state, None)
        elif isinstance(data, (Panel, TrajectoryPanel)):
            panel = data
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, got {type(data)}"
            )
        self._panel = cast(Any, panel)
        self._validate_panel_support(panel)

        self.reward_spec_ = reward_spec
        if reward_spec is not None:
            feature_matrix = np.asarray(reward_spec.feature_matrix[:, 0, :])
            reward_fn = LinearReward(
                state_features=jnp.asarray(feature_matrix),
                parameter_names=reward_spec.parameter_names,
                n_actions=self.n_actions,
            )
        else:
            created_reward = self._create_reward()
            if not isinstance(created_reward, LinearReward):
                raise ValueError(
                    "AIRL requires a 2D state-only feature_matrix. Use AIRL2 "
                    "for action-dependent rewards."
                )
            reward_fn = created_reward

        self._reward_fn = cast(Any, reward_fn)
        tensor = self._build_transition_tensor(np.asarray(transitions))
        if np.asarray(transitions).ndim != 3:
            raise ValueError("AIRL transitions must have shape (n_actions, n_states, n_states)")
        self._validate_transition_model(tensor)
        tensor_array = np.asarray(tensor)
        self.transitions_ = np.asarray(transitions)
        self.transition_tensor_ = tensor_array
        self.transition_model_ = tensor
        self.transition_source_ = "supplied action-specific tensor"
        problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )
        self._problem = cast(Any, problem)

        rank = self._identification_diagnostics()
        if rank["feature_rank"] < rank["num_features"]:
            raise ValueError(
                f"state feature rank {rank['feature_rank']} is below "
                f"the {rank['num_features']} reward features"
            )
        self.diagnostics_ = self._contract_diagnostics(self.n_states, tensor, rank)
        self.diagnostics_["identification"].update(
            {
                "target": "state-only reward up to an additive constant and transferred behavior",
                "normalization": "reward centered over states; logit shock scale fixed at 1.0",
            }
        )

        estimator = AIRLEstimator(
            AIRLConfig(
                reward_type="linear",
                reward_arg="state",
                reward_lr=self.reward_lr,
                discriminator_steps=self.discriminator_steps,
                generator_solver=self.generator_solver,
                generator_tol=self.generator_tol,
                generator_max_iter=self.generator_max_iter,
                generator_reward=self.generator_reward,
                policy_step_size=self.policy_step_size,
                max_rounds=self.max_rounds,
                min_rounds=self.min_rounds,
                convergence_tol=self.convergence_tol,
                shaping_l2_penalty=self.shaping_l2_penalty,
                compute_se=False,
                seed=self.seed,
                verbose=self.verbose,
            )
        )
        try:
            result = estimator.estimate(
                panel=panel,
                utility=reward_fn,
                problem=problem,
                transitions=jnp.asarray(tensor_array),
            )
        except Exception as exc:
            self.termination_reason_ = "execution_failure"
            self.failure_reason_ = f"{type(exc).__name__}: {exc}"
            raise RuntimeError("AIRL estimation failed during optimization") from exc
        self._result = cast(Any, result)

        self._extract_airl_results()
        if self.compute_se:
            self._fit_functional_bootstrap()
        return self

    def _extract_airl_results(self) -> None:
        """Populate the shared public result fields without structural claims."""
        assert self._result is not None
        assert self._reward_fn is not None
        params = np.asarray(self._result.parameters, dtype=float)
        names = list(self._result.parameter_names)
        raw_reward = np.asarray(self._reward_fn.compute(jnp.asarray(params)))[:, 0]
        centered_reward = raw_reward - raw_reward.mean()

        self.params_ = {name: float(value) for name, value in zip(names, params)}
        self.coef_ = params
        self.se_ = None
        self.pvalues_ = None
        self.reward_ = centered_reward
        self.policy_ = np.asarray(self._result.policy)
        self.value_function_ = np.asarray(self._result.value_function)
        self.value_ = self.value_function_
        self.log_likelihood_ = float(self._result.log_likelihood)
        self.converged_ = bool(self._result.converged)
        convergence_message = self._result.convergence_message
        self.termination_reason_ = (
            "converged" if self.converged_ else str(convergence_message or "not_converged")
        )
        self.failure_reason_ = cast(Any, None if self.converged_ else self.termination_reason_)
        self.n_iter_ = int(self._result.num_iterations)
        self.fit_time_ = float(self._result.estimation_time)
        self.n_observations_ = int(self._result.num_observations)
        self.result_ = self._result
        self.is_fitted_ = True
        if self.diagnostics_ is not None:
            self.diagnostics_["optimization"] = {
                "converged": self.converged_,
                "termination_reason": self.termination_reason_,
                "failure_reason": self.failure_reason_,
                "iterations": self.n_iter_,
                "fit_time_seconds": self.fit_time_,
                "final_discriminator_loss": self._result.metadata.get("final_disc_loss"),
            }
        if not self.converged_:
            warnings.warn(
                "AIRL reached max_rounds before the policy-change stopping rule; "
                "inspect diagnostics_ before using the fitted reward.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        """Centered state-only reward repeated over the action axis."""
        if self.reward_ is None:
            return None
        reward = np.asarray(self.reward_, dtype=float)
        return cast(np.ndarray, np.repeat(reward[:, None], self.n_actions, axis=1))

    def _fit_functional_bootstrap(self) -> None:
        """Trajectory bootstrap for reward and policy functionals."""
        assert self._panel is not None
        assert self.reward_spec_ is not None or self.feature_matrix is not None
        rng = np.random.default_rng(self.se_seed)
        reward_draws: list[np.ndarray] = []
        policy_draws: list[np.ndarray] = []
        failures: list[str] = []
        trajectories = list(self._panel.trajectories)
        for draw in range(self.n_bootstrap):
            indices = rng.integers(0, len(trajectories), size=len(trajectories))
            training_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            panel = Panel(trajectories=[trajectories[index] for index in indices])
            clone = AIRL(
                n_states=self.n_states,
                n_actions=self.n_actions,
                discount=self.discount,
                feature_matrix=self.feature_matrix,
                feature_names=self.feature_names,
                reward_lr=self.reward_lr,
                discriminator_steps=self.discriminator_steps,
                generator_solver=self.generator_solver,
                generator_tol=self.generator_tol,
                generator_max_iter=self.generator_max_iter,
                generator_reward=self.generator_reward,
                policy_step_size=self.policy_step_size,
                max_rounds=self.max_rounds,
                min_rounds=self.min_rounds,
                convergence_tol=self.convergence_tol,
                shaping_l2_penalty=self.shaping_l2_penalty,
                compute_se=False,
                seed=training_seed,
                verbose=False,
            )
            try:
                clone.fit(
                    panel,
                    transitions=np.asarray(self.transition_tensor_),
                    reward=self.reward_spec_,
                )
                reward_draws.append(np.asarray(clone.reward_))
                policy_draws.append(np.asarray(clone.policy_))
            except Exception as exc:
                failures.append(f"draw {draw}: {type(exc).__name__}: {exc}")

        rewards = np.asarray(reward_draws)
        policies = np.asarray(policy_draws)
        if len(rewards) < 2:
            raise RuntimeError("AIRL trajectory bootstrap produced fewer than two successful draws")
        estimates = np.concatenate([np.asarray(self.reward_), np.asarray(self.policy_).ravel()])
        draws = np.concatenate([rewards, policies.reshape(len(policies), -1)], axis=1)
        names = tuple(
            [f"reward[{state}]" for state in range(self.n_states)]
            + [
                f"policy[{state},{action}]"
                for state in range(self.n_states)
                for action in range(self.n_actions)
            ]
        )
        standard_errors = draws.std(axis=0, ddof=1)
        intervals = np.quantile(
            draws,
            [0.025, 0.975],
            axis=0,
            method="inverted_cdf",
        ).T
        self.bootstrap_ = FunctionalBootstrapResult(
            method="pairs_cluster",
            unit="individual_trajectory",
            n_requested=self.n_bootstrap,
            n_successful=len(rewards),
            seed=self.se_seed,
            estimand_names=names,
            estimates=estimates,
            standard_errors=standard_errors,
            intervals=intervals,
            reward_draws=rewards,
            policy_draws=policies,
            failures=tuple(failures),
        )

    def counterfactual(
        self,
        *,
        params: dict[str, float] | np.ndarray | None = None,
        transitions: np.ndarray | DeterministicTransitions | None = None,
        description: str | None = None,
    ) -> CounterfactualResult:
        """Re-solve behavior after transitions or reward parameters change."""
        if isinstance(transitions, DeterministicTransitions):
            raise ValueError("AIRL counterfactuals require a dense transition tensor")
        result = super().counterfactual(
            params=params,
            transitions=transitions,
            description=description or "AIRL counterfactual",
        )
        result.metadata.update(
            {
                "estimator": "AIRL",
                "reward_scope": "state_only",
                "transfer_interpretation": (
                    "requires the state-only and decomposability conditions"
                ),
            }
        )
        result.transitions = (
            None if self.transition_tensor_ is None else jnp.asarray(self.transition_tensor_)
        )
        if transitions is not None:
            result.counterfactual_transitions = jnp.asarray(transitions)
        return result

    def conf_int(self, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        """Return percentile intervals for normalized reward and policy cells."""
        if self.bootstrap_ is None:
            raise RuntimeError(
                "functional intervals require compute_se=True and a completed bootstrap"
            )
        bootstrap = cast(FunctionalBootstrapResult, self.bootstrap_)
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be finite and lie strictly between 0 and 1")
        draws = np.concatenate(
            [
                bootstrap.reward_draws,
                bootstrap.policy_draws.reshape(bootstrap.n_successful, -1),
            ],
            axis=1,
        )
        intervals = np.quantile(
            draws,
            [alpha / 2.0, 1.0 - alpha / 2.0],
            axis=0,
            method="inverted_cdf",
        ).T
        return {
            name: (float(lower), float(upper))
            for name, (lower, upper) in zip(bootstrap.estimand_names, intervals)
        }

    def summary(self, alpha: float = 0.05) -> str:
        """Return a compact manager-readable AIRL fit summary."""
        if self._result is None:
            return "Estimator\nAIRL\n\nNot fitted. Call fit() first."
        diagnostics = self.diagnostics_ or {}
        data = diagnostics.get("data", {})
        optimization = diagnostics.get("optimization", {})
        uncertainty = (
            "Not computed for this fit"
            if self.bootstrap_ is None
            else (
                "Trajectory bootstrap: "
                f"{self.bootstrap_.n_successful}/{self.bootstrap_.n_requested} draws"
            )
        )
        return "\n".join(
            [
                "Estimator",
                "AIRL (state-only tabular adversarial IRL)",
                "",
                "Data",
                f"Observations: {self.n_observations_}",
                f"Individuals: {data.get('n_individuals', 'unavailable')}",
                f"State coverage: {data.get('state_coverage', float('nan')):.3f}",
                "",
                "Model",
                f"States: {self.n_states}",
                f"Actions: {self.n_actions}",
                f"Discount: {self.discount:.6g}",
                "Reward: state-only linear basis",
                "",
                "Pre-estimation checks",
                "Transition orientation: (n_actions, n_states, n_states)",
                "Reward feature rank: "
                f"{diagnostics.get('identification', {}).get('feature_rank', 'unavailable')}",
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
                "Transfer interpretation requires state-only rewards and AIRL's "
                "decomposability conditions.",
                "Raw adversarial weights are not structural coefficients.",
            ]
        )
