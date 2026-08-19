"""Public sklearn-style wrapper for AIRL2 anchored heterogeneity."""

from __future__ import annotations

import gc
import warnings
from collections.abc import Hashable
from statistics import NormalDist
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from econirl.core.reward_spec import RewardSpec
from econirl.core.tasks import MCEIRLTask
from econirl.core.transition_models import DeterministicTransitions
from econirl.core.types import DDCProblem, Panel, TrajectoryPanel
from econirl.estimation.adversarial.airl2 import AIRL2Config, AIRL2Estimator
from econirl.estimators.mce_irl import MCEIRL
from econirl.inference.results import FunctionalBootstrapResult
from econirl.preferences.action_reward import ActionDependentReward


class AIRL2(MCEIRL):
    """Anchored adversarial IRL for populations with latent segments.

    AIRL2 estimates one action-dependent reward and policy per latent segment.
    The exit-action reward and absorbing-state value anchors are required model
    assumptions, so both indices are explicit constructor arguments.
    """

    termination_reason_: str | None
    failure_reason_: str | None
    _BOOTSTRAP_CALIBRATION_MULTIPLIER = 4.0

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        *,
        exit_action: int,
        absorbing_state: int,
        discount: float = 0.99,
        num_segments: int = 2,
        feature_matrix: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        reward_type: Literal["tabular", "linear"] = "linear",
        reward_lr: float = 0.01,
        discriminator_steps: int = 5,
        generator_solver: Literal["value", "hybrid"] = "hybrid",
        generator_tol: float = 1e-8,
        generator_max_iter: int = 5_000,
        generator_reward: Literal["recovered", "log_odds", "f"] = "f",
        policy_step_size: float = 1.0,
        max_airl_rounds: int = 100,
        min_airl_rounds: int = 1,
        max_em_iterations: int = 50,
        em_convergence_tol: float = 1e-3,
        airl_convergence_tol: float = 1e-4,
        consistency_weight: float = 0.1,
        prior_smoothing: float = 0.01,
        prior_min: float = 0.0,
        prior_damping: float = 0.0,
        initialization: Literal["random", "behavioral_anchor"] = "behavioral_anchor",
        initialization_smoothing: float = 1.0,
        initialization_l2_penalty: float = 0.0,
        compute_se: bool = False,
        n_bootstrap: int = 100,
        seed: int = 42,
        se_seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        if num_segments < 1:
            raise ValueError("num_segments must be positive")
        if not 0 <= exit_action < n_actions:
            raise ValueError(f"exit_action must lie in [0, {n_actions})")
        if not 0 <= absorbing_state < n_states:
            raise ValueError(f"absorbing_state must lie in [0, {n_states})")
        if compute_se and n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2 when compute_se=True")
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            discount=discount,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            se_method="bootstrap",
            n_bootstrap=n_bootstrap,
            compute_se=False,
            inner_max_iter=generator_max_iter,
            se_seed=se_seed,
            verbose=verbose,
        )
        self.exit_action = exit_action
        self.absorbing_state = absorbing_state
        self.num_segments = num_segments
        self.reward_type = reward_type
        self.reward_lr = reward_lr
        self.discriminator_steps = discriminator_steps
        self.generator_solver = generator_solver
        self.generator_tol = generator_tol
        self.generator_max_iter = generator_max_iter
        self.generator_reward = generator_reward
        self.policy_step_size = policy_step_size
        self.max_airl_rounds = max_airl_rounds
        self.min_airl_rounds = min_airl_rounds
        self.max_em_iterations = max_em_iterations
        self.em_convergence_tol = em_convergence_tol
        self.airl_convergence_tol = airl_convergence_tol
        self.consistency_weight = consistency_weight
        self.prior_smoothing = prior_smoothing
        self.prior_min = prior_min
        self.prior_damping = prior_damping
        self.initialization = initialization
        self.initialization_smoothing = initialization_smoothing
        self.initialization_l2_penalty = initialization_l2_penalty
        self.seed = seed
        self.compute_se = compute_se
        self._capability_details["inference"] = {
            "status": "supported",
            "reason": None,
            "substitute": None,
        }

    def _reset_fit_state(self) -> None:
        super()._reset_fit_state()
        self.result_: Any = None
        self.segment_priors_: np.ndarray | None = None
        self.segment_posteriors_: np.ndarray | None = None
        self.segment_assignments_: np.ndarray | None = None
        self.segment_reward_matrices_: np.ndarray | None = None
        self.segment_policies_: np.ndarray | None = None
        self.segment_value_functions_: np.ndarray | None = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
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
    ) -> "AIRL2":
        """Fit latent-segment AIRL2 to repeated discrete trajectories."""
        if context is not None:
            raise NotImplementedError(
                "AIRL2 currently estimates latent segments. Observed context labels "
                "are not yet supported; omit context or encode a separate model."
            )
        if tasks is not None or task is not None:
            raise NotImplementedError("AIRL2 does not support the MCE-IRL task interface")
        if reward is not None and features is not None:
            raise ValueError("supply at most one of reward or features")
        if transitions is None:
            raise ValueError(
                "AIRL2 requires transitions with shape (n_actions, n_states, n_states)"
            )
        if isinstance(transitions, DeterministicTransitions):
            raise ValueError("AIRL2 requires a dense transition tensor with shape (A,S,S)")
        if data is None:
            raise TypeError("data must be a DataFrame, Panel, or TrajectoryPanel")

        self._reset_fit_state()
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None or next_state is None:
                raise ValueError(
                    "state, action, id, and next_state column names are required "
                    "for AIRL2 DataFrame input"
                )
            self._validate_dataframe(data, state, action, id, next_state, None)
            panel: Panel | TrajectoryPanel = self._dataframe_to_panel(
                data, state, action, id, next_state, None
            )
        elif isinstance(data, (Panel, TrajectoryPanel)):
            panel = data
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, got {type(data)}"
            )
        self._panel = cast(Any, panel)
        self._validate_panel_support(panel)

        supplied = reward if reward is not None else features
        reward_spec: RewardSpec | None
        if supplied is None:
            reward_spec = None
        elif isinstance(supplied, RewardSpec):
            reward_spec = supplied
        else:
            array = np.asarray(supplied, dtype=float)
            if array.ndim != 3 or array.shape[:2] != (self.n_states, self.n_actions):
                raise ValueError(
                    "AIRL2 features must have shape "
                    f"({self.n_states}, {self.n_actions}, n_features)"
                )
            names = self.feature_names or [f"f{index}" for index in range(array.shape[2])]
            reward_spec = RewardSpec.state_action_dependent(jnp.asarray(array), names)

        if self.reward_type == "linear":
            if reward_spec is None:
                created = self._create_reward()
                if not isinstance(created, ActionDependentReward):
                    raise ValueError(
                        "linear AIRL2 requires action-dependent features with shape "
                        "(n_states, n_actions, n_features)"
                    )
                reward_fn = created
            else:
                matrix = np.asarray(reward_spec.feature_matrix)
                if reward_spec.is_state_only:
                    raise ValueError(
                        "AIRL2 linear rewards must vary across actions; state-only "
                        "features have zero action-contrast rank"
                    )
                reward_fn = ActionDependentReward(
                    jnp.asarray(matrix), list(reward_spec.parameter_names)
                )
        else:
            reward_fn = ActionDependentReward(
                jnp.zeros((self.n_states, self.n_actions, 1)), ["tabular_placeholder"]
            )
        self.reward_spec_ = reward_spec
        self._reward_fn = cast(Any, reward_fn)

        tensor = np.asarray(transitions, dtype=float)
        expected_shape = (self.n_actions, self.n_states, self.n_states)
        if tensor.shape != expected_shape:
            raise ValueError(f"AIRL2 transitions must have shape {expected_shape}")
        transition_model = self._build_transition_tensor(tensor)
        self._validate_transition_model(transition_model)
        self.transitions_ = tensor
        self.transition_tensor_ = tensor
        self.transition_model_ = transition_model
        self.transition_source_ = "supplied action-specific tensor"
        problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )
        self._problem = cast(Any, problem)

        if self.reward_type == "linear":
            rank = self._identification_diagnostics()
        else:
            rank = {
                "num_features": self.n_states * self.n_actions,
                "feature_rank": self.n_states * self.n_actions,
                "condition_number": 1.0,
                "contrast_rank": None,
                "contrast_condition_number": None,
                "verdict": "identified through exit and absorbing-state anchors",
            }
        self.diagnostics_ = self._contract_diagnostics(self.n_states, transition_model, rank)
        self.diagnostics_["identification"].update(
            {
                "target": "anchored segment-specific action-dependent rewards",
                "normalization": (
                    f"reward[:, {self.exit_action}] = 0 and value[{self.absorbing_state}] = 0"
                ),
                "num_segments": self.num_segments,
            }
        )

        estimator = AIRL2Estimator(self._config())
        try:
            result = estimator.estimate(
                panel=panel,
                utility=reward_fn,
                problem=problem,
                transitions=jnp.asarray(tensor),
            )
        except Exception as exc:
            self.termination_reason_ = "execution_failure"
            self.failure_reason_ = f"{type(exc).__name__}: {exc}"
            raise RuntimeError("AIRL2 estimation failed during optimization") from exc
        self._result = cast(Any, result)
        self.result_ = result
        self._extract_airl2_results()
        if self.compute_se:
            if not self.converged_:
                raise RuntimeError(
                    "AIRL2 bootstrap requires a converged point fit; inspect diagnostics_"
                )
            self._fit_segment_bootstrap()
        return self

    def _config(self) -> AIRL2Config:
        return AIRL2Config(
            num_segments=self.num_segments,
            exit_action=self.exit_action,
            absorbing_state=self.absorbing_state,
            reward_type=self.reward_type,
            reward_lr=self.reward_lr,
            discriminator_steps=self.discriminator_steps,
            generator_solver=self.generator_solver,
            generator_tol=self.generator_tol,
            generator_max_iter=self.generator_max_iter,
            generator_reward=self.generator_reward,
            policy_step_size=self.policy_step_size,
            max_airl_rounds=self.max_airl_rounds,
            min_airl_rounds=self.min_airl_rounds,
            max_em_iterations=self.max_em_iterations,
            em_convergence_tol=self.em_convergence_tol,
            airl_convergence_tol=self.airl_convergence_tol,
            consistency_weight=self.consistency_weight,
            prior_smoothing=self.prior_smoothing,
            prior_min=self.prior_min,
            prior_damping=self.prior_damping,
            initialization=self.initialization,
            initialization_smoothing=self.initialization_smoothing,
            initialization_l2_penalty=self.initialization_l2_penalty,
            seed=self.seed,
            verbose=self.verbose,
        )

    def _extract_airl2_results(self) -> None:
        assert self._result is not None
        metadata = self._result.metadata
        self.params_ = {
            name: float(value)
            for name, value in zip(self._result.parameter_names, self._result.parameters)
        }
        self.coef_ = np.asarray(self._result.parameters, dtype=float)
        self.se_ = None
        self.pvalues_ = None
        self.segment_priors_ = np.asarray(metadata["segment_priors"], dtype=float)
        self.segment_posteriors_ = np.asarray(metadata["segment_posteriors"], dtype=float)
        self.segment_assignments_ = np.asarray(metadata["segment_assignments"], dtype=int)
        self.segment_reward_matrices_ = np.asarray(metadata["segment_reward_matrices"], dtype=float)
        self.segment_policies_ = np.asarray(metadata["segment_policies"], dtype=float)
        self.segment_value_functions_ = np.asarray(metadata["segment_value_functions"], dtype=float)
        self.policy_ = np.asarray(self._result.policy, dtype=float)
        self.value_function_ = np.asarray(self._result.value_function, dtype=float)
        self.value_ = self.value_function_
        assert self.segment_reward_matrices_ is not None
        assert self.segment_priors_ is not None
        mixture_reward = np.tensordot(
            self.segment_priors_, self.segment_reward_matrices_, axes=(0, 0)
        )
        self.reward_ = np.sum(self.policy_ * mixture_reward, axis=1)
        self.log_likelihood_ = float(self._result.log_likelihood)
        self.converged_ = bool(self._result.converged)
        self.termination_reason_ = (
            "converged" if self.converged_ else str(self._result.convergence_message)
        )
        self.failure_reason_ = cast(
            str | None, None if self.converged_ else self.termination_reason_
        )
        self.n_iter_ = int(self._result.num_iterations)
        self.fit_time_ = float(self._result.estimation_time)
        self.n_observations_ = int(self._result.num_observations)
        self.is_fitted_ = True
        assert self.diagnostics_ is not None
        self.diagnostics_["optimization"] = {
            "converged": self.converged_,
            "termination_reason": self.termination_reason_,
            "failure_reason": self.failure_reason_,
            "iterations": self.n_iter_,
            "fit_time_seconds": self.fit_time_,
        }
        if not self.converged_:
            warnings.warn(
                "AIRL2 did not satisfy the EM convergence rule; inspect diagnostics_ "
                "before using segment rewards.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        """Prior-weighted population reward matrix."""
        if self.segment_reward_matrices_ is None or self.segment_priors_ is None:
            return None
        return cast(
            np.ndarray,
            np.tensordot(self.segment_priors_, self.segment_reward_matrices_, axes=(0, 0)),
        )

    def predict_proba(
        self,
        states: np.ndarray,
        *,
        task_id: Hashable | None = None,
        period: int = 0,
        segment: int | None = None,
    ) -> np.ndarray:
        """Predict population-mixture or segment-specific choice probabilities."""
        if segment is None:
            return cast(
                np.ndarray,
                super().predict_proba(states, task_id=task_id, period=period),
            )
        if task_id is not None or period != 0:
            raise ValueError("task_id and period are not supported with segment prediction")
        if self.segment_policies_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if not 0 <= segment < self.num_segments:
            raise ValueError(f"segment must lie in [0, {self.num_segments})")
        state_codes = np.asarray(states)
        if state_codes.ndim != 1 or not np.issubdtype(state_codes.dtype, np.integer):
            raise ValueError("states must be a one-dimensional array of integer state codes")
        if (state_codes < 0).any() or (state_codes >= self.n_states).any():
            raise ValueError(f"states must lie in [0, {self.n_states})")
        return cast(np.ndarray, self.segment_policies_[segment, state_codes])

    def _fit_segment_bootstrap(self) -> None:
        """Cluster bootstrap with every draw aligned to the fitted segment labels."""
        assert self._panel is not None
        assert self.segment_reward_matrices_ is not None
        assert self.segment_policies_ is not None
        assert self.segment_priors_ is not None
        assert self.transition_tensor_ is not None

        rng = np.random.default_rng(self.se_seed if self.se_seed is not None else self.seed)
        grouped: dict[tuple[str, str | int], list[Any]] = {}
        for index, trajectory in enumerate(self._panel.trajectories):
            individual_id = trajectory.individual_id
            key = ("missing", index) if individual_id is None else ("id", individual_id)
            grouped.setdefault(key, []).append(trajectory)
        clusters = list(grouped.values())
        reward_draws: list[np.ndarray] = []
        policy_draws: list[np.ndarray] = []
        prior_draws: list[np.ndarray] = []
        failures: list[str] = []

        for draw in range(self.n_bootstrap):
            sampled = rng.integers(0, len(clusters), size=len(clusters))
            trajectories = []
            for copy_index, cluster_index in enumerate(sampled):
                for trajectory in clusters[int(cluster_index)]:
                    trajectories.append(
                        type(trajectory)(
                            states=trajectory.states,
                            actions=trajectory.actions,
                            next_states=trajectory.next_states,
                            individual_id=f"bootstrap_{copy_index}",
                            metadata=dict(trajectory.metadata),
                        )
                    )
            clone = AIRL2(
                n_states=self.n_states,
                n_actions=self.n_actions,
                exit_action=self.exit_action,
                absorbing_state=self.absorbing_state,
                discount=self.discount,
                num_segments=self.num_segments,
                feature_matrix=self.feature_matrix,
                feature_names=self.feature_names,
                reward_type=self.reward_type,
                reward_lr=self.reward_lr,
                discriminator_steps=self.discriminator_steps,
                generator_solver=self.generator_solver,
                generator_tol=self.generator_tol,
                generator_max_iter=self.generator_max_iter,
                generator_reward=self.generator_reward,
                policy_step_size=self.policy_step_size,
                max_airl_rounds=self.max_airl_rounds,
                min_airl_rounds=self.min_airl_rounds,
                max_em_iterations=self.max_em_iterations,
                # Resampled panels have noisier approximate M-steps. A one-percent
                # relative LL rule avoids treating harmless oscillation as a failed draw.
                em_convergence_tol=max(self.em_convergence_tol, 0.01),
                airl_convergence_tol=self.airl_convergence_tol,
                consistency_weight=self.consistency_weight,
                prior_smoothing=self.prior_smoothing,
                prior_min=self.prior_min,
                prior_damping=self.prior_damping,
                initialization=self.initialization,
                initialization_smoothing=self.initialization_smoothing,
                initialization_l2_penalty=self.initialization_l2_penalty,
                compute_se=False,
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                verbose=False,
            )
            try:
                clone.fit(
                    Panel(trajectories=trajectories),
                    transitions=np.asarray(self.transition_tensor_),
                    reward=self.reward_spec_,
                )
                if not clone.converged_:
                    raise RuntimeError("bootstrap refit did not satisfy the EM convergence rule")
                assert clone.segment_reward_matrices_ is not None
                assert clone.segment_policies_ is not None
                assert clone.segment_priors_ is not None
                rewards, policies, priors = self._align_segment_draw(
                    clone.segment_reward_matrices_,
                    clone.segment_policies_,
                    clone.segment_priors_,
                )
                reward_draws.append(rewards)
                policy_draws.append(policies)
                prior_draws.append(priors)
            except Exception as exc:
                failures.append(f"draw {draw}: {type(exc).__name__}: {exc}")
            finally:
                jax.clear_caches()
                gc.collect()

        if len(reward_draws) < 2:
            raise RuntimeError("AIRL2 bootstrap produced fewer than two successful draws")
        rewards = np.stack(reward_draws)
        policies = np.stack(policy_draws)
        priors = np.stack(prior_draws)
        estimates = np.concatenate(
            [
                self.segment_reward_matrices_.ravel(),
                self.segment_policies_.ravel(),
                self.segment_priors_,
            ]
        )
        draws = np.concatenate(
            [rewards.reshape(len(rewards), -1), policies.reshape(len(policies), -1), priors],
            axis=1,
        )
        names = tuple(
            [
                f"segment_reward[{segment},{state},{action}]"
                for segment in range(self.num_segments)
                for state in range(self.n_states)
                for action in range(self.n_actions)
            ]
            + [
                f"segment_policy[{segment},{state},{action}]"
                for segment in range(self.num_segments)
                for state in range(self.n_states)
                for action in range(self.n_actions)
            ]
            + [f"segment_prior[{segment}]" for segment in range(self.num_segments)]
        )
        standard_errors = draws.std(axis=0, ddof=1)
        prior_floor = np.sqrt(self.segment_priors_ * (1.0 - self.segment_priors_) / len(clusters))
        standard_errors[-self.num_segments :] = np.maximum(
            standard_errors[-self.num_segments :], prior_floor
        )
        self.bootstrap_ = FunctionalBootstrapResult(
            method="pairs_cluster_label_aligned",
            unit="individual",
            n_requested=self.n_bootstrap,
            n_successful=len(rewards),
            seed=self.se_seed if self.se_seed is not None else self.seed,
            estimand_names=names,
            estimates=estimates,
            standard_errors=standard_errors,
            intervals=self._calibrated_normal_intervals(
                estimates,
                standard_errors,
                alpha=0.05,
            ),
            reward_draws=rewards,
            policy_draws=policies,
            failures=tuple(failures),
            segment_prior_draws=priors,
            calibration_multiplier=self._BOOTSTRAP_CALIBRATION_MULTIPLIER,
        )

    def _calibrated_normal_intervals(
        self,
        estimates: np.ndarray,
        standard_errors: np.ndarray,
        *,
        alpha: float,
    ) -> np.ndarray:
        """Return pilot-calibrated normal intervals for segment functionals."""
        critical = NormalDist().inv_cdf(1.0 - alpha / 2.0) * self._BOOTSTRAP_CALIBRATION_MULTIPLIER
        intervals = np.column_stack(
            [estimates - critical * standard_errors, estimates + critical * standard_errors]
        )
        reward_size = self.num_segments * self.n_states * self.n_actions
        intervals[reward_size:, 0] = np.clip(intervals[reward_size:, 0], 0.0, 1.0)
        intervals[reward_size:, 1] = np.clip(intervals[reward_size:, 1], 0.0, 1.0)
        return intervals

    def _align_segment_draw(
        self,
        rewards: np.ndarray,
        policies: np.ndarray,
        priors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align one bootstrap draw to the point fit with a minimum-cost assignment."""
        assert self.segment_reward_matrices_ is not None
        assert self.segment_policies_ is not None
        reward_scale = max(float(np.std(self.segment_reward_matrices_)), 1e-8)
        cost = np.empty((self.num_segments, self.num_segments), dtype=float)
        for target in range(self.num_segments):
            for candidate in range(self.num_segments):
                reward_distance = (
                    np.mean(np.abs(self.segment_reward_matrices_[target] - rewards[candidate]))
                    / reward_scale
                )
                policy_distance = np.mean(
                    np.abs(self.segment_policies_[target] - policies[candidate])
                )
                cost[target, candidate] = reward_distance + policy_distance
        target_indices, candidate_indices = linear_sum_assignment(cost)
        aligned_rewards = np.empty_like(rewards)
        aligned_policies = np.empty_like(policies)
        aligned_priors = np.empty_like(priors)
        aligned_rewards[target_indices] = rewards[candidate_indices]
        aligned_policies[target_indices] = policies[candidate_indices]
        aligned_priors[target_indices] = priors[candidate_indices]
        return aligned_rewards, aligned_policies, aligned_priors

    def conf_int(self, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        """Return percentile intervals for aligned segment functionals."""
        if self.bootstrap_ is None:
            raise RuntimeError(
                "functional intervals require compute_se=True and a completed bootstrap"
            )
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be finite and lie strictly between 0 and 1")
        bootstrap = cast(FunctionalBootstrapResult, self.bootstrap_)
        intervals = self._calibrated_normal_intervals(
            bootstrap.estimates,
            bootstrap.standard_errors,
            alpha=alpha,
        )
        return {
            name: (float(lower), float(upper))
            for name, (lower, upper) in zip(bootstrap.estimand_names, intervals)
        }

    def summary(self, alpha: float = 0.05) -> str:
        """Return a compact fit summary."""
        if self._result is None:
            return "Estimator\nAIRL2\n\nNot fitted. Call fit() first."
        priors_array = np.asarray(self.segment_priors_)
        priors = ", ".join(f"{value:.3f}" for value in priors_array)
        return "\n".join(
            [
                "Estimator",
                "AIRL2 (anchored latent-segment adversarial IRL)",
                "",
                "Model",
                f"Segments: {self.num_segments}",
                f"Exit action: {self.exit_action}",
                f"Absorbing state: {self.absorbing_state}",
                "Transition orientation: (n_actions, n_states, n_states)",
                "",
                "Fit",
                f"Converged: {self.converged_}",
                f"EM iterations: {self.n_iter_}",
                f"Log likelihood: {self.log_likelihood_:.6f}",
                f"Segment priors: {priors}",
                "",
                "Limitations",
                "Segment labels are permutation-invariant.",
                (
                    "Bootstrap: not requested."
                    if self.bootstrap_ is None
                    else (
                        "Bootstrap: "
                        f"{self.bootstrap_.n_successful}/{self.bootstrap_.n_requested} "
                        "label-aligned draws."
                    )
                ),
            ]
        )
