"""Sklearn-style MCE IRL estimator.

Maximum Causal Entropy Inverse Reinforcement Learning with sklearn-style API.
"""

from __future__ import annotations

import warnings
from typing import Hashable, Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.reward_spec import RewardSpec
from econirl.core.tasks import (
    CompiledMCEIRLTasks,
    MCEIRLTask,
    compile_mce_irl_tasks,
)
from econirl.core.transition_models import DeterministicTransitions, TransitionModel
from econirl.core.types import DDCProblem, Panel, Trajectory, TrajectoryPanel
from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.reward import LinearReward
from econirl.simulation.counterfactual import (
    CounterfactualResult,
    CounterfactualType,
)
from econirl.transitions import TransitionEstimator


def estimate_empirical_transitions(
    panel: Panel | TrajectoryPanel,
    n_actions: int,
    n_states: int,
) -> np.ndarray:
    """Estimate a per-action transition kernel ``P(s'|s,a)`` from observed data.

    Counts every observed ``(state, action, next_state)`` triple in the panel
    into an ``(n_actions, n_states, n_states)`` tensor and row-normalizes.  This
    is the general-MDP counterpart to the Rust-bus increment estimator: pass the
    result as ``transitions=`` to :meth:`MCEIRL.fit`.  Unobserved ``(state,
    action)`` rows fall back to staying in place.

    Parameters
    ----------
    panel : Panel or TrajectoryPanel
        Demonstrations with ``states``, ``actions``, and ``next_states``.
    n_actions, n_states : int
        Shape of the kernel to build.

    Returns
    -------
    numpy.ndarray
        Row-stochastic kernel of shape ``(n_actions, n_states, n_states)``.
    """
    trajectories = getattr(panel, "trajectories", None)
    if trajectories is None:
        raise TypeError("panel must be a Panel/TrajectoryPanel with .trajectories")

    counts = np.zeros((n_actions, n_states, n_states), dtype=np.float64)
    for traj in trajectories:
        s = np.asarray(traj.states, dtype=int)
        a = np.asarray(traj.actions, dtype=int)
        sp = np.asarray(traj.next_states, dtype=int)
        np.add.at(counts, (a, s, sp), 1.0)

    row_sums = counts.sum(axis=2, keepdims=True)
    kernel = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    empty = row_sums[..., 0] == 0
    rows, cols = np.nonzero(empty)
    kernel[rows, cols, cols] = 1.0  # unobserved (a, s): stay in place
    return kernel


class MCEIRL:
    """Sklearn-style Maximum Causal Entropy IRL estimator.

    Maximum Causal Entropy IRL (Ziebart 2010) recovers reward function
    parameters from demonstrated behavior, properly accounting for the
    causal structure of sequential decisions.

    Parameters
    ----------
    n_states : int, default=90
        Number of discrete states.
    n_actions : int, default=2
        Number of discrete actions.
    discount : float, default=0.99
        Time discount factor (beta). Use <0.999 for numerical stability.
    feature_matrix : numpy.ndarray, optional
        Feature matrix. State-only features have shape
        ``(n_states, n_features)``. Action-dependent features have shape
        ``(n_states, n_actions, n_features)``. For multi-action models,
        ``fit`` raises if neither ``feature_matrix`` nor ``reward`` is
        supplied; the old implicit state-index fallback is not a validated
        structural specification.
    feature_names : list[str], optional
        Names for each feature.
    se_method : str, default="bootstrap"
        Method for standard errors: "bootstrap", "asymptotic", or "hessian".
    n_bootstrap : int, default=100
        Number of bootstrap samples for SE computation.
    compute_se : bool, default=True
        Whether to compute standard errors.
    inner_max_iter : int, default=10000
        Maximum iterations for the infinite-horizon Bellman solve.
    horizon : int, optional
        Finite number of decision periods. Required when ``discount=1``.
    terminal_states : numpy.ndarray, optional
        Boolean terminal-state mask for a single-task fit.
    l2_regularization : float, default=0
        Per-observation L2 penalty. Positive values make the stationarity
        equation ``mu_expert - mu_model - 2 * penalty * theta = 0``.
    se_seed : int, optional
        Random seed for trajectory bootstrap standard errors.
    verbose : bool, default=False
        Print progress messages.

    Attributes
    ----------
    params_ : dict
        Estimated reward parameters {name: value}.
    se_ : dict
        Standard errors for each parameter.
    coef_ : numpy.ndarray
        Coefficients as array.
    reward_ : numpy.ndarray
        Policy-weighted state reward summary R(s) = sum_a pi(a|s) R(s,a), shape
        (n_states,). The structural state-action reward R(s,a) - the canonical MCE
        object (Gleave & Toyer 2022) - is in ``reward_matrix_``.
    policy_ : numpy.ndarray
        Learned policy π(a|s), shape (n_states, n_actions).
    value_function_ : numpy.ndarray
        Value function V(s) for each state.
    state_visitation_ : numpy.ndarray
        Expected state visitation frequencies.
    log_likelihood_ : float
        Log-likelihood of the data under learned model.
    converged_ : bool
        Whether the optimizer, stationarity, occupancy, and Bellman checks pass.
    time_policy_ : numpy.ndarray or None
        Period-specific policy for a finite-horizon fit.
    task_policy_ : dict or None
        Period-specific or stationary policy slices by task identifier.
    termination_reason_ : str or None
        Joint convergence or the first failed residual check.

    Examples
    --------
    >>> from econirl.estimators import MCEIRL
    >>> from econirl.datasets import load_rust_bus
    >>>
    >>> df = load_rust_bus()
    >>>
    >>> # State features: linear and quadratic mileage cost
    >>> n_states = 90
    >>> s = np.arange(n_states)
    >>> features = np.column_stack([s / 100, (s / 100) ** 2])
    >>>
    >>> model = MCEIRL(
    ...     n_states=n_states,
    ...     discount=0.99,
    ...     feature_matrix=features,
    ...     feature_names=["linear", "quadratic"],
    ...     verbose=True,
    ... )
    >>> model.fit(df, state="mileage_bin", action="replaced", id="bus_id")
    >>> print(model.summary())

    Notes
    -----
    For a general (non-bus) MDP, do not rely on the wrapper to infer dynamics.
    Pass a full 3D transition tensor ``transitions`` of shape ``(n_actions,
    n_states, n_states)`` (or build one from data with
    :func:`estimate_empirical_transitions`) and supply the observed next state
    via ``fit(..., next_state="next_state_col")``.  ``transitions=None`` only
    estimates the 2-action Rust-bus keep/replace kernel; a 2D matrix fills the
    non-keep actions with the bus "reset to state 0" kernel (a warning is
    raised); and a ``>2``-action MDP without explicit transitions is rejected.

    References
    ----------
    Ziebart, B. D. (2010). Modeling purposeful adaptive behavior with the
        principle of maximum causal entropy. PhD thesis, CMU.
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.99,
        feature_matrix: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        se_method: Literal["bootstrap", "asymptotic", "hessian"] = "bootstrap",
        n_bootstrap: int = 100,
        compute_se: bool = True,
        inner_max_iter: int = 10000,
        horizon: int | None = None,
        terminal_states: np.ndarray | None = None,
        l2_regularization: float = 0.0,
        se_seed: int | None = None,
        verbose: bool = False,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.feature_matrix = feature_matrix
        self.feature_names = feature_names
        self.se_method = se_method
        self.n_bootstrap = n_bootstrap
        self.compute_se = compute_se
        self.inner_max_iter = inner_max_iter
        self.horizon = horizon
        self.terminal_states = terminal_states
        self.l2_regularization = l2_regularization
        self.se_seed = se_seed
        self.verbose = verbose

        # Fitted attributes
        self.params_: dict | None = None
        self.se_: dict | None = None
        self.pvalues_: dict | None = None
        self.coef_: np.ndarray | None = None
        self.reward_: np.ndarray | None = None
        self.policy_: np.ndarray | None = None
        self.value_function_: np.ndarray | None = None
        self.value_: np.ndarray | None = None
        self.state_visitation_: np.ndarray | None = None
        self.transitions_: np.ndarray | DeterministicTransitions | None = None
        self.transition_model_: TransitionModel | None = None
        self.log_likelihood_: float | None = None
        self.converged_: bool | None = None
        self.reward_spec_: RewardSpec | None = None
        self.time_policy_: np.ndarray | None = None
        self.time_value_function_: np.ndarray | None = None
        self.task_policy_: dict[Hashable, np.ndarray] | None = None
        self.task_value_function_: dict[Hashable, np.ndarray] | None = None
        self.termination_reason_: str | None = None
        self.feature_residual_: float | None = None
        self.occupancy_residual_: float | None = None
        self.bellman_residual_: float | None = None

        # Internal
        self._result = None
        self._panel = None
        self._reward_fn = None
        self._problem = None
        self._compiled_tasks: CompiledMCEIRLTasks | None = None
        self._effective_terminal_states: np.ndarray | None = None
        self._tasks: list[MCEIRLTask] | None = None
        self._source_panel: Panel | None = None
        self._source_feature_matrix: np.ndarray | jnp.ndarray | None = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        next_state: str | None = None,
        transitions: np.ndarray | DeterministicTransitions | None = None,
        reward: RewardSpec | None = None,
        tasks: list[MCEIRLTask] | None = None,
        task: str | None = None,
    ) -> "MCEIRL":
        """Fit the MCE IRL estimator.

        Parameters
        ----------
        data : pandas.DataFrame or Panel or TrajectoryPanel
            Panel data with demonstrations.  When a DataFrame is passed,
            ``state``, ``action``, and ``id`` column names are required.
            When a Panel/TrajectoryPanel is passed, column names are ignored.
        state : str, optional
            Column name for state variable (required for DataFrame input).
        action : str, optional
            Column name for action variable (required for DataFrame input).
        id : str, optional
            Column name for individual/trajectory identifier (required for
            DataFrame input).
        next_state : str, optional
            Column name for the observed next state (DataFrame input only).
            When given, these observed transitions are used directly.  When
            omitted, interior next-states are taken from the following row and
            the final period is synthesized from the action (Rust-bus rule),
            which is only correct for a bus-shaped problem.
        transitions : numpy.ndarray, optional
            Transition kernel.  Either a full 3D ``(n_actions, n_states,
            n_states)`` tensor (used as given) or a 2D ``(n_states, n_states)``
            keep-action kernel (the other actions are filled with the Rust-bus
            replacement kernel, with a warning).  If None, a 2-action bus kernel
            is estimated from the data; a >2-action MDP must supply transitions
            explicitly.
        reward : RewardSpec, optional
            Reward/utility specification.  If provided, overrides the
            ``feature_matrix`` and ``feature_names`` parameters passed at
            construction time.
        tasks : list[MCEIRLTask], optional
            Finite-horizon task definitions over one shared deterministic
            transition system. Reward parameters remain shared across tasks.
        task : str, optional
            DataFrame column containing the task identifier. Every trajectory
            must map to exactly one supplied task.

        Returns
        -------
        self : MCEIRL
            Fitted estimator.
        """
        # --- Handle reward spec ---
        if reward is not None:
            self.reward_spec_ = reward

        # --- Handle data: DataFrame or Panel/TrajectoryPanel ---
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required when data is a DataFrame"
                )
            self._panel = self._dataframe_to_panel(
                data,
                state,
                action,
                id,
                next_state,
                task,
            )
        elif isinstance(data, (Panel, TrajectoryPanel)):
            self._panel = data
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, got {type(data)}"
            )
        self._compiled_tasks = None
        self._tasks = tasks
        self._source_panel = self._panel

        # Estimate transitions
        if transitions is None:
            if self.n_actions > 2:
                raise ValueError(
                    "MCEIRL cannot infer per-action transitions from data for a "
                    f"{self.n_actions}-action MDP. Pass transitions=<(n_actions, "
                    "n_states, n_states) array>, or build one with "
                    "estimate_empirical_transitions(panel, n_actions, n_states) "
                    "from econirl.estimators. The built-in increment "
                    "estimator only models the Rust-bus keep/replace dynamics and "
                    "is not valid for a general MDP."
                )
            trans_est = TransitionEstimator(n_states=self.n_states, max_increase=2)
            trans_est.fit(self._panel)
            self.transitions_ = trans_est.matrix_
        else:
            self.transitions_ = (
                transitions
                if isinstance(transitions, DeterministicTransitions)
                else np.asarray(transitions)
            )

        # Create reward function (RewardSpec overrides feature_matrix)
        if self.reward_spec_ is not None:
            self._reward_fn = self.reward_spec_.to_linear_reward()
        else:
            self._reward_fn = self._create_reward()
        self._source_feature_matrix = getattr(
            self._reward_fn,
            "feature_matrix",
            getattr(self._reward_fn, "state_features", None),
        )

        transition_tensor = self._build_transition_tensor(self.transitions_)
        effective_n_states = self.n_states
        effective_horizon = self.horizon
        effective_terminal_states = self.terminal_states
        if tasks is not None:
            if not isinstance(transition_tensor, DeterministicTransitions):
                raise TypeError(
                    "tasks require DeterministicTransitions so active subgraphs "
                    "can be compiled without dense state duplication"
                )
            feature_matrix = getattr(self._reward_fn, "feature_matrix", None)
            if feature_matrix is None:
                feature_matrix = getattr(self._reward_fn, "state_features", None)
            if feature_matrix is None:
                raise TypeError(
                    "task compilation requires a linear state or state-action feature matrix"
                )
            compiled = compile_mce_irl_tasks(
                tasks,
                transition_tensor,
                feature_matrix,
                self._panel,
            )
            self._compiled_tasks = compiled
            self._panel = compiled.panel
            transition_tensor = compiled.transitions
            effective_n_states = compiled.transitions.num_states
            effective_horizon = compiled.horizon
            effective_terminal_states = np.asarray(compiled.terminal_states)
            self._reward_fn = self._reward_with_features(compiled.feature_matrix)

        self.transition_model_ = transition_tensor
        self._effective_terminal_states = (
            None
            if effective_terminal_states is None
            else np.asarray(effective_terminal_states, dtype=bool)
        )
        self._problem = DDCProblem(
            num_states=effective_n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
            num_periods=effective_horizon,
        )

        self._warn_if_unidentified()

        # Create estimator with config
        config = MCEIRLConfig(
            se_method=self.se_method,
            n_bootstrap=self.n_bootstrap,
            compute_se=self.compute_se,
            inner_max_iter=self.inner_max_iter,
            l2_regularization=self.l2_regularization,
            verbose=self.verbose,
        )
        estimator = MCEIRLEstimator(config=config)

        # Estimate
        self._result = estimator.estimate(
            panel=self._panel,
            utility=self._reward_fn,
            problem=self._problem,
            transitions=transition_tensor,
            n_bootstrap=self.n_bootstrap,
            se_seed=self.se_seed,
            terminal_states=effective_terminal_states,
            initial_dist=(
                None if self._compiled_tasks is None else self._compiled_tasks.initial_state_dist
            ),
        )

        # Extract results
        self._extract_results()

        return self

    def _dataframe_to_panel(
        self,
        data: pd.DataFrame,
        state: str,
        action: str,
        id: str,
        next_state: str | None = None,
        task: str | None = None,
    ) -> Panel:
        """Convert DataFrame to Panel."""
        trajectories = []

        for ind_id, group in data.groupby(id, sort=True):
            sorted_group = group.sort_index()
            task_id = None
            if task is not None:
                task_values = sorted_group[task].drop_duplicates()
                if len(task_values) != 1:
                    raise ValueError(f"trajectory {ind_id!r} contains multiple task identifiers")
                task_id = task_values.iloc[0]

            states = sorted_group[state].values.astype(np.int64)
            actions = sorted_group[action].values.astype(np.int64)

            if next_state is not None:
                # Use the observed next-states directly.
                next_states = sorted_group[next_state].values.astype(np.int64)
            else:
                # No observed next-state: interior from the following row, final
                # period synthesized from the action (Rust-bus rule).
                next_states = np.zeros_like(states)
                next_states[:-1] = states[1:]
                if len(states) > 0:
                    last_action = actions[-1]
                    if last_action == 1:
                        next_states[-1] = 0
                    else:
                        next_states[-1] = min(states[-1] + 1, self.n_states - 1)

            traj = Trajectory(
                states=np.array(states, dtype=np.int64),
                actions=np.array(actions, dtype=np.int64),
                next_states=np.array(next_states, dtype=np.int64),
                individual_id=ind_id,
                metadata={} if task_id is None else {"task_id": task_id},
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    def _reward_with_features(
        self,
        feature_matrix: jnp.ndarray,
    ) -> LinearReward | ActionDependentReward:
        """Rebuild the linear reward on compiled task-state features."""
        parameter_names = list(self._reward_fn.parameter_names)
        if feature_matrix.ndim == 3:
            return ActionDependentReward(
                feature_matrix=feature_matrix,
                parameter_names=parameter_names,
            )
        return LinearReward(
            state_features=feature_matrix,
            parameter_names=parameter_names,
            n_actions=self.n_actions,
        )

    def _build_transition_tensor(
        self,
        keep_transitions: np.ndarray | DeterministicTransitions,
    ) -> TransitionModel:
        """Build transition tensor for both actions."""
        if isinstance(keep_transitions, DeterministicTransitions):
            expected_shape = (self.n_states, self.n_actions)
            if keep_transitions.next_state.shape != expected_shape:
                raise ValueError(
                    "deterministic transitions must have shape "
                    f"{expected_shape}, got {keep_transitions.next_state.shape}"
                )
            return keep_transitions

        keep_transitions = np.asarray(keep_transitions, dtype=np.float32)
        if keep_transitions.ndim == 3:
            expected_shape = (self.n_actions, self.n_states, self.n_states)
            if keep_transitions.shape != expected_shape:
                raise ValueError(
                    f"3D transitions must have shape {expected_shape}, got {keep_transitions.shape}"
                )
            return jnp.array(keep_transitions)

        # 2D input specifies only the keep-action (a=0) kernel.
        if self.n_actions > 2:
            raise ValueError(
                "A 2D transition matrix only specifies the keep-action (a=0) "
                f"kernel, but n_actions={self.n_actions}. Pass a full 3D "
                "(n_actions, n_states, n_states) tensor so every action's "
                "dynamics are defined."
            )

        warnings.warn(
            "MCEIRL received a 2D transition matrix (keep-action kernel only). "
            "Action 1 transitions are set to the Rust-bus replacement kernel "
            "(reset to state 0). For a general MDP pass a 3D (n_actions, "
            "n_states, n_states) `transitions` array to fit().",
            UserWarning,
            stacklevel=2,
        )

        n = self.n_states
        transitions = np.zeros((self.n_actions, n, n), dtype=np.float32)

        # Action 0 (keep): use provided transitions
        transitions[0] = keep_transitions

        # Action 1 (replace): Rust-bus reset-to-state-0 kernel.
        for action in range(1, self.n_actions):
            for s in range(n):
                transitions[action, s, :] = transitions[0, 0, :]

        return jnp.array(transitions)

    def _warn_if_unidentified(self) -> None:
        """Warn when action-dependent features have a rank-deficient contrast.

        Action-specific reward parameters are identified only if the
        action-contrast design phi(s, a) - phi(s, 0) has full column rank.
        When it does not, the parameters lie on a ridge and recovery can fail
        even with correct transitions, which is a feature-design problem rather
        than an estimator bug.
        """
        feature_matrix = getattr(self._reward_fn, "feature_matrix", None)
        if feature_matrix is None:
            return  # state-only reward: identification is through the dynamics
        fm = np.asarray(feature_matrix)
        if fm.ndim != 3:
            return
        _, _, k = fm.shape
        contrast = (fm[:, 1:, :] - fm[:, :1, :]).reshape(-1, k)
        rank = int(np.linalg.matrix_rank(contrast))
        if rank < k:
            warnings.warn(
                f"Action-contrast feature rank is {rank} < {k} features. "
                "Action-specific reward parameters are not identified (they lie "
                "on a ridge); recovered coefficients and per-action feature "
                "residuals may be unreliable even with correct transitions. "
                "Check the feature design before trusting parameter estimates.",
                UserWarning,
                stacklevel=2,
            )

    def _create_reward(self) -> LinearReward | ActionDependentReward:
        """Create reward function."""
        if self.feature_matrix is None:
            if self.n_actions > 1:
                raise ValueError(
                    "MCEIRL requires an explicit reward specification for "
                    "multi-action structural recovery. Pass `reward=RewardSpec(...)` "
                    "to fit(), or pass `feature_matrix` at construction time. "
                    "The old state-index fallback is not identified for "
                    "multi-action MCE-IRL."
                )
            features = jnp.expand_dims(
                jnp.arange(self.n_states, dtype=jnp.float32),
                axis=1,
            )
            n_features = 1
        else:
            features = jnp.array(self.feature_matrix, dtype=jnp.float32)
            if features.ndim == 2:
                n_features = features.shape[1]
            elif features.ndim == 3:
                if features.shape[:2] != (self.n_states, self.n_actions):
                    raise ValueError(
                        "3D feature_matrix must have shape "
                        f"({self.n_states}, {self.n_actions}, n_features), "
                        f"got {features.shape}"
                    )
                n_features = features.shape[2]
            else:
                raise ValueError(
                    "feature_matrix must be 2D (state-only) or 3D "
                    f"(state-action), got shape {features.shape}"
                )

        if self.feature_names is not None:
            param_names = list(self.feature_names)
        else:
            param_names = [f"f{i}" for i in range(n_features)]

        if len(param_names) != n_features:
            raise ValueError(
                f"feature_names length {len(param_names)} must match feature dimension {n_features}"
            )

        if features.ndim == 3:
            return ActionDependentReward(
                feature_matrix=features,
                parameter_names=param_names,
            )

        return LinearReward(
            state_features=features,
            parameter_names=param_names,
            n_actions=self.n_actions,
        )

    def _extract_results(self) -> None:
        """Extract results into sklearn-style attributes."""
        if self._result is None:
            return

        params = np.asarray(self._result.parameters)
        param_names = self._result.parameter_names

        self.params_ = {name: float(val) for name, val in zip(param_names, params)}
        self.coef_ = params.copy()

        # Standard errors from metadata
        if self._result.metadata and "standard_errors" in self._result.metadata:
            se_values = self._result.metadata["standard_errors"]
            if se_values is not None:
                self.se_ = {name: float(val) for name, val in zip(param_names, se_values)}
            else:
                se = np.asarray(self._result.standard_errors)
                self.se_ = {name: float(val) for name, val in zip(param_names, se)}
        else:
            se = np.asarray(self._result.standard_errors)
            self.se_ = {name: float(val) for name, val in zip(param_names, se)}

        # P-values from t-statistics (Wald test)
        if self.se_ is not None:
            pvalues: dict[str, float] = {}
            for name in self.params_:
                se_val = self.se_[name]
                if se_val and se_val > 0 and np.isfinite(se_val):
                    t_stat = self.params_[name] / se_val
                    pvalues[name] = float(2 * (1 - scipy_norm.cdf(abs(t_stat))))
                else:
                    pvalues[name] = float("nan")
            self.pvalues_ = pvalues

        # Reward function R(s): policy-weighted over actions,
        # R(s) = sum_a pi(a|s) R(s,a). For a state-only (linear) reward every
        # action column is identical, so this reduces to that reward; for an
        # action-dependent reward it is the correct per-state summary.
        reward_params = jnp.array(params, dtype=jnp.float32)
        reward_matrix = np.asarray(self._reward_fn.compute(reward_params))
        if self._result.policy is not None:
            policy = np.asarray(self._result.policy)
            self.reward_ = (policy * reward_matrix).sum(axis=1)
        else:
            self.reward_ = reward_matrix[:, 0]

        # Policy
        if self._result.policy is not None:
            self.policy_ = np.asarray(self._result.policy)
        if self._result.metadata:
            time_policy = self._result.metadata.get("time_policy")
            self.time_policy_ = None if time_policy is None else np.asarray(time_policy)
            self.termination_reason_ = self._result.metadata.get("termination_reason")
            self.feature_residual_ = self._result.metadata.get(
                "stationarity_residual",
                self._result.metadata.get("feature_difference"),
            )
            self.occupancy_residual_ = self._result.metadata.get("occupancy_moment_residual")
            self.bellman_residual_ = self._result.metadata.get("bellman_residual")

        # Value function
        if self._result.value_function is not None:
            self.value_function_ = np.asarray(self._result.value_function)
            self.value_ = self.value_function_
        if self._result.metadata:
            time_value = self._result.metadata.get("time_value_function")
            self.time_value_function_ = None if time_value is None else np.asarray(time_value)

        if self._compiled_tasks is not None:
            self.task_policy_ = {}
            self.task_value_function_ = {}
            for task_id, task_slice in self._compiled_tasks.task_slices.items():
                if self.time_policy_ is not None:
                    self.task_policy_[task_id] = self.time_policy_[:, task_slice, :]
                elif self.policy_ is not None:
                    self.task_policy_[task_id] = self.policy_[task_slice]
                if self.time_value_function_ is not None:
                    self.task_value_function_[task_id] = self.time_value_function_[:, task_slice]
                elif self.value_function_ is not None:
                    self.task_value_function_[task_id] = self.value_function_[task_slice]

        # State visitation
        if self._result.metadata and "state_visitation" in self._result.metadata:
            self.state_visitation_ = np.array(self._result.metadata["state_visitation"])

        self.log_likelihood_ = float(self._result.log_likelihood)
        self.converged_ = bool(self._result.converged)

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        """Structural reward matrix R(s,a) of shape (n_states, n_actions).

        Computes the reward matrix from the fitted parameters and the
        reward function. Returns None if the model has not been fitted.
        """
        if self.params_ is None or self._reward_fn is None or self._result is None:
            return None
        param_names = self._result.parameter_names
        param_vector = jnp.array(
            [self.params_[name] for name in param_names],
            dtype=jnp.float32,
        )
        reward_matrix = self._reward_fn.compute(param_vector)
        return np.asarray(reward_matrix)

    def predict_proba(
        self,
        states: np.ndarray,
        *,
        task_id: Hashable | None = None,
        period: int = 0,
    ) -> np.ndarray:
        """Predict choice probabilities.

        Parameters
        ----------
        states : numpy.ndarray
            Array of state indices.

        Returns
        -------
        proba : numpy.ndarray
            Choice probabilities, shape (len(states), n_actions).
        """
        if self.policy_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        states = np.asarray(states, dtype=np.int64)
        if self._compiled_tasks is not None:
            if task_id is None:
                raise ValueError("task_id is required for a multi-task fit")
            if task_id not in self._compiled_tasks.global_to_local:
                raise KeyError(f"unknown task_id {task_id!r}")
            mapping = self._compiled_tasks.global_to_local[task_id]
            try:
                local_states = np.asarray([mapping[int(s)] for s in states])
            except KeyError as exc:
                raise ValueError(f"state {exc.args[0]} is not active for task {task_id!r}") from exc
            task_policy = self.task_policy_[task_id]
            if task_policy.ndim == 3:
                if period < 0 or period >= task_policy.shape[0]:
                    raise ValueError(f"period must lie in [0, {task_policy.shape[0]})")
                return task_policy[period, local_states]
            return task_policy[local_states]
        if self.time_policy_ is not None:
            if period < 0 or period >= self.time_policy_.shape[0]:
                raise ValueError(f"period must lie in [0, {self.time_policy_.shape[0]})")
            return self.time_policy_[period, states]
        return self.policy_[states]

    def simulate(
        self,
        n_trajectories: int,
        *,
        task_id: Hashable | None = None,
        n_periods: int | None = None,
        seed: int | None = None,
    ) -> TrajectoryPanel:
        """Simulate trajectories from the fitted policy and dynamics."""
        if self.policy_ is None or self.transition_model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if n_trajectories < 1:
            raise ValueError("n_trajectories must be positive")

        rng = np.random.default_rng(seed)
        horizon = n_periods
        if horizon is None:
            horizon = self._problem.num_periods
        if horizon is None:
            horizon = max(self._panel.num_periods_per_individual)
        if horizon < 1:
            raise ValueError("n_periods must be positive")

        state_offset = 0
        local_to_global = None
        if self._compiled_tasks is not None:
            if task_id is None:
                raise ValueError("task_id is required for a multi-task fit")
            if task_id not in self._compiled_tasks.task_slices:
                raise KeyError(f"unknown task_id {task_id!r}")
            task_slice = self._compiled_tasks.task_slices[task_id]
            state_offset = task_slice.start
            local_to_global = self._compiled_tasks.local_to_global[task_id]
            initial_dist = self._compiled_tasks.task_initial_state_dist[task_id]
        else:
            initial_dist = np.zeros(self._problem.num_states, dtype=float)
            for trajectory in self._panel.trajectories:
                initial_dist[int(trajectory.states[0])] += 1.0
            initial_dist /= initial_dist.sum()

        terminal = self._effective_terminal_states
        trajectories = []
        for individual in range(n_trajectories):
            local_initial = int(rng.choice(len(initial_dist), p=initial_dist))
            state = state_offset + local_initial
            states = []
            actions = []
            next_states = []
            for period in range(horizon):
                if terminal is not None and terminal[state]:
                    break
                if self.time_policy_ is None:
                    action_prob = self.policy_[state]
                else:
                    action_prob = self.time_policy_[
                        min(period, self.time_policy_.shape[0] - 1),
                        state,
                    ]
                action = int(rng.choice(self.n_actions, p=action_prob))
                if isinstance(self.transition_model_, DeterministicTransitions):
                    successor = int(self.transition_model_.next_state[state, action])
                else:
                    successor = int(
                        rng.choice(
                            self._problem.num_states,
                            p=np.asarray(self.transition_model_[action, state]),
                        )
                    )

                if local_to_global is None:
                    output_state = state
                    output_successor = successor
                else:
                    output_state = int(local_to_global[state - state_offset])
                    output_successor = int(local_to_global[successor - state_offset])
                states.append(output_state)
                actions.append(action)
                next_states.append(output_successor)
                state = successor

            trajectories.append(
                Trajectory(
                    states=jnp.asarray(states, dtype=jnp.int32),
                    actions=jnp.asarray(actions, dtype=jnp.int32),
                    next_states=jnp.asarray(next_states, dtype=jnp.int32),
                    individual_id=individual,
                    metadata={} if task_id is None else {"task_id": task_id},
                )
            )
        return TrajectoryPanel(trajectories=trajectories)

    def counterfactual(
        self,
        *,
        params: dict[str, float] | np.ndarray | None = None,
        transitions: np.ndarray | DeterministicTransitions | None = None,
        description: str | None = None,
    ) -> CounterfactualResult:
        """Re-solve one reward or transition counterfactual."""
        if self.params_ is None or self.transition_model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if (params is None) == (transitions is None):
            raise ValueError("supply exactly one of params or transitions")

        baseline_params = np.asarray(self.coef_, dtype=float)
        changed_params = baseline_params.copy()
        changed_transitions: TransitionModel = self.transition_model_
        if params is not None:
            if isinstance(params, dict):
                unknown = set(params).difference(self.params_)
                if unknown:
                    raise KeyError(f"unknown reward parameters: {sorted(unknown)}")
                names = list(self._result.parameter_names)
                for name, value in params.items():
                    changed_params[names.index(name)] = float(value)
            else:
                changed_params = np.asarray(params, dtype=float)
                if changed_params.shape != baseline_params.shape:
                    raise ValueError(f"params must have shape {baseline_params.shape}")
            counterfactual_type = CounterfactualType.REWARD_CHANGE
        else:
            if (
                self._compiled_tasks is not None
                and isinstance(transitions, DeterministicTransitions)
                and transitions.num_states == self.n_states
            ):
                assert self._tasks is not None
                assert self._source_feature_matrix is not None
                assert self._source_panel is not None
                compiled = compile_mce_irl_tasks(
                    self._tasks,
                    transitions,
                    self._source_feature_matrix,
                    self._source_panel,
                    validate_demonstrations=False,
                )
                changed_transitions = compiled.transitions
            else:
                changed_transitions = (
                    transitions
                    if isinstance(transitions, DeterministicTransitions)
                    else jnp.asarray(transitions)
                )
            counterfactual_type = CounterfactualType.ENVIRONMENT_CHANGE

        operator = SoftBellmanOperator(
            self._problem,
            changed_transitions,
            terminal_states=self._effective_terminal_states,
        )
        reward_matrix = self._reward_fn.compute(jnp.asarray(changed_params))
        solver = MCEIRLEstimator(MCEIRLConfig(compute_se=False))
        values, policy, _ = solver._soft_value_iteration(
            operator,
            reward_matrix,
            num_periods=self._problem.num_periods,
        )
        if values.ndim == 2:
            counterfactual_value = values[0]
            counterfactual_policy = policy[0]
        else:
            counterfactual_value = values
            counterfactual_policy = policy

        baseline_value = jnp.asarray(self.value_function_)
        baseline_policy = jnp.asarray(self.policy_)
        value_change = counterfactual_value - baseline_value
        result_params = {
            name: float(value)
            for name, value in zip(
                self._result.parameter_names,
                changed_params,
            )
        }
        return CounterfactualResult(
            baseline_policy=baseline_policy,
            counterfactual_policy=counterfactual_policy,
            baseline_value=baseline_value,
            counterfactual_value=counterfactual_value,
            policy_change=counterfactual_policy - baseline_policy,
            value_change=value_change,
            welfare_change=float(jnp.mean(value_change)),
            counterfactual_type=counterfactual_type,
            description=description or "MCE-IRL counterfactual",
            metadata={
                "reward_level_identified": False,
                "changed_primitive": ("reward_parameters" if params is not None else "transitions"),
            },
            params=result_params,
        )

    def conf_int(self, alpha: float = 0.05) -> dict:
        """Compute confidence intervals for parameters.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level.  Returns (1 - alpha) confidence intervals.

        Returns
        -------
        dict
            ``{param_name: (lower, upper)}`` confidence intervals.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.params_ is None or self.se_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        z = scipy_norm.ppf(1 - alpha / 2)
        intervals: dict[str, tuple[float, float]] = {}
        for name in self.params_:
            est = self.params_[name]
            se = self.se_[name]
            intervals[name] = (est - z * se, est + z * se)
        return intervals

    def summary(self) -> str:
        """Generate formatted summary of results."""
        if self._result is None:
            return "MCEIRL: Not fitted yet. Call fit() first."

        lines = []
        lines.append("=" * 70)
        lines.append("Maximum Causal Entropy IRL Results".center(70))
        lines.append("=" * 70)
        lines.append(f"{'Method:':<25} MCE IRL (Ziebart 2010)")
        lines.append(f"{'Discount Factor (β):':<25} {self.discount}")
        fitted_states = self._problem.num_states if self._problem is not None else self.n_states
        lines.append(f"{'No. States:':<25} {fitted_states}")
        lines.append(f"{'No. Actions:':<25} {self.n_actions}")
        if self._compiled_tasks is not None:
            lines.append(f"{'No. Tasks:':<25} {len(self._compiled_tasks.task_slices)}")
        lines.append(f"{'Log-Likelihood:':<25} {self.log_likelihood_:,.2f}")
        lines.append(f"{'Converged:':<25} {'Yes' if self.converged_ else 'No'}")
        if self.termination_reason_ is not None:
            lines.append(f"{'Termination:':<25} {self.termination_reason_}")
        if self.feature_residual_ is not None:
            lines.append(f"{'Feature residual:':<25} {self.feature_residual_:.6g}")
        if self.occupancy_residual_ is not None:
            lines.append(f"{'Occupancy residual:':<25} {self.occupancy_residual_:.6g}")
        if self.bellman_residual_ is not None:
            lines.append(f"{'Bellman residual:':<25} {self.bellman_residual_:.6g}")
        lines.append("-" * 70)
        lines.append("")
        lines.append("Parameter Estimates:")
        lines.append("-" * 70)
        lines.append(
            f"{'Parameter':<20} {'Estimate':>12} {'Std Err':>12} {'t-stat':>10} {'95% CI':>20}"
        )
        lines.append("-" * 70)

        for name in self.params_:
            param = self.params_[name]
            se = self.se_.get(name, float("nan")) if self.se_ else float("nan")

            if np.isfinite(se) and se > 0:
                t_stat = param / se
                ci_low = param - 1.96 * se
                ci_high = param + 1.96 * se
                ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]"
            else:
                t_stat = float("nan")
                ci_str = "[nan, nan]"

            lines.append(f"{name:<20} {param:>12.4f} {se:>12.4f} {t_stat:>10.2f} {ci_str:>20}")

        lines.append("-" * 70)

        # Feature matching diagnostics
        if self._result.metadata:
            emp = self._result.metadata.get("empirical_features", [])
            exp = self._result.metadata.get("final_expected_features", [])
            diff = self._result.metadata.get("feature_difference", 0)

            lines.append("")
            lines.append("Feature Matching Diagnostics:")
            lines.append(f"  Feature difference (||μ_D - μ_π||): {diff:.6f}")
            if emp and exp:
                lines.append(f"  Empirical features: {[f'{x:.4f}' for x in emp]}")
                lines.append(f"  Expected features:  {[f'{x:.4f}' for x in exp]}")

        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        fitted = self.params_ is not None
        return (
            f"MCEIRL(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, fitted={fitted})"
        )
