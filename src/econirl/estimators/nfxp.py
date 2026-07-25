"""Sklearn-style NFXP estimator for dynamic discrete choice models.

This module provides an NFXP class with a scikit-learn style API that wraps
the underlying NFXPEstimator from econirl.estimation.nfxp. It accepts pandas
DataFrames with column names instead of the low-level Panel API.

Example:
    >>> from econirl.estimators import NFXP
    >>> import pandas as pd
    >>>
    >>> # Load bus replacement data
    >>> df = pd.read_csv("zurcher_bus.csv")
    >>>
    >>> # Create estimator and fit
    >>> from econirl.datasets import rust_bus_reward_spec
    >>> model = NFXP(n_states=90, discount=0.9999, utility=rust_bus_reward_spec(90))
    >>> model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> # Access results sklearn-style
    >>> print(model.params_)        # {"operating_cost": 0.001, "replacement_cost": 9.35}
    >>> print(model.coef_)          # numpy array [0.001, 9.35]
    >>> print(model.log_likelihood_)
    >>> print(model.summary())
    >>>
    >>> # Predict choice probabilities
    >>> proba = model.predict_proba(states=np.array([0, 10, 50]))
    >>>
    >>> # Simulate new data under estimated policy
    >>> sim_df = model.simulate(n_agents=100, n_periods=50, seed=42)
    >>>
    >>> # Counterfactual analysis: what if the replacement cost was higher?
    >>> cf_result = model.counterfactual(replacement_cost=15.0)
    >>> print(cf_result.policy)  # New policy under higher replacement cost
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import DDCProblem, Panel, Trajectory, TrajectoryPanel
from econirl.estimation.nfxp import NFXPEstimator
from econirl.simulation.counterfactual import (
    CounterfactualResult,
    counterfactual_policy,
    counterfactual_transitions,
)
from econirl.transitions import TransitionEstimator


class NFXP:
    """Sklearn-style NFXP estimator for dynamic discrete choice models.

    The Nested Fixed Point (NFXP) algorithm estimates utility parameters by
    nesting the solution of the Bellman equation within likelihood maximization.
    This is the classic approach from Rust (1987, 1988).

    Parameters
    ----------
    n_states : int, default=90
        Number of discrete states (e.g., mileage bins).
    n_actions : int, default=2
        Number of discrete actions (e.g., keep/replace).
    discount : float, default=0.9999
        Time discount factor (beta).
    utility : RewardSpec
        Utility specification as a ``RewardSpec``.  For the classic Rust bus
        model, use ``rust_bus_reward_spec(n_states)`` from
        ``econirl.datasets``.
    se_method : str, default="robust"
        Method for computing standard errors. Options: "asymptotic", "robust",
        "clustered", "bootstrap", and "full_likelihood_bhhh".
    n_bootstrap : int, default=400
        Number of pairs-cluster bootstrap replications when
        ``se_method="bootstrap"``.
    se_seed : int, optional
        Random seed for bootstrap standard errors.
    verbose : bool, default=False
        Whether to print progress messages during estimation.

    Attributes
    ----------
    params_ : dict
        Estimated parameters after fitting. Keys are parameter names
        (e.g., "theta_c", "RC") and values are point estimates.
    se_ : dict
        Standard errors for each parameter.
    coef_ : numpy.ndarray
        Coefficients as a numpy array (sklearn convention).
    log_likelihood_ : float
        Maximized log-likelihood value.
    pvalues_ : dict
        P-values for each parameter (from Wald t-test).
    policy_ : numpy.ndarray
        Estimated choice probabilities P(a|s) of shape (n_states, n_actions).
    value_ : numpy.ndarray
        Estimated value function V(s) of shape (n_states,).
    value_function_ : numpy.ndarray
        Alias for ``value_`` (backward compatibility).
    transitions_ : numpy.ndarray
        Original transition input. This is a two-dimensional keep matrix when
        transitions are estimated by the wrapper, or the supplied array.
    transition_tensor_ : numpy.ndarray
        Canonical action-specific transition tensor with shape
        ``(n_actions, n_states, n_states)``.
    transition_source_ : str
        Whether transitions were estimated from the fitted panel or supplied.
    converged_ : bool
        Whether the optimization converged.
    reward_spec_ : RewardSpec
        The reward specification used for estimation.

    Examples
    --------
    >>> from econirl.estimators import NFXP
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ...     "bus_id": [0, 0, 1, 1],
    ...     "mileage": [10, 20, 15, 30],
    ...     "replaced": [0, 0, 0, 1],
    ... })
    >>>
    >>> model = NFXP(n_states=90)
    >>> model.fit(df, state="mileage", action="replaced", id="bus_id")
    >>> print(model.params_)
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.9999,
        utility: RewardSpec | None = None,
        se_method: Literal[
            "asymptotic",
            "robust",
            "clustered",
            "bootstrap",
            "full_likelihood_bhhh",
        ] = "robust",
        n_bootstrap: int = 400,
        se_seed: int | None = None,
        verbose: bool = False,
    ):
        """Initialize the NFXP estimator.

        Parameters
        ----------
        n_states : int, default=90
            Number of discrete states.
        n_actions : int, default=2
            Number of discrete actions.
        discount : float, default=0.9999
            Time discount factor (beta).
        utility : RewardSpec
            Utility specification as a ``RewardSpec``.  For the classic Rust
            bus model, use ``rust_bus_reward_spec(n_states)`` from
            ``econirl.datasets``.
        se_method : str, default="robust"
            Method for computing standard errors.
        n_bootstrap : int, default=400
            Number of pairs-cluster bootstrap replications when
            ``se_method="bootstrap"``.
        se_seed : int, optional
            Random seed for bootstrap standard errors.
        verbose : bool, default=False
            Whether to print progress messages.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.utility = utility
        # White (1982), Cameron-Miller (2015), and Horowitz (2001) back the
        # robust, clustered, and bootstrap choices; asymptotic and the
        # full-likelihood BHHH replication profile follow Rust.
        self.se_method = se_method
        self.n_bootstrap = n_bootstrap
        self.se_seed = se_seed
        self.verbose = verbose

        # Fitted attributes (set after fit())
        self.params_: dict[str, float] | None = None
        self.se_: dict[str, float] | None = None
        self.transition_se_: dict[str, float] | None = None
        self.joint_se_: dict[str, float] | None = None
        self.joint_covariance_: np.ndarray | None = None
        self.joint_parameter_names_: list[str] | None = None
        self.pvalues_: dict[str, float] | None = None
        self.coef_: np.ndarray | None = None
        self.log_likelihood_: float | None = None
        self.value_function_: np.ndarray | None = None
        self.policy_: np.ndarray | None = None
        self.value_: np.ndarray | None = None
        self.transitions_: np.ndarray | None = None
        self.transition_tensor_: np.ndarray | None = None
        self.transition_source_: str | None = None
        self.converged_: bool | None = None
        self.reward_spec_: RewardSpec | None = None

        # Internal storage
        self._result = None
        self._panel = None
        self._utility_fn = None
        self._problem = None
        self._transition_probabilities = None
        self._transition_increments = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        transitions: np.ndarray | None = None,
        reward: RewardSpec | None = None,
    ) -> "NFXP":
        """Fit the NFXP estimator to data.

        Parameters
        ----------
        data : pandas.DataFrame or Panel or TrajectoryPanel
            Panel data with observations.  When a DataFrame is passed,
            ``state``, ``action``, and ``id`` column names are required.
            When a Panel/TrajectoryPanel is passed, column names are ignored.
        state : str, optional
            Column name for the state variable (required for DataFrame input).
        action : str, optional
            Column name for the action variable (required for DataFrame input).
        id : str, optional
            Column name for the individual identifier (required for DataFrame
            input).
        transitions : numpy.ndarray, optional
            Pre-estimated transition probabilities. Two forms are accepted.
            A 2D array of shape (n_states, n_states) is the keep-kernel for
            action 0; the replace action is built by resetting to state 0
            (the Rust reset-on-replace default). A 3D array of shape
            (n_actions, n_states, n_states) is a full per-action transition
            tensor and is used as given, for action-dependent transitions.
            If None, transitions are estimated from the data.
        reward : RewardSpec, optional
            Reward/utility specification.  If provided, overrides the
            ``utility`` parameter passed at construction time.

        Returns
        -------
        self : NFXP
            Returns self for method chaining.
        """
        # Resolve reward spec: explicit argument > constructor parameter
        reward_spec = reward if reward is not None else self.utility

        # --- Handle data: DataFrame or Panel/TrajectoryPanel ---
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required when data is a DataFrame"
                )
            self._panel = TrajectoryPanel.from_dataframe(data, state=state, action=action, id=id)
            self._validate_dataframe(data, state=state, action=action)
        elif isinstance(data, (Panel, TrajectoryPanel)):
            self._panel = data
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, got {type(data)}"
            )

        # --- Handle reward: RewardSpec ---
        if isinstance(reward_spec, RewardSpec):
            self.reward_spec_ = reward_spec
            self._utility_fn = reward_spec.to_linear_utility()
        else:
            raise ValueError(
                "utility must be a RewardSpec; the 'linear_cost' preset was "
                "removed. Build features explicitly, e.g. "
                "rust_bus_reward_spec(n_states) from econirl.datasets for "
                "the Rust bus."
            )

        # Validate the feature matrix matches the declared state/action space so a
        # mismatch raises a clear error instead of a cryptic broadcasting TypeError.
        feat = np.asarray(self._utility_fn.feature_matrix)
        if feat.ndim != 3 or feat.shape[0] != self.n_states or feat.shape[1] != self.n_actions:
            raise ValueError(
                f"reward/feature matrix has shape {feat.shape}; expected "
                f"(n_states={self.n_states}, n_actions={self.n_actions}, n_features). "
                "Check that n_states and n_actions match your RewardSpec features."
            )

        # Estimate transitions if not provided
        if transitions is None:
            trans_estimator = TransitionEstimator(
                n_states=self.n_states,
                max_increase=2,
            )
            trans_estimator.fit(self._panel)
            # First stage. The standard errors condition on this transition
            # estimate; the two-step correction for its estimation error
            # (Rust 1994 p.3109; Amemiya 1976) is omitted, usually negligible.
            self.transitions_ = trans_estimator.matrix_
            self._transition_probabilities = trans_estimator.probs_
            self._transition_increments = trans_estimator.transition_increments_
            self.transition_source_ = "estimated from fitted panel"
        else:
            if self.se_method == "full_likelihood_bhhh":
                raise ValueError(
                    "se_method='full_likelihood_bhhh' requires transitions to be "
                    "estimated from the fitted panel by NFXP.fit(). Use the core "
                    "NFXPEstimator API to pass transition_probabilities and "
                    "transition_increments with an explicit transition matrix."
                )
            self.transitions_ = np.asarray(transitions)
            self._transition_probabilities = None
            self._transition_increments = None
            self.transition_source_ = (
                "supplied action-specific tensor"
                if self.transitions_.ndim == 3
                else "supplied keep-transition matrix"
            )

        # Build full transition matrices (for both actions)
        # Action 0 (keep): use estimated transitions
        # Action 1 (replace): reset to state 0, then apply transition
        transition_tensor = self._build_transition_tensor(self.transitions_)
        self.transition_tensor_ = transition_tensor

        # Create problem specification
        self._problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )

        # Create the underlying NFXP estimator
        estimator = NFXPEstimator(
            se_method=self.se_method,
            verbose=self.verbose,
        )

        # Run estimation
        estimate_kwargs = {
            "n_bootstrap": self.n_bootstrap,
            "se_seed": self.se_seed,
            "transition_source": self.transition_source_,
        }
        if self.se_method == "full_likelihood_bhhh":
            estimate_kwargs.update(
                {
                    "transition_probabilities": self._transition_probabilities,
                    "transition_increments": self._transition_increments,
                }
            )

        self._result = estimator.estimate(
            panel=self._panel,
            utility=self._utility_fn,
            problem=self._problem,
            transitions=transition_tensor,
            **estimate_kwargs,
        )

        # Extract results
        self._extract_results()

        return self

    def _validate_dataframe(
        self,
        data: pd.DataFrame,
        state: str,
        action: str,
    ) -> None:
        """Validate the state and action columns of a DataFrame before fitting.

        Raises a clear ``ValueError`` on NaN entries, non-integer or negative
        actions, actions outside ``[0, n_actions)``, or states outside
        ``[0, n_states)``.  Catches malformed input that would otherwise be
        silently coerced to integers and produce meaningless estimates.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data.
        state : str
            Column name for the state variable.
        action : str
            Column name for the action variable.
        """
        state_col = data[state]
        action_col = data[action]

        # NaN in either column.
        if state_col.isna().any():
            raise ValueError(f"state column '{state}' contains NaN values")
        if action_col.isna().any():
            raise ValueError(f"action column '{action}' contains NaN values")

        # Actions must be non-negative integers (reject float-coded like 0.5).
        action_values = np.asarray(action_col.values)
        non_integer = action_values != np.floor(action_values)
        if non_integer.any():
            bad = action_values[non_integer][0]
            raise ValueError(
                f"action column '{action}' contains non-integer value {bad}; "
                "actions must be non-negative integers"
            )
        action_int = action_values.astype(np.int64)
        if (action_int < 0).any():
            bad = action_int[action_int < 0][0]
            raise ValueError(
                f"action column '{action}' contains negative value {bad}; "
                "actions must be non-negative integers"
            )

        # Actions in range [0, n_actions).
        out_of_range = action_int >= self.n_actions
        if out_of_range.any():
            bad = action_int[out_of_range][0]
            raise ValueError(
                f"action column contains out-of-range value {bad}; "
                f"n_actions={self.n_actions} expects actions in "
                f"[0, {self.n_actions})"
            )

        # States in range [0, n_states).
        state_values = np.asarray(state_col.values)
        state_int = state_values.astype(np.int64)
        state_oob = (state_int < 0) | (state_int >= self.n_states)
        if state_oob.any():
            bad = state_int[state_oob][0]
            raise ValueError(
                f"state column contains out-of-range value {bad}; "
                f"n_states={self.n_states} expects states in "
                f"[0, {self.n_states})"
            )

    def _dataframe_to_panel(
        self,
        data: pd.DataFrame,
        state: str,
        action: str,
        id: str,
    ) -> Panel:
        """Convert pandas DataFrame to Panel object.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data.
        state : str
            Column name for state.
        action : str
            Column name for action.
        id : str
            Column name for individual id.

        Returns
        -------
        Panel
            Panel object suitable for estimation.
        """
        trajectories = []

        # Group by individual
        for ind_id, group in data.groupby(id, sort=True):
            # Sort by any time/period column if available, otherwise keep order
            sorted_group = group.sort_index()

            states = sorted_group[state].values.astype(np.int64)
            actions = sorted_group[action].values.astype(np.int64)

            # Compute next states
            # For the last observation, we need to handle it specially
            # We'll use the "wrap around" convention or just use the current state
            next_states = np.zeros_like(states)
            next_states[:-1] = states[1:]
            # For the last observation, apply transition logic based on action
            if len(states) > 0:
                last_state = states[-1]
                last_action = actions[-1]
                if last_action == 1:  # Replace -> reset to 0
                    next_states[-1] = 0
                else:  # Keep -> stay at same state (conservative)
                    next_states[-1] = min(last_state + 1, self.n_states - 1)

            traj = Trajectory(
                states=np.array(states, dtype=np.int64),
                actions=np.array(actions, dtype=np.int64),
                next_states=np.array(next_states, dtype=np.int64),
                individual_id=ind_id,
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    def _build_transition_tensor(self, keep_transitions: np.ndarray) -> np.ndarray:
        """Build full transition tensor for both actions.

        Parameters
        ----------
        keep_transitions : numpy.ndarray
            Either a transition matrix for action=0 (keep) of shape
            (n_states, n_states), or a pre-built full tensor of shape
            (n_actions, n_states, n_states).  When the 3-D form is supplied
            it is validated and returned unchanged, allowing callers to pass
            an arbitrary multi-action transition tensor directly.

        Returns
        -------
        numpy.ndarray
            Transition tensor of shape (n_actions, n_states, n_states).
        """
        keep_transitions = np.asarray(keep_transitions, dtype=np.float32)
        if keep_transitions.ndim == 3:
            expected_shape = (self.n_actions, self.n_states, self.n_states)
            if keep_transitions.shape != expected_shape:
                raise ValueError(
                    f"3D transitions must have shape {expected_shape}, got {keep_transitions.shape}"
                )
            self._validate_transition_rows(keep_transitions)
            return keep_transitions

        if keep_transitions.ndim != 2:
            raise ValueError(
                "transitions must be a 2D keep matrix or a 3D action-specific tensor"
            )
        if self.n_actions != 2:
            raise ValueError(
                "a 2D keep-transition matrix is only defined for n_actions=2; "
                "supply a full (n_actions, n_states, n_states) tensor"
            )
        expected_matrix_shape = (self.n_states, self.n_states)
        if keep_transitions.shape != expected_matrix_shape:
            raise ValueError(
                f"2D transitions must have shape {expected_matrix_shape}, "
                f"got {keep_transitions.shape}"
            )
        self._validate_transition_rows(keep_transitions)

        n = self.n_states
        transitions = np.zeros((self.n_actions, n, n), dtype=np.float32)

        # Action 0 (keep): use provided transitions
        transitions[0] = np.array(keep_transitions, dtype=np.float32)

        # Action 1 (replace): reset to state 0, then transition
        # After replacement, start at state 0 and apply the same transition
        # The row for state 0 in keep_transitions tells us where we go from 0
        for s in range(n):
            # After replacement from any state, we transition as if from state 0
            transitions[1, s, :] = transitions[0, 0, :]

        self._validate_transition_rows(transitions)
        return transitions

    @staticmethod
    def _validate_transition_rows(transitions: np.ndarray) -> None:
        """Validate finite, non-negative probability rows."""
        if not np.isfinite(transitions).all():
            raise ValueError("transitions must contain only finite probabilities")
        if (transitions < 0).any():
            raise ValueError("transitions must contain non-negative probabilities")
        row_sums = transitions.sum(axis=-1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            max_error = float(np.max(np.abs(row_sums - 1.0)))
            raise ValueError(
                "transition rows must sum to 1; "
                f"maximum absolute row-sum error is {max_error:.3g}"
            )

    def _extract_results(self) -> None:
        """Extract results from estimation into sklearn-style attributes."""
        if self._result is None:
            return

        # Parameter estimates
        params = np.asarray(self._result.parameters)
        param_names = self._result.parameter_names

        self.params_ = {name: float(val) for name, val in zip(param_names, params)}
        self.coef_ = params.copy()

        # Standard errors
        se = np.asarray(self._result.standard_errors)
        self.se_ = {name: float(val) for name, val in zip(param_names, se)}

        self.transition_se_ = None
        self.joint_se_ = None
        self.joint_covariance_ = None
        self.joint_parameter_names_ = None
        full_metadata = self._result.metadata.get("full_likelihood_bhhh")
        se_details = self._result.metadata.get("se_details", {})
        if full_metadata is not None:
            joint_names = list(full_metadata["joint_parameter_names"])
            joint_se_raw = se_details.get("joint_standard_errors")
            joint_cov_raw = se_details.get("joint_variance_covariance")
            if joint_se_raw is not None:
                joint_se = np.asarray(joint_se_raw, dtype=float)
            else:
                joint_se = np.array([])
            if joint_cov_raw is not None:
                joint_cov = np.asarray(joint_cov_raw, dtype=float)
            else:
                joint_cov = np.array([])
            if joint_se.size == len(joint_names):
                self.joint_parameter_names_ = joint_names
                self.joint_se_ = {name: float(val) for name, val in zip(joint_names, joint_se)}
                self.transition_se_ = {
                    name: self.joint_se_[name] for name in full_metadata["transition_score_columns"]
                }
            if joint_cov.shape == (len(joint_names), len(joint_names)):
                self.joint_covariance_ = joint_cov

        # Other attributes
        self.log_likelihood_ = float(self._result.log_likelihood)
        self.converged_ = bool(self._result.converged)

        if not self.converged_:
            warnings.warn(
                "NFXP optimization did not converge; parameter estimates and "
                "standard errors may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )

        if self._result.value_function is not None:
            self.value_function_ = np.asarray(self._result.value_function)
            self.value_ = self.value_function_

        # Policy matrix
        if self._result.policy is not None:
            self.policy_ = np.asarray(self._result.policy)

        # p-values from t-statistics
        if self.se_ is not None:
            pvalues: dict[str, float] = {}
            for name in self.params_:
                se_val = self.se_[name]
                if se_val and se_val > 0:
                    t_stat = self.params_[name] / se_val
                    pvalues[name] = float(2 * (1 - scipy_norm.cdf(abs(t_stat))))
                else:
                    pvalues[name] = float("nan")
            self.pvalues_ = pvalues

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        """Structural reward matrix R(s,a) of shape (n_states, n_actions).

        Computes the utility matrix from the fitted parameters and the
        feature specification. Returns None if the model has not been fitted.
        """
        if self.params_ is None or self._utility_fn is None or self._result is None:
            return None
        param_names = self._result.parameter_names
        param_vector = np.array(
            [self.params_[name] for name in param_names],
            dtype=np.float32,
        )
        utility_matrix = self._utility_fn.compute(param_vector)
        return np.asarray(utility_matrix)

    def summary(self, alpha: float = 0.05) -> str:
        """Generate a formatted summary of estimation results.

        Returns
        -------
        str
            Human-readable summary of the estimation.
        """
        if self._result is None:
            return "NFXP: Not fitted yet. Call fit() first."

        return self._result.summary(alpha=alpha)

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

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        """Predict choice probabilities for given states.

        Parameters
        ----------
        states : numpy.ndarray
            Array of state indices.

        Returns
        -------
        numpy.ndarray
            Choice probabilities of shape (len(states), n_actions).
            Each row sums to 1.
        """
        if self._result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        states = np.asarray(states, dtype=np.int64)

        # Get policy (choice probabilities) from result
        policy = np.asarray(self._result.policy)

        # Index into the policy for the requested states
        proba = policy[states]

        return proba

    def simulate(
        self,
        n_agents: int,
        n_periods: int,
        seed: int | None = None,
        init_states: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Simulate choices under the estimated policy.

        Generates synthetic data by simulating agents making decisions
        according to the fitted model. Each agent starts from ``init_states``
        (state 0 for everyone by default) and evolves according to the
        estimated transition probabilities and choice probabilities.

        Parameters
        ----------
        n_agents : int
            Number of agents to simulate.
        n_periods : int
            Number of time periods per agent.
        seed : int, optional
            Random seed for reproducibility.
        init_states : numpy.ndarray, optional
            Initial states. None starts every agent at state 0. A length
            ``n_states`` vector is treated as a start distribution and sampled.
            A length ``n_agents`` array sets each agent's start state directly.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:
            - agent_id: Identifier for each agent (0 to n_agents-1)
            - period: Time period (0 to n_periods-1)
            - state: State at the beginning of the period
            - action: Action taken (sampled from estimated policy)

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = NFXP(n_states=90)
        >>> model.fit(data, state="mileage", action="replaced", id="bus_id")
        >>> sim_data = model.simulate(n_agents=100, n_periods=50, seed=42)
        >>> print(sim_data.head())
        """
        if self._result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        rng = np.random.default_rng(seed)

        # Resolve per-agent initial states. None keeps the historical behaviour
        # (everyone starts at state 0); a length-n_states vector is a start
        # distribution; a length-n_agents vector is explicit per-agent states.
        if init_states is None:
            initial = np.zeros(n_agents, dtype=int)
        else:
            init_states = np.asarray(init_states)
            if init_states.shape == (self.n_states,):
                initial = rng.choice(self.n_states, size=n_agents, p=init_states)
            elif init_states.shape == (n_agents,):
                initial = init_states.astype(int)
            else:
                raise ValueError(
                    f"init_states must be a length-{self.n_states} distribution "
                    f"or a length-{n_agents} array of state indices; "
                    f"got shape {init_states.shape}"
                )

        # Get the policy (choice probabilities)
        policy = np.asarray(self._result.policy)  # shape: (n_states, n_actions)

        transition_tensor = self.transition_tensor_
        if transition_tensor is None:
            transition_tensor = self._build_transition_tensor(self.transitions_)

        # Storage for results
        data = []

        for agent_id in range(n_agents):
            state = int(initial[agent_id])

            for period in range(n_periods):
                # Sample action from policy
                action_probs = policy[state]
                action = rng.choice(self.n_actions, p=action_probs)

                # Record observation
                data.append(
                    {
                        "agent_id": agent_id,
                        "period": period,
                        "state": state,
                        "action": action,
                    }
                )

                trans_probs = transition_tensor[action, state]
                state = rng.choice(self.n_states, p=trans_probs)

        return pd.DataFrame(data)

    def counterfactual(
        self,
        transitions: np.ndarray | None = None,
        **param_changes,
    ) -> CounterfactualResult:
        """Compute outcomes under different parameter values.

        Performs counterfactual analysis by solving the dynamic programming
        problem under alternative parameter values. This enables "what if"
        questions like "what would the policy be if RC was 15 instead of 10?"

        Parameters
        ----------
        transitions : numpy.ndarray, optional
            Alternative transition matrix or action-specific transition tensor.
            When supplied, reward parameters remain fixed.
        **param_changes : float
            Keyword arguments specifying parameter changes.
            Keys must be valid parameter names (e.g., "theta_c", "RC").
            Values are the counterfactual parameter values.

        Returns
        -------
        CounterfactualResult
            Object containing:
            - params: Dictionary of all parameter values used
            - value_function: V(s) under new parameters
            - policy: P(a|s) under new parameters

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If an unknown parameter name is provided.

        Examples
        --------
        >>> model = NFXP(n_states=90)
        >>> model.fit(data, state="mileage", action="replaced", id="bus_id")
        >>>
        >>> # What if replacement cost was higher?
        >>> cf = model.counterfactual(RC=15.0)
        >>> print(f"Original RC: {model.params_['RC']:.2f}")
        >>> print(f"Counterfactual RC: {cf.params['RC']:.2f}")
        >>> print(f"P(replace|state=50) changes from "
        ...       f"{model.predict_proba(np.array([50]))[0,1]:.3f} to "
        ...       f"{cf.policy[50,1]:.3f}")
        >>>
        >>> # Multiple parameter changes
        >>> cf2 = model.counterfactual(RC=15.0, theta_c=0.05)
        """
        if self._result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if transitions is not None and param_changes:
            raise ValueError(
                "change reward parameters or transitions in one counterfactual, not both"
            )

        # Check for invalid parameter names
        valid_params = set(self.params_.keys())
        for param_name in param_changes:
            if param_name not in valid_params:
                raise ValueError(
                    f"Unknown parameter '{param_name}'. "
                    f"Valid parameters are: {sorted(valid_params)}"
                )

        # Build counterfactual parameter dictionary
        cf_params = self.params_.copy()
        cf_params.update(param_changes)

        baseline_tensor = self.transition_tensor_
        if baseline_tensor is None:
            baseline_tensor = self._build_transition_tensor(self.transitions_)

        if transitions is not None:
            new_tensor = self._build_transition_tensor(transitions)
            result = counterfactual_transitions(
                result=self._result,
                new_transitions=new_tensor,
                utility=self._utility_fn,
                problem=self._problem,
                baseline_transitions=baseline_tensor,
            )
        else:
            result = counterfactual_policy(
                result=self._result,
                new_parameters=cf_params,
                utility=self._utility_fn,
                problem=self._problem,
                transitions=baseline_tensor,
            )
        result.params = cf_params
        return result

    def __repr__(self) -> str:
        if self.params_ is not None:
            return (
                f"NFXP(n_states={self.n_states}, n_actions={self.n_actions}, "
                f"discount={self.discount}, fitted=True)"
            )
        return (
            f"NFXP(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, fitted=False)"
        )
