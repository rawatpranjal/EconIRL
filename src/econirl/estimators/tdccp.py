"""Sklearn-style TD-CCP estimator for dynamic discrete choice models.

This module provides a TDCCP class with a scikit-learn style API that wraps
the underlying TDCCPEstimator from econirl.estimation.td_ccp. It accepts pandas
DataFrames with column names instead of the low-level Panel API.

TD-CCP (Temporal-Difference CCP) estimates recursive utility components from
observed state-action transitions. The default linear semi-gradient method
uses a closed-form TD solve. Approximate value iteration is available with
neural networks or a gradient-boosting regressor.

After fitting, ``ev_features_`` shows how much of the continuation value comes
from each structural feature.

Example:
    >>> from econirl.estimators import TDCCP
    >>> import pandas as pd
    >>>
    >>> # Load bus replacement data
    >>> df = pd.read_csv("zurcher_bus.csv")
    >>>
    >>> # Create estimator and fit
    >>> from econirl.datasets import rust_bus_reward_spec
    >>> model = TDCCP(n_states=90, discount=0.9999, utility=rust_bus_reward_spec(90))
    >>> model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> # Access results sklearn-style
    >>> print(model.params_)        # {"operating_cost": 0.001, "replacement_cost": 9.35}
    >>> print(model.summary())
    >>>
    >>> # Interpretable EV decomposition
    >>> print(model.ev_features_)   # shape (n_states, n_features)
    >>>
    >>> # Predict choice probabilities
    >>> proba = model.predict_proba(states=np.array([0, 10, 50]))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import DDCProblem, Panel, TrajectoryPanel
from econirl.estimation.td_ccp import TDCCPConfig, TDCCPEstimator
from econirl.estimators.nfxp import NFXP
from econirl.preprocessing.diagnostics import feature_diagnostics
from econirl.transitions import TransitionEstimator


@dataclass(frozen=True)
class _NormalizedStateEncoder:
    """Pickle-safe scalar state encoder used by the default wrapper path."""

    n_states: int

    def __call__(self, states: jnp.ndarray) -> jnp.ndarray:
        values = jnp.asarray(states, dtype=jnp.float32)
        return jnp.expand_dims(values / max(self.n_states - 1, 1), axis=-1)


@dataclass(frozen=True)
class _ArrayStateEncoder:
    """Pickle-safe lookup encoder for user-supplied state features."""

    features: np.ndarray

    def __call__(self, states: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(self.features)[jnp.asarray(states, dtype=jnp.int32)]


class TDCCP(NFXP):
    """Sklearn-style TD-CCP estimator for dynamic discrete choice models.

    TD-CCP (Temporal-Difference CCP) estimates utility parameters after
    learning recursive terms directly from observed transitions. The default
    method is the paper's linear semi-gradient solve. Neural and generic
    regression AVI are optional alternatives.

    This is particularly useful for large state spaces where matrix-based
    CCP methods are computationally infeasible.

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
        Method for computing standard errors. Options: "robust", "asymptotic",
        "bootstrap".
    n_bootstrap : int, default=400
        Number of pairs-cluster bootstrap replications when
        ``se_method="bootstrap"``.
    se_seed : int, optional
        Random seed for bootstrap standard errors.
    hidden_dim : int, default=64
        Number of hidden units per layer in the EV component networks.
    num_hidden_layers : int, default=2
        Number of hidden layers in the EV component networks.
    avi_iterations : int, default=20
        Number of approximate value iteration rounds.
    epochs_per_avi : int, default=30
        Number of SGD epochs per AVI iteration.
    learning_rate : float, default=1e-3
        Learning rate for neural network training.
    batch_size : int, default=8192
        Mini-batch size for SGD training.
    n_policy_iterations : int, default=3
        Number of NPL-style policy iterations.
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
    ev_features_ : numpy.ndarray or None
        Per-feature EV component values of shape (n_states, n_features).
        Shows how much of the continuation value comes from each structural
        feature. Available after fitting if the underlying estimator
        includes them in metadata.
    converged_ : bool
        Whether the optimization converged.
    reward_spec_ : RewardSpec
        The reward specification used for estimation.

    Examples
    --------
    >>> from econirl.estimators import TDCCP
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ...     "bus_id": [0, 0, 1, 1],
    ...     "mileage": [10, 20, 15, 30],
    ...     "replaced": [0, 0, 0, 1],
    ... })
    >>>
    >>> from econirl.datasets import rust_bus_reward_spec
    >>> model = TDCCP(n_states=90, utility=rust_bus_reward_spec(90))
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.9999,
        utility: RewardSpec | None = None,
        se_method: Literal["robust", "asymptotic", "bootstrap"] = "robust",
        n_bootstrap: int = 400,
        se_seed: int | None = None,
        seed: int | None = None,
        # Method selection (new: Adusumilli-Eckardt 2025)
        method: Literal["semigradient", "neural"] = "semigradient",
        cross_fitting: bool = True,
        robust_se: bool = True,
        # Semi-gradient specific
        basis_dim: int = 8,
        basis_type: Literal["polynomial", "encoded", "tabular"] = "polynomial",
        basis_include_rewards: bool = False,
        basis_ridge: float = 1e-8,
        basis_pinv_rcond: float | None = None,
        basis_action_coding: Literal["separate", "reference"] = "separate",
        # Neural AVI specific
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        avi_iterations: int = 20,
        avi_early_stop_tol: float = 0.01,
        epochs_per_avi: int = 30,
        learning_rate: float = 1e-3,
        batch_size: int = 8192,
        avi_functional_class: Literal["neural", "gbm"] = "neural",
        avi_regressor: Any = None,
        # CCP estimation
        ccp_method: Literal["frequency", "logit"] = "frequency",
        ccp_smoothing: float = 0.01,
        ccp_poly_degree: int = 3,
        ccp_use_encoder: bool = False,
        # NPL iteration (not in paper, optional)
        n_policy_iterations: int = 1,
        split_unit: Literal["individual", "row"] = "individual",
        cross_fit_shuffle: bool = True,
        cross_fit_ccp: bool = True,
        linear_robust_correction: Literal["sensitivity", "backward"] = "sensitivity",
        outer_max_iter: int = 200,
        outer_tol: float = 1e-6,
        theta_l2_penalty: float = 0.0,
        compute_policy: bool = True,
        state_features: np.ndarray | None = None,
        verbose: bool = False,
    ):
        """Initialize the TD-CCP estimator.

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
        method : str, default="semigradient"
            TD method: "semigradient" (fast closed-form, eq 3.5) or
            "neural" (AVI with neural networks, Algorithm 1).
        cross_fitting : bool, default=True
            Use 2-fold cross-fitting (Algorithm 2) for valid inference.
        robust_se : bool, default=True
            Compute locally robust standard errors (Section 4).
        basis_dim : int, default=8
            Number of polynomial basis functions for semi-gradient method.
        basis_type : str, default="polynomial"
            Semi-gradient basis: "polynomial", "encoded", or "tabular".
        basis_include_rewards : bool, default=False
            Include reward features in the encoded semi-gradient basis.
        basis_ridge : float, default=1e-8
            Ridge stabilization for the semi-gradient normal equation.
        basis_pinv_rcond : float, optional
            Pseudoinverse cutoff for nearly singular semi-gradient bases.
        basis_action_coding : str, default="separate"
            Use separate action bases or the paper's binary reference coding.
        hidden_dim : int, default=64
            Hidden units per layer in EV component networks.
        num_hidden_layers : int, default=2
            Number of hidden layers in EV component networks.
        avi_iterations : int, default=20
            Number of approximate value iteration rounds.
        avi_early_stop_tol : float, default=0.01
            Relative-change stopping tolerance for approximate value iteration.
        epochs_per_avi : int, default=30
            Number of SGD epochs per AVI iteration.
        learning_rate : float, default=1e-3
            Learning rate for neural network training.
        batch_size : int, default=8192
            Mini-batch size for SGD training.
        avi_functional_class : str, default="neural"
            Approximation class for AVI. Options are "neural" and "gbm".
        avi_regressor : object, optional
            Custom object or factory with ``fit`` and ``predict`` methods.
        ccp_method : str, default="frequency"
            CCP estimation: "frequency" or "logit".
        ccp_smoothing : float, default=0.01
            Additive smoothing for frequency CCPs.
        ccp_poly_degree : int, default=3
            Polynomial degree for logit CCPs.
        ccp_use_encoder : bool, default=False
            Build logit CCP features from ``state_features``.
        n_policy_iterations : int, default=1
            Number of NPL-style policy iterations. Paper uses 1.
        split_unit : str, default="individual"
            Cross-fitting unit. Individual splitting is the panel-data default.
        cross_fit_shuffle : bool, default=True
            Shuffle split units before assigning the two folds.
        cross_fit_ccp : bool, default=True
            Estimate first-stage CCPs separately within each training fold.
        linear_robust_correction : str, default="sensitivity"
            Locally robust correction for linear semi-gradient fits.
        outer_max_iter : int, default=200
            Maximum iterations for the structural parameter optimizer.
        outer_tol : float, default=1e-6
            Structural parameter optimizer tolerance.
        theta_l2_penalty : float, default=0.0
            Optional L2 penalty on structural parameters.
        compute_policy : bool, default=True
            Solve and store dynamic policy and value objects after estimation.
        state_features : numpy.ndarray, optional
            Encoded state coordinates with shape ``(n_states, n_features)``.
        verbose : bool, default=False
            Whether to print progress messages.
        """
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            discount=discount,
            utility=utility,
            se_method=se_method,
            n_bootstrap=n_bootstrap,
            se_seed=se_seed,
            verbose=verbose,
            seed=seed,
        )
        self.method = method
        self.cross_fitting = cross_fitting
        self.robust_se = robust_se
        self.basis_dim = basis_dim
        self.basis_type = basis_type
        self.basis_include_rewards = basis_include_rewards
        self.basis_ridge = basis_ridge
        self.basis_pinv_rcond = basis_pinv_rcond
        self.basis_action_coding = basis_action_coding
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.avi_iterations = avi_iterations
        self.avi_early_stop_tol = avi_early_stop_tol
        self.epochs_per_avi = epochs_per_avi
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.avi_functional_class = avi_functional_class
        self.avi_regressor = avi_regressor
        self.ccp_method = ccp_method
        self.ccp_smoothing = ccp_smoothing
        self.ccp_poly_degree = ccp_poly_degree
        self.ccp_use_encoder = ccp_use_encoder
        self.n_policy_iterations = n_policy_iterations
        self.split_unit = split_unit
        self.cross_fit_shuffle = cross_fit_shuffle
        self.cross_fit_ccp = cross_fit_ccp
        self.linear_robust_correction = linear_robust_correction
        self.outer_max_iter = outer_max_iter
        self.outer_tol = outer_tol
        self.theta_l2_penalty = theta_l2_penalty
        self.compute_policy = compute_policy
        self.state_features = None if state_features is None else np.asarray(state_features)
        self._reset_fit_state()

    def _reset_fit_state(self) -> None:
        """Reset shared and TD-CCP-specific fitted fields."""
        super()._reset_fit_state()
        self.termination_reason_: str | None = None
        self.failure_reason_: str | None = None
        self.ev_features_: np.ndarray | None = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        *,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        transitions: np.ndarray | None = None,
        features: np.ndarray | None = None,
        context: Any | None = None,
        reward: RewardSpec | None = None,
    ) -> "TDCCP":
        """Fit the TD-CCP estimator to data.

        Parameters
        ----------
        data : pandas.DataFrame or Panel or TrajectoryPanel
            Panel data with observations. When a DataFrame is passed,
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
            Action-specific transition tensor with shape
            ``(n_actions, n_states, n_states)``. If None, transitions are
            estimated from the data for post-fit dynamic calculations.
        reward : RewardSpec, optional
            Reward/utility specification. If provided, overrides the
            ``utility`` parameter passed at construction time.

        Returns
        -------
        self : TDCCP
            Returns self for method chaining.
        """
        self._reset_fit_state()
        if features is not None:
            raise ValueError(
                "TD-CCP linear reward features must be supplied through RewardSpec, "
                "using utility= at construction or reward= in fit()"
            )
        if context is not None:
            raise ValueError("TD-CCP does not use context; omit context=")

        # Resolve reward spec: explicit argument > constructor parameter
        reward_spec = reward if reward is not None else self.utility

        # --- Handle data: DataFrame or Panel/TrajectoryPanel ---
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required when data is a DataFrame"
                )
            missing = [name for name in (state, action, id) if name not in data.columns]
            if missing:
                raise ValueError(f"data is missing required columns: {missing}")
            self._validate_dataframe(data, state=state, action=action, id=id)
            self._panel = TrajectoryPanel.from_dataframe(data, state=state, action=action, id=id)
        elif isinstance(data, (Panel, TrajectoryPanel)):
            self._panel = data
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, got {type(data)}"
            )
        self._validate_panel_support()

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

        utility_fn = self._utility_fn
        if utility_fn is None:
            raise RuntimeError("TD-CCP utility initialization failed")
        feat = np.asarray(utility_fn.feature_matrix)
        expected_prefix = (self.n_states, self.n_actions)
        if feat.ndim != 3 or feat.shape[:2] != expected_prefix:
            raise ValueError(
                f"reward/feature matrix has shape {feat.shape}; expected "
                f"(n_states={self.n_states}, n_actions={self.n_actions}, n_features). "
                "Check that n_states and n_actions match your RewardSpec features."
            )
        rank_diagnostics = feature_diagnostics(feat)
        n_features = int(rank_diagnostics["num_features"])
        if int(rank_diagnostics["feature_rank"]) < n_features:
            raise ValueError("TD-CCP reward design is rank deficient")
        if int(rank_diagnostics["contrast_rank"]) < n_features:
            raise ValueError("TD-CCP reward design is not identified from choices")

        # Estimate transitions if not provided
        if transitions is None:
            trans_estimator = TransitionEstimator(
                n_states=self.n_states,
                max_increase=2,
            )
            trans_estimator.fit(self._panel)
            transition_tensor = NFXP._build_transition_tensor(self, trans_estimator.matrix_)
            self.transition_source_ = "estimated from fitted panel"
        else:
            transition_tensor = self._build_transition_tensor(transitions)
            self.transition_source_ = "supplied action-specific tensor"
        self.transitions_ = np.asarray(transition_tensor)
        self.transition_tensor_ = np.asarray(transition_tensor)
        self.diagnostics_ = self._contract_diagnostics(
            feat,
            self.transition_tensor_,
            rank_diagnostics,
        )

        state_encoder: _NormalizedStateEncoder | _ArrayStateEncoder
        if self.state_features is None:
            state_encoder = _NormalizedStateEncoder(self.n_states)
            state_dim = 1
        else:
            state_features = np.asarray(self.state_features, dtype=np.float64)
            if state_features.ndim != 2 or state_features.shape[0] != self.n_states:
                raise ValueError("state_features must have shape (n_states, n_state_features)")
            if not np.isfinite(state_features).all():
                raise ValueError("state_features must contain only finite values")
            state_encoder = _ArrayStateEncoder(state_features)
            state_dim = int(state_features.shape[1])

        # Create problem specification with an explicit state encoder.
        self._problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
            state_dim=state_dim,
            state_encoder=state_encoder,
        )

        # Create the underlying TD-CCP estimator with all config options
        config = TDCCPConfig(
            method=self.method,
            basis_dim=self.basis_dim,
            basis_type=self.basis_type,
            basis_include_rewards=self.basis_include_rewards,
            basis_ridge=self.basis_ridge,
            basis_pinv_rcond=self.basis_pinv_rcond,
            basis_action_coding=self.basis_action_coding,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            avi_iterations=self.avi_iterations,
            avi_early_stop_tol=self.avi_early_stop_tol,
            epochs_per_avi=self.epochs_per_avi,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            avi_functional_class=self.avi_functional_class,
            avi_regressor=self.avi_regressor,
            ccp_method=self.ccp_method,
            ccp_smoothing=self.ccp_smoothing,
            ccp_poly_degree=self.ccp_poly_degree,
            ccp_use_encoder=self.ccp_use_encoder,
            cross_fitting=self.cross_fitting,
            split_unit=self.split_unit,
            cross_fit_shuffle=self.cross_fit_shuffle,
            cross_fit_ccp=self.cross_fit_ccp,
            linear_robust_correction=self.linear_robust_correction,
            robust_se=self.robust_se,
            n_policy_iterations=self.n_policy_iterations,
            outer_max_iter=self.outer_max_iter,
            outer_tol=self.outer_tol,
            theta_l2_penalty=self.theta_l2_penalty,
            compute_se=True,
            compute_policy=self.compute_policy,
            verbose=self.verbose,
        )
        if self.se_method == "robust" and not self.robust_se:
            raise ValueError("se_method='robust' requires robust_se=True")
        effective_se_method = "asymptotic" if self.se_method == "robust" else self.se_method
        estimator = TDCCPEstimator(
            config=config,
            se_method=effective_se_method,
            seed=0 if self.seed is None else self.seed,
        )

        # Run estimation
        try:
            self._result = estimator.estimate(
                panel=self._panel,
                utility=self._utility_fn,
                problem=self._problem,
                transitions=transition_tensor,
                n_bootstrap=self.n_bootstrap,
                se_seed=self.se_seed,
                transition_source=self.transition_source_,
            )
        except Exception as exc:
            self.termination_reason_ = "execution_failure"
            self.failure_reason_ = f"{type(exc).__name__}: {exc}"
            raise RuntimeError("TD-CCP estimation failed during optimization") from exc
        self._result.metadata["se_method"] = self.se_method

        # Extract results
        self._extract_results()

        return self

    def _build_transition_tensor(self, transitions: np.ndarray) -> np.ndarray:
        """Validate a canonical action-specific transition tensor.

        Parameters
        ----------
        transitions : numpy.ndarray
            Action-specific probabilities with shape
            ``(n_actions, n_states, n_states)``. TD-CCP uses the canonical
            tensor because its transition-free parameter stage does not imply
            a reset-on-action transition law for prediction or counterfactuals.

        Returns
        -------
        numpy.ndarray
            Transition tensor of shape (n_actions, n_states, n_states).
        """
        tensor = np.asarray(transitions, dtype=np.float32)
        expected_shape = (self.n_actions, self.n_states, self.n_states)
        if tensor.ndim != 3:
            raise ValueError(
                "TD-CCP transitions must be a 3D action-specific tensor with "
                f"shape {expected_shape}; got shape {tensor.shape}"
            )
        if tensor.shape != expected_shape:
            raise ValueError(f"3D transitions must have shape {expected_shape}, got {tensor.shape}")
        self._validate_transition_rows(tensor)
        return cast(np.ndarray, tensor)

    def _extract_results(self) -> None:
        """Extract results from estimation into sklearn-style attributes."""
        if self._result is None:
            return
        super()._extract_results()
        self.termination_reason_ = "converged" if self.converged_ else "optimizer_failure"
        self.failure_reason_ = None if self.converged_ else self.termination_reason_
        if self.diagnostics_ is not None:
            self.diagnostics_["optimization"].update(
                {
                    "termination_reason": self.termination_reason_,
                    "failure_reason": self.failure_reason_,
                    "method": self.method,
                    "cross_fitting": self.cross_fitting,
                    "locally_robust": self.robust_se,
                }
            )

        # TD-CCP specific: EV feature components
        if self._result.metadata:
            ev = self._result.metadata.get("ev_features")
            if ev is not None:
                self.ev_features_ = np.asarray(ev)

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        """Structural reward matrix R(s,a) of shape (n_states, n_actions).

        Computes the utility matrix from the fitted parameters and the
        feature specification. Returns None if the model has not been fitted.
        """
        if self.params_ is None or self._utility_fn is None or self._result is None:
            return None
        param_names = self._result.parameter_names
        param_vector = jnp.array(
            [self.params_[name] for name in param_names],
            dtype=jnp.float32,
        )
        utility_matrix = self._utility_fn.compute(param_vector)
        return cast(np.ndarray, np.asarray(utility_matrix))

    def summary(self, alpha: float = 0.05) -> str:
        """Generate a formatted summary of estimation results.

        Returns
        -------
        str
            Human-readable summary of the estimation.
        """
        if self._result is None:
            return "Estimator\nTD-CCP\n\nNot fitted. Call fit() first."

        intervals = self.conf_int(alpha=alpha)
        diagnostics = self.diagnostics_ or {}
        data = diagnostics.get("data", {})
        identification = diagnostics.get("identification", {})
        transitions = diagnostics.get("transitions", {})
        ci_level = 100.0 * (1.0 - alpha)
        parameter_lines = []
        for name, estimate in (self.params_ or {}).items():
            lower, upper = intervals[name]
            parameter_lines.append(
                f"{name}: {estimate:.6g} (SE {(self.se_ or {})[name]:.6g}, "
                f"{ci_level:.1f}% CI [{lower:.6g}, {upper:.6g}])"
            )
        if self.bootstrap_ is None:
            uncertainty_lines = [
                f"Method: {'Algorithm 2 locally robust' if self.robust_se else self.se_method}",
                f"Confidence level: {ci_level:.1f}%",
            ]
        else:
            uncertainty_lines = [
                "Method: pairs-cluster bootstrap",
                f"Confidence level: {ci_level:.1f}%",
                f"Bootstrap unit: {self.bootstrap_.unit}",
                "Bootstrap successful draws: "
                f"{self.bootstrap_.n_successful}/{self.bootstrap_.n_requested}",
            ]
        return "\n".join(
            [
                "Estimator",
                "TD-CCP (temporal-difference conditional choice probability estimation)",
                "",
                "Data",
                f"Observations: {self.n_observations_}",
                f"Individuals: {data.get('n_individuals', 'unavailable')}",
                f"State coverage: {data.get('state_coverage', float('nan')):.3f}",
                "",
                "Model",
                f"States: {self.n_states}",
                f"Actions: {self.n_actions}",
                f"Discount factor: {self.discount:.6g}",
                f"TD method: {self.method}",
                "Shock normalization: Type-I extreme-value scale fixed at 1.0",
                "",
                "Pre-estimation checks",
                f"Identification: {identification.get('verdict', 'unavailable')}",
                f"Action-contrast rank: {identification.get('contrast_rank', 'unavailable')}",
                f"Transition source: {transitions.get('source', 'unavailable')}",
                "",
                "Fit",
                f"Converged: {'yes' if self.converged_ else 'no'}",
                f"Termination: {self.termination_reason_}",
                f"Iterations: {self.n_iter_}",
                f"Fit time: {self.fit_time_:.3f} seconds",
                "",
                "Outcome",
                f"Log likelihood: {self.log_likelihood_:.6f}",
                *parameter_lines,
                "",
                "Uncertainty",
                *uncertainty_lines,
                "",
                "Limitations",
                "Inference is conditional on the reward model and discount factor.",
                "The TD parameter stage does not estimate a transition density.",
                "Prediction and counterfactuals use the stored transition tensor.",
            ]
        )

    def conf_int(self, alpha: float = 0.05) -> dict:
        """Compute confidence intervals for parameters.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level. Returns (1 - alpha) confidence intervals.

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
        return cast(np.ndarray, super().predict_proba(states))

    def __repr__(self) -> str:
        if self.params_ is not None:
            return (
                f"TDCCP(n_states={self.n_states}, n_actions={self.n_actions}, "
                f"discount={self.discount}, fitted=True)"
            )
        return (
            f"TDCCP(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, fitted=False)"
        )
