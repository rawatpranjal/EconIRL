"""Sklearn-style CCP estimator for dynamic discrete choice models.

This module provides a CCP class with a scikit-learn style API that wraps
the underlying CCPEstimator from econirl.estimation.ccp. It implements the
Hotz-Miller (1993) approach and NPL extension (Aguirregabiria-Mira 2002).

The CCP estimator has the same interface as NFXP, allowing users to easily
switch between estimation methods.

Example:
    >>> from econirl.estimators import CCP
    >>> import pandas as pd
    >>>
    >>> # Load bus replacement data
    >>> df = pd.read_csv("zurcher_bus.csv")
    >>>
    >>> # Create estimator and fit (Hotz-Miller by default)
    >>> model = CCP(n_states=90, discount=0.9999)
    >>> model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> # Access results sklearn-style
    >>> print(model.params_)        # {"operating_cost": 0.001, "replacement_cost": 9.35}
    >>> print(model.summary())
    >>>
    >>> # Use NPL (iterative refinement)
    >>> model_npl = CCP(n_states=90, num_policy_iterations=10)
    >>> model_npl.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import DDCProblem, Panel, TrajectoryPanel
from econirl.estimation.ccp import CCPEstimator
from econirl.estimators.nfxp import NFXP
from econirl.preprocessing.diagnostics import feature_diagnostics
from econirl.transitions import TransitionEstimator


class CCP(NFXP):
    """Sklearn-style CCP estimator for dynamic discrete choice models.

    The CCP (Conditional Choice Probability) approach estimates utility
    parameters using the Hotz-Miller inversion theorem. This is typically
    faster than NFXP because it avoids the nested fixed-point computation.

    With num_policy_iterations=1, this is the classic Hotz-Miller estimator.
    With num_policy_iterations>1, this becomes the NPL (Nested Pseudo
    Likelihood) algorithm. A positive value is the maximum number of stages.
    Set it to -1 to require joint parameter and policy fixed-point convergence.

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
        Method for computing standard errors.
    n_bootstrap : int, default=400
        Number of pairs-cluster bootstrap replications when
        ``se_method="bootstrap"``.
    se_seed : int, optional
        Random seed for bootstrap standard errors.
    verbose : bool, default=False
        Whether to print progress messages during estimation.
    num_policy_iterations : int, default=1
        Number of policy iterations. K=1 is Hotz-Miller, K>1 is NPL.
        Set to -1 for convergence-based stopping.

    Attributes
    ----------
    params_ : dict
        Estimated parameters after fitting. Keys are parameter names
        (e.g., "operating_cost", "replacement_cost") and values are point
        estimates.
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
        Transition probability matrix (n_states x n_states).
    transition_tensor_ : numpy.ndarray
        Action-specific transition probabilities with shape
        ``(n_actions, n_states, n_states)``.
    transition_source_ : str
        Whether transitions were estimated from the fitted panel or supplied.
    converged_ : bool
        Whether the requested CCP run completed successfully.
    npl_converged_ : bool
        Whether both the parameter and policy residuals met the NPL tolerance.
    termination_reason_ : str
        Why the CCP run stopped.
    npl_parameter_residual_ : float
        Final L2 change in the parameter vector.
    npl_policy_residual_ : float
        Final maximum absolute policy fixed-point residual.
    reward_spec_ : RewardSpec
        The reward specification used for estimation.

    Examples
    --------
    >>> from econirl.estimators import CCP
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({
    ...     "bus_id": [0, 0, 1, 1],
    ...     "mileage": [10, 20, 15, 30],
    ...     "replaced": [0, 0, 0, 1],
    ... })
    >>>
    >>> # Hotz-Miller (fast, one-step)
    >>> model = CCP(n_states=90)
    >>> model.fit(df, state="mileage", action="replaced", id="bus_id")
    >>>
    >>> # Fixed-stage NPL
    >>> model_npl = CCP(n_states=90, num_policy_iterations=10)
    >>> model_npl.fit(df, state="mileage", action="replaced", id="bus_id")

    References
    ----------
    Hotz, V.J. and Miller, R.A. (1993). "Conditional Choice Probabilities
        and the Estimation of Dynamic Models." RES 60(3), 497-529.
    Aguirregabiria, V. and Mira, P. (2002). "Swapping the Nested Fixed Point
        Algorithm." Econometrica 70(4), 1519-1543.
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.9999,
        utility: RewardSpec | None = None,
        se_method: Literal["asymptotic", "robust", "clustered", "bootstrap"] = "robust",
        n_bootstrap: int = 400,
        se_seed: int | None = None,
        verbose: bool = False,
        num_policy_iterations: int = 1,
    ):
        """Initialize the CCP estimator.

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
        num_policy_iterations : int, default=1
            Maximum number of NPL stages (K=1 is Hotz-Miller).
        """
        # Initialize parent with shared parameters
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            discount=discount,
            utility=utility,
            se_method=se_method,
            n_bootstrap=n_bootstrap,
            se_seed=se_seed,
            verbose=verbose,
        )
        # CCP-specific parameter
        self.num_policy_iterations = num_policy_iterations
        self.npl_converged_: bool | None = None
        self.termination_reason_: str | None = None
        self.npl_parameter_residual_: float | None = None
        self.npl_policy_residual_: float | None = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        transitions: np.ndarray | None = None,
        reward: RewardSpec | None = None,
    ) -> "CCP":
        """Fit the CCP estimator to data.

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
            Pre-estimated transition probabilities. A 2D array of shape
            ``(n_states, n_states)`` is treated as the keep-action matrix.
            A 3D array of shape ``(n_actions, n_states, n_states)`` is used as
            the full action-specific transition tensor.
            If None, transitions are estimated from the data.
        reward : RewardSpec, optional
            Reward/utility specification.  If provided, overrides the
            ``utility`` parameter passed at construction time.

        Returns
        -------
        self : CCP
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

        feat = np.asarray(self._utility_fn.feature_matrix)
        expected_prefix = (self.n_states, self.n_actions)
        if feat.ndim != 3 or feat.shape[:2] != expected_prefix:
            raise ValueError(
                f"reward/feature matrix has shape {feat.shape}; expected "
                f"(n_states={self.n_states}, n_actions={self.n_actions}, n_features). "
                "Check that n_states and n_actions match your RewardSpec features."
            )

        support = self._check_ccp_support(feat)

        # Estimate transitions if not provided
        if transitions is None:
            trans_estimator = TransitionEstimator(
                n_states=self.n_states,
                max_increase=2,
            )
            trans_estimator.fit(self._panel)
            self.transitions_ = trans_estimator.matrix_
            self._transition_probabilities = trans_estimator.probs_
            self._transition_increments = trans_estimator.transition_increments_
            self.transition_source_ = "estimated from fitted panel"
        else:
            self.transitions_ = np.asarray(transitions)
            self._transition_probabilities = None
            self._transition_increments = None
            self.transition_source_ = (
                "supplied action-specific tensor"
                if self.transitions_.ndim == 3
                else "supplied keep-transition matrix"
            )

        # Build full transition matrices (for both actions)
        transition_tensor = self._build_transition_tensor(self.transitions_)
        self.transition_tensor_ = transition_tensor

        # Create problem specification
        self._problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )

        # Create the underlying CCP estimator (this is the key difference from NFXP)
        estimator = CCPEstimator(
            num_policy_iterations=self.num_policy_iterations,
            se_method=self.se_method,
            verbose=self.verbose,
        )

        # Run estimation
        self._result = estimator.estimate(
            panel=self._panel,
            utility=self._utility_fn,
            problem=self._problem,
            transitions=transition_tensor,
            n_bootstrap=self.n_bootstrap,
            se_seed=self.se_seed,
            transition_source=self.transition_source_,
        )
        self._result.metadata["ccp_support"] = support

        # Extract results
        self._extract_results()
        self.npl_converged_ = bool(self._result.metadata["npl_converged"])
        self.termination_reason_ = str(self._result.metadata["termination_reason"])
        parameter_residual = self._result.metadata["npl_parameter_residual"]
        policy_residual = self._result.metadata["npl_policy_residual"]
        self.npl_parameter_residual_ = (
            None if parameter_residual is None else float(parameter_residual)
        )
        self.npl_policy_residual_ = None if policy_residual is None else float(policy_residual)

        return self

    def _check_ccp_support(self, feature_matrix: np.ndarray) -> dict[str, object]:
        """Check identification and empirical policy support before fitting."""
        diagnostics = feature_diagnostics(feature_matrix)
        num_features = int(diagnostics["num_features"])
        feature_rank = int(diagnostics["feature_rank"])
        contrast_rank = int(diagnostics["contrast_rank"])
        if feature_rank < num_features:
            raise ValueError(
                f"CCP reward design is rank deficient: feature rank {feature_rank} < {num_features}"
            )
        if contrast_rank < num_features:
            raise ValueError(
                f"CCP reward design is not identified from choices: "
                f"action-contrast rank {contrast_rank} < {num_features}"
            )

        states = np.asarray(self._panel.get_all_states(), dtype=np.int64)
        actions = np.asarray(self._panel.get_all_actions(), dtype=np.int64)
        if states.size == 0:
            raise ValueError("CCP requires at least one panel observation")
        if (states < 0).any() or (states >= self.n_states).any():
            raise ValueError(f"panel states must lie in [0, {self.n_states})")
        if (actions < 0).any() or (actions >= self.n_actions).any():
            raise ValueError(f"panel actions must lie in [0, {self.n_actions})")

        counts = np.zeros((self.n_states, self.n_actions), dtype=np.int64)
        np.add.at(counts, (states, actions), 1)
        action_totals = counts.sum(axis=0)
        missing_actions = np.flatnonzero(action_totals == 0)
        if missing_actions.size:
            raise ValueError(
                "CCP cannot estimate a choice model when actions are absent "
                f"from the panel: {missing_actions.tolist()}"
            )

        state_totals = counts.sum(axis=1)
        visited = state_totals > 0
        unvisited_states = int((~visited).sum())
        single_action_states = int(((counts > 0).sum(axis=1)[visited] == 1).sum())
        thin_states = int((state_totals[visited] < 5).sum())
        messages = []
        if unvisited_states:
            messages.append(f"{unvisited_states} states are unvisited")
        if single_action_states:
            messages.append(f"{single_action_states} visited states contain only one action")
        if thin_states:
            messages.append(f"{thin_states} visited states contain fewer than five observations")
        for label, value in (
            ("feature", float(diagnostics["condition_number"])),
            ("action-contrast", float(diagnostics["contrast_condition_number"])),
        ):
            if not np.isfinite(value) or value > 1e6:
                messages.append(f"{label} condition number is {value:.3g}")
        if messages:
            warnings.warn(
                "CCP first-stage support is thin: "
                + "; ".join(messages)
                + ". Inspect the PRE-ESTIMATION CHECKS before interpreting the fit.",
                RuntimeWarning,
                stacklevel=2,
            )

        return {
            **diagnostics,
            "states_visited": int(visited.sum()),
            "state_action_coverage": float((counts > 0).sum() / counts.size),
            "single_action_states": single_action_states,
            "thin_states": thin_states,
            "action_counts": action_totals.tolist(),
            "warnings": messages,
        }

    def summary(self, alpha: float = 0.05) -> str:
        """Generate a formatted summary of estimation results.

        Returns
        -------
        str
            Human-readable summary of the estimation.
        """
        if self._result is None:
            return "CCP: Not fitted yet. Call fit() first."

        return self._result.summary(alpha=alpha)

    def __repr__(self) -> str:
        if self.params_ is not None:
            return (
                f"CCP(n_states={self.n_states}, n_actions={self.n_actions}, "
                f"discount={self.discount}, num_policy_iterations={self.num_policy_iterations}, "
                f"fitted=True)"
            )
        return (
            f"CCP(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, num_policy_iterations={self.num_policy_iterations}, "
            f"fitted=False)"
        )
