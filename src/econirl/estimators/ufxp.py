"""Sklearn-style UFXP estimator for dynamic discrete choice models.

This module provides a UFXP class with a scikit-learn style API that wraps the
underlying UFXPEstimator from econirl.estimation.ufxp. It implements the
unnested fixed point approach of Bray ("Unnesting the Fixed Point in the
Estimation of Dynamic Programs") in the optimally weighted form of Oguz and
Bray (2026), which is closed form for linear utility and as asymptotically
efficient as maximum likelihood.

The UFXP estimator has the same interface as NFXP and CCP, allowing users to
switch between estimation methods.

Example:
    >>> from econirl.estimators import UFXP
    >>> import pandas as pd
    >>>
    >>> df = pd.read_csv("zurcher_bus.csv")
    >>>
    >>> model = UFXP(n_states=90, discount=0.9999)
    >>> model.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    >>>
    >>> print(model.params_)
    >>> print(model.summary())
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import DDCProblem, Panel, TrajectoryPanel
from econirl.estimation.ufxp import UFXPEstimator
from econirl.estimators.nfxp import NFXP
from econirl.transitions import TransitionEstimator


class UFXP(NFXP):
    """Sklearn-style UFXP estimator for dynamic discrete choice models.

    UFXP (Unnested Fixed Point) estimates utility parameters from the
    first-order conditions of Bellman's equation. The value-function
    dependence of those conditions is eliminated by dual fixed points computed
    once, before the parameter search, so no dynamic program is ever solved
    inside an optimizer. With the default optimal weighting (the OUFXP form of
    Oguz and Bray 2026) and a linear utility, the estimator is a single
    closed-form weighted moment solve that is as asymptotically efficient as
    maximum likelihood, with standard errors from the efficient moment
    variance.

    Parameters
    ----------
    n_states : int, default=90
        Number of discrete states (e.g., mileage bins).
    n_actions : int, default=2
        Number of discrete actions (e.g., keep/replace).
    discount : float, default=0.9999
        Time discount factor (beta).
    utility : str or RewardSpec, default="linear_cost"
        Utility specification.  Pass ``"linear_cost"`` for the classic bus
        engine model, or a ``RewardSpec`` for custom features.
    weights : str, default="optimal"
        ``"optimal"`` is the efficient OUFXP weighting with standard errors;
        ``"random"`` is the plain random-projection UFXP (no standard errors).
    num_projections : int, default=32
        Number of random projections when ``weights="random"``.
    verbose : bool, default=False
        Whether to print progress messages during estimation.

    Attributes
    ----------
    params_ : dict
        Estimated parameters after fitting.
    se_ : dict
        Standard errors for each parameter (``weights="optimal"`` only).
    coef_ : numpy.ndarray
        Coefficients as a numpy array (sklearn convention).
    log_likelihood_ : float
        Log-likelihood evaluated at the estimate.
    policy_ : numpy.ndarray
        Estimated choice probabilities P(a|s) of shape (n_states, n_actions).
    value_ : numpy.ndarray
        Estimated value function V(s) of shape (n_states,).
    transitions_ : numpy.ndarray
        Transition probability matrix (n_states x n_states).
    converged_ : bool
        Whether the closed-form solve had full rank.

    References
    ----------
    Bray, R. "Unnesting the Fixed Point in the Estimation of Dynamic
        Programs."
    Oguz, E. and Bray, R. (2026). "Training Neural Networks Embedded in
        Dynamic Discrete Choice Models."
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.9999,
        utility: str | RewardSpec = "linear_cost",
        weights: Literal["optimal", "random"] = "optimal",
        num_projections: int = 32,
        verbose: bool = False,
    ):
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            discount=discount,
            utility=utility,
            se_method="asymptotic",
            verbose=verbose,
        )
        self.weights = weights
        self.num_projections = num_projections

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        transitions: np.ndarray | None = None,
        reward: RewardSpec | None = None,
    ) -> "UFXP":
        """Fit the UFXP estimator to data.

        Parameters
        ----------
        data : pandas.DataFrame or Panel or TrajectoryPanel
            Panel data with observations.  When a DataFrame is passed,
            ``state``, ``action``, and ``id`` column names are required.
        state, action, id : str, optional
            Column names (required for DataFrame input).
        transitions : numpy.ndarray, optional
            Pre-estimated transition matrix of shape (n_states, n_states).
            If None, transitions are estimated from the data.
        reward : RewardSpec, optional
            Reward/utility specification overriding the constructor's.

        Returns
        -------
        self : UFXP
        """
        reward_spec = reward if reward is not None else self.utility

        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required "
                    "when data is a DataFrame"
                )
            self._panel = TrajectoryPanel.from_dataframe(
                data, state=state, action=action, id=id
            )
        elif isinstance(data, (Panel, TrajectoryPanel)):
            self._panel = data
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, "
                f"got {type(data)}"
            )

        if isinstance(reward_spec, RewardSpec):
            self.reward_spec_ = reward_spec
            self._utility_fn = reward_spec.to_linear_utility()
        elif reward_spec == "linear_cost":
            self._utility_fn = self._create_utility()
            self.reward_spec_ = RewardSpec(
                self._utility_fn.feature_matrix,
                self._utility_fn.parameter_names,
            )
        else:
            raise ValueError(f"Unknown reward/utility specification: {reward_spec}")

        if transitions is None:
            trans_estimator = TransitionEstimator(
                n_states=self.n_states,
                max_increase=2,
            )
            trans_estimator.fit(self._panel)
            self.transitions_ = trans_estimator.matrix_
        else:
            self.transitions_ = np.asarray(transitions)

        transition_tensor = self._build_transition_tensor(self.transitions_)

        self._problem = DDCProblem(
            num_states=self.n_states,
            num_actions=self.n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )

        estimator = UFXPEstimator(
            weights=self.weights,
            num_projections=self.num_projections,
            verbose=self.verbose,
        )

        self._result = estimator.estimate(
            panel=self._panel,
            utility=self._utility_fn,
            problem=self._problem,
            transitions=transition_tensor,
        )

        self._extract_results()

        return self

    def summary(self) -> str:
        """Generate a formatted summary of estimation results."""
        if self._result is None:
            return "UFXP: Not fitted yet. Call fit() first."

        return self._result.summary()

    def __repr__(self) -> str:
        fitted = self.params_ is not None
        return (
            f"UFXP(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, weights={self.weights!r}, "
            f"fitted={fitted})"
        )
