"""GenPQR: IRL via Binary Classification (van der Laan, Kallus & Bibaut 2025).

Two-stage estimator. Stage 1: estimate log CCPs from panel data. Stage 2:
solve the anchor-normalized Bellman fixed point for Q, then recover the
reward as r(s,a) = Q(s,a) - Q(s, a_anchor).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.special import softmax
from scipy.stats import norm as scipy_norm

from econirl.core.types import Panel, Trajectory, TrajectoryPanel
from econirl.transitions import TransitionEstimator


class GenPQR:
    """IRL via policy estimation and Q-evaluation.

    Two-stage estimator from van der Laan, Kallus & Bibaut (2025).
    Stage 1 estimates conditional choice probabilities from panel data.
    Stage 2 solves the anchor-normalized Bellman fixed point for Q and
    recovers reward as r(s,a) = Q(s,a) - Q(s, a_anchor).

    The anchor normalization pins reward at the anchor action to zero for
    all states, which matches the Rust (1987) DDC convention when anchor
    action 0 is the no-replacement alternative.

    Parameters
    ----------
    n_states : int, default=90
        Number of discrete states.
    n_actions : int, default=2
        Number of discrete actions.
    discount : float, default=0.99
        Time discount factor (beta). Must be in (0, 1).
    feature_matrix : numpy.ndarray, optional
        State-feature matrix (n_states, n_features) for linear projection
        of the estimated reward. If None, reward_ contains the full table.
    feature_names : list[str], optional
        Names for each feature column.
    anchor_action : int, default=0
        Action index whose reward is fixed to zero. Sets the normalization.
    se_method : str, default="bootstrap"
        Standard error method. Only "bootstrap" is supported in v1.
    n_bootstrap : int, default=200
        Bootstrap resamples for SE computation.
    se_seed : int, optional
        Random seed for reproducible bootstraps.
    verbose : bool, default=False
        Print progress.

    Attributes
    ----------
    reward_ : numpy.ndarray
        State-action reward table, shape (n_states, n_actions).
        reward_[:, anchor_action] is identically zero.
    policy_ : numpy.ndarray
        Softmax policy pi(a|s), shape (n_states, n_actions).
    value_function_ : numpy.ndarray
        Soft value V(s) = Q_anchor(s) + log sum_a exp(reward_(s,a)),
        shape (n_states,).
    params_ : dict, optional
        Linear projection coefficients {name: value}. None if no
        feature_matrix was supplied.
    se_ : dict, optional
        Bootstrap standard errors for params_. None if no feature_matrix.
    pvalues_ : dict, optional
        Two-sided p-values from Wald tests on params_.
    coef_ : numpy.ndarray, optional
        params_ as an array, shape (n_features,).
    converged_ : bool
        True when the linear solve completed without numerical issues.
    transitions_ : numpy.ndarray
        Transition tensor used, shape (n_actions, n_states, n_states).
    """

    def __init__(
        self,
        n_states: int = 90,
        n_actions: int = 2,
        discount: float = 0.99,
        feature_matrix: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        anchor_action: int = 0,
        se_method: Literal["bootstrap"] = "bootstrap",
        n_bootstrap: int = 200,
        se_seed: int | None = None,
        verbose: bool = False,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.feature_matrix = feature_matrix
        self.feature_names = feature_names
        self.anchor_action = anchor_action
        self.se_method = se_method
        self.n_bootstrap = n_bootstrap
        self.se_seed = se_seed
        self.verbose = verbose

        self.reward_: np.ndarray | None = None
        self.policy_: np.ndarray | None = None
        self.value_function_: np.ndarray | None = None
        self.transitions_: np.ndarray | None = None
        self.params_: dict | None = None
        self.se_: dict | None = None
        self.pvalues_: dict | None = None
        self.coef_: np.ndarray | None = None
        self.converged_: bool | None = None

        self._panel: Panel | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        next_state: str | None = None,
        transitions: np.ndarray | None = None,
        reward=None,
    ) -> "GenPQR":
        """Fit the GenPQR estimator.

        Parameters
        ----------
        data : DataFrame or Panel or TrajectoryPanel
            Panel demonstrations.  Column name args required for DataFrame.
        state, action, id : str
            Column names (required when data is a DataFrame).
        next_state : str, optional
            Next-state column for DataFrame input.
        transitions : numpy.ndarray, optional
            Transition tensor (n_actions, n_states, n_states). If None,
            infers the 2-action Rust-bus kernel (bus data only).
        reward : ignored
            Accepted for API compatibility. GenPQR does not use a prior
            reward specification.

        Returns
        -------
        self : GenPQR
        """
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError("state, action, and id required for DataFrame input")
            self._panel = self._dataframe_to_panel(data, state, action, id, next_state)
        elif isinstance(data, (Panel, TrajectoryPanel)):
            self._panel = data
        else:
            raise TypeError(f"data must be DataFrame, Panel, or TrajectoryPanel, got {type(data)}")

        if transitions is None:
            if self.n_actions > 2:
                raise ValueError(
                    f"GenPQR cannot infer per-action transitions for a "
                    f"{self.n_actions}-action MDP. Pass transitions of shape "
                    f"({self.n_actions}, {self.n_states}, {self.n_states})."
                )
            trans_est = TransitionEstimator(n_states=self.n_states, max_increase=2)
            trans_est.fit(self._panel)
            T = self._bus_kernel_to_3d(np.asarray(trans_est.matrix_))
        else:
            T = np.asarray(transitions, dtype=np.float64)
            if T.ndim != 3 or T.shape != (self.n_actions, self.n_states, self.n_states):
                raise ValueError(
                    f"transitions must have shape "
                    f"({self.n_actions}, {self.n_states}, {self.n_states}), got {T.shape}"
                )

        self.transitions_ = T

        reward_mat, Q_anc = self._estimate(self._panel, T)
        self.reward_ = reward_mat
        self.policy_ = softmax(reward_mat, axis=1)

        # Soft value: V(s) = Q_anc(s) + log sum_a exp(r(s,a))
        log_partition = np.log(np.exp(reward_mat).sum(axis=1))
        self.value_function_ = Q_anc + log_partition

        if self.feature_matrix is not None:
            coef, se = self._fit_features(reward_mat, T)
            fm = np.asarray(self.feature_matrix)
            n_features = fm.shape[1] if fm.ndim > 1 else 1
            names = self.feature_names or [f"f{i}" for i in range(n_features)]
            self.coef_ = coef
            self.params_ = {n: float(v) for n, v in zip(names, coef)}
            if se is not None:
                self.se_ = {n: float(v) for n, v in zip(names, se)}
                self.pvalues_ = {
                    n: float(2 * (1 - scipy_norm.cdf(abs(self.params_[n] / self.se_[n]))))
                    if self.se_[n] > 0
                    else float("nan")
                    for n in names
                }

        self.converged_ = True
        return self

    def summary(self) -> str:
        """Formatted summary of fitted results."""
        if self.reward_ is None:
            return "GenPQR: not fitted. Call fit() first."
        lines = [
            "=" * 60,
            "GenPQR (van der Laan, Kallus & Bibaut 2025)".center(60),
            "=" * 60,
            f"{'Discount:':<25} {self.discount}",
            f"{'States / Actions:':<25} {self.n_states} / {self.n_actions}",
            f"{'Anchor action:':<25} {self.anchor_action}",
            f"{'Converged:':<25} {'Yes' if self.converged_ else 'No'}",
        ]
        if self.params_ is not None:
            lines.append("-" * 60)
            lines.append("Feature coefficients:")
            for name, val in self.params_.items():
                se_str = f"{self.se_[name]:.4f}" if self.se_ else "n/a"
                lines.append(f"  {name:<20} {val:>10.4f}   SE {se_str}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        fitted = self.reward_ is not None
        return (
            f"GenPQR(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"discount={self.discount}, fitted={fitted})"
        )

    # ------------------------------------------------------------------
    # Internal: estimation
    # ------------------------------------------------------------------

    def _estimate(
        self, panel: Panel | TrajectoryPanel, T: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Two-stage estimation.

        Returns
        -------
        reward_mat : ndarray, shape (n_states, n_actions)
            r(s,a) = Q(s,a) - Q(s, anchor), with anchor column = 0.
        Q_anc : ndarray, shape (n_states,)
            Q values under the anchor action.
        """
        u = self._empirical_log_ccps(panel)  # (n_states, n_actions)
        P_anc = T[self.anchor_action]  # (n_states, n_states)

        # Stage 2: solve (I - gamma P_anc) Q_anc = u_anc
        A = np.eye(self.n_states) - self.discount * P_anc
        Q_anc = solve(A, u[:, self.anchor_action], assume_a="gen")  # (n_states,)

        # Q(s,a) = u(s,a) + gamma sum_{s'} P(s'|s,a) Q_anc(s')
        # (T @ Q_anc) has shape (n_actions, n_states); .T -> (n_states, n_actions)
        Q = u + self.discount * (T @ Q_anc).T

        reward = Q - Q_anc[:, np.newaxis]
        return reward, Q_anc

    def _empirical_log_ccps(self, panel: Panel | TrajectoryPanel) -> np.ndarray:
        """Log CCPs estimated from panel via Laplace-smoothed counts."""
        counts = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        for traj in panel.trajectories:
            s = np.asarray(traj.states, dtype=int)
            a = np.asarray(traj.actions, dtype=int)
            np.add.at(counts, (s, a), 1.0)
        counts += 0.5  # Laplace smoothing
        row_sums = counts.sum(axis=1, keepdims=True)
        ccps = counts / row_sums
        return np.log(ccps)

    def _fit_features(
        self, reward_mat: np.ndarray, T: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """OLS of non-anchor reward onto feature_matrix, with bootstrap SE."""
        fm = np.asarray(self.feature_matrix, dtype=np.float64)
        if fm.ndim == 1:
            fm = fm[:, np.newaxis]

        non_anchor = [a for a in range(self.n_actions) if a != self.anchor_action]
        if len(non_anchor) == 1:
            r_vec = reward_mat[:, non_anchor[0]]
        else:
            r_vec = reward_mat[:, non_anchor].mean(axis=1)

        coef = np.linalg.lstsq(fm, r_vec, rcond=None)[0]

        se = None
        if self.n_bootstrap > 0 and self._panel is not None:
            rng = np.random.default_rng(self.se_seed)
            trajs = self._panel.trajectories
            n_ind = len(trajs)
            boot_coefs: list[np.ndarray] = []
            for _ in range(self.n_bootstrap):
                idx = rng.integers(0, n_ind, size=n_ind)
                boot_panel = Panel(trajectories=[trajs[i] for i in idx])
                r_boot, _ = self._estimate(boot_panel, T)
                r_vec_boot = (
                    r_boot[:, non_anchor[0]]
                    if len(non_anchor) == 1
                    else r_boot[:, non_anchor].mean(axis=1)
                )
                c_boot = np.linalg.lstsq(fm, r_vec_boot, rcond=None)[0]
                boot_coefs.append(c_boot)
            se = np.std(boot_coefs, axis=0)

        return coef, se

    # ------------------------------------------------------------------
    # Internal: data conversion
    # ------------------------------------------------------------------

    def _bus_kernel_to_3d(self, keep_kernel: np.ndarray) -> np.ndarray:
        """Expand 2D keep-action kernel to (n_actions, n_states, n_states).

        Non-keep actions use the Rust-bus replacement convention: the
        post-replacement distribution equals the keep-action's distribution
        from state 0.
        """
        n = self.n_states
        T = np.zeros((self.n_actions, n, n), dtype=np.float64)
        T[0] = keep_kernel
        for a in range(1, self.n_actions):
            for s in range(n):
                T[a, s, :] = T[0, 0, :]
        return T

    def _dataframe_to_panel(
        self,
        data: pd.DataFrame,
        state: str,
        action: str,
        id: str,
        next_state: str | None = None,
    ) -> Panel:
        """Convert DataFrame to Panel, following the MCEIRL convention."""
        trajectories = []
        for ind_id, group in data.groupby(id, sort=True):
            sorted_group = group.sort_index()
            states = sorted_group[state].values.astype(np.int64)
            actions = sorted_group[action].values.astype(np.int64)

            if next_state is not None:
                next_states = sorted_group[next_state].values.astype(np.int64)
            else:
                next_states = np.zeros_like(states)
                next_states[:-1] = states[1:]
                if len(states) > 0:
                    last_action = actions[-1]
                    if last_action == 1:
                        next_states[-1] = 0
                    else:
                        next_states[-1] = min(states[-1] + 1, self.n_states - 1)

            trajectories.append(
                Trajectory(
                    states=states,
                    actions=actions,
                    next_states=next_states,
                    individual_id=ind_id,
                )
            )
        return Panel(trajectories=trajectories)
