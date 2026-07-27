"""MCEIRLNeural: Neural Maximum Causal Entropy IRL.

Supports two reward parameterizations:
- ``reward_type="state_action"`` (default): learns R(s,a) via a neural
  network that takes [state_features, action_onehot] as input.  This is
  more general and correctly handles environments with action-dependent
  rewards (e.g., gridworlds where moving has a cost but staying is free).
- ``reward_type="state"``: learns R(s) only, broadcasting the same reward
  to all actions (original behaviour).

Training loop (MCE-IRL objective, Ziebart 2010):
    for epoch in range(max_epochs):
        1. Compute reward matrix R(s,a) for all (state, action) pairs
        2. Solve soft Bellman with this reward (transitions required)
        3. Compute state visitation frequencies via forward pass
        4. Loss = -E_expert[R] + E_policy[R]  (feature matching)
        5. Backprop through reward network

After training, implied rewards are projected onto features via
least-squares to extract interpretable theta (same as NeuralGLADIUS).

Reference:
    Ziebart, B. D. (2010). Modeling purposeful adaptive behavior with the
        principle of maximum causal entropy. PhD thesis, CMU.
    Wulfmeier, M., Ondruska, P., & Posner, I. (2015). Maximum entropy
        deep inverse reinforcement learning. arXiv:1507.04888.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.occupancy import compute_state_action_visitation
from econirl.core.reward_spec import RewardSpec
from econirl.core.solvers import hybrid_iteration, value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory, TrajectoryPanel
from econirl.estimators.neural_base import NeuralEstimatorMixin
from econirl.simulation.counterfactual import CounterfactualResult, CounterfactualType

# ---------------------------------------------------------------------------
# Internal network modules (Equinox)
# ---------------------------------------------------------------------------


class _StateRewardNetwork(eqx.Module):
    """R(s) reward network.

    Input: state features of shape (state_dim,).
    Output: scalar reward.
    """

    layers: list
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, num_layers + 1)
        layers = []
        in_dim = state_dim
        for i in range(num_layers):
            layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys[i]))
            in_dim = hidden_dim
        self.layers = layers
        self.output_layer = eqx.nn.Linear(in_dim, 1, key=keys[-1])

    def __call__(self, state_feat: jax.Array) -> jax.Array:
        """Compute R(s) for a single state.

        Parameters
        ----------
        state_feat : jax.Array
            State features of shape (state_dim,).

        Returns
        -------
        jax.Array
            Scalar reward.
        """
        x = state_feat
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        return self.output_layer(x).squeeze(-1)


class _StateActionRewardNetwork(eqx.Module):
    """R(s,a) reward network.

    Input: concatenation of state features (state_dim,) and action
    one-hot encoding (n_actions,).
    Output: scalar reward.
    """

    layers: list
    output_layer: eqx.nn.Linear
    _n_actions: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        self._n_actions = n_actions
        input_dim = state_dim + n_actions
        keys = jax.random.split(key, num_layers + 1)
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys[i]))
            in_dim = hidden_dim
        self.layers = layers
        self.output_layer = eqx.nn.Linear(in_dim, 1, key=keys[-1])

    def __call__(self, state_feat: jax.Array, action_onehot: jax.Array) -> jax.Array:
        """Compute R(s,a) for a single (state, action) pair.

        Parameters
        ----------
        state_feat : jax.Array
            State features of shape (state_dim,).
        action_onehot : jax.Array
            One-hot action encoding of shape (n_actions,).

        Returns
        -------
        jax.Array
            Scalar reward.
        """
        x = jnp.concatenate([state_feat, action_onehot])
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        return self.output_layer(x).squeeze(-1)

    def all_actions(self, state_feat: jax.Array) -> jax.Array:
        """Compute R(s,a) for all actions at every state.

        Parameters
        ----------
        state_feat : jax.Array
            State features of shape (S, state_dim).

        Returns
        -------
        jax.Array
            Reward matrix of shape (S, A).
        """
        S = state_feat.shape[0]
        A = self._n_actions
        eye = jnp.eye(A)
        # Expand state features: (S, state_dim) -> (S*A, state_dim)
        sf_expanded = jnp.repeat(state_feat, A, axis=0)
        # Tile action one-hots: (A, A) -> (S*A, A)
        act_expanded = jnp.tile(eye, (S, 1))
        # Apply network to all (state, action) pairs in one vmap call
        rewards = jax.vmap(self)(sf_expanded, act_expanded)
        return rewards.reshape(S, A)


# ---------------------------------------------------------------------------
# MCEIRLNeural estimator
# ---------------------------------------------------------------------------


class MCEIRLNeural(NeuralEstimatorMixin):
    """Neural Maximum Causal Entropy IRL.

    Learns a neural reward function using the MCE-IRL objective:
    maximize E_expert[R] - log Z(R)

    where Z(R) is the partition function (soft value at initial state).

    Supports two reward types:

    - ``reward_type="state_action"`` (default): R(s,a) via a network that
      takes [state_features, action_onehot].  This is more general and
      correctly handles action-dependent rewards.
    - ``reward_type="state"``: R(s) broadcast to all actions (original).

    For v1, transitions must be available so that exact soft value iteration
    and state visitation frequencies can be computed.

    Parameters
    ----------
    n_states : int, optional
        Number of discrete states.  Inferred from data if None.
    n_actions : int, optional
        Number of discrete actions.  Inferred from data if None.
    discount : float, default=0.95
        Time discount factor beta.
    reward_type : str, default="state_action"
        Type of reward function: ``"state_action"`` for R(s,a) or
        ``"state"`` for R(s) broadcast to all actions.
    reward_hidden_dim : int, default=64
        Hidden dimension for the reward MLP.
    reward_num_layers : int, default=2
        Number of hidden layers in the reward MLP.
    reward_network : callable, optional
        Factory for a custom reward architecture, overriding the default MLP.
        Called as ``reward_network(state_dim, n_actions, key)`` and must return
        an Equinox module that maps the full state-feature matrix of shape
        ``(n_states, state_dim)`` to the reward: shape ``(n_states,)`` for a
        state reward or ``(n_states, n_actions)`` for a state-action reward. Set
        ``reward_type`` to match the module's output. Use this to plug in deeper
        MLPs, residual or Fourier-feature nets, or convolutional reward fields
        (e.g. a CoordConv over the grid). When ``None`` the default MLP is used.
    max_epochs : int, default=200
        Maximum number of training epochs.
    lr : float, default=1e-3
        Learning rate for Adam optimizer.
    inner_solver : str, default="hybrid"
        Solver for soft value iteration: "hybrid" or "value".
    inner_tol : float, default=1e-8
        Convergence tolerance for inner solver.
    inner_max_iter : int, default=5000
        Maximum iterations for inner solver.
    state_encoder : callable, optional
        Function mapping state indices (int array) to feature vectors.
        Receives shape (B,) and should return shape (B, state_dim).
        If None, a default normalizing encoder is created.
    state_dim : int, optional
        Dimension of state features.  Required if state_encoder is provided.
    feature_names : list of str, optional
        Names for features when projecting rewards onto linear features.
    anchor_action : int, optional
        Action whose reward is fixed to zero. This is useful for identified
        action-dependent IRL designs with a normalized outside/exit action.
    absorbing_state : int, optional
        State whose reward row is fixed to zero.
    seed : int, default=0
        Random seed for network initialization.
    verbose : bool, default=False
        Whether to print progress during training.

    Attributes
    ----------
    params_ : dict or None
        Projected structural parameters after fitting.  None if no
        features were provided for projection.
    se_ : dict or None
        Pseudo standard errors from the projection regression.
    pvalues_ : dict or None
        P-values from Wald t-test on pseudo SEs.
    coef_ : numpy.ndarray or None
        Coefficient array (same values as ``params_`` in array form).
    policy_ : numpy.ndarray or None
        Estimated choice probabilities P(a|s) of shape (n_states, n_actions).
    value_ : numpy.ndarray or None
        Estimated value function V(s) of shape (n_states,).
    reward_ : numpy.ndarray or None
        Neural reward.  Shape (n_states,) for ``reward_type="state"``
        or (n_states, n_actions) for ``reward_type="state_action"``.
    projection_r2_ : float or None
        R-squared of the feature projection.
    converged_ : bool or None
        Whether training converged.
    n_epochs_ : int or None
        Number of training epochs completed.

    Examples
    --------
    >>> from econirl.estimators import MCEIRLNeural
    >>> import numpy as np
    >>>
    >>> # R(s,a) -- default, more general
    >>> model = MCEIRLNeural(n_states=25, n_actions=4, discount=0.95)
    >>> model.fit(data=df, state="state", action="action", id="agent_id",
    ...           transitions=T)
    >>> print(model.reward_.shape)  # (25, 4)
    >>> print(model.policy_.shape)  # (25, 4)
    >>>
    >>> # R(s) -- state-only, backward compatible
    >>> model = MCEIRLNeural(n_states=25, n_actions=4, reward_type="state")
    >>> model.fit(...)
    >>> print(model.reward_.shape)  # (25,)
    >>>
    >>> # Custom architecture: any net mapping (S, state_dim) -> (S,) or (S, A)
    >>> import equinox as eqx, jax
    >>> class DeeperMLP(eqx.Module):
    ...     layers: list
    ...     def __init__(self, state_dim, n_actions, key):
    ...         k1, k2 = jax.random.split(key)
    ...         self.layers = [eqx.nn.Linear(state_dim, 64, key=k1),
    ...                        eqx.nn.Linear(64, 1, key=k2)]
    ...     def __call__(self, X):  # (S, state_dim) -> (S,)
    ...         f = lambda x: self.layers[1](jax.nn.tanh(self.layers[0](x))).squeeze(-1)
    ...         return jax.vmap(f)(X)
    >>> model = MCEIRLNeural(n_states=25, n_actions=4, reward_type="state",
    ...                      reward_network=lambda sd, na, key: DeeperMLP(sd, na, key))
    >>> model.fit(...)  # CoordConv / CNN reward fields plug in the same way
    """

    def __init__(
        self,
        n_states: int | None = None,
        n_actions: int | None = None,
        discount: float = 0.95,
        reward_type: str = "state_action",
        reward_hidden_dim: int = 64,
        reward_num_layers: int = 2,
        reward_network: Callable | None = None,
        max_epochs: int = 200,
        lr: float = 1e-3,
        occupancy_tol: float = 1e-2,
        patience: int = 100,
        improvement_tol: float = 1e-5,
        inner_solver: str = "hybrid",
        inner_tol: float = 1e-8,
        inner_max_iter: int = 5000,
        state_encoder: Callable | None = None,
        state_dim: int | None = None,
        feature_names: list[str] | None = None,
        anchor_action: int | None = 0,
        anchor_state: int | None = 0,
        absorbing_state: int | None = None,
        seed: int = 0,
        verbose: bool = False,
    ):
        if reward_type not in ("state", "state_action"):
            raise ValueError(f"reward_type must be 'state' or 'state_action', got '{reward_type}'")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must lie in [0, 1]")
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if patience < 1:
            raise ValueError("patience must be positive")
        if occupancy_tol <= 0 or inner_tol <= 0 or improvement_tol < 0:
            raise ValueError("solver tolerances must be positive")
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.reward_type = reward_type
        self.reward_hidden_dim = reward_hidden_dim
        self.reward_num_layers = reward_num_layers
        self.reward_network = reward_network
        self.max_epochs = max_epochs
        self.lr = lr
        self.occupancy_tol = occupancy_tol
        self.patience = patience
        self.improvement_tol = improvement_tol
        self.inner_solver = inner_solver
        self.inner_tol = inner_tol
        self.inner_max_iter = inner_max_iter
        self.state_encoder = state_encoder
        self.state_dim = state_dim
        self.feature_names = feature_names
        self.anchor_action = anchor_action
        self.anchor_state = absorbing_state if absorbing_state is not None else anchor_state
        self.absorbing_state = absorbing_state
        self.seed = seed
        self.verbose = verbose

        self.params_: dict[str, float] | None = None
        self.se_: None = None
        self.pvalues_: None = None
        self.coef_: np.ndarray | None = None
        self.policy_: np.ndarray | None = None
        self.value_: np.ndarray | None = None
        self.reward_: np.ndarray | None = None
        self.projection_r2_: float | None = None
        self.projection_diagnostics_: dict[str, object] | None = None
        self.converged_: bool | None = None
        self.termination_reason_: str | None = None
        self.n_epochs_: int | None = None
        self.best_epoch_: int | None = None
        self.training_loss_: float | None = None
        self.feature_difference_: float | None = None
        self.occupancy_moment_residual_: float | None = None
        self.bellman_residual_: float | None = None
        self.n_observations_: int | None = None
        self.diagnostics_: dict[str, object] | None = None
        self.transitions_: np.ndarray | None = None
        self.action_mask_: np.ndarray | None = None

        self._reward_net: Any = None
        self._state_encoder: Callable | None = None
        self._state_dim: int | None = None
        self._n_states: int | None = None
        self._n_actions: int | None = None
        self._empirical_sa: jnp.ndarray | None = None
        self._initial_distribution: jnp.ndarray | None = None
        self._action_mask_jax: jax.Array | None = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        features: RewardSpec | np.ndarray | None = None,
        transitions: np.ndarray | None = None,
        action_mask: np.ndarray | None = None,
        context: object = None,
    ) -> "MCEIRLNeural":
        """Fit the neural reward map under known transitions.

        ``transitions`` must use ``(n_actions, n_states, n_states)`` orientation.
        ``action_mask``, when supplied, has shape ``(n_states, n_actions)``.
        Feature inputs are projection diagnostics and do not turn neural weights
        into structural parameters.
        """
        del context
        if transitions is None:
            raise ValueError(
                "MCEIRLNeural requires transitions. Pass an (n_actions, n_states, n_states) array."
            )
        if isinstance(data, pd.DataFrame):
            for column_name, label in ((state, "state"), (action, "action")):
                if column_name is None or column_name not in data:
                    continue
                numeric = pd.to_numeric(data[column_name], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
                    raise ValueError(f"{label} values must be finite integer codes")

        panel, all_states, all_actions, all_next = self._extract_data(data, state, action, id)
        if all_states.size == 0:
            raise ValueError("data must contain at least one observation")
        n_states = self.n_states or int(max(all_states.max(), all_next.max())) + 1
        n_actions = self.n_actions or int(all_actions.max()) + 1
        self._n_states = n_states
        self._n_actions = n_actions
        self.n_observations_ = int(all_states.size)

        if np.any(all_states < 0) or np.any(all_states >= n_states):
            raise ValueError("observed states fall outside [0, n_states)")
        if np.any(all_next < 0) or np.any(all_next >= n_states):
            raise ValueError("observed next states fall outside [0, n_states)")
        if np.any(all_actions < 0) or np.any(all_actions >= n_actions):
            raise ValueError("observed actions fall outside [0, n_actions)")

        transition_array = np.asarray(transitions, dtype=np.float64)
        expected_shape = (n_actions, n_states, n_states)
        if transition_array.shape != expected_shape:
            raise ValueError(
                "transitions must have shape "
                f"{expected_shape} in (actions, states, next_states) orientation; "
                f"got {transition_array.shape}"
            )
        if not np.isfinite(transition_array).all() or np.any(transition_array < 0):
            raise ValueError("transitions must be finite and nonnegative")
        row_error = float(np.max(np.abs(transition_array.sum(axis=2) - 1.0)))
        if row_error > 1e-6:
            raise ValueError(
                f"transition rows must sum to one; maximum row error is {row_error:.3g}"
            )

        if action_mask is None:
            mask = np.ones((n_states, n_actions), dtype=bool)
        else:
            mask = np.asarray(action_mask, dtype=bool)
            if mask.shape != (n_states, n_actions):
                raise ValueError(
                    f"action_mask must have shape {(n_states, n_actions)}, got {mask.shape}"
                )
            if np.any(mask.sum(axis=1) == 0):
                raise ValueError("every state must retain at least one available action")
        if np.any(~mask[all_states, all_actions]):
            raise ValueError("demonstrations contain actions marked unavailable")

        if self.reward_type == "state_action":
            if self.anchor_action is None or not 0 <= self.anchor_action < n_actions:
                raise ValueError(f"state_action rewards require anchor_action in [0, {n_actions})")
        elif self.anchor_state is None or not 0 <= self.anchor_state < n_states:
            raise ValueError(f"state rewards require anchor_state in [0, {n_states})")

        counts = np.zeros((n_states, n_actions), dtype=np.int64)
        np.add.at(counts, (all_states, all_actions), 1)
        available_cells = int(mask.sum())
        observed_available = int(np.count_nonzero((counts > 0) & mask))
        action_shares = np.bincount(all_actions, minlength=n_actions) / all_actions.size
        self.diagnostics_ = {
            "transition_orientation": "(n_actions, n_states, n_states)",
            "max_transition_row_error": row_error,
            "observed_states": int(np.unique(all_states).size),
            "state_coverage": float(np.unique(all_states).size / n_states),
            "state_action_coverage": float(observed_available / available_cells),
            "single_action_states": int(np.sum(mask.sum(axis=1) == 1)),
            "min_action_share": float(action_shares.min()),
            "normalization": (
                f"anchor_action={self.anchor_action}"
                if self.reward_type == "state_action"
                else f"anchor_state={self.anchor_state}"
            ),
        }

        self.transitions_ = transition_array.astype(np.float32)
        self.action_mask_ = mask
        self._action_mask_jax = jnp.asarray(mask)
        transitions_jax = jnp.asarray(self.transitions_)
        self._build_encoder(n_states)
        assert self._state_dim is not None
        assert self._state_encoder is not None
        empirical_sa = self._compute_empirical_occupancy(
            panel, n_states, n_actions, discount=self.discount
        )
        self._empirical_sa = empirical_sa
        self._initial_distribution = self._compute_initial_distribution(panel, n_states)

        key = jax.random.PRNGKey(self.seed)
        if self.reward_network is not None:
            self._reward_net = self.reward_network(self._state_dim, n_actions, key)
        elif self.reward_type == "state_action":
            self._reward_net = _StateActionRewardNetwork(
                self._state_dim,
                n_actions,
                self.reward_hidden_dim,
                self.reward_num_layers,
                key=key,
            )
        else:
            self._reward_net = _StateRewardNetwork(
                self._state_dim,
                self.reward_hidden_dim,
                self.reward_num_layers,
                key=key,
            )

        self._train_mce(transitions_jax, empirical_sa, n_states, n_actions)
        self._extract_final(transitions_jax, n_states, n_actions)

        if features is not None:
            self._project_onto_features(features, n_states, n_actions)
        else:
            self.params_ = None
            self.coef_ = None
            self.projection_r2_ = None
            self.projection_diagnostics_ = None
        self.se_ = None
        self.pvalues_ = None
        return self

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_data(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None,
        action: str | None,
        id: str | None,
    ) -> tuple[TrajectoryPanel, np.ndarray, np.ndarray, np.ndarray]:
        """Extract state/action/next_state arrays from input data."""
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required when data is a DataFrame"
                )
            panel = TrajectoryPanel.from_dataframe(data, state=state, action=action, id=id)
            all_states = np.asarray(panel.all_states, dtype=np.int64)
            all_actions = np.asarray(panel.all_actions, dtype=np.int64)
            all_next = np.asarray(panel.all_next_states, dtype=np.int64)
        elif isinstance(data, (Panel, TrajectoryPanel)):
            panel = TrajectoryPanel.from_panel(data)
            all_states = np.asarray(panel.get_all_states(), dtype=np.int64)
            all_actions = np.asarray(panel.get_all_actions(), dtype=np.int64)
            all_next = np.asarray(panel.get_all_next_states(), dtype=np.int64)
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, got {type(data)}"
            )

        return panel, all_states, all_actions, all_next

    # ------------------------------------------------------------------
    # Encoder setup
    # ------------------------------------------------------------------

    def _build_encoder(self, n_states: int) -> None:
        """Build default state encoder if not provided."""
        if self.state_encoder is not None:
            self._state_encoder = self.state_encoder
            self._state_dim = self.state_dim or 1
        else:
            max_s = max(n_states - 1, 1)

            def _default_encoder(s, _ms=max_s):
                s_float = jnp.asarray(s, dtype=jnp.float32)
                return (s_float / _ms).reshape(-1, 1)

            self._state_encoder = _default_encoder
            self._state_dim = 1

    # ------------------------------------------------------------------
    # Empirical occupancy
    # ------------------------------------------------------------------

    def _compute_empirical_occupancy(
        self,
        panel: TrajectoryPanel,
        n_states: int,
        n_actions: int,
        discount: float = 1.0,
    ) -> jnp.ndarray:
        """Compute empirical state-action occupancy from demonstrations.

        Returns
        -------
        jnp.ndarray
            State-action occupancy of shape (n_states, n_actions).
            Normalized to sum to 1.
        """
        sa_counts = np.zeros((n_states, n_actions), dtype=np.float32)
        total = 0.0
        for traj in panel.trajectories:
            states = np.asarray(traj.states, dtype=np.int64)
            actions = np.asarray(traj.actions, dtype=np.int64)
            if len(states) == 0:
                continue
            if discount == 1.0:
                weights = np.ones(len(states), dtype=np.float32)
            else:
                weights = np.power(float(discount), np.arange(len(states))).astype(np.float32)
            flat_idx = states * n_actions + actions
            np.add.at(sa_counts.ravel(), flat_idx, weights)
            total += float(weights.sum())
        if total > 0:
            sa_counts = sa_counts / total
        return jnp.array(sa_counts)

    def _compute_initial_distribution(
        self,
        panel: TrajectoryPanel,
        n_states: int,
    ) -> jnp.ndarray:
        """Compute the empirical initial-state distribution."""
        counts = np.zeros(n_states, dtype=np.float32)
        for traj in panel.trajectories:
            if len(traj.states):
                counts[int(traj.states[0])] += 1.0
        total = counts.sum()
        if total > 0:
            counts = counts / total
        else:
            counts[:] = 1.0 / n_states
        return jnp.asarray(counts)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _net_reward_matrix(
        self,
        reward_net,
        state_feat: jax.Array,
        n_states: int,
        n_actions: int,
    ) -> jax.Array:
        """Raw R(s,a) from the reward net, before absorbing/anchor masking.

        Default per-state nets are vmapped over states (state reward) or queried
        for all actions (state-action reward). A custom ``reward_network`` is
        called once on the full ``(S, state_dim)`` feature matrix and must return
        ``(S,)`` (state reward) or ``(S, A)`` (state-action reward).
        """
        if self.reward_network is not None:
            # Feed the custom net canonical-dtype features (x64-aware) so strict
            # ops like conv (which do not auto-promote) match their weights.
            sf = state_feat.astype(jnp.result_type(float))
            out = jnp.asarray(reward_net(sf))
            if out.ndim == 1:
                return jnp.broadcast_to(out[:, None], (n_states, n_actions))
            return out
        if self.reward_type == "state_action":
            return jnp.asarray(reward_net.all_actions(state_feat))
        rewards_s = jax.vmap(reward_net)(state_feat)
        return jnp.broadcast_to(rewards_s[:, None], (n_states, n_actions))

    def _compute_reward_matrix(
        self,
        reward_net,
        state_feat: jax.Array,
        n_states: int,
        n_actions: int,
    ) -> jax.Array:
        """Compute the anchored reward matrix and apply availability."""
        rewards = self._net_reward_matrix(reward_net, state_feat, n_states, n_actions)
        if self.reward_type == "state_action":
            assert self.anchor_action is not None
            rewards = rewards.at[:, int(self.anchor_action)].set(0.0)
        else:
            assert self.anchor_state is not None
            rewards = rewards.at[int(self.anchor_state), :].set(0.0)
        if self._action_mask_jax is not None:
            rewards = jnp.where(self._action_mask_jax, rewards, -1e9)
        return rewards

    def _train_mce(
        self,
        transitions: jnp.ndarray,
        empirical_sa: jnp.ndarray,
        n_states: int,
        n_actions: int,
    ) -> None:
        """Train the reward network against discounted occupancy moments."""
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=self.lr, weight_decay=1e-5),
        )
        opt_state = optimizer.init(eqx.filter(self._reward_net, eqx.is_array))
        problem = DDCProblem(
            num_states=n_states,
            num_actions=n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )
        bellman = SoftBellmanOperator(problem=problem, transitions=transitions)
        best_loss = float("inf")
        best_net = self._reward_net
        best_epoch = 0
        patience_counter = 0
        all_state_indices = jnp.arange(n_states)
        assert self._state_encoder is not None
        reward_net = self._reward_net

        from tqdm import tqdm

        pbar = tqdm(
            range(self.max_epochs),
            desc="MCE-IRL-NN",
            disable=not self.verbose,
            leave=True,
        )
        for epoch in pbar:
            state_feat = self._state_encoder(all_state_indices)
            reward_matrix = self._compute_reward_matrix(reward_net, state_feat, n_states, n_actions)
            if self.inner_solver == "hybrid":
                result = hybrid_iteration(
                    bellman, reward_matrix, tol=self.inner_tol, max_iter=self.inner_max_iter
                )
            else:
                result = value_iteration(
                    bellman, reward_matrix, tol=self.inner_tol, max_iter=self.inner_max_iter
                )
            if not result.converged or not np.isfinite(result.final_error):
                self.termination_reason_ = "inner_solver_failed"
                break
            policy_sa = self._forward_pass(result.policy, transitions, n_states, self.discount)
            grad_r = jax.lax.stop_gradient(policy_sa - empirical_sa)

            def surrogate_loss(net):
                reward = self._compute_reward_matrix(net, state_feat, n_states, n_actions)
                return jnp.sum(reward * grad_r)

            _, grads = eqx.filter_value_and_grad(surrogate_loss)(reward_net)
            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(reward_net, eqx.is_array)
            )
            reward_net = eqx.apply_updates(reward_net, updates)
            loss_val = float(jnp.sum(grad_r**2))
            if not np.isfinite(loss_val):
                self.termination_reason_ = "nonfinite_training_loss"
                break
            feature_diff = float(jnp.linalg.norm(grad_r))
            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                fdiff=f"{feature_diff:.4f}",
                best=f"{best_loss:.4f}",
                no_imp=patience_counter,
            )
            if loss_val < best_loss - self.improvement_tol:
                best_loss = loss_val
                best_net = reward_net
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.termination_reason_ = "training_plateau"
                    break
        else:
            self.termination_reason_ = "max_epochs_reached"

        self._reward_net = best_net
        self.n_epochs_ = epoch + 1
        self.best_epoch_ = best_epoch
        self.training_loss_ = best_loss
        self.feature_difference_ = float(np.sqrt(best_loss))

    def _forward_pass(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        n_states: int,
        discount: float,
    ) -> jnp.ndarray:
        """Compute normalized discounted state-action visitation.

        Parameters
        ----------
        policy : jnp.ndarray
            Policy pi(a|s), shape (n_states, n_actions).
        transitions : jnp.ndarray
            Transition matrices ``P(s'|s,a)``, shape (n_actions, n_states, n_states).
        n_states : int
            Number of states.
        discount : float
            Discount factor.

        Returns
        -------
        jnp.ndarray
            State-action visitation frequencies, shape (n_states, n_actions).
        """
        problem = DDCProblem(
            num_states=n_states,
            num_actions=policy.shape[1],
            discount_factor=discount,
            scale_parameter=1.0,
        )
        return compute_state_action_visitation(
            policy,
            transitions,
            problem,
            self._initial_distribution,
        )

    # ------------------------------------------------------------------
    # Post-training extraction
    # ------------------------------------------------------------------

    def _extract_final(
        self,
        transitions: jnp.ndarray,
        n_states: int,
        n_actions: int,
    ) -> None:
        """Extract fitted reward, policy, values, and convergence diagnostics."""
        all_state_indices = jnp.arange(n_states)
        assert self._state_encoder is not None
        state_feat = self._state_encoder(all_state_indices)
        reward_matrix = self._compute_reward_matrix(
            self._reward_net, state_feat, n_states, n_actions
        )
        problem = DDCProblem(
            num_states=n_states,
            num_actions=n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )
        bellman = SoftBellmanOperator(problem=problem, transitions=transitions)
        if self.inner_solver == "hybrid":
            result = hybrid_iteration(
                bellman, reward_matrix, tol=self.inner_tol, max_iter=self.inner_max_iter
            )
        else:
            result = value_iteration(
                bellman, reward_matrix, tol=self.inner_tol, max_iter=self.inner_max_iter
            )

        self.policy_ = np.asarray(result.policy)
        self.value_ = np.asarray(result.V)
        self.reward_ = (
            np.asarray(reward_matrix)
            if self.reward_type == "state_action"
            else np.asarray(reward_matrix[:, 0])
        )
        self.bellman_residual_ = float(result.final_error)
        if self._empirical_sa is not None:
            policy_sa = self._forward_pass(result.policy, transitions, n_states, self.discount)
            residual = self._empirical_sa - policy_sa
            self.feature_difference_ = float(jnp.linalg.norm(residual))
            self.occupancy_moment_residual_ = float(jnp.max(jnp.abs(residual)))
        finite = all(
            np.isfinite(value).all() for value in (self.policy_, self.value_, self.reward_)
        )
        occupancy_ok = (
            self.occupancy_moment_residual_ is not None
            and self.occupancy_moment_residual_ <= self.occupancy_tol
        )
        self.converged_ = bool(result.converged and finite and occupancy_ok)
        if not result.converged:
            self.termination_reason_ = "inner_solver_failed"
        elif not finite:
            self.termination_reason_ = "nonfinite_fitted_state"
        elif not occupancy_ok:
            self.termination_reason_ = "occupancy_tolerance_not_met"
        elif self.termination_reason_ == "max_epochs_reached":
            self.termination_reason_ = "converged_at_max_epochs"
        else:
            self.termination_reason_ = "converged"

    def _project_onto_features(
        self,
        features: RewardSpec | np.ndarray,
        n_states: int,
        n_actions: int,
    ) -> None:
        """Project the fitted reward map onto a supplied linear basis.

        This is descriptive geometry of one fitted reward map. It is not
        sampling inference for the neural estimator.
        """
        if isinstance(features, RewardSpec):
            feat_3d = np.asarray(features.feature_matrix, dtype=np.float64)
            names = list(features.parameter_names)
        else:
            feat_arr = np.asarray(features, dtype=np.float64)
            if feat_arr.ndim == 3:
                feat_3d = feat_arr
            elif feat_arr.ndim == 2:
                feat_3d = np.broadcast_to(
                    feat_arr[:, None, :],
                    (feat_arr.shape[0], n_actions, feat_arr.shape[1]),
                ).copy()
            else:
                raise ValueError(
                    f"features must be 2D (S, K) or 3D (S, A, K), got {feat_arr.ndim}D"
                )
            names = self.feature_names or [f"f{i}" for i in range(feat_3d.shape[-1])]
        if feat_3d.shape[:2] != (n_states, n_actions):
            raise ValueError(
                "features must align with states and actions; "
                f"got {feat_3d.shape[:2]}, expected {(n_states, n_actions)}"
            )
        rewards = np.asarray(self.reward_matrix_, dtype=np.float64)
        if self.reward_type == "state_action":
            mask = (
                self.action_mask_
                if self.action_mask_ is not None
                else np.ones((n_states, n_actions), dtype=bool)
            )
            phi = feat_3d[mask]
            target = rewards[mask]
        else:
            phi = feat_3d[:, 0, :]
            target = rewards[:, 0]
        rank = int(np.linalg.matrix_rank(phi))
        condition = float(np.linalg.cond(phi))
        if rank < phi.shape[1]:
            raise ValueError(
                f"projection feature matrix is rank deficient: rank {rank} < {phi.shape[1]}"
            )
        theta, _, r2 = self._project_parameters(phi, target)
        residual = (
            target
            - np.column_stack([phi, np.ones(phi.shape[0])])
            @ np.r_[
                theta,
                np.linalg.lstsq(np.column_stack([phi, np.ones(phi.shape[0])]), target, rcond=None)[
                    0
                ][-1],
            ]
        )
        self.params_ = {name: float(value) for name, value in zip(names, theta)}
        self.coef_ = theta
        self.projection_r2_ = float(r2)
        self.projection_diagnostics_ = {
            "rank": rank,
            "num_features": int(phi.shape[1]),
            "condition_number": condition,
            "r_squared": float(r2),
            "residual_scale": float(np.sqrt(np.mean(residual**2))),
            "sampling_inference": False,
        }
        self.se_ = None
        self.pvalues_ = None

    # ------------------------------------------------------------------
    # Prediction methods
    # ------------------------------------------------------------------

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        """Reward matrix R(s,a) of shape (n_states, n_actions).

        For ``reward_type="state_action"``, ``self.reward_`` already has
        shape (n_states, n_actions) and is returned directly.  For
        ``reward_type="state"``, the state-only reward is broadcast to all
        actions.
        """
        if self.reward_ is None:
            return None
        if self.reward_.ndim == 2:
            return self.reward_
        # State-only reward: broadcast to all actions
        n_actions = self._n_actions or self.n_actions
        assert n_actions is not None
        return np.tile(self.reward_[:, np.newaxis], (1, n_actions))

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

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.policy_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        states = np.asarray(states, dtype=np.int64)
        return np.asarray(self.policy_[states])

    def simulate(
        self,
        n_trajectories: int,
        n_periods: int,
        *,
        seed: int = 0,
        initial_distribution: np.ndarray | None = None,
    ) -> TrajectoryPanel:
        """Simulate trajectories from the fitted policy and transition kernel."""
        if self.policy_ is None or self.transitions_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if n_trajectories < 1 or n_periods < 1:
            raise ValueError("n_trajectories and n_periods must be positive")
        distribution = (
            np.asarray(initial_distribution, dtype=float)
            if initial_distribution is not None
            else np.asarray(self._initial_distribution, dtype=float)
        )
        if distribution.shape != (self.policy_.shape[0],):
            raise ValueError(f"initial_distribution must have shape {(self.policy_.shape[0],)}")
        if np.any(distribution < 0) or not np.isclose(distribution.sum(), 1.0):
            raise ValueError("initial_distribution must be a probability vector")
        rng = np.random.default_rng(seed)
        trajectories: list[Trajectory] = []
        for individual in range(n_trajectories):
            current = int(rng.choice(self.policy_.shape[0], p=distribution))
            states: list[int] = []
            actions: list[int] = []
            next_states: list[int] = []
            for _ in range(n_periods):
                chosen = int(rng.choice(self.policy_.shape[1], p=self.policy_[current]))
                following = int(
                    rng.choice(
                        self.policy_.shape[0],
                        p=self.transitions_[chosen, current],
                    )
                )
                states.append(current)
                actions.append(chosen)
                next_states.append(following)
                current = following
            trajectories.append(
                Trajectory(
                    states=jnp.asarray(states),
                    actions=jnp.asarray(actions),
                    next_states=jnp.asarray(next_states),
                    individual_id=individual,
                )
            )
        return TrajectoryPanel(trajectories=trajectories)

    def counterfactual(
        self,
        *,
        reward_delta: np.ndarray | None = None,
        transitions: np.ndarray | None = None,
        action_mask: np.ndarray | None = None,
        description: str | None = None,
    ) -> CounterfactualResult:
        """Re-solve one reward, transition, or action-availability change."""
        if self.policy_ is None or self.value_ is None or self.transitions_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        supplied = sum(value is not None for value in (reward_delta, transitions, action_mask))
        if supplied != 1:
            raise ValueError("supply exactly one of reward_delta, transitions, or action_mask")
        baseline_reward = np.asarray(self.reward_matrix_, dtype=np.float64)
        changed_reward = baseline_reward.copy()
        changed_transitions = np.asarray(self.transitions_, dtype=np.float64)
        changed_mask = np.asarray(self.action_mask_, dtype=bool)
        changed_primitive: str
        cf_type = CounterfactualType.ENVIRONMENT_CHANGE

        if reward_delta is not None:
            delta = np.asarray(reward_delta, dtype=np.float64)
            if self.reward_type == "state" and delta.shape == (baseline_reward.shape[0],):
                delta = np.broadcast_to(delta[:, None], baseline_reward.shape)
            if delta.shape != baseline_reward.shape:
                raise ValueError(f"reward_delta must have shape {baseline_reward.shape}")
            if not np.isfinite(delta).all():
                raise ValueError("reward_delta must be finite")
            changed_reward += delta
            if self.reward_type == "state_action":
                assert self.anchor_action is not None
                changed_reward[:, int(self.anchor_action)] = 0.0
            else:
                assert self.anchor_state is not None
                changed_reward[int(self.anchor_state), :] = 0.0
            changed_primitive = "reward"
            cf_type = CounterfactualType.REWARD_CHANGE
        elif transitions is not None:
            candidate = np.asarray(transitions, dtype=np.float64)
            if candidate.shape != changed_transitions.shape:
                raise ValueError(f"transitions must have shape {changed_transitions.shape}")
            if np.any(candidate < 0) or not np.isfinite(candidate).all():
                raise ValueError("transitions must be finite and nonnegative")
            if np.max(np.abs(candidate.sum(axis=2) - 1.0)) > 1e-6:
                raise ValueError("transition rows must sum to one")
            changed_transitions = candidate
            changed_primitive = "transitions"
        else:
            candidate = np.asarray(action_mask, dtype=bool)
            if candidate.shape != changed_mask.shape:
                raise ValueError(f"action_mask must have shape {changed_mask.shape}")
            if np.any(candidate.sum(axis=1) == 0):
                raise ValueError("every state must retain at least one available action")
            if np.any(candidate & ~changed_mask):
                raise ValueError(
                    "action_mask counterfactuals may only remove actions that "
                    "were available during fitting"
                )
            changed_mask = candidate
            changed_primitive = "action_availability"

        solved_reward = np.where(changed_mask, changed_reward, -1e9)
        problem = DDCProblem(
            num_states=changed_reward.shape[0],
            num_actions=changed_reward.shape[1],
            discount_factor=self.discount,
            scale_parameter=1.0,
        )
        operator = SoftBellmanOperator(
            problem=problem,
            transitions=jnp.asarray(changed_transitions),
        )
        if self.inner_solver == "hybrid":
            result = hybrid_iteration(
                operator,
                jnp.asarray(solved_reward),
                tol=self.inner_tol,
                max_iter=self.inner_max_iter,
            )
        else:
            result = value_iteration(
                operator,
                jnp.asarray(solved_reward),
                tol=self.inner_tol,
                max_iter=self.inner_max_iter,
            )
        if not result.converged:
            raise RuntimeError("counterfactual Bellman solve did not converge")
        baseline_policy = jnp.asarray(self.policy_)
        baseline_value = jnp.asarray(self.value_)
        value_change = result.V - baseline_value
        return CounterfactualResult(
            baseline_policy=baseline_policy,
            counterfactual_policy=result.policy,
            baseline_value=baseline_value,
            counterfactual_value=result.V,
            policy_change=result.policy - baseline_policy,
            value_change=value_change,
            welfare_change=float(jnp.mean(value_change)),
            counterfactual_type=cf_type,
            description=description or f"Neural MCE-IRL {changed_primitive} counterfactual",
            metadata={
                "changed_primitive": changed_primitive,
                "reward_normalization": (
                    f"anchor_action={self.anchor_action}"
                    if self.reward_type == "state_action"
                    else f"anchor_state={self.anchor_state}"
                ),
                "neural_weights_interpretable": False,
            },
            transitions=jnp.asarray(self.transitions_),
            counterfactual_transitions=jnp.asarray(changed_transitions),
            params=dict(self.params_ or {}),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def conf_int(self, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        """Confidence intervals are not supported for neural reward maps."""
        del alpha
        raise NotImplementedError(
            "MCEIRLNeural does not report sampling confidence intervals. "
            "Projected coefficients describe one fitted reward map; use "
            "repeated panel and training seeds to assess stability."
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a concise report of the fitted neural reward map."""
        if self.policy_ is None:
            return "MCEIRLNeural: Not fitted yet. Call fit() first."
        assert self.diagnostics_ is not None
        lines = [
            "MCEIRLNeural (Neural MCE-IRL)",
            f"Observations: {self.n_observations_}",
            f"States x actions: {self._n_states} x {self._n_actions}",
            f"Reward type: {self.reward_type}",
            f"Normalization: {self.diagnostics_['normalization']}",
            f"Network: {self.reward_num_layers} layers x {self.reward_hidden_dim} hidden",
            f"Epochs: {self.n_epochs_} (best {self.best_epoch_})",
            f"Converged: {self.converged_}",
            f"Termination: {self.termination_reason_}",
            f"Occupancy residual: {self.occupancy_moment_residual_:.6g}",
            f"Bellman residual: {self.bellman_residual_:.6g}",
            "Sampling inference: not supported",
        ]
        if self.projection_diagnostics_ is not None:
            lines.extend(
                [
                    f"Projection R-squared: {self.projection_r2_:.6g}",
                    "Projected coefficients: descriptive only",
                ]
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self.policy_ is not None
        return (
            f"MCEIRLNeural(n_states={self._n_states or self.n_states}, "
            f"n_actions={self._n_actions or self.n_actions}, "
            f"discount={self.discount}, "
            f"fitted={fitted})"
        )
