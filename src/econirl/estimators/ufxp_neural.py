"""Neural-utility UFXP (Oguz and Bray 2026).

The unnested fixed point trained with a neural utility. The linear UFXP scores
random projections of the Bellman first-order conditions and, because the value
function is removed by a precomputed dual (Proposition 2), solves a single
least-squares problem. This estimator keeps the same precompute and replaces the
closed-form solve with a gradient loop over a neural utility ``u_w(s, a)``.

The point of the method: the dual ``lam = (I - beta F_P')^{-1} W`` is the only
linear solve, computed once before training. Every gradient step is one network
forward pass through the objective

    Q(w) = sum_i ( b_i - <Z_i, du_w> + lam_i' u_P^w )^2,
    du_w[s, a] = u_w(s, a) - u_w(s, ref),
    u_P^w[s]   = sum_a P_hat[s, a] u_w(s, a),

with ``b_i`` precomputed and utility-free. No Bellman equation is solved inside
the loop, which is what lets a neural utility train at scale.

Scope (Phase 1): the neural-utility training and behavior recovery, with the
learned utility projected onto features for an interpretable coefficient vector
and pseudo standard errors. The paper's optimal-weight second step (neural
OUFXP) and its efficient standard errors are not implemented here; the standard
errors reported are projection pseudo-SEs, not the efficient UFXP variance.
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from econirl.core.types import DDCProblem, Panel, TrajectoryPanel
from econirl.estimation.ufxp import _ufxp_precompute, _ufxp_random_duals
from econirl.estimators.neural_base import NeuralEstimatorMixin
from econirl.preferences.linear import LinearUtility


class _UtilityNet(eqx.Module):
    """MLP utility ``u_w(phi)`` mapping a feature vector in R^K to a scalar."""

    layers: list
    output_layer: eqx.nn.Linear

    def __init__(self, n_features: int, hidden_dim: int, num_layers: int, *, key):
        keys = jax.random.split(key, num_layers + 1)
        layers = []
        in_dim = n_features
        for i in range(num_layers):
            layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys[i]))
            in_dim = hidden_dim
        self.layers = layers
        self.output_layer = eqx.nn.Linear(in_dim, 1, key=keys[-1])

    def __call__(self, phi_sa: jax.Array) -> jax.Array:
        x = phi_sa
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        return self.output_layer(x).squeeze(-1)

    def utility_matrix(self, phi: jax.Array) -> jax.Array:
        """Utility ``u_w(s, a)`` for every state-action pair; ``phi`` is (S, A, K)."""
        S, A, K = phi.shape
        u = jax.vmap(self)(phi.reshape(S * A, K))
        return u.reshape(S, A)


class NeuralUFXP(NeuralEstimatorMixin):
    """Neural-utility UFXP estimator (Oguz and Bray 2026).

    Trains a neural utility ``u_w(s, a)`` by minimizing the UFXP random-projection
    objective, reusing the linear estimator's precomputed dual so no Bellman
    equation is solved during training.

    Parameters
    ----------
    n_states, n_actions : int, optional
        Sizes of the state and action spaces. Inferred from the data if None.
    discount : float, default=0.95
        Discount factor ``beta``.
    scale : float, default=1.0
        Logit scale ``sigma``.
    num_projections : int, default=64
        Number of random projections ``m``.
    reward_hidden_dim : int, default=64
        Hidden width of the utility network.
    reward_num_layers : int, default=2
        Hidden depth of the utility network.
    max_epochs : int, default=2000
        Adam steps over the projection objective.
    lr : float, default=1e-2
        Adam learning rate.
    gradient_clip : float, default=10.0
        Global-norm gradient clip (<=0 disables).
    ccp_min_count : int, default=1
        Minimum visits for a state's first-order conditions to be scored.
    ccp_smoothing : float, default=1e-6
        Additive smoothing for the frequency CCPs.
    seed : int, default=0
        Seed for the projections and the network initialization.
    verbose : bool, default=False
        Print the objective during training.

    Attributes
    ----------
    policy_ : numpy.ndarray
        Estimated choice probabilities, shape (n_states, n_actions).
    value_ : numpy.ndarray
        Estimated value function, shape (n_states,).
    reward_ : numpy.ndarray
        Learned utility ``u_w(s, a)``, shape (n_states, n_actions).
    params_ : dict
        The learned utility projected onto the features. The objective constrains
        the choice-relevant utility, not the utility level, so this is a
        best-effort linear summary of a partially identified function; a low
        ``projection_r2_`` flags that the utility is not linear in the features.
    se_ : dict
        Projection pseudo standard errors (not the efficient UFXP variance).
    coef_ : numpy.ndarray
        Projected coefficients in array form.
    projection_r2_ : float
        R-squared of the feature projection.
    converged_ : bool
        Whether the objective decreased to a finite value.
    """

    def __init__(
        self,
        n_states: int | None = None,
        n_actions: int | None = None,
        discount: float = 0.95,
        scale: float = 1.0,
        num_projections: int = 64,
        reward_hidden_dim: int = 64,
        reward_num_layers: int = 2,
        max_epochs: int = 2000,
        lr: float = 1e-2,
        gradient_clip: float = 10.0,
        ccp_min_count: int = 1,
        ccp_smoothing: float = 1e-6,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.scale = scale
        self.num_projections = num_projections
        self.reward_hidden_dim = reward_hidden_dim
        self.reward_num_layers = reward_num_layers
        self.max_epochs = max_epochs
        self.lr = lr
        self.gradient_clip = gradient_clip
        self.ccp_min_count = ccp_min_count
        self.ccp_smoothing = ccp_smoothing
        self.seed = seed
        self.verbose = verbose

        self.policy_ = None
        self.value_ = None
        self.reward_ = None
        self.params_ = None
        self.se_ = None
        self.coef_ = None
        self.projection_r2_ = None
        self.converged_ = None
        self.n_epochs_ = None

    def _extract_panel(self, data, state, action, id) -> TrajectoryPanel:
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError("state, action, and id column names are required "
                                 "when data is a DataFrame")
            return TrajectoryPanel.from_dataframe(data, state=state, action=action, id=id)
        if isinstance(data, (Panel, TrajectoryPanel)):
            return TrajectoryPanel.from_panel(data)
        raise TypeError(f"data must be a DataFrame, Panel, or TrajectoryPanel, got {type(data)}")

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        features: np.ndarray | None = None,
        transitions: np.ndarray | None = None,
    ) -> "NeuralUFXP":
        """Fit the neural utility to data.

        Parameters
        ----------
        data : pandas.DataFrame or Panel or TrajectoryPanel
            Panel of observed choices.
        state, action, id : str, optional
            Column names (required when ``data`` is a DataFrame).
        features : numpy.ndarray
            Reward features ``phi(s, a)`` of shape (n_states, n_actions, K). The
            utility network maps each feature vector to a scalar utility, so the
            features set the inputs the network can combine.
        transitions : numpy.ndarray
            Transition matrices ``P(s'|s,a)`` of shape (n_actions, n_states, n_states).
        """
        if features is None:
            raise ValueError("NeuralUFXP requires features (n_states, n_actions, K).")
        if transitions is None:
            raise ValueError("NeuralUFXP requires transitions (n_actions, n_states, n_states).")
        t0 = time.time()

        phi = np.asarray(getattr(features, "features", features), dtype=np.float64)
        names = list(getattr(features, "names", [])) or [f"theta_{i}" for i in range(phi.shape[2])]
        panel = self._extract_panel(data, state, action, id)
        A, S, _ = np.asarray(transitions).shape
        problem = DDCProblem(num_states=S, num_actions=A,
                             discount_factor=self.discount, scale_parameter=self.scale)
        utility = LinearUtility(feature_matrix=jnp.asarray(phi), parameter_names=names)

        # Reuse the linear estimator's theta-independent precompute and duals.
        pre = _ufxp_precompute(panel, utility, transitions, problem,
                               self.ccp_min_count, self.ccp_smoothing)
        Z, lam = _ufxp_random_duals(pre, self.num_projections, self.seed)
        # b_i = <Z_i, logratio> + lam_i' ent  -- precomputed, utility-free.
        b = np.einsum("msa,sa->m", Z, pre.logratio) + lam.T @ pre.ent

        phi_j = jnp.asarray(pre.phi)
        Z_j = jnp.asarray(Z)
        lamT = jnp.asarray(lam.T)        # (m, S)
        b_j = jnp.asarray(b)             # (m,)
        P_j = jnp.asarray(pre.P)         # (S, A)
        ref = pre.ref

        key = jax.random.PRNGKey(self.seed)
        net = _UtilityNet(phi.shape[2], self.reward_hidden_dim,
                          self.reward_num_layers, key=key)

        def objective(model):
            uall = model.utility_matrix(phi_j)            # (S, A)
            du = uall[:, :ref] - uall[:, ref:ref + 1]     # (S, A-1)
            u_P = jnp.sum(P_j * uall, axis=1)             # (S,)
            resid = b_j - jnp.einsum("msa,sa->m", Z_j, du) + lamT @ u_P
            return jnp.sum(resid ** 2)

        transforms = []
        if self.gradient_clip > 0:
            transforms.append(optax.clip_by_global_norm(self.gradient_clip))
        transforms.append(optax.adam(self.lr))
        opt = optax.chain(*transforms)
        opt_state = opt.init(eqx.filter(net, eqx.is_inexact_array))

        @eqx.filter_jit
        def step(model, opt_state):
            loss, grads = eqx.filter_value_and_grad(objective)(model)
            updates, opt_state = opt.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        init_loss = float(objective(net))
        last = init_loss
        for epoch in range(self.max_epochs):
            net, opt_state, loss = step(net, opt_state)
            last = float(loss)
            if self.verbose and epoch % 200 == 0:
                print(f"  epoch {epoch:5d}  Q={last:.6e}")
        self.n_epochs_ = self.max_epochs
        self.converged_ = bool(np.isfinite(last) and last <= init_loss)

        # One soft-Bellman solve at the learned utility for policy and value.
        from econirl.core.bellman import SoftBellmanOperator
        from econirl.core.solvers import hybrid_iteration

        reward_mat = net.utility_matrix(phi_j)
        op = SoftBellmanOperator(problem, jnp.asarray(transitions, dtype=jnp.float64))
        sol = hybrid_iteration(op, reward_mat, tol=1e-8, max_iter=5000)
        self.policy_ = np.asarray(sol.policy)
        self.value_ = np.asarray(sol.V)
        self.reward_ = np.asarray(reward_mat)

        # Project the learned utility onto the features for interpretable coefs.
        theta, se, r2 = self._project_parameters(
            phi.reshape(S * A, phi.shape[2]), np.asarray(reward_mat).reshape(S * A))
        self.coef_ = np.asarray(theta)
        self.params_ = dict(zip(names, [float(v) for v in theta]))
        self.se_ = dict(zip(names, [float(v) for v in se]))
        self.projection_r2_ = float(r2)
        self._fit_time = time.time() - t0
        return self
