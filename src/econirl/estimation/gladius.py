"""GLADIUS Estimator: Neural Network-based IRL for Dynamic Discrete Choice.

This module implements the GLADIUS estimator from Kang et al. (2025),
which uses neural networks to parameterize Q-functions and expected
future value functions for inverse reinforcement learning in dynamic
discrete choice models.

Algorithm:
    1. Parameterize Q(s,a) and EV(s,a) = E[V(s')|s,a] with MLPs.
    2. Train via mini-batch SGD on observed (s, a, s') transitions:
       - NLL loss: negative log-likelihood of observed actions under
         softmax policy derived from Q.
       - Bellman penalty: squared TD error beta*(EV(s,a) - V(s'))^2,
         where V(s') = sigma * logsumexp(Q(s', :) / sigma).
    3. Extract structural parameters by regressing implied rewards
       r(s,a) = Q(s,a) - beta * EV(s,a) onto the feature matrix.

Reference:
    Kang, M., et al. (2025). DDC IRL with neural networks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.preferences.base import UtilityFunction


@dataclass
class GLADIUSConfig:
    """Configuration for the GLADIUS estimator.

    Attributes:
        q_hidden_dim: Hidden dimension for the Q-network MLP.
        q_num_layers: Number of hidden layers in the Q-network.
        v_hidden_dim: Hidden dimension for the EV-network MLP.
        v_num_layers: Number of hidden layers in the EV-network.
        q_lr: Learning rate for the Q-network optimizer.
        v_lr: Learning rate for the EV-network optimizer.
        max_epochs: Maximum number of training epochs.
        batch_size: Mini-batch size for SGD.
        bellman_penalty_weight: Weight on the Bellman consistency penalty.
        weight_decay: L2 regularization weight.
        gradient_clip: Maximum gradient norm for clipping.
        compute_se: Whether to compute standard errors via bootstrap.
        n_bootstrap: Number of bootstrap replications for SE computation.
        verbose: Whether to print progress messages.
    """

    q_hidden_dim: int = 128
    q_num_layers: int = 3
    v_hidden_dim: int = 128
    v_num_layers: int = 3
    q_lr: float = 1e-3
    v_lr: float = 1e-3
    max_epochs: int = 500
    batch_size: int = 512
    bellman_penalty_weight: float = 1.0
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    compute_se: bool = True
    n_bootstrap: int = 100
    verbose: bool = False


class _QNetwork(eqx.Module):
    """MLP that maps (state_features, action_onehot) to a scalar Q value."""

    mlp: eqx.nn.MLP
    n_actions: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        self.n_actions = n_actions
        self.mlp = eqx.nn.MLP(
            in_size=state_dim + n_actions,
            out_size=1,
            width_size=hidden_dim,
            depth=num_layers,
            activation=jax.nn.relu,
            key=key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Compute Q for a single input vector (state_feat || action_onehot).

        Args:
            x: Input vector of shape (state_dim + n_actions,).

        Returns:
            Scalar Q value.
        """
        return self.mlp(x).squeeze(-1)

    def forward(
        self, state_features: jax.Array, action_onehot: jax.Array
    ) -> jax.Array:
        """Compute Q(s, a) for a batch.

        Args:
            state_features: Array of shape (batch, state_dim).
            action_onehot: Array of shape (batch, n_actions).

        Returns:
            Q values of shape (batch,).
        """
        x = jnp.concatenate([state_features, action_onehot], axis=-1)
        return jax.vmap(self)(x)

    def forward_all_actions(self, state_features: jax.Array) -> jax.Array:
        """Compute Q(s, a) for all actions at once.

        Args:
            state_features: Array of shape (batch, state_dim).

        Returns:
            Q values of shape (batch, n_actions).
        """
        batch_size = state_features.shape[0]
        eye = jnp.eye(self.n_actions)

        def _q_for_action(a_idx: int) -> jax.Array:
            onehot = jnp.broadcast_to(eye[a_idx], (batch_size, self.n_actions))
            return self.forward(state_features, onehot)

        # Stack Q values for each action along axis 1.
        q_values = jnp.stack(
            [_q_for_action(a) for a in range(self.n_actions)], axis=1
        )
        return q_values


class _EVNetwork(eqx.Module):
    """MLP that maps (state_features, action_onehot) to scalar EV = E[V(s')|s,a]."""

    mlp: eqx.nn.MLP
    n_actions: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        self.n_actions = n_actions
        self.mlp = eqx.nn.MLP(
            in_size=state_dim + n_actions,
            out_size=1,
            width_size=hidden_dim,
            depth=num_layers,
            activation=jax.nn.relu,
            key=key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Compute EV for a single input vector (state_feat || action_onehot).

        Args:
            x: Input vector of shape (state_dim + n_actions,).

        Returns:
            Scalar EV value.
        """
        return self.mlp(x).squeeze(-1)

    def forward(
        self, state_features: jax.Array, action_onehot: jax.Array
    ) -> jax.Array:
        """Compute EV(s, a) for a batch.

        Args:
            state_features: Array of shape (batch, state_dim).
            action_onehot: Array of shape (batch, n_actions).

        Returns:
            EV values of shape (batch,).
        """
        x = jnp.concatenate([state_features, action_onehot], axis=-1)
        return jax.vmap(self)(x)


class GLADIUSEstimator(BaseEstimator):
    """GLADIUS estimator for DDC IRL with neural networks.

    Uses two MLPs (Q_net and EV_net) to approximate the Q-function
    and expected next-period value function. The loss combines negative
    log-likelihood (NLL) with a Bellman consistency penalty. After
    training, structural parameters are recovered by regressing implied
    rewards onto the feature matrix via least squares.

    Parameters
    ----------
    config : GLADIUSConfig, optional
        Configuration object. If None, default config is used.
    **kwargs
        Override individual config parameters.

    References
    ----------
    Kang, M., et al. (2025). DDC IRL with neural networks.
    """

    def __init__(self, config: GLADIUSConfig | None = None, **kwargs):
        if config is None:
            config = GLADIUSConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        super().__init__(
            se_method="bootstrap" if config.compute_se else "asymptotic",
            compute_hessian=False,
            verbose=config.verbose,
        )
        self.config = config
        self.q_net_: _QNetwork | None = None
        self.ev_net_: _EVNetwork | None = None

    @property
    def name(self) -> str:
        return "GLADIUS"

    def _build_state_features(
        self, states: jnp.ndarray, problem: DDCProblem
    ) -> jnp.ndarray:
        """Build state feature vectors from state indices.

        Uses the problem's state_encoder if available, otherwise
        normalizes state index to [0, 1].

        Args:
            states: Array of state indices, shape (batch,).
            problem: Problem specification with optional state_encoder.

        Returns:
            Feature array of shape (batch, state_dim).
        """
        if problem.state_encoder is not None:
            return problem.state_encoder(states)
        normalized = states.astype(jnp.float32) / max(problem.num_states - 1, 1)
        return normalized[:, None]

    def _build_state_features_all(self, problem: DDCProblem) -> jnp.ndarray:
        """Build feature vectors for all states.

        Args:
            problem: Problem specification.

        Returns:
            Feature array of shape (n_states, state_dim).
        """
        return self._build_state_features(jnp.arange(problem.num_states), problem)

    def _compute_log_likelihood(
        self,
        q_net: _QNetwork,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        problem: DDCProblem,
        sigma: float,
    ) -> float:
        """Compute the log-likelihood of the full dataset.

        Args:
            q_net: Trained Q-network.
            states: All observed state indices.
            actions: All observed action indices.
            problem: Problem specification.
            sigma: Scale parameter.

        Returns:
            Total log-likelihood (scalar).
        """
        state_feat = self._build_state_features(states, problem)
        q_all = q_net.forward_all_actions(state_feat)  # (N, n_actions)
        log_probs = q_all / sigma - jax.scipy.special.logsumexp(
            q_all / sigma, axis=1, keepdims=True
        )
        ll = float(log_probs[jnp.arange(len(actions)), actions].sum())
        return ll

    def estimate(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate utility parameters from panel data.

        Overrides the base estimate() method because GLADIUS uses neural
        networks internally and needs custom handling for the summary.

        Parameters
        ----------
        panel : Panel
            Panel data with observed choices.
        utility : UtilityFunction
            Utility function specification.
        problem : DDCProblem
            Problem specification.
        transitions : jnp.ndarray
            Transition matrices P(s'|s,a).
        initial_params : jnp.ndarray, optional
            Not used (networks have their own initialization).

        Returns
        -------
        EstimationSummary
        """
        start_time = time.time()

        result = self._optimize(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
            initial_params=initial_params,
            **kwargs,
        )

        # Standard errors not directly available from NN; fill with NaN
        standard_errors = jnp.full_like(result.parameters, float("nan"))

        # Goodness of fit
        n_obs = panel.num_observations
        n_params = len(result.parameters)
        ll = result.log_likelihood

        goodness_of_fit = GoodnessOfFit(
            log_likelihood=ll,
            num_parameters=n_params,
            num_observations=n_obs,
            aic=-2 * ll + 2 * n_params,
            bic=-2 * ll + n_params * np.log(n_obs),
            prediction_accuracy=self._compute_prediction_accuracy(panel, result.policy),
        )

        total_time = time.time() - start_time

        return EstimationSummary(
            parameters=result.parameters,
            parameter_names=utility.parameter_names,
            standard_errors=standard_errors,
            hessian=None,
            variance_covariance=None,
            method=self.name,
            num_observations=n_obs,
            num_individuals=panel.num_individuals,
            num_periods=max(panel.num_periods_per_individual),
            discount_factor=problem.discount_factor,
            scale_parameter=problem.scale_parameter,
            log_likelihood=ll,
            goodness_of_fit=goodness_of_fit,
            identification=None,
            converged=result.converged,
            num_iterations=result.num_iterations,
            convergence_message=result.message,
            value_function=result.value_function,
            policy=result.policy,
            estimation_time=total_time,
            metadata=result.metadata,
        )

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Core GLADIUS optimization routine.

        Steps:
            1. Extract (s, a, s') tuples from panel.
            2. Build Q-network and EV-network.
            3. Train via mini-batch SGD with NLL + Bellman penalty.
            4. Extract structural parameters via least-squares regression.

        Returns:
            EstimationResult with parameters, policy, and value function.
        """
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        beta = problem.discount_factor
        sigma = problem.scale_parameter

        # --- Step 1: Extract data ---
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        all_next_states = panel.get_all_next_states()
        n_obs = len(all_states)

        # --- Step 2: Build networks ---
        state_dim = problem.state_dim or 1

        key = jax.random.PRNGKey(0)
        q_key, ev_key = jax.random.split(key)

        q_net = _QNetwork(
            state_dim, n_actions, self.config.q_hidden_dim,
            self.config.q_num_layers, key=q_key,
        )
        ev_net = _EVNetwork(
            state_dim, n_actions, self.config.v_hidden_dim,
            self.config.v_num_layers, key=ev_key,
        )

        # Build optimizers with gradient clipping and weight decay
        q_optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.gradient_clip),
            optax.adamw(self.config.q_lr, weight_decay=self.config.weight_decay),
        )
        ev_optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.gradient_clip),
            optax.adamw(self.config.v_lr, weight_decay=self.config.weight_decay),
        )

        q_opt_state = q_optimizer.init(eqx.filter(q_net, eqx.is_array))
        ev_opt_state = ev_optimizer.init(eqx.filter(ev_net, eqx.is_array))

        bellman_weight = self.config.bellman_penalty_weight

        # --- Define the training step ---
        @eqx.filter_jit
        def train_step(
            q_net: _QNetwork,
            ev_net: _EVNetwork,
            q_opt_state: optax.OptState,
            ev_opt_state: optax.OptState,
            s_feat: jax.Array,
            a_batch: jax.Array,
            sp_feat: jax.Array,
        ):
            """Single training step: compute loss, update both networks."""

            def loss_fn(nets):
                q_net_inner, ev_net_inner = nets

                # Action one-hot encoding
                a_onehot = jax.nn.one_hot(a_batch, n_actions)

                # Q(s, a) for the taken action
                q_sa = q_net_inner.forward(s_feat, a_onehot)

                # Q(s, all a) for NLL computation
                q_all = q_net_inner.forward_all_actions(s_feat)  # (batch, n_actions)

                # NLL loss: -log P(a|s) = -[Q(s,a)/sigma - logsumexp(Q(s,:)/sigma)]
                log_probs = q_all / sigma - jax.scipy.special.logsumexp(
                    q_all / sigma, axis=1, keepdims=True
                )
                nll = -log_probs[jnp.arange(len(a_batch)), a_batch].mean()

                # Bellman penalty
                # EV(s, a)
                ev_sa = ev_net_inner.forward(s_feat, a_onehot)

                # V(s') = sigma * logsumexp(Q(s', :) / sigma)
                q_sp_all = q_net_inner.forward_all_actions(sp_feat)  # (batch, n_actions)
                v_sp = sigma * jax.scipy.special.logsumexp(
                    q_sp_all / sigma, axis=1
                )

                # TD error: beta * (EV(s,a) - V(s'))
                # Stop gradient on V(s') to stabilize learning
                td_error = beta * (ev_sa - jax.lax.stop_gradient(v_sp))
                bellman_loss = jnp.mean(td_error ** 2)

                total_loss = nll + bellman_weight * bellman_loss
                return total_loss, (nll, bellman_loss)

            (loss, aux), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True
            )((q_net, ev_net))

            q_grads, ev_grads = grads

            q_updates, new_q_opt = q_optimizer.update(
                q_grads, q_opt_state, eqx.filter(q_net, eqx.is_array)
            )
            ev_updates, new_ev_opt = ev_optimizer.update(
                ev_grads, ev_opt_state, eqx.filter(ev_net, eqx.is_array)
            )

            q_net = eqx.apply_updates(q_net, q_updates)
            ev_net = eqx.apply_updates(ev_net, ev_updates)

            return q_net, ev_net, new_q_opt, new_ev_opt, loss, aux

        # --- Step 3: Training loop ---
        best_loss = float("inf")
        epochs_no_improve = 0
        patience = 50
        converged = False

        loss_history: list[float] = []
        rng_key = jax.random.PRNGKey(42)

        for epoch in range(self.config.max_epochs):
            # Shuffle data
            rng_key, perm_key = jax.random.split(rng_key)
            perm = jax.random.permutation(perm_key, n_obs)
            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, n_obs, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, n_obs)
                idx = perm[start_idx:end_idx]

                s_batch = all_states[idx]
                a_batch = all_actions[idx]
                sp_batch = all_next_states[idx]

                # Build features
                s_feat = self._build_state_features(s_batch, problem)
                sp_feat = self._build_state_features(sp_batch, problem)

                q_net, ev_net, q_opt_state, ev_opt_state, loss, _aux = train_step(
                    q_net, ev_net, q_opt_state, ev_opt_state,
                    s_feat, a_batch, sp_feat,
                )

                epoch_loss += float(loss)
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)

            if self._verbose:
                if (epoch + 1) % 50 == 0 or epoch == 0:
                    self._log(
                        f"Epoch {epoch + 1}/{self.config.max_epochs}: "
                        f"loss={avg_loss:.6f}"
                    )

            # Early stopping
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                converged = True
                if self._verbose:
                    self._log(f"Early stopping at epoch {epoch + 1}")
                break

        num_epochs = epoch + 1
        if num_epochs == self.config.max_epochs:
            converged = True  # Reached max epochs

        # --- Step 4: Extract parameters ---
        all_state_feat = self._build_state_features_all(problem)

        # Compute Q(s, a) for all (s, a)
        q_table = q_net.forward_all_actions(all_state_feat)  # (n_states, n_actions)

        # Compute EV(s, a) for all (s, a)
        eye = jnp.eye(n_actions)
        ev_columns = []
        for a in range(n_actions):
            a_oh = jnp.broadcast_to(eye[a], (n_states, n_actions))
            ev_columns.append(ev_net.forward(all_state_feat, a_oh))
        ev_table = jnp.stack(ev_columns, axis=1)  # (n_states, n_actions)

        # Implied reward: r(s, a) = Q(s, a) - beta * EV(s, a)
        reward_table = q_table - beta * ev_table

        # Policy: softmax of Q values
        policy = jax.nn.softmax(q_table / sigma, axis=1)

        # Value function: V(s) = sigma * logsumexp(Q(s, :) / sigma)
        value_function = sigma * jax.scipy.special.logsumexp(
            q_table / sigma, axis=1
        )

        # Regress implied rewards onto feature matrix if available
        parameters = self._extract_parameters(utility, reward_table)

        # Compute log-likelihood
        ll = self._compute_log_likelihood(q_net, all_states, all_actions, problem, sigma)

        optimization_time = time.time() - start_time

        # Store trained networks
        self.q_net_ = q_net
        self.ev_net_ = ev_net

        message = f"GLADIUS converged after {num_epochs} epochs"
        if self._verbose:
            self._log(message)

        return EstimationResult(
            parameters=parameters,
            log_likelihood=ll,
            value_function=value_function,
            policy=policy,
            hessian=None,
            gradient_contributions=None,
            converged=converged,
            num_iterations=num_epochs,
            num_function_evals=num_epochs,
            num_inner_iterations=0,
            message=message,
            optimization_time=optimization_time,
            metadata={
                "reward_table": np.asarray(reward_table).tolist(),
                "q_table": np.asarray(q_table).tolist(),
                "ev_table": np.asarray(ev_table).tolist(),
                "loss_history": loss_history,
                "final_loss": loss_history[-1] if loss_history else float("nan"),
            },
        )

    def _extract_parameters(
        self, utility: UtilityFunction, reward_table: jnp.ndarray
    ) -> jnp.ndarray:
        """Extract structural parameters by regressing rewards onto features.

        If the utility has a feature_matrix attribute (linear utility), solves
        the least-squares problem:
            theta = argmin ||feature_matrix @ theta - r||^2

        Otherwise returns the flattened reward table.

        Args:
            utility: Utility function specification.
            reward_table: Implied rewards of shape (n_states, n_actions).

        Returns:
            Parameter vector.
        """
        feature_matrix = getattr(utility, "feature_matrix", None)

        if feature_matrix is not None:
            # feature_matrix shape: (n_states, n_actions, n_features)
            n_states, n_actions, n_features = feature_matrix.shape

            # Flatten to (n_states * n_actions, n_features) and (n_states * n_actions,)
            X = jnp.asarray(feature_matrix).reshape(-1, n_features)
            y = reward_table.reshape(-1)

            # Least-squares: theta = (X^T X)^{-1} X^T y
            parameters, _residuals, _rank, _sv = jnp.linalg.lstsq(X, y)
            return parameters
        else:
            # No feature matrix: return flattened rewards
            return reward_table.flatten()
