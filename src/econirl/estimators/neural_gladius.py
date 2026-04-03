"""NeuralGLADIUS: Context-aware Q-learning with Bellman consistency penalty.

Learns Q(s,a,ctx) and EV(s,a,ctx) via mini-batch training, then extracts
structural parameters by projecting implied rewards onto features.

No transition matrix is needed. Supports context conditioning through
pluggable state and context encoders.

Reference:
    Kang, M., et al. (2025). DDC IRL with neural networks.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pandas as pd
from scipy.stats import norm as scipy_norm

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import Panel, TrajectoryPanel
from econirl.estimators.neural_base import NeuralEstimatorMixin

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _is_torch_tensor(values: object) -> bool:
    return torch is not None and isinstance(values, torch.Tensor)


def _to_numpy(values: object) -> np.ndarray:
    if _is_torch_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _to_jax_float(values: object) -> jax.Array:
    if _is_torch_tensor(values):
        return jnp.asarray(values.detach().cpu().numpy(), dtype=jnp.float32)
    return jnp.asarray(values, dtype=jnp.float32)


def _to_jax_int(values: object) -> jax.Array:
    if _is_torch_tensor(values):
        return jnp.asarray(values.detach().cpu().numpy(), dtype=jnp.int32)
    return jnp.asarray(values, dtype=jnp.int32)


def _return_like(values: jax.Array, *templates: object) -> object:
    if any(_is_torch_tensor(template) for template in templates):
        if torch is None:  # pragma: no cover
            raise RuntimeError("Torch is required for torch tensor outputs.")
        return torch.tensor(np.asarray(values).copy())
    return values


class _MLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        n_hidden = max(num_layers, 0)
        keys = jr.split(key, n_hidden + 1)
        layers: list[eqx.nn.Linear] = []
        current_dim = in_dim
        for idx in range(n_hidden):
            layers.append(eqx.nn.Linear(current_dim, hidden_dim, key=keys[idx]))
            current_dim = hidden_dim
        self.layers = tuple(layers)
        self.output_layer = eqx.nn.Linear(current_dim, out_dim, key=keys[-1])

    def _forward_single(self, x: jax.Array) -> jax.Array:
        h = x
        for layer in self.layers:
            h = jax.nn.relu(layer(h))
        return self.output_layer(h)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.ndim == 1:
            return self._forward_single(x)
        return jax.vmap(self._forward_single)(x)

    def eval(self) -> _MLP:
        return self


class _ContextQNetwork(eqx.Module):
    n_actions: int = eqx.field(static=True)
    net: _MLP

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        n_actions: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        self.n_actions = n_actions
        self.net = _MLP(
            state_dim + context_dim + n_actions,
            1,
            hidden_dim,
            num_layers,
            key=key,
        )

    def __call__(
        self,
        state_feat: object,
        ctx_feat: object,
        action_onehot: object,
    ) -> object:
        sf = _to_jax_float(state_feat)
        cf = _to_jax_float(ctx_feat)
        ao = _to_jax_float(action_onehot)
        x = jnp.concatenate([sf, cf, ao], axis=-1)
        out = jnp.squeeze(self.net(x), axis=-1)
        return _return_like(out, state_feat, ctx_feat, action_onehot)

    def all_actions(
        self,
        state_feat: object,
        ctx_feat: object,
        n_actions: int,
    ) -> object:
        sf = _to_jax_float(state_feat)
        cf = _to_jax_float(ctx_feat)
        actions = jnp.eye(n_actions, dtype=jnp.float32)
        sf_exp = jnp.repeat(sf[:, None, :], n_actions, axis=1)
        cf_exp = jnp.repeat(cf[:, None, :], n_actions, axis=1)
        a_exp = jnp.repeat(actions[None, :, :], sf.shape[0], axis=0)
        x = jnp.concatenate([sf_exp, cf_exp, a_exp], axis=-1)
        out = jnp.squeeze(jax.vmap(self.net)(x), axis=-1)
        return _return_like(out, state_feat, ctx_feat)

    def eval(self) -> _ContextQNetwork:
        return self


class NeuralGLADIUS(NeuralEstimatorMixin):
    """Context-aware GLADIUS estimator with sklearn-style API."""

    def __init__(
        self,
        n_actions: int = 8,
        discount: float = 0.95,
        scale: float = 1.0,
        q_hidden_dim: int = 128,
        q_num_layers: int = 3,
        ev_hidden_dim: int = 128,
        ev_num_layers: int = 3,
        batch_size: int = 512,
        max_epochs: int = 500,
        lr: float = 1e-3,
        bellman_weight: float = 1.0,
        gradient_clip: float = 1.0,
        patience: int = 50,
        alternating_updates: bool = True,
        lr_decay_rate: float = 0.001,
        tikhonov_annealing: bool = False,
        tikhonov_initial_weight: float = 100.0,
        anchor_action: int | None = None,
        state_encoder: Callable[[object], object] | None = None,
        context_encoder: Callable[[object], object] | None = None,
        state_dim: int | None = None,
        context_dim: int = 0,
        feature_names: list[str] | None = None,
        verbose: bool = False,
    ):
        self.n_actions = n_actions
        self.discount = discount
        self.scale = scale
        self.q_hidden_dim = q_hidden_dim
        self.q_num_layers = q_num_layers
        self.ev_hidden_dim = ev_hidden_dim
        self.ev_num_layers = ev_num_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.bellman_weight = bellman_weight
        self.gradient_clip = gradient_clip
        self.patience = patience
        self.alternating_updates = alternating_updates
        self.lr_decay_rate = lr_decay_rate
        self.tikhonov_annealing = tikhonov_annealing
        self.tikhonov_initial_weight = tikhonov_initial_weight
        self.anchor_action = anchor_action
        self.state_encoder = state_encoder
        self.context_encoder = context_encoder
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.feature_names = feature_names
        self.verbose = verbose

        self.params_: dict[str, float] | None = None
        self.se_: dict[str, float] | None = None
        self.pvalues_: dict[str, float] | None = None
        self.coef_: np.ndarray | None = None
        self.policy_: np.ndarray | None = None
        self.value_: np.ndarray | None = None
        self.projection_r2_: float | None = None
        self.converged_: bool | None = None
        self.n_epochs_: int | None = None

        self._q_net: _ContextQNetwork | None = None
        self._ev_net: _ContextQNetwork | None = None
        self._state_encoder: Callable[[object], jax.Array] | None = None
        self._context_encoder: Callable[[object], jax.Array] | None = None
        self._state_dim: int | None = None
        self._context_dim: int | None = None
        self._n_states: int | None = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        context: str | object | None = None,
        features: RewardSpec | object | None = None,
        transitions: object = None,
    ) -> NeuralGLADIUS:
        all_states, all_actions, all_next, all_contexts = self._extract_data(
            data, state, action, id, context
        )

        n_states = int(np.asarray(all_states).max()) + 1
        self._n_states = n_states
        self._build_encoders(all_states, all_contexts, n_states)

        key = jr.PRNGKey(np.random.randint(0, 2**31 - 1))
        q_key, ev_key = jr.split(key, 2)
        self._q_net = _ContextQNetwork(
            self._state_dim,
            self._context_dim,
            self.n_actions,
            self.q_hidden_dim,
            self.q_num_layers,
            key=q_key,
        )
        self._ev_net = _ContextQNetwork(
            self._state_dim,
            self._context_dim,
            self.n_actions,
            self.ev_hidden_dim,
            self.ev_num_layers,
            key=ev_key,
        )

        self._train(all_states, all_actions, all_next, all_contexts)
        self._extract_policy_and_value(all_states, all_contexts, n_states)

        if features is not None:
            self._project_onto_features(
                features, all_states, all_actions, all_contexts
            )
        else:
            self.params_ = None
            self.se_ = None
            self.pvalues_ = None
            self.projection_r2_ = None
            self.coef_ = None

        return self

    def _extract_data(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None,
        action: str | None,
        id: str | None,
        context: str | object | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required "
                    "when data is a DataFrame"
                )
            panel = TrajectoryPanel.from_dataframe(
                data, state=state, action=action, id=id
            )
            all_states = jnp.asarray(panel.all_states, dtype=jnp.int32)
            all_actions = jnp.asarray(panel.all_actions, dtype=jnp.int32)
            all_next = jnp.asarray(panel.all_next_states, dtype=jnp.int32)

            if isinstance(context, str):
                all_contexts = self._extract_context_from_df(data, id, context, panel)
            elif context is not None:
                all_contexts = _to_jax_int(context)
            else:
                all_contexts = jnp.zeros(len(all_states), dtype=jnp.int32)
        elif isinstance(data, (Panel, TrajectoryPanel)):
            all_states = jnp.asarray(data.get_all_states(), dtype=jnp.int32)
            all_actions = jnp.asarray(data.get_all_actions(), dtype=jnp.int32)
            all_next = jnp.asarray(data.get_all_next_states(), dtype=jnp.int32)
            if context is not None:
                all_contexts = _to_jax_int(context)
            else:
                all_contexts = jnp.zeros(len(all_states), dtype=jnp.int32)
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, got {type(data)}"
            )

        return all_states, all_actions, all_next, all_contexts

    def _extract_context_from_df(
        self,
        df: pd.DataFrame,
        id_col: str,
        context_col: str,
        panel: TrajectoryPanel,
    ) -> jax.Array:
        contexts: list[int] = []
        for _, group in df.groupby(id_col, sort=True):
            group = group.sort_index()
            contexts.extend(group[context_col].values.tolist())
        return jnp.asarray(contexts, dtype=jnp.int32)

    def _call_encoder(self, encoder: Callable[[object], object], values: object) -> jax.Array:
        try:
            encoded = encoder(values)
        except Exception as err:
            if torch is None:
                raise err
            torch_values = torch.tensor(_to_numpy(values).copy(), dtype=torch.long)
            encoded = encoder(torch_values)
        return _to_jax_float(encoded)

    def _build_encoders(
        self,
        all_states: jax.Array,
        all_contexts: jax.Array,
        n_states: int,
    ) -> None:
        if self.state_encoder is not None:
            self._state_encoder = lambda s: self._call_encoder(self.state_encoder, s)
            self._state_dim = self.state_dim or 1
        else:
            max_s = max(n_states - 1, 1)
            self._state_encoder = lambda s, _ms=max_s: (
                _to_jax_float(s) / float(_ms)
            ).reshape(-1, 1)
            self._state_dim = 1

        if self.context_encoder is not None:
            self._context_encoder = lambda c: self._call_encoder(self.context_encoder, c)
            self._context_dim = self.context_dim or 1
        else:
            n_ctx = max(int(np.asarray(all_contexts).max()), 1) if len(all_contexts) else 1
            self._context_encoder = lambda c, _mc=n_ctx: (
                _to_jax_float(c) / float(_mc)
            ).reshape(-1, 1)
            self._context_dim = 1

    def _train(
        self,
        states: jax.Array,
        actions: jax.Array,
        next_states: jax.Array,
        contexts: jax.Array,
    ) -> None:
        def lr_schedule(step: jax.Array) -> jax.Array:
            return self.lr / (1.0 + self.lr_decay_rate * step)

        q_transforms = []
        ev_transforms = []
        if self.gradient_clip > 0:
            q_transforms.append(optax.clip_by_global_norm(self.gradient_clip))
            ev_transforms.append(optax.clip_by_global_norm(self.gradient_clip))
        q_transforms.append(optax.adam(lr_schedule))
        ev_transforms.append(optax.adam(lr_schedule))

        q_optimizer = optax.chain(*q_transforms)
        ev_optimizer = optax.chain(*ev_transforms)
        q_net = self._q_net
        ev_net = self._ev_net
        q_opt_state = q_optimizer.init(eqx.filter(q_net, eqx.is_inexact_array))
        ev_opt_state = ev_optimizer.init(eqx.filter(ev_net, eqx.is_inexact_array))

        N = len(states)
        best_loss = float("inf")
        patience_counter = 0

        action_counts = np.bincount(np.asarray(actions), minlength=self.n_actions).astype(np.float32)
        action_counts = np.clip(action_counts, a_min=1.0, a_max=None)
        class_weights = jnp.asarray(N / (self.n_actions * action_counts), dtype=jnp.float32)

        def q_all(net: _ContextQNetwork, s_feat: jax.Array, ctx_feat: jax.Array) -> jax.Array:
            return jnp.asarray(net.all_actions(s_feat, ctx_feat, self.n_actions), dtype=jnp.float32)

        @eqx.filter_value_and_grad
        def ev_loss_fn(
            ev_model: _ContextQNetwork,
            q_model: _ContextQNetwork,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            actions_j: jax.Array,
            ns_feat: jax.Array,
        ) -> jax.Array:
            a_oh = jax.nn.one_hot(actions_j, self.n_actions, dtype=jnp.float32)
            zeta_sa = jnp.asarray(ev_model(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
            q_next_all = q_all(q_model, ns_feat, ctx_feat)
            v_next = self.scale * jax.nn.logsumexp(q_next_all / self.scale, axis=1)
            return jnp.mean((zeta_sa - jax.lax.stop_gradient(v_next)) ** 2)

        @eqx.filter_value_and_grad
        def q_nll_loss_fn(
            q_model: _ContextQNetwork,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            actions_j: jax.Array,
            ce_weight: float,
        ) -> jax.Array:
            qvals = q_all(q_model, s_feat, ctx_feat)
            log_probs = jax.nn.log_softmax(qvals / self.scale, axis=1)
            per_obs_nll = -log_probs[jnp.arange(actions_j.shape[0]), actions_j]
            weights = class_weights[actions_j]
            nll = jnp.mean(per_obs_nll * weights)
            return ce_weight * nll

        @eqx.filter_value_and_grad
        def joint_loss_fn(
            q_model: _ContextQNetwork,
            ev_model: _ContextQNetwork,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            actions_j: jax.Array,
            ns_feat: jax.Array,
            ce_weight: float,
        ) -> jax.Array:
            qvals = q_all(q_model, s_feat, ctx_feat)
            log_probs = jax.nn.log_softmax(qvals / self.scale, axis=1)
            per_obs_nll = -log_probs[jnp.arange(actions_j.shape[0]), actions_j]
            weights = class_weights[actions_j]
            nll = jnp.mean(per_obs_nll * weights)
            a_oh = jax.nn.one_hot(actions_j, self.n_actions, dtype=jnp.float32)
            ev_sa = jnp.asarray(ev_model(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
            q_next_all = q_all(q_model, ns_feat, ctx_feat)
            v_next = self.scale * jax.nn.logsumexp(q_next_all / self.scale, axis=1)
            bellman = jnp.mean((ev_sa - jax.lax.stop_gradient(v_next)) ** 2)
            return ce_weight * nll + self.bellman_weight * bellman

        @eqx.filter_jit
        def ev_step(
            ev_model: _ContextQNetwork,
            ev_state: optax.OptState,
            q_model: _ContextQNetwork,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            actions_j: jax.Array,
            ns_feat: jax.Array,
        ) -> tuple[_ContextQNetwork, optax.OptState, jax.Array]:
            loss, grads = ev_loss_fn(ev_model, q_model, s_feat, ctx_feat, actions_j, ns_feat)
            updates, ev_state = ev_optimizer.update(grads, ev_state, ev_model)
            ev_model = eqx.apply_updates(ev_model, updates)
            return ev_model, ev_state, loss

        @eqx.filter_jit
        def q_step(
            q_model: _ContextQNetwork,
            q_state: optax.OptState,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            actions_j: jax.Array,
            ce_weight: float,
        ) -> tuple[_ContextQNetwork, optax.OptState, jax.Array]:
            loss, grads = q_nll_loss_fn(q_model, s_feat, ctx_feat, actions_j, ce_weight)
            updates, q_state = q_optimizer.update(grads, q_state, q_model)
            q_model = eqx.apply_updates(q_model, updates)
            return q_model, q_state, loss

        @eqx.filter_jit
        def joint_step(
            q_model: _ContextQNetwork,
            q_state: optax.OptState,
            ev_model: _ContextQNetwork,
            ev_state: optax.OptState,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            actions_j: jax.Array,
            ns_feat: jax.Array,
            ce_weight: float,
        ) -> tuple[_ContextQNetwork, optax.OptState, _ContextQNetwork, optax.OptState, jax.Array]:
            loss, (q_grads, ev_grads) = eqx.filter_value_and_grad(joint_loss_fn, arg=(0, 1))(
                q_model, ev_model, s_feat, ctx_feat, actions_j, ns_feat, ce_weight
            )
            q_updates, q_state = q_optimizer.update(q_grads, q_state, q_model)
            ev_updates, ev_state = ev_optimizer.update(ev_grads, ev_state, ev_model)
            q_model = eqx.apply_updates(q_model, q_updates)
            ev_model = eqx.apply_updates(ev_model, ev_updates)
            return q_model, q_state, ev_model, ev_state, loss

        best_q = q_net
        best_ev = ev_net

        for epoch in range(self.max_epochs):
            perm = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches = 0
            batch_idx = 0
            ce_weight = (
                self.tikhonov_initial_weight / (1.0 + epoch)
                if self.tikhonov_annealing
                else 1.0
            )

            for start in range(0, N, self.batch_size):
                idx = perm[start : start + self.batch_size]
                s = states[idx]
                a = actions[idx]
                ns = next_states[idx]
                ctx = contexts[idx]

                s_feat = self._state_encoder(s)
                ns_feat = self._state_encoder(ns)
                ctx_feat = self._context_encoder(ctx)

                if self.alternating_updates and batch_idx % 2 == 0:
                    ev_net, ev_opt_state, loss = ev_step(
                        ev_net, ev_opt_state, q_net, s_feat, ctx_feat, a, ns_feat
                    )
                elif self.alternating_updates and batch_idx % 2 == 1:
                    q_net, q_opt_state, loss = q_step(
                        q_net, q_opt_state, s_feat, ctx_feat, a, ce_weight
                    )
                else:
                    q_net, q_opt_state, ev_net, ev_opt_state, loss = joint_step(
                        q_net,
                        q_opt_state,
                        ev_net,
                        ev_opt_state,
                        s_feat,
                        ctx_feat,
                        a,
                        ns_feat,
                        ce_weight,
                    )

                epoch_loss += float(loss)
                n_batches += 1
                batch_idx += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}")

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                best_q = q_net
                best_ev = ev_net
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        self._q_net = best_q
        self._ev_net = best_ev
        self.converged_ = patience_counter >= self.patience or epoch == self.max_epochs - 1
        self.n_epochs_ = epoch + 1

    def _extract_policy_and_value(
        self,
        all_states: jax.Array,
        all_contexts: jax.Array,
        n_states: int,
    ) -> None:
        unique_states = jnp.arange(n_states, dtype=jnp.int32)
        ctx_default = jnp.zeros(n_states, dtype=jnp.int32)
        s_feat = self._state_encoder(unique_states)
        ctx_feat = self._context_encoder(ctx_default)
        qvals = jnp.asarray(self._q_net.all_actions(s_feat, ctx_feat, self.n_actions), dtype=jnp.float32)
        policy = jax.nn.softmax(qvals / self.scale, axis=1)
        value = self.scale * jax.nn.logsumexp(qvals / self.scale, axis=1)
        self.policy_ = np.asarray(policy)
        self.value_ = np.asarray(value)

    def _project_onto_features(
        self,
        features: RewardSpec | object,
        states: jax.Array,
        actions: jax.Array,
        contexts: jax.Array,
    ) -> None:
        if isinstance(features, RewardSpec):
            feat_matrix = features.feature_matrix
            names = features.parameter_names
        else:
            feat_matrix = features
            names = self.feature_names or [f"f{i}" for i in range(np.asarray(features).shape[-1])]

        n_s = self._n_states
        unique_states = jnp.arange(n_s, dtype=jnp.int32)
        unique_ctx = jnp.zeros(n_s, dtype=jnp.int32)
        s_feat = self._state_encoder(unique_states)
        ctx_feat = self._context_encoder(unique_ctx)
        q_all = jnp.asarray(self._q_net.all_actions(s_feat, ctx_feat, self.n_actions), dtype=jnp.float32)
        action_ids = jnp.arange(self.n_actions, dtype=jnp.int32)
        action_oh = jax.nn.one_hot(action_ids, self.n_actions, dtype=jnp.float32)

        def reward_for_action(a_oh_single: jax.Array) -> jax.Array:
            tiled = jnp.repeat(a_oh_single[None, :], n_s, axis=0)
            ev_a = jnp.asarray(self._ev_net(s_feat, ctx_feat, tiled), dtype=jnp.float32)
            return ev_a

        ev_all = jax.vmap(reward_for_action)(action_oh).T
        r_all = q_all - self.discount * ev_all

        feat_np = _to_numpy(feat_matrix)
        dr_list = []
        dphi_list = []
        for a_idx in range(1, self.n_actions):
            dr_list.append(np.asarray(r_all[:, a_idx] - r_all[:, 0]))
            dphi_list.append(feat_np[:n_s, a_idx, :] - feat_np[:n_s, 0, :])

        rewards = np.concatenate(dr_list, axis=0).astype(np.float32)
        phi = np.concatenate(dphi_list, axis=0).astype(np.float32)

        theta, se, r2 = self._project_parameters(phi, rewards)
        self.params_ = {n: float(v) for n, v in zip(names, theta)}
        self.se_ = {n: float(v) for n, v in zip(names, se)}
        self.pvalues_ = self._compute_pvalues(self.params_, self.se_)
        self.projection_r2_ = r2
        self.coef_ = np.asarray(theta)

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        if self._q_net is None or self._ev_net is None or self._n_states is None:
            return None
        n_s = self._n_states
        unique_states = jnp.arange(n_s, dtype=jnp.int32)
        ctx_default = jnp.zeros(n_s, dtype=jnp.int32)
        s_feat = self._state_encoder(unique_states)
        ctx_feat = self._context_encoder(ctx_default)
        q_all = jnp.asarray(self._q_net.all_actions(s_feat, ctx_feat, self.n_actions), dtype=jnp.float32)
        action_ids = jnp.arange(self.n_actions, dtype=jnp.int32)
        action_oh = jax.nn.one_hot(action_ids, self.n_actions, dtype=jnp.float32)

        def ev_for_action(a_oh_single: jax.Array) -> jax.Array:
            tiled = jnp.repeat(a_oh_single[None, :], n_s, axis=0)
            return jnp.asarray(self._ev_net(s_feat, ctx_feat, tiled), dtype=jnp.float32)

        ev_all = jax.vmap(ev_for_action)(action_oh).T
        return np.asarray(q_all - self.discount * ev_all)

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        if self.policy_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        states = np.asarray(states, dtype=np.int64)
        return self.policy_[states]

    def predict_q_from_features(
        self,
        state_features: object,
        contexts: object | None = None,
    ) -> np.ndarray:
        if self._q_net is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        s_feat = _to_jax_float(state_features)
        if s_feat.ndim == 1:
            s_feat = s_feat[None, :]
        if contexts is None:
            contexts = jnp.zeros(s_feat.shape[0], dtype=jnp.int32)
        ctx_feat = self._context_encoder(contexts)
        qvals = self._q_net.all_actions(s_feat, ctx_feat, self.n_actions)
        return np.asarray(qvals)

    def predict_reward_from_features(
        self,
        state_features: object,
        actions: object,
        contexts: object | None = None,
    ) -> np.ndarray:
        if self._q_net is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        s_feat = _to_jax_float(state_features)
        if s_feat.ndim == 1:
            s_feat = s_feat[None, :]
        actions_j = _to_jax_int(actions)
        if actions_j.ndim == 0:
            actions_j = actions_j[None]
        if contexts is None:
            contexts = jnp.zeros(s_feat.shape[0], dtype=jnp.int32)
        ctx_feat = self._context_encoder(contexts)
        a_oh = jax.nn.one_hot(actions_j, self.n_actions, dtype=jnp.float32)
        q_vals = jnp.asarray(self._q_net(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
        ev_vals = jnp.asarray(self._ev_net(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
        return np.asarray(q_vals - self.discount * ev_vals)

    def predict_reward(
        self,
        states: object,
        actions: object,
        contexts: object | None = None,
    ) -> object:
        if self._q_net is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        states_j = _to_jax_int(states)
        actions_j = _to_jax_int(actions)
        if contexts is None:
            contexts_j = jnp.zeros(states_j.shape[0], dtype=jnp.int32)
        else:
            contexts_j = _to_jax_int(contexts)
        s_feat = self._state_encoder(states_j)
        ctx_feat = self._context_encoder(contexts_j)
        a_oh = jax.nn.one_hot(actions_j, self.n_actions, dtype=jnp.float32)
        q_vals = jnp.asarray(self._q_net(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
        ev_vals = jnp.asarray(self._ev_net(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
        rewards = q_vals - self.discount * ev_vals
        return _return_like(rewards, states, actions, contexts)

    def conf_int(self, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        if self.params_ is None or self.se_ is None:
            raise RuntimeError(
                "No projected parameters available. "
                "Call fit() with features= to extract structural parameters."
            )
        z = scipy_norm.ppf(1 - alpha / 2)
        intervals: dict[str, tuple[float, float]] = {}
        for name in self.params_:
            est = self.params_[name]
            se = self.se_[name]
            if np.isfinite(se):
                intervals[name] = (est - z * se, est + z * se)
            else:
                intervals[name] = (float("nan"), float("nan"))
        return intervals

    def summary(self) -> str:
        if self.policy_ is None:
            return "NeuralGLADIUS: Not fitted yet. Call fit() first."
        n_obs = self._n_states if self._n_states is not None else None
        return self._format_neural_summary(
            method_name="NeuralGLADIUS",
            params=self.params_,
            se=self.se_,
            pvalues=self.pvalues_,
            projection_r2=self.projection_r2_,
            n_observations=n_obs,
            n_epochs=self.n_epochs_,
            converged=self.converged_,
            discount=self.discount,
            scale=self.scale,
            context_dim=self._context_dim,
            extra_lines=[
                f"Q-network: {self.q_num_layers} layers x {self.q_hidden_dim} hidden",
                f"EV-network: {self.ev_num_layers} layers x {self.ev_hidden_dim} hidden",
            ],
        )

    def __repr__(self) -> str:
        fitted = self.policy_ is not None
        return (
            f"NeuralGLADIUS(n_actions={self.n_actions}, "
            f"discount={self.discount}, "
            f"fitted={fitted})"
        )
