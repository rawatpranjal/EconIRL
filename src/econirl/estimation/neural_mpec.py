"""Neural MPEC estimator: value-lifted structural estimation with neural reward.

This is the neural-network sibling of the tabular MPEC estimator
(:class:`~econirl.estimation.mpec.MPECEstimator`). Both belong to the
"direct optimization" family: they lift the value function into the optimizer
and solve a single problem, imposing the soft Bellman equation rather than
nesting a fixed-point solve inside every step. Tabular MPEC carries a linear
reward and a value vector and enforces the Bellman equation as a hard equality
constraint. Neural MPEC replaces both with networks, so the equality relaxes to
a soft penalty evaluated over a set of collocation states.

The estimator co-trains a reward network ``u_theta(s, a)`` and a value network
``V_phi(s)`` by minimizing the logit negative log-likelihood plus a Bellman
residual penalty::

    EV_phi(s, a) = sum_s' P_a(s, s') V_phi(s')         # exact, transitions known
    Q(s, a)      = u_theta(s, a) + beta * EV_phi(s, a)
    pi(a | s)    = softmax_a( Q(s, .) / sigma )
    NLL          = - mean_it log pi(a_it | s_it)
    resid(s)     = V_phi(s) - sigma * logsumexp_a( Q(s, .) / sigma )
    loss         = NLL + (rho / 2) * sum_{s in C} w(s) * resid(s) ** 2

Because the transition kernel ``P`` is known (the structural setting), the
expected continuation value is a matrix-vector product computed exactly, with no
double-sampling of next states. The reward of a reference action is anchored to
zero, the location normalization that point-identifies the reward level
(Magnac and Thesmar 2002).

Unlike GLADIUS (the model-free neural cousin), Neural MPEC uses the known
transitions and a single value network instead of a learned expected-value
network. It is consistent: as the sample grows, the recovered reward and value
converge to the truth. Under a correctly specified linear reward it is less
efficient than the parametric tabular MPEC; under a misspecified (nonlinear)
reward the flexible network avoids the linear method's bias floor.

See ``docs/simulation_studies/direct_optimization.md`` for the comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.preferences.base import UtilityFunction


@dataclass
class NeuralMPECConfig:
    """Configuration for the Neural MPEC estimator.

    Attributes:
        width: Hidden-layer width of both the reward and value MLPs.
        depth: Number of hidden layers (1 = shallow). The validated default
            recovers the canonical structural cell.
        bellman_penalty_weight: ``rho``, the weight on the soft Bellman-residual
            penalty. Large values pull ``V_phi`` tightly onto its own Bellman
            image; the likelihood is largely insensitive to it once the residual
            is small.
        learning_rate: Adam learning rate.
        max_epochs: Number of full-batch gradient steps.
        reference_action: Action whose reward is anchored to zero (the location
            normalization). Negative indexes from the end, so ``-1`` (default)
            anchors the last action. Set to ``None`` to learn all actions' rewards
            with no anchor, in which case the reward level is identified only up to
            a per-state constant and reward-level metrics are not meaningful.
        collocation: Which states carry the Bellman penalty. ``"all"`` weights
            every state uniformly (the exact residual on a tabular grid);
            ``"observed"`` weights states by their empirical visit frequency.
        seed: PRNG seed for network initialization.
    """

    width: int = 32
    depth: int = 2
    bellman_penalty_weight: float = 1.0
    learning_rate: float = 5e-3
    max_epochs: int = 4000
    reference_action: int | None = -1
    collocation: Literal["all", "observed"] = "all"
    seed: int = 0


class _RewardNet(eqx.Module):
    """Neural flow reward ``u_theta(s, a)`` over one-hot state inputs.

    Outputs one value per non-reference action; the reference action's column is
    inserted as zero so its reward is pinned to the location-normalization anchor.
    When ``reference_action`` is None, all actions are learned.
    """

    mlp: eqx.nn.MLP
    n_actions: int = eqx.field(static=True)
    ref: int | None = eqx.field(static=True)

    def __init__(self, n_states, n_actions, width, depth, ref, *, key):
        self.n_actions = n_actions
        self.ref = ref
        out = n_actions - 1 if ref is not None else n_actions
        self.mlp = eqx.nn.MLP(
            in_size=n_states, out_size=out, width_size=width, depth=depth,
            activation=jax.nn.tanh, key=key,
        )

    def all_actions(self, onehot):  # (S, S) -> (S, A)
        raw = jax.vmap(self.mlp)(onehot)
        if self.ref is None:
            return raw
        # Scatter the learned columns around a zero reference column.
        S = raw.shape[0]
        cols, j = [], 0
        for a in range(self.n_actions):
            if a == self.ref:
                cols.append(jnp.zeros((S, 1), dtype=raw.dtype))
            else:
                cols.append(raw[:, j:j + 1])
                j += 1
        return jnp.concatenate(cols, axis=1)


class _ValueNet(eqx.Module):
    """Neural value function ``V_phi(s)`` over one-hot state inputs."""

    mlp: eqx.nn.MLP

    def __init__(self, n_states, width, depth, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=n_states, out_size=1, width_size=width, depth=depth,
            activation=jax.nn.tanh, key=key,
        )

    def all_states(self, onehot):  # (S, S) -> (S,)
        return jax.vmap(self.mlp)(onehot).squeeze(-1)


class _NeuralMPEC(eqx.Module):
    reward: _RewardNet
    value: _ValueNet


class NeuralMPECEstimator(BaseEstimator):
    """Neural MPEC estimator for dynamic discrete choice models.

    Co-trains a neural reward and a neural value function in a single loop with a
    soft Bellman-residual penalty, using the known transition kernel to compute
    the expected continuation value exactly. The neural sibling of
    :class:`~econirl.estimation.mpec.MPECEstimator`.

    Standard errors are not produced (the asymptotic score of the tabular MPEC
    does not carry over to the network parameters); ``standard_errors`` is filled
    with NaN, matching the other neural estimators.

    Example:
        >>> est = NeuralMPECEstimator(NeuralMPECConfig(reference_action=2))
        >>> summary = est.estimate(panel, utility, problem, transitions)
        >>> reward = summary.metadata["reward_table"]   # (n_states, n_actions)
    """

    def __init__(
        self,
        config: NeuralMPECConfig | None = None,
        verbose: bool = False,
    ):
        super().__init__(se_method="asymptotic", compute_hessian=False, verbose=verbose)
        self._config = config or NeuralMPECConfig()

    @property
    def name(self) -> str:
        return "Neural MPEC"

    def estimate(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate the reward and value networks from panel data.

        Overrides the base ``estimate`` because the networks have no analytic
        standard errors; the summary is built directly with NaN standard errors,
        mirroring the GLADIUS path.
        """
        start_time = time.time()
        result = self._optimize(
            panel=panel, utility=utility, problem=problem,
            transitions=transitions, initial_params=initial_params, **kwargs,
        )
        standard_errors = jnp.full_like(result.parameters, float("nan"))

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
            estimation_time=time.time() - start_time,
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
        cfg = self._config
        start_time = time.time()

        S = problem.num_states
        A = problem.num_actions
        beta = float(problem.discount_factor)
        sigma = float(problem.scale_parameter)
        ref = None if cfg.reference_action is None else int(cfg.reference_action) % A

        onehot = jnp.eye(S, dtype=jnp.float64)
        T = jnp.asarray(transitions, dtype=jnp.float64)
        obs_s = jnp.asarray(np.asarray(panel.get_all_states()), dtype=jnp.int32)
        obs_a = jnp.asarray(np.asarray(panel.get_all_actions()), dtype=jnp.int32)

        # Collocation weights (sum to one).
        if cfg.collocation == "observed":
            counts = np.bincount(np.asarray(obs_s), minlength=S).astype(np.float64)
            w = jnp.asarray(counts / counts.sum(), dtype=jnp.float64)
        else:
            w = jnp.ones(S, dtype=jnp.float64) / S

        key = jax.random.PRNGKey(cfg.seed)
        k_r, k_v = jax.random.split(key)
        model = _NeuralMPEC(
            reward=_RewardNet(S, A, cfg.width, cfg.depth, ref, key=k_r),
            value=_ValueNet(S, cfg.width, cfg.depth, key=k_v),
        )

        def loss_fn(m):
            u_all = m.reward.all_actions(onehot)            # (S, A)
            V_all = m.value.all_states(onehot)              # (S,)
            EV = jnp.einsum("ast,t->as", T, V_all)          # (A, S), exact known-P
            Q = u_all + beta * EV.T                         # (S, A)
            logp = jax.nn.log_softmax(Q / sigma, axis=1)
            nll = -logp[obs_s, obs_a].mean()
            resid = V_all - sigma * jax.scipy.special.logsumexp(Q / sigma, axis=1)
            penalty = jnp.sum(w * resid ** 2)
            return nll + 0.5 * cfg.bellman_penalty_weight * penalty

        opt = optax.adam(cfg.learning_rate)
        opt_state = opt.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def step(m, ostate):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
            updates, ostate = opt.update(grads, ostate, eqx.filter(m, eqx.is_array))
            return eqx.apply_updates(m, updates), ostate, loss

        from tqdm import tqdm
        pbar = tqdm(range(cfg.max_epochs), desc="Neural MPEC",
                    disable=not self._verbose, leave=True)
        final_loss = float("nan")
        for _ in pbar:
            model, opt_state, loss = step(model, opt_state)
            final_loss = float(loss)
            pbar.set_postfix({"loss": f"{final_loss:.4f}"})

        # Final objects.
        u_all = np.asarray(model.reward.all_actions(onehot))            # (S, A)
        V_all = np.asarray(model.value.all_states(onehot))              # (S,)
        EV = np.einsum("ast,t->as", np.asarray(T), V_all)
        Q = u_all + beta * EV.T
        policy = np.asarray(jax.nn.softmax(jnp.asarray(Q) / sigma, axis=1))
        logp = np.asarray(jax.nn.log_softmax(jnp.asarray(Q) / sigma, axis=1))
        ll = float(logp[np.asarray(obs_s), np.asarray(obs_a)].sum())
        resid = V_all - sigma * np.asarray(
            jax.scipy.special.logsumexp(jnp.asarray(Q) / sigma, axis=1)
        )

        parameters = _project_reward_to_params(utility, jnp.asarray(u_all))

        return EstimationResult(
            parameters=parameters,
            log_likelihood=ll,
            value_function=jnp.asarray(V_all, dtype=jnp.float32),
            policy=jnp.asarray(policy, dtype=jnp.float32),
            hessian=None,
            gradient_contributions=None,
            converged=True,
            num_iterations=cfg.max_epochs,
            num_function_evals=cfg.max_epochs,
            num_inner_iterations=0,  # single loop, no nested Bellman solve
            message=f"Neural MPEC trained for {cfg.max_epochs} epochs",
            optimization_time=time.time() - start_time,
            metadata={
                "reward_table": u_all.tolist(),
                "value_function": V_all.tolist(),
                "max_bellman_residual": float(np.abs(resid).max()),
                "reference_action": ref,
                "collocation": cfg.collocation,
                "bellman_penalty_weight": cfg.bellman_penalty_weight,
                "final_loss": final_loss,
            },
        )


def _project_reward_to_params(
    utility: UtilityFunction, reward_table: jnp.ndarray
) -> jnp.ndarray:
    """Project the neural reward onto the linear feature gauge for reporting.

    Mirrors the GLADIUS convention (action-difference least squares relative to
    action 0) so ``.parameters`` lines up with ``utility.parameter_names``. The
    rewards are partially identified, so this is a convenience reading in the
    linear gauge, not a structural estimate.
    """
    feature_matrix = getattr(utility, "feature_matrix", None)
    if feature_matrix is None:
        return reward_table.flatten()

    feature_matrix = jnp.asarray(feature_matrix)
    n_states, n_actions, _ = feature_matrix.shape
    dr_list, dphi_list = [], []
    for a in range(1, n_actions):
        dr_list.append(reward_table[:, a] - reward_table[:, 0])
        dphi_list.append(feature_matrix[:, a, :] - feature_matrix[:, 0, :])
    X = jnp.concatenate(dphi_list, axis=0)
    y = jnp.concatenate(dr_list, axis=0)
    parameters, _res, _rank, _sv = jnp.linalg.lstsq(X, y)
    return parameters
