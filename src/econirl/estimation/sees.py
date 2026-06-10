"""Sieve Estimator (SEES) for dynamic discrete choice models.

Approximates a Bellman solution object with sieve basis functions, then
performs penalized maximum likelihood jointly over structural parameters theta
and basis coefficients alpha. The default mode approximates the value function
V(s); action-value, continuation-value, policy-logit, and collocation variants
are available through the ``solution`` configuration.

This avoids both the costly inner fixed-point loop of NFXP and the
neural network training of NNES, using a closed-form basis expansion
that can be solved with standard nonlinear optimization.

Algorithm:
    1. Construct sieve basis Psi(s) of dimension K (Fourier or polynomial)
    2. Approximate V, Q, EV, or policy logits with basis coefficients alpha
    3. Compute the implied softmax policy
    4. Maximize a likelihood minus a Bellman/equilibrium residual penalty

Reference:
    Luo, Y. & Sang, Y. (2024). "Efficient Estimation of Structural Models
    via Sieves." Working Paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.optimizer import minimize_lbfgsb
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod
from econirl.preferences.base import UtilityFunction


SEESSolution = Literal["value", "q", "ev", "policy", "collocation"]
VALID_SEES_SOLUTIONS: tuple[str, ...] = (
    "value",
    "q",
    "ev",
    "policy",
    "collocation",
)


@dataclass
class SEESConfig:
    """Configuration for SEES estimator.

    Attributes:
        basis_type: Sieve basis type. "bspline" uses cubic B-splines per
            Luo and Sang (2024). "polynomial" uses monomials on [-1, 1].
            "fourier" is an econirl extension not in the original paper.
        basis_dim: Number of basis functions.
        penalty_weight: Weight omega on the equilibrium penalty
            (Luo and Sang 2024, equation 3). Penalizes the Bellman
            equation violation ||V - T(V; theta)||^2. Higher values
            enforce the Bellman constraint more strongly, pushing the
            estimator toward MLE. The paper recommends increasing omega
            until the confidence interval stabilizes. If
            penalty_schedule is set, that value is used instead.
        penalty_schedule: Optional callable mapping sample size n to a
            penalty weight, implementing the omega_n -> infinity schedule
            of Luo and Sang (2024). When set, supersedes penalty_weight.
            Example: lambda n: 1.0 * n ** 0.5.
        spline_degree: Polynomial degree for the B-spline basis (default 3).
        state_basis_mode: "index" always builds the historical basis over
            state indices; "encoded" builds the sieve over
            problem.state_encoder states; "auto" uses encoded features only
            for high-dimensional encoded state spaces and otherwise keeps the
            index basis.
        solution: Solution object approximated by the sieve. "value" is the
            existing V-SEES estimator; "q", "ev", and "policy" approximate
            action values, expected continuation values, and policy logits;
            "collocation" is V-SEES with the Bellman penalty evaluated on a
            deterministic state subset.
        num_theta_starts: Number of deterministic theta starts to try. The
            first start is the supplied initial_params or utility default.
            Additional starts include static-logit and neutral variants.
        warm_start_value: Whether to initialize sieve coefficients by solving
            the Bellman equation at the initial theta and projecting the value
            function into the sieve basis.
        max_iter: Maximum L-BFGS-B iterations.
        tol: Gradient tolerance for convergence.
        compute_se: Whether to compute standard errors.
        se_method: Standard error method.
        verbose: Whether to print progress.
    """

    basis_type: str = "bspline"
    basis_dim: int = 8
    penalty_weight: float = 10.0
    penalty_schedule: Callable[[int], float] | None = None
    spline_degree: int = 3
    state_basis_mode: str = "auto"
    solution: SEESSolution = "value"
    num_theta_starts: int = 1
    warm_start_value: bool = True
    max_iter: int = 500
    tol: float = 1e-6
    compute_se: bool = True
    se_method: SEMethod = "asymptotic"
    verbose: bool = False


class SEESEstimator(BaseEstimator):
    """Sieve Estimator for dynamic discrete choice.

    Approximates a Bellman solution object with basis functions and jointly
    optimizes structural parameters and basis coefficients via penalized MLE.
    Standard errors use the Schur complement to marginalize out basis
    coefficients alpha, giving the marginal Hessian for theta.

    Args:
        config: SEESConfig or keyword arguments matching SEESConfig fields.

    Example:
        >>> estimator = SEESEstimator(basis_type="fourier", basis_dim=8)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
        >>> # Access basis coefficients
        >>> result.metadata["alpha"]
    """

    def __init__(
        self,
        basis_type: str = "bspline",
        basis_dim: int = 8,
        penalty_weight: float = 10.0,
        penalty_schedule: Callable[[int], float] | None = None,
        spline_degree: int = 3,
        state_basis_mode: str = "auto",
        solution: SEESSolution = "value",
        num_theta_starts: int = 1,
        warm_start_value: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
        compute_se: bool = True,
        se_method: SEMethod = "asymptotic",
        verbose: bool = False,
        config: SEESConfig | None = None,
    ):
        if config is not None:
            basis_type = config.basis_type
            basis_dim = config.basis_dim
            penalty_weight = config.penalty_weight
            penalty_schedule = config.penalty_schedule
            spline_degree = config.spline_degree
            state_basis_mode = config.state_basis_mode
            solution = config.solution
            num_theta_starts = config.num_theta_starts
            warm_start_value = config.warm_start_value
            max_iter = config.max_iter
            tol = config.tol
            compute_se = config.compute_se
            se_method = config.se_method
            verbose = config.verbose

        super().__init__(
            se_method=se_method,
            compute_hessian=compute_se,
            verbose=verbose,
        )
        self._basis_type = basis_type
        self._basis_dim = basis_dim
        self._penalty_weight = penalty_weight
        self._penalty_schedule = penalty_schedule
        self._spline_degree = spline_degree
        if state_basis_mode not in {"auto", "index", "encoded"}:
            raise ValueError(
                "state_basis_mode must be one of 'auto', 'index', or 'encoded'"
            )
        if solution not in VALID_SEES_SOLUTIONS:
            raise ValueError(
                "solution must be one of "
                + ", ".join(repr(value) for value in VALID_SEES_SOLUTIONS)
            )
        if num_theta_starts < 1:
            raise ValueError("num_theta_starts must be at least 1")
        self._state_basis_mode = state_basis_mode
        self._solution = solution
        self._num_theta_starts = int(num_theta_starts)
        self._warm_start_value = warm_start_value
        self._max_iter = max_iter
        self._tol = tol
        self._compute_se = compute_se
        self._config = SEESConfig(
            basis_type=basis_type,
            basis_dim=basis_dim,
            penalty_weight=penalty_weight,
            penalty_schedule=penalty_schedule,
            spline_degree=spline_degree,
            state_basis_mode=state_basis_mode,
            solution=solution,
            num_theta_starts=num_theta_starts,
            warm_start_value=warm_start_value,
            max_iter=max_iter,
            tol=tol,
            compute_se=compute_se,
            se_method=se_method,
            verbose=verbose,
        )
        self._last_basis_metadata: dict[str, object] = {}

    @property
    def name(self) -> str:
        return f"SEES-{self._solution} ({self._basis_type}, Luo & Sang 2024)"

    @property
    def config(self) -> SEESConfig:
        """Return current configuration."""
        return self._config

    def _build_basis(
        self,
        n_states: int,
        problem: DDCProblem | None = None,
    ) -> jnp.ndarray:
        """Construct sieve basis matrix Psi(s).

        Args:
            n_states: Number of discrete states.
            problem: Optional DDCProblem. When supplied and
                state_basis_mode permits it, high-dimensional state encoders
                are used to build an encoded-state basis.

        Returns:
            Basis matrix, shape (n_states, basis_dim).
        """
        use_encoded = self._use_encoded_basis(problem)
        if use_encoded:
            return self._build_encoded_state_basis(problem)

        self._last_basis_metadata = {
            "basis_source": "state_index",
            "basis_family": self._basis_type,
            "state_feature_dim": None,
            "configured_basis_dim": self._basis_dim,
        }
        return self._build_index_basis(n_states)

    def _use_encoded_basis(self, problem: DDCProblem | None) -> bool:
        if self._state_basis_mode == "index":
            return False
        if problem is None or problem.state_encoder is None:
            if self._state_basis_mode == "encoded":
                raise ValueError("state_basis_mode='encoded' requires problem.state_encoder")
            return False
        if self._state_basis_mode == "encoded":
            return True
        return (problem.state_dim or 0) > 2

    def _build_index_basis(self, n_states: int) -> jnp.ndarray:
        """Construct the historical index-based sieve basis."""
        # Normalized state values in [0, 1]
        s_norm = jnp.linspace(0, 1, n_states)

        if self._basis_type == "fourier":
            # Fourier basis: [1, cos(pi*s), sin(pi*s), cos(2pi*s), sin(2pi*s), ...]
            basis = jnp.zeros((n_states, self._basis_dim))
            basis = basis.at[:, 0].set(1.0)  # Constant term
            for k in range(1, self._basis_dim):
                freq = (k + 1) // 2
                if k % 2 == 1:
                    basis = basis.at[:, k].set(jnp.cos(freq * np.pi * s_norm))
                else:
                    basis = basis.at[:, k].set(jnp.sin(freq * np.pi * s_norm))
            return basis

        elif self._basis_type == "polynomial":
            # Monomial basis: [1, s, s^2, s^3, ...] on [-1, 1]
            s_cheb = 2 * s_norm - 1  # Map to [-1, 1] for conditioning
            basis = jnp.zeros((n_states, self._basis_dim))
            for k in range(self._basis_dim):
                basis = basis.at[:, k].set(s_cheb ** k)
            return basis

        elif self._basis_type == "bspline":
            # Cubic B-spline basis per Luo and Sang (2024). The K basis
            # functions span the unit interval with equally-spaced knots
            # and degree `spline_degree` (default 3 = cubic).
            from scipy.interpolate import BSpline as _BSpline

            degree = self._spline_degree
            K = self._basis_dim
            n_interior = K - degree - 1
            if n_interior < 0:
                raise ValueError(
                    f"basis_dim ({K}) must exceed spline_degree ({degree}) "
                    f"to admit a clamped B-spline basis."
                )
            interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
            knots = np.concatenate([
                np.zeros(degree + 1),
                interior,
                np.ones(degree + 1),
            ])
            s_grid = np.asarray(s_norm)
            basis_np = np.zeros((n_states, K))
            for k in range(K):
                coeffs = np.zeros(K)
                coeffs[k] = 1.0
                spline = _BSpline(knots, coeffs, degree, extrapolate=False)
                basis_np[:, k] = np.nan_to_num(spline(s_grid), nan=0.0)
            return jnp.array(basis_np, dtype=jnp.float64)

        else:
            raise ValueError(f"Unknown basis type: {self._basis_type}")

    def _build_encoded_state_basis(self, problem: DDCProblem | None) -> jnp.ndarray:
        """Build a stable basis over encoded state features.

        The high-dimensional known-truth DGP exposes a finite grid through
        problem.state_encoder. A Gaussian RBF dictionary on those encoded
        states is then orthonormalized with an SVD. With basis_dim >= S this
        spans every finite-state value function; with fewer columns it is a
        smooth feature-aware sieve.
        """
        if problem is None or problem.state_encoder is None:
            raise ValueError("encoded-state SEES basis requires problem.state_encoder")

        states = jnp.arange(problem.num_states, dtype=jnp.int32)
        features = np.asarray(problem.state_encoder(states), dtype=np.float64)
        if features.ndim == 1:
            features = features[:, None]
        if features.shape[0] != problem.num_states:
            raise ValueError(
                "problem.state_encoder must return one row per state; "
                f"got {features.shape[0]} rows for {problem.num_states} states"
            )

        mean = features.mean(axis=0, keepdims=True)
        scale = np.maximum(features.std(axis=0, keepdims=True), 1e-8)
        z = (features - mean) / scale
        sqdist = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)
        nearest_dist = np.min(
            np.where(sqdist > 1e-12, sqdist, np.inf),
            axis=1,
        )
        finite_nearest = nearest_dist[np.isfinite(nearest_dist)]
        bandwidth_sq = (
            float(np.median(finite_nearest)) if finite_nearest.size else 1.0
        )
        bandwidth_sq = max(bandwidth_sq, 1e-6)

        requested = max(1, int(self._basis_dim))
        n_centers = min(requested, problem.num_states)
        if n_centers == problem.num_states:
            center_idx = np.arange(problem.num_states)
        else:
            center_idx = np.unique(
                np.linspace(0, problem.num_states - 1, n_centers).round().astype(int)
            )
            while center_idx.size < n_centers:
                missing = [
                    idx for idx in range(problem.num_states) if idx not in set(center_idx)
                ]
                center_idx = np.sort(np.concatenate([center_idx, missing[:1]]))

        raw_rbf = np.exp(-0.5 * sqdist[:, center_idx] / bandwidth_sq)
        if requested >= problem.num_states:
            raw = raw_rbf
        else:
            raw = np.column_stack([np.ones(problem.num_states), raw_rbf])
            raw = raw[:, : min(requested, raw.shape[1])]

        # Orthonormalize for stable alpha scaling and projection.
        u, singular_values, _ = np.linalg.svd(raw, full_matrices=False)
        rank = int(np.sum(singular_values > 1e-10))
        if rank == 0:
            raise ValueError("encoded-state basis is numerically rank deficient")
        use = min(requested, rank)
        basis = u[:, :use] * np.sqrt(float(problem.num_states))

        self._last_basis_metadata = {
            "basis_source": "encoded_state",
            "basis_family": "rbf_svd",
            "state_feature_dim": int(features.shape[1]),
            "configured_basis_dim": self._basis_dim,
            "actual_basis_dim": int(use),
            "rbf_bandwidth_sq": bandwidth_sq,
            "encoded_basis_rank": rank,
        }
        return jnp.array(basis, dtype=jnp.float64)

    def _project_value_solution(
        self,
        basis: jnp.ndarray,
        feature_matrix: jnp.ndarray,
        initial_params: jnp.ndarray,
        problem: DDCProblem,
        transitions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, float, bool]:
        """Project the Bellman solution at initial theta into the sieve."""
        flow_u = jnp.einsum("sak,k->sa", feature_matrix, initial_params)
        operator = SoftBellmanOperator(problem, transitions)
        solution = value_iteration(
            operator,
            flow_u,
            tol=1e-10,
            max_iter=10_000,
        )
        basis_np = np.asarray(basis, dtype=np.float64)
        value_np = np.asarray(solution.V, dtype=np.float64)
        alpha_np, *_ = np.linalg.lstsq(basis_np, value_np, rcond=None)
        projected = basis_np @ alpha_np
        projection_rmse = float(np.sqrt(np.mean((projected - value_np) ** 2)))
        return (
            jnp.array(alpha_np, dtype=jnp.float64),
            projection_rmse,
            bool(solution.converged),
        )

    def _project_state_action_solution(
        self,
        basis: jnp.ndarray,
        target: jnp.ndarray,
    ) -> tuple[jnp.ndarray, float]:
        """Project a state-action object into action-specific basis columns."""
        basis_np = np.asarray(basis, dtype=np.float64)
        target_np = np.asarray(target, dtype=np.float64)
        if target_np.ndim != 2:
            raise ValueError("state-action target must have shape (states, actions)")

        coeffs = []
        fitted = np.zeros_like(target_np)
        for action in range(target_np.shape[1]):
            alpha_a, *_ = np.linalg.lstsq(
                basis_np,
                target_np[:, action],
                rcond=None,
            )
            coeffs.append(alpha_a)
            fitted[:, action] = basis_np @ alpha_a
        rmse = float(np.sqrt(np.mean((fitted - target_np) ** 2)))
        return (
            jnp.array(np.stack(coeffs, axis=0), dtype=jnp.float64).reshape(-1),
            rmse,
        )

    def _collocation_state_indices(
        self,
        n_states: int,
        obs_states: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Return deterministic collocation states plus observed states."""
        n_collocation = min(n_states, max(1, int(self._basis_dim)))
        base = np.linspace(0, n_states - 1, n_collocation).round().astype(np.int32)
        if obs_states is not None:
            base = np.concatenate([base, np.asarray(obs_states, dtype=np.int32)])
        return jnp.asarray(np.unique(base), dtype=jnp.int32)

    def _evaluate_solution_outputs(
        self,
        theta: jnp.ndarray,
        alpha: jnp.ndarray,
        *,
        basis: jnp.ndarray,
        feature_matrix: jnp.ndarray,
        problem: DDCProblem,
        transitions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evaluate the SEES solution object and its Bellman residual."""
        n_states = problem.num_states
        n_actions = problem.num_actions
        sigma = problem.scale_parameter
        beta = problem.discount_factor

        theta = jnp.asarray(theta, dtype=jnp.float64)
        alpha = jnp.asarray(alpha, dtype=jnp.float64)
        basis = jnp.asarray(basis, dtype=jnp.float64)
        feature_matrix = jnp.asarray(feature_matrix, dtype=jnp.float64)
        transitions_f64 = jnp.asarray(transitions, dtype=jnp.float64)
        basis_cols = int(basis.shape[1])

        flow_u = jnp.einsum("sak,k->sa", feature_matrix, theta)
        expected_basis = jnp.einsum("ast,tk->ask", transitions_f64, basis)

        def expected_value(value: jnp.ndarray) -> jnp.ndarray:
            return jnp.einsum("ast,t->as", transitions_f64, value).T

        def state_action_from_alpha(alpha_flat: jnp.ndarray) -> jnp.ndarray:
            alpha_by_action = alpha_flat.reshape((n_actions, basis_cols))
            return jnp.einsum("sk,ak->sa", basis, alpha_by_action)

        def center_actions(values: jnp.ndarray) -> jnp.ndarray:
            return values - values.mean(axis=1, keepdims=True)

        def policy_value(flow_u: jnp.ndarray, policy: jnp.ndarray) -> jnp.ndarray:
            clipped = jnp.clip(policy, 1e-12, 1.0)
            entropy_flow = -sigma * jnp.sum(policy * jnp.log(clipped), axis=1)
            reward_pi = jnp.sum(policy * flow_u, axis=1) + entropy_flow
            transition_pi = jnp.einsum("sa,ast->st", policy, transitions_f64)
            lhs = jnp.eye(n_states, dtype=jnp.float64) - beta * transition_pi
            return jnp.linalg.solve(lhs, reward_pi)

        if self._solution in {"value", "collocation"}:
            value_approx = basis @ alpha
            continuation = beta * jnp.einsum("ask,k->sa", expected_basis, alpha)
            q_vals = flow_u + continuation
            logits = q_vals / sigma
            policy = jax.nn.softmax(logits, axis=1)
            bellman_value = sigma * jax.scipy.special.logsumexp(logits, axis=1)
            residual = value_approx - bellman_value
            return logits, bellman_value, policy, residual, q_vals

        if self._solution == "q":
            q_vals = state_action_from_alpha(alpha)
            logits = q_vals / sigma
            policy = jax.nn.softmax(logits, axis=1)
            value = sigma * jax.scipy.special.logsumexp(logits, axis=1)
            target_q = flow_u + beta * expected_value(value)
            residual = q_vals - target_q
            return logits, value, policy, residual, q_vals

        if self._solution == "ev":
            continuation_value = state_action_from_alpha(alpha)
            q_vals = flow_u + beta * continuation_value
            logits = q_vals / sigma
            policy = jax.nn.softmax(logits, axis=1)
            value = sigma * jax.scipy.special.logsumexp(logits, axis=1)
            target_continuation = expected_value(value)
            residual = continuation_value - target_continuation
            return logits, value, policy, residual, q_vals

        raw_logits = state_action_from_alpha(alpha)
        logits = center_actions(raw_logits)
        policy = jax.nn.softmax(logits, axis=1)
        value = policy_value(flow_u, policy)
        target_q = flow_u + beta * expected_value(value)
        target_logits = center_actions(target_q / sigma)
        residual = logits - target_logits
        return logits, value, policy, residual, target_q

    def _static_logit_start(
        self,
        state_action_counts: jnp.ndarray,
        utility: UtilityFunction,
        problem: DDCProblem,
    ) -> jnp.ndarray | None:
        """Estimate a static-logit theta start from observed choices."""
        feature_matrix = jnp.asarray(utility.feature_matrix, dtype=jnp.float64)
        n_obs = jnp.maximum(state_action_counts.sum(), 1.0)
        sigma = problem.scale_parameter
        ridge = 1e-4

        def static_objective(theta: jnp.ndarray) -> jnp.ndarray:
            flow_u = jnp.einsum("sak,k->sa", feature_matrix, theta)
            log_probs = jax.nn.log_softmax(flow_u / sigma, axis=1)
            nll = -jnp.sum(state_action_counts * log_probs) / n_obs
            return nll + ridge * jnp.sum(theta**2)

        lower_theta, upper_theta = utility.get_parameter_bounds()
        bounds = (
            jnp.asarray(lower_theta, dtype=jnp.float64),
            jnp.asarray(upper_theta, dtype=jnp.float64),
        )
        try:
            result = minimize_lbfgsb(
                static_objective,
                utility.get_initial_parameters(),
                bounds=bounds,
                maxiter=min(200, self._max_iter),
                tol=max(self._tol, 1e-8),
                verbose=False,
                desc="SEES static-logit start",
                param_names=utility.parameter_names,
            )
        except Exception:
            return None
        theta = jnp.asarray(result.x, dtype=jnp.float64)
        if bool(jnp.all(jnp.isfinite(theta))):
            return theta
        return None

    def _theta_start_candidates(
        self,
        initial_params: jnp.ndarray,
        state_action_counts: jnp.ndarray,
        utility: UtilityFunction,
        problem: DDCProblem,
    ) -> list[jnp.ndarray]:
        """Build deterministic theta starts for multistart SEES."""
        candidates: list[jnp.ndarray] = []

        def add(theta: jnp.ndarray | None) -> None:
            if theta is None:
                return
            theta = jnp.asarray(theta, dtype=jnp.float64)
            if theta.shape != (utility.num_parameters,):
                return
            if not bool(jnp.all(jnp.isfinite(theta))):
                return
            for existing in candidates:
                if np.allclose(np.asarray(theta), np.asarray(existing), atol=1e-8):
                    return
            candidates.append(theta)

        add(initial_params)
        if self._num_theta_starts == 1:
            return candidates

        default_theta = jnp.asarray(
            utility.get_initial_parameters(),
            dtype=jnp.float64,
        )
        static_theta = self._static_logit_start(
            state_action_counts,
            utility,
            problem,
        )
        add(default_theta)
        add(static_theta)

        if static_theta is not None:
            neutral_static = static_theta.at[0].set(0.0)
            add(neutral_static)
            add(0.5 * static_theta)

        add(jnp.zeros(utility.num_parameters, dtype=jnp.float64))

        return candidates[: self._num_theta_starts]

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run sieve estimation (Luo and Sang 2024, equation 4).

        Jointly optimizes structural parameters theta and basis
        coefficients alpha by maximizing the penalized criterion:

            LL(theta, alpha) - omega * ||V(alpha) - T(V(alpha); theta)||^2

        where T is the soft Bellman operator. The penalty enforces
        the equilibrium condition V = T(V; theta).
        """
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        if self._penalty_schedule is not None:
            n_obs = sum(len(t.actions) for t in panel.trajectories)
            omega = float(self._penalty_schedule(n_obs))
        else:
            omega = self._penalty_weight

        feature_matrix = jnp.array(utility.feature_matrix, dtype=jnp.float64)
        n_theta = utility.num_parameters
        obs_states = panel.get_all_states()
        obs_actions = panel.get_all_actions()
        n_obs = int(panel.num_observations)

        state_action_counts = jnp.zeros((n_states, n_actions), dtype=jnp.float64)
        state_action_counts = state_action_counts.at[obs_states, obs_actions].add(1.0)

        # Build sieve basis
        basis = self._build_basis(n_states, problem)  # (S, basis_dim)
        basis_metadata = dict(self._last_basis_metadata)
        basis_cols = int(basis.shape[1])
        n_alpha = (
            basis_cols
            if self._solution in {"value", "collocation"}
            else n_actions * basis_cols
        )

        transitions_f64 = jnp.array(transitions, dtype=jnp.float64)

        if initial_params is None:
            initial_params = utility.get_initial_parameters()
        initial_params = jnp.array(initial_params, dtype=jnp.float64)

        collocation_indices = self._collocation_state_indices(n_states, obs_states)

        def expected_value(value: jnp.ndarray) -> jnp.ndarray:
            return jnp.einsum("ast,t->as", transitions_f64, value).T

        def initial_alpha_for(
            theta_start: jnp.ndarray,
        ) -> tuple[jnp.ndarray, float, bool]:
            """Build initial alpha for a specific theta start."""
            if not self._warm_start_value:
                return jnp.zeros(n_alpha, dtype=jnp.float64), float("nan"), False

            flow_u0 = jnp.einsum("sak,k->sa", feature_matrix, theta_start)
            operator = SoftBellmanOperator(problem, transitions_f64)
            solution0 = value_iteration(
                operator,
                flow_u0,
                tol=1e-10,
                max_iter=10_000,
            )
            projection_converged = bool(solution0.converged)
            if self._solution in {"value", "collocation"}:
                basis_np = np.asarray(basis, dtype=np.float64)
                value_np = np.asarray(solution0.V, dtype=np.float64)
                alpha_np, *_ = np.linalg.lstsq(basis_np, value_np, rcond=None)
                projected = basis_np @ alpha_np
                projection_rmse = float(
                    np.sqrt(np.mean((projected - value_np) ** 2))
                )
                initial_alpha = jnp.array(alpha_np, dtype=jnp.float64)
            elif self._solution == "q":
                initial_alpha, projection_rmse = self._project_state_action_solution(
                    basis,
                    solution0.Q,
                )
            elif self._solution == "ev":
                continuation0 = expected_value(solution0.V)
                initial_alpha, projection_rmse = self._project_state_action_solution(
                    basis,
                    continuation0,
                )
            else:
                policy0 = np.asarray(solution0.policy, dtype=np.float64)
                logits0 = np.log(np.clip(policy0, 1e-12, 1.0))
                logits0 = logits0 - logits0.mean(axis=1, keepdims=True)
                initial_alpha, projection_rmse = self._project_state_action_solution(
                    basis,
                    jnp.array(logits0, dtype=jnp.float64),
                )
            return initial_alpha, projection_rmse, projection_converged

        def solution_outputs(
            theta: jnp.ndarray,
            alpha: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            return self._evaluate_solution_outputs(
                theta,
                alpha,
                basis=basis,
                feature_matrix=feature_matrix,
                problem=problem,
                transitions=transitions_f64,
            )

        def criterion_parts(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            theta = x[:n_theta]
            alpha = x[n_theta:]
            logits, _, _, residual, _ = solution_outputs(theta, alpha)

            log_probs = jax.nn.log_softmax(logits, axis=1)
            ll_sum = jnp.sum(state_action_counts * log_probs)

            if self._solution == "collocation":
                penalty_residual = residual[collocation_indices]
            else:
                penalty_residual = residual
            bellman_mse = jnp.mean(penalty_residual ** 2)

            return ll_sum, bellman_mse

        def penalized_criterion_mean(x: jnp.ndarray) -> jnp.ndarray:
            ll_sum, bellman_mse = criterion_parts(x)
            return ll_sum / n_obs - omega * bellman_mse

        def penalized_criterion_sum(x: jnp.ndarray) -> jnp.ndarray:
            return n_obs * penalized_criterion_mean(x)

        self._ll_fn = penalized_criterion_sum

        # JAX autodiff objective (minimization: negate the penalized log-likelihood)
        _neg_penalized_ll = jax.jit(lambda x: -penalized_criterion_mean(x))

        # Bounds: theta bounds from utility, alpha bounded loosely
        lower_theta, upper_theta = utility.get_parameter_bounds()
        alpha_bound = 1e5
        lower = jnp.concatenate([
            lower_theta.astype(jnp.float64),
            jnp.full((n_alpha,), -alpha_bound, dtype=jnp.float64),
        ])
        upper = jnp.concatenate([
            upper_theta.astype(jnp.float64),
            jnp.full((n_alpha,), alpha_bound, dtype=jnp.float64),
        ])

        self._log(
            f"SEES: {n_theta} structural + {n_alpha} basis params, "
            f"solution={self._solution}, omega={omega}, "
            f"basis_source={basis_metadata.get('basis_source')}, "
            f"theta_starts={self._num_theta_starts}"
        )

        theta_starts = self._theta_start_candidates(
            initial_params,
            state_action_counts,
            utility,
            problem,
        )
        best_run: tuple[object, float, bool, jnp.ndarray, jnp.ndarray] | None = None
        start_records: list[dict[str, object]] = []
        total_iterations = 0
        total_function_evals = 0
        for start_idx, theta_start in enumerate(theta_starts):
            initial_alpha, start_projection_rmse, start_projection_converged = (
                initial_alpha_for(theta_start)
            )
            x0 = jnp.concatenate([theta_start, initial_alpha])
            result_i = minimize_lbfgsb(
                _neg_penalized_ll,
                x0,
                bounds=(lower, upper),
                maxiter=self._max_iter,
                tol=self._tol,
                verbose=self._verbose,
                desc=f"SEES L-BFGS-B start {start_idx + 1}/{len(theta_starts)}",
                param_names=utility.parameter_names,
            )
            total_iterations += int(result_i.nit)
            total_function_evals += int(result_i.nfev)
            fun_i = float(result_i.fun)
            start_records.append(
                {
                    "start_index": start_idx,
                    "theta_start": np.asarray(theta_start, dtype=float).tolist(),
                    "objective": fun_i,
                    "converged": bool(result_i.success),
                    "iterations": int(result_i.nit),
                    "initial_solution_projection_rmse": start_projection_rmse,
                    "initial_solution_projection_converged": (
                        start_projection_converged
                    ),
                }
            )
            if not np.isfinite(fun_i):
                continue
            if best_run is None or fun_i < float(best_run[0].fun):
                best_run = (
                    result_i,
                    start_projection_rmse,
                    start_projection_converged,
                    theta_start,
                    initial_alpha,
                )

        if best_run is None:
            raise RuntimeError("SEES optimization failed for every theta start")

        result, projection_rmse, projection_converged, selected_theta_start, _ = best_run

        x_opt = jnp.array(result.x, dtype=jnp.float64)
        theta_opt = x_opt[:n_theta]
        alpha_opt = x_opt[n_theta:]
        try:
            final_gradient_norm = float(
                jnp.linalg.norm(jax.grad(_neg_penalized_ll)(x_opt))
            )
        except Exception:
            final_gradient_norm = float("nan")

        self._log(f"theta: {np.asarray(theta_opt)}")
        self._log(f"alpha: {np.asarray(alpha_opt)}")

        # Compute final policy and value function
        logits, V, policy, bellman_residual, q_vals = solution_outputs(
            theta_opt,
            alpha_opt,
        )

        # Compute pure log-likelihood (without penalty) for reporting
        log_probs = jax.nn.log_softmax(logits, axis=1)
        ll_opt = float(log_probs[obs_states, obs_actions].sum())

        # Bellman violation at solution
        bellman_viol = float(jnp.max(jnp.abs(bellman_residual)))
        bellman_rmse = float(jnp.sqrt(jnp.mean(bellman_residual**2)))
        collocation_viol = float("nan")
        collocation_rmse = float("nan")
        if self._solution == "collocation":
            collocation_residual = bellman_residual[collocation_indices]
            collocation_viol = float(jnp.max(jnp.abs(collocation_residual)))
            collocation_rmse = float(
                jnp.sqrt(jnp.mean(collocation_residual**2))
            )

        # Hessian for theta SEs via Schur complement (Corollary 3.1)
        # H = H_theta_theta - H'_beta_theta @ H_beta_beta^{-1} @ H_beta_theta
        hessian = None
        if self._compute_se:
            self._log("Computing Hessian for standard errors (Schur complement)")

            full_hessian = jax.hessian(penalized_criterion_sum)(
                jnp.array(x_opt, dtype=jnp.float64)
            )

            H_tt = full_hessian[:n_theta, :n_theta]
            H_ta = full_hessian[:n_theta, n_theta:]
            H_at = full_hessian[n_theta:, :n_theta]
            H_aa = full_hessian[n_theta:, n_theta:]

            try:
                hessian = H_tt - H_ta @ jnp.linalg.solve(H_aa, H_at)
                if not bool(jnp.all(jnp.isfinite(hessian))):
                    raise np.linalg.LinAlgError("non-finite Schur complement")
            except Exception:
                self._log("WARNING: H_aa singular, using ridge-regularized Schur complement")
                ridge = 1e-8 * jnp.eye(H_aa.shape[0], dtype=H_aa.dtype)
                hessian = H_tt - H_ta @ jnp.linalg.solve(H_aa - ridge, H_at)

        elapsed = time.time() - start_time

        return EstimationResult(
            parameters=theta_opt,
            log_likelihood=ll_opt,
            value_function=V,
            policy=policy,
            hessian=hessian,
            converged=result.success,
            num_iterations=result.nit,
            num_function_evals=result.nfev,
            message=f"SEES: {result.message}",
            optimization_time=elapsed,
            metadata={
                "alpha": alpha_opt,
                "solution_type": self._solution,
                "num_theta_starts": len(theta_starts),
                "selected_theta_start": np.asarray(
                    selected_theta_start,
                    dtype=float,
                ).tolist(),
                "theta_start_results": start_records,
                "multistart_total_iterations": total_iterations,
                "multistart_total_function_evals": total_function_evals,
                "selected_objective": float(result.fun),
                "selected_gradient_norm": final_gradient_norm,
                "optimizer_success": bool(result.success),
                "basis_type": self._basis_type,
                "basis_dim": n_alpha,
                "state_basis_dim": basis_cols,
                "configured_basis_dim": self._basis_dim,
                "basis_matrix": basis,
                "bellman_violation": bellman_viol,
                "bellman_rmse": bellman_rmse,
                "collocation_violation": collocation_viol,
                "collocation_rmse": collocation_rmse,
                "collocation_state_count": int(collocation_indices.shape[0]),
                "alpha_shape": (
                    (basis_cols,)
                    if self._solution in {"value", "collocation"}
                    else (n_actions, basis_cols)
                ),
                "penalty_weight": omega,
                "penalty_objective_scale": "mean_loglik_minus_omega_bellman_mse",
                "warm_start_value": self._warm_start_value,
                "initial_value_projection_rmse": projection_rmse,
                "initial_solution_projection_rmse": projection_rmse,
                "initial_value_projection_converged": projection_converged,
                "initial_solution_projection_converged": projection_converged,
                **basis_metadata,
            },
        )
