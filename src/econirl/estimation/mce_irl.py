"""Maximum Causal Entropy Inverse Reinforcement Learning (MCE IRL).

This module implements Maximum Causal Entropy IRL from Ziebart's 2010 thesis,
which properly accounts for the causal structure in sequential decision-making.

The key difference from standard MaxEnt IRL is the backward pass uses soft
value iteration which respects that agents don't know future randomness
at decision time.

Algorithm (following Ziebart 2010):
    BACKWARD PASS:
    1. Initialize Q(s,a) and V(s) at terminal states
    2. Propagate backwards using soft Bellman:
       Q(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')
       V(s) = softmax(Q(s,·))

    FORWARD PASS:
    3. Compute state visitation frequencies:
       D(s) = ρ₀(s) + Σ_{s',a} D(s') π(a|s') P(s|s',a)

    GRADIENT:
    4. Δθ = E_expert[∇R] - E_policy[∇R]
          = empirical_features - Σ_s D(s) ∇R(s)

Gradient Computation — Two Approaches
======================================

This implementation supports two gradient modes, corresponding to two different
ways of computing "what features does the policy visit":

(A) OCCUPANCY MODE (default, ``gradient_mode="occupancy"``)
------------------------------------------------------------
The correct MCE IRL gradient (Ziebart 2010, Chapter 9):

    ∇_θ L(θ) = E_demo[φ(s,a)] - E_π[φ(s,a)]
             = (1/N) Σ_{i,t} φ(s_{it}, a_{it})        [demo: actual expert actions]
               - Σ_s D_π(s) · Σ_a π(a|s) · φ(s,a)    [model: D_π × policy features]
             = empirical_features - D_π @ policy_weighted_features

Demo side: empirical_features uses actual expert (s,a) pairs from data.
Model side: D_π(s) from forward pass × Σ_a π(a|s) φ(s,a,k) per state.

Why this works for ALL feature types (state-only, state-action, Rust-style):
- For state-action φ(s,a): model side correctly uses D_π(s) instead of D_demo(s).
  This is the root fix: feature_matching mode (buggy) used D_demo for the model side,
  causing wrong gradients when D_demo ≠ D_π.
- For state-only φ(s): policy_weighted[s,k] = Σ_a π(a|s)·φ(s,k) = φ(s,k),
  and empirical_features ≈ D_demo @ φ, so gradient ≈ (D_demo - D_π) @ φ —
  non-zero whenever D_π ≠ D_demo, which happens until π = π*.

Key insight: model side uses D_π (depends on θ through the policy), so the
gradient is non-zero even for state-only rewards. The occupancy changes as
θ changes, providing gradient signal for all reward parameterizations.

Reference: Ziebart (2010) Theorem 6.1, feature matching at the optimum:
    E_demo[φ] = E_π*[φ]   ↔   ∇L = 0   ↔   θ = θ*

(B) FEATURE MATCHING MODE (legacy, ``gradient_mode="feature_matching"``)
-------------------------------------------------------------------------
Follows Ziebart (2010) thesis Chapter 9 and Wulfmeier (2016) Deep MaxEnt IRL.

    ∇_θ L(θ) = E_D[φ(S,A)] - E_π[φ(S,A)]

where both expectations use per-observation averaging over the EMPIRICAL
state sequence from demonstrations:
    E_D[φ]_k = (1/N) Σ_{i,t} φ(s_{it}, a_{it})          [actual expert actions]
    E_π[φ]_k = (1/N) Σ_{i,t} Σ_a π(a|s_{it}) φ(s_{it},a)  [policy at demo states]

Wulfmeier (2016) Algorithm 1 makes this explicit: the backward pass computes a
soft policy, the forward pass propagates it through demo states to get expected
features, and the gradient is the difference.

FAILURE CASE for state-only φ(s):
    E_π[φ]_k = (1/N) Σ_{i,t} Σ_a π(a|s_{it}) φ(s_{it},k)
             = (1/N) Σ_{i,t} φ(s_{it},k) · [Σ_a π(a|s_{it})]  ← policy sums to 1
             = (1/N) Σ_{i,t} φ(s_{it},k)
             = E_D[φ]_k   ← always equal, regardless of policy!
    ∴ gradient = 0 always.  The policy has NO effect on the gradient.

References:
    Ziebart et al. (2008). Maximum entropy inverse reinforcement learning. AAAI.
        Algorithm 1, Eq. 6 (state visitation gradient).
    Ziebart (2010). Modeling purposeful adaptive behavior with the principle of
        maximum causal entropy. PhD thesis, CMU.
    Gleave & Toyer (2022). A primer on maximum causal entropy inverse reinforcement
        learning. arXiv:2203.11409. Eq. 59.
    Wulfmeier, Ondruska & Posner (2016). Maximum entropy deep inverse reinforcement
        learning. IJCAI. Algorithm 1 (forward-backward IRL passes).
    imitation library: https://github.com/HumanCompatibleAI/imitation
        Uses loss = dot(D_π - D_demo, r), then backprop through reward network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy import optimize
from tqdm import tqdm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration, value_iteration, backward_induction
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.base import BaseUtilityFunction
from econirl.preferences.reward import LinearReward


@dataclass
class MCEIRLConfig:
    """Configuration for MCE IRL estimation."""

    # Optimization
    optimizer: Literal["L-BFGS-B", "BFGS", "gradient"] = "L-BFGS-B"
    learning_rate: float = 0.02  # Lower than typical SGD; works well with Adam
    outer_tol: float = 1e-6
    outer_max_iter: int = 200
    gradient_clip: float = 1.0  # Max gradient norm (prevents divergence)
    use_adam: bool = True  # Use Adam optimizer (adaptive learning rate)
    adam_beta1: float = 0.9  # Adam first moment decay
    adam_beta2: float = 0.999  # Adam second moment decay
    adam_eps: float = 1e-8  # Adam numerical stability

    # Inner solver (soft value iteration)
    inner_solver: Literal["value", "hybrid"] = "hybrid"  # hybrid is faster
    inner_tol: float = 1e-8
    inner_max_iter: int = 10000  # Higher for high discount factors
    switch_tol: float = 1e-3  # For hybrid: switch to NK when error < this

    # State visitation computation
    svf_tol: float = 1e-8
    svf_max_iter: int = 1000

    # Inference
    compute_se: bool = True
    se_method: Literal["bootstrap", "asymptotic", "hessian"] = "bootstrap"
    n_bootstrap: int = 100

    # Gradient computation mode
    gradient_mode: Literal["occupancy", "feature_matching"] = "occupancy"
    """Gradient computation mode.

    "occupancy" (default, Ziebart 2008 Algorithm 1):
        Compares state occupancy distributions D_demo(s) vs D_pi(s), both
        weighted by the CURRENT policy's expected features:
            ∇L_k = Σ_s (D_demo(s) - D_pi(s)) · [Σ_a π(a|s) φ(s,a,k)]

        This is equivalent to the `imitation` library's approach:
            loss = dot(D_π - D_demo, r)  [then backprop through r = θ^T φ]
        (Gleave 2022, Eq. 59; Ziebart 2008, Sec. 3)

        Works correctly for ALL feature types:
        - State-only φ(s): Σ_a π(a|s)·φ(s,k) = φ(s,k) [π sums to 1]
        - State-action φ(s,a): correctly marginalizes via D_pi(s)·Σ_a π(a|s)·φ(s,a,k)
        - Rust-style: same as state-action case

    "feature_matching" (legacy):
        Direct feature matching: E_demo[φ] - E_π[φ(s_demo,·)]
        The model side uses the EMPIRICAL state distribution from demos,
        NOT the model's D_pi(s). For state-action features, this means:
            expected_k = (1/N) Σ_{i,t} Σ_a π(a|s_{it}) φ(s_{it},a,k)
        FAILS for state-only features: Σ_a π(a|s)·φ(s,k) = φ(s,k), so
        expected = empirical → zero gradient always.
    """

    # Verbosity
    verbose: bool = False


class MCEIRLEstimator(BaseEstimator):
    """Maximum Causal Entropy IRL Estimator.

    Recovers reward function parameters from demonstrated behavior using
    the maximum causal entropy principle (Ziebart 2010).

    This differs from standard MaxEnt IRL in that it properly accounts for
    the causal structure of sequential decisions - agents act before observing
    future randomness.

    Parameters
    ----------
    config : MCEIRLConfig, optional
        Configuration object with algorithm parameters.
    **kwargs
        Override individual config parameters.

    Attributes
    ----------
    config : MCEIRLConfig
        Configuration object.

    Examples
    --------
    >>> from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
    >>> from econirl.preferences.reward import LinearReward
    >>>
    >>> # Create estimator with custom config
    >>> config = MCEIRLConfig(verbose=True, n_bootstrap=200)
    >>> estimator = MCEIRLEstimator(config=config)
    >>>
    >>> # Estimate reward from demonstrations
    >>> result = estimator.estimate(panel, reward_fn, problem, transitions)
    >>> print(result.summary())
    """

    def __init__(
        self,
        config: MCEIRLConfig | None = None,
        **kwargs,
    ):
        # Build config from defaults + overrides
        if config is None:
            config = MCEIRLConfig(**kwargs)
        else:
            # Apply any kwargs as overrides
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Map "hessian" to "asymptotic" for compatibility with standard_errors module
        # (they are semantically equivalent: both use inverse Hessian for variance)
        if config.compute_se:
            if config.se_method == "hessian":
                effective_se_method = "asymptotic"
            else:
                effective_se_method = config.se_method
        else:
            effective_se_method = "asymptotic"

        super().__init__(
            se_method=effective_se_method,
            compute_hessian=config.compute_se,
            verbose=config.verbose,
        )
        self.config = config

    @property
    def name(self) -> str:
        return "MCE IRL (Ziebart 2010)"

    def _soft_value_iteration(
        self,
        operator: SoftBellmanOperator,
        reward_matrix: torch.Tensor,
        num_periods: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Run soft value iteration (backward pass).

        For infinite horizon: uses contraction/hybrid iteration.
        For finite horizon: uses backward induction (deterministic, no convergence needed).

        Parameters
        ----------
        operator : SoftBellmanOperator
            Bellman operator with problem and transitions.
        reward_matrix : torch.Tensor
            Reward matrix R(s,a), shape (n_states, n_actions).
        num_periods : int, optional
            If set, use finite-horizon backward induction.

        Returns
        -------
        V : torch.Tensor
            Soft value function, shape (n_states,) for infinite horizon,
            or shape (num_periods, n_states) for finite horizon.
        policy : torch.Tensor
            Soft policy π(a|s), shape (n_states, n_actions) for infinite horizon,
            or shape (num_periods, n_states, n_actions) for finite horizon.
        converged : bool
            Whether the iteration converged (always True for finite horizon).
        """
        if num_periods is not None:
            # Finite horizon: backward induction (deterministic, no convergence needed)
            fh_result = backward_induction(operator, reward_matrix, num_periods)
            return fh_result.V, fh_result.policy, True

        if self.config.inner_solver == "hybrid":
            result = hybrid_iteration(
                operator,
                reward_matrix,
                tol=self.config.inner_tol,
                max_iter=self.config.inner_max_iter,
                switch_tol=self.config.switch_tol,
            )
            return result.V, result.policy, result.converged
        else:
            # Pure value iteration (original implementation)
            result = value_iteration(
                operator,
                reward_matrix,
                tol=self.config.inner_tol,
                max_iter=self.config.inner_max_iter,
            )
            return result.V, result.policy, result.converged

    def _compute_state_visitation(
        self,
        policy: torch.Tensor,
        transitions: torch.Tensor,
        problem: DDCProblem,
        initial_dist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute expected state visitation frequencies (forward pass).

        Uses forward message passing to compute the expected number of times
        each state is visited under the policy.

        Parameters
        ----------
        policy : torch.Tensor
            Policy π(a|s), shape (n_states, n_actions).
        transitions : torch.Tensor
            Transition matrices P(s'|s,a), shape (n_actions, n_states, n_states).
        problem : DDCProblem
            Problem specification.
        initial_dist : torch.Tensor, optional
            Initial state distribution ρ₀. If None, uses uniform.

        Returns
        -------
        D : torch.Tensor
            State visitation frequencies, shape (n_states,).
        """
        n_states = problem.num_states
        gamma = problem.discount_factor

        if initial_dist is None:
            D = torch.ones(n_states, dtype=policy.dtype) / n_states
        else:
            D = initial_dist.clone()

        # Compute policy-weighted transition: P_π(s'|s) = Σ_a π(a|s) P(s'|s,a)
        # transitions: (n_actions, n_states, n_states) = [a, from_s, to_s]
        # policy: (n_states, n_actions) = [s, a]
        P_pi = torch.einsum("sa,ast->st", policy, transitions)

        # Fixed point iteration: D = ρ₀ + γ P_π^T D
        # Equivalently: D = (I - γ P_π^T)^{-1} ρ₀
        # But we use iteration for numerical stability

        rho0 = D.clone()
        for i in range(self.config.svf_max_iter):
            D_new = rho0 + gamma * (P_pi.T @ D)

            delta = torch.abs(D_new - D).max().item()
            D = D_new

            if delta < self.config.svf_tol:
                break

        # Normalize to probability distribution
        D = D / D.sum()

        return D

    def _compute_empirical_features(
        self,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
    ) -> torch.Tensor:
        """Compute empirical feature expectations from demonstrations.

        Computes the average state-action features visited by the demonstrator:
            μ_D = (1/N) Σ_{i,t} φ(s_{i,t}, a_{i,t})

        For state-only features: μ_D = (1/N) Σ_{i,t} φ(s_{i,t})

        Handles both:
        - LinearReward (state-only): 2D feature matrix (n_states, n_features)
        - ActionDependentReward: 3D feature matrix (n_states, n_actions, n_features)
        """
        # Get feature matrix - handle both 2D and 3D cases
        if isinstance(reward_fn, ActionDependentReward):
            feature_matrix = reward_fn.feature_matrix  # (n_states, n_actions, n_features)
        elif isinstance(reward_fn, LinearReward):
            feature_matrix = reward_fn.state_features  # (n_states, n_features)
        else:
            # Try to get feature_matrix, fall back to state_features
            if hasattr(reward_fn, 'feature_matrix'):
                feature_matrix = reward_fn.feature_matrix
            elif hasattr(reward_fn, 'state_features'):
                feature_matrix = reward_fn.state_features
            else:
                raise ValueError(f"Unsupported reward function type: {type(reward_fn)}")

        total_obs = sum(len(traj) for traj in panel.trajectories)

        if feature_matrix.ndim == 3:
            # Action-dependent features: (n_states, n_actions, n_features)
            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            feature_sum = feature_matrix[all_states, all_actions, :].sum(dim=0)
        else:
            # State-only features: (n_states, n_features)
            all_states = panel.get_all_states()
            feature_sum = feature_matrix[all_states, :].sum(dim=0)

        if total_obs > 0:
            return feature_sum / total_obs
        return feature_sum

    def _compute_expected_features(
        self,
        panel: Panel,
        policy: torch.Tensor,
        reward_fn: BaseUtilityFunction,
        transitions: torch.Tensor | None = None,
        initial_dist: torch.Tensor | None = None,
        discount: float = 0.9,
    ) -> torch.Tensor:
        """Compute expected feature expectations under the learned policy.

        This implements the "feature matching" (legacy) mode. It is used by
        the ``feature_matching`` gradient mode and for post-hoc reporting.

        For action-dependent features (3D), computes expectations over the
        EMPIRICAL state sequence, weighting by the current policy's action
        probabilities:

            μ_π = (1/N) Σ_{i,t} Σ_a π(a|s_{i,t}) φ(s_{i,t}, a)

        Note on limitations (see module docstring for full discussion):
        This method uses the EMPIRICAL state distribution (s_{i,t} from demos),
        NOT the model's D_π(s). For state-only features φ(s), the policy factor
        Σ_a π(a|s)=1 cancels, making μ_π identical to μ_D regardless of policy.
        This renders the gradient zero and optimization ineffective for state-only
        rewards. Use "occupancy" gradient mode to avoid this issue.

        For state-only features (2D), computes expectations using the model's
        state visitation distribution (the correct approach for this case):

            μ_π = Σ_s d_π(s) φ(s)

        Why occupancy mode fixes this for ALL feature types:
        In occupancy mode, both the demo side and model side use state
        occupancy distributions D_demo(s) and D_π(s). The gradient becomes:
            ∇L_k = Σ_s (D_demo(s) - D_π(s)) · Σ_a π(a|s) φ(s,a,k)
        Since D_π depends on θ through the policy, this is non-zero even for
        state-only features — the key insight from Ziebart (2008) Eq. 6.

        Parameters
        ----------
        panel : Panel
            Panel data containing the empirical state sequence.
        policy : torch.Tensor
            Policy probabilities π(a|s), shape (n_states, n_actions).
        reward_fn : BaseUtilityFunction
            Reward function with feature matrix.
        transitions : torch.Tensor, optional
            Transition matrices, shape (n_actions, n_states, n_states).
            Required for state-only features.
        initial_dist : torch.Tensor, optional
            Initial state distribution. Required for state-only features.
        discount : float
            Discount factor for state visitation computation.

        Returns
        -------
        torch.Tensor
            Expected features, shape (n_features,).
        """
        # Get feature matrix - handle both 2D and 3D cases
        if isinstance(reward_fn, ActionDependentReward):
            feature_matrix = reward_fn.feature_matrix  # (n_states, n_actions, n_features)
        elif isinstance(reward_fn, LinearReward):
            feature_matrix = reward_fn.state_features  # (n_states, n_features)
        else:
            # Try to get feature_matrix, fall back to state_features
            if hasattr(reward_fn, 'feature_matrix'):
                feature_matrix = reward_fn.feature_matrix
            elif hasattr(reward_fn, 'state_features'):
                feature_matrix = reward_fn.state_features
            else:
                raise ValueError(f"Unsupported reward function type: {type(reward_fn)}")

        total_obs = sum(len(traj) for traj in panel.trajectories)

        if feature_matrix.ndim == 3:
            # Action-dependent: (n_states, n_actions, n_features)
            # E[φ] = (1/N) Σ_{i,t} Σ_a π(a|s_{i,t}) * φ(s_{i,t}, a, k)
            # Vectorized: gather all states, then batch multiply
            all_states = torch.cat([traj.states for traj in panel.trajectories])
            # policy[all_states] → (N, n_actions), feature_matrix[all_states] → (N, n_actions, n_features)
            # Sum over actions and observations
            feature_sum = (policy[all_states].unsqueeze(-1) * feature_matrix[all_states]).sum(dim=(0, 1))

            if total_obs > 0:
                return feature_sum / total_obs
            return feature_sum
        else:
            # State-only: (n_states, n_features)
            # Use state visitation frequencies under the current policy
            n_states = feature_matrix.shape[0]

            if transitions is not None and initial_dist is not None:
                # Create a temporary DDCProblem for the existing _compute_state_visitation
                problem = DDCProblem(
                    num_states=n_states,
                    num_actions=policy.shape[1],
                    discount_factor=discount,
                )
                d = self._compute_state_visitation(
                    policy, transitions, problem, initial_dist
                )
                # E_π[φ] = Σ_s d_π(s) * φ(s)
                return d @ feature_matrix
            else:
                # Fallback: iterate over empirical states
                all_states = panel.get_all_states()
                feature_sum = feature_matrix[all_states, :].sum(dim=0)
                if total_obs > 0:
                    return feature_sum / total_obs
                return feature_sum

    def _compute_expected_features_finite_horizon(
        self,
        panel: Panel,
        policy: torch.Tensor,
        reward_fn: BaseUtilityFunction,
        num_periods: int,
    ) -> torch.Tensor:
        """Compute expected features under time-indexed policy for finite horizon.

        For each observation (s_it, t), uses the period-specific policy π_t(a|s)
        to compute expected features: (1/N) Σ_{i,t} Σ_a π_t(a|s_{it}) φ(s_{it}, a)

        Parameters
        ----------
        policy : torch.Tensor
            Time-indexed policy, shape (num_periods, n_states, n_actions).
        """
        if hasattr(reward_fn, 'feature_matrix'):
            feature_matrix = reward_fn.feature_matrix
        elif hasattr(reward_fn, 'state_features'):
            feature_matrix = reward_fn.state_features
        else:
            raise ValueError(f"Unsupported reward function type: {type(reward_fn)}")

        total_obs = sum(len(traj) for traj in panel.trajectories)

        if feature_matrix.ndim == 3:
            # Action-dependent: (n_states, n_actions, n_features)
            n_features = feature_matrix.shape[2]
            feature_sum = torch.zeros(n_features)

            for traj in panel.trajectories:
                for t in range(len(traj.states)):
                    period = min(t, num_periods - 1)
                    s = traj.states[t].item()
                    # Σ_a π_t(a|s) * φ(s, a, k)
                    feature_sum += (policy[period, s].unsqueeze(-1) * feature_matrix[s]).sum(dim=0)

            if total_obs > 0:
                return feature_sum / total_obs
            return feature_sum
        else:
            # State-only features: same as infinite horizon (no time dependence in features)
            all_states = panel.get_all_states()
            feature_sum = feature_matrix[all_states, :].sum(dim=0)
            if total_obs > 0:
                return feature_sum / total_obs
            return feature_sum

    def _compute_initial_distribution(
        self,
        panel: Panel,
        n_states: int,
    ) -> torch.Tensor:
        """Compute initial state distribution from data."""
        counts = torch.zeros(n_states, dtype=torch.float32)
        init_states = torch.tensor(
            [traj.states[0].item() for traj in panel.trajectories if len(traj) > 0],
            dtype=torch.long,
        )
        counts.scatter_add_(0, init_states, torch.ones_like(init_states, dtype=torch.float32))

        if counts.sum() > 0:
            return counts / counts.sum()
        return torch.ones(n_states) / n_states

    def _optimize(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        true_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run MCE IRL optimization.

        Parameters
        ----------
        true_params : torch.Tensor, optional
            True parameters for debugging. If provided, RMSE is shown in progress bar.
        """
        start_time = time.time()

        reward_fn = utility
        operator = SoftBellmanOperator(problem, transitions)

        # Initialize parameters
        if initial_params is None:
            params = reward_fn.get_initial_parameters()
        else:
            params = initial_params.clone()

        # Determine horizon type up front — needed for pre-loop setup below.
        finite_horizon = problem.num_periods is not None
        num_periods = problem.num_periods

        # Compute empirical features (constant throughout optimization).
        # For feature_matching mode: μ_D = (1/N) Σ_{i,t} φ(s_{it}, a_{it})
        # For occupancy mode: used only for logging (gradient is computed differently)
        empirical_features = self._compute_empirical_features(panel, reward_fn)
        initial_dist = self._compute_initial_distribution(panel, problem.num_states)

        self._log(f"Empirical features: {empirical_features}")
        self._log(f"Initial distribution entropy: {-(initial_dist * torch.log(initial_dist + 1e-10)).sum():.3f}")

        # ---------------------------------------------------------------
        # Pre-extract feature matrix for gradient computation (outside loop
        # to avoid repeated attribute lookups per iteration).
        # Shape: (n_states, n_features) for state-only, or
        #        (n_states, n_actions, n_features) for state-action features.
        # ---------------------------------------------------------------
        if isinstance(reward_fn, ActionDependentReward):
            feature_matrix = reward_fn.feature_matrix  # (n_states, n_actions, n_features)
        elif isinstance(reward_fn, LinearReward):
            feature_matrix = reward_fn.state_features  # (n_states, n_features)
        else:
            if hasattr(reward_fn, 'feature_matrix'):
                feature_matrix = reward_fn.feature_matrix
            elif hasattr(reward_fn, 'state_features'):
                feature_matrix = reward_fn.state_features
            else:
                raise ValueError(f"Unsupported reward function type: {type(reward_fn)}")

        # ---------------------------------------------------------------
        # Note: D_demo is NOT precomputed here. The occupancy mode uses
        # empirical_features (already computed above) for the demo side, which
        # contains the actual expert actions: (1/N) Σ_{i,t} φ(s_{it}, a_{it}).
        # This is correct because the MCE IRL gradient on the demo side is
        # E_demo[∇r] = E_demo[φ], which uses actual expert actions, not
        # the current policy. Using actual actions gives the correct gradient
        # for state-action features where the demo actions encode the expert's
        # behavior, not just which states they visited.
        D_demo = None  # Kept for type consistency; unused in occupancy mode

        # Tracking
        n_function_evals = 0
        inner_not_converged = 0

        # ---------------------------------------------------------------
        # Choose optimizer: scipy L-BFGS-B/BFGS or Adam/SGD.
        #
        # L-BFGS-B uses second-order curvature information (BFGS approximation
        # to the Hessian inverse) and line search, giving much faster convergence
        # than Adam for small-to-medium parameter spaces. MaxEnt IRL uses
        # L-BFGS-B and routinely achieves cosine similarity ≈ 0.997.
        #
        # Adam is simpler to implement but has a constant effective step size
        # (lr / sqrt(v)), which causes parameters to keep moving even near the
        # optimum. This leads to divergence unless lr is very small, making it
        # poorly suited for IRL where LL surfaces can be flat.
        #
        # Activation rule:
        #   - use_adam=False AND optimizer in ("L-BFGS-B", "BFGS") → scipy
        #   - use_adam=True  → Adam (legacy; use_adam flag overrides optimizer)
        #   - use_adam=False, optimizer="gradient"                  → SGD
        # ---------------------------------------------------------------
        use_scipy = (not self.config.use_adam) and (self.config.optimizer in ("L-BFGS-B", "BFGS"))

        if use_scipy:
            # -----------------------------------------------------------
            # SCIPY L-BFGS-B / BFGS PATH
            # Closure packs tensor → numpy → scipy, computes value+gradient,
            # returns numpy so scipy can apply its line search.
            # -----------------------------------------------------------
            self._log(f"Starting MCE IRL with scipy {self.config.optimizer}")

            _call_count = [0]
            _inner_nc = [0]

            all_states_idx = panel.get_all_states()
            all_actions_idx = panel.get_all_actions()

            def _objective_and_gradient(theta_np: np.ndarray):
                """Closure: returns (-LL, -gradient) for scipy.minimize (minimization)."""
                _call_count[0] += 1
                params_t = torch.tensor(theta_np, dtype=torch.float32)

                # Backward pass: soft Bellman → policy
                reward_matrix = reward_fn.compute(params_t)
                V_t, policy_t, inner_ok = self._soft_value_iteration(
                    operator, reward_matrix, num_periods=num_periods
                )
                if not inner_ok:
                    _inner_nc[0] += 1

                # Compute gradient in ascent direction (empirical - model)
                if finite_horizon:
                    expected = self._compute_expected_features_finite_horizon(
                        panel, policy_t, reward_fn, num_periods
                    )
                elif self.config.gradient_mode == "occupancy":
                    # Occupancy mode: model side uses D_π (not D_demo)
                    D_pi_t = self._compute_state_visitation(
                        policy_t, transitions, problem, initial_dist
                    )
                    if feature_matrix.ndim == 3:
                        pw = torch.einsum("sa,sak->sk", policy_t, feature_matrix)
                    else:
                        pw = feature_matrix
                    expected = D_pi_t @ pw
                else:
                    expected = self._compute_expected_features(
                        panel, policy_t, reward_fn,
                        transitions=transitions,
                        initial_dist=initial_dist,
                        discount=problem.discount_factor,
                    )

                # Ascent gradient: ∇L = empirical - expected
                # scipy minimizes → return negated value and gradient
                grad_asc = empirical_features - expected

                log_probs = operator.compute_log_choice_probabilities(reward_matrix, V_t)
                ll_val = log_probs[all_states_idx, all_actions_idx].sum().item()

                return -ll_val, (-grad_asc).numpy().astype(np.float64)

            scipy_result = optimize.minimize(
                _objective_and_gradient,
                params.numpy().astype(np.float64),
                method=self.config.optimizer,
                jac=True,
                options={
                    "maxiter": self.config.outer_max_iter,
                    "ftol": self.config.outer_tol ** 2,
                    "gtol": self.config.outer_tol,
                    "disp": False,  # suppress scipy's own output; use self._log instead
                },
            )

            final_params = torch.tensor(scipy_result.x, dtype=torch.float32)
            converged = bool(scipy_result.success)
            n_function_evals = scipy_result.nfev
            inner_not_converged = _inner_nc[0]

        else:
            # -----------------------------------------------------------
            # ADAM / SGD PATH (legacy; use_adam=True or optimizer="gradient")
            # -----------------------------------------------------------
            if self.config.use_adam:
                m = torch.zeros_like(params)  # First moment
                v = torch.zeros_like(params)  # Second moment
                optimizer_name = "Adam"
            else:
                m = v = None
                optimizer_name = "SGD"

            self._log(f"Starting MCE IRL with {optimizer_name} (lr={self.config.learning_rate})")

            best_obj = float('inf')
            best_params = params.clone()
            patience_counter = 0
            max_patience = 20
            grad_norm = float('inf')

            # Use tqdm for progress tracking
            pbar = tqdm(
                range(self.config.outer_max_iter),
                desc="MCE IRL",
                disable=not self.config.verbose,
                leave=True,
            )

            for i in pbar:
                # Forward and backward passes
                reward_matrix = reward_fn.compute(params)
                V, policy, inner_converged = self._soft_value_iteration(
                    operator, reward_matrix, num_periods=num_periods
                )
                if not inner_converged:
                    inner_not_converged += 1

                # ===========================================================
                # GRADIENT COMPUTATION — Two approaches (see module docstring)
                #
                # (A) Occupancy mode [Ziebart 2008 Eq.6, imitation library]:
                #     ∇L(θ) = Σ_s (D_demo(s) - D_π(s)) · ∂r(s)/∂θ
                #     where D_demo and D_π are state visitation frequencies.
                #     Works for ALL feature types (state-only, action-dep, neural).
                #     Key: D_π depends on θ through the policy, so gradient is
                #     non-zero even when reward features don't vary by action.
                #
                # (B) Feature matching mode [Ziebart thesis Ch.9, Gleave 2022]:
                #     ∇L(θ) = E_D[φ(S,A)] - E_π[φ(S_D,·)]
                #     where E_π uses the DEMO state distribution, not D_π(s).
                #     Only works when features vary across actions — for
                #     state-only features, E_π = E_D → gradient is always zero.
                # ===========================================================

                if finite_horizon:
                    # Finite horizon: always use feature_matching regardless of gradient_mode.
                    # Occupancy mode for finite horizon would require time-indexed state
                    # distributions D_π(s,t), which is not yet implemented.
                    expected_features = self._compute_expected_features_finite_horizon(
                        panel, policy, reward_fn, num_periods
                    )
                    gradient = empirical_features - expected_features
                    obj = 0.5 * torch.sum(gradient ** 2).item()

                elif self.config.gradient_mode == "occupancy":
                    # -----------------------------------------------------------
                    # OCCUPANCY MODE — Correct MCE IRL gradient (Ziebart 2010 Ch.9)
                    #
                    # The true gradient of log-likelihood L(θ) = Σ_{i,t} log π_θ(a|s) is:
                    #
                    #   ∇L_k = E_demo[φ_k(s,a)] - E_π[φ_k(s,a)]
                    #         = (1/N) Σ_{i,t} φ(s_{it}, a_{it}, k)          [demo side]
                    #           - Σ_s D_π(s) · Σ_a π(a|s) · φ(s,a,k)       [model side]
                    #         = empirical_features[k] - D_π @ policy_weighted_features[k]
                    #
                    # Demo side: empirical_features uses ACTUAL expert actions (fixed from data).
                    # Model side: D_π(s) from forward pass × policy-weighted features.
                    #
                    # This fixes the bug in "feature_matching" mode where the model side
                    # incorrectly used D_demo (empirical states) instead of D_π (model states).
                    # Using D_demo for the model side gives the wrong gradient for action-
                    # dependent features when D_demo ≠ D_π (Ziebart 2008 Eq. 6).
                    #
                    # For state-only φ(s):
                    #   policy_weighted[s,k] = Σ_a π(a|s) φ(s,k) = φ(s,k)   [π sums to 1]
                    #   empirical_features[k] = (1/N) Σ_{i,t} φ(s_{it},k) ≈ D_demo @ φ[k]
                    #   gradient[k] ≈ (D_demo - D_pi) @ φ[k]   ← non-zero, correct!
                    # -----------------------------------------------------------

                    # Step 1: Compute model state occupancy D_π(s) via forward pass.
                    # D_π(s) = expected discounted frequency of visiting state s under π_θ.
                    # Solved via iteration: D = ρ₀ + γ · P_π^T · D   (until convergence)
                    # Shape: (n_states,), normalized to sum to 1.
                    # Reference: Ziebart (2010) Chapter 9, forward message passing.
                    D_pi = self._compute_state_visitation(
                        policy, transitions, problem, initial_dist
                    )

                    # Step 2: Compute policy-weighted features at each state.
                    # policy_weighted_features[s, k] = Σ_a π(a|s) · φ(s, a, k)
                    # = the expected feature k at state s under the current policy.
                    # For state-action features (3D φ): sum over actions with π weights.
                    # For state-only features (2D φ): result is just φ(s,k) (π sums to 1).
                    # Shape: (n_states, n_features) in both cases.
                    if feature_matrix.ndim == 3:
                        # State-action features: φ has shape (n_states, n_actions, n_features)
                        # Σ_a π(a|s)·φ(s,a,k) → shape (n_states, n_features)
                        policy_weighted_features = torch.einsum(
                            "sa,sak->sk", policy, feature_matrix
                        )
                    else:
                        # State-only features: φ has shape (n_states, n_features)
                        # Σ_a π(a|s)·φ(s,k) = φ(s,k)·Σ_a π(a|s) = φ(s,k)  [no-op]
                        policy_weighted_features = feature_matrix

                    # Step 3: Compute expected features under the model's state distribution.
                    # model_features[k] = Σ_s D_π(s) · policy_weighted_features[s, k]
                    # = E_π[φ_k]  weighted by model's state visitation
                    # Shape: (n_features,)
                    model_features = D_pi @ policy_weighted_features

                    # Step 4: Gradient in ascent direction = empirical - model.
                    # gradient[k] = E_demo[φ_k] - E_π[φ_k]
                    # When expert features > model features for component k: increase θ_k.
                    # When model features > expert features for component k: decrease θ_k.
                    # At the optimum, gradient → 0 (feature matching condition).
                    gradient = empirical_features - model_features

                    # Objective for tracking convergence: squared gradient norm.
                    # At optimum, E_demo[φ] ≈ E_π[φ], so gradient → 0 and obj → 0.
                    obj = 0.5 * torch.sum(gradient ** 2).item()

                else:
                    # -----------------------------------------------------------
                    # FEATURE MATCHING MODE (legacy)
                    # Implements: ∇L(θ) = μ_D - μ_π  (feature expectations)
                    # where μ_π uses the empirical state distribution from demos.
                    # -----------------------------------------------------------
                    expected_features = self._compute_expected_features(
                        panel, policy, reward_fn,
                        transitions=transitions,
                        initial_dist=initial_dist,
                        discount=problem.discount_factor,
                    )
                    # Feature matching gradient: μ_D - μ_π (ascent direction)
                    # Increase θ_k where expert features exceed model's expected features.
                    gradient = empirical_features - expected_features
                    obj = 0.5 * torch.sum(gradient ** 2).item()

                # Gradient clipping (prevents divergence in early iterations when
                # rewards are far from optimal and value function hasn't converged)
                grad_norm = torch.norm(gradient).item()
                if self.config.gradient_clip > 0 and grad_norm > self.config.gradient_clip:
                    gradient = gradient * (self.config.gradient_clip / grad_norm)
                    grad_norm = self.config.gradient_clip

                # Track best
                if obj < best_obj:
                    best_obj = obj
                    best_params = params.clone()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Update progress bar with metrics
                postfix = {
                    "obj": f"{obj:.6f}",
                    "||grad||": f"{grad_norm:.4f}",
                }
                if true_params is not None:
                    rmse = torch.sqrt(torch.mean((params - true_params) ** 2)).item()
                    postfix["RMSE"] = f"{rmse:.6f}"
                pbar.set_postfix(postfix)

                # Check convergence
                if grad_norm < self.config.outer_tol:
                    converged = True
                    break

                if patience_counter > max_patience:
                    pbar.set_description("MCE IRL (early stop)")
                    break

                # Gradient step with Adam or SGD
                if self.config.use_adam:
                    # Adam update (Kingma & Ba, 2014)
                    t = i + 1  # Timestep (1-indexed)
                    m = self.config.adam_beta1 * m + (1 - self.config.adam_beta1) * gradient
                    v = self.config.adam_beta2 * v + (1 - self.config.adam_beta2) * (gradient ** 2)
                    # Bias correction
                    m_hat = m / (1 - self.config.adam_beta1 ** t)
                    v_hat = v / (1 - self.config.adam_beta2 ** t)
                    # Update
                    params = params + self.config.learning_rate * m_hat / (torch.sqrt(v_hat) + self.config.adam_eps)
                else:
                    # Simple SGD
                    params = params + self.config.learning_rate * gradient

                n_function_evals += 1

            pbar.close()
            final_params = best_params
            converged = grad_norm < self.config.outer_tol if grad_norm else False

        # Final solution
        reward_matrix = reward_fn.compute(final_params)
        V, policy, _ = self._soft_value_iteration(
            operator, reward_matrix, num_periods=num_periods
        )

        if finite_horizon:
            final_expected = self._compute_expected_features_finite_horizon(
                panel, policy, reward_fn, num_periods
            )
            D = torch.zeros(problem.num_states)

            # Finite-horizon LL: use period-specific policies
            sigma = problem.scale_parameter
            import torch.nn.functional as Func
            fh_result = backward_induction(operator, reward_matrix, num_periods)
            log_policy = Func.log_softmax(fh_result.Q / sigma, dim=-1)  # (T, S, A)
            ll = 0.0
            for traj in panel.trajectories:
                for t in range(len(traj.states)):
                    period = min(t, num_periods - 1)
                    s = traj.states[t].item()
                    a = traj.actions[t].item()
                    ll += log_policy[period, s, a].item()

            # Flatten to period-0 for EstimationResult (base class expects 2D)
            V = V[0] if V.dim() > 1 else V
            policy = policy[0] if policy.dim() > 2 else policy
            feature_diff = torch.norm(empirical_features - final_expected).item()
        elif self.config.gradient_mode == "occupancy":
            # Occupancy mode: compute final feature difference with correct MCE IRL formula.
            # Demo side = empirical_features (actual expert actions).
            # Model side = D_π @ policy_weighted_features (model's state distribution).
            D = self._compute_state_visitation(policy, transitions, problem, initial_dist)

            # policy_weighted_features[s,k] = Σ_a π(a|s) φ(s,a,k)
            # Shape: (n_states, n_features)
            if feature_matrix.ndim == 3:
                _policy_wt_final = torch.einsum("sa,sak->sk", policy, feature_matrix)
            else:
                _policy_wt_final = feature_matrix

            # Model's expected features: Σ_s D_π(s) · policy_weighted_features[s]
            # Shape: (n_features,)
            final_expected = D @ _policy_wt_final
            # Feature difference: same formula as gradient = empirical - model
            feature_diff = torch.norm(empirical_features - final_expected).item()

            log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)
            ll = log_probs[panel.get_all_states(), panel.get_all_actions()].sum().item()
        else:
            D = self._compute_state_visitation(policy, transitions, problem, initial_dist)
            final_expected = self._compute_expected_features(panel, policy, reward_fn)
            feature_diff = torch.norm(empirical_features - final_expected).item()
            log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)
            ll = log_probs[panel.get_all_states(), panel.get_all_actions()].sum().item()

        # Inference
        hessian = None
        gradient_contributions = None
        standard_errors = None

        if self.config.compute_se:
            self._log(f"Computing standard errors via {self.config.se_method}")

            if self.config.se_method == "bootstrap":
                standard_errors = self._bootstrap_inference(
                    panel, reward_fn, problem, transitions, final_params, initial_dist
                )
            else:
                # Numerical Hessian
                hessian = self._numerical_hessian(
                    final_params, panel, reward_fn, problem, transitions, initial_dist
                )

        optimization_time = time.time() - start_time

        self._log(f"Optimization complete: feature_diff={feature_diff:.6f}, LL={ll:.2f}")
        if inner_not_converged > 0:
            self._log(f"Warning: Inner loop did not converge {inner_not_converged} times")

        return EstimationResult(
            parameters=final_params,
            log_likelihood=ll,
            value_function=V,
            policy=policy,
            hessian=hessian,
            gradient_contributions=gradient_contributions,
            converged=converged,
            num_iterations=n_function_evals,
            num_function_evals=n_function_evals,
            num_inner_iterations=0,
            message="",
            optimization_time=optimization_time,
            metadata={
                "optimizer": self.config.optimizer,
                "empirical_features": empirical_features.tolist(),
                "final_expected_features": final_expected.tolist(),
                "feature_difference": feature_diff,
                "state_visitation": D.tolist(),
                "inner_not_converged": inner_not_converged,
                "standard_errors": standard_errors.tolist() if standard_errors is not None else None,
            },
        )

    def _bootstrap_inference(
        self,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        point_estimate: torch.Tensor,
        initial_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Compute standard errors via bootstrap.

        Resamples trajectories and re-estimates parameters to get
        sampling distribution.
        """
        n_params = len(point_estimate)
        bootstrap_estimates = torch.zeros((self.config.n_bootstrap, n_params))

        trajectories = panel.trajectories
        n_traj = len(trajectories)

        operator = SoftBellmanOperator(problem, transitions)

        for b in range(self.config.n_bootstrap):
            # Resample trajectories with replacement
            indices = np.random.choice(n_traj, size=n_traj, replace=True)
            boot_trajectories = [trajectories[i] for i in indices]
            boot_panel = Panel(trajectories=boot_trajectories)

            # Compute empirical features for bootstrap sample
            empirical_features = self._compute_empirical_features(boot_panel, reward_fn)
            boot_initial = self._compute_initial_distribution(boot_panel, problem.num_states)

            # Quick optimization from point estimate
            params = point_estimate.clone()

            for _ in range(50):  # Fewer iterations for bootstrap
                reward_matrix = reward_fn.compute(params)
                V, policy, _ = self._soft_value_iteration(operator, reward_matrix)
                expected_features = self._compute_expected_features(boot_panel, policy, reward_fn)

                gradient = expected_features - empirical_features
                params = params - 0.1 * gradient

                if torch.norm(gradient) < 0.01:
                    break

            bootstrap_estimates[b] = params

            if self.config.verbose and (b + 1) % 20 == 0:
                self._log(f"Bootstrap {b + 1}/{self.config.n_bootstrap}")

        # Standard errors = std of bootstrap estimates
        standard_errors = bootstrap_estimates.std(dim=0)

        return standard_errors

    def _numerical_hessian(
        self,
        params: torch.Tensor,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_dist: torch.Tensor,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Compute numerical Hessian of the log-likelihood.

        Uses central differences with adaptive step size for numerical stability.
        The Hessian at a maximum should be negative semi-definite, so we ensure
        the returned Hessian has this property by projecting onto the negative
        semi-definite cone if needed.

        Parameters
        ----------
        params : torch.Tensor
            Parameter vector at which to compute Hessian.
        panel : Panel
            Panel data for computing log-likelihood.
        reward_fn : LinearReward
            Reward function specification.
        problem : DDCProblem
            Problem specification.
        transitions : torch.Tensor
            Transition matrices.
        initial_dist : torch.Tensor
            Initial state distribution (unused but kept for API consistency).
        eps : float
            Step size for finite differences. Default 1e-3 for stability.

        Returns
        -------
        torch.Tensor
            Hessian matrix, shape (n_params, n_params).
            Guaranteed to be negative semi-definite for valid inference.

        Notes
        -----
        Step size selection follows Gill, Murray, and Wright (1981) guidance:
        - Use adaptive step h_i = max(eps, min(|params[i]| * eps, 0.1))
        - Lower bound (eps) prevents division by zero for zero-valued parameters
        - Upper bound (0.1) prevents excessively large steps for large parameters
          that would introduce truncation error and numerical instability
        - The default eps=1e-3 balances truncation error (favors larger h)
          against rounding error (favors smaller h) for float32 precision
        """
        operator = SoftBellmanOperator(problem, transitions)
        n_params = len(params)
        hessian = torch.zeros((n_params, n_params), dtype=params.dtype)

        def ll_at(p):
            reward_matrix = reward_fn.compute(p)
            V, policy, _ = self._soft_value_iteration(operator, reward_matrix)
            log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)

            ll = log_probs[panel.get_all_states(), panel.get_all_actions()].sum().item()
            return ll

        # Compute Hessian using central differences
        # Use adaptive step size based on parameter magnitude with bounds
        for i in range(n_params):
            # Adaptive step: larger for larger params, bounded between eps and 0.1
            h_i = max(eps, min(abs(params[i].item()) * eps, 0.1))

            for j in range(i, n_params):
                h_j = max(eps, min(abs(params[j].item()) * eps, 0.1))

                if i == j:
                    # Diagonal: use standard 2nd derivative formula
                    # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
                    p_plus = params.clone()
                    p_plus[i] += h_i
                    p_minus = params.clone()
                    p_minus[i] -= h_i

                    ll_plus = ll_at(p_plus)
                    ll_0 = ll_at(params)
                    ll_minus = ll_at(p_minus)

                    h_ii = (ll_plus - 2 * ll_0 + ll_minus) / (h_i * h_i)
                    hessian[i, i] = h_ii
                else:
                    # Off-diagonal: use 4-point formula for mixed partial
                    p_pp = params.clone()
                    p_pp[i] += h_i
                    p_pp[j] += h_j

                    p_pm = params.clone()
                    p_pm[i] += h_i
                    p_pm[j] -= h_j

                    p_mp = params.clone()
                    p_mp[i] -= h_i
                    p_mp[j] += h_j

                    p_mm = params.clone()
                    p_mm[i] -= h_i
                    p_mm[j] -= h_j

                    h_ij = (ll_at(p_pp) - ll_at(p_pm) - ll_at(p_mp) + ll_at(p_mm)) / (4 * h_i * h_j)
                    hessian[i, j] = h_ij
                    hessian[j, i] = h_ij

        # Ensure Hessian is negative semi-definite (required at a maximum)
        # If not, project onto negative semi-definite cone
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        if (eigenvalues > 0).any():
            # Some eigenvalues are positive - not at a proper maximum
            # Project to negative semi-definite by clamping positive eigenvalues
            self._log("Warning: Hessian not negative semi-definite, projecting")
            eigenvalues_clamped = torch.clamp(eigenvalues, max=-1e-8)
            hessian = eigenvectors @ torch.diag(eigenvalues_clamped) @ eigenvectors.T

        return hessian
