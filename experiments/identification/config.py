"""Central configuration for identification experiments.

All hyperparameters live here so individual scripts stay clean.
The DGP mimics the serialized content consumption setting from
Lee, Sudhir & Wang (2026): episodes, buy/wait/exit actions,
deterministic transitions, anchor identification.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DGPConfig:
    """Data generating process specification.

    The environment has S regular episode states plus one absorbing state.
    Three actions: buy (advance), wait (stay), exit (absorb).
    Reward is action-dependent with exit anchored to zero.
    """

    num_episodes: int = 20
    discount_factor: float = 0.95
    scale_parameter: float = 1.0

    # True reward parameters (5 free parameters)
    # r(s, buy)  = alpha_buy + theta_e * quality(s) + theta_w * w(s)
    # r(s, wait) = theta_wait_base + theta_w_wait * w(s)
    # r(s, exit) = 0 (anchor)
    alpha_buy: float = -1.5
    theta_e: float = 2.0
    theta_w: float = -0.8
    theta_wait_base: float = -0.3
    theta_w_wait: float = -0.5

    @property
    def num_states(self) -> int:
        return self.num_episodes + 1  # +1 for absorbing state

    @property
    def num_actions(self) -> int:
        return 3  # buy=0, wait=1, exit=2

    @property
    def absorbing_state(self) -> int:
        return self.num_episodes  # last index

    @property
    def exit_action(self) -> int:
        return 2


@dataclass(frozen=True)
class EstimationConfig:
    """Hyperparameters for estimation methods."""

    # Data generation
    n_individuals: int = 500
    n_periods: int = 80
    seed: int = 42

    # AIRL
    airl_max_rounds: int = 300
    airl_reward_lr: float = 0.01
    airl_disc_steps: int = 5
    airl_convergence_tol: float = 1e-4

    # IQ-Learn
    iq_max_iter: int = 5000
    iq_learning_rate: float = 0.005
    iq_alpha: float = 1.0

    # Reduced-form Q
    rf_max_iter: int = 2000
    rf_learning_rate: float = 0.01


@dataclass(frozen=True)
class CounterfactualConfig:
    """Counterfactual experiment specifications."""

    # Type I: shift the wait-cost feature w(s) by this amount
    type_i_wait_shift: float = -0.3

    # Type II: buy advances by k episodes instead of 1
    type_ii_skip_values: tuple[int, ...] = (1, 2, 3, 5, 7)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    dgp: DGPConfig = field(default_factory=DGPConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    counterfactual: CounterfactualConfig = field(default_factory=CounterfactualConfig)
