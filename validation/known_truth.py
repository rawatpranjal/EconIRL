"""Known-truth synthetic validation harness.

This module is experiment infrastructure, deliberately kept out of the
public package API. It defines one adaptable DGP, exact truth objects,
pre-estimation checks, estimator contracts, hard recovery gates, and a
small CLI for oracle and estimator runs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.occupancy import (
    compute_state_action_visitation,
    compute_state_visitation,
)
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.environments.shapeshifter import (
    ShapeshifterConfig,
    ShapeshifterEnvironment,
)
from econirl.inference.results import EstimationSummary
from econirl.preferences.action_reward import ActionDependentReward

# --- Configuration ---
StateMode = Literal["low_dim", "high_dim"]
RewardMode = Literal["state_only", "action_dependent", "neural"]
RewardDim = Literal["low", "high"]
HeterogeneityMode = Literal["none", "latent_segments"]
InitialStateMode = Literal["start", "uniform_regular"]
CounterfactualKind = Literal["type_a", "type_b", "type_c"]


@dataclass(frozen=True)
class KnownTruthDGPConfig:
    """Specification for the configurable known-truth DGP.

    The state space is represented by discrete indices so exact Bellman
    solutions remain available. High-dimensional modes expose richer state
    encodings and reward features on top of that common grid.
    """

    state_mode: StateMode = "low_dim"
    reward_mode: RewardMode = "action_dependent"
    reward_dim: RewardDim = "low"
    heterogeneity: HeterogeneityMode = "none"
    num_regular_states: int = 20
    num_actions: int = 3
    high_state_dim: int = 12
    high_reward_features: int = 24
    num_segments: int = 2
    discount_factor: float = 0.95
    scale_parameter: float = 1.0
    seed: int = 42
    initial_state_mode: InitialStateMode = "uniform_regular"
    exit_action: int = 2
    transition_noise: float = 0.05
    feature_scale: float = 1.0

    @property
    def num_states(self) -> int:
        return self.num_regular_states + 1

    @property
    def absorbing_state(self) -> int:
        return self.num_regular_states

    @property
    def uses_exit_anchor(self) -> bool:
        return 0 <= self.exit_action < self.num_actions

    def validate(self) -> None:
        if self.num_regular_states < 3:
            raise ValueError("num_regular_states must be at least 3")
        if self.num_actions < 2:
            raise ValueError("num_actions must be at least 2")
        if not 0 <= self.discount_factor < 1:
            raise ValueError("discount_factor must be in [0, 1)")
        if self.scale_parameter <= 0:
            raise ValueError("scale_parameter must be positive")
        if self.heterogeneity == "latent_segments" and self.num_segments < 2:
            raise ValueError("latent_segments requires num_segments >= 2")
        if self.reward_dim == "high" and self.high_reward_features < 8:
            raise ValueError("high_reward_features must be at least 8")
        if self.state_mode == "high_dim" and self.high_state_dim < 4:
            raise ValueError("high_state_dim must be at least 4")
        if self.uses_exit_anchor and self.exit_action >= self.num_actions:
            raise ValueError("exit_action must be a valid action index")
        if not 0 <= self.transition_noise < 1:
            raise ValueError("transition_noise must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShapeshifterKnownTruthConfig:
    """Known-truth harness adapter for the Shapeshifter flexible DGP.

    Shapeshifter has no absorbing state or exit action. The bridge exposes the
    small set of fields the shared harness needs while preserving the native
    Shapeshifter configuration as the source of truth.
    """

    env_config: ShapeshifterConfig
    state_mode: StateMode = "low_dim"
    heterogeneity: HeterogeneityMode = "none"

    @property
    def reward_mode(self) -> RewardMode:
        if self.env_config.reward_type == "neural":
            return "neural"
        if self.env_config.action_dependent:
            return "action_dependent"
        return "state_only"

    @property
    def reward_dim(self) -> RewardDim:
        if self.env_config.feature_type == "neural" or self.env_config.num_features > 8:
            return "high"
        return "low"

    @property
    def num_states(self) -> int:
        return self.env_config.total_states

    @property
    def num_actions(self) -> int:
        return self.env_config.num_actions

    @property
    def num_regular_states(self) -> int:
        return self.env_config.total_states

    @property
    def absorbing_state(self) -> None:
        return None

    @property
    def exit_action(self) -> None:
        return None

    @property
    def uses_exit_anchor(self) -> bool:
        return False

    @property
    def num_segments(self) -> int:
        return 1

    @property
    def seed(self) -> int:
        return self.env_config.seed

    def validate(self) -> None:
        if self.env_config.total_states < 2:
            raise ValueError("Shapeshifter total_states must be at least 2")
        if self.env_config.num_actions < 2:
            raise ValueError("Shapeshifter num_actions must be at least 2")
        if self.heterogeneity != "none":
            raise ValueError("Shapeshifter known-truth bridge is homogeneous")
        if not 0 <= self.env_config.discount_factor < 1:
            raise ValueError("discount_factor must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "shapeshifter",
            "env_config": asdict(self.env_config),
            "state_mode": self.state_mode,
            "reward_mode": self.reward_mode,
            "reward_dim": self.reward_dim,
            "heterogeneity": self.heterogeneity,
        }


@dataclass(frozen=True)
class ContentHeterogeneityKnownTruthConfig:
    """Paper-style serialized-content DGP for AIRL-Het validation.

    The generic high-dimensional grid is useful as a stress test, but it does
    not encode the core structure of the AIRL-Het paper setting: repeated
    series for each user, latent segment membership that is constant across
    those series, a pay/wait/exit choice set, high-dimensional observed
    content controls, and an exit-action reward anchor.
    """

    num_chapters: int = 5
    wait_bins: int = 3
    price_levels: int = 2
    quality_levels: int = 2
    num_segments: int = 2
    discount_factor: float = 0.92
    scale_parameter: float = 0.85
    seed: int = 4506
    exit_action: int = 2
    transition_noise: float = 0.0
    books_per_user: int = 4
    segment_probabilities: tuple[float, ...] = (0.48, 0.52)
    high_state_dim: int = 18
    high_reward_features: int = 20

    @property
    def state_mode(self) -> StateMode:
        return "high_dim"

    @property
    def reward_mode(self) -> RewardMode:
        return "action_dependent"

    @property
    def reward_dim(self) -> RewardDim:
        return "high"

    @property
    def heterogeneity(self) -> HeterogeneityMode:
        return "latent_segments"

    @property
    def num_actions(self) -> int:
        return 3

    @property
    def num_regular_states(self) -> int:
        return (
            self.num_chapters
            * self.wait_bins
            * self.price_levels
            * self.quality_levels
        )

    @property
    def num_states(self) -> int:
        return self.num_regular_states + 1

    @property
    def absorbing_state(self) -> int:
        return self.num_regular_states

    @property
    def uses_exit_anchor(self) -> bool:
        return True

    def validate(self) -> None:
        if self.num_chapters < 3:
            raise ValueError("num_chapters must be at least 3")
        if self.wait_bins < 2:
            raise ValueError("wait_bins must be at least 2")
        if self.price_levels < 2 or self.quality_levels < 2:
            raise ValueError("price_levels and quality_levels must be at least 2")
        if self.num_segments != 2:
            raise ValueError("content AIRL-Het DGP is calibrated for two segments")
        if not 0 <= self.discount_factor < 1:
            raise ValueError("discount_factor must be in [0, 1)")
        if self.scale_parameter <= 0:
            raise ValueError("scale_parameter must be positive")
        if self.exit_action != 2:
            raise ValueError("content AIRL-Het DGP expects action 2 to be exit")
        if not 0 <= self.transition_noise < 1:
            raise ValueError("transition_noise must be in [0, 1)")
        if self.books_per_user < 1:
            raise ValueError("books_per_user must be positive")
        if len(self.segment_probabilities) != self.num_segments:
            raise ValueError("segment_probabilities must match num_segments")
        if any(weight < 0 for weight in self.segment_probabilities):
            raise ValueError("segment_probabilities must be nonnegative")
        if sum(self.segment_probabilities) <= 0:
            raise ValueError("segment_probabilities must have positive mass")
        if self.high_state_dim < 7:
            raise ValueError("high_state_dim must be at least 7")
        if self.high_reward_features < 11:
            raise ValueError("high_reward_features must be at least 11")
        if self.high_reward_features > 32:
            raise ValueError("high_reward_features must be at most 32")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = "content_heterogeneity"
        payload["state_mode"] = self.state_mode
        payload["reward_mode"] = self.reward_mode
        payload["reward_dim"] = self.reward_dim
        payload["heterogeneity"] = self.heterogeneity
        payload["num_regular_states"] = self.num_regular_states
        payload["num_states"] = self.num_states
        payload["num_actions"] = self.num_actions
        payload["absorbing_state"] = self.absorbing_state
        return payload


@dataclass(frozen=True)
class SimulationConfig:
    """Panel simulation controls."""

    n_individuals: int = 500
    n_periods: int = 80
    seed: int = 42
    show_progress: bool = False


@dataclass(frozen=True)
class CounterfactualConfig:
    """Default oracle counterfactual controls."""

    type_a_shift: float = -0.25
    type_b_skip: int = 2
    type_c_action: int = 1
    type_c_penalty: float = -1_000.0


# --- DGP and Simulation ---
@dataclass(frozen=True)
class KnownTruthDGP:
    """A fully specified synthetic DGP with all truth objects exposed."""

    config: (
        KnownTruthDGPConfig
        | ShapeshifterKnownTruthConfig
        | ContentHeterogeneityKnownTruthConfig
    )
    problem: DDCProblem
    transitions: jnp.ndarray
    feature_matrix: jnp.ndarray
    state_features: jnp.ndarray
    parameter_names: list[str]
    true_parameters: jnp.ndarray
    reward_matrix: jnp.ndarray
    initial_distribution: jnp.ndarray
    segment_probabilities: jnp.ndarray | None = None

    @property
    def num_segments(self) -> int:
        if self.true_parameters.ndim == 1:
            return 1
        return int(self.true_parameters.shape[0])

    @property
    def homogeneous_parameters(self) -> jnp.ndarray:
        if self.true_parameters.ndim == 1:
            return self.true_parameters
        weights = self.segment_probabilities
        if weights is None:
            weights = jnp.ones(self.num_segments) / self.num_segments
        return jnp.einsum("g,gk->k", weights, self.true_parameters)

    @property
    def homogeneous_reward(self) -> jnp.ndarray:
        if self.reward_matrix.ndim == 2:
            return self.reward_matrix
        weights = self.segment_probabilities
        if weights is None:
            weights = jnp.ones(self.num_segments) / self.num_segments
        return jnp.einsum("g,gsa->sa", weights, self.reward_matrix)

    def utility(self) -> ActionDependentReward:
        num_features = int(self.feature_matrix.shape[-1])
        if len(self.parameter_names) == num_features:
            names = self.parameter_names
        else:
            names = [f"feature_{k}" for k in range(num_features)]
        return ActionDependentReward(self.feature_matrix, names)

    def metadata(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "parameter_names": self.parameter_names,
            "true_parameters": np.asarray(self.true_parameters).tolist(),
            "segment_probabilities": (
                None
                if self.segment_probabilities is None
                else np.asarray(self.segment_probabilities).tolist()
            ),
        }


def build_known_truth_dgp(
    config: (
        KnownTruthDGPConfig
        | ShapeshifterKnownTruthConfig
        | ContentHeterogeneityKnownTruthConfig
        | None
    ) = None,
) -> KnownTruthDGP:
    """Build a known-truth DGP from a config."""

    if config is None:
        config = KnownTruthDGPConfig()
    if isinstance(config, ShapeshifterKnownTruthConfig):
        return build_shapeshifter_known_truth_dgp(config)
    if isinstance(config, ContentHeterogeneityKnownTruthConfig):
        return build_content_heterogeneity_known_truth_dgp(config)
    config.validate()

    transitions = _build_transitions(config)
    state_features = _build_state_features(config)
    feature_matrix, parameter_names = _build_reward_features(config, state_features)
    true_parameters = _build_parameters(config, len(parameter_names))
    reward_matrix = _compute_rewards(feature_matrix, true_parameters)
    initial_distribution = _build_initial_distribution(config)

    problem = DDCProblem(
        num_states=config.num_states,
        num_actions=config.num_actions,
        discount_factor=config.discount_factor,
        scale_parameter=config.scale_parameter,
        state_dim=int(state_features.shape[1]),
        state_encoder=lambda states: state_features[states],
    )

    segment_probabilities = None
    if config.heterogeneity == "latent_segments":
        segment_probabilities = jnp.ones(config.num_segments, dtype=jnp.float32)
        segment_probabilities = segment_probabilities / segment_probabilities.sum()

    return KnownTruthDGP(
        config=config,
        problem=problem,
        transitions=transitions,
        feature_matrix=feature_matrix,
        state_features=state_features,
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        reward_matrix=reward_matrix,
        initial_distribution=initial_distribution,
        segment_probabilities=segment_probabilities,
    )


def build_shapeshifter_known_truth_dgp(
    config: ShapeshifterKnownTruthConfig,
) -> KnownTruthDGP:
    """Build a known-truth DGP from the Shapeshifter flexible environment."""

    config.validate()
    env = ShapeshifterEnvironment(config.env_config)
    state_indices = jnp.arange(env.num_states, dtype=jnp.int32)
    state_features = env.encode_states(state_indices)

    if config.env_config.reward_type == "linear":
        parameter_names = env.parameter_names
        true_parameters = env.get_true_parameter_vector()
    else:
        parameter_names = []
        true_parameters = jnp.asarray([], dtype=jnp.float32)

    return KnownTruthDGP(
        config=config,
        problem=env.problem_spec,
        transitions=jnp.asarray(env.transition_matrices, dtype=jnp.float32),
        feature_matrix=jnp.asarray(env.feature_matrix, dtype=jnp.float32),
        state_features=jnp.asarray(state_features, dtype=jnp.float32),
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        reward_matrix=jnp.asarray(env.true_reward_matrix, dtype=jnp.float32),
        initial_distribution=jnp.asarray(
            env._get_initial_state_distribution(), dtype=jnp.float32
        ),
        segment_probabilities=None,
    )


def build_content_heterogeneity_known_truth_dgp(
    config: ContentHeterogeneityKnownTruthConfig,
) -> KnownTruthDGP:
    """Build the serialized-content latent-segment AIRL-Het DGP."""

    config.validate()
    transitions = _build_content_transitions(config)
    state_features = _build_content_state_features(config)
    feature_matrix, parameter_names = _build_content_reward_features(
        config, state_features
    )
    true_parameters = _build_content_parameters(config, parameter_names)
    reward_matrix = _compute_rewards(feature_matrix, true_parameters)
    initial_distribution = _build_content_initial_distribution(config)

    problem = DDCProblem(
        num_states=config.num_states,
        num_actions=config.num_actions,
        discount_factor=config.discount_factor,
        scale_parameter=config.scale_parameter,
        state_dim=int(state_features.shape[1]),
        state_encoder=lambda states: state_features[states],
    )

    segment_probabilities = np.asarray(config.segment_probabilities, dtype=np.float64)
    segment_probabilities = segment_probabilities / segment_probabilities.sum()

    return KnownTruthDGP(
        config=config,
        problem=problem,
        transitions=transitions,
        feature_matrix=feature_matrix,
        state_features=state_features,
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        reward_matrix=reward_matrix,
        initial_distribution=initial_distribution,
        segment_probabilities=jnp.asarray(segment_probabilities, dtype=jnp.float32),
    )


def _content_state_index(
    config: ContentHeterogeneityKnownTruthConfig,
    chapter: int,
    wait_bin: int,
    price_level: int,
    quality_level: int,
) -> int:
    return (
        (((chapter * config.wait_bins) + wait_bin) * config.price_levels + price_level)
        * config.quality_levels
        + quality_level
    )


def _content_state_tuple(
    config: ContentHeterogeneityKnownTruthConfig,
    state: int,
) -> tuple[int, int, int, int]:
    quality = state % config.quality_levels
    state //= config.quality_levels
    price = state % config.price_levels
    state //= config.price_levels
    wait_bin = state % config.wait_bins
    chapter = state // config.wait_bins
    return chapter, wait_bin, price, quality


def _build_content_initial_distribution(
    config: ContentHeterogeneityKnownTruthConfig,
) -> jnp.ndarray:
    dist = np.zeros(config.num_states, dtype=np.float64)
    for price in range(config.price_levels):
        for quality in range(config.quality_levels):
            state = _content_state_index(config, 0, 0, price, quality)
            dist[state] = 1.0
    dist = dist / dist.sum()
    return jnp.asarray(dist, dtype=jnp.float32)


def _build_content_transitions(
    config: ContentHeterogeneityKnownTruthConfig,
) -> jnp.ndarray:
    transitions = np.zeros(
        (config.num_actions, config.num_states, config.num_states),
        dtype=np.float64,
    )
    absorbing = config.absorbing_state
    last_chapter = config.num_chapters - 1
    last_wait = config.wait_bins - 1

    for state in range(config.num_regular_states):
        chapter, wait_bin, price, quality = _content_state_tuple(config, state)

        if chapter == last_chapter:
            read_target = absorbing
            wait_target = absorbing if wait_bin == last_wait else _content_state_index(
                config, chapter, wait_bin + 1, price, quality
            )
        else:
            read_target = _content_state_index(config, chapter + 1, 0, price, quality)
            if wait_bin == last_wait:
                wait_target = _content_state_index(
                    config, chapter + 1, 0, price, quality
                )
            else:
                wait_target = _content_state_index(
                    config, chapter, wait_bin + 1, price, quality
                )

        targets = {0: read_target, 1: wait_target, config.exit_action: absorbing}
        for action, target in targets.items():
            transitions[action, state, target] += 1.0 - config.transition_noise
            if config.transition_noise > 0 and action != config.exit_action:
                fallback = _content_state_index(config, chapter, 0, price, quality)
                transitions[action, state, fallback] += config.transition_noise

    transitions[:, absorbing, absorbing] = 1.0
    transitions = transitions / transitions.sum(axis=2, keepdims=True)
    return jnp.asarray(transitions, dtype=jnp.float32)


def _build_content_state_features(
    config: ContentHeterogeneityKnownTruthConfig,
) -> jnp.ndarray:
    regular_rows: list[list[float]] = []
    denom_chapter = max(config.num_chapters - 1, 1)
    denom_wait = max(config.wait_bins - 1, 1)
    denom_price = max(config.price_levels - 1, 1)
    denom_quality = max(config.quality_levels - 1, 1)

    for state in range(config.num_regular_states):
        chapter, wait_bin, price, quality = _content_state_tuple(config, state)
        progress = chapter / denom_chapter
        wait_norm = wait_bin / denom_wait
        price_norm = price / denom_price
        quality_norm = quality / denom_quality
        cliffhanger = (
            0.45 * quality_norm
            + 0.35 * np.sin(np.pi * (chapter + 1) / config.num_chapters)
            + 0.20 * (1.0 if chapter in {2, config.num_chapters - 2} else 0.0)
        )
        final_chapter = 1.0 if chapter == config.num_chapters - 1 else 0.0
        unlock_ready = 1.0 if wait_bin == config.wait_bins - 1 else 0.0
        regular_rows.append(
            [
                progress,
                wait_norm,
                price_norm,
                quality_norm,
                cliffhanger,
                final_chapter,
                unlock_ready,
            ]
        )

    regular = np.asarray(regular_rows, dtype=np.float64)
    progress = regular[:, 0]
    wait_norm = regular[:, 1]
    price = regular[:, 2]
    quality = regular[:, 3]
    cliffhanger = regular[:, 4]
    unlock_ready = regular[:, 6]
    extras = [
        progress * quality,
        progress * price,
        wait_norm * price,
        wait_norm * quality,
        cliffhanger * price,
        cliffhanger * wait_norm,
        unlock_ready * price,
        np.sin(np.pi * progress),
        np.cos(np.pi * progress),
        np.sin(2.0 * np.pi * progress),
        np.cos(2.0 * np.pi * progress),
    ]
    for extra in extras:
        if regular.shape[1] >= config.high_state_dim:
            break
        regular = np.column_stack([regular, _standardize_feature(extra)])

    while regular.shape[1] < config.high_state_dim:
        idx = regular.shape[1] - 6
        wave = np.sin((idx + 1) * np.pi * progress + 0.3 * idx * price)
        regular = np.column_stack([regular, _standardize_feature(wave)])

    absorbing = np.zeros((1, regular.shape[1]), dtype=np.float64)
    features = np.vstack([regular, absorbing])
    return jnp.asarray(features, dtype=jnp.float32)


def _standardize_feature(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean()
    scale = values.std()
    if scale < 1e-8:
        return centered
    return centered / scale


def _build_content_reward_features(
    config: ContentHeterogeneityKnownTruthConfig,
    state_features: jnp.ndarray,
) -> tuple[jnp.ndarray, list[str]]:
    sf = np.asarray(state_features, dtype=np.float64)
    progress = sf[:, 0]
    wait_norm = sf[:, 1]
    price = sf[:, 2]
    quality = sf[:, 3]
    cliffhanger = sf[:, 4]
    final_chapter = sf[:, 5]
    unlock_ready = sf[:, 6]
    episode_wave = (
        np.sin(np.pi * progress)
        if sf.shape[1] <= 8
        else sf[:, 14]
    )
    late_unlock = final_chapter * unlock_ready

    read_candidates = [
        ("read_intercept", np.ones(config.num_states)),
        ("read_quality", quality),
        ("read_cliffhanger", cliffhanger),
        ("read_progress", progress),
        ("read_price", price),
        ("read_unlock_ready", unlock_ready),
        ("read_price_quality", price * quality),
        ("read_progress_quality", progress * quality),
        ("read_price_cliffhanger", price * cliffhanger),
        ("read_wait_delay", wait_norm),
        ("read_final_chapter", final_chapter),
        ("read_episode_wave", episode_wave),
        ("read_unlock_price", unlock_ready * price),
        ("read_late_unlock", late_unlock),
        ("read_progress_price", progress * price),
        ("read_wait_cliffhanger", wait_norm * cliffhanger),
    ]
    wait_candidates = [
        ("wait_intercept", np.ones(config.num_states)),
        ("wait_quality", quality),
        ("wait_delay", wait_norm),
        ("wait_unlock_ready", unlock_ready),
        ("wait_progress", progress),
        ("wait_price", price),
        ("wait_price_quality", price * quality),
        ("wait_cliffhanger", cliffhanger),
        ("wait_progress_quality", progress * quality),
        ("wait_final_chapter", final_chapter),
        ("wait_price_delay", price * wait_norm),
        ("wait_cliffhanger_delay", cliffhanger * wait_norm),
        ("wait_unlock_price", unlock_ready * price),
        ("wait_late_unlock", late_unlock),
        ("wait_episode_wave", episode_wave),
        ("wait_progress_price", progress * price),
    ]
    target_read = min(
        len(read_candidates),
        max(6, config.high_reward_features // 2),
    )
    target_wait = min(
        len(wait_candidates),
        max(5, config.high_reward_features - target_read),
    )
    selected_read = read_candidates[:target_read]
    selected_wait = wait_candidates[:target_wait]
    names = [name for name, _ in selected_read] + [
        name for name, _ in selected_wait
    ]
    features = np.zeros(
        (config.num_states, config.num_actions, len(names)),
        dtype=np.float64,
    )
    regular = slice(0, config.num_regular_states)
    for col, (_, values) in enumerate(selected_read):
        values = np.asarray(values, dtype=np.float64)
        if col >= 6:
            values = _standardize_feature(values[: config.num_regular_states])
            features[regular, 0, col] = values
        else:
            features[regular, 0, col] = values[regular]

    wait_offset = len(selected_read)
    for pos, (_, values) in enumerate(selected_wait):
        values = np.asarray(values, dtype=np.float64)
        col = wait_offset + pos
        if pos >= 5:
            values = _standardize_feature(values[: config.num_regular_states])
            features[regular, 1, col] = values
        else:
            features[regular, 1, col] = values[regular]

    features[:, config.exit_action, :] = 0.0
    features[config.absorbing_state, :, :] = 0.0
    return jnp.asarray(features, dtype=jnp.float32), names


def _build_content_parameters(
    config: ContentHeterogeneityKnownTruthConfig,
    parameter_names: list[str],
) -> jnp.ndarray:
    del config
    binge_reader = {
        "read_intercept": 0.05,
        "read_quality": 0.85,
        "read_cliffhanger": 0.45,
        "read_progress": 0.18,
        "read_price": -0.45,
        "read_unlock_ready": 0.10,
        "read_price_quality": 0.00,
        "read_progress_quality": 0.00,
        "read_price_cliffhanger": 0.00,
        "read_wait_delay": 0.00,
        "read_final_chapter": 0.00,
        "read_episode_wave": 0.00,
        "read_unlock_price": 0.00,
        "read_late_unlock": 0.00,
        "read_progress_price": 0.00,
        "read_wait_cliffhanger": 0.00,
        "wait_intercept": -0.45,
        "wait_quality": 0.20,
        "wait_delay": -0.25,
        "wait_unlock_ready": 0.25,
        "wait_progress": -0.10,
        "wait_price": 0.00,
        "wait_price_quality": 0.00,
        "wait_cliffhanger": 0.00,
        "wait_progress_quality": 0.00,
        "wait_final_chapter": 0.00,
        "wait_price_delay": 0.00,
        "wait_cliffhanger_delay": 0.00,
        "wait_unlock_price": 0.00,
        "wait_late_unlock": 0.00,
        "wait_episode_wave": 0.00,
        "wait_progress_price": 0.00,
    }
    patient_reader = {
        "read_intercept": -0.20,
        "read_quality": 0.65,
        "read_cliffhanger": 0.25,
        "read_progress": 0.05,
        "read_price": -1.00,
        "read_unlock_ready": 0.45,
        "read_price_quality": 0.00,
        "read_progress_quality": 0.00,
        "read_price_cliffhanger": 0.00,
        "read_wait_delay": 0.00,
        "read_final_chapter": 0.00,
        "read_episode_wave": 0.00,
        "read_unlock_price": 0.00,
        "read_late_unlock": 0.00,
        "read_progress_price": 0.00,
        "read_wait_cliffhanger": 0.00,
        "wait_intercept": -0.10,
        "wait_quality": 0.35,
        "wait_delay": -0.05,
        "wait_unlock_ready": 0.65,
        "wait_progress": 0.10,
        "wait_price": 0.00,
        "wait_price_quality": 0.00,
        "wait_cliffhanger": 0.00,
        "wait_progress_quality": 0.00,
        "wait_final_chapter": 0.00,
        "wait_price_delay": 0.00,
        "wait_cliffhanger_delay": 0.00,
        "wait_unlock_price": 0.00,
        "wait_late_unlock": 0.00,
        "wait_episode_wave": 0.00,
        "wait_progress_price": 0.00,
    }
    return jnp.asarray(
        np.vstack(
            [
                [binge_reader.get(name, 0.0) for name in parameter_names],
                [patient_reader.get(name, 0.0) for name in parameter_names],
            ]
        ),
        dtype=jnp.float32,
    )


def simulate_known_truth_panel(
    dgp: KnownTruthDGP,
    config: SimulationConfig | None = None,
) -> Panel:
    """Simulate panel data from the known optimal policy."""


    if config is None:
        config = SimulationConfig()
    if isinstance(dgp.config, ContentHeterogeneityKnownTruthConfig):
        return _simulate_content_heterogeneity_panel(dgp, config)

    rng = np.random.default_rng(config.seed)
    solutions = [
        solve_known_truth(dgp, segment_index=g)
        for g in range(dgp.num_segments)
    ]

    iterator = range(config.n_individuals)
    if config.show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="simulate known-truth panel")

    trajectories: list[Trajectory] = []
    segment_labels: list[int] = []
    segment_probs = (
        np.asarray(dgp.segment_probabilities)
        if dgp.segment_probabilities is not None
        else np.ones(1)
    )

    for i in iterator:
        segment = int(rng.choice(dgp.num_segments, p=segment_probs))
        segment_labels.append(segment)
        policy = np.asarray(solutions[segment].policy)

        state = int(rng.choice(dgp.problem.num_states, p=np.asarray(dgp.initial_distribution)))
        states = np.empty(config.n_periods, dtype=np.int32)
        actions = np.empty(config.n_periods, dtype=np.int32)
        next_states = np.empty(config.n_periods, dtype=np.int32)

        for t in range(config.n_periods):
            action_probs = policy[state].astype(np.float64)
            action_probs = action_probs / action_probs.sum()
            action = int(rng.choice(dgp.problem.num_actions, p=action_probs))
            transition_probs = np.asarray(dgp.transitions[action, state], dtype=np.float64)
            transition_probs = transition_probs / transition_probs.sum()
            next_state = int(rng.choice(dgp.problem.num_states, p=transition_probs))

            states[t] = state
            actions[t] = action
            next_states[t] = next_state
            state = next_state

        trajectories.append(
            Trajectory(
                states=jnp.array(states, dtype=jnp.int32),
                actions=jnp.array(actions, dtype=jnp.int32),
                next_states=jnp.array(next_states, dtype=jnp.int32),
                individual_id=i,
                metadata={"segment": segment},
            )
        )

    metadata = dgp.metadata()
    metadata.update(
        {
            "simulation": {
                "n_individuals": config.n_individuals,
                "n_periods": config.n_periods,
                "seed": config.seed,
            },
            "segment_labels": segment_labels,
        }
    )
    return Panel(trajectories=trajectories, metadata=metadata)


def _simulate_content_heterogeneity_panel(
    dgp: KnownTruthDGP,
    config: SimulationConfig,
) -> Panel:
    """Simulate repeated content series with fixed latent user types."""

    dgp_config = dgp.config
    if not isinstance(dgp_config, ContentHeterogeneityKnownTruthConfig):
        raise TypeError("content simulation requires ContentHeterogeneityKnownTruthConfig")

    rng = np.random.default_rng(config.seed)
    solutions = [
        solve_known_truth(dgp, segment_index=g)
        for g in range(dgp.num_segments)
    ]
    segment_probs = np.asarray(dgp.segment_probabilities, dtype=np.float64)
    segment_probs = segment_probs / segment_probs.sum()
    initial_distribution = np.asarray(dgp.initial_distribution, dtype=np.float64)

    iterator = range(config.n_individuals)
    if config.show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="simulate content known-truth panel")

    trajectories: list[Trajectory] = []
    trajectory_segment_labels: list[int] = []
    user_segment_labels: list[int] = []

    for user_id in iterator:
        segment = int(rng.choice(dgp.num_segments, p=segment_probs))
        user_segment_labels.append(segment)
        policy = np.asarray(solutions[segment].policy, dtype=np.float64)

        for book_id in range(dgp_config.books_per_user):
            state = int(rng.choice(dgp.problem.num_states, p=initial_distribution))
            states: list[int] = []
            actions: list[int] = []
            next_states: list[int] = []

            for _ in range(config.n_periods):
                action_probs = policy[state].astype(np.float64)
                action_probs = action_probs / action_probs.sum()
                action = int(rng.choice(dgp.problem.num_actions, p=action_probs))
                transition_probs = np.asarray(
                    dgp.transitions[action, state],
                    dtype=np.float64,
                )
                transition_probs = transition_probs / transition_probs.sum()
                next_state = int(rng.choice(dgp.problem.num_states, p=transition_probs))

                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                state = next_state
                if state == dgp_config.absorbing_state:
                    break

            trajectories.append(
                Trajectory(
                    states=jnp.asarray(states, dtype=jnp.int32),
                    actions=jnp.asarray(actions, dtype=jnp.int32),
                    next_states=jnp.asarray(next_states, dtype=jnp.int32),
                    individual_id=user_id,
                    metadata={"segment": segment, "book_id": book_id},
                )
            )
            trajectory_segment_labels.append(segment)

    metadata = dgp.metadata()
    metadata.update(
        {
            "simulation": {
                "n_individuals": config.n_individuals,
                "n_periods": config.n_periods,
                "seed": config.seed,
                "books_per_user": dgp_config.books_per_user,
                "trajectory_unit": "book",
            },
            "segment_labels": trajectory_segment_labels,
            "user_segment_labels": user_segment_labels,
        }
    )
    return Panel(trajectories=trajectories, metadata=metadata)


def _build_initial_distribution(config: KnownTruthDGPConfig) -> jnp.ndarray:
    dist = np.zeros(config.num_states, dtype=np.float64)
    if config.initial_state_mode == "start":
        dist[0] = 1.0
    else:
        dist[: config.num_regular_states] = 1.0 / config.num_regular_states
    return jnp.array(dist, dtype=jnp.float32)


def _build_transitions(config: KnownTruthDGPConfig) -> jnp.ndarray:
    transitions = np.zeros(
        (config.num_actions, config.num_states, config.num_states), dtype=np.float64
    )
    absorbing = config.absorbing_state
    non_exit_actions = [a for a in range(config.num_actions) if a != config.exit_action]

    for state in range(config.num_regular_states):
        for action in range(config.num_actions):
            if action == config.exit_action:
                transitions[action, state, absorbing] = 1.0
                continue

            action_position = non_exit_actions.index(action)
            if action_position == 0:
                target = min(state + 1, config.num_regular_states - 1)
            elif action_position == 1:
                target = state
            elif action_position % 2 == 0:
                target = min(state + 2, config.num_regular_states - 1)
            else:
                target = max(state - 1, 0)

            noise = config.transition_noise
            transitions[action, state, target] += 1.0 - noise
            if noise > 0:
                left = max(target - 1, 0)
                right = min(target + 1, config.num_regular_states - 1)
                if left == right:
                    transitions[action, state, target] += noise
                else:
                    transitions[action, state, left] += noise / 2.0
                    transitions[action, state, right] += noise / 2.0

    transitions[:, absorbing, absorbing] = 1.0
    transitions = transitions / transitions.sum(axis=2, keepdims=True)
    return jnp.array(transitions, dtype=jnp.float32)


def _build_state_features(config: KnownTruthDGPConfig) -> jnp.ndarray:
    regular = config.num_regular_states
    progress = np.linspace(0.0, 1.0, regular, dtype=np.float64)

    if config.state_mode == "low_dim":
        features = np.column_stack(
            [
                progress,
                np.sin(2.0 * np.pi * progress),
            ]
        )
    else:
        rng = np.random.default_rng(config.seed + 17)
        cols = [
            progress,
            progress**2,
            progress**3,
            np.sin(2.0 * np.pi * progress),
            np.cos(2.0 * np.pi * progress),
        ]
        while len(cols) < config.high_state_dim:
            freq = rng.uniform(0.5, 4.0)
            phase = rng.uniform(-np.pi, np.pi)
            cols.append(np.sin(freq * np.pi * progress + phase))
        features = np.column_stack(cols[: config.high_state_dim])

    absorbing = np.zeros((1, features.shape[1]), dtype=np.float64)
    features = np.vstack([features, absorbing])
    return jnp.array(features * config.feature_scale, dtype=jnp.float32)


def _build_reward_features(
    config: KnownTruthDGPConfig,
    state_features: jnp.ndarray,
) -> tuple[jnp.ndarray, list[str]]:
    S, A = config.num_states, config.num_actions
    non_exit_actions = [a for a in range(A) if a != config.exit_action]
    progress = np.asarray(state_features[:, 0])
    wave = np.asarray(state_features[:, 1]) if state_features.shape[1] > 1 else progress

    if config.reward_dim == "low":
        if config.reward_mode == "state_only":
            names = ["state_intercept", "state_progress", "state_wave"]
            features = np.zeros((S, A, len(names)), dtype=np.float64)
            base = np.column_stack([np.ones(S), progress, wave])
            for action in non_exit_actions:
                features[:, action, :] = base
        else:
            names = []
            features = np.zeros((S, A, 2 * len(non_exit_actions)), dtype=np.float64)
            col = 0
            for action in non_exit_actions:
                names.extend([f"action_{action}_intercept", f"action_{action}_progress"])
                features[:, action, col] = 1.0
                features[:, action, col + 1] = progress
                col += 2
            features[:, :, :] += 0.05 * wave[:, None, None]
            features[:, config.exit_action, :] = 0.0
    else:
        K = config.high_reward_features
        names = [f"theta_{k}" for k in range(K)]
        features = np.zeros((S, A, K), dtype=np.float64)
        basis = _expand_state_basis(np.asarray(state_features), K)
        if config.reward_mode == "state_only":
            for action in non_exit_actions:
                features[:, action, :] = basis
        else:
            rng = np.random.default_rng(config.seed + 31)
            action_embeddings = rng.normal(size=(A, K))
            action_embeddings = action_embeddings / np.maximum(
                np.linalg.norm(action_embeddings, axis=1, keepdims=True), 1e-8
            )
            for action in non_exit_actions:
                features[:, action, :] = basis * (1.0 + 0.5 * action_embeddings[action])
        features[:, config.exit_action, :] = 0.0

    features[config.absorbing_state, :, :] = 0.0
    return jnp.array(features, dtype=jnp.float32), names


def _expand_state_basis(state_features: np.ndarray, n_features: int) -> np.ndarray:
    """Build a stable high-dimensional reward basis.

    The high-dimensional DGP is meant to stress estimators with many reward
    features, not to create near-collinearity. Raw powers and random waves of
    the same one-dimensional progress variable can be full rank but extremely
    ill-conditioned on the finite grid. We therefore generate an overcomplete
    deterministic dictionary and take its orthogonal principal components on
    the regular states.
    """
    n_states = state_features.shape[0]
    regular_mask = np.linalg.norm(state_features, axis=1) > 1e-12
    if int(regular_mask.sum()) < n_features:
        regular_mask[:] = True

    progress = state_features[:, 0]
    dictionary: list[np.ndarray] = []
    for j in range(state_features.shape[1]):
        col = state_features[:, j]
        dictionary.append(col)
        dictionary.append(col**2)
        dictionary.append(np.sin(np.pi * col))
        dictionary.append(np.cos(np.pi * col))

    freq = 1
    while len(dictionary) < max(4 * n_features, n_features + 8):
        dictionary.append(np.sin(freq * np.pi * progress))
        dictionary.append(np.cos(freq * np.pi * progress))
        freq += 1

    raw = np.column_stack(dictionary)
    raw_regular = raw[regular_mask]
    raw_regular = raw_regular - raw_regular.mean(axis=0, keepdims=True)
    scale = np.maximum(raw_regular.std(axis=0, keepdims=True), 1e-8)
    raw_regular = raw_regular / scale

    u, singular_values, _ = np.linalg.svd(raw_regular, full_matrices=False)
    available = int(np.sum(singular_values > 1e-10))
    needed = n_features - 1
    use = min(available, needed)

    basis = np.zeros((n_states, n_features), dtype=np.float64)
    basis[:, 0] = 1.0
    if use:
        basis[regular_mask, 1 : 1 + use] = (
            u[:, :use] * np.sqrt(float(regular_mask.sum()))
        )
    if use < needed:
        fallback = raw_regular[:, : needed - use]
        basis[regular_mask, 1 + use :] = fallback
    return basis


def _build_parameters(config: KnownTruthDGPConfig, n_params: int) -> jnp.ndarray:
    rng = np.random.default_rng(config.seed + 53)
    if config.reward_dim == "low":
        if config.reward_mode == "action_dependent":
            non_exit_actions = [
                action
                for action in range(config.num_actions)
                if action != config.exit_action
            ]
            intercepts = np.linspace(0.10, 0.00, len(non_exit_actions))
            slopes = np.linspace(0.50, -0.20, len(non_exit_actions))
            base = np.empty(n_params, dtype=np.float64)
            for idx in range(len(non_exit_actions)):
                base[2 * idx] = intercepts[idx]
                base[2 * idx + 1] = slopes[idx]
        else:
            base = np.array([0.15, 0.40, -0.20], dtype=np.float64)[:n_params]
    else:
        base = rng.normal(size=n_params) / np.sqrt(n_params)
        keep = max(4, n_params // 3)
        base[keep:] *= 0.25

    if config.heterogeneity == "none":
        return jnp.array(base, dtype=jnp.float32)

    segments = []
    for g in range(config.num_segments):
        direction = -1.0 if g % 2 else 1.0
        perturb = direction * 0.35 * np.roll(base, g + 1)
        segments.append(base + perturb)
    return jnp.array(np.vstack(segments), dtype=jnp.float32)


def _compute_rewards(feature_matrix: jnp.ndarray, parameters: jnp.ndarray) -> jnp.ndarray:
    if parameters.ndim == 1:
        return jnp.einsum("sak,k->sa", feature_matrix, parameters)
    return jnp.einsum("sak,gk->gsa", feature_matrix, parameters)


# --- Truth Solver ---
@dataclass(frozen=True)
class KnownTruthSolution:
    """Exact Bellman solution and occupancy objects for one segment."""

    segment_index: int
    reward_matrix: jnp.ndarray
    Q: jnp.ndarray
    V: jnp.ndarray
    policy: jnp.ndarray
    state_occupancy: jnp.ndarray
    state_action_occupancy: jnp.ndarray
    converged: bool
    num_iterations: int
    final_error: float


def get_segment_reward(dgp: KnownTruthDGP, segment_index: int = 0) -> jnp.ndarray:
    if dgp.reward_matrix.ndim == 2:
        if segment_index != 0:
            raise IndexError("homogeneous DGP has only segment 0")
        return dgp.reward_matrix
    if not 0 <= segment_index < dgp.reward_matrix.shape[0]:
        raise IndexError(f"segment_index {segment_index} is out of range")
    return dgp.reward_matrix[segment_index]


def solve_known_truth(
    dgp: KnownTruthDGP,
    segment_index: int = 0,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> KnownTruthSolution:
    """Solve the DGP exactly under the true reward."""

    reward = get_segment_reward(dgp, segment_index)
    operator = SoftBellmanOperator(dgp.problem, dgp.transitions)
    result = value_iteration(operator, reward, tol=tol, max_iter=max_iter)
    state_occ = compute_state_visitation(
        result.policy,
        dgp.transitions,
        dgp.problem,
        dgp.initial_distribution,
    )
    state_action_occ = compute_state_action_visitation(
        result.policy,
        dgp.transitions,
        dgp.problem,
        dgp.initial_distribution,
    )
    return KnownTruthSolution(
        segment_index=segment_index,
        reward_matrix=reward,
        Q=result.Q,
        V=result.V,
        policy=result.policy,
        state_occupancy=state_occ,
        state_action_occupancy=state_action_occ,
        converged=result.converged,
        num_iterations=result.num_iterations,
        final_error=result.final_error,
    )


# --- Counterfactual Oracles ---
@dataclass(frozen=True)
class CounterfactualDGP:
    """A counterfactual environment derived from a baseline DGP."""

    kind: CounterfactualKind
    description: str
    baseline: KnownTruthDGP
    reward_matrix: jnp.ndarray
    transitions: jnp.ndarray
    disabled_action: int | None = None


@dataclass(frozen=True)
class CounterfactualOracle:
    """Baseline and counterfactual oracle solutions for one segment."""

    counterfactual: CounterfactualDGP
    segment_index: int
    baseline_solution: KnownTruthSolution
    counterfactual_solution: KnownTruthSolution


def build_counterfactual(
    dgp: KnownTruthDGP,
    kind: CounterfactualKind,
    config: CounterfactualConfig | None = None,
) -> CounterfactualDGP:
    """Build a Type A, Type B, or Type C counterfactual DGP."""

    if config is None:
        config = CounterfactualConfig()

    reward = dgp.reward_matrix
    transitions = dgp.transitions
    disabled_action = None

    if kind == "type_a":
        shift = _state_shift(dgp, config.type_a_shift)
        reward = reward + shift
        description = "Type A reward feature shift with baseline transitions"
    elif kind == "type_b":
        transitions = _skip_transitions(dgp, config.type_b_skip)
        description = "Type B transition change with baseline reward"
    elif kind == "type_c":
        disabled_action = config.type_c_action
        reward = _penalize_action(dgp, reward, disabled_action, config.type_c_penalty)
        description = "Type C action design intervention by disabling one action"
    else:
        raise ValueError(f"unknown counterfactual kind {kind!r}")

    return CounterfactualDGP(
        kind=kind,
        description=description,
        baseline=dgp,
        reward_matrix=reward,
        transitions=transitions,
        disabled_action=disabled_action,
    )


def solve_counterfactual_oracle(
    dgp: KnownTruthDGP,
    kind: CounterfactualKind,
    segment_index: int = 0,
    config: CounterfactualConfig | None = None,
) -> CounterfactualOracle:
    """Solve baseline and counterfactual policies for one segment."""

    counterfactual = build_counterfactual(dgp, kind, config)
    cf_dgp = _replace_truth_objects(
        dgp,
        reward_matrix=counterfactual.reward_matrix,
        transitions=counterfactual.transitions,
    )
    return CounterfactualOracle(
        counterfactual=counterfactual,
        segment_index=segment_index,
        baseline_solution=solve_known_truth(dgp, segment_index),
        counterfactual_solution=solve_known_truth(cf_dgp, segment_index),
    )


def _replace_truth_objects(
    dgp: KnownTruthDGP,
    reward_matrix: jnp.ndarray,
    transitions: jnp.ndarray,
) -> KnownTruthDGP:
    return KnownTruthDGP(
        config=dgp.config,
        problem=dgp.problem,
        transitions=transitions,
        feature_matrix=dgp.feature_matrix,
        state_features=dgp.state_features,
        parameter_names=dgp.parameter_names,
        true_parameters=dgp.true_parameters,
        reward_matrix=reward_matrix,
        initial_distribution=dgp.initial_distribution,
        segment_probabilities=dgp.segment_probabilities,
    )


def _absorbing_state(dgp: KnownTruthDGP) -> int | None:
    value = getattr(dgp.config, "absorbing_state", None)
    if value is None:
        return None
    return int(value)


def _exit_action(dgp: KnownTruthDGP) -> int | None:
    if not bool(getattr(dgp.config, "uses_exit_anchor", False)):
        return None
    value = getattr(dgp.config, "exit_action", None)
    if value is None:
        return None
    return int(value)


def _regular_state_mask(dgp: KnownTruthDGP) -> jnp.ndarray:
    mask = jnp.ones(dgp.problem.num_states, dtype=bool)
    absorbing = _absorbing_state(dgp)
    if absorbing is not None and 0 <= absorbing < dgp.problem.num_states:
        mask = mask.at[absorbing].set(False)
    return mask


def _regular_state_limit(dgp: KnownTruthDGP) -> int:
    absorbing = _absorbing_state(dgp)
    if absorbing is None:
        return dgp.problem.num_states
    return min(int(getattr(dgp.config, "num_regular_states", absorbing)), dgp.problem.num_states)


def _state_shift(dgp: KnownTruthDGP, amount: float) -> jnp.ndarray:
    progress = dgp.state_features[:, 0]
    regular_mask = _regular_state_mask(dgp)
    action_mask = jnp.ones(dgp.problem.num_actions, dtype=jnp.float32)
    exit_action = _exit_action(dgp)
    if exit_action is not None:
        action_mask = action_mask.at[exit_action].set(0.0)
    shift = amount * progress[:, None] * action_mask[None, :]
    shift = jnp.where(regular_mask[:, None], shift, 0.0)
    if dgp.reward_matrix.ndim == 3:
        return shift[None, :, :]
    return shift


def _skip_transitions(dgp: KnownTruthDGP, skip: int) -> jnp.ndarray:
    transitions = np.asarray(dgp.transitions).copy()
    advance_action = 0
    exit_action = _exit_action(dgp)
    if exit_action is not None and advance_action == exit_action:
        advance_action = 1
    transitions[advance_action, :, :] = 0.0
    absorbing = _absorbing_state(dgp)
    regular_limit = _regular_state_limit(dgp)
    for state in range(regular_limit):
        if absorbing is None:
            target = (state + skip) % regular_limit
        else:
            target = min(state + skip, regular_limit - 1)
        transitions[advance_action, state, target] = 1.0
    if absorbing is not None:
        transitions[advance_action, absorbing, absorbing] = 1.0
    transitions = transitions / transitions.sum(axis=2, keepdims=True)
    return jnp.array(transitions, dtype=jnp.float32)


def _penalize_action(
    dgp: KnownTruthDGP,
    reward: jnp.ndarray,
    action: int,
    penalty: float,
) -> jnp.ndarray:
    if not 0 <= action < dgp.problem.num_actions:
        raise ValueError(f"action {action} is out of range")
    exit_action = _exit_action(dgp)
    if exit_action is not None and action == exit_action:
        raise ValueError("Type C should not disable the anchor exit action")
    regular_limit = _regular_state_limit(dgp)
    if reward.ndim == 2:
        return reward.at[:regular_limit, action].add(penalty)
    return reward.at[:, :regular_limit, action].add(penalty)


# --- Pre-Estimation Diagnostics ---
@dataclass(frozen=True)
class PreEstimationDiagnostics:
    """Diagnostics that should be checked before estimator execution."""

    feature_rank: int
    num_features: int
    condition_number: float
    is_action_dependent: bool
    max_transition_row_error: float
    observed_states: int | None = None
    num_states: int | None = None
    single_action_states: int | None = None
    state_action_coverage: float | None = None
    action_shares: list[float] | None = None
    min_action_share: float | None = None
    min_positive_ccp: float | None = None
    anchor_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


def run_pre_estimation_diagnostics(
    dgp: KnownTruthDGP,
    panel: Panel | None = None,
    condition_threshold: float = 1e6,
) -> PreEstimationDiagnostics:
    """Run structural and sample diagnostics before fitting an estimator."""

    features = np.asarray(dgp.feature_matrix, dtype=np.float64)
    flat_features = features.reshape(-1, features.shape[-1])
    nonzero_rows = flat_features[np.linalg.norm(flat_features, axis=1) > 1e-12]
    if nonzero_rows.size == 0:
        feature_rank = 0
        condition_number = float("inf")
    else:
        feature_rank = int(np.linalg.matrix_rank(nonzero_rows))
        condition_number = _safe_condition_number(nonzero_rows)

    transitions = np.asarray(dgp.transitions, dtype=np.float64)
    row_sums = transitions.sum(axis=2)
    max_transition_row_error = float(np.max(np.abs(row_sums - 1.0)))

    exit_action = _exit_action(dgp)
    non_exit_actions = [
        a for a in range(dgp.problem.num_actions) if a != exit_action
    ]
    action_features = features[:, non_exit_actions, :]
    action_reference = action_features[:, :1, :]
    action_diff = np.max(np.abs(action_features - action_reference))
    is_action_dependent = bool(action_diff > 1e-8)

    anchor_valid = True
    absorbing = _absorbing_state(dgp)
    if exit_action is not None and absorbing is not None:
        regular_limit = _regular_state_limit(dgp)
        exit_reward = np.asarray(dgp.homogeneous_reward[:, exit_action])
        absorbing_reward = np.asarray(dgp.homogeneous_reward[absorbing, :])
        exit_transitions = transitions[exit_action, :regular_limit]
        anchor_target = exit_transitions[:, absorbing]
        anchor_valid = bool(
            np.max(np.abs(exit_reward)) < 1e-6
            and np.max(np.abs(absorbing_reward)) < 1e-6
            and np.min(anchor_target) > 1.0 - 1e-6
        )

    errors: list[str] = []
    warnings: list[str] = []
    if feature_rank < features.shape[-1]:
        errors.append(
            f"feature rank {feature_rank} is less than {features.shape[-1]} features"
        )
    if condition_number > condition_threshold:
        warnings.append(f"feature condition number {condition_number:.3g} is high")
    if max_transition_row_error > 1e-6:
        errors.append(
            f"transition rows are not stochastic, max error {max_transition_row_error:.3g}"
        )
    if not anchor_valid:
        errors.append("exit or absorbing-state anchor is invalid")

    observed_states = None
    single_action_states = None
    coverage = None
    action_shares = None
    min_action_share = None
    min_positive_ccp = None
    if panel is not None:
        states = np.asarray(panel.get_all_states(), dtype=np.int64)
        actions = np.asarray(panel.get_all_actions(), dtype=np.int64)
        counts = np.zeros((dgp.problem.num_states, dgp.problem.num_actions), dtype=np.float64)
        for state, action in zip(states, actions):
            counts[state, action] += 1.0
        action_counts = counts.sum(axis=0)
        action_shares = (action_counts / max(action_counts.sum(), 1.0)).tolist()
        min_action_share = float(np.min(action_shares))
        observed_state_mask = counts.sum(axis=1) > 0
        observed_states = int(observed_state_mask.sum())
        single_action_states = int(
            np.logical_and((counts > 0).sum(axis=1) == 1, observed_state_mask).sum()
        )
        coverage = float((counts > 0).sum() / counts.size)
        positive = counts[counts > 0]
        min_positive_ccp = None
        if positive.size:
            row_sums = counts.sum(axis=1, keepdims=True)
            ccps = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
            min_positive_ccp = float(ccps[ccps > 0].min())
        if observed_states < dgp.problem.num_states:
            warnings.append(
                f"{observed_states} of {dgp.problem.num_states} states are observed"
            )
        if single_action_states > 0:
            warnings.append(f"{single_action_states} observed states have one action")
        if min_action_share < 0.02:
            warnings.append(f"minimum action share {min_action_share:.3g} is very low")

    return PreEstimationDiagnostics(
        feature_rank=feature_rank,
        num_features=features.shape[-1],
        condition_number=condition_number,
        is_action_dependent=is_action_dependent,
        max_transition_row_error=max_transition_row_error,
        observed_states=observed_states,
        num_states=dgp.problem.num_states if panel is not None else None,
        single_action_states=single_action_states,
        state_action_coverage=coverage,
        action_shares=action_shares,
        min_action_share=min_action_share,
        min_positive_ccp=min_positive_ccp,
        anchor_valid=anchor_valid,
        errors=errors,
        warnings=warnings,
    )


def _safe_condition_number(x: np.ndarray) -> float:
    try:
        value = float(np.linalg.cond(x))
    except np.linalg.LinAlgError:
        value = float("inf")
    if not np.isfinite(value):
        return float("inf")
    return value


# --- Recovery Metrics ---
@dataclass(frozen=True)
class PolicyMetrics:
    l1: float
    linf: float
    tv: float
    kl: float


@dataclass(frozen=True)
class CounterfactualMetrics:
    policy: PolicyMetrics
    value_rmse: float
    regret: float


@dataclass(frozen=True)
class ParameterMetrics:
    rmse: float
    relative_rmse: float
    max_abs_error: float
    cosine_similarity: float


def rmse(estimated: jnp.ndarray, truth: jnp.ndarray) -> float:
    estimated = jnp.asarray(estimated)
    truth = jnp.asarray(truth)
    return float(jnp.sqrt(jnp.mean((estimated - truth) ** 2)))


def normalized_rmse(
    estimated: jnp.ndarray,
    truth: jnp.ndarray,
    mask: jnp.ndarray | np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """RMSE after the standard IRL location/scale normalization.

    This is for IRL reward-like objects where additive constants and positive
    rescaling are not the structural claim. It does not remove potential-based
    shaping or change the policy gate.
    """

    estimated = jnp.asarray(estimated, dtype=jnp.float64)
    truth = jnp.asarray(truth, dtype=jnp.float64)
    if mask is not None:
        mask_arr = jnp.asarray(mask, dtype=bool)
        estimated = estimated[mask_arr]
        truth = truth[mask_arr]
    if estimated.size == 0:
        estimated = jnp.ravel(jnp.asarray(estimated, dtype=jnp.float64))
        truth = jnp.ravel(jnp.asarray(truth, dtype=jnp.float64))

    truth_centered = truth - jnp.mean(truth)
    estimated_centered = estimated - jnp.mean(estimated)
    truth_scale = jnp.sqrt(jnp.mean(truth_centered**2))
    estimated_scale = jnp.sqrt(jnp.mean(estimated_centered**2))
    truth_norm = truth_centered / jnp.maximum(truth_scale, eps)
    estimated_norm = estimated_centered / jnp.maximum(estimated_scale, eps)
    return rmse(estimated_norm, truth_norm)


def _affine_align_for_recovery(
    estimated: jnp.ndarray,
    truth: jnp.ndarray,
    mask: jnp.ndarray | np.ndarray | None = None,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Map an IRL reward estimate onto the truth location/scale convention."""

    estimated = jnp.asarray(estimated, dtype=jnp.float64)
    truth = jnp.asarray(truth, dtype=jnp.float64)
    if mask is None:
        est_vec = estimated.reshape(-1)
        truth_vec = truth.reshape(-1)
    else:
        mask_arr = jnp.asarray(mask, dtype=bool)
        est_vec = estimated[mask_arr]
        truth_vec = truth[mask_arr]

    est_centered = est_vec - jnp.mean(est_vec)
    truth_centered = truth_vec - jnp.mean(truth_vec)
    denom = jnp.sum(est_centered**2)
    scale = jnp.where(
        denom > eps,
        jnp.sum(est_centered * truth_centered) / denom,
        1.0,
    )
    scale = jnp.where(scale > eps, scale, 1.0)
    offset = jnp.mean(truth_vec) - scale * jnp.mean(est_vec)
    return scale * estimated + offset


def _anchor_project_reward(dgp: Any, reward: jnp.ndarray) -> jnp.ndarray | None:
    """Project a reward table into the DGP's exit-action anchor convention."""

    exit_action = _exit_action(dgp)
    if exit_action is None or not 0 <= exit_action < dgp.problem.num_actions:
        return None

    projected = jnp.asarray(reward, dtype=jnp.float64)
    projected = projected - projected[:, exit_action][:, None]
    projected = projected.at[:, exit_action].set(0.0)

    absorbing = _absorbing_state(dgp)
    if absorbing is not None and 0 <= absorbing < dgp.problem.num_states:
        projected = projected.at[absorbing, :].set(0.0)
    return projected


def parameter_metrics(
    truth: jnp.ndarray,
    estimated: jnp.ndarray,
    eps: float = 1e-12,
) -> ParameterMetrics:
    """Compare estimated structural parameters to known truth."""

    truth = jnp.asarray(truth)
    estimated = jnp.asarray(estimated)
    error = estimated - truth
    error_rmse = jnp.sqrt(jnp.mean(error**2))
    truth_rms = jnp.sqrt(jnp.mean(truth**2))
    max_abs = jnp.max(jnp.abs(error))
    denom = jnp.linalg.norm(truth) * jnp.linalg.norm(estimated)
    cosine = jnp.where(
        denom > eps,
        jnp.dot(truth, estimated) / denom,
        jnp.nan,
    )
    return ParameterMetrics(
        rmse=float(error_rmse),
        relative_rmse=float(error_rmse / jnp.maximum(truth_rms, eps)),
        max_abs_error=float(max_abs),
        cosine_similarity=float(cosine),
    )


def policy_divergence(
    truth: jnp.ndarray,
    estimated: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    eps: float = 1e-12,
) -> PolicyMetrics:
    truth = jnp.asarray(truth)
    estimated = jnp.asarray(estimated)
    if weights is None:
        weights = jnp.ones(truth.shape[0]) / truth.shape[0]
    weights = weights / weights.sum()
    diff = jnp.abs(truth - estimated)
    l1_by_state = diff.sum(axis=1)
    l1 = jnp.sum(weights * l1_by_state)
    linf = jnp.max(l1_by_state)
    tv = 0.5 * l1
    p = jnp.clip(truth, eps, 1.0)
    q = jnp.clip(estimated, eps, 1.0)
    kl = jnp.sum(weights * jnp.sum(p * (jnp.log(p) - jnp.log(q)), axis=1))
    return PolicyMetrics(l1=float(l1), linf=float(linf), tv=float(tv), kl=float(kl))


def evaluate_policy_value(
    reward: jnp.ndarray,
    transitions: jnp.ndarray,
    policy: jnp.ndarray,
    discount_factor: float,
    scale_parameter: float = 1.0,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Evaluate a stochastic policy under the soft-logit Bellman objective."""

    clipped_policy = jnp.clip(policy, eps, 1.0)
    entropy_flow = -scale_parameter * jnp.sum(policy * jnp.log(clipped_policy), axis=1)
    reward_pi = jnp.sum(policy * reward, axis=1) + entropy_flow
    transition_pi = jnp.einsum("sa,ast->st", policy, transitions)
    lhs = jnp.eye(reward.shape[0]) - discount_factor * transition_pi
    return jnp.linalg.solve(lhs, reward_pi)


def q_from_value(
    reward: jnp.ndarray,
    value: jnp.ndarray,
    transitions: jnp.ndarray,
    discount_factor: float,
) -> jnp.ndarray:
    """Compute Q(s,a) from a reward matrix and continuation value."""

    continuation = jnp.einsum("ast,t->as", transitions, value).T
    return reward + discount_factor * continuation


def counterfactual_metrics(
    oracle_policy: jnp.ndarray,
    oracle_value: jnp.ndarray,
    estimated_policy: jnp.ndarray,
    reward: jnp.ndarray,
    transitions: jnp.ndarray,
    discount_factor: float,
    initial_distribution: jnp.ndarray,
    scale_parameter: float = 1.0,
) -> CounterfactualMetrics:
    estimated_value = evaluate_policy_value(
        reward=reward,
        transitions=transitions,
        policy=estimated_policy,
        discount_factor=discount_factor,
        scale_parameter=scale_parameter,
    )
    policy_metrics = policy_divergence(oracle_policy, estimated_policy)
    value_error = rmse(estimated_value, oracle_value)
    regret_by_state = oracle_value - estimated_value
    regret = float(jnp.dot(initial_distribution, regret_by_state))
    return CounterfactualMetrics(
        policy=policy_metrics,
        value_rmse=value_error,
        regret=regret,
    )


def evaluate_estimator_against_truth(
    dgp: Any,
    summary: Any,
    *,
    panel: Panel | None = None,
    segment_index: int = 0,
    counterfactual_kinds: tuple[str, ...] = ("type_a", "type_b", "type_c"),
) -> dict[str, Any]:
    """Compute estimator-independent known-truth recovery metrics.

    The estimator policy for counterfactuals is obtained by solving the
    intervention under the estimator's recovered reward model, then evaluating
    that policy in the true counterfactual environment.
    """

    if dgp.num_segments > 1 and summary.metadata.get("segment_reward_matrices") is not None:
        return evaluate_segmented_estimator_against_truth(
            dgp,
            summary,
            panel=panel,
            counterfactual_kinds=counterfactual_kinds,
        )


    truth = solve_known_truth(dgp, segment_index=segment_index)
    estimated_params = jnp.asarray(summary.parameters)
    true_params = jnp.asarray(dgp.homogeneous_parameters)

    metrics: dict[str, Any] = {
        "parameters": None,
        "reward_rmse": None,
        "reward_normalized_rmse": None,
        "raw_bellman_reward_rmse": None,
        "raw_bellman_reward_normalized_rmse": None,
        "projected_reward_rmse": None,
        "projected_reward_normalized_rmse": None,
        "value_rmse": None,
        "value_normalized_rmse": None,
        "q_rmse": None,
        "q_normalized_rmse": None,
        "policy": None,
        "counterfactuals": {},
    }

    if estimated_params.shape == true_params.shape and true_params.size > 0:
        metrics["parameters"] = parameter_metrics(true_params, estimated_params)

    true_reward = get_segment_reward(dgp, segment_index)
    reward_mask = _reward_recovery_mask(dgp)
    raw_bellman_reward = _extract_metadata_reward_table(
        dgp,
        summary,
        "raw_bellman_reward_table",
        "reward_table",
    )
    if raw_bellman_reward is not None:
        metrics["raw_bellman_reward_rmse"] = rmse(raw_bellman_reward, true_reward)
        metrics["raw_bellman_reward_normalized_rmse"] = normalized_rmse(
            raw_bellman_reward,
            true_reward,
            reward_mask,
        )

    projected_reward = _extract_metadata_reward_table(
        dgp,
        summary,
        "projected_reward_matrix",
    )
    if (
        projected_reward is None
        and estimated_params.shape == true_params.shape
        and true_params.size > 0
    ):
        projected_reward = dgp.utility().compute(estimated_params)
    if projected_reward is not None:
        metrics["projected_reward_rmse"] = rmse(projected_reward, true_reward)
        metrics["projected_reward_normalized_rmse"] = normalized_rmse(
            projected_reward,
            true_reward,
            reward_mask,
        )

    if (
        summary.metadata.get("counterfactual_reward_source")
        == "raw_bellman_reward_table"
        and raw_bellman_reward is not None
    ):
        estimated_reward = raw_bellman_reward
    else:
        estimated_reward = _extract_estimated_reward(dgp, summary, estimated_params)
    counterfactual_estimated_reward = estimated_reward
    if estimated_reward is not None:
        metrics["reward_rmse"] = rmse(estimated_reward, true_reward)
        metrics["reward_normalized_rmse"] = normalized_rmse(
            estimated_reward,
            true_reward,
            reward_mask,
        )
        estimated_anchor_reward = _anchor_project_reward(dgp, estimated_reward)
        true_anchor_reward = _anchor_project_reward(dgp, true_reward)
        if estimated_anchor_reward is not None and true_anchor_reward is not None:
            metrics["anchor_projected_reward_rmse"] = rmse(
                estimated_anchor_reward[reward_mask],
                true_anchor_reward[reward_mask],
            )
            metrics["anchor_projected_reward_normalized_rmse"] = normalized_rmse(
                estimated_anchor_reward,
                true_anchor_reward,
                reward_mask,
            )
        if summary.metadata.get("counterfactual_reward_normalization") == "affine":
            counterfactual_estimated_reward = _affine_align_for_recovery(
                estimated_reward,
                true_reward,
                reward_mask,
            )

    if summary.value_function is not None:
        estimated_value = jnp.asarray(summary.value_function)
        metrics["value_rmse"] = rmse(estimated_value, truth.V)
        metrics["value_normalized_rmse"] = normalized_rmse(
            estimated_value,
            truth.V,
            _value_recovery_mask(dgp),
        )
        if estimated_reward is not None:
            estimated_q = q_from_value(
                estimated_reward,
                estimated_value,
                dgp.transitions,
                dgp.problem.discount_factor,
            )
            metrics["q_rmse"] = rmse(estimated_q, truth.Q)
            metrics["q_normalized_rmse"] = normalized_rmse(
                estimated_q,
                truth.Q,
                _q_recovery_mask(dgp),
            )

    if summary.policy is not None:
        estimated_policy = jnp.asarray(summary.policy)
        metrics["policy"] = policy_divergence(truth.policy, estimated_policy)

        if counterfactual_estimated_reward is not None:
            for kind in counterfactual_kinds:
                oracle = solve_counterfactual_oracle(dgp, kind, segment_index=segment_index)
                cf_reward = oracle.counterfactual_solution.reward_matrix
                reward_delta = cf_reward - true_reward
                estimated_cf_reward = counterfactual_estimated_reward + reward_delta
                operator = SoftBellmanOperator(dgp.problem, oracle.counterfactual.transitions)
                estimated_cf = value_iteration(
                    operator,
                    estimated_cf_reward,
                    tol=1e-8,
                    max_iter=10_000,
                )
                metrics["counterfactuals"][kind] = counterfactual_metrics(
                    oracle_policy=oracle.counterfactual_solution.policy,
                    oracle_value=oracle.counterfactual_solution.V,
                    estimated_policy=estimated_cf.policy,
                    reward=cf_reward,
                    transitions=oracle.counterfactual.transitions,
                    discount_factor=dgp.problem.discount_factor,
                    initial_distribution=dgp.initial_distribution,
                    scale_parameter=dgp.problem.scale_parameter,
                )

    return metrics


def evaluate_segmented_estimator_against_truth(
    dgp: KnownTruthDGP,
    summary: EstimationSummary,
    *,
    panel: Panel | None = None,
    counterfactual_kinds: tuple[str, ...] = ("type_a", "type_b", "type_c"),
) -> dict[str, Any]:
    """Evaluate latent-segment estimators after resolving label switching."""

    metadata = summary.metadata
    estimated_rewards = jnp.asarray(metadata["segment_reward_matrices"])
    estimated_policies = jnp.asarray(metadata.get("segment_policies", []))
    estimated_values = jnp.asarray(metadata.get("segment_value_functions", []))

    true_count = int(dgp.num_segments)
    estimated_count = int(estimated_rewards.shape[0])
    matched_count = min(true_count, estimated_count)
    reward_mask = _reward_recovery_mask(dgp)
    value_mask = _value_recovery_mask(dgp)
    q_mask = _q_recovery_mask(dgp)

    reward_cost = np.full((estimated_count, true_count), np.inf, dtype=float)
    for est_idx in range(estimated_count):
        for true_idx in range(true_count):
            reward_cost[est_idx, true_idx] = normalized_rmse(
                estimated_rewards[est_idx],
                get_segment_reward(dgp, true_idx),
                reward_mask,
            )

    best_perm: tuple[int, ...] = tuple(range(matched_count))
    best_cost = float("inf")
    for perm in itertools.permutations(range(estimated_count), matched_count):
        cost = sum(reward_cost[perm[true_idx], true_idx] for true_idx in range(matched_count))
        if cost < best_cost:
            best_cost = cost
            best_perm = tuple(int(idx) for idx in perm)

    true_to_estimated = list(best_perm)
    estimated_to_true = {
        estimated_idx: true_idx
        for true_idx, estimated_idx in enumerate(true_to_estimated)
    }

    reward_errors: list[float] = []
    value_errors: list[float] = []
    q_errors: list[float] = []
    policy_metrics: list[PolicyMetrics] = []
    segment_counterfactuals: dict[str, list[CounterfactualMetrics]] = {
        kind: [] for kind in counterfactual_kinds
    }

    for true_idx, estimated_idx in enumerate(true_to_estimated):
        truth = solve_known_truth(dgp, segment_index=true_idx)
        true_reward = get_segment_reward(dgp, true_idx)
        estimated_reward = estimated_rewards[estimated_idx]
        reward_errors.append(
            normalized_rmse(estimated_reward, true_reward, reward_mask)
        )

        estimated_policy = estimated_policies[estimated_idx]
        policy_metrics.append(policy_divergence(truth.policy, estimated_policy))

        estimated_value = estimated_values[estimated_idx]
        value_errors.append(normalized_rmse(estimated_value, truth.V, value_mask))
        estimated_q = q_from_value(
            estimated_reward,
            estimated_value,
            dgp.transitions,
            dgp.problem.discount_factor,
        )
        q_errors.append(normalized_rmse(estimated_q, truth.Q, q_mask))

        for kind in counterfactual_kinds:
            oracle = solve_counterfactual_oracle(dgp, kind, segment_index=true_idx)
            cf_reward = oracle.counterfactual_solution.reward_matrix
            reward_delta = cf_reward - true_reward
            estimated_cf_reward = estimated_reward + reward_delta
            operator = SoftBellmanOperator(dgp.problem, oracle.counterfactual.transitions)
            estimated_cf = value_iteration(
                operator,
                estimated_cf_reward,
                tol=1e-8,
                max_iter=10_000,
            )
            segment_counterfactuals[kind].append(
                counterfactual_metrics(
                    oracle_policy=oracle.counterfactual_solution.policy,
                    oracle_value=oracle.counterfactual_solution.V,
                    estimated_policy=estimated_cf.policy,
                    reward=cf_reward,
                    transitions=oracle.counterfactual.transitions,
                    discount_factor=dgp.problem.discount_factor,
                    initial_distribution=dgp.initial_distribution,
                    scale_parameter=dgp.problem.scale_parameter,
                )
            )

    priors = np.asarray(metadata.get("segment_priors", []), dtype=float)
    true_priors = np.asarray(
        dgp.segment_probabilities
        if dgp.segment_probabilities is not None
        else np.ones(true_count) / true_count,
        dtype=float,
    )
    aligned_priors = np.full(true_count, np.nan, dtype=float)
    if priors.shape[0] == estimated_count:
        for true_idx, estimated_idx in enumerate(true_to_estimated):
            aligned_priors[true_idx] = priors[estimated_idx]
    prior_abs_error = np.abs(aligned_priors - true_priors)

    assignment_accuracy = None
    labels = None if panel is None else panel.metadata.get("segment_labels")
    assignments = metadata.get("segment_assignments")
    if labels is not None and assignments is not None:
        labels_arr = np.asarray(labels, dtype=int)
        assignments_arr = np.asarray(assignments, dtype=int)
        if labels_arr.shape[0] == assignments_arr.shape[0]:
            mapped = np.array(
                [estimated_to_true.get(int(idx), -1) for idx in assignments_arr],
                dtype=int,
            )
            assignment_accuracy = float(np.mean(mapped == labels_arr))

    cf_max_regret = {
        kind: max((cf.regret for cf in values), default=float("inf"))
        for kind, values in segment_counterfactuals.items()
    }

    return {
        "parameters": None,
        "reward_rmse": None,
        "reward_normalized_rmse": max(reward_errors, default=float("inf")),
        "value_rmse": None,
        "value_normalized_rmse": max(value_errors, default=float("inf")),
        "q_rmse": None,
        "q_normalized_rmse": max(q_errors, default=float("inf")),
        "policy": None,
        "counterfactuals": {},
        "num_true_segments": true_count,
        "num_estimated_segments": estimated_count,
        "segment_permutation": {
            "true_to_estimated": true_to_estimated,
            "estimated_to_true": {
                str(key): value for key, value in estimated_to_true.items()
            },
        },
        "segment_reward_normalized_rmse": reward_errors,
        "max_segment_reward_normalized_rmse": max(reward_errors, default=float("inf")),
        "segment_value_normalized_rmse": value_errors,
        "max_segment_value_normalized_rmse": max(value_errors, default=float("inf")),
        "segment_q_normalized_rmse": q_errors,
        "max_segment_q_normalized_rmse": max(q_errors, default=float("inf")),
        "segment_policy": policy_metrics,
        "max_segment_policy_tv": max(
            (metric.tv for metric in policy_metrics),
            default=float("inf"),
        ),
        "segment_counterfactuals": segment_counterfactuals,
        "max_segment_counterfactual_regret": cf_max_regret,
        "aligned_segment_priors": aligned_priors.tolist(),
        "segment_prior_l1": float(np.nansum(prior_abs_error)),
        "segment_prior_max_abs_error": float(np.nanmax(prior_abs_error)),
        "segment_assignment_accuracy": assignment_accuracy,
    }


def _reward_recovery_mask(dgp: KnownTruthDGP) -> jnp.ndarray:
    mask = np.ones(
        (dgp.problem.num_states, dgp.problem.num_actions),
        dtype=bool,
    )
    absorbing = _absorbing_state(dgp)
    if absorbing is not None and 0 <= absorbing < dgp.problem.num_states:
        mask[absorbing, :] = False
    exit_action = _exit_action(dgp)
    if exit_action is not None and 0 <= exit_action < dgp.problem.num_actions:
        mask[:, exit_action] = False
    return jnp.asarray(mask)


def _value_recovery_mask(dgp: KnownTruthDGP) -> jnp.ndarray:
    mask = np.ones(dgp.problem.num_states, dtype=bool)
    absorbing = _absorbing_state(dgp)
    if absorbing is not None and 0 <= absorbing < dgp.problem.num_states:
        mask[absorbing] = False
    return jnp.asarray(mask)


def _q_recovery_mask(dgp: KnownTruthDGP) -> jnp.ndarray:
    mask = np.ones(
        (dgp.problem.num_states, dgp.problem.num_actions),
        dtype=bool,
    )
    absorbing = _absorbing_state(dgp)
    if absorbing is not None and 0 <= absorbing < dgp.problem.num_states:
        mask[absorbing, :] = False
    return jnp.asarray(mask)


def _extract_metadata_reward_table(
    dgp: Any,
    summary: Any,
    *keys: str,
) -> jnp.ndarray | None:
    for key in keys:
        metadata_reward = summary.metadata.get(key)
        if metadata_reward is None:
            continue
        reward = jnp.asarray(metadata_reward)
        if reward.shape == dgp.homogeneous_reward.shape:
            return reward
    return None


def _extract_estimated_reward(
    dgp: Any,
    summary: Any,
    estimated_params: jnp.ndarray,
) -> jnp.ndarray | None:
    """Recover an estimator reward matrix when its output supports one."""

    true_params = jnp.asarray(dgp.homogeneous_parameters)
    metadata_reward = summary.metadata.get("reward_matrix")
    if metadata_reward is not None:
        reward = jnp.asarray(metadata_reward)
        if reward.shape == dgp.homogeneous_reward.shape:
            return reward

    if estimated_params.shape == true_params.shape and true_params.size > 0:
        return dgp.utility().compute(estimated_params)

    if estimated_params.size == dgp.problem.num_states * dgp.problem.num_actions:
        return estimated_params.reshape((dgp.problem.num_states, dgp.problem.num_actions))

    return None


# --- Estimator Contracts ---
Support = Literal["valid", "valid_with_normalization", "diagnostic_only", "unsupported"]


@dataclass(frozen=True)
class EstimatorContract:
    """Estimator requirements and validation targets."""

    name: str
    code_path: str
    paper_paths: tuple[str, ...]
    required_reward_modes: tuple[str, ...]
    required_state_modes: tuple[str, ...]
    requires_transitions: bool
    recovers: tuple[str, ...]
    type_a_support: Support
    type_b_support: Support
    type_c_support: Support
    gpu_recommended: bool = False
    notes: str = ""


ESTIMATOR_CONTRACTS: dict[str, EstimatorContract] = {
    "NFXP": EstimatorContract(
        name="NFXP",
        code_path="src/econirl/estimation/nfxp.py",
        paper_paths=(
            "internal_docs/estimators/nfxp.md",
        ),
        required_reward_modes=("action_dependent",),
        required_state_modes=("low_dim",),
        requires_transitions=True,
        recovers=("theta", "reward", "policy", "value", "Q"),
        type_a_support="valid",
        type_b_support="valid",
        type_c_support="valid",
        notes="Exact structural reference for manageable tabular state spaces.",
    ),
    "CCP": EstimatorContract(
        name="CCP",
        code_path="src/econirl/estimation/ccp.py",
        paper_paths=(
            "internal_docs/estimators/ccp.md",
        ),
        required_reward_modes=("action_dependent",),
        required_state_modes=("low_dim",),
        requires_transitions=True,
        recovers=("theta", "reward", "policy", "value"),
        type_a_support="valid",
        type_b_support="valid",
        type_c_support="valid",
        notes="Use enough NPL iterations before treating this as MLE-like.",
    ),
    "MPEC": EstimatorContract(
        name="MPEC",
        code_path="src/econirl/estimation/mpec.py",
        paper_paths=(
            "internal_docs/estimators/mpec.md",
        ),
        required_reward_modes=("action_dependent",),
        required_state_modes=("low_dim",),
        requires_transitions=True,
        recovers=("theta", "reward", "policy", "value"),
        type_a_support="valid",
        type_b_support="valid",
        type_c_support="valid",
    ),
    "MCE-IRL": EstimatorContract(
        name="MCE-IRL",
        code_path="src/econirl/estimation/mce_irl.py",
        paper_paths=("internal_docs/estimators/mce_irl.md",),
        required_reward_modes=("action_dependent",),
        required_state_modes=("low_dim",),
        requires_transitions=True,
        recovers=("reward", "policy", "occupancy"),
        type_a_support="valid_with_normalization",
        type_b_support="valid_with_normalization",
        type_c_support="valid_with_normalization",
        notes="Reward comparisons require the accepted IRL normalization.",
    ),
    "MCE-IRL Deep": EstimatorContract(
        name="MCE-IRL Deep",
        code_path="src/econirl/estimators/mceirl_neural.py",
        paper_paths=(
            "internal_docs/estimators/deep_mce_irl.md",
        ),
        required_reward_modes=("state_only", "action_dependent", "neural"),
        required_state_modes=("low_dim",),
        requires_transitions=True,
        recovers=("reward", "policy", "value", "Q", "occupancy"),
        type_a_support="valid_with_normalization",
        type_b_support="valid_with_normalization",
        type_c_support="valid_with_normalization",
        gpu_recommended=True,
        notes=(
            "Validated targets are reward-map, policy, value, Q, occupancy, "
            "and counterfactual recovery under known transitions. Neural "
            "network weights themselves are not structural parameters."
        ),
    ),
    "TD-CCP": EstimatorContract(
        name="TD-CCP",
        code_path="src/econirl/estimation/td_ccp.py",
        paper_paths=("internal_docs/estimators/tdccp.md",),
        required_reward_modes=("action_dependent",),
        required_state_modes=("low_dim", "high_dim"),
        requires_transitions=False,
        recovers=("theta", "policy", "value"),
        type_a_support="valid",
        type_b_support="valid",
        type_c_support="valid",
        gpu_recommended=True,
    ),
    "NNES": EstimatorContract(
        name="NNES",
        code_path="src/econirl/estimation/nnes.py",
        paper_paths=("internal_docs/estimators/nnes.md",),
        required_reward_modes=("action_dependent",),
        required_state_modes=("low_dim", "high_dim"),
        requires_transitions=True,
        recovers=("theta", "policy", "value"),
        type_a_support="valid",
        type_b_support="valid",
        type_c_support="valid",
        gpu_recommended=True,
    ),
    "SEES": EstimatorContract(
        name="SEES",
        code_path="src/econirl/estimation/sees.py",
        paper_paths=("internal_docs/estimators/sees.md",),
        required_reward_modes=("action_dependent",),
        required_state_modes=("low_dim", "high_dim"),
        requires_transitions=True,
        recovers=("theta", "policy", "value"),
        type_a_support="valid",
        type_b_support="valid",
        type_c_support="valid",
    ),
    "GLADIUS": EstimatorContract(
        name="GLADIUS",
        code_path="src/econirl/estimation/gladius.py",
        paper_paths=("internal_docs/estimators/gladius.md",),
        required_reward_modes=("action_dependent", "state_only"),
        required_state_modes=("high_dim",),
        requires_transitions=True,
        recovers=("Q", "reward_projection", "policy"),
        type_a_support="valid_with_normalization",
        type_b_support="diagnostic_only",
        type_c_support="diagnostic_only",
        gpu_recommended=True,
        notes=(
            "Known-truth path uses a stable anchor-moment Q loss when known "
            "anchor rewards are supplied; literal paper minimax mode remains "
            "diagnostic because it is numerically unstable in this harness."
        ),
    ),
    "IQ-Learn": EstimatorContract(
        name="IQ-Learn",
        code_path="src/econirl/estimation/iq_learn.py",
        paper_paths=("internal_docs/estimators/iq_learn.md",),
        required_reward_modes=("action_dependent", "state_only"),
        required_state_modes=("low_dim", "high_dim"),
        requires_transitions=True,
        recovers=("Q", "reward", "policy"),
        type_a_support="valid_with_normalization",
        type_b_support="diagnostic_only",
        type_c_support="diagnostic_only",
        gpu_recommended=True,
    ),
    "AIRL": EstimatorContract(
        name="AIRL",
        code_path="src/econirl/estimation/adversarial/airl.py",
        paper_paths=("internal_docs/estimators/airl.md",),
        required_reward_modes=("state_only", "action_dependent"),
        required_state_modes=("low_dim", "high_dim"),
        requires_transitions=False,
        recovers=("reward", "policy"),
        type_a_support="valid_with_normalization",
        type_b_support="valid_with_normalization",
        type_c_support="valid_with_normalization",
        gpu_recommended=True,
        notes=(
            "State-only AIRL matches Fu et al.; action-dependent AIRL must use "
            "an anchor action to be interpretable."
        ),
    ),
    "AIRL-Het": EstimatorContract(
        name="AIRL-Het",
        code_path="src/econirl/estimation/adversarial/airl_het.py",
        paper_paths=(
            "internal_docs/estimators/airl.md",
            "internal_docs/estimators/airl_het.md",
        ),
        required_reward_modes=("state_only", "action_dependent"),
        required_state_modes=("low_dim", "high_dim"),
        requires_transitions=False,
        recovers=("segment_reward", "segment_policy", "segment_membership"),
        type_a_support="valid_with_normalization",
        type_b_support="valid_with_normalization",
        type_c_support="valid_with_normalization",
        gpu_recommended=True,
        notes="Must be run on a latent-segment DGP for the main validation.",
    ),
    "f-IRL": EstimatorContract(
        name="f-IRL",
        code_path="src/econirl/estimation/f_irl.py",
        paper_paths=("src/econirl/estimation/f_irl.py",),
        required_reward_modes=("action_dependent", "state_only"),
        required_state_modes=("low_dim",),
        requires_transitions=True,
        recovers=("occupancy", "reward", "policy"),
        type_a_support="valid_with_normalization",
        type_b_support="diagnostic_only",
        type_c_support="diagnostic_only",
    ),
}


REQUIRED_ESTIMATORS: tuple[str, ...] = tuple(ESTIMATOR_CONTRACTS)


def get_estimator_contract(name: str) -> EstimatorContract:
    try:
        return ESTIMATOR_CONTRACTS[name]
    except KeyError as exc:
        raise KeyError(f"unknown known-truth estimator {name!r}") from exc


# --- Estimator Adapters and Gates ---
@dataclass(frozen=True)
class CompatibilityReport:
    estimator: str
    compatible: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecoveryGate:
    name: str
    value: float | bool
    operator: str
    threshold: float | bool
    passed: bool


@dataclass(frozen=True)
class EstimatorRun:
    estimator: str
    summary: EstimationSummary
    diagnostics: PreEstimationDiagnostics
    compatibility: CompatibilityReport
    metrics: dict[str, Any] = field(default_factory=dict)
    gates: list[RecoveryGate] = field(default_factory=list)


class RecoveryGateFailure(AssertionError):
    """Raised when a non-smoke known-truth run fails hard recovery gates."""


def check_estimator_compatibility(
    estimator_name: str,
    dgp: KnownTruthDGP,
    diagnostics: PreEstimationDiagnostics | None = None,
) -> CompatibilityReport:
    """Check whether an estimator should run on a DGP cell."""

    contract = get_estimator_contract(estimator_name)
    if diagnostics is None:
        diagnostics = run_pre_estimation_diagnostics(dgp)

    errors = list(diagnostics.errors)
    warnings = list(diagnostics.warnings)

    if dgp.config.reward_mode not in contract.required_reward_modes:
        errors.append(
            f"{estimator_name} does not support reward mode {dgp.config.reward_mode}"
        )
    if dgp.config.state_mode not in contract.required_state_modes:
        errors.append(
            f"{estimator_name} does not support state mode {dgp.config.state_mode}"
        )
    if "theta" in contract.recovers and not diagnostics.is_action_dependent:
        errors.append(f"{estimator_name} needs action-dependent features for theta recovery")
    if estimator_name == "NFXP" and dgp.config.heterogeneity != "none":
        errors.append("NFXP main validation requires a homogeneous DGP")
    if estimator_name == "MPEC" and dgp.config.heterogeneity != "none":
        errors.append("MPEC main validation requires a homogeneous DGP")
    if estimator_name == "SEES" and dgp.config.heterogeneity != "none":
        errors.append("SEES main validation requires a homogeneous DGP")
    if estimator_name == "NNES" and dgp.config.heterogeneity != "none":
        errors.append("NNES main validation requires a homogeneous DGP")
    if (
        estimator_name in {"NFXP", "MPEC", "SEES", "NNES"}
        and diagnostics.min_action_share is not None
        and diagnostics.min_action_share < 0.05
    ):
        errors.append(
            f"{estimator_name} requires empirical action support; minimum action share is "
            f"{diagnostics.min_action_share:.3g}"
        )
    if estimator_name == "AIRL-Het" and dgp.config.heterogeneity != "latent_segments":
        errors.append("AIRL-Het main validation requires a latent-segment DGP")
    if estimator_name != "AIRL-Het" and dgp.config.heterogeneity == "latent_segments":
        warnings.append(
            f"{estimator_name} will be evaluated on mixture-average behavior unless segmented"
        )
    if contract.requires_transitions:
        row_error = diagnostics.max_transition_row_error
        if row_error > 1e-6:
            errors.append(f"{estimator_name} requires stochastic transitions")

    return CompatibilityReport(
        estimator=estimator_name,
        compatible=not errors,
        errors=errors,
        warnings=warnings,
    )


class _MCEIRLNeuralKnownTruthAdapter:
    """Adapter from the sklearn-style neural MCE wrapper to EstimationSummary."""

    def __init__(
        self,
        dgp: KnownTruthDGP,
        *,
        smoke: bool = False,
        verbose: bool = False,
    ) -> None:
        self.dgp = dgp
        self.smoke = smoke
        self.verbose = verbose

    def estimate(
        self,
        panel: Panel,
        utility: Any,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
    ) -> EstimationSummary:
        from econirl.estimators.mceirl_neural import MCEIRLNeural

        del utility, initial_params
        start = time.time()
        state_features = jnp.asarray(self.dgp.state_features)
        true_params = jnp.asarray(self.dgp.homogeneous_parameters)
        is_neural_reward_truth = true_params.size == 0
        env_config = getattr(self.dgp.config, "env_config", None)
        truth_action_dependent = bool(
            getattr(
                env_config,
                "action_dependent",
                getattr(self.dgp.config, "reward_mode", "") == "action_dependent",
            )
        )
        reward_type = (
            "state_action"
            if truth_action_dependent
            else "state"
        )
        network_source = env_config if env_config is not None else self.dgp.config
        network_width = (
            int(network_source.network_width)
            if hasattr(network_source, "network_width")
            else (16 if self.smoke else 32)
        )
        network_depth = (
            int(network_source.network_depth)
            if hasattr(network_source, "network_depth")
            else (1 if self.smoke else 2)
        )

        def state_encoder(states: jnp.ndarray) -> jnp.ndarray:
            if problem.state_encoder is not None:
                return problem.state_encoder(jnp.asarray(states, dtype=jnp.int32))
            return state_features[jnp.asarray(states, dtype=jnp.int32)]

        model = MCEIRLNeural(
            n_states=problem.num_states,
            n_actions=problem.num_actions,
            discount=problem.discount_factor,
            reward_type=reward_type,
            reward_hidden_dim=min(network_width, 16) if self.smoke else network_width,
            reward_num_layers=1 if self.smoke else network_depth,
            max_epochs=80 if self.smoke else 1_200,
            lr=3e-3,
            inner_solver="value",
            inner_tol=1e-8,
            inner_max_iter=1_000 if self.smoke else 5_000,
            state_encoder=state_encoder,
            state_dim=problem.state_dim,
            feature_names=list(self.dgp.parameter_names),
            anchor_action=0 if reward_type == "state_action" else None,
            absorbing_state=_absorbing_state(self.dgp),
            seed=2,
            verbose=self.verbose,
        )
        projection_features = (
            np.asarray(self.dgp.feature_matrix)
            if len(self.dgp.parameter_names) == int(self.dgp.feature_matrix.shape[-1])
            else None
        )
        projection_condition_number = None
        projected_parameter_identified = False
        if projection_features is not None:
            projection_condition_number = _safe_condition_number(
                np.asarray(projection_features).reshape(-1, projection_features.shape[-1])
            )
            projected_parameter_identified = bool(
                true_params.size > 0 and projection_condition_number <= 100.0
            )
        model.fit(
            panel,
            transitions=np.asarray(transitions),
            features=projection_features,
        )

        raw_reward_matrix = np.asarray(model.reward_matrix_, dtype=np.float64)
        if model.policy_ is None:
            raise RuntimeError("MCEIRLNeural did not produce a policy")

        projected_parameters = jnp.asarray([], dtype=jnp.float64)
        projected_reward_matrix = None
        projected_solution = None
        if model.coef_ is not None and true_params.size > 0:
            projected_parameters = jnp.asarray(model.coef_, dtype=jnp.float64)
            projected_reward_matrix = np.asarray(
                self.dgp.utility().compute(projected_parameters),
                dtype=np.float64,
            )
            operator = SoftBellmanOperator(
                problem, jnp.asarray(transitions, dtype=jnp.float64)
            )
            projected_solution = value_iteration(
                operator,
                jnp.asarray(projected_reward_matrix, dtype=jnp.float64),
                tol=1e-8,
                max_iter=10_000,
            )

        if is_neural_reward_truth or projected_solution is None:
            reward_matrix = raw_reward_matrix
            policy = jnp.asarray(model.policy_, dtype=jnp.float64)
            value_function = jnp.asarray(model.value_, dtype=jnp.float64)
            reward_validation_target = "raw_neural_reward_matrix"
        else:
            reward_matrix = projected_reward_matrix
            policy = jnp.asarray(projected_solution.policy, dtype=jnp.float64)
            value_function = jnp.asarray(projected_solution.V, dtype=jnp.float64)
            reward_validation_target = "projected_linear_reward_matrix"

        states = np.asarray(panel.get_all_states(), dtype=np.int64)
        actions = np.asarray(panel.get_all_actions(), dtype=np.int64)
        log_probs = np.log(np.clip(np.asarray(policy)[states, actions], 1e-12, 1.0))
        ll = float(log_probs.sum())

        parameter_names = list(self.dgp.parameter_names)
        se_values = None
        if model.se_ is not None:
            se_values = [model.se_.get(name, np.nan) for name in parameter_names]
        return EstimationSummary(
            parameters=projected_parameters,
            parameter_names=parameter_names,
            standard_errors=(
                jnp.asarray(se_values, dtype=jnp.float64)
                if se_values is not None
                else jnp.full_like(projected_parameters, jnp.nan)
            ),
            method="MCE-IRL Deep",
            num_observations=panel.num_observations,
            num_individuals=panel.num_individuals,
            num_periods=max(panel.num_periods_per_individual),
            discount_factor=problem.discount_factor,
            scale_parameter=problem.scale_parameter,
            log_likelihood=ll,
            converged=bool(model.converged_),
            num_iterations=int(model.n_epochs_ or 0),
            convergence_message=(
                "Neural MCE reward training completed"
                if model.converged_
                else "Neural MCE reward training stopped before convergence"
            ),
            value_function=value_function,
            policy=policy,
            estimation_time=time.time() - start,
            metadata={
                "estimator": "MCE-IRL Deep",
                "reward_type": model.reward_type,
                "truth_reward_type": getattr(
                    getattr(self.dgp.config, "env_config", self.dgp.config),
                    "reward_type",
                    getattr(self.dgp.config, "reward_mode", None),
                ),
                "truth_feature_type": getattr(
                    getattr(self.dgp.config, "env_config", self.dgp.config),
                    "feature_type",
                    None,
                ),
                "feature_difference": model.feature_difference_,
                "feature_diff": model.feature_difference_,
                "occupancy_moment_residual": model.occupancy_moment_residual_,
                "reward_matrix": reward_matrix,
                "raw_neural_reward_matrix": raw_reward_matrix,
                "reward_validation_target": reward_validation_target,
                "counterfactual_reward_normalization": "affine",
                "projection_r2": model.projection_r2_,
                "projection_condition_number": projection_condition_number,
                "projected_parameter_identified": projected_parameter_identified,
                "projected_parameters": np.asarray(projected_parameters).tolist(),
                "projected_parameter_names": list(self.dgp.parameter_names),
                "raw_neural_feature_difference": model.feature_difference_,
                "raw_neural_occupancy_moment_residual": model.occupancy_moment_residual_,
                "anchor_action": model.anchor_action,
                "absorbing_state": model.absorbing_state,
                "network_hidden_dim": model.reward_hidden_dim,
                "network_num_layers": model.reward_num_layers,
                "learning_rate": model.lr,
                "seed": model.seed,
            },
        )


def make_estimator(
    estimator_name: str,
    dgp: KnownTruthDGP,
    *,
    smoke: bool = False,
    verbose: bool = False,
) -> Any:
    """Instantiate an estimator with known-truth defaults.

    Smoke settings are intentionally small. Medium-scale runs should use the
    same factory with smoke disabled and estimator-specific config overrides
    added in the run matrix.
    """

    if estimator_name == "NFXP":
        from econirl.estimation.nfxp import NFXPEstimator

        return NFXPEstimator(
            optimizer="BHHH",
            inner_solver="hybrid",
            inner_tol=1e-9 if smoke else 1e-12,
            outer_max_iter=30 if smoke else 500,
            inner_max_iter=2_000 if smoke else 100_000,
            compute_hessian=not smoke,
            verbose=verbose,
        )
    if estimator_name == "CCP":
        from econirl.estimation.ccp import CCPEstimator

        return CCPEstimator(
            num_policy_iterations=3 if smoke else 10,
            outer_max_iter=50 if smoke else 500,
            se_method="asymptotic" if smoke else "robust",
            compute_hessian=not smoke,
            verbose=verbose,
        )
    if estimator_name == "MPEC":
        from econirl.estimation.mpec import MPECConfig, MPECEstimator

        return MPECEstimator(
            config=MPECConfig(
                solver="sqp",
                outer_max_iter=30 if smoke else 200,
                tol=1e-6 if smoke else 1e-8,
                constraint_tol=1e-5 if smoke else 1e-6,
            ),
            se_method="asymptotic" if smoke else "robust",
            compute_hessian=not smoke,
            verbose=verbose,
        )
    if estimator_name == "MCE-IRL":
        from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator

        return MCEIRLEstimator(
            config=MCEIRLConfig(
                optimizer="root",
                outer_max_iter=60 if smoke else 300,
                outer_tol=1e-8,
                inner_max_iter=1_000 if smoke else 10_000,
                compute_se=False,
                verbose=verbose,
            )
        )
    if estimator_name == "MCE-IRL Deep":
        return _MCEIRLNeuralKnownTruthAdapter(
            dgp,
            smoke=smoke,
            verbose=verbose,
        )
    if estimator_name == "TD-CCP":
        from econirl.estimation.td_ccp import TDCCPConfig, TDCCPEstimator

        high_dim = dgp.config.state_mode == "high_dim"
        return TDCCPEstimator(
            config=TDCCPConfig(
                method="semigradient",
                basis_type="encoded",
                basis_dim=3,
                basis_include_rewards=True,
                ccp_smoothing=0.1 if high_dim else 0.01,
                cross_fitting=False,
                robust_se=False,
                outer_max_iter=50 if smoke else 500,
                outer_tol=1e-6 if smoke else 1e-8,
                theta_l2_penalty=1_000.0 if high_dim and not smoke else 0.0,
                compute_se=False,
                verbose=verbose,
            )
        )
    if estimator_name == "NNES":
        from econirl.estimation.nnes import NNESEstimator

        return NNESEstimator(
            hidden_dim=16 if smoke else 32,
            v_epochs=20 if smoke else 500,
            outer_max_iter=30 if smoke else 200,
            outer_tol=1e-4,
            n_outer_iterations=1 if smoke else 3,
            compute_se=False,
            verbose=verbose,
        )
    if estimator_name == "SEES":
        from econirl.estimation.sees import SEESEstimator

        basis_dim = min(dgp.problem.num_states, 6 if smoke else 21)
        if not smoke and dgp.config.state_mode == "high_dim":
            basis_dim = dgp.problem.num_states
        penalty_weight = 100.0
        if not smoke and dgp.config.state_mode == "high_dim":
            penalty_weight = 10_000.0
        return SEESEstimator(
            basis_type="bspline",
            basis_dim=basis_dim,
            penalty_weight=penalty_weight,
            max_iter=40 if smoke else 1_000,
            tol=1e-7,
            compute_se=not smoke,
            verbose=verbose,
        )
    if estimator_name == "GLADIUS":
        from econirl.estimation.gladius import GLADIUSConfig, GLADIUSEstimator

        anchor_action = _exit_action(dgp)
        anchor_rewards = None
        if anchor_action is not None:
            anchor_rewards = tuple(
                float(x) for x in np.asarray(dgp.homogeneous_reward[:, anchor_action])
            )
        return GLADIUSEstimator(
            config=GLADIUSConfig(
                q_hidden_dim=16 if smoke else 128,
                v_hidden_dim=16 if smoke else 128,
                q_num_layers=1 if smoke else 3,
                v_num_layers=1 if smoke else 3,
                max_epochs=10 if smoke else 500,
                batch_size=128 if smoke else 512,
                anchor_action=anchor_action,
                anchor_rewards=anchor_rewards,
                anchor_bellman_mode="anchor_moment",
                compute_se=False,
                verbose=verbose,
            )
        )
    if estimator_name == "IQ-Learn":
        from econirl.estimation.iq_learn import IQLearnConfig, IQLearnEstimator

        neural_q = (
            dgp.config.state_mode == "high_dim"
            and dgp.config.reward_dim == "high"
            and not smoke
        )
        return IQLearnEstimator(
            config=IQLearnConfig(
                q_type="neural" if neural_q else "tabular",
                max_iter=25 if smoke else (800 if neural_q else 500),
                optimizer="adam" if (smoke or neural_q) else "L-BFGS-B",
                learning_rate=0.003 if neural_q else 0.01,
                hidden_dim=64 if neural_q else 32,
                num_layers=2,
                seed=int(dgp.config.seed),
                convergence_tol=1e-4 if not smoke else 1e-6,
                verbose=verbose,
            )
        )
    if estimator_name == "AIRL":
        from econirl.estimation.adversarial.airl import AIRLConfig, AIRLEstimator

        reward_arg = (
            "state_action"
            if dgp.config.reward_mode == "action_dependent"
            else "state"
        )
        paper_identification_case = bool(
            reward_arg == "state"
            and getattr(dgp.config, "exit_action", None) is None
            and getattr(dgp.config, "absorbing_state", None) is None
            and dgp.homogeneous_parameters.size > 0
        )
        reward_type = "linear" if paper_identification_case else "tabular"
        anchor_action = dgp.config.exit_action if reward_arg == "state_action" else None
        absorbing_state = (
            dgp.config.absorbing_state if reward_arg == "state_action" else None
        )
        return AIRLEstimator(
            config=AIRLConfig(
                reward_type=reward_type,
                reward_arg=reward_arg,
                anchor_action=anchor_action,
                absorbing_state=absorbing_state,
                reward_lr=0.02,
                discriminator_steps=3 if smoke else 5,
                policy_step_size=(
                    0.3
                    if smoke
                    else (0.1 if paper_identification_case else 0.3)
                ),
                max_rounds=10 if smoke else 200,
                min_rounds=(
                    3
                    if smoke
                    else (150 if paper_identification_case else 20)
                ),
                convergence_tol=(
                    1e-4
                    if smoke
                    else (0.01 if paper_identification_case else 1e-4)
                ),
                generator_reward="f" if paper_identification_case else "recovered",
                generator_max_iter=500 if smoke else 5_000,
                compute_se=False,
                verbose=verbose,
            )
        )
    if estimator_name == "AIRL-Het":
        from econirl.estimation.adversarial.airl_het import AIRLHetConfig, AIRLHetEstimator

        content_cell = isinstance(dgp.config, ContentHeterogeneityKnownTruthConfig)
        return AIRLHetEstimator(
            config=AIRLHetConfig(
                num_segments=dgp.config.num_segments,
                exit_action=dgp.config.exit_action,
                absorbing_state=dgp.config.absorbing_state,
                reward_type="linear",
                reward_lr=(
                    0.001
                    if content_cell
                    else 0.02
                ),
                discriminator_steps=2 if content_cell else (3 if smoke else 5),
                policy_step_size=0.1 if content_cell else (0.3 if smoke else 0.1),
                generator_reward="f",
                max_airl_rounds=3 if content_cell else (5 if smoke else 80),
                min_airl_rounds=1 if content_cell else (2 if smoke else 20),
                max_em_iterations=8 if content_cell else (3 if smoke else 30),
                airl_convergence_tol=1e-4 if smoke else 0.01,
                em_convergence_tol=1e-3,
                prior_min=0.05,
                prior_damping=0.8 if content_cell else 0.2,
                consistency_weight=1.0 if content_cell else 0.1,
                antisymmetric_init=not content_cell,
                initialization="behavioral_anchor" if content_cell else "random",
                initialization_smoothing=1.0,
                initialization_l2_penalty=10.0 if content_cell else 0.0,
                generator_max_iter=500 if smoke else 5_000,
                verbose=verbose,
            )
        )
    if estimator_name == "f-IRL":
        from econirl.estimation.f_irl import FIRLEstimator

        paper_state_marginal = (
            dgp.config.reward_mode == "state_only" and _exit_action(dgp) is None
        )
        return FIRLEstimator(
            f_divergence="fkl",
            lr=0.20 if paper_state_marginal else 0.50,
            max_iter=20 if smoke else (250 if paper_state_marginal else 500),
            inner_max_iter=500 if smoke else 5_000,
            marginal_space="state" if paper_state_marginal else "state_action",
            reward_scope="state" if paper_state_marginal else "state_action",
            selection_metric="occupancy_l1" if paper_state_marginal else "log_likelihood",
            compute_se=False,
            verbose=verbose,
        )
    raise KeyError(f"unknown known-truth estimator {estimator_name!r}")


def run_estimator(
    estimator_name: str,
    dgp: KnownTruthDGP,
    panel: Panel,
    *,
    smoke: bool = False,
    verbose: bool = False,
    initial_params: jnp.ndarray | None = None,
    enforce_gates: bool | None = None,
) -> EstimatorRun:
    """Run one estimator after compatibility and pre-estimation checks."""

    if enforce_gates is None:
        enforce_gates = not smoke

    diagnostics = run_pre_estimation_diagnostics(dgp, panel)
    compatibility = check_estimator_compatibility(estimator_name, dgp, diagnostics)
    if not compatibility.compatible:
        joined = "; ".join(compatibility.errors)
        raise ValueError(f"{estimator_name} is incompatible with this DGP: {joined}")

    estimator = make_estimator(estimator_name, dgp, smoke=smoke, verbose=verbose)
    if initial_params is None and estimator_name in {"NFXP", "NNES", "SEES"}:
        perturbation_scale = 0.02
        if estimator_name == "SEES" and dgp.config.state_mode == "high_dim":
            perturbation_scale = 0.001
        initial_params = known_truth_initial_params(
            dgp,
            perturbation_scale=perturbation_scale,
        )
    summary = estimator.estimate(
        panel=panel,
        utility=dgp.utility(),
        problem=dgp.problem,
        transitions=dgp.transitions,
        initial_params=initial_params,
    )
    metrics = evaluate_estimator_against_truth(dgp, summary, panel=panel)
    gates = recovery_gates(estimator_name, summary, metrics, smoke=smoke)
    if enforce_gates:
        failed = [gate for gate in gates if not gate.passed]
        if failed:
            details = "; ".join(
                f"{gate.name}={gate.value} {gate.operator} {gate.threshold}"
                for gate in failed
            )
            raise RecoveryGateFailure(
                f"{estimator_name} failed known-truth recovery gates: {details}"
            )
    return EstimatorRun(
        estimator=estimator_name,
        summary=summary,
        diagnostics=diagnostics,
        compatibility=compatibility,
        metrics=metrics,
        gates=gates,
    )


def contract_for(estimator_name: str) -> EstimatorContract:
    return get_estimator_contract(estimator_name)


def known_truth_initial_params(
    dgp: KnownTruthDGP,
    *,
    perturbation_scale: float = 0.02,
) -> jnp.ndarray:
    """Deterministic known-truth starting point for structural validation runs."""

    truth = np.asarray(dgp.homogeneous_parameters, dtype=np.float64)
    rng = np.random.default_rng(dgp.config.seed + 7_919)
    scale = np.maximum(np.abs(truth), 1.0)
    perturbation = perturbation_scale * scale * rng.normal(size=truth.shape)
    return jnp.array(truth + perturbation, dtype=jnp.float32)


def recovery_gates(
    estimator_name: str,
    summary: EstimationSummary,
    metrics: dict[str, Any],
    *,
    smoke: bool,
) -> list[RecoveryGate]:
    """Return hard known-truth recovery gates for non-smoke validation."""

    if smoke:
        return []
    if estimator_name == "CCP":
        se_available = summary.standard_errors is not None and bool(
            jnp.all(jnp.isfinite(jnp.asarray(summary.standard_errors)))
        )
        checks = [
            _numeric_gate(
                "npl_iterations",
                float(summary.num_iterations),
                ">=",
                5.0,
            ),
            _bool_gate("standard_errors_finite", se_available, True),
            _numeric_gate(
                "parameter_cosine",
                metrics["parameters"].cosine_similarity,
                ">=",
                0.98,
            ),
            _numeric_gate(
                "parameter_relative_rmse",
                metrics["parameters"].relative_rmse,
                "<=",
                0.15,
            ),
            _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.03),
            _numeric_gate("value_rmse", metrics["value_rmse"], "<=", 0.10),
            _numeric_gate("q_rmse", metrics["q_rmse"], "<=", 0.10),
        ]
        for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
            checks.append(
                _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.05)
            )
        return checks

    if estimator_name == "MPEC":
        se_available = summary.standard_errors is not None and bool(
            jnp.all(jnp.isfinite(jnp.asarray(summary.standard_errors)))
        )
        constraint_violation = float(
            summary.metadata.get("final_constraint_violation", float("inf"))
        )
        checks = [
            _bool_gate("converged", bool(summary.converged), True),
            _numeric_gate(
                "constraint_violation",
                constraint_violation,
                "<=",
                1e-6,
            ),
            _bool_gate("standard_errors_finite", se_available, True),
            _numeric_gate(
                "parameter_cosine",
                metrics["parameters"].cosine_similarity,
                ">=",
                0.98,
            ),
            _numeric_gate(
                "parameter_relative_rmse",
                metrics["parameters"].relative_rmse,
                "<=",
                0.15,
            ),
            _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.03),
            _numeric_gate("value_rmse", metrics["value_rmse"], "<=", 0.10),
            _numeric_gate("q_rmse", metrics["q_rmse"], "<=", 0.10),
        ]
        for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
            checks.append(
                _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.05)
            )
        return checks

    if estimator_name == "SEES":
        se_available = summary.standard_errors is not None and bool(
            jnp.all(jnp.isfinite(jnp.asarray(summary.standard_errors)))
        )
        bellman_violation = float(summary.metadata.get("bellman_violation", float("inf")))
        checks = [
            _numeric_gate("bellman_violation", bellman_violation, "<=", 0.05),
            _bool_gate("standard_errors_finite", se_available, True),
            _numeric_gate(
                "parameter_cosine",
                metrics["parameters"].cosine_similarity,
                ">=",
                0.99,
            ),
            _numeric_gate(
                "parameter_relative_rmse",
                metrics["parameters"].relative_rmse,
                "<=",
                0.15,
            ),
            _numeric_gate("reward_rmse", metrics["reward_rmse"], "<=", 0.03),
            _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.02),
            _numeric_gate("value_rmse", metrics["value_rmse"], "<=", 0.10),
            _numeric_gate("q_rmse", metrics["q_rmse"], "<=", 0.10),
        ]
        for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
            checks.append(
                _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.01)
            )
        return checks

    if estimator_name == "NNES":
        v_losses = summary.metadata.get("v_loss_per_outer", ())
        final_v_loss = float(v_losses[-1]) if v_losses else float("inf")
        checks = [
            _numeric_gate(
                "npl_outer_iterations",
                float(summary.metadata.get("n_outer_iterations", 0)),
                ">=",
                3.0,
            ),
            _numeric_gate("final_v_loss", final_v_loss, "<=", 0.05),
            _numeric_gate(
                "parameter_cosine",
                metrics["parameters"].cosine_similarity,
                ">=",
                0.95,
            ),
            _numeric_gate(
                "parameter_relative_rmse",
                metrics["parameters"].relative_rmse,
                "<=",
                0.30,
            ),
            _numeric_gate("reward_rmse", metrics["reward_rmse"], "<=", 0.08),
            _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.03),
            _numeric_gate("value_rmse", metrics["value_rmse"], "<=", 0.20),
            _numeric_gate("q_rmse", metrics["q_rmse"], "<=", 0.20),
        ]
        for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
            checks.append(
                _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.05)
            )
        return checks

    if estimator_name == "MCE-IRL":
        feature_residual = float(
            summary.metadata.get(
                "feature_difference",
                summary.metadata.get("feature_diff", float("inf")),
            )
        )
        occupancy_moment_residual = float(
            summary.metadata.get("occupancy_moment_residual", float("inf"))
        )
        checks = [
            _bool_gate("converged", bool(summary.converged), True),
            _numeric_gate("feature_residual", feature_residual, "<=", 0.02),
            _numeric_gate(
                "occupancy_moment_residual",
                occupancy_moment_residual,
                "<=",
                0.02,
            ),
            _numeric_gate(
                "reward_normalized_rmse",
                metrics["reward_normalized_rmse"],
                "<=",
                0.10,
            ),
            _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.03),
            _numeric_gate(
                "value_normalized_rmse",
                metrics["value_normalized_rmse"],
                "<=",
                0.10,
            ),
            _numeric_gate(
                "q_normalized_rmse",
                metrics["q_normalized_rmse"],
                "<=",
                0.10,
            ),
        ]
        for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
            checks.append(
                _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.05)
            )
        return checks

    if estimator_name == "MCE-IRL Deep":
        occupancy_moment_residual = float(
            summary.metadata.get("occupancy_moment_residual", float("inf"))
        )
        neural_reward_target = (
            summary.metadata.get("reward_validation_target")
            == "raw_neural_reward_matrix"
            and metrics["parameters"] is None
        )
        reward_threshold = 0.15 if neural_reward_target else 0.10
        policy_threshold = 0.05 if neural_reward_target else 0.03
        value_threshold = 0.15 if neural_reward_target else 0.10
        q_threshold = 0.15 if neural_reward_target else 0.10
        cf_threshold = 0.08
        checks = [
            _bool_gate("converged", bool(summary.converged), True),
            _numeric_gate(
                "occupancy_moment_residual",
                occupancy_moment_residual,
                "<=",
                0.03 if neural_reward_target else 0.02,
            ),
            _numeric_gate(
                "reward_normalized_rmse",
                metrics["reward_normalized_rmse"],
                "<=",
                reward_threshold,
            ),
            _numeric_gate("policy_tv", metrics["policy"].tv, "<=", policy_threshold),
            _numeric_gate(
                "value_normalized_rmse",
                metrics["value_normalized_rmse"],
                "<=",
                value_threshold,
            ),
            _numeric_gate(
                "q_normalized_rmse",
                metrics["q_normalized_rmse"],
                "<=",
                q_threshold,
            ),
        ]
        parameter_gates_apply = bool(
            metrics["parameters"] is not None
            and summary.metadata.get("projected_parameter_identified", True)
        )
        if parameter_gates_apply:
            checks.insert(
                2,
                _numeric_gate(
                    "projected_parameter_cosine",
                    metrics["parameters"].cosine_similarity,
                    ">=",
                    0.98,
                ),
            )
            checks.insert(
                3,
                _numeric_gate(
                    "projected_parameter_relative_rmse",
                    metrics["parameters"].relative_rmse,
                    "<=",
                    0.15,
                ),
            )
        for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
            checks.append(
                _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", cf_threshold)
            )
        return checks

    if estimator_name == "AIRL":
        checks = [
            _bool_gate("converged", bool(summary.converged), True),
            _numeric_gate(
                "reward_normalized_rmse",
                metrics["reward_normalized_rmse"],
                "<=",
                0.15,
            ),
            _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.05),
            _numeric_gate(
                "value_normalized_rmse",
                metrics["value_normalized_rmse"],
                "<=",
                0.15,
            ),
            _numeric_gate(
                "q_normalized_rmse",
                metrics["q_normalized_rmse"],
                "<=",
                0.15,
            ),
        ]
        for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
            checks.append(
                _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.08)
            )
        return checks

    if estimator_name == "AIRL-Het":
        assignment_accuracy = metrics["segment_assignment_accuracy"]
        if assignment_accuracy is None:
            assignment_accuracy = float("-inf")
        checks = [
            _bool_gate(
                "num_segments_match",
                metrics["num_estimated_segments"] == metrics["num_true_segments"],
                True,
            ),
            _bool_gate("converged", bool(summary.converged), True),
            _numeric_gate(
                "segment_prior_l1",
                metrics["segment_prior_l1"],
                "<=",
                0.35,
            ),
            _numeric_gate(
                "segment_assignment_accuracy",
                assignment_accuracy,
                ">=",
                0.70,
            ),
            _numeric_gate(
                "max_segment_reward_normalized_rmse",
                metrics["max_segment_reward_normalized_rmse"],
                "<=",
                0.30,
            ),
            _numeric_gate(
                "max_segment_policy_tv",
                metrics["max_segment_policy_tv"],
                "<=",
                0.12,
            ),
            _numeric_gate(
                "max_segment_value_normalized_rmse",
                metrics["max_segment_value_normalized_rmse"],
                "<=",
                0.30,
            ),
            _numeric_gate(
                "max_segment_q_normalized_rmse",
                metrics["max_segment_q_normalized_rmse"],
                "<=",
                0.30,
            ),
        ]
        for kind, regret in sorted(metrics["max_segment_counterfactual_regret"].items()):
            checks.append(_numeric_gate(f"{kind}_max_regret", regret, "<=", 0.12))
        return checks

    if estimator_name == "TD-CCP":
        checks = [
            _bool_gate("converged", bool(summary.converged), True),
            _numeric_gate(
                "parameter_cosine",
                metrics["parameters"].cosine_similarity,
                ">=",
                0.99,
            ),
            _numeric_gate(
                "parameter_relative_rmse",
                metrics["parameters"].relative_rmse,
                "<=",
                0.15,
            ),
            _numeric_gate("reward_rmse", metrics["reward_rmse"], "<=", 0.03),
            _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.02),
            _numeric_gate("value_rmse", metrics["value_rmse"], "<=", 0.10),
            _numeric_gate("q_rmse", metrics["q_rmse"], "<=", 0.10),
        ]
        for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
            checks.append(
                _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.01)
            )
        return checks

    if estimator_name == "f-IRL":
        occupancy_l1 = float(summary.metadata.get("occupancy_l1", float("inf")))
        reward_range = float(summary.metadata.get("reward_range", 0.0))
        paper_state_marginal = (
            summary.metadata.get("marginal_space") == "state"
            and summary.metadata.get("reward_scope") == "state"
        )
        if paper_state_marginal:
            checks = [
                _bool_gate("converged", bool(summary.converged), True),
                _numeric_gate("state_marginal_l1", occupancy_l1, "<=", 0.08),
                _numeric_gate("reward_range", reward_range, ">=", 1e-3),
                _numeric_gate(
                    "reward_normalized_rmse",
                    metrics["reward_normalized_rmse"],
                    "<=",
                    0.30,
                ),
                _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.08),
                _numeric_gate(
                    "value_normalized_rmse",
                    metrics["value_normalized_rmse"],
                    "<=",
                    0.30,
                ),
                _numeric_gate(
                    "q_normalized_rmse",
                    metrics["q_normalized_rmse"],
                    "<=",
                    0.30,
                ),
            ]
            for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
                checks.append(
                    _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.05)
                )
            return checks
        checks = [
            _bool_gate("converged", bool(summary.converged), True),
            _numeric_gate("occupancy_l1", occupancy_l1, "<=", 0.40),
            _numeric_gate("reward_range", reward_range, ">=", 1e-3),
        ]
        return checks

    if estimator_name == "GLADIUS":
        final_loss = float(summary.metadata.get("final_loss", float("inf")))
        checks = [
            _bool_gate("converged", bool(summary.converged), True),
            _numeric_gate("final_loss", final_loss, "<=", 2.0),
            _numeric_gate(
                "parameter_cosine",
                metrics["parameters"].cosine_similarity,
                ">=",
                0.90,
            ),
            _numeric_gate(
                "parameter_relative_rmse",
                metrics["parameters"].relative_rmse,
                "<=",
                0.50,
            ),
            _numeric_gate(
                "raw_bellman_reward_normalized_rmse",
                metrics["raw_bellman_reward_normalized_rmse"],
                "<=",
                0.30,
            ),
            _numeric_gate(
                "projected_reward_normalized_rmse",
                metrics["projected_reward_normalized_rmse"],
                "<=",
                0.30,
            ),
            _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.12),
            _numeric_gate(
                "value_normalized_rmse",
                metrics["value_normalized_rmse"],
                "<=",
                0.30,
            ),
            _numeric_gate(
                "q_normalized_rmse",
                metrics["q_normalized_rmse"],
                "<=",
                0.30,
            ),
        ]
        for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
            checks.append(
                _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.12)
            )
        return checks

    if estimator_name == "IQ-Learn":
        checks = [
            _bool_gate("converged", bool(summary.converged), True),
            _numeric_gate(
                "expert_state_coverage",
                float(summary.metadata.get("expert_state_coverage", 0.0)),
                ">=",
                1.0,
            ),
            _numeric_gate(
                "expert_state_action_coverage",
                float(summary.metadata.get("expert_state_action_coverage", 0.0)),
                ">=",
                0.95,
            ),
            _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.05),
            _numeric_gate(
                "raw_bellman_reward_normalized_rmse",
                metrics["raw_bellman_reward_normalized_rmse"],
                "<=",
                0.10,
            ),
            _numeric_gate(
                "projected_reward_normalized_rmse",
                metrics["projected_reward_normalized_rmse"],
                "<=",
                0.10,
            ),
            _numeric_gate(
                "value_normalized_rmse",
                metrics["value_normalized_rmse"],
                "<=",
                0.10,
            ),
            _numeric_gate(
                "q_normalized_rmse",
                metrics["q_normalized_rmse"],
                "<=",
                0.10,
            ),
        ]
        for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
            checks.append(
                _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.05)
            )
        return checks

    if estimator_name != "NFXP":
        raise NotImplementedError(
            f"No hard non-smoke recovery gates are implemented for {estimator_name}"
        )

    checks = [
        _bool_gate("converged", bool(summary.converged), True),
        _numeric_gate(
            "parameter_cosine",
            metrics["parameters"].cosine_similarity,
            ">=",
            0.98,
        ),
        _numeric_gate(
            "parameter_relative_rmse",
            metrics["parameters"].relative_rmse,
            "<=",
            0.15,
        ),
        _numeric_gate("policy_tv", metrics["policy"].tv, "<=", 0.03),
        _numeric_gate("value_rmse", metrics["value_rmse"], "<=", 0.10),
    ]
    for kind, cf_metrics in sorted(metrics["counterfactuals"].items()):
        checks.append(
            _numeric_gate(f"{kind}_regret", cf_metrics.regret, "<=", 0.05)
        )
    return checks


def _numeric_gate(
    name: str,
    value: float,
    operator: str,
    threshold: float,
) -> RecoveryGate:
    if operator == "<=":
        passed = value <= threshold
    elif operator == ">=":
        passed = value >= threshold
    else:
        raise ValueError(f"unknown gate operator {operator!r}")
    return RecoveryGate(
        name=name,
        value=float(value),
        operator=operator,
        threshold=float(threshold),
        passed=bool(passed),
    )


def _bool_gate(name: str, value: bool, threshold: bool) -> RecoveryGate:
    return RecoveryGate(
        name=name,
        value=bool(value),
        operator="is",
        threshold=bool(threshold),
        passed=bool(value) == bool(threshold),
    )


# --- Artifacts ---
def stable_hash(payload: Any, length: int = 12) -> str:
    encoded = json.dumps(to_jsonable(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return value


# --- Cell Matrix ---
@dataclass(frozen=True)
class KnownTruthCell:
    cell_id: str
    dgp_config: (
        KnownTruthDGPConfig
        | ShapeshifterKnownTruthConfig
        | ContentHeterogeneityKnownTruthConfig
    )
    simulation_config: SimulationConfig = field(default_factory=SimulationConfig)
    description: str = ""


DEFAULT_CELLS: tuple[KnownTruthCell, ...] = (
    KnownTruthCell(
        cell_id="canonical_low_action",
        dgp_config=KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=20,
            seed=42,
        ),
        simulation_config=SimulationConfig(n_individuals=2_000, n_periods=80, seed=42),
        description="Universal DGP preset: low-dimensional action-dependent structural benchmark.",
    ),
    KnownTruthCell(
        cell_id="canonical_low_state_only",
        dgp_config=KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="state_only",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=20,
            seed=43,
        ),
        description="Universal DGP preset: state-only reward benchmark for AIRL-style assumptions.",
    ),
    KnownTruthCell(
        cell_id="airl_paper_identification",
        dgp_config=ShapeshifterKnownTruthConfig(
            env_config=ShapeshifterConfig(
                num_states=16,
                num_actions=4,
                num_features=4,
                reward_type="linear",
                feature_type="linear",
                action_dependent=False,
                stochastic_transitions=False,
                stochastic_rewards=False,
                num_periods=None,
                discount_factor=0.95,
                scale_parameter=1.0,
                state_dim=1,
                reward_scale=1.0,
                seed=1710,
            ),
        ),
        simulation_config=SimulationConfig(n_individuals=300, n_periods=80, seed=1711),
        description=(
            "AIRL paper-assumption diagnostic: deterministic transitions, no "
            "exit/absorbing anchor, full observability, and a state-only "
            "reward tiled across every action."
        ),
    ),
    KnownTruthCell(
        cell_id="f_irl_paper_state_marginal",
        dgp_config=ShapeshifterKnownTruthConfig(
            env_config=ShapeshifterConfig(
                num_states=8,
                num_actions=3,
                num_features=3,
                reward_type="linear",
                feature_type="linear",
                action_dependent=False,
                stochastic_transitions=False,
                stochastic_rewards=False,
                num_periods=None,
                discount_factor=0.95,
                scale_parameter=1.0,
                state_dim=1,
                reward_scale=1.0,
                seed=4400,
            ),
        ),
        simulation_config=SimulationConfig(n_individuals=1_000, n_periods=100, seed=4401),
        description=(
            "f-IRL paper-side state-marginal cell: deterministic dynamics, "
            "full observability, no exit/absorbing anchor, and a state-only "
            "reward tiled across every action."
        ),
    ),
    KnownTruthCell(
        cell_id="canonical_high_action",
        dgp_config=KnownTruthDGPConfig(
            state_mode="high_dim",
            reward_mode="action_dependent",
            reward_dim="high",
            heterogeneity="none",
            num_regular_states=80,
            high_state_dim=16,
            high_reward_features=32,
            seed=44,
        ),
        simulation_config=SimulationConfig(n_individuals=2_000, n_periods=80, seed=44),
        description="Universal DGP preset: high-dimensional state and reward stress benchmark.",
    ),
    KnownTruthCell(
        cell_id="gladius_paper_high_state",
        dgp_config=KnownTruthDGPConfig(
            state_mode="high_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=20,
            high_state_dim=64,
            seed=144,
            initial_state_mode="uniform_regular",
            transition_noise=0.05,
        ),
        simulation_config=SimulationConfig(n_individuals=1_000, n_periods=100, seed=145),
        description=(
            "GLADIUS paper-side high-dimensional-state cell: many nuisance "
            "state controls, a low-dimensional action-dependent reward basis, "
            "and an exit-action reward anchor."
        ),
    ),
    KnownTruthCell(
        cell_id="gladius_paper_high_state_scaled",
        dgp_config=KnownTruthDGPConfig(
            state_mode="high_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=20,
            high_state_dim=128,
            seed=244,
            initial_state_mode="uniform_regular",
            transition_noise=0.05,
        ),
        simulation_config=SimulationConfig(n_individuals=1_000, n_periods=100, seed=245),
        description=(
            "GLADIUS scaled paper-side cell: doubled nuisance state dimension "
            "with the same low-dimensional anchored reward structure."
        ),
    ),
    KnownTruthCell(
        cell_id="canonical_latent_segments",
        dgp_config=KnownTruthDGPConfig(
            state_mode="high_dim",
            reward_mode="action_dependent",
            reward_dim="high",
            heterogeneity="latent_segments",
            num_regular_states=60,
            high_state_dim=12,
            high_reward_features=24,
            num_segments=2,
            seed=45,
        ),
        description="Universal DGP preset: latent-segment benchmark for heterogeneous estimators.",
    ),
    KnownTruthCell(
        cell_id="airl_het_paper_identification",
        dgp_config=ContentHeterogeneityKnownTruthConfig(
            num_chapters=5,
            wait_bins=3,
            price_levels=2,
            quality_levels=2,
            books_per_user=4,
            discount_factor=0.92,
            scale_parameter=0.85,
            seed=4506,
        ),
        simulation_config=SimulationConfig(n_individuals=800, n_periods=16, seed=4507),
        description=(
            "AIRL-Het serialized-content identification cell: two latent "
            "segments, repeated books per user, pay/wait/exit actions, an "
            "exit reward anchor, deterministic chapter transitions, and "
            "known finite reward features."
        ),
    ),
    KnownTruthCell(
        cell_id="deep_mce_neural_reward",
        dgp_config=ShapeshifterKnownTruthConfig(
            env_config=ShapeshifterConfig(
                num_states=32,
                num_actions=3,
                num_features=4,
                reward_type="neural",
                feature_type="linear",
                action_dependent=True,
                stochastic_transitions=True,
                stochastic_rewards=False,
                num_periods=None,
                discount_factor=0.95,
                scale_parameter=1.0,
                state_dim=1,
                network_width=32,
                network_depth=2,
                reward_scale=3.0,
                seed=146,
            ),
        ),
        simulation_config=SimulationConfig(n_individuals=2_000, n_periods=80, seed=146),
        description=(
            "Deep MCE primary Shapeshifter cell: frozen nonlinear neural "
            "reward with action 0 anchored to zero, linear features, known "
            "stochastic transitions, and full state-action reward-map validation."
        ),
    ),
    KnownTruthCell(
        cell_id="deep_mce_neural_features",
        dgp_config=ShapeshifterKnownTruthConfig(
            env_config=ShapeshifterConfig(
                num_states=32,
                num_actions=3,
                num_features=4,
                reward_type="linear",
                feature_type="neural",
                action_dependent=True,
                stochastic_transitions=True,
                stochastic_rewards=False,
                num_periods=None,
                discount_factor=0.95,
                scale_parameter=1.0,
                state_dim=1,
                network_width=32,
                network_depth=2,
                reward_scale=3.0,
                seed=147,
            ),
        ),
        simulation_config=SimulationConfig(n_individuals=2_000, n_periods=80, seed=147),
        description=(
            "Deep MCE Shapeshifter finite-theta cell: frozen neural features "
            "with action 0 anchored to zero and a linear reward, retained to "
            "test structural theta recovery when the feature matrix is the "
            "true reward basis."
        ),
    ),
    KnownTruthCell(
        cell_id="deep_mce_neural_reward_features",
        dgp_config=ShapeshifterKnownTruthConfig(
            env_config=ShapeshifterConfig(
                num_states=32,
                num_actions=3,
                num_features=4,
                reward_type="neural",
                feature_type="neural",
                action_dependent=True,
                stochastic_transitions=True,
                stochastic_rewards=False,
                num_periods=None,
                discount_factor=0.95,
                scale_parameter=1.0,
                state_dim=1,
                network_width=32,
                network_depth=2,
                reward_scale=3.0,
                seed=148,
            ),
        ),
        simulation_config=SimulationConfig(n_individuals=2_000, n_periods=80, seed=148),
        description=(
            "Deep MCE hard Shapeshifter stress test: frozen neural reward and "
            "frozen neural features with action 0 anchored to zero. Failures "
            "are retained as failed gates, not converted into success prose."
        ),
    ),
)


CELL_ALIASES: dict[str, str] = {
    "low_state_action_reward": "canonical_low_action",
    "low_state_state_only_reward": "canonical_low_state_only",
    "airl_original_conditions": "airl_paper_identification",
    "f_irl_original_conditions": "f_irl_paper_state_marginal",
    "high_state_high_reward": "canonical_high_action",
    "gladius_paper": "gladius_paper_high_state",
    "gladius_high_state": "gladius_paper_high_state",
    "gladius_scaled": "gladius_paper_high_state_scaled",
    "latent_segments": "canonical_latent_segments",
    "deep_mce_state_reward_32": "deep_mce_neural_reward",
}


def get_cell(cell_id: str) -> KnownTruthCell:
    resolved = CELL_ALIASES.get(cell_id, cell_id)
    for cell in DEFAULT_CELLS:
        if cell.cell_id == resolved:
            if resolved == cell_id:
                return cell
            return replace(cell, cell_id=cell_id)
    raise KeyError(f"unknown known-truth cell {cell_id!r}")


# --- CLI entrypoints ---

def run_cell_estimator(
    cell_id: str,
    estimator: str,
    output_dir: Path,
    *,
    smoke: bool = False,
    show_progress: bool = False,
    verbose: bool = False,
) -> Path:
    """Run one estimator on one known-truth cell and write result.json."""

    cell = get_cell(cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)
    simulation_config = _simulation_config(cell.simulation_config, smoke, show_progress)
    panel = simulate_known_truth_panel(dgp, simulation_config)
    config_hash = stable_hash(
        {
            "cell": cell.dgp_config.to_dict(),
            "simulation": simulation_config,
            "estimator": estimator,
            "smoke": smoke,
        }
    )
    run_dir = output_dir / f"{cell_id}_{estimator.lower().replace('-', '')}_{config_hash}"
    try:
        result = run_estimator(estimator, dgp, panel, smoke=smoke, verbose=verbose)
        payload = {
            "cell": cell,
            "simulation": simulation_config,
            "estimator": estimator,
            "diagnostics": result.diagnostics,
            "compatibility": result.compatibility,
            "summary": _summary_payload(result.summary),
            "metrics": result.metrics,
            "gates": result.gates,
            "exception": None,
        }
    except Exception:
        payload = {
            "cell": cell,
            "simulation": simulation_config,
            "estimator": estimator,
            "exception": traceback.format_exc(),
        }
        write_json(run_dir / "result.json", payload)
        raise
    write_json(run_dir / "result.json", payload)
    return run_dir


def run_oracle_cell(cell_id: str, output_dir: Path) -> Path:
    """Build one known-truth cell and write its oracle artifacts."""

    cell = get_cell(cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)
    panel = simulate_known_truth_panel(dgp, cell.simulation_config)
    diagnostics = run_pre_estimation_diagnostics(dgp, panel)
    solutions = [solve_known_truth(dgp, segment_index=g) for g in range(dgp.num_segments)]
    counterfactuals = {
        kind: [
            solve_counterfactual_oracle(dgp, kind, segment_index=g)
            for g in range(dgp.num_segments)
        ]
        for kind in ("type_a", "type_b", "type_c")
    }

    config_hash = stable_hash(cell.dgp_config.to_dict())
    cell_dir = output_dir / f"{cell.cell_id}_{config_hash}"
    write_json(
        cell_dir / "oracle.json",
        {
            "cell": cell,
            "diagnostics": diagnostics,
            "solutions": solutions,
            "counterfactuals": counterfactuals,
            "panel_metadata": panel.metadata,
        },
    )
    return cell_dir


def _summary_payload(summary: EstimationSummary) -> dict[str, Any]:
    return {
        "method": summary.method,
        "converged": bool(summary.converged),
        "num_iterations": int(summary.num_iterations),
        "log_likelihood": (
            None if summary.log_likelihood is None else float(summary.log_likelihood)
        ),
        "parameters": np.asarray(summary.parameters).tolist(),
        "parameter_names": list(summary.parameter_names),
        "standard_errors": np.asarray(summary.standard_errors).tolist(),
        "num_observations": int(summary.num_observations),
        "num_individuals": int(summary.num_individuals),
        "estimation_time": float(summary.estimation_time),
        "convergence_message": summary.convergence_message,
        "goodness_of_fit": summary.goodness_of_fit,
        "metadata": summary.metadata,
        "value_function": (
            None
            if summary.value_function is None
            else np.asarray(summary.value_function).tolist()
        ),
        "policy": None if summary.policy is None else np.asarray(summary.policy).tolist(),
    }


def _simulation_config(
    base: SimulationConfig,
    smoke: bool,
    show_progress: bool,
) -> SimulationConfig:
    if smoke:
        return replace(
            base,
            n_individuals=min(base.n_individuals, 40),
            n_periods=min(base.n_periods, 20),
            show_progress=show_progress,
        )
    return replace(base, show_progress=show_progress)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", default="canonical_low_action")
    parser.add_argument("--estimator")
    parser.add_argument("--output-dir", default="outputs/known_truth")
    parser.add_argument("--oracles", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.estimator:
        path = run_cell_estimator(
            args.cell_id,
            args.estimator,
            output_dir,
            smoke=args.smoke,
            show_progress=args.show_progress,
            verbose=args.verbose,
        )
        print(path)
        return

    cell_ids = [cell.cell_id for cell in DEFAULT_CELLS] if args.cell_id == "all" else [args.cell_id]
    for cell_id in cell_ids:
        path = run_oracle_cell(cell_id, output_dir)
        print(path)


if __name__ == "__main__":
    main()
