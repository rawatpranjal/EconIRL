"""Grid-topology form factory.

Builds :class:`~econirl.forms.base.Form` objects for N x N gridworld
problems.

Reward forms:
- ``"linear"``: :class:`~econirl.environments.gridworld.GridworldEnvironment`
  -- deterministic grid, 5 actions, 3 features (step_penalty,
  terminal_reward, distance_weight).
- ``"nonlinear"``: :class:`~econirl.environments.objectworld.ObjectworldEnvironment`
  -- grid with colored objects; reward is a nonlinear function of distances
  to objects.  True parameters are placeholder (reward is not recovered
  exactly by any linear estimator).
- ``"neural"``: :class:`~econirl.environments.shapeshifter.ShapeshifterEnvironment`
  with ``reward_type="neural"`` and ``feature_type="neural"``.  State count
  is ``grid_size^2`` (product space via ``state_dim=2``).
"""

from __future__ import annotations

from econirl.environments.gridworld import GridworldEnvironment
from econirl.environments.objectworld import ObjectworldEnvironment
from econirl.environments.shapeshifter import ShapeshifterConfig, ShapeshifterEnvironment
from econirl.forms.base import Form, FormSpec


def grid(
    reward_form: str = "linear",
    grid_size: int = 5,
    seed: int = 0,
    **kw,
) -> Form:
    """Build a grid-topology :class:`~econirl.forms.base.Form`.

    Args:
        reward_form: ``"linear"``, ``"nonlinear"``, or ``"neural"``.
        grid_size: Side length N of the N x N grid.  Ignored for neural
            (Shapeshifter uses its own state parameterization).
        seed: Random seed for the environment.
        **kw: Extra keyword arguments forwarded to the underlying
            environment constructor.

    Returns:
        A :class:`~econirl.forms.base.Form` wrapping the appropriate
        :class:`~econirl.environments.base.DDCEnvironment`.

    Raises:
        ValueError: For an unknown ``reward_form``.
    """
    if reward_form == "linear":
        env = GridworldEnvironment(grid_size=grid_size, seed=seed, **kw)
        spec = FormSpec(
            topology="grid",
            reward_form="linear",
            num_states=env.num_states,
            num_actions=env.num_actions,
            name=f"grid-linear-{grid_size}x{grid_size}",
        )
        return Form(spec=spec, env=env)

    if reward_form == "nonlinear":
        env = ObjectworldEnvironment(grid_size=grid_size, seed=seed, **kw)
        # Objectworld's reward is a nonlinear function of object distances;
        # true_parameters() returns placeholder values (no finite linear theta).
        spec = FormSpec(
            topology="grid",
            reward_form="nonlinear",
            num_states=env.num_states,
            num_actions=env.num_actions,
            name=f"grid-nonlinear-{grid_size}x{grid_size}",
        )
        return Form(spec=spec, env=env)

    if reward_form == "neural":
        # ShapeshifterEnvironment with num_states=grid_size per axis and
        # state_dim=2 gives a grid_size^2 product state space.
        num_actions = kw.pop("num_actions", 5)
        cfg_kw = dict(
            num_states=grid_size,
            num_actions=num_actions,
            state_dim=2,
            reward_type="neural",
            feature_type="neural",
            seed=seed,
        )
        cfg_kw.update(kw)
        env = ShapeshifterEnvironment(ShapeshifterConfig(**cfg_kw))
        spec = FormSpec(
            topology="grid",
            reward_form="neural",
            num_states=env.num_states,
            num_actions=env.num_actions,
            name=f"grid-neural-{grid_size}x{grid_size}",
        )
        return Form(spec=spec, env=env)

    raise ValueError(
        f"grid: unknown reward_form {reward_form!r}.  "
        "Supported: 'linear', 'nonlinear', 'neural'."
    )
