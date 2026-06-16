"""Graph-topology form factory.

Builds :class:`~econirl.forms.base.Form` objects for road-network route-choice
problems.

Reward forms:
- ``"linear"``: :func:`~econirl.environments.road_network.road_network`
  -- random geometric graph, ``num_actions`` nearest-neighbour moves,
  K=3 edge-dependent features (edge_cost, amenity, dist_to_goal).
  Known-truth theta = [1.0, 0.5, 1.0]; contrast_rank = 3.
- ``"nonlinear"``: raises :exc:`NotImplementedError` (deferred).
"""

from __future__ import annotations

from econirl.environments.road_network import road_network
from econirl.forms.base import Form, FormSpec


def graph(
    reward_form: str = "linear",
    num_nodes: int = 20,
    num_actions: int = 4,
    seed: int = 0,
    **kw,
) -> Form:
    """Build a graph-topology :class:`~econirl.forms.base.Form`.

    Args:
        reward_form: ``"linear"`` (default).  ``"nonlinear"`` raises
            :exc:`NotImplementedError`.
        num_nodes: Number of graph nodes (= number of MDP states).
        num_actions: Number of discrete actions (nearest-neighbour choices).
        seed: Random seed for node placement and amenity draws.
        **kw: Extra keyword arguments forwarded to
            :func:`~econirl.environments.road_network.road_network`
            (e.g. ``connectivity``, ``transition_noise``, ``goal_node``).

    Returns:
        A :class:`~econirl.forms.base.Form` wrapping the road-network
        :class:`~econirl.environments.array_mdp.ArrayMDP`.

    Raises:
        NotImplementedError: For ``reward_form="nonlinear"`` (deferred).
        ValueError: For an unknown ``reward_form``.
    """
    if reward_form == "linear":
        env = road_network(
            num_nodes=num_nodes,
            num_actions=num_actions,
            reward_form="linear",
            seed=seed,
            **kw,
        )
        spec = FormSpec(
            topology="graph",
            reward_form="linear",
            num_states=env.num_states,
            num_actions=env.num_actions,
            has_transitions=True,
            name=f"graph-linear-N{num_nodes}-A{num_actions}",
        )
        return Form(spec=spec, env=env)

    if reward_form == "nonlinear":
        raise NotImplementedError(
            "graph nonlinear is not yet implemented.  "
            "Use reward_form='linear'.  A nonlinear road-network variant "
            "is deferred to a later chunk."
        )

    raise ValueError(
        f"graph: unknown reward_form {reward_form!r}.  "
        "Supported: 'linear', 'nonlinear'."
    )
