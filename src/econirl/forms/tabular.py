"""Tabular-topology form factory.

Builds :class:`~econirl.forms.base.Form` objects whose state space is a
flat, unstructured set of indices -- the most general "abstract MDP"
topology.

Reward forms:
- ``"linear"``: :func:`random_mdp` -> :class:`ArrayMDP`.  Action-dependent
  polynomial features; exact theta known.
- ``"neural"``: :class:`ShapeshifterEnvironment` with ``reward_type="neural"``
  and ``feature_type="neural"``.  Frozen MLP reward; no finite theta.
- ``"nonlinear"``: Not implemented for tabular topology.  ArrayMDP carries
  only a linear reward, and no clean nonlinear-in-x tabular environment
  exists in the package.  Raise :exc:`NotImplementedError`.
"""

from __future__ import annotations

from econirl.environments.array_mdp import ArrayMDP  # noqa: F401 – re-exported for callers
from econirl.environments.random_mdp import random_mdp
from econirl.environments.shapeshifter import ShapeshifterConfig, ShapeshifterEnvironment
from econirl.forms.base import Form, FormSpec


def tabular(
    reward_form: str = "linear",
    num_states: int = 30,
    num_actions: int = 3,
    seed: int = 0,
    **kw,
) -> Form:
    """Build a tabular-topology :class:`~econirl.forms.base.Form`.

    Args:
        reward_form: ``"linear"`` or ``"neural"``.
        num_states: Number of discrete states (linear only; ignored for
            neural where the Shapeshifter config is used directly).
        num_actions: Number of discrete actions.
        seed: Random seed for the environment.
        **kw: Extra keyword arguments forwarded to the underlying
            environment constructor.

    Returns:
        A :class:`~econirl.forms.base.Form` wrapping the appropriate
        :class:`~econirl.environments.base.DDCEnvironment`.

    Raises:
        NotImplementedError: For ``reward_form="nonlinear"`` (no clean
            tabular nonlinear-in-x environment in the package).
        ValueError: For an unknown ``reward_form``.
    """
    if reward_form == "linear":
        num_features = kw.pop("num_features", 3)
        env = random_mdp(
            num_states=num_states,
            num_actions=num_actions,
            num_features=num_features,
            seed=seed,
            **kw,
        )
        spec = FormSpec(
            topology="tabular",
            reward_form="linear",
            num_states=env.num_states,
            num_actions=env.num_actions,
            has_transitions=True,
            name=f"tabular-linear-S{env.num_states}-A{env.num_actions}",
        )
        return Form(spec=spec, env=env)

    if reward_form == "neural":
        cfg_kw = dict(
            num_states=num_states,
            num_actions=num_actions,
            reward_type="neural",
            feature_type="neural",
            seed=seed,
        )
        cfg_kw.update(kw)
        env = ShapeshifterEnvironment(ShapeshifterConfig(**cfg_kw))
        spec = FormSpec(
            topology="tabular",
            reward_form="neural",
            num_states=env.num_states,
            num_actions=env.num_actions,
            has_transitions=True,
            name=f"tabular-neural-S{env.num_states}-A{env.num_actions}",
        )
        return Form(spec=spec, env=env)

    if reward_form == "nonlinear":
        raise NotImplementedError(
            "tabular nonlinear is not implemented: ArrayMDP uses a linear reward "
            "by design, and there is no clean nonlinear-in-x tabular environment "
            "in the package.  Use reward_form='neural' for nonlinear reward, or "
            "topology='grid' with reward_form='nonlinear' (ObjectworldEnvironment)."
        )

    raise ValueError(
        f"tabular: unknown reward_form {reward_form!r}.  "
        "Supported: 'linear', 'neural'."
    )
