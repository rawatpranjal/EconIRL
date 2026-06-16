"""Factored-topology form factory.

Builds :class:`~econirl.forms.base.Form` objects for problems with a
factored state space (multiple independent components).

Reward forms:
- ``"linear"``:
  :class:`~econirl.environments.multi_component_bus.MultiComponentBusEnvironment`
  (K independent bus components, each with M mileage bins, 2 actions).
  Raises ``ValueError`` for K >= 4 (state space too large for dense tensors).
- ``"neural"``: :class:`~econirl.environments.shapeshifter.ShapeshifterEnvironment`
  with ``state_dim=K`` and ``reward_type="neural"``.  Each axis uses
  ``num_states_per_dim`` states; total states = ``num_states_per_dim ** K``.
- ``"nonlinear"``: Not implemented (no clean factored nonlinear environment
  in the package).
"""

from __future__ import annotations

from econirl.environments.multi_component_bus import MultiComponentBusEnvironment
from econirl.environments.shapeshifter import ShapeshifterConfig, ShapeshifterEnvironment
from econirl.forms.base import Form, FormSpec


def factored(
    reward_form: str = "linear",
    K: int = 2,
    M: int = 20,
    seed: int = 0,
    **kw,
) -> Form:
    """Build a factored-topology :class:`~econirl.forms.base.Form`.

    Args:
        reward_form: ``"linear"`` or ``"neural"``.
        K: Number of independent components (linear only; raises for K >= 4).
        M: Number of mileage bins per component (linear only).
        seed: Random seed for the environment.
        **kw: Extra keyword arguments forwarded to the underlying
            environment constructor.

    Returns:
        A :class:`~econirl.forms.base.Form` wrapping the appropriate
        :class:`~econirl.environments.base.DDCEnvironment`.

    Raises:
        ValueError: If ``K >= 4`` for linear (state space is too large
            for dense tensors) or for an unknown ``reward_form``.
        NotImplementedError: For ``reward_form="nonlinear"``.
    """
    if reward_form == "linear":
        # MultiComponentBusEnvironment raises ValueError internally for K >= 4;
        # pass through directly so the error message is informative.
        env = MultiComponentBusEnvironment(K=K, M=M, seed=seed, **kw)
        spec = FormSpec(
            topology="factored",
            reward_form="linear",
            num_states=env.num_states,
            num_actions=env.num_actions,
            has_transitions=True,
            name=f"factored-linear-K{K}-M{M}",
        )
        return Form(spec=spec, env=env)

    if reward_form == "neural":
        # Use ShapeshifterEnvironment with state_dim=K; each axis has M states,
        # giving M^K total flat states (product-space encoding).
        num_actions = kw.pop("num_actions", 2)
        cfg_kw = dict(
            num_states=M,
            num_actions=num_actions,
            state_dim=K,
            reward_type="neural",
            feature_type="neural",
            seed=seed,
        )
        cfg_kw.update(kw)
        env = ShapeshifterEnvironment(ShapeshifterConfig(**cfg_kw))
        spec = FormSpec(
            topology="factored",
            reward_form="neural",
            num_states=env.num_states,
            num_actions=env.num_actions,
            has_transitions=True,
            name=f"factored-neural-K{K}-M{M}",
        )
        return Form(spec=spec, env=env)

    if reward_form == "nonlinear":
        raise NotImplementedError(
            "factored nonlinear is not implemented: there is no clean factored "
            "nonlinear-reward environment in the package.  Use "
            "reward_form='neural' for a nonlinear reward in a factored state space."
        )

    raise ValueError(
        f"factored: unknown reward_form {reward_form!r}.  "
        "Supported: 'linear', 'neural'."
    )
