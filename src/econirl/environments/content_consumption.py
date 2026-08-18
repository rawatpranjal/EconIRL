"""Content-consumption viewer-session MDP generator.

Builds a known-truth content-consumption dynamic discrete choice model as an
:class:`ArrayMDP`.  A viewer sits in a session and each period chooses which
content category to consume, or to leave.  Consuming a category raises that
category's satiation (it grows boring), while the categories left alone recover.
The viewer trades current enjoyment against accumulated satiation, a per-period
time cost, and a variety bonus for keeping fresh categories on the menu, until
they decide to ``leave`` and end the session.

This is the **heterogeneity** environment of the suite: the dynamics are
type-agnostic and the same for every viewer, while different reward ``theta``
vectors describe different viewer types (a binge-watcher vs. a variety-seeker,
say).  Pass a different ``theta`` to get a different type; the state space,
actions, transitions, and features never change.

**State** -- a per-category satiation profile ``(s_0, ..., s_{C-1})`` with each
``s_c`` in ``{0, ..., L-1}`` (``L = satiation_levels``), flat-indexed in row-major
("odometer") order over the ``L ** C`` product, plus one extra absorbing
"session ended" state at the top index.  The defaults (``C = 3`` categories,
``L = 4`` levels) give ``4 ** 3 = 64`` regular states + 1 absorbing = **65**.

**Actions** -- ``{0 = consume category 0, ..., C-1 = consume category C-1,
C = leave}`` (``C + 1`` actions).  With the defaults: ``{0, 1, 2}`` consume
A/B/C and action ``3`` is ``leave``.  ``leave`` is the **anchor exit action**
(the outside option, zero reward) and the highest action index, which is the
base action the action-contrast identification check differences against.

**Timing within a period** (at satiation profile ``s``, action ``a``):

1. If ``a == leave`` (``a == C``): the session ends, transition to the
   absorbing ``session_ended_state``.
2. Otherwise consume category ``c = a``: ``s_c`` rises by one (capped at
   ``L - 1``) and every other category decays by one (floored at ``0``).
   Transition to that new profile deterministically.

The absorbing state is self-absorbing: every action from it returns to it.

**Reward** is linear in four features, encoded so a positive ``theta`` reads as a
positive weight on that channel::

    phi[s, a, 0] = base_quality[c]            # enjoyment: intrinsic category appeal
    phi[s, a, 1] = -satiation_of(s, c)        # satiation cost on the consumed category
    phi[s, a, 2] = -1.0                        # flat per-period time cost
    phi[s, a, 3] = (# categories at satiation 0)  # variety bonus (fresh menu)

    theta = [enjoyment, satiation_cost, time_cost, variety_bonus]

For the ``leave`` action and for the absorbing state, all features are zero, so
the exit action and the session-ended state both carry zero flow reward -- the
anchors AIRL2 expects.

**Identification** -- ``enjoyment`` is category-specific intrinsic quality, kept
*decoupled* from satiation so it is not an affine function of the satiation
level.  That decoupling is what keeps the action-contrast design full rank:
:func:`~econirl.preprocessing.diagnostics.feature_diagnostics` returns
``contrast_rank = 4`` with a condition number near 10.  (Tying enjoyment to
``1 - k * satiation`` instead collapses it onto the satiation-cost feature and
drops the contrast rank to 3.)

**Transition tensor orientation** -- ``transitions`` has shape
``(num_actions, num_states, num_states)`` with ``transitions[a, s, s'] =
P(s' | s, a)``, matching the estimator-facing convention.

The constructed :class:`ArrayMDP` carries two extra attributes so a study and the
AIRL2 anchors line up with the environment:

- ``leave_action`` -- the index of the ``leave`` action (``= C``).
- ``session_ended_state`` -- the index of the absorbing state (``= L ** C``).
"""

from __future__ import annotations

from itertools import product

import numpy as np

from econirl.environments.array_mdp import ArrayMDP

# Known-truth feature order and default reward weights (a generic mixed viewer;
# the heterogeneity study overrides theta per viewer type).
_DEFAULT_NAMES: list[str] = ["enjoyment", "satiation_cost", "time_cost", "variety_bonus"]
_DEFAULT_THETA: np.ndarray = np.array([1.5, 0.8, 0.5, 0.6], dtype=np.float64)


def content_consumption(
    n_categories: int = 3,
    satiation_levels: int = 4,
    base_quality: np.ndarray | None = None,
    reward_form: str = "linear",
    discount_factor: float = 0.95,
    scale_parameter: float = 1.0,
    theta: np.ndarray | None = None,
    seed: int = 0,
) -> ArrayMDP:
    """Construct a content-consumption viewer-session :class:`ArrayMDP`.

    Args:
        n_categories: Number of content categories ``C`` (consume actions).
            With ``n_categories = 3`` the consume actions are ``{0, 1, 2}`` and
            ``leave`` is action ``3``.
        satiation_levels: Number of per-category satiation levels ``L``.  Regular
            state count is ``L ** C``; total states is ``L ** C + 1``.
        base_quality: Optional length-``C`` array of intrinsic per-category
            enjoyment.  Defaults to a descending ramp so categories differ in
            appeal (this difference is what keeps enjoyment identified).
        reward_form: ``"linear"`` (default).  Any other value raises.
        discount_factor: Time discount ``beta`` in ``[0, 1)``.
        scale_parameter: Logit scale ``sigma > 0``.
        theta: Optional override for the reward weights ``[enjoyment,
            satiation_cost, time_cost, variety_bonus]``.  Different ``theta``
            give different viewer types.  Defaults to ``[1.5, 0.8, 0.5, 0.6]``.
        seed: Random seed recorded on the environment (the DGP is deterministic
            in the structural knobs; the seed only drives ``reset``/``step``).

    Returns:
        An :class:`ArrayMDP` with ``S = satiation_levels ** n_categories + 1``
        states, ``A = n_categories + 1`` actions, ``K = 4`` reward features,
        known-truth theta, and the attributes ``leave_action`` and
        ``session_ended_state``.

    Raises:
        ValueError: For ``reward_form != "linear"``, ``n_categories < 2``,
            ``satiation_levels < 2``, or a ``base_quality`` of the wrong length.
    """
    if reward_form != "linear":
        raise ValueError(
            f"content_consumption: unknown reward_form {reward_form!r}. Supported: 'linear'."
        )
    if n_categories < 2:
        raise ValueError(f"content_consumption: n_categories must be >= 2, got {n_categories}.")
    if satiation_levels < 2:
        raise ValueError(
            f"content_consumption: satiation_levels must be >= 2, got {satiation_levels}."
        )

    C = int(n_categories)
    L = int(satiation_levels)

    if base_quality is None:
        # Descending intrinsic appeal across categories, decoupled from satiation
        # so the enjoyment feature stays identified (see module docstring).
        base = np.linspace(1.0, 0.4, C, dtype=np.float64)
    else:
        base = np.asarray(base_quality, dtype=np.float64).reshape(-1)
        if base.shape[0] != C:
            raise ValueError(
                f"content_consumption: base_quality must have length n_categories "
                f"({C}); got {base.shape[0]}."
            )

    # Enumerate satiation profiles in row-major ("odometer") order. Profile
    # ``profiles[i]`` is the satiation vector of regular state ``i``.
    profiles = [np.asarray(p, dtype=np.int64) for p in product(range(L), repeat=C)]
    index_of = {tuple(int(x) for x in p): i for i, p in enumerate(profiles)}

    n_regular = L**C
    S = n_regular + 1
    A = C + 1
    leave_action = C
    session_ended_state = n_regular  # the top index, the absorbing state

    # ------------------------------------------------------------------
    # Transition tensor (A, S, S): T[a, s, s'] = P(s' | s, a). Deterministic.
    #   consume c: s_c += 1 (cap L-1); every other category -= 1 (floor 0).
    #   leave:     -> session_ended_state.
    # ------------------------------------------------------------------
    T = np.zeros((A, S, S), dtype=np.float64)
    for i, sat in enumerate(profiles):
        for c in range(C):
            nxt = sat.copy()
            nxt[c] = min(int(nxt[c]) + 1, L - 1)
            for other in range(C):
                if other != c:
                    nxt[other] = max(int(nxt[other]) - 1, 0)
            T[c, i, index_of[tuple(int(x) for x in nxt)]] = 1.0
        T[leave_action, i, session_ended_state] = 1.0
    # Absorbing: every action self-loops.
    T[:, session_ended_state, session_ended_state] = 1.0

    # ------------------------------------------------------------------
    # Feature tensor (S, A, K=4). Consume actions carry features; the leave
    # action and the absorbing state are zero (the anchors).
    #   phi[s, a, 0] = base_quality[c]                 enjoyment
    #   phi[s, a, 1] = -satiation_of(s, c)             satiation cost
    #   phi[s, a, 2] = -1.0                             time cost
    #   phi[s, a, 3] = #{categories at satiation 0}     variety bonus
    # ------------------------------------------------------------------
    features = np.zeros((S, A, 4), dtype=np.float64)
    for i, sat in enumerate(profiles):
        variety = float(np.sum(sat == 0))
        for c in range(C):
            features[i, c, 0] = base[c]
            features[i, c, 1] = -float(sat[c])
            features[i, c, 2] = -1.0
            features[i, c, 3] = variety
        # leave action (index C) stays all zeros -> zero-reward exit anchor.
    # Absorbing state row already all zeros.

    theta_vec = _DEFAULT_THETA.copy() if theta is None else np.asarray(theta, dtype=np.float64)

    env = ArrayMDP(
        transitions=T,
        features=features,
        theta=theta_vec,
        discount_factor=discount_factor,
        scale_parameter=scale_parameter,
        parameter_names=list(_DEFAULT_NAMES),
        seed=seed,
    )

    # Expose the anchor indices so the study and AIRL2 line up with the env.
    env.leave_action = leave_action
    env.session_ended_state = session_ended_state
    env.n_categories = C
    env.satiation_levels = L
    return env
