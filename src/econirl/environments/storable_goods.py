"""Storable-goods consumer-stockpiling MDP generator (Hendel-Nevo).

Builds a known-truth single-type storable-goods dynamic discrete choice model as
an :class:`ArrayMDP`.  A household consumes one unit per period and chooses each
period whether to buy a pack of ``pack_size`` units.  The shelf price alternates
between a low "sale" regime and a high "regular" regime following an exogenous
two-state Markov chain.  The household stockpiles on sale to avoid paying the
regular price later, traded off against a per-unit holding cost.

**State** — a pair ``(i, p)``: inventory ``i`` in ``{0, ..., I_max}`` and price
regime ``p`` in ``{0 = sale, 1 = regular}``.  Flat index ``s = i * 2 + p``, so
the default ``I_max = 9`` gives ``10 x 2 = 20`` states.

**Actions** — ``{0 = do not buy, 1 = buy a pack of pack_size units}`` (2 actions).

**Timing within a period** (at state ``(i, p)``, action ``a``):

1. Purchase: ``available = i + pack_size * a`` units on hand.
2. Consume one unit if any is available.
3. Carry forward: ``next_i = clip(available - 1, 0, I_max)``.  Capping *after*
   subtracting consumption keeps the top inventory level reachable as a
   next-state and caps storage; over-buying above capacity is paid for but not
   carried.
4. Stockout: incurred when ``available == 0`` (the household wanted to consume
   but had nothing), i.e. exactly when ``i == 0`` and ``a == 0``.

**Reward** is linear in three features, sign-encoded so positive theta = positive
marginal cost::

    phi[s, a, 0] = -price_level(p) * pack_size * a   # spending on the purchase
    phi[s, a, 1] = -next_i                            # holding cost on carried units
    phi[s, a, 2] = -1{available == 0}                 # stockout disutility

    theta = [spend, holding, stockout] = [1.0, 0.2, 3.0]

**Identification** — the stockout feature is action-dependent (buying at ``i == 0``
avoids the stockout), so all three reward parameters are identified from choices.
:func:`~econirl.preprocessing.diagnostics.feature_diagnostics` returns
``contrast_rank = 3``.  Price varies exogenously, so the spending response to
price identifies the spend coefficient.

**Transition tensor orientation** — ``transitions`` has shape
``(num_actions, num_states, num_states)`` with ``transitions[a, s, s'] =
P(s' | s, a)``, matching the estimator-facing convention.
"""

from __future__ import annotations

import numpy as np

from econirl.environments.array_mdp import ArrayMDP

# Known-truth linear reward parameters (fixed, documented above). Tuned so the
# optimal policy genuinely stockpiles: stockout (3.0) exceeds even the regular
# per-unit price (2.0), so the household always wants stock; the sale-vs-regular
# price gap drives buying on sale; the small holding cost (0.2) caps the buffer.
_DEFAULT_THETA: np.ndarray = np.array([1.0, 0.2, 3.0], dtype=np.float64)
_DEFAULT_NAMES: list[str] = ["spend", "holding", "stockout"]


def storable_goods(
    max_inventory: int = 9,
    pack_size: int = 3,
    price_levels: tuple[float, float] = (1.0, 2.0),
    sale_persistence: float = 0.5,
    regular_persistence: float = 0.8,
    reward_form: str = "linear",
    discount_factor: float = 0.95,
    scale_parameter: float = 1.0,
    theta: np.ndarray | None = None,
    seed: int = 0,
) -> ArrayMDP:
    """Construct a storable-goods consumer-stockpiling :class:`ArrayMDP`.

    Args:
        max_inventory: Largest carried inventory ``I_max``.  States count is
            ``(I_max + 1) * 2``.
        pack_size: Units added by the buy action ``B``.
        price_levels: ``(sale, regular)`` per-unit shelf prices.
        sale_persistence: ``P(sale -> sale)`` in the price Markov chain.
        regular_persistence: ``P(regular -> regular)`` in the price Markov chain.
        reward_form: ``"linear"`` (default).  Any other value raises.
        discount_factor: Time discount ``beta`` in ``[0, 1)``.
        scale_parameter: Logit scale ``sigma > 0``.
        theta: Optional override for the known-truth reward parameters
            ``[spend, holding, stockout]``.  Defaults to ``[1.0, 0.2, 3.0]``.
        seed: Random seed recorded on the environment (the DGP is deterministic
            in the structural knobs; the seed only drives ``reset``/``step``).

    Returns:
        An :class:`ArrayMDP` with ``S = (max_inventory + 1) * 2`` states,
        ``A = 2`` actions, ``K = 3`` reward features, and known-truth theta.

    Raises:
        ValueError: For ``reward_form != "linear"``, non-positive ``pack_size``,
            ``max_inventory < 1``, or persistences outside ``[0, 1]``.
    """
    if reward_form != "linear":
        raise ValueError(
            f"storable_goods: unknown reward_form {reward_form!r}. "
            "Supported: 'linear'."
        )
    if max_inventory < 1:
        raise ValueError(f"storable_goods: max_inventory must be >= 1, got {max_inventory}.")
    if pack_size < 1:
        raise ValueError(f"storable_goods: pack_size must be >= 1, got {pack_size}.")
    for nm, v in (("sale_persistence", sale_persistence),
                  ("regular_persistence", regular_persistence)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"storable_goods: {nm} must be in [0, 1], got {v}.")

    I_max = int(max_inventory)
    B = int(pack_size)
    n_inv = I_max + 1
    n_price = 2
    S = n_inv * n_price
    A = 2
    sale_price, reg_price = float(price_levels[0]), float(price_levels[1])

    def flat(i: int, p: int) -> int:
        return i * n_price + p

    # ------------------------------------------------------------------
    # Price regime Markov chain (exogenous). price_trans[p, p'].
    #   p = 0 (sale): stays sale with prob sale_persistence.
    #   p = 1 (regular): stays regular with prob regular_persistence.
    # ------------------------------------------------------------------
    price_trans = np.array(
        [[sale_persistence, 1.0 - sale_persistence],
         [1.0 - regular_persistence, regular_persistence]],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Transition tensor (A, S, S): T[a, s, s'] = P(s' | s, a).
    # Inventory evolves deterministically given (i, a); only price is random.
    # ------------------------------------------------------------------
    T = np.zeros((A, S, S), dtype=np.float64)
    for i in range(n_inv):
        for p in range(n_price):
            s = flat(i, p)
            for a in range(A):
                available = i + B * a
                next_i = int(np.clip(available - 1, 0, I_max))
                for p_next in range(n_price):
                    T[a, s, flat(next_i, p_next)] = price_trans[p, p_next]

    # Defensive normalisation against floating-point drift.
    row_sums = T.sum(axis=2, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    T = T / row_sums

    # ------------------------------------------------------------------
    # Feature tensor (S, A, K=3), sign-encoded (positive theta = positive cost):
    #   phi[s, a, 0] = -spend       = -price_level(p) * B * a
    #   phi[s, a, 1] = -holding     = -next_i (carried inventory)
    #   phi[s, a, 2] = -stockout    = -1{available == 0}  (action-dependent)
    # ------------------------------------------------------------------
    features = np.zeros((S, A, 3), dtype=np.float64)
    for i in range(n_inv):
        for p in range(n_price):
            s = flat(i, p)
            price_level = sale_price if p == 0 else reg_price
            for a in range(A):
                available = i + B * a
                next_i = int(np.clip(available - 1, 0, I_max))
                spend = price_level * B * a
                stockout = 1.0 if available == 0 else 0.0
                features[s, a, 0] = -spend
                features[s, a, 1] = -float(next_i)
                features[s, a, 2] = -stockout

    theta_vec = _DEFAULT_THETA.copy() if theta is None else np.asarray(theta, dtype=np.float64)

    return ArrayMDP(
        transitions=T,
        features=features,
        theta=theta_vec,
        discount_factor=discount_factor,
        scale_parameter=scale_parameter,
        parameter_names=list(_DEFAULT_NAMES),
        seed=seed,
    )
