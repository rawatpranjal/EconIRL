"""Synthetic road-network MDP generator.

Builds a known-truth graph/route-choice :class:`ArrayMDP` on a random
geometric graph: nodes are scattered uniformly in the unit square, edges
connect nodes within Euclidean radius ``connectivity``, and a minimum spanning
tree (MST) over Euclidean distances is added to guarantee full connectivity.
An agent at node ``s`` chooses among its ``num_actions`` nearest neighbours
(sorted by Euclidean distance); actions beyond the node degree self-loop.

**Identification** — reward features are *edge-dependent* (they describe the
traversed edge, not the state), so all three parameters are identified from
observed route choices.  :func:`~econirl.preprocessing.diagnostics.feature_diagnostics`
returns ``contrast_rank = 3``.

**Goal node** — defaults to the node geometrically closest to corner (1, 1),
giving a well-separated destination in the unit square.

**Known-truth theta** ::

    theta = [1.0, 0.5, 1.0]
    names = ["edge_cost", "amenity", "goal"]

Features are sign-encoded so positive theta values are intuitive:
``phi[0] = -edge_cost`` (penalise long edges), ``phi[1] = amenity(s')``
(reward destination amenity), ``phi[2] = -dist_to_goal(s')`` (penalise
distance to the goal).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path

from econirl.environments.array_mdp import ArrayMDP

# Known-truth linear reward parameters (fixed, documented above).
_DEFAULT_THETA: np.ndarray = np.array([1.0, 0.5, 1.0], dtype=np.float64)
_DEFAULT_NAMES: list[str] = ["edge_cost", "amenity", "goal"]


def road_network(
    num_nodes: int = 20,
    num_actions: int = 4,
    reward_form: str = "linear",
    connectivity: float = 0.25,
    transition_noise: float = 0.0,
    goal_node: int | None = None,
    discount_factor: float = 0.95,
    scale_parameter: float = 1.0,
    seed: int = 0,
) -> ArrayMDP:
    """Construct a road-network :class:`ArrayMDP`.

    Nodes are placed uniformly at random in the unit square.  Edges connect
    pairs within Euclidean radius ``connectivity``; a full MST is overlaid
    so the graph is always connected regardless of ``connectivity``.

    Action ``a`` at state ``s`` moves to the ``a``-th nearest neighbour of
    ``s`` (0-indexed, sorted ascending by Euclidean distance).  If
    ``a >= degree(s)`` the agent self-loops (stays at ``s``).

    Args:
        num_nodes: Number of graph nodes (= number of MDP states ``S``).
        num_actions: Number of discrete actions ``A``.
        reward_form: ``"linear"`` (default).  ``"nonlinear"`` raises
            :exc:`NotImplementedError`.
        connectivity: Euclidean radius for drawing radius-based edges.
            Larger values give denser graphs.
        transition_noise: Probability of moving to a uniformly random
            neighbour of ``s`` instead of the intended one.  0.0 = fully
            deterministic.  Noise is not applied to self-loop actions.
        goal_node: Destination node used for ``dist_to_goal`` features.
            Defaults to the node closest to corner (1, 1).
        discount_factor: Time discount ``beta`` in ``[0, 1)``.
        scale_parameter: Logit scale ``sigma > 0``.
        seed: Random seed.  The same seed reproduces the identical graph.

    Returns:
        An :class:`ArrayMDP` with ``S = num_nodes`` states, ``A = num_actions``
        actions, ``K = 3`` reward features, and known-truth linear theta.

    Raises:
        ValueError: For ``reward_form`` not in ``{"linear", "nonlinear"}``.
        NotImplementedError: For ``reward_form="nonlinear"`` (deferred).
    """
    if reward_form not in ("linear", "nonlinear"):
        raise ValueError(
            f"road_network: unknown reward_form {reward_form!r}.  "
            "Supported: 'linear', 'nonlinear'."
        )
    if reward_form == "nonlinear":
        raise NotImplementedError(
            "road_network: nonlinear reward is not yet implemented.  "
            "Use reward_form='linear'.  A nonlinear variant (fixed MLP over the "
            "3 edge features) is deferred to a later chunk."
        )

    rng = np.random.default_rng(seed)
    N = num_nodes

    # ------------------------------------------------------------------
    # 1. Graph geometry
    # ------------------------------------------------------------------
    coords = rng.uniform(size=(N, 2))  # (N, 2) uniform in unit square

    # Pairwise Euclidean distances — (N, N), diagonal = 0
    diff = coords[:, None, :] - coords[None, :, :]    # (N, N, 2)
    pairwise_dist = np.sqrt((diff ** 2).sum(axis=-1))  # (N, N)
    np.fill_diagonal(pairwise_dist, 0.0)

    # Radius-based adjacency (exclude self-loops)
    radius_adj = (pairwise_dist > 0) & (pairwise_dist <= connectivity)  # (N, N) bool

    # Minimum spanning tree to guarantee full graph connectivity.
    # minimum_spanning_tree treats zero as absent; pairwise_dist is positive
    # off-diagonal, so passing it finds the MST of the complete graph.
    mst_sparse = minimum_spanning_tree(csr_matrix(pairwise_dist))
    mst_arr = np.asarray(mst_sparse.todense())
    mst_adj = (mst_arr > 0) | (mst_arr.T > 0)  # symmetrize: MST is undirected

    # Final adjacency = radius-based ∪ MST
    adj = radius_adj | mst_adj  # (N, N) bool

    # ------------------------------------------------------------------
    # 2. Per-node features: amenity scalars and shortest-path distances
    # ------------------------------------------------------------------
    amenity = rng.standard_normal(N)  # per-node amenity drawn once, seeded

    if goal_node is None:
        # Node closest to corner (1, 1) — well-separated destination
        goal_node = int(
            np.argmin(np.hypot(coords[:, 0] - 1.0, coords[:, 1] - 1.0))
        )

    # Shortest-path distances from every node to goal_node (Euclidean weights)
    weight_adj = pairwise_dist * adj  # (N, N): 0 where no edge, dist where edge
    sp_dists = shortest_path(
        csr_matrix(weight_adj), directed=False, indices=goal_node
    )
    # sp_dists[s] = undirected shortest-path distance from s to goal_node
    # All finite because adj is connected (MST guarantees it).

    # ------------------------------------------------------------------
    # 3. Sorted neighbour lists: action a -> a-th nearest neighbour
    # ------------------------------------------------------------------
    neighbor_lists: list[np.ndarray] = []
    for s in range(N):
        nbrs = np.where(adj[s])[0]
        order = np.argsort(pairwise_dist[s, nbrs])
        neighbor_lists.append(nbrs[order])

    # ------------------------------------------------------------------
    # 4. Transition tensor (A, S, S)
    #    - action a < degree(s): move to nbrs[a]; with optional noise spread
    #      uniformly over all neighbours.
    #    - action a >= degree(s): deterministic self-loop (noise not applied).
    # ------------------------------------------------------------------
    T = np.zeros((num_actions, N, N), dtype=np.float64)
    for s in range(N):
        nbrs = neighbor_lists[s]
        degree = len(nbrs)
        for a in range(num_actions):
            if a < degree:
                s_next = int(nbrs[a])
                if transition_noise == 0.0:
                    T[a, s, s_next] = 1.0
                else:
                    # Uniform mass over all neighbours, then add deterministic
                    # mass to the intended next-state.
                    uniform = transition_noise / degree
                    for nbr in nbrs:
                        T[a, s, int(nbr)] = uniform
                    T[a, s, s_next] += 1.0 - transition_noise
            else:
                # Self-loop: action index >= degree of this node
                T[a, s, s] = 1.0

    # Defensive normalisation against floating-point drift
    row_sums = T.sum(axis=2, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    T = T / row_sums

    # ------------------------------------------------------------------
    # 5. Feature tensor (S, A, K=3) — action-dependent, edge-descriptive
    #    phi[s, a, 0] = -edge_cost         (neg. Euclidean length; 0 for self-loop)
    #    phi[s, a, 1] =  amenity(s')       (per-node scalar at destination)
    #    phi[s, a, 2] = -dist_to_goal(s')  (neg. shortest-path dist to goal)
    # ------------------------------------------------------------------
    features = np.zeros((N, num_actions, 3), dtype=np.float64)
    for s in range(N):
        nbrs = neighbor_lists[s]
        degree = len(nbrs)
        for a in range(num_actions):
            if a < degree:
                s_next = int(nbrs[a])
                features[s, a, 0] = -pairwise_dist[s, s_next]  # -edge_cost
                features[s, a, 1] = amenity[s_next]             # amenity at dest
                features[s, a, 2] = -sp_dists[s_next]           # -dist to goal
            else:
                # Self-loop: agent stays at s, no edge traversed
                features[s, a, 0] = 0.0
                features[s, a, 1] = amenity[s]
                features[s, a, 2] = -sp_dists[s]

    return ArrayMDP(
        transitions=T,
        features=features,
        theta=_DEFAULT_THETA.copy(),
        discount_factor=discount_factor,
        scale_parameter=scale_parameter,
        parameter_names=list(_DEFAULT_NAMES),
        seed=seed,
    )
