"""Simulation study: high-dimensional route choice on a synthetic road network.

The scale axis of the suite. The small ``route_choice`` study tops out at a
25-node graph. This one runs the same route-choice DGP on a much larger
synthetic road network, where the planning horizon actually matters. It is the
showcase for RHIP (Receding Horizon Inverse Planning), whose single knob -- the
planning horizon ``H`` -- spans a family of classic methods: ``H=0`` recovers
Max-Margin Planning, ``H=1`` a Bayesian-IRL middle ground, and ``H=inf``
recovers Max Causal Entropy IRL exactly.

Generates ``validation/results/study_highdim_route_choice.json`` and renders
``docs/simulation_studies/highdim_route_choice.md`` from it via the shared
harness (``validation/benchmark/harness.py``), inheriting the honesty contract:
raw per-replication facts, page as a pure function of the JSON, crashes first
class, no fabricated metrics.

Usage::

    python scripts/study_highdim_route_choice.py [--verbose]      # run + write JSON
    python scripts/study_highdim_route_choice.py --page           # regenerate the page
    python scripts/study_highdim_route_choice.py --verify         # re-derive table from JSON
    python scripts/study_highdim_route_choice.py --only-estimator NAME   # retry one, merge
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from econirl.environments.road_network import road_network  # noqa: E402
from validation.benchmark.harness import Cell, RosterEntry, main_cli  # noqa: E402
from validation.benchmark.runner import _action_reward, _linear_utility  # noqa: E402

RESULTS_JSON = os.path.join(_ROOT, "validation", "results", "study_highdim_route_choice.json")
PAGE_PATH = os.path.join(_ROOT, "docs", "simulation_studies", "highdim_route_choice.md")
_STATIC = os.path.join(_ROOT, "docs", "_static", "simulation_studies")
FIGURE_PNG = os.path.join(_STATIC, "highdim_route_choice_dgp.png")
RESULTS_FIG = os.path.join(_STATIC, "highdim_route_choice_results.png")
SCALING_FIG = os.path.join(_STATIC, "highdim_route_choice_scaling.png")
NETWORK_FIG = os.path.join(_STATIC, "highdim_route_choice_network.png")
HORIZON_FIG = os.path.join(_STATIC, "highdim_route_choice_horizon.png")

# ---- DGP configuration ----
# 4 actions, beta 0.95. The headline cell is 150 nodes (150 states): genuinely
# high-dimensional versus the 25-node route_choice reference, while the dense
# (A, S, S) kernel stays memory-safe (150 x 4 x 150 floats). The scaling sweep
# adds a 75-node cell so compute and accuracy can be traced against size.
HEADLINE_NODES = 150
SCALING_NODES = (75,)


def _env(num_nodes=HEADLINE_NODES):
    return road_network(num_nodes=num_nodes, num_actions=4, seed=0, discount_factor=0.95)


# ---------------------------------------------------------------------------
# Roster: _run_*(env, panel) -> result with .parameters, .standard_errors,
#         .policy, .value_function, .converged
# ---------------------------------------------------------------------------


def _mce_config():
    """Shared MCE-IRL config used by both MCE-IRL and the RHIP H=inf endpoint.

    Passing the same config object to RHIP(H=inf) via ``mce_config`` makes the
    H=inf endpoint reproduce this exact MCE-IRL run, so RHIP-Hinf == MCE-IRL by
    construction rather than by luck.
    """
    from econirl.estimation import MCEIRLConfig

    return MCEIRLConfig(
        optimizer="gradient",
        learning_rate=0.05,
        outer_max_iter=80,
        inner_max_iter=1000,
        compute_se=False,
        verbose=False,
    )


def _run_rhip(env, panel, horizon):
    from econirl.estimators.rhip import RHIPConfig, RHIPEstimator

    config = RHIPConfig(
        horizon=horizon,
        learning_rate=0.05,
        outer_max_iter=80,
        compute_se=False,
        verbose=False,
    )
    if horizon == float("inf"):
        # Exact MCE-IRL endpoint: delegate to the same config MCE-IRL uses.
        config.mce_config = _mce_config()
    est = RHIPEstimator(config=config)
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_rhip_h0(env, panel):
    return _run_rhip(env, panel, 0)


def _run_rhip_h1(env, panel):
    return _run_rhip(env, panel, 1)


def _run_rhip_h3(env, panel):
    return _run_rhip(env, panel, 3)


def _run_rhip_hinf(env, panel):
    return _run_rhip(env, panel, float("inf"))


def _run_mce_irl(env, panel):
    from econirl.estimation import MCEIRLEstimator

    est = MCEIRLEstimator(config=_mce_config())
    return est.estimate(panel, _action_reward(env), env.problem_spec, env.transition_matrices)


def _run_nfxp(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-10,
        inner_max_iter=100_000,
        compute_hessian=True,
        verbose=False,
    )
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_ccp(env, panel):
    from econirl.estimation import CCPEstimator

    est = CCPEstimator(num_policy_iterations=5, compute_hessian=True, verbose=False)
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


def _run_mpec(env, panel):
    from econirl.estimation.mpec import MPECConfig, MPECEstimator

    # solver="sqp" is the real constrained MLE; "slsqp" is a legacy alias for
    # the augmented-Lagrangian penalty solver and does not check optimality.
    est = MPECEstimator(
        config=MPECConfig(solver="sqp", outer_max_iter=200, tol=1e-8, constraint_tol=1e-6),
        compute_hessian=True,
        verbose=False,
    )
    return est.estimate(panel, _linear_utility(env), env.problem_spec, env.transition_matrices)


# Full roster on the headline cell. The four RHIP horizon variants are the
# spectrum (H=0 -> MMP, H=1 -> BIRL, H=inf -> MCE-IRL); MCE-IRL is the
# IRL baseline; NFXP / CCP / MPEC are the structural reference.
ROSTER = (
    RosterEntry("RHIP-H0", "behavioral", _run_rhip_h0, uses_transitions=True),
    RosterEntry("RHIP-H1", "behavioral", _run_rhip_h1, uses_transitions=True),
    RosterEntry("RHIP-H3", "behavioral", _run_rhip_h3, uses_transitions=True),
    RosterEntry("RHIP-Hinf", "behavioral", _run_rhip_hinf, uses_transitions=True),
    RosterEntry("MCE-IRL", "behavioral", _run_mce_irl, uses_transitions=True),
    RosterEntry("NFXP", "structural", _run_nfxp, uses_transitions=True),
    RosterEntry("CCP", "structural", _run_ccp, uses_transitions=True),
    RosterEntry("MPEC", "structural", _run_mpec, timeout=300, uses_transitions=True),
)

# Scaling sweep roster: a strict subset of ROSTER. Two RHIP horizon endpoints
# (the cheap H=0 and the expensive H=inf) bracket the spectrum; NFXP and MCE-IRL
# anchor the structural and IRL families.
SCALING_ROSTER = (
    RosterEntry("RHIP-Hinf", "behavioral", _run_rhip_hinf, uses_transitions=True),
    RosterEntry("RHIP-H0", "behavioral", _run_rhip_h0, uses_transitions=True),
    RosterEntry("NFXP", "structural", _run_nfxp, uses_transitions=True),
    RosterEntry("MCE-IRL", "behavioral", _run_mce_irl, uses_transitions=True),
)

# ---------------------------------------------------------------------------
# Diagnoses, excluded, cell, narrative
# ---------------------------------------------------------------------------

DIAGNOSES = {
    "RHIP-H0": (
        "Receding Horizon Inverse Planning at horizon zero. No soft backups run, "
        "so the policy is a softmax over the deterministic continuation value. "
        "This is the Max-Margin-Planning end of the spectrum: cheap, and the "
        "least robust to demonstrator noise."
    ),
    "RHIP-H1": (
        "One soft Bellman backup over the deterministic tail. A middle ground "
        "between the deterministic and the fully stochastic planner."
    ),
    "RHIP-H3": (
        "Three soft Bellman backups. It recovers most of the accuracy of the full "
        "stochastic planner (H=inf), a middle point on the horizon spectrum "
        "between the deterministic and the fully stochastic ends."
    ),
    "RHIP-Hinf": (
        "The infinite-horizon endpoint delegates to MCE-IRL with the same config, "
        "so RHIP-Hinf and MCE-IRL are the same computation here. Their rows match "
        "by construction. This anchors the spectrum at the Max Causal Entropy end."
    ),
    "MCE-IRL": (
        "Max Causal Entropy IRL with linear features. Its convergence indicator "
        "reports whether the gradient norm crossed the tolerance. The objective "
        "often plateaus before that, so it can read False while the recovered "
        "policy is accurate. It is the H=inf endpoint of the RHIP spectrum."
    ),
    "NFXP": (
        "Full-solution MLE with a nested Bellman fixed-point inner loop. "
        "Quadratic convergence near the optimum. All three parameters are "
        "identified from the edge-feature contrast."
    ),
    "CCP": (
        "CCP uses a first-step nonparametric policy estimate to avoid the inner "
        "Bellman loop. One policy-iteration step corrects the first-stage bias."
    ),
    "MPEC": (
        "Mathematical programming with equilibrium constraints. Uses solver='sqp' "
        "for real constrained MLE; the legacy 'slsqp' alias checks only Bellman "
        "feasibility. The constrained solve is the slowest of the structural "
        "family at this size."
    ),
}

EXCLUDED = [
    {
        "name": "IQ-Learn, f-IRL",
        "reason": (
            "not separately identified from choices on this problem; "
            "reward is only partially identified from behavior"
        ),
    },
    {
        "name": "NeuralGLADIUS",
        "reason": (
            "model-free policy learner, coverage-fragile at this size; "
            "peripheral nodes go unvisited and its policy estimate degrades "
            "where the panel does not reach"
        ),
    },
    {
        "name": "AIRL",
        "reason": (
            "the public estimator requires a state-only reward, while this "
            "route problem is identified through within-state edge contrasts"
        ),
    },
    {
        "name": "NNES, SEES, TD-CCP, UFXP",
        "reason": (
            "correct structural estimators but slower at 150 states; "
            "NFXP and CCP already cover the structural family"
        ),
    },
]


def _cell(num_nodes, *, headline, roster):
    """One route-choice cell at a given graph size.

    The headline cell carries the DGP figure, the parameter-recovery table, and
    the scorecard. The smaller size is scaling-only: it runs a trimmed roster to
    feed the scaling figure but renders no tables of its own.
    """
    return Cell(
        cell_id="route_highdim" if headline else f"route_highdim_{num_nodes}",
        label=f"High-dimensional route choice ({num_nodes} nodes, 4 actions)",
        description=(
            "Synthetic route choice on a random geometric road network. "
            f"``road_network(num_nodes={num_nodes}, num_actions=4, "
            "discount_factor=0.95, seed=0)``."
        ),
        env_factory=(lambda n=num_nodes: _env(n)),
        roster=roster,
        n_individuals=400,
        n_periods=40,
        seed=42,
        n_replications=10 if headline else 2,
        fit_timeout=300,
        param_block=headline,
        figure=FIGURE_PNG if headline else None,
        results_figure=RESULTS_FIG if headline else None,
        scaling_only=not headline,
    )


# Headline first so single-size runs (--only-cell route_highdim) still work.
CELLS = (
    _cell(HEADLINE_NODES, headline=True, roster=ROSTER),
    *(_cell(n, headline=False, roster=SCALING_ROSTER) for n in SCALING_NODES),
)

NARRATIVE = {
    "title": "High-dimensional route choice on a synthetic road network",
    "intro": (
        "A traveller moves through a road network one step at a time. "
        "Each period, the agent chooses among the nearest neighbours of the "
        "current node. The utility depends on the edge: how long it is, how "
        "attractive the destination is, and how close the destination sits to "
        "a fixed goal node. This is the same route-choice problem as the small "
        "study, run on a much larger graph.\n"
        "\n"
        "## Why this study\n"
        "\n"
        "Real route choice spans hundreds to millions of road segments. At that "
        "scale, the fully stochastic planner that Max Causal Entropy IRL solves "
        "becomes expensive. A shorter planning horizon trades a little accuracy "
        "for a lot of compute. RHIP (Receding Horizon Inverse Planning) makes "
        "the horizon a single knob $H$. Within $H$ steps the agent plans with "
        "the expensive stochastic policy; beyond $H$ it falls back to a cheap "
        "deterministic planner. The horizon recovers three classic methods as "
        "special cases:\n"
        "\n"
        "$$\n"
        "H = 0 \\;\\to\\; \\text{Max-Margin Planning}, \\quad "
        "H = 1 \\;\\to\\; \\text{Bayesian-IRL middle ground}, \\quad "
        "H = \\infty \\;\\to\\; \\text{Max Causal Entropy IRL}.\n"
        "$$\n"
        "\n"
        "The 25-node study cannot show the one thing the horizon is for: "
        "behaviour at scale. This study runs the spectrum on a 150-node graph.\n"
        "\n"
        "## The data-generating process\n"
        "\n"
        "Nodes are scattered uniformly at random in the unit square. Edges "
        "connect pairs within a fixed Euclidean radius. A spanning tree is "
        "overlaid to keep the graph connected. The agent at node $s$ picks among "
        "$A$ nearest neighbours sorted by distance. Actions beyond the node "
        "degree self-loop.\n"
        "\n"
        "The reward for traversing edge $(s, a) \\to s'$ is linear in three "
        "features:\n"
        "\n"
        "$$\n"
        "u_\\theta(s, a) = "
        "\\theta_{\\mathrm{cost}}\\,(-d_{ss'}) + "
        "\\theta_{\\mathrm{am}}\\,\\mathrm{am}(s') + "
        "\\theta_{\\mathrm{goal}}\\,(-\\ell_{s'})\n"
        "$$\n"
        "\n"
        "where $d_{ss'}$ is the Euclidean edge length, $\\mathrm{am}(s')$ is a "
        "node-level amenity draw, and $\\ell_{s'}$ is the shortest-path distance "
        "from $s'$ to a fixed destination node. The true parameters are "
        "$\\theta = [1.0,\\;0.5,\\;1.0]$.\n"
        "\n"
        "Agents discount future payoffs at $\\beta$ and face i.i.d. logit taste "
        "shocks (scale $\\sigma = 1$). Their behaviour solves the soft Bellman "
        "equation. All three parameters are identified from observed route "
        "choices because the features vary across edges, not just states. The "
        "panel scales with the graph (400 agents at 150 nodes) so state coverage "
        "stays adequate. The figure shows simulated paths and the optimal value "
        "function (lower at nodes far from the goal)."
    ),
    "cells": {
        "route_highdim": {
            "after": (
                "The structural family (NFXP, CCP, MPEC) recovers all three "
                "parameters on the same scale as the truth, so Param RMSE applies "
                "to them alone. The RHIP horizon variants and MCE-IRL use the same "
                "linear features but their weights stay out of the recovery table, "
                "because an IRL reward is only partially identified in general. "
                "RHIP-Hinf and MCE-IRL match by construction. Policy TV and regret "
                "are the right scorecards for the behavioral family."
            ),
        },
    },
    "scaling_intro": (
        "The same study at two problem sizes (75 and 150 states). Each line is "
        "one estimator: fit time on the left, policy total variation on the "
        "right. The structural method (NFXP) stays cheap and accurate as the "
        "graph grows. On accuracy the RHIP endpoints bracket the horizon "
        "spectrum: H=0 (deterministic) is the least accurate, H=inf (stochastic) "
        "matches MCE-IRL and is the most accurate. Wall-clock does not track the "
        "horizon here, because the H=inf path reuses the optimized MCE-IRL solver "
        "and so is not the slowest. With only two sizes the lines trace a "
        "direction, not an asymptotic rate."
    ),
    "script": "scripts/study_highdim_route_choice.py",
    "results_rel": "validation/results/study_highdim_route_choice.json",
    "extra_sections": (
        "## Reward and structure\n"
        "\n"
        "The 150-node network sits in the unit square. Each dot is a node, each "
        "line an edge. The color is the optimal value $V^*(s)$ at the true "
        "parameters: value rises toward the goal node (the star). This is the "
        "spatial scale a short planning horizon is built for.\n"
        "\n"
        "![Road network layout colored by optimal value]"
        "(../_static/simulation_studies/highdim_route_choice_network.png)\n"
        "\n"
        "The horizon $H$ is the single knob that spans a family of methods. The "
        "figure traces policy total variation across the four horizons. $H=0$ is "
        "the Max-Margin-Planning end. $H=\\infty$ matches Max Causal Entropy IRL. "
        "Accuracy improves smoothly as the horizon grows. Wall-clock fit time is "
        "not shown: the $H=\\infty$ path reuses the optimized MCE-IRL solver and "
        "runs faster than the finite-horizon path, which would invert the "
        "planning cost the horizon is meant to trade.\n"
        "\n"
        "![Policy total variation across the planning horizon]"
        "(../_static/simulation_studies/highdim_route_choice_horizon.png)\n"
    ),
}


# Construction seed for the road_network geometry (matches ``_env``).
_NETWORK_SEED = 0


def _make_network_fig(data):  # noqa: ARG001 - pure from the env it rebuilds
    from validation.benchmark.figures import network_plot

    network_plot(_env(HEADLINE_NODES), NETWORK_FIG, seed=_NETWORK_SEED)


def _make_horizon_fig(data):
    from validation.benchmark.figures import horizon_frontier

    horizon_frontier(data, "route_highdim", HORIZON_FIG)


EXTRA_FIGURES = [(NETWORK_FIG, _make_network_fig), (HORIZON_FIG, _make_horizon_fig)]


if __name__ == "__main__":
    main_cli(
        cells=CELLS,
        title="Simulation study: high-dimensional route choice on a road network",
        narrative=NARRATIVE,
        diagnoses=DIAGNOSES,
        excluded=EXCLUDED,
        results_json=RESULTS_JSON,
        page_path=PAGE_PATH,
        scaling_figure=SCALING_FIG,
        extra_figures=EXTRA_FIGURES,
    )
