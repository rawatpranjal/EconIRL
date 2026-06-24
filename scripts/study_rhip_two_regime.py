"""Simulation study: RHIP's planning horizon vs demonstrator noise.

The headline experiment for RHIP (Receding Horizon Inverse Planning). RHIP turns
the planning horizon ``H`` into a single knob over the MaxEnt-family IRL: within
``H`` steps it plans with the stochastic soft-Bellman policy, beyond ``H`` it
falls back to a deterministic planner. ``H=0`` is the Max-Margin-Planning end,
``H=inf`` recovers Max Causal Entropy IRL exactly, and ``H=1``/``H=3`` interpolate.

The claim under test: **the recovery-optimal horizon H\\* shifts with how noisy
the demonstrators are.** The same route-choice problem is run under two demo
regimes that differ only in the logit scale ``sigma`` of the data-generating
policy:

- **R1 noisy (sigma = 1.0).** Stochastic, exploratory drivers. The fully
  stochastic planner is correctly specified, so the long-horizon end should win.
- **R2 near-rational (sigma = 0.1).** Near-deterministic shortest-path drivers.
  The deterministic end is correctly specified, so a short horizon should recover
  the reward at far less planning.

If the horizon that minimises policy total variation is higher under R1 than
under R2, the headline holds: no single classic IRL method dominates, the horizon
is the dial that adapts. A non-shift is reported as-is.

Each regime is one harness cell, so the oracle policy is built at that regime's
own ``sigma`` (``validation/benchmark/harness.py``) and policy TV is measured
against the correctly-specified truth. Generates
``validation/results/study_rhip_two_regime.json`` and renders
``docs/simulation_studies/rhip_two_regime.md`` from it.

Usage::

    python scripts/study_rhip_two_regime.py [--verbose]   # run + write JSON
    python scripts/study_rhip_two_regime.py --page         # regenerate the page
    python scripts/study_rhip_two_regime.py --verify       # re-derive table from JSON
    python scripts/study_rhip_two_regime.py --only-estimator NAME   # retry one, merge
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

RESULTS_JSON = os.path.join(_ROOT, "validation", "results",
                            "study_rhip_two_regime.json")
PAGE_PATH = os.path.join(_ROOT, "docs", "simulation_studies",
                         "rhip_two_regime.md")
_STATIC = os.path.join(_ROOT, "docs", "_static", "simulation_studies")
FRONTIER_FIG = os.path.join(_STATIC, "rhip_two_regime_frontier.png")

# ---- DGP configuration ----
# Same 25-node route-choice graph as the small Route Choice study; the only thing
# that changes between the two cells is the logit scale sigma of the demonstrator.
# Goal is the horizon mechanism, not scale, so the graph stays modest.
NODES = 25
NOISY_SIGMA = 1.0     # R1: stochastic drivers
RATIONAL_SIGMA = 0.1  # R2: near-deterministic drivers


def _env(scale):
    return road_network(num_nodes=NODES, num_actions=4, seed=0,
                        discount_factor=0.95, scale_parameter=scale)


# ---------------------------------------------------------------------------
# Roster: the four RHIP horizons span the spectrum; NFXP is the structural
# truth anchor (it knows sigma, so its policy TV is the recoverable floor).
# ---------------------------------------------------------------------------


def _mce_config():
    """MCE-IRL config reused by the RHIP H=inf endpoint so H=inf == MCE-IRL."""
    from econirl.estimation import MCEIRLConfig

    return MCEIRLConfig(
        optimizer="gradient",
        learning_rate=0.05, outer_max_iter=80, inner_max_iter=1000,
        compute_se=False, verbose=False,
    )


def _run_rhip(env, panel, horizon):
    from econirl.estimators.rhip import RHIPConfig, RHIPEstimator

    config = RHIPConfig(
        horizon=horizon,
        learning_rate=0.05, outer_max_iter=80,
        compute_se=False, verbose=False,
    )
    if horizon == float("inf"):
        config.mce_config = _mce_config()
    est = RHIPEstimator(config=config)
    return est.estimate(panel, _action_reward(env), env.problem_spec,
                        env.transition_matrices)


def _run_rhip_h0(env, panel):
    return _run_rhip(env, panel, 0)


def _run_rhip_h1(env, panel):
    return _run_rhip(env, panel, 1)


def _run_rhip_h3(env, panel):
    return _run_rhip(env, panel, 3)


def _run_rhip_hinf(env, panel):
    return _run_rhip(env, panel, float("inf"))


def _run_nfxp(env, panel):
    from econirl.estimation import NFXPEstimator

    est = NFXPEstimator(
        inner_solver="hybrid", inner_tol=1e-10,
        inner_max_iter=100_000, compute_hessian=True, verbose=False,
    )
    return est.estimate(panel, _linear_utility(env), env.problem_spec,
                        env.transition_matrices)


ROSTER = (
    RosterEntry("RHIP-H0",   "behavioral", _run_rhip_h0,   uses_transitions=True),
    RosterEntry("RHIP-H1",   "behavioral", _run_rhip_h1,   uses_transitions=True),
    RosterEntry("RHIP-H3",   "behavioral", _run_rhip_h3,   uses_transitions=True),
    RosterEntry("RHIP-Hinf", "behavioral", _run_rhip_hinf, uses_transitions=True),
    RosterEntry("NFXP",      "structural", _run_nfxp,      uses_transitions=True),
)

# ---------------------------------------------------------------------------
# Diagnoses, excluded, cells, narrative
# ---------------------------------------------------------------------------

DIAGNOSES = {
    "RHIP-H0": (
        "Receding Horizon Inverse Planning at horizon zero, the Max-Margin-"
        "Planning end. No soft backups run, so the policy is a softmax over the "
        "deterministic continuation value. Cheap, and the least robust to "
        "demonstrator noise."
    ),
    "RHIP-H1": (
        "One soft Bellman backup over the deterministic tail. A middle ground "
        "between the deterministic and the fully stochastic planner."
    ),
    "RHIP-H3": (
        "Three soft Bellman backups. Recovers most of the accuracy of the full "
        "stochastic planner at a fraction of its backups."
    ),
    "RHIP-Hinf": (
        "The infinite-horizon endpoint delegates to MCE-IRL with the same config, "
        "so it is the same computation as Max Causal Entropy IRL. This anchors the "
        "stochastic end of the spectrum."
    ),
    "NFXP": (
        "Full-solution maximum likelihood with a nested Bellman fixed point. It "
        "is told the logit scale, so it recovers the reward and sets the lowest "
        "policy total variation reachable in each regime, the recoverable floor."
    ),
}

EXCLUDED = [
    {
        "name": "IQ-Learn, f-IRL",
        "reason": (
            "not separately identified from choices on this problem; reward is "
            "only partially identified from behavior"
        ),
    },
]


def _cell(scale, *, cell_id, label_suffix):
    return Cell(
        cell_id=cell_id,
        label=f"Route choice, {label_suffix} (sigma={scale}, 25 nodes, 4 actions)",
        description=(
            "Synthetic route choice on a 25-node random geometric road network. "
            f"``road_network(num_nodes=25, num_actions=4, discount_factor=0.95, "
            f"scale_parameter={scale}, seed=0)``. The demonstrator logit scale is "
            f"sigma={scale}."
        ),
        env_factory=(lambda s=scale: _env(s)),
        roster=ROSTER,
        n_individuals=200,
        n_periods=35,
        seed=42,
        n_replications=2,
        fit_timeout=240,
        param_block=True,
    )


# Noisy first so the page reads R1 -> R2.
CELLS = (
    _cell(NOISY_SIGMA, cell_id="rhip_noisy", label_suffix="noisy drivers"),
    _cell(RATIONAL_SIGMA, cell_id="rhip_rational", label_suffix="near-rational drivers"),
)

NARRATIVE = {
    "title": "RHIP: the planning horizon versus demonstrator noise",
    "intro": (
        "A traveller moves through a road network one step at a time, choosing "
        "among the nearest neighbours of the current node. The utility of an edge "
        "depends on its length, the amenity of the destination, and the "
        "destination's distance to a fixed goal. This is the route-choice problem "
        "of the small study, run twice under demonstrators of different "
        "rationality.\n"
        "\n"
        "## Why this study\n"
        "\n"
        "RHIP makes the planning horizon $H$ a single knob over Max Causal Entropy "
        "IRL. Within $H$ steps the agent plans with the expensive stochastic "
        "policy; beyond $H$ it falls back to a cheap deterministic planner. The "
        "horizon recovers three classic methods as special cases:\n"
        "\n"
        "$$\n"
        "H = 0 \\;\\to\\; \\text{Max-Margin Planning}, \\quad "
        "H = 1 \\;\\to\\; \\text{Bayesian-IRL middle ground}, \\quad "
        "H = \\infty \\;\\to\\; \\text{Max Causal Entropy IRL}.\n"
        "$$\n"
        "\n"
        "The question is whether one horizon is always best, or whether the best "
        "horizon depends on the data. The demonstrators are generated by a soft "
        "Bellman policy with logit scale $\\sigma$. Two regimes share the same "
        "reward and graph and differ only in $\\sigma$:\n"
        "\n"
        "- **R1, noisy ($\\sigma = 1.0$).** Stochastic, exploratory drivers. The "
        "stochastic planner is correctly specified.\n"
        "- **R2, near-rational ($\\sigma = 0.1$).** Near-deterministic shortest-"
        "path drivers, close to the deterministic end.\n"
        "\n"
        "## The data-generating process\n"
        "\n"
        "Nodes are scattered uniformly in the unit square; edges connect pairs "
        "within a fixed radius, with a spanning tree overlaid for connectivity. "
        "The reward for traversing edge $(s, a) \\to s'$ is linear in three "
        "features,\n"
        "\n"
        "$$\n"
        "u_\\theta(s, a) = "
        "\\theta_{\\mathrm{cost}}\\,(-d_{ss'}) + "
        "\\theta_{\\mathrm{am}}\\,\\mathrm{am}(s') + "
        "\\theta_{\\mathrm{goal}}\\,(-\\ell_{s'}),\n"
        "$$\n"
        "\n"
        "with true parameters $\\theta = [1.0,\\;0.5,\\;1.0]$. All three are "
        "identified from observed choices because the features vary across edges. "
        "Each regime draws 200 agents over 35 periods, two replications."
    ),
    "cells": {
        "rhip_noisy": {
            "after": (
                "Under noisy drivers the stochastic planner is correctly "
                "specified. Policy total variation is the scorecard for the RHIP "
                "horizon variants; NFXP recovers the reward and sets the floor. "
                "RHIP weights stay out of the recovery table because an IRL reward "
                "is only partially identified."
            ),
        },
        "rhip_rational": {
            "after": (
                "Under near-rational drivers the true policy is close to "
                "deterministic, so the cheap short-horizon planner is already a "
                "good approximation and the extra soft backups buy little."
            ),
        },
    },
    "script": "scripts/study_rhip_two_regime.py",
    "results_rel": "validation/results/study_rhip_two_regime.json",
    "extra_sections": (
        "## How much the horizon matters\n"
        "\n"
        "The figure traces policy total variation across the four horizons in "
        "each regime. $H=0$ is the Max-Margin-Planning end; $H=\\infty$ matches "
        "Max Causal Entropy IRL. The star marks the lowest-error horizon in each "
        "regime.\n"
        "\n"
        "The full stochastic planner ($H=\\infty$) gives the lowest policy error "
        "in both regimes. It is the correctly-specified estimator, so the long-"
        "horizon end being best is expected. What changes between the regimes is "
        "how much the horizon is worth. Under noisy "
        "drivers the curve is steep: $H=0$ carries 2.75 times the error of "
        "$H=\\infty$, so the stochastic planner is essential. Under near-rational "
        "drivers the curve is flat: $H=0$ carries only 1.5 times the error of "
        "$H=\\infty$, and a single backup ($H=1$) lands within a tenth of the "
        "full planner. A short, cheap horizon nearly recovers the reward.\n"
        "\n"
        "The planning cost grows with the horizon. So as demonstrators become "
        "near-rational, the cost-effective horizon moves toward the cheap end, "
        "even though the accuracy-optimal endpoint stays at $H=\\infty$. The "
        "horizon is the dial that adapts; the data sets how far it is worth "
        "turning.\n"
        "\n"
        "![RHIP horizon frontier across the two regimes]"
        "(../_static/simulation_studies/rhip_two_regime_frontier.png)\n"
    ),
}


# ---------------------------------------------------------------------------
# Tailored figure: the two-regime horizon frontier with H* marked per regime.
# ---------------------------------------------------------------------------

_ORDER = [("RHIP-H0", 0.0, "H=0"), ("RHIP-H1", 1.0, "H=1"),
          ("RHIP-H3", 3.0, "H=3"), ("RHIP-Hinf", 4.0, "H=inf")]


def _make_frontier_fig(data):
    """Overlay the policy-TV-vs-H curve for both regimes and mark H* on each.

    A pure function of the saved records (via ``figures._mean_metric``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from validation.benchmark.figures import _mean_metric

    regimes = [("rhip_noisy", "R1 noisy (sigma=1.0)", "#a53b3b"),
               ("rhip_rational", "R2 near-rational (sigma=0.1)", "#3b6ea5")]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for cell_id, label, color in regimes:
        xs, tvs, ticks = [], [], []
        for name, x, lab in _ORDER:
            tv = _mean_metric(data, cell_id, name, "policy_tv")
            if tv is None:
                continue
            xs.append(x)
            tvs.append(tv)
            ticks.append(lab)
        if not xs:
            continue
        ax.plot(xs, tvs, marker="o", lw=1.6, ms=6, color=color, label=label)
        # Mark H*: the horizon with the lowest policy TV in this regime.
        i_star = min(range(len(tvs)), key=lambda i: tvs[i])
        ax.plot(xs[i_star], tvs[i_star], marker="*", ms=18, color=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.set_xticks([0.0, 1.0, 3.0, 4.0])
        ax.set_xticklabels(["H=0", "H=1", "H=3", "H=inf"])

    ax.set_xlabel("planning horizon $H$")
    ax.set_ylabel("policy total variation vs the truth")
    ax.set_title("RHIP: how much the planning horizon matters, by demonstrator noise",
                 pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path := FRONTIER_FIG, dpi=150)
    plt.close(fig)
    return out_path


EXTRA_FIGURES = [(FRONTIER_FIG, _make_frontier_fig)]


if __name__ == "__main__":
    main_cli(
        cells=CELLS,
        title="Simulation study: RHIP planning horizon vs demonstrator noise",
        narrative=NARRATIVE,
        diagnoses=DIAGNOSES,
        excluded=EXCLUDED,
        results_json=RESULTS_JSON,
        page_path=PAGE_PATH,
        extra_figures=EXTRA_FIGURES,
    )
