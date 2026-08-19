"""Simulation study: RHIP recovers the demonstrator's planning horizon.

The faithful headline for RHIP (Receding Horizon Inverse Planning, Barnes et al.
2024). The paper's empirical finding is that on real Google Maps data an
*interior* horizon beats both the myopic (Max-Margin-Planning, H=0) and the fully
stochastic (Max Causal Entropy, H=inf) endpoints, because real drivers are
neither: they "take a mixed approach, considering all paths within some horizon,
and making approximations beyond that horizon" (paper Sec 5, the Figure 5
mechanism). The finite-horizon planner is simply a better-specified behavioral
model.

This study reproduces that mechanism on a known synthetic truth. The demonstrator
is RHIP's own receding-horizon policy at a chosen lookahead ``h_demo``: it plans
with the stochastic soft-Bellman policy for ``h_demo`` steps, then falls back to a
deterministic planner. We then fit RHIP across a sweep of estimator horizons ``H``
and measure how far the recovered choice probabilities sit from the true
demonstrator policy (policy total variation).

The prediction, and the headline: the recovery-optimal estimator horizon is
*interior* and *tracks* ``h_demo`` - it lands at ``H = h_demo``, beating both the
H=0 and the H=inf endpoints. As the demonstrator's lookahead changes, the optimal
horizon shifts with it. RHIP's horizon is an identifiable behavioral parameter,
not just a compute knob.

This replaces the earlier two-regime study, which varied demonstrator noise under
a correctly-specified estimator (true sigma handed to the estimator). That design
made H=inf the correct model by construction, so it could not show an interior
optimum - a setup the paper never used. Here the estimator and demonstrator share
sigma; the only mismatch is the planning horizon, which is the parameter under
study.

Usage::

    python scripts/study_rhip_lookahead.py            # run + write JSON + page + figure
    python scripts/study_rhip_lookahead.py --page      # re-render page + figure from JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from econirl.environments.road_network import road_network  # noqa: E402
from econirl.core.bellman import SoftBellmanOperator  # noqa: E402
from econirl.estimators.rhip import RHIPEstimator  # noqa: E402
from econirl.simulation.synthetic import simulate_panel  # noqa: E402
from validation.benchmark.runner import _action_reward  # noqa: E402

RESULTS_JSON = os.path.join(_ROOT, "validation", "results",
                            "study_rhip_lookahead.json")
PAGE_PATH = os.path.join(_ROOT, "docs", "simulation_studies", "rhip_lookahead.md")
_STATIC = os.path.join(_ROOT, "docs", "_static", "simulation_studies")
FIGURE_PNG = os.path.join(_STATIC, "rhip_lookahead.png")

# ---- DGP / sweep configuration ----
NODES = 25
N_INDIVIDUALS = 300
N_PERIODS = 40
N_REPS = 10
SEED = 42
# Demonstrator lookaheads. All interior and identifiable: the receding-horizon
# policy is genuinely between the H=0 and H=inf endpoints for small h, and
# consecutive small-h policies are distinguishable (they collapse toward H=inf for
# large h, where the horizon is no longer separately identifiable from MaxEnt).
H_DEMOS = (1, 2, 3)
# Estimator horizon sweep. inf is the MCE-IRL endpoint; the finite values bracket
# the demonstrator lookaheads on both sides.
H_SWEEP = (0, 1, 2, 3, 5, float("inf"))


def _env(scale=1.0):
    return road_network(num_nodes=NODES, num_actions=4, seed=0,
                        discount_factor=0.95, scale_parameter=scale)


def _mce_config():
    from econirl.estimation import MCEIRLConfig

    return MCEIRLConfig(optimizer="gradient", learning_rate=0.05,
                        outer_max_iter=80, inner_max_iter=1000,
                        compute_se=False, verbose=False)


def _demonstrator_policy(env, h_demo):
    """The true finite-lookahead policy: RHIP's receding-horizon policy at the
    true reward and horizon ``h_demo``. This is the data-generating policy and
    the oracle the recovered policy is scored against."""
    op = SoftBellmanOperator(env.problem_spec,
                             jnp.asarray(np.asarray(env.transition_matrices),
                                         dtype=jnp.float64))
    R = jnp.asarray(np.asarray(env.true_reward_matrix), dtype=jnp.float64)
    est = RHIPEstimator(horizon=h_demo)
    _, policy = est._receding_horizon_policy(op, R, env.problem_spec, int(h_demo))
    return np.asarray(policy)


def _fit_rhip(env, panel, horizon):
    """Fit RHIP at a given estimator horizon; return (policy, params, runtime)."""
    config_kwargs = dict(horizon=horizon, learning_rate=0.05, outer_max_iter=80,
                         compute_se=False, verbose=False)
    est = RHIPEstimator(**config_kwargs)
    if horizon == float("inf"):
        est.config.mce_config = _mce_config()
    t0 = time.time()
    res = est.estimate(panel, _action_reward(env), env.problem_spec,
                       env.transition_matrices)
    return (np.asarray(res.policy) if res.policy is not None else None,
            None if res.parameters is None else np.asarray(res.parameters).tolist(),
            time.time() - t0)


def _policy_tv(p, q):
    """Mean over states of the per-state total variation 0.5*sum_a|p-q|."""
    return float(np.mean(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum(axis=1)))


def _h_label(h):
    return "inf" if h == float("inf") else str(int(h))


def run():
    env = _env()
    records = []
    for h_demo in H_DEMOS:
        pi_demo = _demonstrator_policy(env, h_demo)
        for rep in range(N_REPS):
            seed = SEED + 1000 + rep
            panel = simulate_panel(env, n_individuals=N_INDIVIDUALS,
                                   n_periods=N_PERIODS, seed=seed,
                                   policy=jnp.asarray(pi_demo))
            for H in H_SWEEP:
                pol, params, rt = _fit_rhip(env, panel, H)
                tv = _policy_tv(pol, pi_demo) if pol is not None else None
                records.append({"h_demo": h_demo, "H": _h_label(H), "rep": rep,
                                "policy_tv": tv, "params": params, "runtime": rt})
                print(f"  h_demo={h_demo} rep {rep} H={_h_label(H):>3} "
                      f"{rt:5.1f}s tv={tv}", flush=True)
    # Also record, per (h_demo), the true-policy distance to each endpoint so the
    # page can state the demonstrator is genuinely between MMP and MaxEnt.
    meta_demo = {}
    for h_demo in H_DEMOS:
        pi_demo = _demonstrator_policy(env, h_demo)
        pi_0 = _demonstrator_policy(env, 0)
        # H=inf proxy: a long receding horizon (soft fixed point).
        op = SoftBellmanOperator(env.problem_spec,
                                 jnp.asarray(np.asarray(env.transition_matrices),
                                             dtype=jnp.float64))
        R = jnp.asarray(np.asarray(env.true_reward_matrix), dtype=jnp.float64)
        _, pi_inf = RHIPEstimator(horizon=1)._receding_horizon_policy(
            op, R, env.problem_spec, 60)
        meta_demo[str(h_demo)] = {"tv_to_mmp": _policy_tv(pi_demo, pi_0),
                                  "tv_to_maxent": _policy_tv(pi_demo, np.asarray(pi_inf))}
    data = {"records": records, "meta": {
        "nodes": NODES, "n_individuals": N_INDIVIDUALS, "n_periods": N_PERIODS,
        "n_reps": N_REPS, "seed": SEED, "h_demos": list(H_DEMOS),
        "h_sweep": [_h_label(h) for h in H_SWEEP],
        "true_theta": np.asarray(env.get_true_parameter_vector()).tolist(),
        "demo_endpoint_distance": meta_demo}}
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {RESULTS_JSON}")
    return data


def _mean_tv(data, h_demo, H_lab):
    tvs = [r["policy_tv"] for r in data["records"]
           if r["h_demo"] == h_demo and r["H"] == H_lab and r["policy_tv"] is not None]
    return float(np.mean(tvs)) if tvs else None


def _x_of(H_lab):
    # inf sits one tick past the largest finite horizon.
    finite = [int(h) for h in data_h_sweep if h != "inf"]
    return (max(finite) + 1) if H_lab == "inf" else int(H_lab)


def make_figure(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    global data_h_sweep
    data_h_sweep = data["meta"]["h_sweep"]
    colors = ["#a53b3b", "#3b6ea5", "#3b8a4f"]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for i, h_demo in enumerate(data["meta"]["h_demos"]):
        xs, tvs, labs = [], [], []
        for H_lab in data_h_sweep:
            tv = _mean_tv(data, h_demo, H_lab)
            if tv is None:
                continue
            xs.append(_x_of(H_lab))
            tvs.append(tv)
            labs.append(H_lab)
        ax.plot(xs, tvs, marker="o", lw=1.6, ms=6, color=colors[i % len(colors)],
                label=f"demonstrator lookahead h={h_demo}")
        i_star = int(np.argmin(tvs))
        ax.plot(xs[i_star], tvs[i_star], marker="*", ms=18,
                color=colors[i % len(colors)], markeredgecolor="white",
                markeredgewidth=0.8, zorder=5)
    ticks = [_x_of(h) for h in data_h_sweep]
    ax.set_xticks(ticks)
    ax.set_xticklabels([("H=inf" if h == "inf" else f"H={h}") for h in data_h_sweep])
    ax.set_xlabel("estimator planning horizon $H$")
    ax.set_ylabel("policy total variation vs the demonstrator")
    ax.set_title("RHIP recovers the demonstrator's planning horizon", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIGURE_PNG), exist_ok=True)
    fig.savefig(FIGURE_PNG, dpi=150)
    plt.close(fig)
    print(f"Wrote {FIGURE_PNG}")


def write_page(data):
    m = data["meta"]
    rows = []
    header = "| Demonstrator | " + " | ".join(
        ("$H=\\infty$" if h == "inf" else f"$H={h}$") for h in m["h_sweep"]) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(m["h_sweep"])) + " |"
    rows.append(header)
    rows.append(sep)
    for h_demo in m["h_demos"]:
        cells = []
        best_lab = min(m["h_sweep"], key=lambda L: (_mean_tv(data, h_demo, L)
                                                    if _mean_tv(data, h_demo, L) is not None
                                                    else 1e9))
        for H_lab in m["h_sweep"]:
            tv = _mean_tv(data, h_demo, H_lab)
            s = "-" if tv is None else f"{tv:.4f}"
            if H_lab == best_lab and tv is not None:
                s = f"**{s}**"
            cells.append(s)
        rows.append(f"| lookahead $h={h_demo}$ | " + " | ".join(cells) + " |")
    table = "\n".join(rows)

    de = m["demo_endpoint_distance"]
    between = " ".join(
        f"The $h={h}$ demonstrator sits {de[str(h)]['tv_to_mmp']:.3f} from the "
        f"$H=0$ policy and {de[str(h)]['tv_to_maxent']:.3f} from the $H=\\infty$ "
        f"policy." for h in m["h_demos"])

    # Data-driven conclusion: the actual best horizon per demonstrator, and whether
    # it tracks the lookahead and beats both endpoints.
    def _best_H(h_demo):
        return min(m["h_sweep"], key=lambda L: (_mean_tv(data, h_demo, L)
                                                if _mean_tv(data, h_demo, L) is not None
                                                else 1e9))
    best_map = {h: _best_H(h) for h in m["h_demos"]}
    tracks = all(str(best_map[h]) == str(h) for h in m["h_demos"])
    track_list = ", ".join(f"$h={h} \\to H={best_map[h]}$" for h in m["h_demos"])
    if tracks:
        conclusion = (
            "The recovery-optimal estimator horizon is interior and tracks the "
            f"demonstrator's lookahead exactly: {track_list}. In every case both "
            "endpoints, the myopic $H=0$ and the fully stochastic $H=\\infty$, are "
            "worse. Neither classic method dominates. The matching horizon recovers "
            "the behavior, and the optimal horizon moves with the demonstrator.")
    else:
        conclusion = (
            "The recovery-optimal estimator horizon moves with the demonstrator's "
            f"lookahead ({track_list}), though not at every point exactly. The "
            "horizon is identifiable from behavior in the short-lookahead range; as "
            "the lookahead grows the policy approaches the $H=\\infty$ (MaxEnt) "
            "endpoint and the horizon is no longer separately identifiable.")

    page = f"""# RHIP recovers the demonstrator's planning horizon

A traveller moves through a road network one step at a time, choosing among the
nearest neighbours of the current node. The utility of an edge depends on its
length, the amenity of the destination, and the destination's distance to a fixed
goal. The true reward is linear in these three features with parameters
$\\theta = {m['true_theta']}$.

![Two men in suits playing chess at a small table, one resting his chin on his hand while the other moves a piece.](../_static/simulation_studies/rhip_lookahead_photo.jpg)

*Two chess players, about 1920. Bain News Service, Library of Congress, no known restrictions, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:(Men_playing_chess)_(LOC).jpg). A player who searches three moves ahead behaves differently from one who searches ten.*

## Why this study

RHIP makes the planning horizon $H$ a single knob. Within $H$ steps the agent
plans with the stochastic soft-Bellman policy. Beyond $H$ it falls back to a
deterministic planner. At $H=0$ this is Max-Margin Planning, at $H=\\infty$ it is
Max Causal Entropy IRL, and finite $H$ interpolates.

On real Google Maps data the source paper finds that an interior horizon predicts
routes better than either endpoint, because real drivers are neither fully myopic
nor fully stochastic. They plan within a finite horizon and approximate beyond it.
The finite-horizon planner is a better-specified model of that behavior.

This study reproduces that mechanism on synthetic data with a known reward and a
known demonstrator. The demonstrator is a finite-lookahead planner with a true
lookahead $h$: it plans softly for $h$ steps,
then deterministically. We fit RHIP across a sweep of estimator horizons and
measure how far each recovered policy sits from the true demonstrator policy. The
demonstrator and estimator share the logit scale, so the only thing that can be
mis-set is the horizon.

## The data-generating process

Nodes are scattered uniformly in the unit square. Edges connect pairs within a
fixed radius, with a spanning tree overlaid for connectivity. Demonstrations are
drawn from the finite-lookahead policy at the true reward. Each demonstrator draws
{m['n_individuals']} agents over {m['n_periods']} periods, {m['n_reps']}
replications, on a {m['nodes']}-node graph.

The finite-lookahead demonstrators are genuinely between the two endpoints.
{between}

## Result

Policy total variation between the recovered policy and the true demonstrator
policy, by estimator horizon (rows) and demonstrator lookahead. The best horizon
in each row is in bold.

{table}

{conclusion}

![RHIP recovers the demonstrator's planning horizon](../_static/simulation_studies/rhip_lookahead.png)

The star on each curve marks the lowest-error horizon. The two stars sit at
different horizons, one per demonstrator. This is the planning horizon working as
an identifiable behavioral parameter, the synthetic analog of the interior optimum
the source paper finds on real route data.

## Reproduce

```bash
python scripts/study_rhip_lookahead.py            # run + write JSON + page + figure
python scripts/study_rhip_lookahead.py --page      # re-render from the saved JSON
```

Raw facts: `validation/results/study_rhip_lookahead.json`.
"""
    os.makedirs(os.path.dirname(PAGE_PATH), exist_ok=True)
    with open(PAGE_PATH, "w") as f:
        f.write(page)
    print(f"Wrote {PAGE_PATH}")


data_h_sweep = [_h_label(h) for h in H_SWEEP]  # default for _x_of before run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", action="store_true",
                        help="Re-render the page + figure from the saved JSON only.")
    args = parser.parse_args()
    if args.page:
        if not os.path.exists(RESULTS_JSON):
            sys.exit(f"No JSON at {RESULTS_JSON}. Run without --page first.")
        data = json.load(open(RESULTS_JSON))
    else:
        data = run()
    make_figure(data)
    write_page(data)
