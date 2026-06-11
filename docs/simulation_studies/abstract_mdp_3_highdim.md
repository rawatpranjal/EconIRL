# Abstract MDP 3: high-dimensional

A super-high-dimensional random MDP. At this scale the dense transition tensor and
the repeated exact Bellman solves that NFXP and MPEC depend on are infeasible, so
those estimators are omitted by design, and the page says so rather than reporting
a number they could not produce. What remains feasible are the machine-learning
powered estimators that approximate the value function or the policy with neural
networks (TD-CCP, GLADIUS, NNES, Deep MCE-IRL). The lesson is that scale forces a
move from exact structural solves to learned approximations.

Environment: `random_mdp` (or Shapeshifter) at a few thousand states. The page
states explicitly which estimators were omitted and why.

```{note}
Status: results pending. This page is generated from
`validation/results/sim_abstract_mdp_3.json` once the run completes.
```
