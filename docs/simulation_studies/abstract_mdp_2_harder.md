# Abstract MDP 2: harder

A larger random MDP where the full nested-fixed-point inner solve stops being
cheap. Vanilla NFXP, which solves the Bellman equation to convergence inside
every likelihood evaluation, becomes slow here, but its refinements that avoid or
shortcut the inner solve (CCP, MPEC, NNES, and faster inner solvers) stay
practical. The lesson of this page is where the classic full-solve approach
starts to cost, and which refinements buy the speed back without losing accuracy.

Environment: `random_mdp` at roughly 300 states. The page reports runtime
prominently alongside recovery and counterfactual regret, and states honestly if
an estimator was too slow to include.

```{note}
Status: results pending. This page is generated from
`validation/results/sim_abstract_mdp_2.json` once the run completes.
```
