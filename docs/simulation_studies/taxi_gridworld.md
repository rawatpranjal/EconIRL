# Taxi gridworld

A gridworld navigation task in the tradition of Ziebart's maximum-entropy inverse
reinforcement learning (Ziebart et al 2008), where the original application was
recovering a taxi driver's route preferences. An agent moves on an N by N grid
toward a goal; the reward trades off a step penalty against reaching the terminal
cell. This is the natural home turf for the entropy-based IRL estimators, so the
roster is weighted toward MaxEnt-IRL, MCE-IRL, and AIRL alongside the structural
methods.

Environment: `GridworldEnvironment`. Roster: structural plus the IRL family.

```{note}
Status: results pending. This page is generated from
`validation/results/sim_taxi_gridworld.json` once the run completes.
```
