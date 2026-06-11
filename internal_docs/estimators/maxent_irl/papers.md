# MaxEnt IRL Paper Context

Primary sources: Ng and Russell (2000), Abbeel and Ng (2004), and Ziebart et
al. (2008). Use this folder for the non-causal or legacy maximum-entropy
feature-matching estimator, and keep it distinct from MCE-IRL.

Public citations live in `../../../docs/references.md`. Broader paper routing
lives in `../../papers/index.md`.

## Paper-To-Package Translation

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Trajectory distribution | Demonstration trajectories | Used for feature-count matching. |
| Cumulative reward | Sum of state/action features along a trajectory | Distinct from causal soft-Bellman reward use. |
| Partition function | Normalization over feasible trajectories | Can be expensive or environment-specific. |
| Feature expectation matching | Gradient signal | Expert features minus model features. |
| Maximum entropy ambiguity | Baseline behavior model | Not automatically structural for DDC timing. |
| Learned reward | Wrapper/contrib estimator output | Should be compared cautiously and under a gauge. |

## Distinction To Preserve

Do not collapse these estimators:

- MaxEnt IRL: entropy over full trajectories.
- MCE-IRL: causal entropy over sequential decisions.
- Deep MCE-IRL: neural reward map inside the causal MCE contract.
- Deep MaxEnt IRL: neural reward in the older trajectory MaxEnt family.

This distinction matters for the package positioning. The released comparison
package should treat MCE-IRL as the entropy method that aligns with structural
dynamic discrete choice timing.

## Old Primer Material To Retain

The retired primer had useful contrast material:

- the Boltzmann trajectory distribution;
- the trajectory feature-count gradient;
- the non-causal limitation;
- a gridworld comparison against MCE and behavioral cloning.

Do not restore the TeX primer. Preserve any needed derivations here or in
`index.md`.
