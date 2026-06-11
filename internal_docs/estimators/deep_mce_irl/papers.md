# Deep MCE-IRL Paper Context

Primary sources: Ziebart (2010) for causal entropy IRL and Wulfmeier,
Ondruska, and Posner (2015) for neural maximum-entropy reward learning. Public
citations live in `../../../docs/references.md`. Broader paper routing lives in
`../../papers/index.md`.

## Paper-To-Package Translation

The paper-level idea is to learn a reward representation rich enough to capture
nonlinear structure while retaining entropy-regularized planning. The package
implementation keeps the planning side causal through the MCE solver and
evaluates the learned object through reward-map and behavior metrics.

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Deep reward model | Neural reward map `f_eta(x)` | Validate reward matrix, not raw weights. |
| Maximum entropy planning | MCE soft Bellman solver | The maintained comparison is causal MCE, not non-causal trajectory MaxEnt. |
| Demonstration likelihood | Occupancy/moment objective | Reported through residuals and induced policy quality. |
| Reward ambiguity | Anchor action and normalization | Required before reward-map comparison. |
| Learned representation | Supplied encodings or frozen neural features | Current validation controls the encoding environment. |
| Parameter recovery | Projection-specific finite-theta check | Meaningful only in identified projected cells. |
| Generalization | Counterfactual perturbations | Local validation in known simulated environments. |

## Internal Derivation Tasks

When expanding or revising this page, keep these derivations separate:

1. MCE base objective and soft Bellman recursion.
2. Neural reward-map parameterization.
3. Neural feature plus linear reward parameterization.
4. Reward anchoring and gauge comparison.
5. Projected reward matrix construction.
6. Why raw neural weights are not comparable across equivalent networks.
7. How reward-map error, policy TV, value error, Q error, and counterfactual
   regret answer different validation questions.

## Assumptions To Preserve

- Transitions are known in the current validation cells.
- Encodings are supplied by the validation design.
- The optimizer target is an MCE-induced occupancy objective.
- Reward comparisons require a fixed gauge.
- Neural-parameter comparisons are not structural unless an explicit finite
  projection is identified.

## Old Primer Material To Retain

The retired primer correctly emphasized:

- the neural reward model;
- occupancy mismatch as the behavioral fitting target;
- anchor-action gauge handling;
- separation between reward matrices, neural weights, and projected rewards;
- validation by policy, value, Q, and counterfactual metrics.

Do not restore the old TeX workflow. Bring any missing detail into Markdown
under `index.md`, this file, or public RTD source.
