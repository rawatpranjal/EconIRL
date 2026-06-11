# MCE-IRL Paper Context

Primary sources: Ziebart et al. (2008) for maximum-entropy IRL and Ziebart
(2010) for maximum causal entropy occupancy matching. Public citations live in
`../../../docs/references.md`. Broader paper routing lives in
`../../papers/index.md`.

## Paper-To-Package Translation

The paper object is a causal-entropy distribution over policies in a controlled
Markov process. The package object is a finite-state estimator with explicit
transition arrays, feature arrays, fitted reward parameters, and validation
artifacts.

Map paper concepts into package concepts as follows:

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Reward function | `r_theta(s, a) = phi(s, a)' theta` | Linear in the current maintained release cell. |
| Causal entropy | Soft Bellman recursion | Implemented through log-sum-exp value iteration. |
| Expected feature counts | Discounted occupancy moments | Reported through feature and occupancy residuals. |
| Expert demonstrations | State-action panel | Converted into empirical occupancy and feature moments. |
| Known dynamics | Transition tensor | Treated as fixed in current validation. |
| Reward ambiguity | Anchor and normalization | Required before reward and theta comparisons. |
| Policy recovery | Soft-optimal policy under fitted reward | Main behavioral object for validation. |
| Counterfactuals | Re-solved MDP under perturbations | Valid only under support and transition assumptions. |

## Derivation Checklist

Internal derivations should keep these steps explicit:

1. Define the finite MDP, discount factor, transition tensor, reward basis, and
   expert occupancy.
2. State the causal entropy objective and the feature-matching constraint.
3. Derive the soft Bellman equations.
4. Show that the policy is `exp(Q - V)`.
5. Express the gradient as expert feature moments minus model feature moments.
6. Explain the reward gauge and the package normalization.
7. Connect the fitted object to policy, value, Q, and counterfactual metrics.

## Assumptions To Preserve

- Transitions are known or estimated outside this estimator.
- Observations are encoded in the same state-action system as the transition
  tensor.
- Feature moments are informative after anchoring.
- The soft-optimal policy is the maintained behavioral model.
- Counterfactuals are structural only if the fitted reward and transition
  primitives remain valid under the perturbation.

## Old Primer Material To Retain

The retired primer contained the right high-level ingredients:

- causal entropy rather than non-causal trajectory entropy;
- the soft Bellman recursion;
- feature-moment matching;
- equivalence intuition with dynamic discrete choice likelihoods;
- identification limits from reward shaping and constants;
- a validation run based on known low/high rewards.

Do not restore the TeX primer or paper build workflow. Move any missing useful
content into `index.md`, this file, or public RTD source as Markdown.
