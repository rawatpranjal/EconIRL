# Deep MaxEnt IRL Paper Context

Primary source: Wulfmeier, Ondruska, and Posner (2015/2016) for maximum-entropy
deep inverse reinforcement learning. Public citations live in
`../../../docs/references.md`. Broader paper routing lives in
`../../papers/index.md`.

## Paper-To-Package Translation

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Neural reward network | State/action embedding MLP | Returns reward matrix. |
| MaxEnt planning | Soft value iteration | Contrib implementation. |
| Visitation matching | State-action visitation comparison | Training signal. |
| Nonlinear reward | Flattened reward matrix output | Not interpretable neural theta. |
| Deep MCE relation | Separate public estimator | Do not merge without validation parity. |
| Release evidence | Missing | Needs tracked validation JSON. |

## Derivation Checklist

1. Define neural reward parameterization.
2. Define soft planning and induced policy.
3. Define state-action visitation frequencies.
4. Derive feature or visitation mismatch loss.
5. Explain reward gauge and neural parameter non-identification.
6. Compare explicitly with Deep MCE-IRL before public docs reuse.

## Release Gap

The estimator has tests but no tracked validation artifact. Treat it as contrib
until reward-map, policy, value/Q, and counterfactual evidence are added.
