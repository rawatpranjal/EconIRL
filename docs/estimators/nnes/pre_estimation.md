# Pre-Estimation Checks

NNES can fail for reasons that are visible before the neural value path starts.
Run these checks before treating a NNES result as structural evidence.

| Check | Why it matters |
| --- | --- |
| Feature rank | Rank below the number of reward parameters means theta is not identified. |
| Feature condition number | A high condition number signals unstable reward estimates. |
| Transition model | NNES is model-based, so transition probabilities must be known or estimated before policy evaluation. |
| Transition row sums | NNES needs valid transition probabilities for continuation values. |
| Logit DDC primitives | The paper's orthogonality result is for additive Type-I extreme-value shocks and separated transition dynamics. |
| State coverage | Unobserved states weaken value-network training and recovery checks. |
| State-action coverage | Sparse action support weakens the likelihood and counterfactual fit. |
| Initial CCP support | The NPL path starts from empirical or first-stage CCPs; weak support makes that start noisy. |
| Reward normalization | Reward level and scale need a valid anchor. |
| Value anchoring | High-discount problems need the value-level normalization to avoid drift. |
| Value-network loss | A high final loss can contaminate recovered policy and value objects. |

## Results Checks

These rows come from the generated NNES results file. See
[Simulation Study](validation.md) for the generator script, rendered table
source, and JSON.

| Check | Low-dimensional | High-dimensional primary |
| --- | ---: | ---: |
| Feature rank | 4 / 4 | 32 / 32 |
| Feature condition number | 4.512 | 1.377 |
| Transition row error | 2.42e-8 | 2.42e-8 |
| Observed states | 21 / 21 | 81 / 81 |
| State-action coverage | 1.000 | 0.959 |
| Minimum action share | 0.325 | 0.281 |
| Outer NPL iterations | 3 | 3 |
| Final V loss | 0.000399 | 0.029932 |

The primary cell is the high-dimensional encoded-state DGP. It checks that the
neural value path still recovers known reward, policy, value, Q, and
counterfactual objects.

## Common Risk Patterns

A low-rank reward matrix can make several reward parameters observationally
equivalent. Sparse action support can make the fitted policy look plausible
while the structural parameters drift. Wrong transition orientation can
produce valid-looking arrays and wrong continuation values.

An unanchored value network can drift in high-discount problems because the
absolute value level is weakly identified by choice data. A neural value
network with poor fit can pass data likelihood checks while failing value, Q,
or counterfactual recovery checks. These risks are why the simulation table
reports both structural recovery and final value-network loss.
