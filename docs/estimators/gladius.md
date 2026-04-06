# GLADIUS

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Model-Free | Kang, Yoganarasimhan, and Jain (2025) | Linear (projected) | No | No | Yes |

## What this estimator does

Structural estimators like NFXP and CCP require enumerating the full state space and storing transition matrices. IRL estimators like MCE-IRL also require transitions, while AIRL lacks interpretable structural parameters. Kang, Yoganarasimhan, and Jain (2025) introduce GLADIUS, which parameterizes the Q-function and expected continuation value with neural networks, trains them via alternating SGD, and then projects the learned neural Q-values onto linear features to recover interpretable structural parameters.

The key theoretical result is a Polyak-Lojasiewicz condition on the composite objective (NLL plus Bellman MSE) that guarantees global convergence. A dual variable $\zeta(s,a) \approx \mathbb{E}[V(s')|s,a]$ separates the irreducible transition variance from the Bellman constraint, eliminating double-sampling bias. However, without observed rewards, Q is trained via NLL only and the Bellman penalty does not flow gradients through Q. This means Q-values are identified only up to a state-dependent constant $c(s)$ that propagates asymmetrically through transitions. On the Rust bus, the replacement cost is recovered within 8 percent but the operating cost is overestimated by roughly 40 percent. This is a structural limitation, not a tuning problem.

## How it works

After training the Q-network and EV-network by alternating SGD, the implied reward is $\hat{r}(s,a) = Q(s,a) - \beta \zeta(s,a)$. Because Q is identified only up to a state-dependent constant, structural parameters are recovered by action-difference projection:

$$
\hat\theta = (\Delta\Phi^\top \Delta\Phi)^{-1} \Delta\Phi^\top \Delta\hat{r},
$$

where $\Delta$ denotes action differences that cancel the unidentified constant. The projection $R^2$ measures how well the neural reward surface is explained by linear features. Standard errors are not available analytically because the neural training loop does not produce a well-defined likelihood.

## When to use it

GLADIUS is the appropriate choice when the state space is too large for tabular methods and the analyst wants to check whether linear features are sufficient via the $R^2$ diagnostic. When transitions are known and the state space is manageable, NFXP recovers both parameters within 5 percent and is strictly preferable. The structural identification bias in the IRL setting means GLADIUS should be used for continuous-state environments or when rewards are observed in the data, which is the paper's intended use case where the bi-conjugate Bellman error anchors Q-values.

## References

- Kang, J., Yoganarasimhan, H., and Jain, V. (2025). GLADIUS: ERM for Offline IRL. Working paper.

The full derivation, algorithm, and simulation results are in the [GLADIUS primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/gladius.pdf).
