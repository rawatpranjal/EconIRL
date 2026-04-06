# Max Margin IRL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Abbeel and Ng (2004) | Linear | Yes | No | No |

## What this estimator does

Abbeel and Ng (2004) framed IRL as an apprenticeship learning problem. Rather than recovering the exact reward function, the goal is to find reward weights under which the expert policy outperforms every alternative policy by the largest possible margin. The algorithm works by iteratively generating violating policies that compete with the expert, then solving a quadratic program to find reward weights that maximize the separation. The reward weights are constrained to have unit norm to avoid the trivial solution.

The constraint generation loop converges in a small number of iterations. At each step, the algorithm solves the MDP under the current reward to find the best response policy. If that policy's feature expectations are close to the expert's, the algorithm has converged. Otherwise the new policy is added to the constraint set and the QP is re-solved. The recovered reward is not unique. Any reward in the margin polytope is consistent with the data.

## How it works

The algorithm solves

$$
\max_{\theta, t} \; t \quad \text{s.t.} \quad \theta^\top \bar\mu_E - \theta^\top \bar\mu_i \geq t \;\; \forall i, \quad \|\theta\|_2 \leq 1,
$$

where $\bar\mu_E$ is the expert's feature expectation and $\bar\mu_i$ is the feature expectation of the $i$-th violating policy. The QP is small because only the feature expectation vectors enter the constraints, not the full policies.

## When to use it

Max Margin IRL requires no probabilistic model of the demonstrator's decision process. It only assumes the expert is approximately optimal. The limitation is the absence of a likelihood, so there is no natural way to compute standard errors, perform hypothesis tests, or compare models. The unit-norm constraint also makes the scale of the reward arbitrary. For formal inference and model comparison, MCE-IRL or NFXP are preferable. For a more scalable margin-based approach, see Max Margin Planning.

## References

- Abbeel, P. and Ng, A. Y. (2004). Apprenticeship Learning via Inverse Reinforcement Learning. *ICML 2004*.

A primer for this estimator is not yet available.
