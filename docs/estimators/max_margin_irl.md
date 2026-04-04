# Max Margin IRL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Abbeel and Ng (2004), Ratliff et al. (2006) | Linear | Yes | No | No |

## Background

Abbeel and Ng (2004) framed IRL as an apprenticeship learning problem. Rather than recovering the exact reward function, the goal is to find reward weights under which the expert policy outperforms every alternative policy by the largest possible margin. The algorithm works by iteratively generating "violating" policies that compete with the expert, then solving a quadratic program to find reward weights that maximize the separation between the expert's feature expectations and those of the most competitive alternative.

The constraint generation loop is the core mechanism. At each iteration, the algorithm solves the MDP under the current reward weights to find the best response policy. If that policy's feature expectations are close to the expert's, the algorithm has converged. Otherwise, the new policy is added to the constraint set and the QP is re-solved. The reward weights are constrained to have unit norm to avoid the trivial solution of setting all weights to zero.

## Key Equations

$$
\max_{\theta, \, t} \; t
\quad \text{s.t.} \quad \theta^\top \bar\mu_E - \theta^\top \bar\mu_i \geq t \;\; \forall\, i, \quad \|\theta\|_2 \leq 1,
$$

where $\bar\mu_E$ is the expert's feature expectation vector and $\bar\mu_i$ is the feature expectation of the $i$-th violating policy. The variable $t$ is the margin.

## Pseudocode

```
MaxMarginIRL(D, phi, p, beta, sigma, max_iterations):
  1. Compute expert feature expectations mu_E from D
  2. Initialize theta randomly, constraint set C = {}
  3. For each iteration:
     a. Solve MDP under R(s) = theta . phi(s) to get policy pi_i
     b. Compute feature expectations mu_i under pi_i
     c. If ||mu_E - mu_i|| < tol, stop
     d. Add mu_i to constraint set C
     e. Solve QP: max t s.t. theta'*mu_E - theta'*mu_j >= t for all j in C,
        ||theta|| <= 1
     f. Update theta from QP solution
  4. Return theta, achieved margin t
```

## Strengths and Limitations

Max Margin IRL requires no probabilistic model of the demonstrator's decision process. It only assumes the expert is approximately optimal and finds the reward that makes the expert look best relative to alternatives. The constraint generation is finite and typically converges in a small number of iterations. The QP is small because only the feature expectation vectors enter the constraints, not the full policies.

The limitation is that Max Margin IRL does not produce a likelihood, so there is no natural way to compute standard errors, perform hypothesis tests, or compare models via information criteria. The recovered reward is also not unique. Any reward in the margin polytope is consistent with the data, and the algorithm picks one based on the norm constraint rather than a statistical criterion. The unit-norm constraint means the scale of the reward is arbitrary. For formal inference and model comparison, likelihood-based methods like MCE-IRL or NFXP are preferable.

## References

- Abbeel, P. and Ng, A. Y. (2004). Apprenticeship Learning via Inverse Reinforcement Learning. *Proceedings of the 21st International Conference on Machine Learning (ICML)*.
- Ratliff, N., Bagnell, J. A., and Zinkevich, M. (2006). Maximum Margin Planning. *Proceedings of the 23rd International Conference on Machine Learning (ICML)*.
