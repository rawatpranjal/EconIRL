# Max Margin Planning

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Ratliff, Bagnell, and Zinkevich (2006) | Linear | Yes | Bootstrap only | No |

## What this estimator does

Max Margin IRL (Abbeel and Ng 2004) learns reward weights via a QP with constraint generation, but the QP grows with each iteration and becomes expensive for large state spaces. Ratliff, Bagnell, and Zinkevich (2006) reformulated the problem using subgradient descent with loss-augmented inference. Instead of accumulating constraints in a growing QP, each iteration solves one MDP with a modified reward that includes a task-specific loss function. The subgradient of the structured hinge loss is the feature difference between the loss-augmented policy and the expert. This makes MMP as scalable as the underlying MDP solver.

## How it works

The estimator minimizes the regularized structured hinge loss:

$$
\min_\theta \frac{\lambda}{2} \|\theta\|^2 + \frac{1}{N} \sum_i \max_\pi \left[ \Delta(\pi^*, \pi) + \theta^\top (\phi(\pi) - \phi(\pi^*)) \right].
$$

The inner maximization is loss-augmented inference, which solves the MDP with reward $R_{\text{aug}}(s,a) = \theta^\top \phi(s,a) + \Delta(s,a)$. The subgradient at each iteration is $\lambda \theta + (\bar\mu_{\text{aug}} - \bar\mu_{\text{expert}})$, where $\bar\mu_{\text{aug}}$ is the feature expectation of the loss-augmented policy. Standard errors require bootstrap because there is no probabilistic model.

## When to use it

MMP is the scalable version of margin-based IRL. Its per-iteration cost is constant and matches MCE-IRL. The loss-augmented inference step allows incorporating task-specific loss functions that penalize specific types of policy errors. The limitation relative to MCE-IRL is the lack of a probabilistic model and the sensitivity to the choice of loss function and regularization strength. For formal inference, use likelihood-based methods.

## References

- Ratliff, N., Bagnell, J. A., and Zinkevich, M. (2006). Maximum Margin Planning. *ICML 2006*.

A primer for this estimator is not yet available.
