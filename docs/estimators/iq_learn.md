# IQ-Learn

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Model-Free | Garg et al. (2021) | Tabular or Neural | Yes | No | No |

## What this estimator does

IQ-Learn parameterizes the Q-function directly and recovers the reward as a byproduct through the inverse Bellman operator. The key finding is that the IQ-Learn chi-squared objective is algebraically equivalent to a penalized conditional log-likelihood. The first term is the standard multinomial logit likelihood that fits expert actions through a softmax policy. The second term penalizes the squared magnitude of the Bellman-implied reward at expert-visited states. No inner Bellman loop is needed because the smoothed value function is an explicit closed-form function of Q.

Unlike NFXP, IQ-Learn does not enforce the Bellman equation as a hard constraint. The chi-squared regularizer penalizes the implied reward magnitude, selecting from the equivalence class of observationally equivalent rewards by favoring solutions with small implied reward. This normalization has no economic interpretation, unlike the anchor constraints in AAIRL which set the exit payoff to zero. Unlike GLADIUS, IQ-Learn does not explicitly enforce Bellman consistency, so the recovered Q may not satisfy the Bellman equation and the implied reward may not be a valid structural primitive.

## How it works

The estimator maximizes the penalized log-likelihood

$$
\text{LL}(Q) - \frac{1}{4\alpha\sigma}\sum_{i=1}^{N} \hat{r}_i^2,
$$

where $\hat{r}_i = Q(s_i,a_i) - \beta \sum_{s'} p(s'|s_i,a_i) V(s')$ is the Bellman-implied reward and $V(s) = \sigma \log \sum_a \exp(Q(s,a)/\sigma)$. The Q-function is the single free parameter that determines both the policy via softmax and the implied reward via the inverse Bellman operator. L-BFGS-B optimizes over the Q-table in the tabular setting. Standard errors are not available because the regularization trades off fit against reward magnitude.

## When to use it

IQ-Learn is suited for quick exploratory estimation where the Bellman equation need not hold exactly and where the researcher wants a single-pass estimator without inner loops. On a 2-state MDP it achieves lower log-likelihood than NFXP because the regularization penalty trades fit for smaller implied rewards. For structural counterfactuals requiring Bellman-consistent rewards, NFXP or GLADIUS is preferred. For publication-grade estimates with inference, use NFXP or CCP.

## References

- Garg, D., Chakraborty, S., Cundy, C., Song, J., and Ermon, S. (2021). IQ-Learn: Inverse soft-Q Learning for Imitation. *NeurIPS 2021*.

The full derivation, algorithm, and simulation results are in the [IQ-Learn primer (PDF)](https://github.com/rawatpranjal/econirl/blob/main/papers/econirl_package/primers/iq_learn.pdf).
