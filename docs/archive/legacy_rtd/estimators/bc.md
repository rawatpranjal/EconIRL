# BC

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Baseline | Ross et al. (2011) | None | No | No | Yes |

## What this estimator does

Behavioral cloning counts how often each action was chosen in each state. No model, no optimization, no value function. It is the mandatory baseline for every evaluation. If a sophisticated estimator cannot beat BC, it is not learning anything useful from the MDP structure. Ross et al. (2011) showed that BC errors grow as $O(T^2 \varepsilon)$ with horizon length $T$, while methods that recover the true reward achieve $O(\varepsilon)$ regardless of $T$. That gap is why structural estimation matters.

## How it works

The estimated policy is the empirical frequency:

$$
\hat\pi(a \mid s) = \frac{N(s,a)}{N(s)}.
$$

There is no optimization step. BC cannot recover parameters, rewards, or value functions.

## When to use it

BC should always be the first step. It validates whether the data has fundamental sequential structure that model-based methods can exploit. If structural estimators fail to beat BC on out-of-sample data, the environment lacks the forward-looking structure that justifies the additional complexity.

## References

- Ross, S., Gordon, G. J., and Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *AISTATS 2011*.

A primer for this estimator is not yet available.
