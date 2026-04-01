# Theory

### The Problem

Imagine someone making decisions over time. Each period, they see a situation (a "state" $s$), pick an action $a$, get a payoff $r(s,a)$, and move to a new situation $s'$. They care about the future, but not as much as the present, so they discount future payoffs by a factor $\beta$ between 0 and 1. Their goal is to maximize total discounted payoffs $E\{\sum_{t=0}^\infty \beta^t r(s_t, a_t)\}$.

This setup is called a Markov decision process (MDP). The solution is an optimal policy $\delta^*(s) = \arg\max_a Q^*(s,a)$, where $Q^*(s,a)$ is the value of taking action $a$ in state $s$ and then behaving optimally forever after. It satisfies

$$
Q^*(s,a) = r(s,a) + \beta \sum_{s'} p(s' \mid s,a) \, V^*(s'),
$$

where $V^*(s) = \max_a Q^*(s,a)$. This is the Bellman equation. It says the value of any action equals its immediate payoff plus the discounted value of wherever you end up.

The estimation problem runs this logic backward. We watch people make choices but we do not know their preferences. We want to figure out the payoff function $r$ (or its parameters $\theta$) that explains what we see. Economists call this structural estimation of dynamic discrete choice (DDC) models. Machine learning researchers call it inverse reinforcement learning (IRL). The math is the same.

### Why People in the Same Situation Make Different Choices

The optimal policy $\delta^*(s)$ is deterministic. Everyone in the same state should do the same thing. But in real data, people in the same situation often choose differently. A deterministic model cannot explain this.

Rust (1987) fixed this by adding unobserved "taste shocks" $\varepsilon = (\varepsilon_1, \ldots, \varepsilon_{|\mathcal{A}|})$, one for each action. The agent sees these shocks, but we as researchers do not. The payoff for action $a$ becomes $r(s,a) + \varepsilon_a$. From the agent's perspective, the choice is still deterministic given $(s, \varepsilon)$. But from our perspective, it looks random because we cannot see $\varepsilon$.

Following McFadden (1973), assume the shocks follow an extreme value type 1 (Gumbel) distribution with scale $\sigma > 0$ and mean zero. Also assume the shocks are independent across time periods. These two assumptions buy us closed-form expressions for everything.

### The Soft Bellman Equations

The Gumbel distribution has a remarkable property. When you take the expected maximum over Gumbel-distributed options, you get a simple formula. Averaging $V(s,\varepsilon) = \max_a [Q(s,a) + \varepsilon_a]$ over the shocks gives

$$
V(s) = \sigma \log \sum_{a \in \mathcal{A}(s)} \exp\!\big(Q(s,a)/\sigma\big).
$$

This is called the log-sum-exp function. It is a smooth approximation to the max. When $\sigma$ is small, it behaves like a hard maximum. When $\sigma$ is large, it spreads probability more evenly across actions.

Plugging this into the Bellman equation gives the soft Bellman operator $\Lambda_\sigma$,

$$
\Lambda_\sigma(Q)(s,a) = r(s,a) + \beta \sum_{s'} p(s' \mid s,a) \; \sigma \log \sum_{a'} \exp\!\big(Q(s',a')/\sigma\big).
$$

This operator is a contraction (it shrinks distances by a factor of $\beta$), so it has a unique fixed point $Q^* = \Lambda_\sigma(Q^*)$.

The probability that someone picks action $a$ in state $s$ is the familiar logit formula,

$$
\pi(a \mid s) = \frac{\exp\!\big(Q(s,a)/\sigma\big)}{\sum_{b} \exp\!\big(Q(s,b)/\sigma\big)}.
$$

Actions with higher $Q$-values get chosen more often. The parameter $\sigma$ controls how sensitive choices are to value differences. Small $\sigma$ means nearly deterministic choices. Large $\sigma$ means nearly random choices.

With a non-uniform base measure $\mu(a \mid s) > 0$, the formula becomes $\pi(a \mid s) = \mu(a \mid s) \exp(Q(s,a)/\sigma) / \sum_b \mu(b \mid s) \exp(Q(s,b)/\sigma)$. The uniform case $\mu = 1/|\mathcal{A}|$ is the standard model.

### The Bridge to Reinforcement Learning

The same equations come from a totally different angle. Suppose an agent picks a policy to maximize expected payoff minus a penalty for straying too far from some default behavior $\mu$,

$$
V(s) = \max_{\pi(\cdot \mid s)} \bigg\{ \sum_a \pi(a \mid s) \, Q(s,a) - \sigma \, \mathrm{KL}\!\big(\pi(\cdot \mid s) \| \mu(\cdot \mid s)\big) \bigg\}.
$$

The solution to this optimization is the same log-sum-exp value and the same logit policy. So the DDC model with taste shocks and the entropy-regularized RL model with a KL penalty are the same thing. The parameter $\sigma$ plays two roles at once. In economics, it is the spread of unobserved taste shocks. In machine learning, it is the strength of the entropy bonus that keeps the policy from becoming too extreme. Rust and Rawat (2026, Appendix A) prove this equivalence formally.

### Notation Summary

| Symbol | Definition |
|--------|-----------|
| $s \in \mathcal{S}$, $a \in \mathcal{A}(s)$ | State, action |
| $\beta \in (0,1)$ | Discount factor |
| $\sigma > 0$ | Scale of taste shocks (or strength of entropy bonus) |
| $\mu(a \mid s) > 0$ | Default (base) policy |
| $\vec{r}(s,a) = (r_1(s,a), \ldots, r_K(s,a))$ | Feature vector |
| $r_\theta(s,a) = \vec{r}(s,a) \cdot \theta$ | Payoff function with parameters $\theta \in \mathbb{R}^K$ |
| $p(s' \mid s,a)$ | Transition probability |
| $Q(s,a)$, $V(s)$ | Action value, state value |
| $\pi(a \mid s)$ | Choice probability (CCP) or policy |
| $\mathcal{D} = \{\tau_i\}_{i=1}^N$ | Observed trajectories, $\tau_i = \{(s_{it}, a_{it})\}_{t=0}^{T_i}$ |
| $R(s,a) \in \mathbb{R}^K$ | Continuation feature vector (discounted future features) |
| $Q_r(s,a)$, $Q_\varepsilon(s,a)$ | Payoff part and taste-shock part of $Q$ |

### Splitting Q into Two Pieces

When the payoff is a weighted sum of features, $r_\theta(s,a) = \vec{r}(s,a) \cdot \theta$, the action value splits neatly into two parts: $Q(s,a) = Q_r(s,a) + Q_\varepsilon(s,a)$.

The first part depends on $\theta$. It equals $Q_r(s,a) = R(s,a)^\top \theta$, where $R$ is the continuation feature vector. Think of $R_k(s,a)$ as "how much of feature $k$ will the agent accumulate, in discounted terms, starting from $(s,a)$." It solves

$$
R(s,a) = \vec{r}(s,a) + \beta \sum_{s'} p(s' \mid s,a) \sum_{a'} \pi(a' \mid s') \, R(s',a').
$$

The second part, $Q_\varepsilon$, captures the option value of having choices in the future. It depends only on the current policy, not on $\theta$,

$$
Q_\varepsilon(s,a) = \beta \sum_{s'} p(s' \mid s,a) \sum_{a'} \pi(a' \mid s') \big[-\sigma \log \pi(a' \mid s') + Q_\varepsilon(s',a')\big].
$$

This split is the foundation of the CCP and TD-CCP estimators. It lets us separate what depends on the unknown parameters from what can be estimated directly from data.

### Likelihood Functions

We observe trajectories $\mathcal{D} = \{\tau_i\}_{i=1}^N$. The full likelihood includes both choices and transitions,

$$
\mathcal{L}^f_\mathcal{D}(\theta) = \prod_{i=1}^N \prod_{t=1}^{T_i} \pi_\theta(a_{it} \mid s_{it}) \, p_\theta(s_{it} \mid s_{it-1}, a_{it-1}).
$$

The partial likelihood uses choices only,

$$
\mathcal{L}^p_\mathcal{D}(\theta) = \prod_{i=1}^N \prod_{t=1}^{T_i} \pi_\theta(a_{it} \mid s_{it}).
$$

Most estimators maximize the partial log-likelihood $\ell^p(\theta) = \log \mathcal{L}^p_\mathcal{D}(\theta)$. The transition part drops out when transitions do not depend on $\theta$.

### Identification

Looking at data alone, we can only tell apart actions whose $Q$-values differ. Specifically,

$$
\log\!\big(\pi(a \mid s) / \pi(a' \mid s)\big) = \big(Q(s,a) - Q(s,a')\big) / \sigma.
$$

We can read off differences in $Q$ but not its level. And since $Q$ builds on $r$, the level of $r$ is not identified either.

In fact, many different reward functions produce the exact same behavior. Ng, Harada, and Russell (1999) showed that for any function $h(s)$,

$$
r_h(s,a) = r(s,a) + \beta \sum_{s'} p(s' \mid s,a) \, h(s') - h(s)
$$

gives the same policy as $r$. You can shift rewards by any "potential" $h$ and nothing changes. But if the environment changes, these equivalent rewards start to disagree. This is why getting identification right matters for counterfactual analysis.

To pin down $\theta$, Rust and Rawat (2026) use two rules. First, the average reward across actions is zero in every state: $E_{a \sim \mu}[r_\theta(s,a)] = 0$. Second, the gap between two reference $Q$-values equals a known target: $Q^*_\theta(\bar{s}, a^+) - Q^*_\theta(\bar{s}, a^-) = \Delta^*$. Together these remove the ambiguity. An alternative approach, used in CCP and GLADIUS, is to assume the reward of one "anchor" action is known in each state.

### Equivalence Theorem

All estimators on this page share the same soft Bellman foundation. Rust and Rawat (2026, Theorem A.6) show that, with proper identification restrictions, it does not matter what kind of data you start from. Maximum likelihood on choice data (NFXP) and feature matching on demonstrations (MCE-IRL) both converge to the same policy as the sample grows,

$$
\pi^*_{\hat\theta_{\mathrm{NFXP}}} = \pi^*_{\hat\theta_{\mathrm{IRL}}} \;\longrightarrow\; \pi^*_{\theta^\star}.
$$

This is the central result connecting structural econometrics and inverse reinforcement learning.
