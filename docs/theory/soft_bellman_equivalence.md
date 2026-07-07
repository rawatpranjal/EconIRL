# Soft Bellman and DDC-MaxEnt Equivalence

Proofs and proof sketches on this page draw from {ref}`Kang (2026)
<kang-2026-lecture>`, {ref}`Rawat and Rust (2026) <rawat-rust-2026>`, and the
classical sources cited inline.

Read this page first if the same softmax equation seems to appear under two
different names. In dynamic discrete choice, the randomness comes from
unobserved Type-I extreme-value shocks. In maximum-entropy IRL, it comes from an
entropy term in the agent's objective. Under the unit-scale normalization, the
observable policy equation is the same.

## Setup

This page uses the unit convention: the Type-I extreme-value shock scale is one
and the maximum-entropy temperature is one. With a general scale $\tau>0$, the
same formulas become

$$
V_Q^\tau(s)=\tau\log\sum_a\exp(Q(s,a)/\tau),
\qquad
\pi_Q^\tau(a\mid s)=
\frac{\exp(Q(s,a)/\tau)}{\sum_b\exp(Q(s,b)/\tau)}.
$$

The DDC and maximum-entropy formulations line up literally only when they use
the same scale. In applications, the logit scale is a normalization: changing it
rescales utility units unless some outside information fixes the scale.

Let $Q$ be a bounded choice-specific value function and define

$$
V_Q(s) = \log \sum_{a \in A} \exp Q(s,a).
$$

For reward $r$, transition kernel $P$, and discount factor $\beta$, the unit
soft Bellman operator is

$$
(TQ)(s,a)
= r(s,a)
+ \beta \mathbb{E}\left[V_Q(s') \mid s,a\right].
$$

The fixed point $Q^* = TQ^*$ implies the policy

$$
\pi^*(a \mid s)
= \frac{\exp Q^*(s,a)}
       {\sum_b \exp Q^*(s,b)}.
$$

The next sections show why this is the right object for both DDC and
maximum-entropy IRL.

## Soft Bellman Contraction

The contraction proof matters because NFXP, MPEC, and many IRL algorithms rely
on the same fixed-point existence argument.

**Claim.** For bounded $Q_1,Q_2$,

$$
\|TQ_1 - TQ_2\|_\infty
\le \beta \|Q_1 - Q_2\|_\infty .
$$

**Proof.** The reward cancels in the difference. The only nonlinear term is
log-sum-exp. For any two action-value vectors $x,y$,

$$
\left|\log \sum_a e^{x_a} - \log \sum_a e^{y_a}\right|
\le \max_a |x_a-y_a|.
$$

This follows by setting $\Delta=\max_a |x_a-y_a|$, so
$e^{-\Delta}\sum_a e^{y_a}\le \sum_a e^{x_a}\le
e^\Delta\sum_a e^{y_a}$, then taking logs. Therefore

$$
\begin{aligned}
|(TQ_1)(s,a)-(TQ_2)(s,a)|
&\le \beta
\mathbb{E}\left[
|V_{Q_1}(s')-V_{Q_2}(s')|
\mid s,a
\right] \\
&\le \beta \|Q_1-Q_2\|_\infty .
\end{aligned}
$$

Taking the supremum over $(s,a)$ gives the result. Since $\beta<1$, Banach's
fixed-point theorem gives a unique bounded fixed point.

**Estimator consequence.** NFXP can solve the inner dynamic program for each
candidate reward, MPEC can impose the same fixed point as a constraint, and
GLADIUS can use the Bellman equation as an identifying restriction.

## Entropy-Regularized Choice

Maximum-entropy IRL solves, at each state, a finite choice problem:

$$
\max_{q \in \Delta(A)}
\left\{\sum_a q_a Q(s,a) - \sum_a q_a \log q_a\right\}.
$$

The first-order condition is

$$
Q(s,a)-\log q_a-1-\eta=0,
$$

so $q_a \propto \exp Q(s,a)$. Normalizing gives

$$
q_a =
\frac{\exp Q(s,a)}{\sum_b \exp Q(s,b)},
$$

and substituting the optimizer gives the value

$$
\log \sum_a \exp Q(s,a).
$$

Thus entropy-regularized control produces exactly the soft value $V_Q$ and the
softmax policy above.

This is the causal-entropy version. The entropy term is policy randomness at
each decision time, not randomness in the next state. Trajectory entropy and
causal entropy coincide when transitions are deterministic. With stochastic
transitions, trajectory entropy also rewards environmental randomness, so it is
not the same object as the logit DDC shock.

## Type-I Extreme-Value Choice

DDC starts with a different story. The agent observes

$$
Q(s,a)+\epsilon_a
$$

for each action, where the shocks are independent unit-scale Type-I
extreme-value shocks. The Gumbel-max identity gives

$$
\Pr\left(a=\arg\max_b\{Q(s,b)+\epsilon_b\}\right)
=
\frac{\exp Q(s,a)}{\sum_b \exp Q(s,b)}.
$$

Under the mean-zero normalization for the shocks, the expected maximum is the
same log-sum-exp value used in the entropy problem:

$$
\mathbb{E}_\epsilon\left[
\max_a \{Q(s,a)+\epsilon_a\}
\right]
=
\log \sum_a \exp Q(s,a).
$$

So the integrated DDC Bellman equation is also

$$
Q^*(s,a)
= r(s,a)
+ \beta \mathbb{E}\left[
\log \sum_b \exp Q^*(s',b)
\mid s,a
\right].
$$

If the shock mean is not normalized to zero, the Bellman equation is shifted by
a common constant. If the shock mean is $\mu_\epsilon$, the fixed point shifts
by $\beta\mu_\epsilon/(1-\beta)$ relative to the mean-zero convention. The
softmax policy is unchanged because adding the same constant to all actions at a
state does not affect action probabilities. The clean equality with
unit-temperature maximum-entropy IRL uses the mean-zero normalization.

## Equivalence Theorem

**Theorem.** Fix $(S,A,P,\beta,r)$. Mean-zero unit-scale DDC and
unit-temperature maximum-entropy IRL have the same bounded fixed point $Q^*$,
the same policy $\pi^*$, and the same distribution over observed
$(s,a,s')$ trajectories.

**Proof.** The entropy derivation and the Gumbel derivation both reduce to the
same soft Bellman fixed point. The contraction result above gives uniqueness of
that fixed point. Applying the same softmax map to the same $Q^*$ gives the same
policy. The observed trajectory law depends only on the initial distribution,
the transition kernel, and the policy, so the trajectory law is the same.

**Estimator consequence.** The structural DDC pages and the IRL pages can share
one notation for $Q$, $V_Q$, and $\pi_Q$. The difference between estimators is
not the forward policy equation; it is how each estimator identifies or
estimates the reward behind that equation.
