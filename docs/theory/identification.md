# Identification and Anchors

Proofs and proof sketches on this page draw from {ref}`Kang (2026)
<kang-2026-lecture>`, {ref}`Rawat and Rust (2026) <rawat-rust-2026>`, and the
classical sources cited inline.

Read this page before interpreting a recovered reward. Choice data identify
relative action values first. Reward levels and some reward shapes require
normalizations or restrictions.

The arguments below take the transition kernel $P$ and discount factor $\beta$
as known inputs. In empirical work, $P$ may be estimated in a first stage and
then treated as known for the reward-recovery step. That is still a substantive
rational-expectations assumption: the transition law used in the estimator must
match the transition law the agent expects.

## What Behavior Identifies First

The softmax policy satisfies

$$
\log \pi(a \mid s)-\log \pi(b \mid s)
= Q(s,a)-Q(s,b).
$$

So if the expert policy is known, the within-state differences in $Q$ are
known. The state-wise level $c(s)$ is not: replacing $Q(s,a)$ by
$Q(s,a)+c(s)$ leaves the softmax policy unchanged.

This is the first place readers get lost. Policy fit and reward recovery are
not the same claim. A method can match choices and still leave the reward
unidentified.

## Potential-Based Non-Identification

Let $r$ produce the soft Bellman fixed point $Q^*$ and policy $\pi^*$. For any
bounded function $\Phi:S\to\mathbb{R}$, define

$$
\tilde r(s,a)
= r(s,a)+\Phi(s)
- \beta \mathbb{E}\left[\Phi(s') \mid s,a\right],
$$

and

$$
\tilde Q(s,a)=Q^*(s,a)+\Phi(s).
$$

**Claim.** $\tilde Q$ is the soft Bellman fixed point for $\tilde r$, and it
induces the same policy as $Q^*$.

**Proof.** Apply the soft Bellman operator for $\tilde r$:

$$
\begin{aligned}
\tilde r(s,a)
&+\beta \mathbb{E}\left[
\log \sum_b \exp \tilde Q(s',b)
\mid s,a
\right] \\
&= r(s,a)+\Phi(s)-\beta\mathbb{E}[\Phi(s')\mid s,a]
+\beta\mathbb{E}\left[
\log \sum_b \exp(Q^*(s',b)+\Phi(s'))
\mid s,a
\right] \\
&= r(s,a)+\Phi(s)-\beta\mathbb{E}[\Phi(s')\mid s,a]
+\beta\mathbb{E}\left[
V_{Q^*}(s')+\Phi(s')
\mid s,a
\right] \\
&= Q^*(s,a)+\Phi(s)
= \tilde Q(s,a).
\end{aligned}
$$

The policy is unchanged because the same $\Phi(s)$ is added to every action at
state $s$:

$$
\frac{\exp(Q^*(s,a)+\Phi(s))}
{\sum_b \exp(Q^*(s,b)+\Phi(s))}
=
\frac{\exp Q^*(s,a)}
{\sum_b \exp Q^*(s,b)}.
$$

**Estimator consequence.** Without more structure, no estimator using only
behavior can distinguish $r$ from $\tilde r$. This is why the estimator pages
talk about reward normalizations, anchors, state-only restrictions, and
counterfactual caveats.

The caveat becomes operational under counterfactuals. The shaped reward above is
observationally equivalent under the original transition law $P$. If a
counterfactual changes the law to $\lambda(s'\mid s,a)$, the shaping correction
becomes

$$
\Phi(s)-\beta\mathbb{E}_{\lambda}[\Phi(s')\mid s,a],
$$

which need not equal the correction under $P$. Two rewards that fit the same
observed policy can therefore imply different policies after the transition law
changes.

## Anchor-Action Identification

The DDC route fixes the missing state-wise level by declaring one action's
reward known in each state. Let $a_0(s)$ be the anchor action and assume

$$
r(s,a_0(s)) = r_0(s)
$$

is known. The common normalization is $r_0(s)=0$, but the argument works with
any known anchor payoff.

**Theorem.** If $(\pi^*,P,\beta)$ are known, the expert policy has support on
the anchor action, and the anchor reward is known, then $Q^*$ and $r$ are
identified on the covered support.

**Proof.** There are three steps.

First, Hotz-Miller inversion gives action-value differences:

$$
Q^*(s,a)-Q^*(s,a_0)
=
\log \pi^*(a\mid s)-\log \pi^*(a_0\mid s).
$$

Second, write $V^*(s)=\log\sum_b\exp Q^*(s,b)$. The softmax identity at the
anchor gives

$$
Q^*(s,a_0)=V^*(s)+\log\pi^*(a_0\mid s).
$$

The Bellman equation at the anchor gives

$$
Q^*(s,a_0)
= r_0(s)+\beta\mathbb{E}[V^*(s')\mid s,a_0].
$$

Combining the last two displays yields a linear fixed-point equation in $V^*$
alone:

$$
V^*(s)
= r_0(s)
+\beta\mathbb{E}[V^*(s')\mid s,a_0]
-\log\pi^*(a_0\mid s).
$$

This operator is a $\beta$-contraction, so $V^*$ is uniquely determined.

Third, recover the anchor value from
$Q^*(s,a_0)=V^*(s)+\log\pi^*(a_0\mid s)$, recover all other $Q^*(s,a)$ from the
log-probability differences, and recover reward by rearranging the Bellman
equation:

$$
r(s,a)
= Q^*(s,a)
-\beta\mathbb{E}[V^*(s')\mid s,a].
$$

That pins down the reward on the support where the policy and transition
objects are known.

The anchor is an identifying assumption, not something the choices prove. If the
researcher supplies the wrong anchor payoff, the recovered reward will still be
internally consistent with the observed policy and the supplied anchor, but it
will not equal the primitive reward. Counterfactual predictions can then differ
from the behavior that the true reward would imply.

## What Each Core Estimator Uses

| Estimator | Identification route | What to check before interpreting reward |
| --- | --- | --- |
| NFXP | Parametric reward plus full Bellman solve. | Transition model, logit scale, reward normalization, feature rank. |
| CCP | Hotz-Miller inversion plus transition expectations. | Choice-probability support and transition estimation. |
| TD-CCP | Same finite reward target, recursive terms learned from successor tuples. | Successor support, basis span, and standard-error path. |
| MCE-IRL | Feature-matching / likelihood in a known controlled process. | Reward in supplied feature span and fixed normalization. |
| AIRL | State-only decomposition under the original AIRL conditions. | State-only reward, deterministic or decomposable dynamics, exact-saddle caveat. |
| AIRL-Het | Action-dependent reward with economic anchors. | Exit action, absorbing state, segment support. |
| GLADIUS | Softmax likelihood plus anchor Bellman risk over $Q$. | Anchor action, support, continuation-value regression, realizability. |

Use this table as a sanity check. If the right-hand column is not credible in
your application, the estimated object may still predict choices, but it should
not be read as a point-identified primitive reward.
