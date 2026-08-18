# IRL Identification Boundaries

Proofs and proof sketches on this page draw from {ref}`Kang (2026)
<kang-2026-lecture>`, {ref}`Rawat and Rust (2026) <rawat-rust-2026>`, and the
classical sources cited inline.

Read this page as a boundary map. Many IRL objectives fit behavior well. Fewer
identify the reward object needed for structural interpretation or
counterfactual policy analysis.

## MCE-IRL

Maximum causal entropy IRL chooses a policy with high causal entropy subject to
matching expert feature counts. With reward $r_\theta(s,a)=\phi(s,a)^\top\theta$,
the dual likelihood gradient has the form

$$
\nabla_\theta \ell(\theta)
=
\mathbb{E}_{\mathrm{expert}}
\left[\sum_t \beta^t\phi(s_t,a_t)\right]
-
\mathbb{E}_{\pi_\theta}
\left[\sum_t \beta^t\phi(s_t,a_t)\right].
$$

At the optimum, model feature counts match expert feature counts. This is a
reward-recovery statement only inside the supplied feature span and under the
normalization used by the soft Bellman equation.

**Estimator consequence.** MCE-IRL is useful when the reward basis is credible
and transitions are known or supplied. Neural MCE-IRL relaxes the reward basis
but does not make raw network weights the estimand.

## AIRL: Advantage First, Reward Later

AIRL uses a discriminator whose logit has potential-shaped form:

$$
f_{g,h}(s,a,s')
=
g(s)+\beta h(s')-h(s).
$$

The discriminator probability is

$$
D(s,a,s')
=
\frac{\exp f_{g,h}(s,a,s')}
{\exp f_{g,h}(s,a,s')+\pi(a\mid s)}.
$$

At an exact population saddle where the generated policy matches the expert and
the discriminator class is rich enough, $D=1/2$ on the matched support. The
discriminator equation then implies

$$
f_{g,h}(s,a,s')=\log\pi^*(a\mid s)=Q^*(s,a)-V^*(s).
$$

That object is the soft advantage. It is not yet the reward. The remaining
question is whether the decomposition of the advantage into $g$ and
$\beta h(s')-h(s)$ is unique.

## AIRL Disentanglement Conditions

AIRL's clean reward result needs load-bearing restrictions. In the state-only
case with deterministic transitions $s'=T(s,a)$,

$$
Q^*(s,a)-V^*(s)
=
r(s)+\beta V^*(T(s,a))-V^*(s).
$$

If the AIRL saddle also gives

$$
g(s)+\beta h(T(s,a))-h(s)
=
r(s)+\beta V^*(T(s,a))-V^*(s),
$$

then set $\Delta(s)=h(s)-V^*(s)$ and $\delta(s)=g(s)-r(s)$. Rearranging gives

$$
\delta(s)-\Delta(s)+\beta\Delta(T(s,a))=0.
$$

Under the decomposability condition on the transition graph, the only way this
can hold for all supported transitions is for $\Delta$ to be constant. Then
$h=V^*+c$ and $g=r+(1-\beta)c$. AIRL recovers the state-only reward up to a
constant.

The restrictions are the point:

- If $g$ can depend on actions, potential-shaped reward ambiguity returns.
- If transitions are stochastic, the realized $h(s')$ term does not equal the
  conditional expectation in the Bellman equation.
- If the transition graph is not connected enough, different components can
  carry different constants.
- If the adversarial game does not reach the population saddle, the identity
  above does not apply.

**Estimator consequence.** The AIRL page should be read with the state-only
condition in mind. The AIRL2 page uses extra economic anchors instead of
relying only on AIRL's original state-only decomposition.

The stochastic-transition problem has two parts. First, the target advantage
$Q^*(s,a)-V^*(s)$ does not depend on the realized next state $s'$, while the
AIRL score $g(s)+\beta h(s')-h(s)$ generally does. Second, even if one replaces
$h(s')$ by a conditional expectation, reward recovery still requires a
completeness or normalization condition that rules out nonconstant shaping
potentials.

## GAIL and Occupancy Matching

GAIL minimizes a divergence between the generated and expert occupancy measures.
At the discriminator optimum,

$$
\log\frac{D^*(s,a)}{1-D^*(s,a)}
=
\log\frac{d^{\pi^*}(s,a)}{d^\pi(s,a)}.
$$

This is a density-ratio object, not a Bellman reward. Matching occupancies can
imitate behavior, but it does not impose the pointwise equation

$$
Q(s,a)=r(s,a)+\beta\mathbb{E}[V(s')\mid s,a].
$$

The limitation can be seen from the Bellman-flow identity. For normalized
discounted occupancy $d^\pi$,

$$
\mathbb{E}_{d^\pi}\!\left[
Q(s,a)-\beta\mathbb{E}[Q(s',a')\mid s,a]
\right]
=(1-\beta)\mathbb{E}[Q(s_0,a_0)].
$$

This is a weighted average over the states and actions visited by $\pi$. A
weighted average Bellman residual can be zero while the pointwise residual is
positive in one region and negative in another. Occupancy matching therefore
controls where the policy goes; it does not by itself enforce the Bellman
equation of a primitive reward.

**Estimator consequence.** Occupancy matching is an imitation route. It needs
additional structure before its discriminator can be read as a primitive reward.

## IQ-Learn

IQ-Learn parameterizes $Q$ and defines a reward by Bellman rearrangement:

$$
r_Q(s,a)=Q(s,a)-\beta\mathbb{E}[V_Q(s')\mid s,a].
$$

This guarantees Bellman consistency for the reward induced by the chosen $Q$.
But the softmax part of the objective is invariant to state-only shifts:

$$
Q_c(s,a)=Q(s,a)+c(s).
$$

The induced reward changes by

$$
r_{Q_c}(s,a)
=
r_Q(s,a)+c(s)-\beta\mathbb{E}[c(s')\mid s,a],
$$

which is the same potential-shaped ambiguity from the identification page. A
regularizer can choose one representative from this class, but that choice is a
penalty preference unless an identifying normalization is added.

The key cancellation is in the objective. The return-gap part can be written as

$$
\mathbb{E}_{d^{\pi^*}}[r_Q(s,a)]
-(1-\beta)\mathbb{E}[V_Q(s_0)].
$$

Under $Q_c(s,a)=Q(s,a)+c(s)$, the reward term changes by
$(1-\beta)\mathbb{E}[c(s_0)]$ and the initial-value term changes by the same
amount with the opposite sign. The policy-fit part is therefore invariant to
state-only shifts; only the regularizer chooses a representative.

**Estimator consequence.** IQ-Learn is useful as an inverse soft-Q diagnostic,
but the public docs should not describe its regularized reward as a uniquely
identified primitive utility.

## Boundary Table

| Method | What the objective directly controls | Reward interpretation |
| --- | --- | --- |
| MCE-IRL | Feature-count matching / soft likelihood. | Reward in the supplied feature span. |
| Neural MCE-IRL | Soft likelihood with a neural reward map. | Reward map, not raw weights; needs support and normalization. |
| AIRL | Advantage decomposition at an adversarial saddle. | State-only reward under original AIRL restrictions. |
| AIRL2 | Anchored action-dependent rewards with latent segments. | Segment-specific reward under anchor and segment support conditions. |
| GAIL | Occupancy matching. | Imitation unless extra Bellman identification is added. |
| IQ-Learn | Soft-Q fit plus regularized Bellman-implied reward. | Representative selected by regularization unless anchored. |
