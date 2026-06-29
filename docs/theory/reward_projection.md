# Reward Projection and Feature Rank

Proofs and proof sketches on this page draw from {ref}`Kang (2026)
<kang-2026-lecture>`, {ref}`Rawat and Rust (2026) <rawat-rust-2026>`, and the
classical sources cited inline.

Read this page when a previous page says that a reward is recovered. A recovered
reward function is not automatically a recovered parameter vector. EconIRL often
reports a finite parameter $\theta$, so one more step is needed: the recovered
reward contrasts must live in the span of the supplied features.

## Setup

Let $a_0(s)$ be the reference or anchor action in state $s$. Suppose the reward
model is linear:

$$
r_\theta(s,a)=\phi(s,a)^\top\theta.
$$

The policy identifies action contrasts first, so define

$$
\Delta r(s,a)=r(s,a)-r(s,a_0(s)),
\qquad
\Delta\phi(s,a)=\phi(s,a)-\phi(s,a_0(s)).
$$

Stack the covered non-anchor state-action pairs into a vector $y$ and matrix
$X$:

$$
y_{s,a}=\Delta r(s,a),
\qquad
X_{s,a,\cdot}=\Delta\phi(s,a)^\top .
$$

The parameter recovery equation is then

$$
y=X\theta.
$$

This is the last algebraic bridge from behavior to finite parameters.

## Action-Difference Rank Theorem

**Theorem.** If the recovered reward contrasts satisfy $y=X\theta_0$ and
$X$ has full column rank, then $\theta_0$ is uniquely identified from the
recovered reward contrasts. If $X$ is rank deficient, $\theta_0$ is not uniquely
identified.

**Proof.** If $X$ has full column rank, then $X^\top X$ is invertible. The
normal equations have the unique solution

$$
\theta_0=(X^\top X)^{-1}X^\top y.
$$

If $X$ is rank deficient, there is a nonzero vector $v$ with $Xv=0$. Then

$$
X(\theta_0+v)=X\theta_0=y.
$$

The two parameter vectors produce exactly the same recovered reward contrasts
on the covered support. No estimator using only those contrasts can distinguish
them.

**Estimator consequence.** The feature-rank condition in NFXP, CCP, TD-CCP,
MCE-IRL, AIRL-Het, and GLADIUS is not a numerical preference. It is the
identification condition that turns recovered rewards or reward contrasts into
finite parameters.

## Why State-Only Features Fail for Action Rewards

Suppose the feature map is copied across actions:

$$
\phi(s,a)=\psi(s)
\quad\text{for every }a.
$$

Then

$$
\Delta\phi(s,a)=\psi(s)-\psi(s)=0,
$$

so $X=0$. The action-difference equation becomes

$$
\Delta r(s,a)=0^\top\theta,
$$

which carries no information about $\theta$ unless every action contrast is
actually zero.

**Proof.** This is the rank theorem with rank$(X)=0$. If $\theta$ is feasible,
then every $\theta+v$ is also feasible because $Xv=0$ for all $v$.

**Estimator consequence.** State-only features can be useful when the model is
explicitly a state-reward model, as in the original AIRL guarantee. They cannot
identify action-dependent payoff differences unless the model supplies another
source of action variation.

## Projection When the Feature Span Is Wrong

If the recovered reward contrast $y$ is not exactly in the column span of $X$,
there is no structural parameter $\theta$ that reproduces the reward perfectly.
The estimand becomes a weighted projection:

$$
\theta_W
=\arg\min_\theta (y-X\theta)^\top W (y-X\theta),
$$

where $W$ is a positive semidefinite weighting matrix over covered
state-action contrasts. If $X^\top W X$ is invertible, then

$$
\theta_W=(X^\top W X)^{-1}X^\top W y.
$$

**Proof.** Differentiate the quadratic objective:

$$
\frac{\partial}{\partial\theta}
(y-X\theta)^\top W(y-X\theta)
=-2X^\top W(y-X\theta).
$$

Setting the derivative to zero gives $X^\top W X\theta=X^\top W y$. Invert
$X^\top W X$ to obtain the displayed formula.

**Estimator consequence.** Under misspecification, the reported parameter is a
best linear projection under the chosen weighting, not the literal primitive
reward. This is why reward plots, action contrasts, and counterfactual checks
matter even when a finite parameter is reported.

## Neural Rewards Are Function Objects First

For neural reward estimators, many parameter vectors can represent the same
reward map. Hidden-unit permutations, redundant units, and flat directions in
the network can change the raw weights while leaving $r_\eta(s,a)$ unchanged on
the covered state-action pairs.

Formally, if two neural parameter vectors $\eta_1$ and $\eta_2$ satisfy

$$
r_{\eta_1}(s,a)=r_{\eta_2}(s,a)
\quad\text{for all covered }(s,a),
$$

then every policy, value, reward-contrast, and Bellman-rearranged quantity on
that support is the same. The raw parameter vector is therefore not the
identified object. The identified object is the anchored reward map or its
finite projection.

**Estimator consequence.** Neural MCE-IRL and GLADIUS should be interpreted at
the level of anchored reward maps, action contrasts, induced policies, and
projected parameters. Raw neural weights are implementation details.

## What To Check

| Question | Formal check | Why it matters |
| --- | --- | --- |
| Are action rewards identified? | $X$ has full column rank after anchoring. | Otherwise multiple $\theta$ values produce the same reward contrasts. |
| Are state-only features being used for action rewards? | $\Delta\phi(s,a)$ is not identically zero. | State-only features cannot explain action payoff differences. |
| Is the linear model exact? | $y$ lies in the column span of $X$. | If not, $\theta$ is a projection. |
| Is a neural reward being reported? | Compare reward maps, not raw weights. | Network parameters are many-to-one. |

The broader logic is simple: behavior identifies policy and action-value
contrasts; anchors and Bellman equations identify reward contrasts; feature rank
turns those contrasts into finite parameters.
