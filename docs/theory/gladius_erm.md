# GLADIUS and ERM

Proofs and proof sketches on this page draw from {ref}`Kang (2026)
<kang-2026-lecture>`, {ref}`Rawat and Rust (2026) <rawat-rust-2026>`, and the
classical sources cited inline.

Read this page as the transition-estimation-free route. GLADIUS keeps the
anchor-action identification logic from DDC, but estimates the needed
continuation object directly from observed successor tuples.

## From Identification to a Risk

The identification page showed that two restrictions are enough to identify
$Q^*$ under an anchor action:

1. the softmax likelihood pins down within-state action-value differences;
2. the anchor Bellman equation pins down the state-wise level.

For a candidate $Q$, define

$$
\hat p_Q(a\mid s)=
\frac{\exp Q(s,a)}{\sum_b\exp Q(s,b)}
$$

and let $a_0(s)$ be the anchor action with known reward $r_0(s)$. The population
risk combines the two restrictions:

$$
\mathcal{R}(Q)
=
\mathbb{E}_{(s,a)\sim d^*}
\left[
-\log \hat p_Q(a\mid s)
+
\mathbf{1}\{a=a_0(s)\}
\left(
r_0(s)+\beta\mathbb{E}[V_Q(s')\mid s,a_0]-Q(s,a_0)
\right)^2
\right].
$$

**Identification proof sketch.** The negative log likelihood is minimized when
$\hat p_Q(\cdot\mid s)=\pi^*(\cdot\mid s)$ on covered states, so it fixes
within-state action differences. The anchor Bellman term is nonnegative and is
zero at $Q^*$. A minimizer of the sum must satisfy both restrictions. Those are
exactly the Hotz-Miller difference equation and the anchor fixed-point equation,
so $Q^*$ is recovered on support. Reward follows from

$$
r(s,a)=Q^*(s,a)-\beta\mathbb{E}[V_{Q^*}(s')\mid s,a].
$$

## Why Sampled Squared TD Is Biased

The population Bellman term uses a conditional expectation. Offline data give
one observed next state $s'$ at a time. The tempting sampled residual is

$$
\widehat\Delta(Q)
=
r(s,a)+\beta V_Q(s')-Q(s,a).
$$

It is unbiased before squaring:

$$
\mathbb{E}[\widehat\Delta(Q)\mid s,a]
=
r(s,a)+\beta\mathbb{E}[V_Q(s')\mid s,a]-Q(s,a).
$$

But the square has an extra variance term:

$$
\mathbb{E}[\widehat\Delta(Q)^2\mid s,a]
=
\left(
r(s,a)+\beta\mathbb{E}[V_Q(s')\mid s,a]-Q(s,a)
\right)^2
+
\beta^2\operatorname{Var}(V_Q(s')\mid s,a).
$$

The second term depends on $Q$. It is not a constant that can be ignored. This
is the double-sampling problem in this setting. If transitions are deterministic
conditional on $(s,a)$, the conditional variance is zero and this correction is
unnecessary. The correction matters for the stochastic-transition case.

## Bias Correction

The variance term can be written as a regression loss:

$$
\operatorname{Var}(V_Q(s')\mid s,a)
=
\min_z
\mathbb{E}\left[(V_Q(s')-z)^2\mid s,a\right].
$$

Let $\zeta(s,a)$ approximate the conditional mean

$$
\zeta_Q^*(s,a)=\mathbb{E}[V_Q(s')\mid s,a].
$$

Then the corrected Bellman-error identity is

$$
\begin{aligned}
&\left(
r(s,a)+\beta\mathbb{E}[V_Q(s')\mid s,a]-Q(s,a)
\right)^2 \\
&\quad =
\max_\zeta
\mathbb{E}\left[
\left(r(s,a)+\beta V_Q(s')-Q(s,a)\right)^2
-\beta^2\left(V_Q(s')-\zeta(s,a)\right)^2
\mid s,a
\right].
\end{aligned}
$$

At the anchor action, $r(s,a_0)$ is known, so this corrected term can be used
without estimating the transition density.

## GLADIUS Objective

With parametrized $Q_\theta$ and $\zeta_\phi$, GLADIUS alternates two updates:

$$
\phi \leftarrow
\arg\min_\phi
\sum_i
\left(V_{Q_\theta}(s_i')-\zeta_\phi(s_i,a_i)\right)^2
$$

and

$$
\theta \leftarrow
\arg\min_\theta
\sum_i
\left[
-\log \hat p_{Q_\theta}(a_i\mid s_i)
+
\mathbf{1}\{a_i=a_0(s_i)\}
\left\{
\widehat\Delta_i(Q_\theta)^2
-\beta^2
\left(V_{Q_\theta}(s_i')-\zeta_\phi(s_i,a_i)\right)^2
\right\}
\right],
$$

where

$$
\widehat\Delta_i(Q_\theta)
=
r_0(s_i)+\beta V_{Q_\theta}(s_i')-Q_\theta(s_i,a_0).
$$

The learned reward on supported state-action pairs is

$$
\widehat r(s,a)=\widehat Q(s,a)-\beta\widehat\zeta(s,a).
$$

The anchor-weighted corrected Bellman term identifies the state-wise level of
$Q$. The $\zeta$ regression has a second role: it is fit over observed
state-action pairs so that, at the solution,
$\zeta(s,a)=\mathbb{E}[V_Q(s')\mid s,a]$. That all-pair conditional-mean object
is what makes reward recovery transition-estimation-free for non-anchor actions
as well as the anchor action.

## Why the Anchor Still Matters

The likelihood part cannot see state-wise shifts in $Q$. The corrected Bellman
term fixes those shifts only at the anchor action because that is where the
reward is known. Without the anchor, GLADIUS can still fit policy behavior, but
the reward remains in the potential-shaped equivalence class from the
identification page.

This is why the GLADIUS estimator page emphasizes anchored action contrasts. The
anchor is not a cosmetic normalization; it is the equation that turns a fitted
policy object into a reward object.

## Convergence Logic

The convergence result is not a generic claim that nonconvex neural objectives
are easy. The proof uses complementary geometry:

- the likelihood term controls action-difference directions in $Q$;
- the anchor Bellman term controls the state-wise level left invisible by the
  softmax;
- the $\zeta$ update is a conditional-mean regression;
- empirical-output Jacobian conditions transfer these output-space controls to
  the network parameters.

Under the stated realizability, conditioning, stability, and stepsize
assumptions, the empirical optimization error decreases with iterations and the
population excess risk decreases with sample size. The page should be read as a
conditional guarantee. In particular, the result relies on the network classes
being able to represent the relevant $Q$ and $\zeta$ objects, empirical-output
Jacobian lower bounds transferring output-space control to parameters, and
PL-style descent/ascent geometry for the alternating updates. When support,
conditioning, or realizability is weak, the simulation-study diagnostics are the
place to look.
