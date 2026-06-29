# Classical DDC Proof Routes

Proofs and proof sketches on this page draw from {ref}`Kang (2026)
<kang-2026-lecture>`, {ref}`Rawat and Rust (2026) <rawat-rust-2026>`, and the
classical sources cited inline.

Read this page as the classical route from identification to computation. NFXP,
CCP, and TD-CCP are not three unrelated estimators. They enforce the same
softmax and Bellman restrictions in different ways.

## The Two Equations

The core DDC equations are:

$$
\pi_Q(a\mid s)
=
\frac{\exp Q(s,a)}{\sum_b\exp Q(s,b)}
$$

and

$$
Q(s,a)
= r(s,a)
+\beta \mathbb{E}[V_Q(s')\mid s,a].
$$

The likelihood equation fits observed actions. The Bellman equation connects
the fitted action values to a reward and transition law. The computational
problem is deciding where to impose the Bellman equation.

## NFXP: Put the Bellman Equation Inside the Likelihood

NFXP parameterizes the reward, for example

$$
r_\theta(s,a)=\phi(s,a)^\top\theta,
$$

then solves the Bellman fixed point for every candidate $\theta$:

$$
Q_\theta=T_\theta Q_\theta.
$$

The outer problem maximizes

$$
\sum_{i,t}\log \pi_{Q_\theta}(a_{it}\mid s_{it}).
$$

**Why it works.** The soft Bellman contraction gives a unique
$Q_\theta$ for each $\theta$. Under the DDC assumptions, the likelihood is the
right population criterion for the observed actions.

**What it costs.** Every likelihood evaluation needs an inner dynamic-programming
solve. The transition kernel must be known or estimated first. When the state
space is large, the inner loop is the bottleneck.

**Estimator consequence.** NFXP is the reference tabular estimator. If it is too
slow, later estimators are usually changing the computation, not the basic
structural target.

## CCP: Use Choice Probabilities First

CCP starts from the Hotz-Miller identity:

$$
Q(s,a)-Q(s,a_0)
=
\log\pi(a\mid s)-\log\pi(a_0\mid s).
$$

With an anchor action, the value function solves

$$
V(s)
= r_0(s)
+\beta\mathbb{E}[V(s')\mid s,a_0]
-\log\pi(a_0\mid s).
$$

Once $V$ is solved, recover $Q$ from the log-probability differences and recover
$r$ by the Bellman rearrangement:

$$
r(s,a)=Q(s,a)-\beta\mathbb{E}[V(s')\mid s,a].
$$

**Why it works.** The inversion uses the same softmax equation as NFXP. The
anchor fixed point supplies the missing state-wise levels.

**What it costs.** CCP needs reliable first-stage choice probabilities and a
transition object for the continuation expectations. Weak support or high
dimensional transitions move the hard problem into the first stage.

**Estimator consequence.** CCP is a speed route for finite DDC settings where
choice probabilities and transitions can be estimated with enough support.

## TD-CCP: Use Observed Successor Tuples

TD-CCP keeps the same identification logic but avoids explicit transition-density
estimation in the recursive terms. For the anchor value equation, define the
anchor pseudo-reward

$$
b(s)=r_0(s)-\log\pi(a_0\mid s).
$$

The target value satisfies

$$
V(s)=b(s)+\beta\mathbb{E}[V(s')\mid s,a_0].
$$

With a linear approximation $V_\eta(s)=\rho(s)^\top\eta$, a semi-gradient TD
update uses observed anchor transitions:

$$
\eta_{k+1}
=
\eta_k
+\alpha_k
\left[
b(s_k)+\beta V_{\eta_k}(s_{k+1})-V_{\eta_k}(s_k)
\right]\rho(s_k).
$$

Approximate value iteration instead freezes the old continuation value and fits
a new regression target:

$$
Y_i^{(k)}=b(s_i)+\beta V_k(s_i'),\qquad
V_{k+1}\in\arg\min_{f\in\mathcal{F}}
\sum_i (Y_i^{(k)}-f(s_i))^2.
$$

## Stability and Realizability

The TD route replaces transition-density estimation with observed successor
tuples. That is the gain. The cost is that bootstrapping with function
approximation can be unstable when the data distribution does not match the
transition chain used by the Bellman equation.

If the projection operator is stable and the true value lies in the chosen
function class, fitted value iteration converges to the correct value. If the
true value is outside the class, it converges to a projected fixed point. If the
projected operator is not stable, even that convergence need not hold.

Read the TD-CCP estimator page with this distinction in mind:

- transition-density-free does not mean assumption-free;
- a converged recursive-term fit is not automatically reward identification;
- the basis must span the recursive objects that enter the DDC likelihood.

## Comparison

| Method | Where the Bellman equation enters | Main burden |
| --- | --- | --- |
| NFXP | Inner fixed point for each reward parameter. | Repeated dynamic-programming solves and a transition kernel. |
| CCP | Anchor value equation after estimating choice probabilities. | Choice-probability support and transition expectations. |
| TD-CCP | Successor-sample recursive-term estimation. | Function approximation, support, and projected fixed-point stability. |

The theory pages do not replace the simulation pages. They explain why the
simulation pages check support, parameter recovery, standard errors, and
counterfactual performance instead of treating a fitted likelihood as enough.
