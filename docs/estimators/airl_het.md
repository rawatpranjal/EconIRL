# AIRL-Het

AIRL-Het extends adversarial inverse reinforcement learning to populations with
latent segments. Each segment has its own reward and policy, recovered jointly
through an expectation-maximization loop that alternates between assigning
trajectories to segments and updating each segment's adversarial reward and
policy. Two anchor normalizations, one on the exit action and one on the
absorbing state, pin the action-dependent reward uniquely, so the recovered
reward equals the structural reward rather than a potential-based perturbation
of it. The result is a segment-level reward object that supports structural
counterfactual analysis on heterogeneous populations.

Read this page when one pooled reward is not credible. The anchors and the
latent-segment separation are the main reasons the recovered rewards can be
interpreted segment by segment.

## Source Papers

The estimator builds on {ref}`Fu, Luo, and Levine (2018) <fu-2018>`, which
introduces the adversarial IRL discriminator and the potential-based shaping
construction. {ref}`Lee, Sudhir, and Wang (2026) <lee-sudhir-wang-2026>`
develop the anchored heterogeneous extension: the two-constraint
normalization that uniquely identifies action-dependent rewards in dynamic
discrete choice environments with absorbing exits, and the EM algorithm that
discovers latent segments from pooled trajectory data.

## Notation

Throughout, $s$ indexes the discrete state and $a$ the discrete action, observed
for individual $i$ in period $t$. The latent segment index is
$k \in \{1, \ldots, K\}$. The segment prior probability is $\lambda_k$ and
the posterior probability that trajectory $i$ belongs to segment $k$ is
$q_{ik}$. Segment $k$ has flow reward $g_k(s, a)$ and shaping potential
$h_k(s)$; the discriminator score is
$f_k(s, a, s') = g_k(s, a) + \beta h_k(s') - h_k(s)$, where $\beta$ is the
discount factor. The logit shock scale is $\tau > 0$. The logistic function is
written $\ell(x) = 1/(1 + e^{-x})$. The transition kernel
$P_a(s, s')$ gives the probability of moving to $s'$ from $s$ under action $a$,
stored in $(A, S, S)$ orientation. The integrated value function for segment
$k$ is $V_k(s)$, the choice-specific value is $Q_k(s, a)$, and the
segment-$k$ policy is $\pi_k(a \mid s)$.

## Model

The data are state, action, next-state trajectories
$(s_{it}, a_{it}, s_{i,t+1})$ for a panel of individuals. Each individual
belongs to one latent segment $k$. Segment $k$ maintains a discriminator score:

$$
f_k(s, a, s') = g_k(s, a) + \beta\, h_k(s') - h_k(s).
$$

The segment-$k$ integrated value function satisfies the soft Bellman fixed
point:

$$
V_k(s) = \tau \log \sum_a \exp\!\left(
    \frac{g_k(s, a) + \beta \sum_{s'} P_a(s, s')\, V_k(s')}{\tau}
\right).
$$

This follows from additive separability (AS) and Type-I extreme-value logit
shocks with scale $\tau$: integrating out the shock yields the log-sum-exp
inclusive value (Rust 1987; Ziebart 2008).

The choice-specific value and logit policy follow:

$$
Q_k(s, a) = g_k(s, a) + \beta \sum_{s'} P_a(s, s')\, V_k(s'),
\qquad
\pi_k(a \mid s) =
\frac{\exp(Q_k(s, a) / \tau)}{\sum_b \exp(Q_k(s, b) / \tau)}.
$$

The mixture layer links the segment objects to the observed data through segment
priors $\lambda_k$ and trajectory-level posteriors $q_{ik}$. The mixture
log-likelihood is:

$$
\mathcal{L}_{\mathrm{mix}} = \sum_i \log \sum_k \lambda_k \prod_t \pi_k(a_{it} \mid s_{it}).
$$

The canonical instance is a serialized-content panel: individuals decide each
period whether to read (advance through content), wait, or exit permanently.
Two latent segments differ in their dynamic preferences, and the exit action
absorbs each individual into a terminal state.

## Identification

This is the section that says when the segment rewards are structurally pinned,
not just clustered policies with different labels.

AIRL-Het recovers segment-level structural rewards under the following
assumptions. Absent the anchor normalizations, the discriminator score
$f_k$ is identified only up to potential-based transformations; the two anchors
together uniquely resolve that ambiguity (Lee, Sudhir, and Wang 2026).

- **Conditional independence (CI).** The observed state transition is Markov in
  the current state and action and does not depend on the current logit shock.
- **Additive separability (AS).** The per-period payoff is the systematic reward
  plus an additive choice-specific shock, drawn independently across choices as
  Type-I extreme value with fixed scale $\tau$.
- **Exogenous transitions.** The transition kernel $P_a(s, s')$ is supplied or
  estimated in a first stage, outside the adversarial objective.
- **Exit-action reward anchor.** The exit action carries zero flow reward in
  every state and segment: $g_k(s, a_{\text{exit}}) = 0$ for all $s$ and $k$.
  This pins the reward level.
- **Absorbing-state shaping anchor.** The shaping potential is zero at the
  absorbing terminal state: $h_k(s_{\text{absorb}}) = 0$ for all $k$. Together
  with the exit-action anchor this pins the value potential, so that the
  recovered shaping function equals the true value function $V_k^*$.
- **Action-dependent feature rank.** The reward features must vary across
  actions. State-only features difference out of action contrasts and leave
  the reward parameters unidentified even under correct anchors.
- **Latent segment separability.** The EM assignment is informative only when
  the segments choose differently across states. Populations whose segments
  have nearly identical policies require substantially larger samples to
  separate.

These hold inside a finite discrete state space, a stationary environment, and
a known, fixed discount factor $\beta$. Given them, the structural reward
$g_k = r_k^*$ and value potential $h_k = V_k^*$ are jointly identified for
each segment, up to the alignment of estimated segments to true segments. The
conventional identifiers weaken when the exit action or absorbing state is
misspecified, when the feature matrix is rank-deficient, or when segment
behavioral separation is small.

## Estimator

The discriminator output for segment $k$ is

$$
D_k(s, a, s') = \frac{\exp(f_k(s, a, s'))}{\exp(f_k(s, a, s')) + \pi_k(a \mid s)},
$$

so the log-odds equal $\log D_k - \log(1-D_k) = f_k - \log \pi_k$, which is
the binary classification logit used in both terms of $\mathcal{L}_k$ (Fu et
al. 2018, §3). The discriminator for segment $k$ is trained by weighted binary
cross-entropy. Expert transitions are weighted by the current posterior
$q_{ik}$:

$$
\mathcal{L}_k
= -\sum_i q_{ik} \sum_t \log \ell\!\bigl(f_k(s_{it}, a_{it}, s_{i,t+1})
    - \log \pi_k(a_{it} \mid s_{it})\bigr)
  - \mathbb{E}_{\pi_k}\!\bigl[\log(1 - D_k)\bigr],
$$

where $\ell(x) = 1/(1+e^{-x})$ is the logistic function. Substituting
$D_k = \ell(f_k - \log \pi_k)$ gives
$\log(1 - D_k) = -\log(1 + \exp(f_k - \log \pi_k))$, so the policy term
is $\mathbb{E}_{\pi_k}[\log(1 - D_k)] = -\mathbb{E}_{\pi_k}[\operatorname{logaddexp}(0,\, f_k - \log \pi_k)]$.
The E-step posterior is:

$$
q_{ik} \propto \lambda_k \prod_t \pi_k(a_{it} \mid s_{it}),
$$

computed via log-sum-exp for numerical stability. Segment priors are updated
by the average posterior with Dirichlet smoothing ($N$ = number of
trajectories):

$$
\lambda_k \leftarrow
\frac{\textstyle\frac{1}{N}\sum_i q_{ik} + \alpha}
     {\textstyle\sum_j \bigl(\frac{1}{N}\sum_i q_{ij} + \alpha\bigr)}.
$$

Here $\alpha > 0$ is the Dirichlet concentration (smoothing pseudocount) and
$j$ ranges over all segments $\{1, \ldots, K\}$ so the denominator sums each
segment's smoothed average posterior to ensure the updated priors sum to one.

Standard errors for the reward parameters are not available; the adversarial
objective does not produce a likelihood Hessian, and the `standard_errors`
field is filled with `nan`.

## Algorithm

```text
Algorithm  AIRL-Het (EM adversarial IRL with anchor identification)
Input   panel {(s_it, a_it, s_{i,t+1})}, transitions P in (A,S,S),
        K segments, exit_action index, absorbing_state index,
        discount beta, logit scale tau, EM tolerance em_tol
Output  segment rewards {g_k}, shaping potentials {h_k},
        segment policies {pi_k}, priors {lambda_k}, posteriors {q_ik}

1   initialize g_k, h_k randomly; set lambda_k := 1/K, q_ik := 1/K
2   enforce g_k(s, exit_action) := 0 for all s, k
3   enforce h_k(absorbing_state) := 0 for all k
4   solve soft Bellman for pi_k using hybrid value iteration        # default generator_solver
5   repeat                                                          # EM outer loop
6       # E-step
7       for each trajectory i:
8           log_lik_k(i) := log lambda_k + sum_t log pi_k(a_it | s_it)
9           q_ik := normalize(exp(log_lik_k(i))) via log-sum-exp
10      smooth q_ik within individuals toward per-individual consensus
11      lambda_k := (mean_i q_ik + alpha) / sum_j (mean_i q_ij + alpha)
12      # M-step
13      for each segment k:
14          collect expert transitions weighted by q_ik
15          repeat                                                   # AIRL inner loop
16              sample policy transitions from current pi_k
17              gradient step on discriminator loss L_k(g_k, h_k)
18              enforce g_k(s, exit_action) := 0, h_k(absorbing_state) := 0
19              re-solve soft Bellman for pi_k via hybrid value iteration
20          until max policy change < airl_convergence_tol
21      mixture_ll := sum_i log sum_k lambda_k prod_t pi_k(a_it | s_it)
22  until |delta mixture_ll| / |mixture_ll| < em_tol
23  return {g_k}, {h_k}, {pi_k}, {lambda_k}, {q_ik}
```

The inner solver at steps 4 and 19 defaults to `generator_solver="hybrid"`,
which runs hybrid value iteration (successive approximation until the iterates
are close, then Newton steps for rapid convergence near the fixed point).
The alternative `generator_solver="value"` applies pure successive
approximation throughout, which is more robust from arbitrary starting points
at the cost of slower convergence. The reward parameterization defaults to
`reward_type="tabular"` (one free parameter per state-action pair per segment);
`reward_type="linear"` projects onto a supplied feature matrix.
Initialization defaults to `initialization="random"`; the
`"behavioral_anchor"` scheme clusters trajectories by observed action shares,
inverts the anchored soft policy into a starting reward, and typically reduces
the number of EM iterations required.

## System View

AIRL-Het adds a segment layer to anchored adversarial reward recovery. Each
person or trajectory receives a persistent segment probability, and each
segment gets its own reward and policy.

```text
Trajectories grouped by person
Exit action, absorbing state, transition model, discount factor
        |
        v
Initialize segment shares and trajectory memberships
        |
        v
For each segment, run anchored AIRL with the two reward anchors
        |
        v
Update which segment each trajectory most likely belongs to
        |
        v
Repeat until memberships and segment policies stabilize
        |
        v
Segment-specific rewards, policies, shares, and assignments
```

The latent segment is a person-level or trajectory-level type. It is not a new
random draw at each decision. That distinction matters for reading the recovered
segments and their counterfactual policies.

## Applicability

| Applicable when | Prefer an alternative when |
| --- | --- |
| The population has latent segments with distinct dynamic preferences. | A single homogeneous reward is a defensible assumption. |
| Individuals have repeated trajectories that help identify segment membership. | Each individual appears only once or very briefly. |
| An exit action and an absorbing terminal state are credibly available. | No credible reward anchor can be specified. |
| Action-dependent structural reward recovery is the target. | State-only reward recovery is sufficient. |
| Segment-specific counterfactual analysis is central. | Only fitted choice probabilities are required. |

AIRL-Het sits at the intersection of adversarial IRL and structural dynamic
discrete choice. Against base AIRL, it adds two things: the anchor
normalization for action-dependent identification, and segment-specific reward
and policy objects for heterogeneous populations. The single-segment
anchored case is accessible via `num_segments=1`. Against the structural
estimators (NFXP, CCP, MPEC, UFXP), AIRL-Het does not maximize a likelihood
and does not report standard errors; it is appropriate when segment-level
reward recovery and counterfactual behavior are the research target rather
than asymptotic inference on structural parameters.

## Usage

```python
from econirl.estimation.adversarial.airl_het import AIRLHetConfig, AIRLHetEstimator

config = AIRLHetConfig(
    num_segments=2,
    exit_action=2,          # index of the anchor action (reward = 0)
    absorbing_state=20,     # index of the absorbing terminal state (V = 0)
    reward_type="tabular",
    max_em_iterations=30,
)
estimator = AIRLHetEstimator(config=config)

summary = estimator.estimate(
    panel=panel,
    utility=utility,
    problem=problem,
    transitions=transitions,
)

print(summary.metadata["segment_priors"])
print(summary.metadata["segment_assignments"][:10])
```

Counterfactual analysis re-solves each segment's dynamic program under a
controlled change to the environment. The segment reward matrices and the
package solver utilities support all three counterfactual families:

```python
from econirl.core.bellman import SoftBellmanOperator
import jax.numpy as jnp

operator = SoftBellmanOperator(problem, transitions)
seg_rewards = summary.metadata["segment_reward_matrices"]

# Type A counterfactual: raise the read-action reward for each segment
exit_action = config.exit_action
for k, rw in enumerate(seg_rewards):
    rw_cf = jnp.array(rw).at[:, 0].add(0.5)
    rw_cf = rw_cf.at[:, exit_action].set(0.0)   # re-enforce anchor
    from econirl.core.solvers import value_iteration
    result = value_iteration(operator, rw_cf)
    print(f"segment {k} counterfactual policy:", result.policy.shape)
```

The [Quick Start](airl_het/quick_start.md) page documents the full set of
fitted attributes, the initialization options, and the reward-type settings.

## Evidence

AIRL-Het is evaluated on a synthetic serialized-content environment with two
latent segments, 61 states, 3 actions (read, wait, exit), 20 reward features, and
known segment-level rewards, policies, values, Q functions, and counterfactual
oracle objects. 58 of 61 states are observed; three states outside the
simulation support are expected given the content dynamics and do not
indicate a data problem, but they bound what segment heterogeneity claims
can support.

Behavioral fit and segment recovery on that cell, from
[aairl.json](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/aairl.json):

| Metric | Value |
| --- | ---: |
| Max segment policy total variation | 0.0591 |
| Segment assignment accuracy | 0.895 |
| Segment prior L1 error | 0.0435 |

Counterfactual regret per intervention family (maximum across segments):

| Counterfactual | Max regret |
| --- | ---: |
| Type A (reward shift) | 0.0145 |
| Type B (transition change) | 0.1189 |
| Type C (action removal) | 0.00687 |

Type B regret is the highest of the three. The transition-change counterfactual
is more demanding for adversarial estimators than reward shifts or action
removals. The recovered reward must transfer across a dynamically different world
without a likelihood-based correction.

AIRL-Het is not included in the current cross-estimator simulation studies,
which target homogeneous populations. For the full study roster, see the
[simulation studies index](../simulation_studies/index.md).

## References

Source papers:

- Fu, J., Luo, K., and Levine, S. (2018). Learning Robust Rewards with
  Adversarial Inverse Reinforcement Learning. _International Conference on
  Learning Representations_.
  {ref}`reference entry <fu-2018>`.
- Lee, P. S., Sudhir, K., and Wang, T. (2026). Consumer Engagement with
  Sequential Content: A Content-Aware Dynamic Choice Model. SSRN working
  paper, abstract 6331041.
  {ref}`reference entry <lee-sudhir-wang-2026>`.

Implementation and reproduction:

- Estimator source: [`econirl.estimation.adversarial.airl_het`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimation/adversarial/airl_het.py).
- Validation runner: [`validation/estimators/aairl/run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/aairl/run.py).
- Results file: [`aairl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/aairl.json).

Pages:

- [Quick Start](airl_het/quick_start.md)
- [Pre-Estimation Checks](airl_het/pre_estimation.md)
- [Simulation Study](airl_het/validation.md)
- [Counterfactuals](airl_het/counterfactuals.md)
- [Serialized-Content Example](airl_het/serialized_content.md)

```{toctree}
:hidden:

airl_het/quick_start
airl_het/pre_estimation
airl_het/validation
airl_het/counterfactuals
airl_het/serialized_content
```
