# Under the Hood

## Model

The data are state, action, next-state triples $(s, a, s')$ from a panel of
individuals, each of whom belongs to one latent segment $k \in \{1, \ldots, K\}$.
The prior probability of segment $k$ is $\lambda_k$. Segment $k$ has flow
reward $g_k(s, a)$, shaping potential $h_k(s)$, and policy $\pi_k(a \mid s)$.
The discriminator score for a transition $(s, a, s')$ under segment $k$ is

$$
f_k(s, a, s') = g_k(s, a) + \beta\, h_k(s') - h_k(s).
$$

The segment-$k$ policy satisfies the soft Bellman equation with logit scale
$\sigma$ and transitions $F_a(s' \mid s)$ in orientation $(A, S, S)$:

$$
V_k(s) = \sigma \log \sum_a \exp\!\left(
    \frac{g_k(s, a) + \beta \sum_{s'} F_a(s' \mid s)\, V_k(s')}{\sigma}
\right),
$$

$$
\pi_k(a \mid s) = \frac{
    \exp\!\bigl(Q_k(s, a) / \sigma\bigr)
}{
    \sum_b \exp\!\bigl(Q_k(s, b) / \sigma\bigr)
},
\quad
Q_k(s, a) = g_k(s, a) + \beta \sum_{s'} F_a(s' \mid s)\, V_k(s').
$$

## Anchor Identification

Without normalization, $f_k$ is identified only up to potential-based
transformations: replacing $g_k$ and $h_k$ with $g_k + \phi(s') - \phi(s)$
and $h_k + c$ leaves every likelihood unchanged. Two constraints remove this
ambiguity:

$$
g_k(s, a_{\text{exit}}) = 0 \quad \text{for all } s, k,
$$

$$
h_k(s_{\text{absorb}}) = 0 \quad \text{for all } k.
$$

The exit-action constraint pins reward levels by requiring that the exit action
carries zero flow reward in every state. The absorbing-state constraint pins the
value potential by requiring that continuation value is zero at the absorbing
terminal state. Together they uniquely recover $g_k = r_k^*$ and
$h_k = V_k^*$, the true structural reward and value function.

## Discriminator Loss

The discriminator $D_k$ classifies transitions as coming from segment-$k$'s
expert or from the learned segment-$k$ policy. The classification logit is
$f_k(s, a, s') - \log \pi_k(a \mid s)$. The weighted binary cross-entropy
loss for segment $k$ is

$$
\mathcal{L}_k
= -\sum_i q_{ik} \sum_t \log \sigma\!\bigl(f_k(s_{it}, a_{it}, s_{i,t+1})
    - \log \pi_k(a_{it} \mid s_{it})\bigr)
  - \mathbb{E}_{\pi_k}\!\bigl[\log(1 - D_k)\bigr],
$$

where $\sigma$ denotes the logistic function and $q_{ik}$ is the posterior
probability that trajectory $i$ belongs to segment $k$.

## EM Algorithm

**E-step.** Compute posterior segment probabilities for each trajectory using
the current segment policies and priors:

$$
q_{ik} \propto \lambda_k \prod_{t=1}^{T_i} \pi_k(a_{it} \mid s_{it}).
$$

Log-sum-exp is used for numerical stability. Trajectories from the same
individual are optionally smoothed toward a within-individual consensus to
encourage consistent segment assignment across repeated series.

**Prior update.** Segment priors are updated by the average posterior with
Dirichlet smoothing $\alpha$:

$$
\lambda_k \leftarrow \frac{\textstyle\sum_i q_{ik} + \alpha}{\textstyle\sum_j \!\bigl(\sum_i q_{ij} + \alpha\bigr)}.
$$

**M-step.** For each segment $k$, run the AIRL inner loop. Expert transitions
are weighted by $q_{ik}$; policy transitions are sampled from $\pi_k$. The
inner loop alternates between updating the discriminator parameters (reward
$g_k$ and shaping potential $h_k$) by gradient descent on $\mathcal{L}_k$ and
re-solving the soft Bellman equation to update $\pi_k$. The anchor constraints
are enforced after every discriminator update.

## Pseudocode

```
initialize segment rewards g_k, priors lambda_k, posteriors q_ik
while EM not converged:
    # E-step
    for each trajectory i:
        compute log-likelihood under each segment-k policy
        update q_ik via log-sum-exp normalization
    smooth within-individual posteriors if consistency_weight > 0
    update priors lambda_k with Dirichlet smoothing

    # M-step
    for each segment k:
        collect expert transitions, weighted by q_ik
        for each AIRL round:
            sample policy transitions from pi_k
            gradient step on discriminator loss L_k (g_k, h_k)
            enforce exit-action and absorbing-state anchors
            re-solve soft Bellman for pi_k

    compute mixture log-likelihood; check EM convergence
return segment rewards, policies, values; priors; posteriors
```

## Implementation Notes

The implementation lives in
`econirl.estimation.adversarial.airl_het.AIRLHetEstimator`. Transition tensors
are expected in $(A, S, S)$ orientation. Reward parameters are concatenated
across segments in order: segment 0 first, then segment 1, and so on. The
segment-specific reward matrices and policies are returned in the result
`metadata` dictionary keyed by `segment_reward_matrices`, `segment_policies`,
and `segment_value_functions`. Standard errors for the reward parameters are
not computed; the `standard_errors` field is filled with `nan`.
