# GLADIUS

GLADIUS learns action-value and continuation-value functions from observed
state, action, and next-state data using neural networks, then recovers
projected structural reward parameters by an action-difference least-squares
step. The method is designed for dynamic discrete choice settings where the
state representation is high-dimensional and repeated Bellman solves over
a candidate parameter are computationally costly. An anchor action with
known rewards pins the absolute level of the learned Q-function; without
the anchor, the network identifies the policy but not the reward in the
full Bellman sense.

## Source Papers

The estimator follows {ref}`Kang, Yoganarasimhan, and Jain (2025) <kang-2025>`,
which proposes an offline empirical-risk minimization objective for IRL in
dynamic discrete choice models, introduces the Q-network and
continuation-value-network architecture, and establishes an anchor-action
identification strategy. The package implementation adapts the paper's
Algorithm 1 to a latent-reward IRL setting where the anchor reward is
supplied by the researcher rather than observed directly.

## Notation

Throughout, $s$ indexes the discrete state and $a$ the discrete action,
observed for individual $i$ in period $t$. The vector $\phi(s, a)$ collects
the reward features and $\theta$ the projected structural parameters. The
discount factor is $\beta$ and the logit shock scale is $\sigma$. The
transition kernel $P_a(s, s')$ gives the probability of moving to $s'$ from
$s$ under action $a$, stored in $(A, S, S)$ orientation. The soft value
function is $V(s)$, the choice-specific value is $Q(s, a)$, and the
conditional choice probability is $\pi(a \mid s)$. Two neural networks
approximate these objects: $Q_\eta$ with parameters $\eta$ approximates
the action-value function, and $\zeta_\xi$ with parameters $\xi$ approximates
the expected continuation value $\mathbb{E}[V(s') \mid s, a]$. The implied
reward is $\hat{r}(s, a) = Q_\eta(s, a) - \beta\,\zeta_\xi(s, a)$. The
anchor action $a_{\text{anch}}$ carries known rewards $r_{\text{anch}}(s)$
for each state, and $\lambda$ is the Bellman penalty weight. The softmax
probability over actions implied by $Q$ is
$\hat{p}_Q(a \mid s) = \exp(Q(s,a)/\sigma) / \sum_{b} \exp(Q(s,b)/\sigma)$,
which equals $\pi(a \mid s)$ when $Q = Q^*$.

## Model

The observed data are state, action, and next-state triples $(s_{it}, a_{it},
s_{i,t+1})$ from a stationary infinite-horizon dynamic discrete choice model.
With discount factor $\beta$ and logit shock scale $\sigma$, the soft Bellman
equation holds:

$$
V(s) = \sigma \log \sum_a \exp\!\left(\frac{Q(s, a)}{\sigma}\right).
$$

The choice-specific value satisfies

$$
Q(s, a) = u(s, a) + \beta\,\mathbb{E}[V(s') \mid s, a],
$$

where $u(s, a) = \phi(s, a)^\top \theta$ is the linear flow payoff. The
conditional choice probability follows the logit rule:

$$
\pi(a \mid s) =
\frac{\exp(Q(s, a) / \sigma)}
     {\sum_b \exp(Q(s, b) / \sigma)}.
$$

The paper derives all results under $\sigma = 1$ (standard T1EV with unit
logit scale, equivalent to setting $\lambda = 1$ in the paper's notation;
see Kang et al. 2025, Section 3.1). The package generalises this to
arbitrary $\sigma$; the equations on this page use the general form.

GLADIUS approximates $Q(s, a)$ and $\mathbb{E}[V(s') \mid s, a]$ by neural
networks rather than solving the Bellman fixed point at each candidate
$\theta$. The canonical evaluation uses synthetic high-dimensional-state
cells in which the true reward, transitions, policy, and value are known,
allowing implied rewards and projected parameters to be compared against
the data-generating truth.

## Identification

GLADIUS identifies the imitation policy and projected structural reward
contrasts under the following assumptions.

- **Additive separability (AS).** The per-period payoff is the flow utility
  plus an additive i.i.d. logit taste shock with scale $\sigma$. This yields
  the soft Bellman representation above without an Euler-Mascheroni constant.
- **Stationarity.** The data-generating process is a stationary infinite-horizon
  dynamic discrete choice model. Transitions and rewards do not vary across
  periods.
- **Anchor availability (the anchor availability condition of Kang et al. (2025)).** An anchor action
  $a_{\text{anch}}$ must exist with known rewards $r_{\text{anch}}(s)$ for every
  state (Assumption 3 of Kang et al. 2025: for all $s \in \mathcal{S}$ there
  exists $a_s$ such that $r(s, a_s)$ is known). To see why this is needed,
  suppose the true Q-function is $Q^*(s, a)$ and consider the shifted function
  $\tilde{Q}(s, a) = Q^*(s, a) + c(s)$ for an arbitrary state-dependent
  constant $c(s)$. The soft value shifts as
  $\tilde{V}(s) = \sigma \log \sum_a \exp\!\bigl((\tilde{Q}(s,a))/\sigma\bigr)
  = V^*(s) + c(s)$, so $c(s)$ cancels in the softmax numerator and denominator
  and $\pi(a \mid s)$ is unchanged: $L_{\text{NLL}}$ alone cannot distinguish
  $Q^*$ from $\tilde{Q}$. The implied reward under $\tilde{Q}$ is
  $\tilde{r}(s, a) = \tilde{Q}(s, a) - \beta\,\mathbb{E}[\tilde{V}(s') \mid s, a]
  = r^*(s, a) + c(s) - \beta\sum_{s'} P(s, a, s')\,c(s')$,
  which does not collapse to $r^*(s, a) + \text{const}$ when the transition
  kernel $P(s, a, \cdot)$ is asymmetric across states. The anchor Bellman loss
  pins $Q_\eta$ at $a_{\text{anch}}$ and eliminates this freedom.
- **Action-dependent feature rank.** The action-difference feature matrix
  $\Delta\phi(s, a) = \phi(s, a) - \phi(s, 0)$, stacked over states and
  non-reference actions, must be full column rank. Because
  $\Delta r(s, a) = \Delta\phi(s, a)^\top \theta$ by the linearity of $u$,
  the OLS normal equations have a unique solution if and only if
  $\Delta\phi$ has full column rank $K$; a rank-deficient matrix leaves the
  projection underdetermined in the direction(s) of the null space.

These hold inside a finite discrete state space with expected-utility
maximization and a known, fixed discount factor $\beta$. Under them, the
action-contrast reward $\Delta r(s, a) = r(s, a) - r(s, 0)$ and the imitation
policy are identified. The reward itself is identified only up to a
state-dependent additive constant; the anchor pins the level of $Q$ for
transitions involving the anchor action. Identification weakens if the anchor
action is absent from some states, the action-difference feature matrix is
rank deficient, or training data provide thin state-action coverage.

**Identification result (Kang et al. 2025, Thm. 3).** Under Additive
Separability, Stationarity, and Anchor Availability, the minimiser of the
expected empirical-risk objective uniquely identifies $Q^*$ over the observed
state-action support. Reward recovery follows as
$r(s, a) = Q^*(s, a) - \beta\,\zeta^*(s, a)$
where $\zeta^*(s, a) = \mathbb{E}[V_{Q^*}(s') \mid s, a]$. The imitation
policy $\pi^*$ is then recovered trivially by taking the softmax of $Q^*$
(this step is not part of Thm. 3). Full column rank of the action-difference
feature matrix $\Delta\phi$ is required for the subsequent OLS projection to
uniquely determine $\theta$; this is a package-level requirement for
structural parameter recovery, not a condition of Thm. 3.

**Minimax reformulation (Kang et al. 2025, Thm. 5).** The minimax empirical-risk
problem (Eq. 12) uniquely identifies $Q^*$ and establishes that the optimal
inner maximiser is $\zeta^*(s,a) = \mathbb{E}[V_{Q^*}(s') \mid s, a]$.

**Convergence result (Kang et al. 2025, Section 6).** Under a
Polyak-Łojasiewicz (PL) condition on the objective landscape,
gradient-based optimisation converges to the global minimiser. The empirical
objective gap decreases at $O(T^{-1})$ in the number of gradient iterations
$T$ (Kang et al. 2025, Thm. 12), and the $L^2(d^*)$ estimation error in $Q^*$
decreases at $O(T^{-1/4})$ at $\alpha = 1/2$ (Thm. 13). No proof is reproduced
here; see Kang et al. (2025) for the full argument.

## Estimator

The Q-network is trained by minimizing the combined objective

$$
L_Q = L_{\text{NLL}} + \lambda\,L_{\text{anch}},
$$

where the negative log-likelihood term fits the observed choice distribution:

$$
L_{\text{NLL}} = -\mathbb{E}\!\left[\log \pi_\eta(a \mid s)\right],
$$

with $\log \pi_\eta(a \mid s)
= Q_\eta(s, a)/\sigma - \log \sum_b \exp(Q_\eta(s, b)/\sigma)$
(the softmax log-probability, expanded in the Algorithm below). The anchor Bellman term pins the Q-level at the
anchor action. The paper's primary form of this term (Eq. 12 of Kang et al. 2025)
is the minimax objective

$$
\min_{Q}\,\max_{\zeta}\;
\mathbb{E}\!\left[
-\log \hat{p}_Q(a \mid s)
+ \mathbf{1}_{a = a_{\text{anch}}}\!\left(
  (\mathcal{T}Q(s,a,s') - Q(s,a))^2
  - \beta^2(V_Q(s') - \zeta(s,a))^2
\right)
\right],
$$

where $\mathcal{T}Q(s,a,s') = r_{\text{anch}}(s) + \beta V_Q(s')$ is the
sampled Bellman operator on the anchor action. The correction term
$-\beta^2(V_Q(s') - \zeta(s,a))^2$ is the key structural contribution of
the paper: the naive squared TD error
$\mathbb{E}[(\mathcal{T}Q - Q)^2 \mid s, a]$ has a double-sampling
bias equal to $\beta^2 \operatorname{Var}[V(s') \mid s, a]$ because the same
next-state draw $s'$ appears in both $\mathcal{T}Q$ and $V_Q(s')$.
Lemma 4 of Kang et al. (2025) shows that the bi-conjugate decomposition
$\mathbb{E}[(\mathcal{T}Q - Q)^2 \mid s, a]
= \mathbb{E}[\ell_{\text{TD}}^2 \mid s, a, s'] - \beta^2 D(Q)(s,a)$,
where $\ell_{\text{TD}}(s,a,s') = \mathcal{T}Q(s,a,s') - Q(s,a)$
is the per-sample TD residual (the signed difference between the sampled
Bellman target and the current Q value) and
$D(Q)(s,a) = \min_\zeta \mathbb{E}[(V_Q(s') - \zeta)^2 \mid s, a]$,
corrects for this bias. The inner $\max$ over $\zeta$ in Eq. 12 exploits
this decomposition: the optimal $\zeta^*$ that solves the inner max is the
conditional expectation $\zeta^*(s, a) = \mathbb{E}[V_{Q^*}(s') \mid s, a]$
(Theorem 5), and the $-\beta^2(V_Q(s') - \zeta)^2$ term cancels the
double-sampling bias when $\zeta$ is at its optimum.

The package defaults to `anchor_bellman_mode="anchor_moment"`, which replaces
the minimax correction with the continuation-moment residual

$$
L_{\text{anch}} =
\mathbb{E}_{a = a_{\text{anch}}}\!\left[
\bigl(Q_\eta(s, a) - r_{\text{anch}}(s) - \beta\,\zeta_\xi(s, a)\bigr)^2
\right].
$$

This variant avoids the explicit minimax and is more stable in the latent-reward
IRL setting (where the paper's observed-reward anchor is not available in
the same form). Setting `anchor_bellman_mode="paper_minimax"` retains the
literal Eq. 12 objective. The two modes differ in sign and structure: the
paper_minimax mode subtracts the variance correction $\beta^2(V_Q(s')-\zeta)^2$
from the squared TD error, whereas the anchor_moment mode treats
$Q - r_{\text{anch}} - \beta\zeta$ as a direct residual.

In the paper, $\zeta(s, a) = \mathbb{E}[V_Q(s') \mid s, a]$ is the
closed-form conditional expectation that solves the inner maximisation in
Theorem 5. It is an analytic object, not a separate network. The package
approximates this conditional expectation with a neural network $\zeta_\xi$
trained by MSE regression:

$$
L_\zeta = \mathbb{E}\!\left[
\bigl(\zeta_\xi(s, a) - V_{Q_\eta}(s')\bigr)^2
\right],
$$

where $V_{Q_\eta}(s') = \sigma \log \sum_b \exp(Q_\eta(s', b) / \sigma)$
is the soft value of the next state under the current Q-network. The
minimiser of $L_\zeta$ over $\zeta_\xi$ is precisely $\mathbb{E}[V_{Q_\eta}(s') \mid s, a]$,
so the learned network targets the analytic object; the distinction from the
paper is that the package replaces the closed-form inner max with a
learned approximation.

After training, the implied reward $\hat{r}(s, a) = Q_\eta(s, a) - \beta\,\zeta_\xi(s, a)$
is projected onto structural features via action-difference least squares:

$$
\hat{\theta} = \arg\min_\theta
\|\Delta\hat{r} - \Delta\phi\,\theta\|^2,
$$

where $\Delta\hat{r}(s, a) = \hat{r}(s, a) - \hat{r}(s, 0)$ and
$\Delta\phi(s, a) = \phi(s, a) - \phi(s, 0)$ are stacked over all states
and all non-reference actions. The projection is solved in closed form.

The action-difference is taken because $Q(s, a)$, and therefore
$\hat{r}(s, a)$, contains a state-dependent additive constant $c(s)$
that is unidentified without the anchor (Kim et al. 2021; Cao et al. 2021):
subtracting the reference action $a = 0$ eliminates $c(s)$ from both
$\Delta\hat{r}$ and $\Delta\phi$, leaving $\Delta r(s,a) = \Delta\phi(s,a)^\top\theta$
as the identified object. Note that the reference action (index 0, a package
convention) and the anchor action $a_{\text{anch}}$ serve distinct roles:
the reference action is the OLS baseline for differencing; the anchor action
pins the absolute level of $Q_\eta$ via the Bellman loss. They need not
be the same action.

## Algorithm

```text
Algorithm  GLADIUS (dual-network, alternating updates, anchor-moment mode)
Input   panel {(s_it, a_it, s_{i,t+1})}, features phi, discount beta,
        logit scale sigma, anchor action a_anch, anchor rewards r_anch,
        Bellman penalty weight lambda, patience P
Output  theta_hat, imitation policy pi, soft value V

1   initialize Q_eta and zeta_xi as MLPs
    # value_scale = 1/(1-beta): implementation trick that rescales MLP outputs
    # to the range of V, keeping gradients well-conditioned (not part of the
    # mathematical formulation)
    # state_encoder: fixed function mapping integer state s to a real feature
    # vector (e.g., one-hot or learned embedding)
2   initialize Adam optimizers with learning-rate decay and gradient clipping
3   for each epoch until max_epochs:
4       shuffle panel into mini-batches
5       for each mini-batch (s, a, s') with index b:
6           s_feat := state_encoder(s);  sp_feat := state_encoder(s')
7           if b is even:                          # zeta step, Q held fixed
8               V_sp := sigma * logsumexp(Q_eta(sp_feat, all a) / sigma)
9               update zeta_xi to minimize (zeta_xi(s_feat, a) - V_sp)^2
10          else:                                  # Q step, zeta held fixed
11              log_pi := Q_eta(s_feat, all a) / sigma
12                        - logsumexp(Q_eta(s_feat, all a) / sigma)
13              L_NLL  := -mean(log_pi[a])
14              mask   := (a == a_anch)
15              L_anch := mean over mask of
16                        (Q_eta(s_feat, a) - r_anch[s] - beta * zeta_xi(s_feat, a))^2
17              update Q_eta to minimize L_NLL + lambda * L_anch
18      if no improvement for P epochs: break      # early stopping
19  r_hat(s, a) := Q_eta(s, a) - beta * zeta_xi(s, a)    # implied reward
20  Delta_r      := r_hat(s, a) - r_hat(s, 0)             # action differences
21  Delta_phi    := phi(s, a)   - phi(s, 0)
22  theta_hat    := lstsq(Delta_phi, Delta_r)              # closed-form OLS
23  pi(a | s)    := softmax(Q_eta(s, all a) / sigma)
24  V(s)         := sigma * logsumexp(Q_eta(s, all a) / sigma)
25  return theta_hat, pi, V
```

The default configuration is `mode="dual"` with `alternating_updates=True`
and `anchor_bellman_mode="anchor_moment"`: separate Q and zeta networks are
trained in alternating mini-batch steps, and the anchor pins $Q_\eta$ via the
continuation-moment residual above. Two alternative modes are available.
`mode="q_only"` drops the zeta network and replaces $\mathbb{E}[V(s') \mid s, a]$
with $\sigma \log \sum_b \exp(Q_\eta(s', b) / \sigma)$; the anchor Bellman
loss is not available in this mode and $Q_\eta$ is trained on $L_{\text{NLL}}$
alone. `anchor_bellman_mode="paper_minimax"` uses the literal bi-conjugate
minimax term from the paper rather than the continuation-moment anchor. It is
available for comparison and is not the default.
Setting `alternating_updates=False` updates both networks jointly in each
step, which is the legacy behavior.

## System View

GLADIUS replaces repeated tabular dynamic-programming solves with learned Q and
continuation objects. After those objects are trained, it reads reward
information from action differences.

```text
Offline panel: state, action, next state
State/action features, discount factor, anchor action
        |
        v
Train Q network to fit observed choices
        |
        v
Train continuation network to predict next-state soft value
        |
        v
Use the anchor action to pin the Q level
        |
        v
Compute implied reward from Q minus discounted continuation
        |
        v
Project action contrasts onto structural reward features
```

The most defensible structural object is the projected action contrast. Raw
Bellman reward levels are weaker unless the paper-style anchor conditions are
met well in the data.

## Applicability

| Applicable when | Prefer an alternative when |
| --- | --- |
| State features are high-dimensional and tabular Bellman solves are costly. | The state space is small and tabular (use NFXP, CCP, or UFXP). |
| An anchor action with known rewards is available. | There is no credible anchor action. |
| Policy imitation and projected reward recovery are the target. | Full structural counterfactual validity is required. |
| A fast neural approximation suffices for model exploration. | Structural standard errors or formal inference are required. |

GLADIUS sits in the behavioral family alongside MCE-IRL, AIRL, and IQ-Learn.
It learns its Q and continuation-value networks offline, from the fixed panel,
without policy rollouts or a dynamic-program solve at each candidate parameter.
That is what lets it scale to state representations that make the structural
family expensive. Against
simpler behavioral methods, GLADIUS adds explicit Bellman structure through
the Q and zeta architecture and the anchor Bellman loss, which provides a
path toward structural parameter recovery that policy-gradient or
behavioral-cloning methods do not have. The current evaluation cells show
strong policy imitation and projected-reward recovery. Raw Bellman reward and
value recovery are weaker, so GLADIUS is best read as a policy-imitation and
action-contrast method on those cells, not a source of counterfactual rewards.

## Usage

```python
from econirl import GLADIUS

model = GLADIUS(
    n_actions=3,
    discount=0.95,
    anchor_action=2,
    anchor_rewards=known_r2,   # known reward for action 2 in each state
    max_epochs=300,
)
model.fit(df, state="state_bin", action="action", id="individual_id",
          features=reward_spec)

print(model.params_)     # projected structural parameters (dict)
print(model.policy_)     # imitation policy, shape (n_states, n_actions)
```

The anchor sets the scale of the recovered reward. Pick an action whose
per-state reward is known, pass its index as `anchor_action`, and pass the
known reward vector as `anchor_rewards`. Without `anchor_rewards` the recovered
reward points in the right direction but its magnitude is understated. When the
state index is not the state, pass a `state_encoder` that maps each state to
its feature vector and set `state_dim` to that width.

The fitted attributes give the projected structural parameters, projection
diagnostics, and the implied policy and value function:

```python
print(model.coef_)              # parameter vector as numpy array
print(model.se_)                # projection standard errors
print(model.projection_r2_)    # R-squared of the action-difference projection
print(model.value_)             # soft value function, shape (n_states,)
print(model.predict_proba([0, 5, 15, 20]))   # policy at selected states
```

Counterfactual analysis uses the projected parameters as inputs to a
structural Bellman solver under an intervened primitive. The re-solved policy
and value can then be compared against the original. The
[Counterfactuals](gladius/counterfactuals.md) page describes the regret
evaluation protocol and the current scope of counterfactual results. The
[Quick Start](gladius/quick_start.md) page documents the full fitted-attribute
table and the lower-level `GLADIUSEstimator` interface.

## Evidence

GLADIUS is evaluated on two synthetic high-dimensional-state cells with
known rewards, transitions, policies, and counterfactual oracle objects.
The primary cell has 21 discrete states encoded into 64-dimensional feature
vectors, a 4-parameter linear reward, and an anchor action with known rewards.
Policy imitation and projected reward recovery are within acceptable bounds. Raw
Bellman reward and value recovery are less accurate.

Behavioral recovery and counterfactual regret on the primary cell (21 states, 3
actions, 1,000 individuals, 100 periods):

| Metric | Value |
| --- | ---: |
| Policy total variation | 0.0369 |
| Q NRMSE | 0.235 |
| Projected reward NRMSE | 0.198 |
| Raw Bellman reward NRMSE | 0.571 |
| Value NRMSE | 0.420 |
| Type A regret (reward shift) | 0.00854 |
| Type B regret (transition change) | 0.0529 |
| Type C regret (action removed) | 0.00852 |

The imitation policy is close to the data-generating policy (TV 0.0369). The
projected parameters show high directional alignment with the truth (parameter
cosine 0.975). That alignment should be read with care. Reward is only partially
identified in general, and the projection matches the ground truth only when the
anchor availability condition holds in the study cell. The raw Bellman reward and
value are recovered less accurately, with NRMSE 0.571 and 0.420. On these cells
GLADIUS recovers the imitation policy and the action-contrast reward well, but not
the absolute reward and value that counterfactual re-solving needs. The small
regret figures alone do not establish structural counterfactual validity.

For cross-estimator behavioral comparisons, including GLADIUS alongside the
structural and IRL rosters, see the
[fleet maintenance simulation study](../simulation_studies/fleet_maintenance.md).

## References

Source papers:

- Kang, E. H., Yoganarasimhan, H., and Jain, L. (2025). An Empirical Risk
  Minimization Approach for Offline Inverse RL and Dynamic Discrete Choice
  Model. *arXiv preprint arXiv:2502.14131*.
  {ref}`reference entry <kang-2025>`.

Implementation and reproduction:

- Estimator source: [`econirl.estimation.gladius`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimation/gladius.py).
- sklearn wrapper: [`econirl.GLADIUS`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimators/neural_gladius.py).
- Validation runner: [`validation/estimators/gladius/run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/gladius/run.py).
- Results files: [`gladius.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius.json), [`gladius_scaled.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius_scaled.json).

Pages:

- [Quick Start](gladius/quick_start.md)
- [Pre-Estimation Checks](gladius/pre_estimation.md)
- [Simulation Study](gladius/validation.md)
- [Counterfactuals](gladius/counterfactuals.md)
- [High-State Example](gladius/high_state_example.md)

```{toctree}
:hidden:

gladius/quick_start
gladius/pre_estimation
gladius/validation
gladius/counterfactuals
gladius/high_state_example
```
