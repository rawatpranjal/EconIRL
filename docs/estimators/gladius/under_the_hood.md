# Under the Hood

## Model

The data are state, action, next-state triples $(s, a, s')$ from a stationary
infinite-horizon dynamic discrete choice model. The agent has flow utility
$u(s,a)$ over discrete actions, discount factor $\beta$, and i.i.d. logit taste
shocks with scale $\sigma$. Under logit shocks the soft Bellman equation holds
without an Euler-gamma constant:

$$
V(s) = \sigma \log \sum_{a} \exp\!\Bigl(\frac{Q(s,a)}{\sigma}\Bigr),
$$

where $Q(s,a)$ is the choice-specific value satisfying

$$
Q(s,a) = u(s,a) + \beta\,\mathbb{E}\bigl[V(s') \mid s,a\bigr].
$$

Choice probabilities follow the softmax of scaled Q-values,

$$
\pi(a \mid s) = \frac{\exp(Q(s,a)/\sigma)}{\sum_{a'}\exp(Q(s,a')/\sigma)}.
$$

Transition tensors are in the package-standard orientation $(A, S, S)$, so
$F_a(s' \mid s)$ is row $s$ of the $a$-th matrix.

## Networks

GLADIUS parameterizes two functions with MLPs. The Q-network
$Q_\eta(s,a)$ maps state features and a one-hot action to a scalar
action-value. The zeta network $\zeta_\xi(s,a)$ approximates the
expected continuation value,

$$
\zeta_\xi(s,a) \approx \mathbb{E}\bigl[V(s') \mid s, a\bigr].
$$

The implied reward is then extracted pointwise:

$$
\hat{r}(s,a) = Q_\eta(s,a) - \beta\,\zeta_\xi(s,a).
$$

## Training Objective

Training uses alternating mini-batch updates following Algorithm 1 of
{ref}`Kang, Yoganarasimhan, and Jain (2025) <kang-2025>`. Even batches update
the zeta network to minimize the squared error against the soft value of the
next state under the current Q-network (Q is frozen during this step):

$$
L_\zeta = \mathbb{E}\!\left[\Bigl(\zeta_\xi(s,a) - V_{Q_\eta}(s')\Bigr)^2\right].
$$

Odd batches update the Q-network via negative log-likelihood of observed actions,

$$
L_{\text{NLL}} = -\mathbb{E}\!\left[\log \pi_\eta(a \mid s)\right],
$$

plus, when known anchor rewards $r_{\text{anch}}(s)$ are supplied for an anchor
action $a_{\text{anch}}$, an anchor Bellman term that pins the absolute level of
$Q$:

$$
L_{\text{anch}} = \mathbb{E}_{a = a_{\text{anch}}}\!\left[
\Bigl(Q_\eta(s,a) - r_{\text{anch}}(s) - \beta\,\zeta_\xi(s,a)\Bigr)^2
\right].
$$

The total Q-network loss is

$$
L_Q = L_{\text{NLL}} + \lambda\,L_{\text{anch}},
$$

where $\lambda$ is the Bellman penalty weight (`bellman_penalty_weight`).
Without an anchor, the Q-network is trained on $L_{\text{NLL}}$ alone and
$Q$ is identified only up to a state-dependent constant.

## Structural Parameter Recovery

IRL rewards are identified only up to a state-dependent additive constant. The
action-difference projection removes that constant. For each action
$a \in \{1, \ldots, A-1\}$ and each state $s$, define

$$
\Delta\hat{r}(s,a) = \hat{r}(s,a) - \hat{r}(s,0),
\qquad
\Delta\varphi(s,a) = \varphi(s,a) - \varphi(s,0),
$$

where $\varphi(s,a)$ is the feature vector for state-action pair $(s,a)$.
Stacking over all states and non-anchor actions gives a least-squares problem,

$$
\hat{\theta} = \arg\min_\theta \|\Delta\hat{r} - \Delta\varphi\,\theta\|^2,
$$

which is solved in closed form. The projection identifies both level and slope
parameters because the state-dependent constant cancels in the action
differences.

## Pseudocode

```
extract (s, a, s') tuples from panel
initialize Q-network and zeta-network
for each epoch:
    for each even mini-batch:
        update zeta to minimize (zeta(s,a) - V_Q(s'))^2   # Q frozen
    for each odd mini-batch:
        update Q to minimize NLL + lambda * anchor_bellman  # zeta frozen
    apply early stopping if patience exceeded
compute implied rewards: r_hat = Q - beta * zeta
stack action-difference rewards and features
solve OLS for theta_hat
report theta_hat, imitation policy, and soft value function
```

## Implementation Notes

The implementation lives in `econirl.estimation.gladius` (lower-level
`GLADIUSEstimator`) and `econirl.estimators.neural_gladius` (sklearn-style
`NeuralGLADIUS`). Networks use a scaling factor of $1/(1-\beta)$ by default
so the MLP output lives in a numerically stable range even when $\beta$ is close
to one. The learning rate decays as $\text{lr}_0 / (1 + \text{decay} \cdot
\text{step})$ following the paper's training schedule.
