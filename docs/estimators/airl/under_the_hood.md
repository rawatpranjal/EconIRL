# Under the Hood

## Model

The data are state, action, next-state triples $(s, a, s')$ from a stationary
infinite-horizon dynamic discrete choice model with discount factor $\beta$,
transition kernels $F_a(s' \mid s)$, and i.i.d. logit taste shocks. The agent's
value function solves the soft Bellman equation (without an Euler-gamma
constant):

$$
V(s) = \log \sum_{a} \exp\!\Bigl(u(s, a) + \beta \sum_{s'} F_a(s' \mid s)\, V(s')\Bigr).
$$

The package uses the $(A, S, S)$ transition tensor convention: `transitions[a, s, s']`
is $F_a(s' \mid s)$ and must be row-stochastic over $s'$.

## Discriminator Decomposition

AIRL trains a discriminator whose logit is constrained to the form

$$
f_{\theta,\phi}(s, a, s') = g_\theta(s) + \beta\, h_\phi(s') - h_\phi(s),
$$

where $g_\theta$ is the reward candidate (state-only in the original
identification setting) and $h_\phi$ is a learned shaping potential. The
discriminator probability is

$$
D(s, a, s') = \frac{\exp(f_{\theta,\phi}(s, a, s'))}{\exp(f_{\theta,\phi}(s, a, s')) + \pi(a \mid s)}.
$$

## Training Objective

The discriminator is trained to separate expert transitions from
policy-generated ones. The adversarial objective is

$$
\max_{\theta,\phi}\; \mathbb{E}_{(s,a,s') \sim \tau^*}\!\bigl[\log D(s,a,s')\bigr]
+ \mathbb{E}_{(s,a,s') \sim \pi}\!\bigl[\log\bigl(1 - D(s,a,s')\bigr)\bigr],
$$

which is binary cross-entropy over the expert-vs-policy classification.
Equivalently, the discriminator is trained on the logit

$$
\ell(s,a,s') = f_{\theta,\phi}(s,a,s') - \log \pi(a \mid s),
$$

and the objective maximizes $\log \sigma(\ell)$ on expert transitions and
$\log(1 - \sigma(\ell))$ on policy transitions, where $\sigma$ is the logistic
function.

## Policy Update

After each discriminator step, the policy is re-solved using the shaped reward
as the flow utility. With `generator_reward="f"` (the default), the generator
reward is the expected shaped score:

$$
r_{\text{shaped}}(s, a) = g_\theta(s) + \beta \sum_{s'} F_a(s' \mid s)\, h_\phi(s') - h_\phi(s).
$$

Value iteration then solves the soft Bellman equation with $r_{\text{shaped}}$
as the flow utility, producing a new policy $\pi$ for the next discriminator
step.

## Identification

At the adversarial optimum, $D = 1/2$ everywhere, which forces
$f_{\theta,\phi}(s,a,s') = \log \pi(a \mid s)$. Under state-only $g_\theta$ and
the decomposability condition of Fu et al. (2018), this pins $g_\theta$ to the
true reward up to a constant. The shaping potential $h_\phi$ absorbs the
dynamics-dependent terms that would otherwise contaminate $g_\theta$.

When $g_\theta$ is state-action, the shaping structure can no longer disentangle
reward from the action-contrast direction; the disentanglement guarantee does
not hold. The action-dependent diagnostic cell confirms this failure in the
package's simulation results.

## Pseudocode

```
initialize reward parameters theta and shaping potential phi to zero
choose an initial policy pi (e.g. uniform)
while the discriminator loss has not converged:
    sample expert transitions (s, a, s') from the demonstration panel
    sample policy transitions (s, a, s') by rolling out pi
    compute discriminator logit f(s,a,s') = g_theta(s) + beta*h_phi(s') - h_phi(s)
    update theta and phi via binary cross-entropy on expert vs. policy transitions
    re-solve the soft Bellman equation using the shaped reward as flow utility
    update pi from the new value function
extract g_theta as the recovered reward
project g_theta onto the reward feature basis if structural parameters are needed
report g_theta, pi, value function, and diagnostics
```

## Implementation Notes

The implementation lives in `econirl.estimation.adversarial.airl`. Reward
parameters are updated with Adam at each round; the shaping potential is
initialized to zero and updated alongside the reward. The `shaping_l2_penalty`
parameter applies a small L2 regularizer to both reward and shaping parameters.
This suppresses drift along directions that leave behavior unchanged during
training. State-only mode projects the reward
matrix onto the state subspace by averaging across actions before computing the
discriminator logit.
