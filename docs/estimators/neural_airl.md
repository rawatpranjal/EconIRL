# NeuralAIRL

## Important Links

- [Core Estimators](core.md)
- [Tabular AIRL](airl.md)
- [Choosing and Comparing Estimators](../comparing_estimators.md)
- [Applied Notebook](https://github.com/rawatpranjal/EconIRL/blob/main/examples/neural-airl/neural_airl_applied_workflow.ipynb)

`NeuralAIRL` estimates a nonlinear state reward from demonstration
trajectories. It is a separate estimator from the tabular `AIRL` class. The
reward, shaping potential, and generator policy use neural function
approximators.

The reward depends on state alone. Observed context and latent heterogeneity are
outside this estimator.

## Source Papers

The estimator follows the structured discriminator in {ref}`Fu, Luo, and
Levine (2018) <fu-2018>`. The finite-state implementation uses exact soft
planning to form the policy target. A policy network learns that target during
the adversarial updates.

## Model

Let $x_s$ be the input vector for state $s$. The discriminator score is

$$
f_{\theta,\phi}(s,a,s') =
g_\theta(x_s) + \beta h_\phi(x_{s'}) - h_\phi(x_s).
$$

The discriminator is

$$
D(s,a,s') =
\frac{\exp f_{\theta,\phi}(s,a,s')}
{\exp f_{\theta,\phi}(s,a,s') + \pi_\psi(a\mid x_s)}.
$$

Here $g_\theta$ is the state reward, $h_\phi$ is the shaping potential, and
$\pi_\psi$ is the generator policy network. The reported policy is re-solved
from the learned state reward and supplied transition tensor.

## Algorithm

```text
Validate the panel, finite state inputs, and transition tensor
Initialize the reward, shaping, and generator policy networks
Repeat until the adversarial stopping rule is met or maximum rounds are reached
    Simulate transitions from the generator policy
    Update the reward and shaping networks with the discriminator loss
    Solve the shaped finite-state reward by soft dynamic programming
    Fit the generator policy network to that policy target
Profile one reward scale against the demonstration choice likelihood
Center the scaled state reward and re-solve the reported policy
Run the fitted-model self-check
```

## Public boundary

The estimator requires all of the following.

- Complete state coverage in the demonstration panel.
- A full-rank state-input matrix.
- A transition tensor with orientation `(n_actions, n_states, n_states)`.
- State-only reward inputs.
- No `context=` argument.

`reward_` is centered over states. Its level is not identified. Neural weights
are not structural coefficients. Recovery and uncertainty therefore target the
normalized reward surface, induced policy, and supported counterfactuals.

## Minimal check

```python
from econirl import AIRL, NeuralAIRL

print(NeuralAIRL.__name__)
print(NeuralAIRL is AIRL)
```

**Result**

```text
NeuralAIRL
False
```

## Fitted objects

| Object | Meaning |
| --- | --- |
| `reward_` | Centered nonlinear state reward. |
| `reward_matrix_` | The state reward repeated over actions. |
| `policy_` | Soft-optimal policy under the fitted reward. |
| `policy_network_` | Generator policy used at the end of adversarial training. |
| `value_` | Soft value function under the fitted reward. |
| `shaping_` | Learned potential values over states. |
| `diagnostics_` | Data, identification, and optimization checks. |
| `bootstrap_` | Trajectory-bootstrap reward and policy functionals, when requested. |

Use `predict_reward()` and `predict_proba()` for fitted states. Use
`counterfactual(transitions=...)` to re-solve the nonlinear reward under new
dynamics. Use `reward_delta=` for a state-reward intervention.

## Limitations

This finite-state implementation does not replicate the deep continuous-control
experiments in Fu et al. (2018). It does not estimate
heterogeneous rewards. It does not attach structural meaning to network
weights. Transfer claims still require the state-only and decomposability
conditions from AIRL.
