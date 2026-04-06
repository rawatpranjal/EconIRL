# f-IRL

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse | Ni et al. (2022) | Tabular | Yes | No | No |

## What this estimator does

Every other IRL method asks you to choose something about the reward, whether features, a network architecture, or a discriminator design. Ni et al. (2022) assign a free reward parameter to every state-action pair and adjust these parameters to make the model's occupancy measure match the expert's as closely as possible, measured by an f-divergence. The analyst chooses the divergence. KL gives a maximum-likelihood flavor, total variation provides robustness to outliers, and chi-squared gives sensitivity to distribution differences. The approach is completely nonparametric and eliminates feature engineering and discriminator instability.

## How it works

The estimator minimizes

$$
\min_r D_f(\rho_E \| \rho_{\pi_r}),
$$

where $\rho(s,a)$ is the discounted occupancy measure and $D_f$ is the chosen f-divergence. Each iteration solves for the optimal policy under the current tabular reward, computes the resulting occupancy measure, and updates the reward via the divergence-specific gradient. The gradient depends on the ratio of expert to model occupancy and takes a different functional form for each divergence choice.

## When to use it

f-IRL is the right choice for exploratory analysis when you have no idea which features matter and want to see what the raw reward landscape looks like before committing to a parametric form. The limitation is that f-IRL is confined to purely tabular state-action spaces and requires dense expert coverage in every cell. It produces no standard errors and no interpretable parameters. For parametric estimation with inference, use MCE-IRL or NFXP.

## References

- Ni, T., Sikchi, H., Wang, Y., Gupta, T., Lee, L., and Eysenbach, B. (2022). f-IRL: Inverse Reinforcement Learning via State Marginal Matching. *CoRL 2022*.

A primer for this estimator is not yet available.
