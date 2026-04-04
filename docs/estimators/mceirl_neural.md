# MCE-IRL (Neural)

| Category | Citation | Reward | Transitions | SEs | Scales |
|----------|----------|--------|-------------|-----|--------|
| Inverse (Neural) | Wulfmeier et al. (2015) | Neural | Yes | Projected | Yes (neural) |

## Background

The linear MCE-IRL estimator restricts the reward to be a weighted combination of hand-crafted features. When the true reward is nonlinear in the state, this restriction introduces specification error that no amount of data can fix. Wulfmeier et al. (2015) replaced the linear reward with a neural network, giving the model the capacity to learn arbitrary reward functions directly from data. The training objective remains the same MCE-IRL feature matching criterion from Ziebart (2010), but the gradient now flows through the reward network via backpropagation.

The training loop computes a full reward matrix R(s,a) from the network at each epoch, solves the soft Bellman equation to get the induced policy, computes state visitation frequencies via a forward pass, and then updates the network to close the gap between the policy's state-action occupancy and the empirical occupancy from demonstrations. The surrogate loss multiplies the reward matrix by the occupancy residual, which produces the correct MCE-IRL gradient when differentiated through the network. After training, the neural reward can be projected onto linear features via ordinary least squares to extract interpretable structural parameters, though the standard errors from this projection are approximate.

## Key Equations

$$
\nabla_\phi \mathcal{L} = \sum_{s,a} \left( D_\pi(s)\,\pi(a \mid s) - \hat{D}(s,a) \right) \nabla_\phi R_\phi(s,a),
$$

where $D_\pi$ is the state visitation frequency under the current policy and $\hat{D}$ is the empirical state-action occupancy from demonstrations. The network parameters $\phi$ are updated via Adam with gradient clipping.

## Pseudocode

```
MCEIRLNeural(D, transitions, beta, max_epochs):
  1. Compute empirical state-action occupancy D_hat from D
  2. Initialize reward network R_phi
  3. For each epoch:
     a. Compute R(s,a) for all (s,a) from the network
     b. Solve soft Bellman to get V and pi
     c. Compute state visitation D_pi via forward pass
     d. Policy occupancy: occ(s,a) = D_pi(s) * pi(a|s)
     e. Residual: grad_R = occ - D_hat
     f. Surrogate loss: L = sum(R * grad_R)
     g. Backpropagate through R_phi, update via Adam
  4. Extract final policy and reward
  5. Optionally project R onto features: theta = argmin ||Phi*theta - R||^2
  6. Return R_phi, pi, projected theta
```

## Strengths and Limitations

Neural MCE-IRL can capture complex nonlinear reward structure that linear models miss. It inherits the sound statistical foundation of the MCE-IRL objective while gaining the representational power of neural networks. The state-action reward variant R(s,a) correctly handles environments where different actions have different intrinsic costs. The feature projection step provides a bridge between the neural reward and interpretable structural parameters, with an R-squared diagnostic that indicates how well the linear approximation fits.

The main cost is the loss of analytical standard errors. The projected standard errors are pseudo-SEs from a least-squares regression of the neural reward onto features, not the true sampling distribution of the structural parameters. Training requires known transition dynamics for the soft Bellman solve, and the neural network introduces hyperparameters for architecture, learning rate, and early stopping. Overfitting is a risk when the state space is small relative to the network capacity. For problems where the reward truly is linear in known features, the standard MCE-IRL estimator is preferable because it provides exact maximum likelihood inference.

## References

- Wulfmeier, M., Ondruska, P., and Posner, I. (2015). Maximum Entropy Deep Inverse Reinforcement Learning. *arXiv:1507.04888*.
- Ziebart, B. D. (2010). Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy. PhD thesis, Carnegie Mellon University.
