# Pseudocode

### Notation

| Symbol | Definition |
|--------|-----------|
| $\mathcal{S}, \mathcal{A}$ | State space, action space |
| $\theta$ | Structural / reward parameters |
| $\beta$ | Discount factor |
| $\phi(s,a)$ | Feature vector |
| $u(s,a) = \phi(s,a)^\top\theta$ | Flow utility |
| $P(s' \mid s,a)$ | Transition probability |
| $V(s)$, $Q(s,a)$ | Value function, action-value function |
| $\pi(a \mid s)$ | Choice probability (policy) |
| $\mathcal{D}$ | Demonstrations: observed $(s_t, a_t)$ trajectories |
| $\bar\phi_\mathcal{D}$ | Empirical feature mean: $\frac{1}{T}\sum_t \phi(s_t, a_t)$ |
| $\mu(s)$ | State visitation frequency |

**Softmax Bellman operator** (used by all algorithms):

$$Q(s,a) = u(s,a) + \beta \sum_{s'} P(s' \mid s,a)\, V(s'), \qquad V(s) = \log\sum_a e^{Q(s,a)}, \qquad \pi(a \mid s) = e^{Q(s,a) - V(s)}$$

---

### Structural Estimators

**NFXP** — Nested Fixed Point (Rust, 1987)

```
1. Outer loop: optimize theta via MLE
2.   Inner loop: solve V = T(V; theta) to fixed point via contraction mapping
3.   Policy: pi(a|s) = exp(Q(s,a) - V(s))
4.   Log-likelihood: L(theta) = sum_{(s,a) in D} log pi(a|s)
5.   Update theta via BFGS
6. Return theta_MLE
```

**CCP** — Conditional Choice Probability / NPL (Hotz & Miller, 1993)

```
1. Estimate CCPs from data: pi_hat(a|s) = count(s,a) / count(s)
2. Social surplus: e(a|s) = euler_constant - log(pi_hat(a|s))
3. Choice-weighted transitions: P^pi(s'|s) = sum_a pi_hat(a|s) P(s'|s,a)
4. Invert: V_bar = (I - beta*P^pi)^{-1} sum_a pi_hat(a|s)[u(s,a;theta) + e(a|s)]
5. Q(s,a) = u(s,a;theta) + beta * P(.|s,a)' * V_bar
6. Optimize theta via MLE on log-softmax(Q)
7. NPL: re-estimate pi_hat from theta, repeat 2-6 until convergence
```

**SEES** — Sieve Estimator (Luo & Sang, 2024)

```
1. Choose sieve basis {b_1(s), ..., b_J(s)}  (e.g., Chebyshev polynomials)
2. Approximate: V(s; alpha) = sum_j alpha_j * b_j(s)
3. Q(s,a) = u(s,a;theta) + beta * P * V(.; alpha)
4. Penalized MLE: max_{theta, alpha}  L(theta, alpha) - lambda * ||V - T(V;theta)||^2
5. Optimize (theta, alpha) jointly via L-BFGS
```

**NNES** — Neural Network Estimator (Nguyen, 2025)

```
1. Parameterize: V_w(s) = NeuralNet_w(s)
2. Q(s,a) = u(s,a;theta) + beta * sum_s' P(s'|s,a) V_w(s')
3. Penalized MLE: max_{theta, w}  L(theta, w) - lambda * ||V_w - T(V_w;theta)||^2
4. Optimize (theta, w) jointly via Adam
```

---

### Entropy-Based IRL

**MCE IRL** — Maximum Causal Entropy (Ziebart, 2010)

```
1. Initialize theta
2. Repeat:
   a. Backward: soft value iteration V, Q from r(s,a) = phi(s,a)'*theta
   b. Policy: pi(a|s) = exp(Q(s,a) - V(s))
   c. Forward: state visitation mu(s) via propagation under pi
   d. Expected features: phi_bar = sum_s mu(s) sum_a pi(a|s) phi(s,a)
   e. Gradient: dL/dtheta = phi_D - phi_bar
   f. Update theta via Adam
3. Return theta
```

**MaxEnt IRL** — Maximum Entropy (Ziebart et al., 2008)

```
1. Initialize theta
2. Repeat:
   a. Reward: r(s,a) = phi(s,a)' * theta
   b. Soft value iteration -> pi(a|s)
   c. State visitation mu(s) via forward propagation under pi
   d. Gradient: dL/dtheta = phi_D - sum_s mu(s) sum_a pi(a|s) phi(s,a)
   e. Update theta
3. Return theta (reward weights)
```

**Deep MaxEnt** — Deep Maximum Entropy (Wulfmeier et al., 2016)

```
1. Initialize neural network f_w
2. Repeat:
   a. Reward: r(s,a) = f_w(phi(s,a))
   b. Soft value iteration -> pi(a|s)
   c. State visitation: mu_D (empirical), mu_pi (under pi)
   d. Reward gradient: dL/dr = mu_D - mu_pi
   e. Backprop: dL/dw = (dL/dr) * (dr/dw)
   f. Update w via Adam
3. Return f_w (learned reward network)
```

**BIRL** — Bayesian IRL (Ramachandran & Amir, 2007)

```
1. Initialize reward R, prior P(R), step size delta
2. For m = 1..M (MCMC):
   a. Propose R' by perturbing one component of R by +/- delta
   b. Solve MDP under R' -> Q*(s,a; R')
   c. Likelihood: P(D|R') = prod_{(s,a)} exp(alpha*Q*(s,a;R')) / Z(s)
   d. Accept R' with prob min(1, P(D|R')P(R') / P(D|R)P(R))
3. Return posterior mean: R_hat = (1/M) sum_m R^(m)
```

---

### Margin-Based IRL

**Max Margin** — Structured Max-Margin (Ratliff et al., 2006)

```
1. For each (s, a*) in D, for each alternative a != a*:
     Constraint: theta'*phi(s,a*) >= theta'*phi(s,a) + loss(a,a*) - xi
2. QP: min ||theta||^2 + C * sum(xi)  s.t. margin constraints, xi >= 0
3. Return theta
```

**Max Margin IRL** — Apprenticeship Learning (Abbeel & Ng, 2004)

```
1. Expert feature expectations: mu_E = E_D[sum_t beta^t phi(s_t,a_t)]
2. Initialize random policy pi_0, compute mu_0
3. Repeat until ||mu_E - closest||_2 < epsilon:
   a. theta = mu_E - argmin_{mu in conv(mu_0,...,mu_i)} ||mu_E - mu||
   b. Solve MDP under theta -> pi_{i+1}
   c. mu_{i+1} = E_{pi_{i+1}}[sum_t beta^t phi(s_t,a_t)]
4. Return theta (reward direction)
```

---

### Distribution Matching

**f-IRL** — f-Divergence IRL (Ni et al., 2022)

```
1. Initialize reward network r_theta(s), discriminator D_w(s)
2. Repeat:
   a. Train D_w to classify expert vs policy states
   b. Density ratio: rho_E(s)/rho_theta(s) = D_w(s) / (1 - D_w(s))
   c. h_f(ratio) for chosen f-divergence (KL, chi^2, TV)
   d. Gradient: dL/dtheta = (1/T) cov(sum_t h_f(ratio_t), sum_t dr_theta/dtheta)
   e. Update theta, recompute policy via soft value iteration
3. Return r_theta
```

---

### Neural Estimators

**TD-CCP** — TD-Learning + CCP (Adusumilli & Eckardt, 2022)

```
1. Train V_w(s) = NeuralNet_w(s) via temporal difference on D:
     L_TD = sum_{(s,a,s')} (V_w(s) - [r(s,a) + beta*V_w(s')])^2
2. Q(s,a) = phi(s,a)'*theta + beta * sum_s' P(s'|s,a) V_w(s')
3. MLE: theta = argmax sum_{(s,a) in D} log softmax(Q(s,.))_a
```

**GLADIUS** — Dual Network IRL (Kang et al., 2025)

```
1. Initialize Q-network Q_w(s,a), EV-network EV_psi(s,a)
2. Joint loss:
     L = -sum_{(s,a)} log softmax(Q_w(s,.))_a                        (NLL)
       + lambda * sum_{(s,a,s')} ||EV_psi(s,a) - beta*V(s')||^2      (Bellman)
   where V(s) = log sum_a exp(Q_w(s,a))
3. Train via mini-batch SGD
4. Rewards: r(s,a) = Q_w(s,a) - beta * EV_psi(s,a)
5. Structural params: theta = (Phi'Phi)^{-1} Phi' r
```

---

### Adversarial Methods

**GAIL** — Generative Adversarial Imitation (Ho & Ermon, 2016)

```
1. Initialize policy pi_w, discriminator D_psi
2. Repeat:
   a. Sample trajectories from pi_w
   b. D_psi: max E_D[log D_psi(s,a)] + E_{pi_w}[log(1 - D_psi(s,a))]
   c. Reward: r(s,a) = -log(1 - D_psi(s,a))
   d. Update pi_w via REINFORCE with reward r
3. Return pi_w (policy only — no transferable reward)
```

**AIRL** — Adversarial Inverse RL (Fu et al., 2018)

```
1. Initialize policy pi_w, reward g_psi(s,a), shaping h_xi(s)
2. Structured discriminator:
     f(s,a,s') = g_psi(s,a) + beta*h_xi(s') - h_xi(s)
     D(s,a,s') = sigmoid(f(s,a,s') - log pi_w(a|s))
3. Repeat:
   a. Update (psi, xi): max E_D[log D] + E_{pi_w}[log(1-D)]
   b. Update pi_w via REINFORCE with reward f
4. Return g_psi (disentangled reward function)
```

**GCL** — Guided Cost Learning (Finn et al., 2016)

```
1. Initialize cost network c_psi(tau), policy pi_w
2. Repeat:
   a. Sample demo tau_D, policy trajectories tau_pi
   b. Partition function via importance sampling:
        Z ≈ (1/M) sum_j exp(-c_psi(tau_j)) / q(tau_j)
   c. Update cost: min_psi E_D[c_psi(tau)] + log Z
   d. Update pi_w to minimize c_psi via REINFORCE
3. Return c_psi (negative reward)
```

---

### Baseline

**BC** — Behavioral Cloning

```
1. pi(a|s) = count(s, a in D) / count(s in D)
2. Return pi (frequency estimate — no reward recovered)
```
