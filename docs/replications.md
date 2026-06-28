# Replications

This page reports numerical replication evidence. A study counts as a paper
replication only when the published design, sample, and estimand are available and
the reported quantities are directly comparable. Each section sets the package
value against the paper's published value, side by side, and ends with a command
to reproduce the run.

## Rust (1987), Table IX, Group 4 (NFXP)

The nested fixed point recovers the bus-engine maintenance and replacement costs
from the original STORDAT Group-4 panel, by full-likelihood BHHH. The package
matches the published Table IX numbers to four or more significant figures, on both
the point estimates and the standard errors.

### beta = 0.9999 (forward-looking)

| Quantity | Package | Paper |
| --- | ---: | ---: |
| theta_1 (maintenance) | 2.2931 | 2.2930 |
| theta_1 standard error | 0.6388 | 0.639 |
| RC (replacement) | 10.0750 | 10.0750 |
| RC standard error | 1.5816 | 1.582 |
| p0 transition | 0.3919 | 0.3919 |
| p1 transition | 0.5953 | 0.5953 |
| full log-likelihood | -3304.1548 | -3304.155 |

### beta = 0 (myopic)

| Quantity | Package | Paper |
| --- | ---: | ---: |
| theta_1 (maintenance) | 71.5134 | 71.5133 |
| theta_1 standard error | 13.7785 | 13.778 |
| RC (replacement) | 7.6358 | 7.6358 |
| RC standard error | 0.7197 | 0.7197 |
| p0 transition | 0.3919 | 0.3919 |
| p1 transition | 0.5953 | 0.5953 |
| full log-likelihood | -3306.0291 | -3306.028 |

The transition probabilities and their log-likelihood are estimated separately
from the cost parameters, so they are the same across both rows.

Reproduce, on the official NFXP data:

```bash
make rust-table-ix                            # fetch the official data, print Table IX
pytest tests/test_rust_tables.py -k TableIX   # lock the published numbers
```

## MPEC (Su and Judd, 2012)

Su and Judd prove that their constrained-optimization estimator and the nested
fixed point solve the same maximum-likelihood problem and return the same
estimates (their Proposition 1). The paper's own evidence is a Monte Carlo on the
simulated bus model, with no real-data estimate table. The replication target is
the equivalence: on the bus-engine data, MPEC recovers the NFXP estimates, and so
the published Rust Table IX numbers.

### STORDAT Group-4 panel, beta = 0.9999

| Quantity | MPEC | NFXP | Paper |
| --- | ---: | ---: | ---: |
| theta_1 (maintenance) | 2.2931 | 2.2931 | 2.2930 |
| theta_1 standard error | 0.6383 | 0.6388 | 0.639 |
| RC (replacement) | 10.0750 | 10.0750 | 10.0750 |
| RC standard error | 1.5815 | 1.5816 | 1.582 |
| choice log-likelihood | -163.5843 | -163.5843 | -163.584 |

The point estimates agree to four or more figures, and the standard errors match
to the precision Rust reports.

### Bundled bus panel, a second discretization

The packaged bus panel bins the same GMC records differently (4329 rows,
transition split 0.3938 / 0.5933 against the STORDAT 0.3919 / 0.5953), which moves
the cost level away from the published table. MPEC still tracks NFXP, which is the
content of Proposition 1.

| beta | Quantity | MPEC | NFXP |
| --- | --- | ---: | ---: |
| 0.9999 | theta_1 | 2.2638 | 2.2636 |
| 0.9999 | RC | 10.1430 | 10.1423 |
| 0.9999 | choice log-likelihood | -163.7111 | -163.7111 |
| 0.975 | theta_1 | 3.7768 | 3.7764 |
| 0.975 | RC | 9.0533 | 9.0529 |
| 0.975 | choice log-likelihood | -164.0198 | -164.0198 |

The cost level differs from the published table because this panel uses a
different binning. The agreement between MPEC and NFXP does not.

Reproduce:

```bash
pytest tests/test_mpec.py::TestMPECvsNFXP      # MPEC matches NFXP (Proposition 1)
# exact Table IX match, after make rust-table-ix has downloaded the data:
pytest tests/test_rust_tables.py::TestMPECStordatProfile
```

## MCE-IRL (Ziebart et al., 2008 and 2010)

Ziebart's reward is a function of state, R(s) linear in state features, on a
deterministic MDP. The paper models Pittsburgh as a deterministic road network
and scores road segments by their features. The published route-choice table
needs the original taxi trajectories, which are not public, so this is a
controlled gridworld recovery, not a paper-number match.

The test is a deterministic 12x12 gridworld. An agent walks from a corner to an
absorbing goal under a state reward over two features, distance to the goal and
distance to the center. Two estimators recover that reward from demonstrations:
maximum causal entropy (MCE-IRL) and its neural-reward form (Neural MCE-IRL, a
multi-layer network over state coordinates).

| Estimator | Reward recovery R² | Policy KL |
| --- | ---: | ---: |
| MCE-IRL (linear) | 1.000 | 0.000 |
| Neural MCE-IRL | 0.988 | 0.014 |

Mean over three seeds. Reward recovery is the R² of the recovered reward against
the true reward across states, where 1.0 is exact. Only the shape of the reward
across states is identified, not its level. Both estimators recover the reward,
the linear one exactly and the neural one nearly so. The neural reward map does
not cost identification on Ziebart's state-reward problem. This is simulation
evidence, not a replication of a published number.

Reproduce:

```bash
python examples/ziebart-mce-irl/run_gridworld.py --grid-size 12
```

## CCP / NPL (Hotz-Miller, 1993 and Aguirregabiria-Mira, 2002)

Aguirregabiria and Mira show that iterating the conditional-choice-probability
estimator, nested pseudo-likelihood (NPL), converges to the nested fixed point
estimates. Their Section 5.2 finding is that the one-step Hotz-Miller estimator
is poor, the gains from extra policy iterations come fast, and the K-step NPL
settles to a fixed point. The replication target is this convergence on the
bus-engine data, since the estimator is equivalent to NFXP in theory rather than
a separate published table.

### Bundled bus panel, Group 4, beta = 0.9999

| Estimator | theta_1 | RC | choice log-likelihood |
| --- | ---: | ---: | ---: |
| NFXP (MLE) | 2.2636 | 10.1423 | -163.7111 |
| Hotz-Miller (K = 1) | 1.2872 | 10.9207 | -168.1879 |
| NPL (K = 5) | 2.2651 | 10.1462 | -163.7113 |
| NPL (K = 20) | 2.2651 | 10.1462 | -163.7113 |

The one-step estimator sits well below the MLE. NPL reaches its fixed point by
the fifth iteration: K = 5 and K = 20 return the same estimates. The fixed point
lands within 0.0002 log-likelihood of the NFXP MLE.

NPL does not attain the MLE here. Its replacement cost differs from NFXP at the
fourth figure, 10.1462 against 10.1423, and its log-likelihood is marginally
lower. The stronger Aguirregabiria-Mira claim, that the NPL and nested fixed
point estimates agree to the twelfth digit, does not reproduce in this
implementation. NFXP remains the exact Rust replication. This is a convergence
reproduction, not a four-figure number match.

Reproduce:

```bash
pytest tests/test_rust_tables.py::TestNPLConvergenceAM2002
```

## AIRL (Fu, Luo, and Levine, 2018)

AIRL learns a reward through an adversarial discriminator. Fu, Luo, and Levine
prove (their Theorems 5.1 and 5.2) that the reward is identified and portable to
new dynamics only when it is a function of state, R(s). A state-action reward
recovers a shaped advantage that re-optimizes correctly in the training dynamics
but not under a changed transition model. Their Section 7.1 task is a 16-state,
4-action MDP with a reward at a single state.

The package reproduces the identification structure:

| Form | Reward | Transitions | Recovers the reward |
| --- | --- | --- | --- |
| AIRL-1 | R(s) | deterministic | yes |
| AIRL-2 (default) | R(s,a) | any | no, a shaped advantage |
| AIRL-2 anchored | R(s,a) with an action anchor | any | yes (see AIRL-Het) |

State-only AIRL recovers the reward on the deterministic 16-state task: normalized
reward error 0.10, policy distance 0.006, counterfactual regret near 0.004. The
action-dependent reward with no anchor does not recover, normalized reward error
1.16 and large counterfactual regret. On the Section 7.1 transfer test, the
state-only reward re-optimizes to optimal behavior under a fresh transition
matrix, while the state-action reward barely beats a random policy.

This is simulation evidence of the paper's identification claims. Section 7.1
reports reward maps and a transfer curve, not a numerical table.

Reproduce:

```bash
python validation/estimators/airl/run.py    # state-only recovers, state-action does not
```

## TD-CCP (Adusumilli and Eckardt, 2025)

Adusumilli and Eckardt estimate dynamic discrete choice models with
temporal-difference learning built on the conditional-choice-probability
approach. Their linear semi-gradient estimator approximates the recursive value
terms with basis functions and needs no transition densities. Their bus-engine
Monte Carlo (Online Appendix, Table B.1) is a Rust-style replacement problem with
one mileage state and a permanent bus type. The manager keeps or replaces each
period under Type-1 extreme-value shocks. The replacement payoff is set to zero,
and the keep payoff is theta0 + theta1 times mileage + theta2 times type. The true
values are theta0 = 2, theta1 = -0.15, theta2 = 1, with discount 0.9.

The package runs the same linear semi-gradient estimator, with a third-order
polynomial basis and logit conditional choice probabilities, on 1000 buses
observed for 30 periods, repeated across Monte Carlo draws. Each parameter's mean
estimate and mean-squared error sit next to the paper.

### Bus-engine recovery, 1000 buses, T = 30, 250 draws

| Parameter | True | Package mean | Paper mean | Package MSE | Paper MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| theta0 (intercept) | 2.0 | 2.043 | 1.979 | 0.0048 | 0.0080 |
| theta1 (mileage) | -0.15 | -0.156 | -0.149 | 0.00009 | 0.00001 |
| theta2 (type) | 1.0 | 0.964 | 1.004 | 0.0067 | 0.0034 |

The estimator recovers all three parameters with small bias, near the paper's
means. The locally robust correction returns nearly the same estimates as the
plain version, which matches the paper's reading that the correction adds little
in this setting. The mileage coefficient is recovered less precisely than the
paper's own code. This is a Monte Carlo recovery reproduction, not a four-figure
number match.

Reproduce:

```bash
PYTHONPATH=src python validation/estimators/tdccp/bus_engine_mc.py --n-reps 250 --lr-reps 50
```

## RHIP (Barnes et al., 2024)

RHIP, Receding Horizon Inverse Planning, generalizes classic IRL through a
planning horizon H. The policy plans with a stochastic soft-Bellman rule for H
steps, then follows a cheap deterministic planner. The paper's Figure 5 finding on
real Google Maps routing is that an interior horizon (H = 10) gives the best route
accuracy, beating both the myopic endpoint and the full MaxEnt endpoint (H
infinite). The paper reads this as better behavioral specification: people plan
over a finite horizon and approximate beyond it.

The package reproduces the mechanism on a controlled graph. Demonstrations come
from a finite-lookahead planner with a known lookahead h. The reward and the shock
scale are held fixed, and only the planning horizon differs from the estimator.
RHIP is then fit across a sweep of horizons H, and the fit is the policy distance
to the demonstrations.

### Recovering the demonstrator's lookahead (25-node graph, 300 trajectories, 3 seeds)

| Demonstrator lookahead h | H = 0 (myopic) | H = h (interior) | H infinite (MaxEnt) | Best H |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.035 | 0.011 | 0.060 | 1 |
| 2 | 0.057 | 0.016 | 0.042 | 2 |
| 3 | 0.068 | 0.012 | 0.031 | 3 |

The numbers are policy distance to the demonstrations, lower is better. For every
demonstrator, the best-fitting horizon is interior and lands on the demonstrator's
lookahead. Both endpoints fit worse. As the demonstrator's lookahead changes, the
recovery-optimal horizon shifts with it, so the horizon is an identifiable
behavioral parameter. This reproduces the Figure 5 mechanism. It is a controlled
recovery reproduction, not a match of the paper's real-world routing numbers, which
need proprietary data.

Reproduce:

```bash
python scripts/study_rhip_lookahead.py
```

## Pending

These estimators have a paper target but no completed replication yet. Each is held
to the same bar: match the published numbers to four or more significant figures, on
both the estimates and the standard errors.

| Estimator | Paper | Status |
| --- | --- | --- |
| NNES | Nguyen (2025) | Not yet evaluated. |
| AIRL-Het | Lee-Sudhir-Wang (2026) | Not yet evaluated. |
| GLADIUS | Kang-Yoganarasimhan-Jain (2025) | Not yet evaluated. |
