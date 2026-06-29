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
estimates (their Lemma 2 and footnote 15). The one-step Hotz-Miller estimator is
poor, the gains from extra policy iterations come fast, and NPL run to its fixed
point reaches the maximum likelihood estimate. The replication target is this
equivalence on the bus-engine data.

### Bundled bus panel, Group 4, beta = 0.9999

| Estimator | theta_1 | RC | choice log-likelihood |
| --- | ---: | ---: | ---: |
| NFXP (MLE) | 2.2636 | 10.1423 | -163.7111 |
| Hotz-Miller (K = 1) | 1.2872 | 10.9207 | -168.1879 |
| NPL (run to convergence) | 2.2640 | 10.1432 | -163.7113 |

The one-step estimator sits well below the MLE. NPL run to its fixed point reaches
it, the operating cost and replacement cost match NFXP to the third and fourth
figure. At this discount factor the choice likelihood is nearly flat in the
replacement cost, so NPL and NFXP both sit about 0.0005 from the exact maximizer
and agree to the fourth figure rather than the twelfth the paper reports on its
own machine. At lower discount factors, where the replacement cost is better
identified, NPL matches the maximum likelihood estimate to five figures. An
earlier build stopped the policy-iteration loop too early and missed the MLE at
the fourth figure; running NPL to its fixed point closes that gap.

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
temporal-difference learning on the conditional-choice-probability representation.
Their linear semi-gradient estimator approximates the recursive value terms h(a, x)
and g(a, x) with basis functions and needs no transition densities. Their bus-engine
Monte Carlo (Online Appendix, Table B.1) is a Rust-style replacement problem with
one mileage state and a permanent bus type s in {1, 2}. The manager keeps or replaces
each period under Type-1 extreme-value shocks. The replacement payoff is set to zero,
and the keep payoff is theta0 + theta1 times mileage + theta2 times type. The true
values are theta0 = 2, theta1 = -0.15, theta2 = 1, with discount 0.9.

The package runs the same linear semi-gradient estimator on the same design, 1000
buses over 30 periods, across 200 Monte Carlo draws, without the locally robust
correction (columns 2 and 3 of Table B.1). Each parameter's mean, standard
deviation, and mean-squared error sit next to the paper.

### Bus-engine recovery, 1000 buses, T = 30, 200 draws

| Parameter | True | Package mean | Paper mean | Package SD | Paper SD | Package MSE | Paper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| theta0 (intercept) | 2.0 | 1.9533 | 1.9788 | 0.0904 | 0.0868 | 0.0103 | 0.0080 |
| theta1 (mileage) | -0.15 | -0.1468 | -0.1492 | 0.0035 | 0.0033 | 0.00002 | 0.00001 |
| theta2 (type) | 1.0 | 1.0029 | 1.0044 | 0.0628 | 0.0583 | 0.0039 | 0.0034 |

The means match the paper to two or three figures, and the standard deviations
match to within a few percent. The linear semi-gradient reaches the paper's
near-maximum-likelihood precision. The mean-squared errors sit within Monte Carlo
noise of the paper, which reports 1000 draws. The nested fixed point reproduces the
same Table B.1 numbers and serves as the oracle reference.

Reproduce:

```bash
PYTHONPATH=src python validation/estimators/tdccp/bus_engine_mc.py --n-reps 200 --skip-lr
```

## NNES (Nguyen, 2025)

Nguyen proposes the neural-network efficient estimator (NNES) for structural
dynamic discrete choice. A neural network approximates the value function inside
the nested pseudo-likelihood, and a Neyman-orthogonality property makes the
structural score insensitive to first-order value-approximation error. The
central result is that NNES attains the semiparametric efficiency bound, it
matches the oracle nested fixed point in both mean and standard deviation (their
Theorem 4.3 and Table 1).

The test is the paper's two-module bus-engine renewal model. Each module has
mileage that grows by an exponential increment when the engine is kept and resets
on replacement, with Type-1 extreme-value shocks at discount 0.9. The keep cost is
c times mileage and the replacement cost is crep. The true values are crep = 2.0
and c = 0.05 for module one, crep = 2.5 and c = 0.08 for module two. Each
replication draws 50 buses over 20 periods. The package runs its NNES and the
nested fixed point on the same panels, across 100 draws.

### Two-module renewal, 50 buses, T = 20, 100 draws

| Module | Parameter | Package NFXP | Package NNES | Paper NFXP | Paper NNES |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | crep | 1.9996 (0.1503) | 1.9980 (0.1498) | 1.9454 (0.1746) | 1.9443 (0.1754) |
| 1 | c | 0.0499 (0.0058) | 0.0497 (0.0057) | 0.0509 (0.0103) | 0.0515 (0.0103) |
| 2 | crep | 2.4834 (0.1842) | 2.4790 (0.1831) | 2.5135 (0.1812) | 2.5823 (0.1903) |
| 2 | c | 0.0792 (0.0077) | 0.0788 (0.0077) | 0.0843 (0.0134) | 0.0872 (0.0141) |

Standard deviations are in parentheses. The package's NNES tracks the nested fixed
point to the third figure on every parameter, the same mean and the same standard
deviation. NNES attains the efficiency bound, which is the paper's result. The
absolute dispersion differs from the paper because this run discretizes the
continuous mileage to a grid, while the paper samples mileage continuously. The
reproduced quantity is the equality between NNES and the nested fixed point.

Reproduce:

```bash
PYTHONPATH=src python validation/estimators/nnes/bus_renewal_efficiency.py --n-reps 100
```

## GLADIUS (Kang, Yoganarasimhan, and Jain, 2025)

GLADIUS learns a reward by inverse Q-learning from offline choices. It fits a
neural Q-network and a continuation-value network, then reads the reward as
r(s, a) = Q(s, a) minus beta times the expected next value. An anchor action with
a known reward sets the absolute level (their Assumption 3).

The paper's first simulation is the Rust bus-engine problem on 20 mileage states.
Maintenance costs theta0 per mileage unit. Replacement costs theta1 and resets the
mileage to one. The true values are theta0 = 1 and theta1 = 5, with discount 0.95.
The reported quantity is the mean absolute percentage error of the recovered
reward, measured on the state-action pairs the expert visits (their Table 2). Most
of those samples sit at low mileage.

The package runs two estimators on this design. The nested fixed point is the
oracle, with the true linear form and known transitions. GLADIUS uses a two-layer
network over mileage, no transitions, and the replacement action as the anchor.

### Bus-engine reward MAPE (percent), 80/20 split, two draws per cell

| Trajectories (H=100) | NFXP oracle | Paper Rust | GLADIUS | Paper GLADIUS |
| --- | ---: | ---: | ---: | ---: |
| 50 | 3.91 | 3.62 | 7.0 | 3.44 |
| 250 | 0.97 | 1.37 | 4.2 | 0.84 |

The nested fixed point reproduces the paper's oracle column to the precision the
paper reports, with recovered costs near [1.0, 5.0]. This confirms the design and
the metric.

GLADIUS recovers the reward direction on the same data. The recovered maintenance
cost rises about one unit per mileage step, as it should. The level is the weak
point. The reward is a difference of two large value terms, so a small level error
in Q shows up in the low-mileage reward, where the metric puts most of its weight.
The package reward MAPE sits above the paper's GLADIUS column, and the gap grows
with the sample size. The fitted-Q anchor target (the package default) pins the
level far better than the literal bi-conjugate variant, which does not pin it.

This reproduces the paper's oracle recovery and the GLADIUS reward direction. It is
not yet a match to the paper's GLADIUS error. That would need the absolute level
pinned as tightly as the oracle.

Reproduce:

```bash
PYTHONPATH=src python validation/estimators/gladius/paper_table2_mape.py --traj 250 --reps 2
```

## Pending

These estimators have a paper target but no completed replication yet. Each is held
to the same bar: match the published numbers to four or more significant figures, on
both the estimates and the standard errors.

| Estimator | Paper | Status |
| --- | --- | --- |
| RHIP | Barnes et al. (2024) | Not yet evaluated. |
| AIRL-Het | Lee-Sudhir-Wang (2026) | Not yet evaluated. |
| UFXP | Oguz and Bray (2026) | Not yet evaluated. |
