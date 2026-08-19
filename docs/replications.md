# Replications

## Important Links

- [CCP estimator](estimators/ccp.md)
- [NFXP estimator](estimators/nfxp.md)
- [CCP validation](estimators/ccp/validation.md)
- [NFXP validation](estimators/nfxp/validation.md)

This page reports numerical replication evidence. A study counts as a paper
replication only when the published design, sample, and estimand are available and
the reported quantities are directly comparable. Each section sets the package
value against the paper's published value, side by side, and ends with a command
to reproduce the run.

Read this page as the paper-number ledger. If a method has simulation evidence
but no directly comparable published table, that distinction is stated rather
than hidden.

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

### STORDAT Group-4 panel, beta = 0.9999

On the official Rust panel, the converged NPL profile reaches the same fixed point
as NFXP. Joint full-likelihood BHHH then includes the structural parameters and
the free transition probabilities. The resulting estimates and standard errors
match Rust's published Table IX values.

| Quantity | CCP / NPL | NFXP | Paper |
| --- | ---: | ---: | ---: |
| theta_1 (maintenance) | 2.2931 | 2.2931 | 2.2930 |
| theta_1 standard error | 0.6388 | 0.6388 | 0.639 |
| RC (replacement) | 10.0750 | 10.0750 | 10.0750 |
| RC standard error | 1.5816 | 1.5816 | 1.582 |
| p0 standard error | 0.0075 | 0.0075 | 0.0075 |
| p1 standard error | 0.0075 | 0.0075 | 0.0075 |
| full log-likelihood | -3304.1548 | -3304.1548 | -3304.155 |

The package computes this joint covariance only after NPL reaches its fixed point
under the Rust residual transition model. Fixed-stage CCP fits report
standard errors conditional on the fitted transition model.

### Bundled bus panel, Group 4, beta = 0.9999

| Estimator | theta_1 | RC | choice log-likelihood |
| --- | ---: | ---: | ---: |
| NFXP (MLE) | 2.2636 | 10.1423 | -163.7111 |
| Hotz-Miller (K = 1) | 1.2872 | 10.9207 | -168.1879 |
| NPL (run to convergence) | 2.2640 | 10.1432 | -163.7113 |

The one-step estimator sits below the MLE. NPL run to its fixed point reaches the
MLE. Its operating and replacement costs match NFXP to the third and fourth
figures. At this discount factor, the choice likelihood is nearly flat in the
replacement cost. NPL and NFXP therefore sit about 0.0005 from the exact maximizer
and agree to the fourth figure. At lower discount factors, where the replacement
cost is better identified, NPL matches the maximum likelihood estimate to five
figures.

Run `make ccp-table-ix` to fetch the official data, reproduce the table, and
write the [JSON receipt](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/ccp_rust_table_ix.json).
Verify a saved receipt without refitting:

```bash
uv run python validation/estimators/ccp/rust_table_ix.py \
  --verify \
  --output validation/results/ccp_rust_table_ix.json
```

**Result**

```text
verified validation/results/ccp_rust_table_ix.json
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
| AIRL-2 anchored | R(s,a) with an action anchor | any | yes (see AIRL2) |

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

Adusumilli and Eckardt estimate finite reward parameters with temporal-
difference recursions. The parameter stage learns continuation terms from
successor tuples without fitting a transition density.

### Official Table E.1 comparison

The official Zenodo code deterministically generates 1,000 bus replacement
panels. Each panel contains 1,000 buses and 30 retained periods after a
1,000-period burn-in. EconIRL uses the paper's seed schedule, basis, logit first
stage, fold assignment, and initial parameter values. The table compares all
published means and empirical standard deviations for the plug-in and locally
robust estimators.

| Method | Parameter | Package mean | Published mean | Package SD | Published SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Plug-in | $\theta_0$ | 1.978596 | 1.978589 | 0.086882 | 0.086880 |
| Plug-in | $\theta_1$ | -0.149204 | -0.149203 | 0.003342 | 0.003342 |
| Plug-in | $\theta_2$ | 1.004454 | 1.004448 | 0.058316 | 0.058315 |
| Locally robust | $\theta_0$ | 1.977512 | 1.977513 | 0.087594 | 0.087594 |
| Locally robust | $\theta_1$ | -0.148897 | -0.148897 | 0.003387 | 0.003387 |
| Locally robust | $\theta_2$ | 1.003724 | 1.003724 | 0.058684 | 0.058684 |

All 12 quantities agree to four or more significant figures. The largest mean
gap is $6.9\times10^{-6}$. The largest standard-deviation gap is
$2.1\times10^{-6}$. This comparison reproduces the distribution of repeated
point estimates. It does not assess standard-error calibration. The optimizer
satisfied its stopping criteria for every locally robust fit. The plug-in
optimizer stopped short of every success criterion in 318 panels. Those finite
estimates remain in the summary and reproduce the published distribution.

See the [Table E.1 result](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/tdccp_table_e1.json)
and the [EconIRL replication runner](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/tdccp/paper_table_e1_mc.py).

### High-dimensional state check

The bus state is augmented with 20 irrelevant variables. Thirty paired panels
are fitted with zero and 20 nuisance variables. The mean parameter error ratio
is 1.006, so the added variables barely change recovery. Shuffling successor
links increases the dynamic coefficient error by a factor of 24.69. This shows
that the result depends on the temporal link rather than a static choice fit.

See the [high-dimensional result](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/tdccp_highdim.json)
and the [simulation program](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/tdccp/highdim_dummies.py).

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

GLADIUS estimates Q and conditional continuation value directly, then recovers
reward without using a transition tensor during fitting. The Table 2 driver
uses the paper's 20 mileage states, equiprobable maintenance increments of 1
through 4, replacement reset, costs `(1, 5)`, discount `0.95`, 100 periods, and
the paper's sample-level reward MAPE.

Qualification runs all six reported sample sizes with 20 seeds each. A cell
passes when its mean MAPE is no larger than the paper mean plus two reported
standard errors. The combined gate also requires no deterioration after the
`N=250` cell. Exact values are stored in
[`gladius_paper_table2.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius_paper_table2.json).

The checked-in author experiment selects the best epoch using true held-out
reward MAPE. The replication matches that simulation-only rule and labels it in
the receipt. This is leakage by ordinary deployment standards, so the public
`GLADIUS.fit` path never accepts oracle rewards or held-out truth.

Five of the six cells land inside the bound. `N=5000` does not, at 0.26 against
a 0.24 bound. GLADIUS is therefore a partial match to Table 2, not a completed
replication. The per-cell numbers and the reasons are on the
[GLADIUS validation page](estimators/gladius/validation.md).

Reproduce:

```bash
PYTHONPATH=src:. uv run python validation/estimators/gladius/paper_table2_mape.py \
    --sweep --reps 20 --max-epochs 800 \
    --out validation/results/gladius_paper_table2.json
```

**Result**

```text
GLADIUS Table 2 qualification gates failed
```

The sharded release procedure is in the
[GLADIUS qualification runbook](estimators/gladius/qualification_runbook.md).

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

## AIRL2 (Lee, Sudhir, and Wang, 2026)

Lee, Sudhir, and Wang extend AIRL to consumers who differ in unobserved ways and
to action-dependent utilities. Their setting is sequential content: a reader of
serialized fiction decides each period whether to continue, paying an access cost,
or to exit. The paper proves (its Theorems 1 to 3) that fixing the exit-action
reward to zero and assuming an absorbing state makes the adversarial discriminator
recover the true reward and value, even under stochastic transitions. An
expectation-maximization layer then infers latent segments and segment-specific
rewards. The empirical study uses proprietary readership data, so this is an
identification reproduction on a controlled problem, not a match of the published
estimates.

The package reproduces the identification on a 61-state, two-segment problem with
an exit-action anchor and an absorbing state, a reward over 20 content features,
and discount 0.92.

### Anchored segment recovery (two segments, priors 0.48 / 0.52)

| Quantity | Value |
| --- | ---: |
| Segment assignment accuracy | 0.895 |
| Segment prior error (L1) | 0.043 |
| Segment policy distance | 0.059 |
| Segment reward error (normalized RMSE) | 0.24, 0.27 |

The estimator recovers which segment each user belongs to with about 90 percent
accuracy, the segment sizes to an L1 of 0.04, and each segment's policy to a
distance of 0.06. The exit anchor and the absorbing state pin the action-dependent
utilities and the latent segments, which is the paper's identification claim. The
published consumption estimates use proprietary data and are not reproduced here.

Reproduce:

```bash
python validation/estimators/airl2/run.py
```

## Pending

These estimators have a paper target but no completed replication yet. Each is held
to the same bar: match the published numbers to four or more significant figures, on
both the estimates and the standard errors.

| Estimator | Paper | Status |
| --- | --- | --- |
| UFXP | Oguz and Bray (2026) | Not yet evaluated. |
