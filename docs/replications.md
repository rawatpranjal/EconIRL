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

The one-step estimator sits well below the MLE. NPL run to its fixed point
reaches it. The operating cost and replacement cost match NFXP to the third and
fourth figure. At this discount factor the choice likelihood is nearly flat in
the replacement cost. NPL and NFXP both sit about 0.0005 from the exact
maximizer and agree to the fourth figure. At lower discount factors, where the
replacement cost is better identified, NPL matches the maximum-likelihood
estimate to five figures.

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
4-action MDP with randomly drawn transition matrices and a reward at a single
state.

The package exposes the identified parts of the AIRL family:

| Public mode | Reward | Required normalization | Recovers the reward |
| --- | --- | --- | --- |
| `AIRL(version="state_only")` | R(s) | state-only restriction and deterministic decomposable dynamics | yes |
| `AIRL(version="anchored")` | R(s,a) | exit-action reward anchor and absorbing-state value anchor | yes |
| `AIRL(version="heterogeneous")` | R_k(s,a) | the same anchors plus latent-segment separation | yes |

Section 7.1 uses MaxEnt IRL, the precursor to the adversarial method. The package
estimator that implements that algorithm is MCE-IRL. The reward recovery (Figure 1)
is matched to the authors' own tabular code, and the transfer behavior (Figure 2) is
shown with our own probe.

Running the authors' recipe at their settings (16 states, 4 actions, seed 1,
sparsity 0.8), a port of their reference code recovers the ground-truth state-only
reward to affine R² 1.000 (Figure 1). The package soft-Bellman policy matches their
soft value iteration to 1e-15. The public MCE-IRL estimator, trained on
demonstrations from the same expert, recovers the state-only reward to affine R²
0.987. A state-action reward instead recovers a shaped advantage. It is uncorrelated
with the ground-truth reward (affine R² near zero), and it still reproduces the
expert policy under the training dynamics (policy distance near zero).

Figure 2 asks whether a learned reward still works when the world changes. The reward
that depends only on the state recovers the true goal, so it keeps steering toward that
goal in every new environment we tried. The reward that also depends on the action
explains the same training behavior, but it quietly absorbs the training environment, so
it usually stops working once the dynamics change. This is not all or nothing. The
action-based reward still works on a minority of new environments and fails on the rest,
so its average is shaky and one test environment tells you little. Read the transfer
column as a direction, not a precise score. State-based rewards carry over, action-based
rewards do not carry over reliably. This is the lesson Fu, Luo, and Levine draw in
Section 7.1.

| Section 7.1 route (seeds 1-3) | Reward affine R^2 | Transfer fraction of optimal |
| --- | ---: | ---: |
| reference code, state-only | 1.000 | 1.000 (steady) |
| MCE-IRL estimator, state-only | 0.987 | 1.000 (steady) |
| reference code, state-action | 0.003 (shaped) | 0.28 (varies widely) |

The transfer test is our own probe, because the authors' public tabular code runs the
recovery experiment only and the Figure 2 tabular transfer curve is not in their
repository. Matching their code is the bar for recovery, because Section 7.1 publishes
reward maps, not a numeric table.

A second deterministic-transition diagnostic exercises the adversarial AIRL
estimator (a different algorithm from the Section 7.1 MaxEnt code). It recovers the
state-only reward to affine R² 0.977 and transfer 1.000, and the unanchored
state-action diagnostic recovers a shaped advantage at affine R² 0.244.

The closest code-style stochastic-transition probe uses the sparse random
transition recipe from the public `inverse_rl` tabular environment: exp(rand)
weights, default `t_sparsity=0.75` sparse zeroing, row normalization, and one
reward-state self-loop action. The source is the public
[`tabular_maxent_irl/simple_env.py`](https://github.com/justinjfu/inverse_rl/blob/master/tabular_maxent_irl/simple_env.py)
environment used in Justin Fu's `inverse_rl` repository. This still is not a
paper-table exact match, because the paper reports a reward map and transfer
curve rather than numeric targets, but it matches the available code recipe more
closely than a dense Dirichlet matrix. From zero, with sampled rollout negatives
and previous-20 replay negatives, state-only AIRL gets reward R² 0.940 and
transfer 1.000.

A denser Dirichlet random-transition probe stresses the adversarial AIRL
estimator from zero, a harder setting than the Section 7.1 MaxEnt code matched
above. It is recorded separately and is not a success. With previous-20 policy
replay negatives and the discriminator log-odds policy reward, the bounded local
probe gets state-only reward R² 0.031 and transfer -0.419. This is an open stress
case for the adversarial estimator, not Section 7.1 evidence.

An exact tabular-occupancy from-zero AIRL diagnostic removes Monte Carlo noise
from the discriminator negatives on the same stochastic MDP. It improves the
state-only row to reward R² 0.586 and transfer 0.825, but it still misses the
paper-style recovery target and is recorded as an open gap.

The same stochastic MDP is identifiable from demonstrations in a non-AIRL
finite-sample inverse-Bellman ceiling. With 5,000 trajectories of length 80, the
empirical state-only inverse reaches reward R² 0.966 and transfer 0.991. That
ceiling isolates the remaining problem to AIRL adversarial training rather than
the random-transition design itself.

A warm-started AIRL continuation now preserves that stochastic signal. Starting
state-only AIRL from the empirical inverse-Bellman reward, sampling policy
negatives as rollouts from the same initial distribution, mixing previous-20
policy negatives, and running 5 adversarial rounds gives reward R² 0.966 and
transfer 0.991. This is useful package evidence for the AIRL training path, but
it is not the paper's from-zero run and should not be labeled as paper-exact
figure replication.

The unanchored state-action diagnostic is excluded from the public `AIRL` entry
point because the paper shows it recovers a shaped advantage, not an identified
structural reward. On the Section 7.1 transfer test, the state-only reward
re-optimizes to near-optimal behavior under a fresh transition matrix.

Reproduce:

```bash
python examples/airl-fu2018/run_fu_reference_match.py --seeds 1 2 3 \
  --out validation/results/airl_fu2018_71_reference_match.json
python examples/airl-fu2018/run_tabular_71.py --seeds 0
python examples/airl-fu2018/run_tabular_71.py \
  --transition-mode stochastic --seeds 0 \
  --n-individuals 250 --n-periods 20 \
  --max-rounds 50 --state-min-rounds 40 --action-min-rounds 10 \
  --discriminator-steps 3 --negative-history 20 \
  --state-generator-reward log_odds --action-generator-reward log_odds \
  --out validation/results/airl_fu2018_71_stochastic_probe.json
python examples/airl-fu2018/run_tabular_71.py \
  --transition-mode sparse_original --seeds 0 \
  --n-individuals 1500 --n-periods 40 \
  --max-rounds 100 --state-min-rounds 100 --action-min-rounds 1 \
  --discriminator-steps 20 --negative-history 20 \
  --policy-sample-mode rollout \
  --state-generator-reward recovered --action-generator-reward recovered \
  --state-policy-step-size 0.1 --reward-lr 0.01 \
  --out validation/results/airl_fu2018_71_sparse_original.json
python examples/airl-fu2018/run_tabular_71.py \
  --transition-mode stochastic --estimator-mode ccp_inverse --seeds 0 \
  --n-individuals 5000 --n-periods 80 \
  --out validation/results/airl_fu2018_71_stochastic_ccp_ceiling.json
python examples/airl-fu2018/run_tabular_71.py \
  --transition-mode stochastic --seeds 0 \
  --n-individuals 5000 --n-periods 80 \
  --max-rounds 100 --state-min-rounds 100 --action-min-rounds 1 \
  --discriminator-steps 20 --negative-history 0 \
  --discriminator-data-mode occupancy \
  --state-generator-reward log_odds --action-generator-reward recovered \
  --state-policy-step-size 0.1 --reward-lr 0.01 \
  --out validation/results/airl_fu2018_71_stochastic_airl_occupancy.json
python examples/airl-fu2018/run_tabular_71.py \
  --transition-mode stochastic --seeds 0 \
  --n-individuals 5000 --n-periods 80 \
  --max-rounds 5 --state-min-rounds 5 --action-min-rounds 1 \
  --discriminator-steps 1 --negative-history 20 \
  --policy-sample-mode rollout \
  --state-generator-reward recovered --action-generator-reward recovered \
  --state-initializer ccp_inverse --reward-lr 0.0005 \
  --out validation/results/airl_fu2018_71_stochastic_airl_ccp_init.json
```

## AIRL Anchored Heterogeneity (Lee, Sudhir, and Wang, 2026)

Lee, Sudhir, and Wang extend AIRL to dynamic discrete choice settings with
action-dependent rewards and latent consumer segments. Their identification
argument adds two anchors: the exit action has known zero flow payoff in every
state, and the absorbing terminal state has zero continuation value. With those
anchors, the AIRL pair recovers the action-dependent reward and value rather
than an arbitrary shaped equivalent. The EM layer assigns users to latent
segments and estimates segment-specific rewards and policies.

The empirical paper replication is blocked without the serialized-fiction
platform data. The paper uses K = 4 latent segments and a large platform panel;
its reported empirical targets include segment marginal effects and
out-of-sample prediction errors. The package cell below is therefore a synthetic
mechanism validation only. It has 61 states, 3 actions, 2 latent segments, exit
action 2, absorbing state 60, and 20 content reward features.

| Synthetic mechanism diagnostic | Package value | Comparison point |
| --- | ---: | --- |
| segment assignment accuracy | 0.895 | 0.5 is random assignment |
| segment prior L1 | 0.0435 | 0 is an exact segment-share match |
| max segment policy TV | 0.0591 | 0 is an exact policy match |
| max segment reward normalized RMSE | 0.2650 | 0 is an exact reward match |
| max segment value normalized RMSE | 0.1420 | 0 is an exact value match |
| max segment Q normalized RMSE | 0.2114 | 0 is an exact Q match |

The same realized synthetic panel has an oracle-MAP assignment reference, using
the true segment policies, of 0.716 for individual book trajectories and 0.901
after pooling all books from the same user. The fitted assignment accuracy of
0.895 is therefore close to the user-pooled oracle reference, while the
finite-panel stochastic choices rule out treating 1.000 as a fitted replication
target.

Segment labels are aligned before comparison because latent labels are
arbitrary up to permutation. In the public API this is
`AIRL(version="heterogeneous")`.

Reproduce:

```bash
python validation/estimators/aairl/run.py
```

## TD-CCP (Adusumilli and Eckardt, 2025)

Adusumilli and Eckardt estimate dynamic discrete choice models with
temporal-difference learning built on the conditional-choice-probability
approach. Their linear semi-gradient estimator approximates the recursive value
terms with basis functions and needs no transition densities. Their bus-engine
Monte Carlo (Online Appendix, Table B.1) is a Rust-style replacement problem with
one mileage state and a permanent bus type s in {1, 2}. The manager keeps or replaces each
period under Type-1 extreme-value shocks. The replacement payoff is set to zero,
and the keep payoff is theta0 + theta1 times mileage + theta2 times type. The true
values are theta0 = 2, theta1 = -0.15, theta2 = 1, with discount 0.9.

The paper reports recovery at the precision of maximum likelihood. The package
reproduces their Table B.1 with the nested fixed point on the same design, 1000
buses observed for 30 periods, across 300 Monte Carlo draws. Each parameter's
mean, standard deviation, and mean-squared error sit next to the paper.

### Bus-engine recovery, 1000 buses, T = 30, 300 draws

| Parameter | True | Package mean | Paper mean | Package SD | Paper SD | Package MSE | Paper MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| theta0 (intercept) | 2.0 | 2.0018 | 1.9788 | 0.0918 | 0.0868 | 0.0084 | 0.0080 |
| theta1 (mileage) | -0.15 | -0.1501 | -0.1492 | 0.0037 | 0.0033 | 0.00001 | 0.00001 |
| theta2 (type) | 1.0 | 1.0009 | 1.0044 | 0.0617 | 0.0583 | 0.0038 | 0.0034 |

The means, standard deviations, and mean-squared errors line up with the paper
across all three parameters. The package's linear semigradient recovers the
same parameter means on this design, but at wider sampling dispersion than
maximum likelihood. The table above reports the maximum-likelihood comparison.

Reproduce:

```bash
PYTHONPATH=src python validation/estimators/tdccp/bus_engine_nfxp.py --n-reps 300
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

### Four-module renewal (d = 4), 60 buses, T = 20, 100 draws

The paper also reports a four-dimensional design (their Table 2). The four modules
are independent, so the oracle solves four one-dimensional problems and the joint
likelihood factorizes. The package estimates each module on its own panel of 1200
observations. The true values are crep = (2.0, 2.5, 1.5, 1.8) and c = (0.05, 0.07,
0.09, 0.11).

| Module | Parameter | Package NFXP | Package NNES | Paper NFXP | Paper NNES |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | crep | 2.0286 (0.1459) | 2.0267 (0.1452) | 1.9733 (0.1366) | 1.8947 (0.1401) |
| 1 | c | 0.0509 (0.0053) | 0.0507 (0.0053) | 0.0501 (0.0053) | 0.0489 (0.0055) |
| 2 | crep | 2.5329 (0.1842) | 2.5269 (0.1827) | 2.4533 (0.1658) | 2.4393 (0.1665) |
| 2 | c | 0.0709 (0.0071) | 0.0704 (0.0070) | 0.0715 (0.0067) | 0.0794 (0.0070) |
| 3 | crep | 1.5125 (0.1383) | 1.5122 (0.1380) | 1.5102 (0.1295) | 1.5111 (0.1301) |
| 3 | c | 0.0913 (0.0088) | 0.0912 (0.0087) | 0.0897 (0.0085) | 0.0890 (0.0085) |
| 4 | crep | 1.8211 (0.1426) | 1.8201 (0.1422) | 1.8222 (0.1467) | 1.9023 (0.1503) |
| 4 | c | 0.1117 (0.0092) | 0.1116 (0.0092) | 0.1104 (0.0098) | 0.1171 (0.0106) |

The equality between NNES and the nested fixed point holds in d = 4 on all eight
parameters, to the third figure in both mean and standard deviation. The means
center on the true values. The package computes the value-function gradient in
closed form. The paper's two derivative methods, finite differences and a gradient
network, both approximate that same gradient, so they coincide here.

Reproduce:

```bash
PYTHONPATH=src python validation/estimators/nnes/bus_renewal_d4.py --n-reps 100
```

## Pending

These estimators have a paper target but no completed replication yet. Each is held
to the same bar: match the published numbers to four or more significant figures, on
both the estimates and the standard errors.

| Estimator | Paper | Status |
| --- | --- | --- |
| RHIP | Barnes et al. (2024) | Not yet evaluated. |
| GLADIUS | Kang-Yoganarasimhan-Jain (2025) | Not yet evaluated. |
| UFXP | Oguz and Bray (2026) | Not yet evaluated. |
