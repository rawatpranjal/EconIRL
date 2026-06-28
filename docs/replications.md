# Replications

This page reports numerical replication evidence. A study counts as a paper
replication only when the published design, sample, and estimand are available and
the reported quantities are directly comparable. Each section sets the package
value against the paper's published value, side by side.

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
from the cost parameters, so they are the same across both rows. The replication
runs through `make rust-table-ix`, which fetches the official NFXP data.

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

## MCE-IRL (Ziebart et al., 2008 and 2010)

The paper reports a route-choice table that needs the original taxi trajectories,
which are not public, and no published gridworld table. The package evidence is a
controlled 5x5 gridworld recovery, not a paper-number match.

| Setting | Reward cosine | Reward RMSE |
| --- | ---: | ---: |
| Gridworld run 1 | 0.9997 | 0.0310 |
| Gridworld run 2 | 0.9998 | 0.0639 |
| Gridworld run 3 | 0.9998 | 0.0514 |

This is simulation evidence, not a replication of a published number.

## Pending

These estimators have a paper target but no completed replication yet. Each is held
to the same bar: match the published numbers to four or more significant figures, on
both the estimates and the standard errors.

| Estimator | Paper | Status |
| --- | --- | --- |
| CCP | Hotz-Miller (1993), Aguirregabiria-Mira (2002) | Not yet evaluated. |
| NNES | Nguyen (2025) | Not yet evaluated. |
| TD-CCP | Adusumilli-Eckardt (2025) | Not yet evaluated. |
| RHIP | Barnes et al. (2024) | Not yet evaluated. |
| AIRL | Fu-Luo-Levine (2018) | Not yet evaluated. |
| AIRL-Het | Lee-Sudhir-Wang (2026) | Not yet evaluated. |
| GLADIUS | Kang-Yoganarasimhan-Jain (2025) | Not yet evaluated. |
