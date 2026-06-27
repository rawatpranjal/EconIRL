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
