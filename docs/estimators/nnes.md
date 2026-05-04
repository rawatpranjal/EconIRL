# NNES

**Reference PDF:** [NNES reference and known-truth validation tutorial](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nnes/nnes.pdf)

NNES estimates structural dynamic discrete choice models with a neural
value-function approximation inside an NPL-style policy iteration. It targets
the same objects as NFXP, CCP, MPEC, and SEES: reward parameters, policy, value
function, Q function, and counterfactual policies.

## Validation Status

NNES passes the enforced known-truth gates on both the easy canonical DGP and
the high-dimensional canonical DGP. Both runs use 2,000 simulated individuals,
80 periods per individual, known transitions, 3 actions, homogeneous rewards,
and the shared Type A/B/C counterfactual checks.

| Cell | States | Reward features | Observations | Converged | L-BFGS-B iterations | Log-likelihood | Final V loss | Hard gates |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `canonical_low_action` | 21 | 4 | 160,000 | true | 10 | -174875.80929267284 | 0.0003994984270360631 | 11 / 11 |
| `canonical_high_action` | 81 | 32 | 160,000 | true | 29 | -160272.88031279508 | 0.029931686287506906 | 11 / 11 |

The high-dimensional preset uses 16-dimensional state features and a
32-parameter action-dependent reward basis. Its pre-estimation diagnostics pass:
feature rank is 32 / 32, the reward-feature condition number is
1.376810450603823, all 81 states are observed, and state-action coverage is
0.9588477366255144.

## Recovery Metrics

| Metric | Gate | Low-dimensional value | High-dimensional value |
| --- | ---: | ---: | ---: |
| Parameter cosine similarity | >= 0.95 | 0.9982397556304932 | 0.991203784942627 |
| Parameter relative RMSE | <= 0.30 | 0.0651790127158165 | 0.13511048257350922 |
| Parameter RMSE | -- | 0.017850005999207497 | 0.01414113026112318 |
| Maximum parameter absolute error | -- | 0.022873617708683014 | 0.03015984036028385 |
| Reward RMSE | <= 0.08 | 0.010209567844867706 | 0.0640120804309845 |
| Policy total variation | <= 0.03 | 0.00564560820338359 | 0.023834346317619222 |
| Policy KL | -- | 0.00009810229540605409 | 0.002693513289699444 |
| Value RMSE | <= 0.20 | 0.019845455486253175 | 0.11562010299193262 |
| Q RMSE | <= 0.20 | 0.0233699468610916 | 0.13714528932947226 |

The easy cell is very tight. The high-dimensional cell is looser on reward,
value, and Q recovery, but it remains inside every hard gate.

## Counterfactual Recovery

| Counterfactual | Regret gate | Low regret | High-dimensional regret | Low policy TV | High-dimensional policy TV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Type A | <= 0.05 | 0.00022427010596255665 | 0.004865025745682183 | 0.005080256555500048 | 0.021578199248338113 |
| Type B | <= 0.05 | 0.00033179297440636394 | 0.005558862016315504 | 0.005504813369390121 | 0.02165145983648699 |
| Type C | <= 0.05 | 0.00005469089622241882 | 0.0013139353505658108 | 0.0026945148023238114 | 0.013052841154959438 |

## What Is Being Validated

The known-truth harness compares the estimator against the exact DGP solution,
not only against likelihood fit. For NNES this means:

- structural reward-parameter recovery;
- recovered reward, value, Q, and policy objects;
- Type A transition/reward-shift counterfactuals;
- Type B transition-change counterfactuals;
- Type C action-restriction counterfactuals.

NNES also has component tests for the algebra inside the profiled NPL step:
the policy-evaluation linear system, the profiled value fixed point, the
profiled Q/policy fixed point, the theta-dependent continuation term, anchor
normalization, and the high-dimensional fixed point.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/nnes/nnes_run.py --quiet-progress
PYTHONPATH=src:. python experiments/known_truth.py --cell-id canonical_high_action --estimator NNES --output-dir /tmp/econirl_nnes_status_high
PYTHONPATH=src:. pytest tests/test_nnes_known_truth_components.py tests/test_known_truth.py -v
```

The primer generator writes `nnes_results.tex`, `nnes_results.json`, and a full
JSON copy under `/tmp/econirl_nnes_primer_known_truth`.

## Code Pointers

- Estimator: `src/econirl/estimation/nnes.py`
- Known-truth harness: `experiments/known_truth.py`
- Component tests: `tests/test_nnes_known_truth_components.py`
- Shared known-truth tests: `tests/test_known_truth.py`
