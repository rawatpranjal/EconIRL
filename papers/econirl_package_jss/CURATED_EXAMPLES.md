# Curated examples for the econirl JSS paper

This document picks the worked examples that go into the paper. The curation principle is to anchor each example to the dataset its source paper uses, then add a paired large-state-space version in the same family so that the classical-versus-neural contrast plays out within the same canonical setting. The result is five datasets organized into three families.

| Dataset | Family | State space | Source-paper anchor | Estimator that wins |
| --- | --- | --- | --- | --- |
| `rust-small`     | Rust bus     | 90 mileage bins, 2 actions      | Rust 1987           | NFXP, CCP, MPEC                  |
| `rust-big`       | Rust bus     | 31-dim augmented, 2 actions     | Kang et al.\ 2025   | GLADIUS, NNES, TD-CCP            |
| `ziebart-small`  | Gridworld    | 100 cells, 5 actions, tabular   | Ziebart 2008        | MCE-IRL, MaxEnt-IRL              |
| `ziebart-big`    | Gridworld    | 2500 cells, 8 actions, neural   | Ziebart 2008 scaled | Deep MCE, AIRL                   |
| `lsw-synthetic`  | Heterogeneity| 30 chapter index, 3 actions, 2 latent types | Lee-Sudhir-Wang 2026 | AIRL-Het |

The five-dataset arc is the spine of Section 4 of the paper. Each estimator in the production registry appears on at least one dataset where it is the right tool. Each dataset produces one headline figure or table whose numbers reproduce from a single script. The first four datasets have direct source-paper anchors. The fifth is a documented semi-synthetic mirror of the source paper because the original platform data is not redistributable.

## Curation principle

A good worked example in a software paper does three things at once. It demonstrates the workflow on data the reader already trusts. It produces a numerical headline that an independent reader can reproduce. It tells the reader something they could not have learned by reading the original method paper alone. PyBLP did this with the original BLP cars example, lavaan did it with the Holzinger-Swineford 1939 dataset. We follow the same template by anchoring each example to a published source paper and by pairing every classical example with a larger variant in the same family so the reader can see where classical methods break.

The small-versus-big pairing matters. A paper that shows only canonical small examples leaves the reader wondering whether the unified API is necessary at all when tabular methods already work. A paper that shows only large neural examples leaves the reader wondering whether the unified API still serves the reader who only ever fits NFXP on small panels. The two together carry the argument that the same workflow scales across the whole spectrum without changing the user-facing surface.

## Family 1. Rust bus

### `rust-small`

**Source paper.** Rust 1987, Econometrica.

**Dataset.** Real Madison Metro maintenance records, 8261 observations, 90 mileage bins, two actions to keep or replace the engine. Loader is `econirl.datasets.load_rust_small`. The package also bundles a synthetic counterpart with 9411 observations and a known data-generating process for parameter-recovery validation.

**Hero estimator.** NFXP. Recovers operating cost $0.001231$ and replacement cost $3.0115$, log-likelihood $-4263.20$, in 0.1 seconds on a single CPU thread.

**Cross-estimator role.** Same panel for the cross-estimator benchmark in Section 4.4. All twelve production estimators run on the same data with the same starting values. NFXP, CCP, MPEC, MCE-IRL, NNES, SEES, TD-CCP, GLADIUS recover the ground truth to four decimal places. AIRL pays two orders of magnitude more wall-clock time. IQ-Learn recovers approximately at near-zero time. f-IRL fails. BC is the lower-bound baseline.

**Headline claim.** When the state space is small and the model is well specified, every structural-family estimator converges to the same answer. The interesting differences are wall-clock time and inferential machinery, not point estimates.

**Script.** `examples/rust-bus-engine/replicate.py` and `examples/rust-bus-engine/benchmark_all_estimators.py`.

### `rust-big`

**Source paper.** Kang, Yoganarasimhan, and Jain 2025 working paper. The high-dimensional Rust extension that augments the original mileage state with a configurable number of dummy state variables.

**Dataset.** `econirl.datasets.load_rust_big`. Real Madison Metro panel from `rust-small` with 30 dummy state variables appended, for a total state-space dimensionality of 31. The dummy variables are drawn independently of the choice so that the true reward depends only on mileage, but tabular value iteration sees a state space of size $90 \times 2^{30}$ and runs out of memory. Action set unchanged.

**Hero estimator.** GLADIUS. Recovers the operating-cost and replacement-cost coefficients on the genuine mileage dimension to within five percent of the `rust-small` NFXP reference, and produces a smooth implied reward over the full augmented state space. Runs in under thirty seconds on a single CPU thread.

**Contrast estimators.** NFXP on the same panel aborts because the inner Bellman solve runs out of memory at 31 state dimensions. The failure is the point. The package surfaces it cleanly through the convergence flag and the identification diagnostics rather than producing silent garbage. NNES and TD-CCP recover the parameters by neural function approximation at higher wall-clock cost than GLADIUS but with structural standard errors available.

**Headline claim.** When the state space breaks tabular value iteration, the same unified pipeline that fits NFXP on `rust-small` admits a documented failure mode and admits successful neural alternatives on the same data with no change in the user-facing workflow. The structural identification of the parameters is preserved because the neural estimators still maximize the structural likelihood.

**Script.** `examples/rust-bus-engine/gladius_highdim.py`.

## Family 2. Ziebart gridworld

### `ziebart-small`

**Source paper.** Ziebart 2008, AAAI. The gridworld validation panel from his thesis, which was the controlled-domain proof of concept for maximum causal entropy IRL.

**Dataset.** `econirl.datasets.load_ziebart_small`. 20000 synthetic trajectories on a $10 \times 10$ grid with five actions and a known reward of step-cost $-0.1$ plus terminal $+10$ at a goal cell.

**Hero estimator.** MCE-IRL. Recovers the step cost and terminal reward to within machine precision. Cosine similarity to the true reward is $0.9999$.

**Contrast estimators.** AIRL and IQ-Learn on the same panel. AIRL recovers the same reward up to an additive constant in roughly 120 seconds. IQ-Learn recovers it under the chi-squared objective in under one second. BC fails because it does not model dynamics.

**Headline claim.** On a controlled domain with known ground truth, the IRL family recovers the reward and the differences come down to wall-clock time and the form of the recovered reward. This is the one place in the paper where the IRL family is benchmarked against itself rather than against the structural family.

**Script.** `examples/taxi-gridworld/run_ziebart_small.py`.

### `ziebart-big`

**Source paper.** Ziebart 2008 scaled up. The same maximum causal entropy IRL problem but on a large enough grid that tabular soft value iteration becomes the bottleneck and feature-based reward parametrization becomes essential.

**Dataset.** `econirl.datasets.load_ziebart_big`. 50000 synthetic trajectories on a $50 \times 50$ grid with eight actions including diagonals, stochastic transitions with slip probability $0.1$, and a feature-based reward of the form $r(s) = \theta^\top \phi(s)$ where $\phi$ is a 16-dimensional radial basis function basis on the grid coordinates. Total state space is 2500 cells and the soft value iteration inner loop becomes the dominant cost for the tabular MCE-IRL estimator.

**Hero estimator.** Deep MCE-IRL with neural reward parametrization. Recovers the reward landscape to a cosine similarity of at least $0.95$ in under five minutes on CPU and under one minute on GPU. Outperforms the tabular MCE-IRL on out-of-sample log-likelihood by a measurable margin because the neural reward smooths across cells that the tabular variant treats as independent.

**Contrast estimators.** MCE-IRL with the linear basis recovers the reward more slowly and at higher variance. AIRL on the same panel recovers a discriminator-based reward in roughly 20 minutes on CPU. BC fails as before.

**Headline claim.** When the gridworld is large enough that the MCE-IRL inner soft value iteration is the bottleneck, the same workflow that fit MCE-IRL on `ziebart-small` admits a deep variant on `ziebart-big` with the same calling convention and recovers a smoother reward at lower wall-clock cost. The contrast tells the IRL-family scalability story exactly as the Rust family tells the structural-family scalability story.

**Script.** `examples/taxi-gridworld/run_ziebart_big.py`.

## Family 3. Lee-Sudhir-Wang serialized content

### `lsw-synthetic`

**Source paper.** Lee, Sudhir, and Wang 2026 working paper on adversarial inverse reinforcement learning with unobserved heterogeneity. The empirical application is a serialized fiction platform with 24000 users, 6000 chapters across 151 books, fifteen months of reading data, and a discrete choice set of pay-and-read, wait-and-read, or quit. The paper identifies two dominant latent consumer segments accounting for 72 percent of consumption and 96 percent of purchases. The platform data is not redistributable so we ship a documented semi-synthetic mirror.

**Dataset.** `econirl.datasets.load_lsw_synthetic`. Semi-synthetic panel of 5000 users reading across 50 books with 30 chapters each over a six-month observation window. State variables are chapter index within the current book, a four-dimensional content embedding, the wait time since the chapter became available, the platform pricing window, and a latent type drawn from a two-component mixture. Three actions are pay-and-read, wait-and-read, and quit. The data-generating process places one latent type as high-patience and monetization-focused with positive utility on pay-and-read, and the other as budget-conscious and patient with positive utility on wait-and-read. Population mixture weight is $0.4$ for the high-patience type and $0.6$ for the patient type, chosen so that the simulated marginal purchase rate matches the four percent reported in the source paper. The script that generates the panel is `code_snippets/build_lsw_synthetic.py` and its docstring documents every parameter.

**Hero estimator.** AIRL-Het with two latent types. Recovers the type-specific reward parameters and the population mixture weights through EM iterations over the discriminator-based reward. Reports asymptotic standard errors on the type weights and on each type's reward coefficients.

**Contrast estimators.** AIRL without heterogeneity on the same panel recovers a population-average reward that smears across the two types and produces type-specific predictions worse than the heterogeneous variant. MCE-IRL with no heterogeneity, same outcome. BC, same outcome. The point of the contrast is that the heterogeneous variant earns its EM iterations on a panel with genuine latent structure.

**Headline claim.** On a panel that mirrors the Lee-Sudhir-Wang serialized content setting, the heterogeneous estimator recovers two distinct latent reward functions with mixture weights matching the source paper, and the homogeneous estimators do not. This is the modern-econometrics centerpiece of the paper because no other open-source package implements adversarial IRL with EM unobserved heterogeneity.

**Script.** `examples/lsw-synthetic/replicate.py` and `code_snippets/build_lsw_synthetic.py`.

## Mapping to the paper sections

The five datasets map onto Section 4 of the paper as follows.

- Section 4.1 walks through `rust-small` with NFXP and the full inference and counterfactual pipeline.
- Section 4.2 walks through `ziebart-small` with MCE-IRL and contrasts AIRL on the same panel.
- Section 4.3 walks through the small-to-big transition. First `rust-big` with GLADIUS as the structural-family scalability story. Then `ziebart-big` with deep MCE-IRL as the IRL-family scalability story.
- Section 4.4 is the modern centerpiece on `lsw-synthetic` with AIRL-Het.
- Section 4.5 is the cross-estimator benchmark on `rust-small` covering all twelve production estimators.

The arc moves from classical structural likelihood on a small state space (4.1), to classical IRL on a controlled domain (4.2), through the small-to-big scalability story in both families (4.3), to modern adversarial IRL with unobserved heterogeneity (4.4), and closes with the cross-estimator benchmark that brings every estimator back to common ground (4.5).

## What needs to be built

Five loaders need to exist before the paper compiles end to end. The first two already exist or trivially reuse existing data. The last three need new data-generating scripts.

| Loader | Status |
| --- | --- |
| `econirl.datasets.load_rust_small`     | exists as `load_rust_bus`, rename and alias |
| `econirl.datasets.load_ziebart_small`  | exists as `load_taxi_gridworld`, rename and alias |
| `econirl.datasets.load_rust_big`       | thin wrapper over `load_rust_small` that appends dummies, build |
| `econirl.datasets.load_ziebart_big`    | new data-generating script, build |
| `econirl.datasets.load_lsw_synthetic`  | new data-generating script, build |

The existing loaders are kept as deprecated aliases for one minor version. The new names match the dataset family naming convention used in the paper, which means the paper, the code, and the documentation all use the same words for the same artifact.

Three scripts need to be written.

1. `code_snippets/build_rust_big.py`. Takes `rust-small` and appends a configurable number of dummy state variables drawn independently of the choice. Default is 30 dummies for a state-space dimensionality of 31. Outputs a CSV plus a metadata JSON.
2. `code_snippets/build_ziebart_big.py`. Generates 50000 trajectories on a $50 \times 50$ grid with eight actions, slip probability $0.1$, and a 16-dimensional radial basis reward. Outputs a CSV plus a metadata JSON declaring the true reward coefficients.
3. `code_snippets/build_lsw_synthetic.py`. Generates the LSW mirror panel as described above. Outputs a CSV plus a metadata JSON declaring the latent-type mixture weights and the type-specific reward coefficients used by the data-generating process.

Each script is deterministic given the random seed of $42$ documented in `BENCHMARK_PROTOCOL.md`. Each script writes its output to `examples/<dataset>/data/` so the data lives next to the example that uses it. The data-generating provenance is recorded in the script docstring and in the metadata JSON.

## Acceptance test

For each of the five datasets the loader returns a `Panel` object with the schema documented in `econirl.core.types`. The headline figure or table for each dataset reproduces from a single script run on a clean Python 3.11 environment with the dependencies pinned in Section 5. Wall-clock budget is under thirty minutes per script except the AIRL-Het example, which takes up to two hours on CPU because of the EM outer loop and finishes in under fifteen minutes on a single A100. The benchmark runner produces a `manifest.csv` and the LaTeX build halts if any output is missing.

## Headline summary

Five datasets in three families. `rust-small` and `rust-big` carry the structural family from canonical to scaled. `ziebart-small` and `ziebart-big` carry the IRL family from canonical to scaled. `lsw-synthetic` carries the modern unobserved-heterogeneity story on a documented mirror of Lee-Sudhir-Wang 2026. Every estimator in the production registry appears on at least one dataset where it is the right tool. Every dataset produces one headline figure or table whose numbers reproduce from a single script. The naming is uniform, the loaders share a signature, and the same workflow runs across all five.
