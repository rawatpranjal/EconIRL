# Estimator Primer Tex Files Design

## Purpose

Each of the 12 estimators in econirl gets a standalone 5-page LaTeX primer that distills the source paper into a self-contained reference document. The primers serve as the bridge between the original academic papers and the code in the repository. They are building blocks that feed into the main econirl package paper.

## File Organization

```
papers/primers/
  preamble.sty              # shared packages, macros, notation commands
  generate_tables.py        # reads JSON results, writes tex table fragments
  tables/                   # auto-generated tex table fragments
    nfxp_results.tex
    ccp_results.tex
    ...
  nfxp.tex                  # NFXP-NK primer (template example)
  ccp.tex
  mpec.tex
  nnes.tex
  tdccp.tex
  sees.tex
  mce_irl.tex
  airl.tex
  airl_het.tex              # moved from papers/lsw_algorithm.tex
  gladius.tex
  f_irl.tex
  iq_learn.tex              # moved from papers/iq_learn_derivation.tex
  bayesian_irl.tex
  gail.tex
  gcl.tex
  max_margin_irl.tex
  max_margin_planning.tex
  deep_maxent.tex
```

Existing stray tex files are moved into this directory:
- `papers/iq_learn_derivation.tex` moves to `papers/primers/iq_learn.tex`
- `papers/lsw_algorithm.tex` moves to `papers/primers/airl_het.tex`

Files inside `papers/econirl_package/` (`aairl_algorithm.tex`, `iq_learn_equivalence.tex`) stay where they are as appendices to the main paper. They cross-reference the primers.

## Template Structure (5 pages)

Each primer follows this exact section structure.

### Section 1: Context and Problem (~0.5 pages)

What problem existed before this estimator. What the estimator does differently. One paragraph of motivation, one of contribution. Ends with the paper citation and a sentence placing the method in the estimator taxonomy (structural forward, neural structural, or inverse).

### Section 2: Model and Notation (~1 page)

Formal setup. Opens with a notation table defining every symbol introduced in this primer beyond the shared notation (s, a, Q, V, pi, beta, sigma, phi, theta are defined in the framework and not redefined). The objective function. The Bellman equation as it appears in this estimator. The key functional forms and loss functions. All equations are numbered and referenced in subsequent sections.

### Section 3: Identification and Theory (~1 page)

The core theoretical results. For structural estimators: consistency, efficiency, and the conditions under which theta is identified. For IRL estimators: what is identified up to what equivalence class, and under what conditions the identification sharpens. Formal propositions with proof sketches or references to the full proof in the source paper. Always addresses counterfactual implications: what policy experiments can and cannot be conducted with the identified object.

### Section 4: Algorithm (~1 page)

Full pseudocode using the `algorithm2e` package. Detailed enough to implement from this page alone. Each major step is annotated with the corresponding class, method, or function in `src/econirl/`. For example: "Step 3 corresponds to `NFXPEstimator._solve_inner()` in `src/econirl/estimation/nfxp.py`." The algorithm block fills most of the page, with brief prose above and below explaining initialization and convergence criteria.

### Section 5: Simulation (~1.5 pages)

A concrete example on a small MDP. Three components:

1. A 5-line Python code snippet showing how to call the estimator using the econirl API. References the corresponding showcase script in `examples/`.

2. An auto-generated results table. The showcase script writes results to a JSON file (this already exists for all showcases built earlier). The `generate_tables.py` script reads the JSON and writes a tex table fragment to `papers/primers/tables/{estimator}_results.tex`. The primer includes this fragment via `\input{tables/{estimator}_results.tex}`.

3. Brief interpretation of the results: what the numbers show about the estimator's strengths and limitations on this problem.

The section ends with a cross-reference to the RTD documentation page (`docs/estimators/{estimator}.md`) and the showcase script path.

## Shared Preamble (`preamble.sty`)

A LaTeX style file that all primers import via `\usepackage{preamble}`. Contains:
- Standard packages: amsmath, amssymb, booktabs, hyperref, algorithm2e, listings
- Notation macros matching the econirl_package paper: `\E` for expectation, `\R` for reals, `\argmin`, `\argmax`
- Theorem environments: proposition, remark, definition
- Code listing style for Python snippets
- Page geometry: 11pt, 1-inch margins
- Consistent hyperlink colors (blue)

## Auto-Updated Results Pipeline

`generate_tables.py` is a single script that:
1. Scans `examples/` for result JSON files (naming convention: `{estimator}_results.json`)
2. For each JSON, extracts the key metrics (parameters, timing, policy accuracy)
3. Writes a tex table fragment to `papers/primers/tables/{estimator}_results.tex`
4. The fragment is a bare `tabular` environment (no `table` wrapper, no caption) so the primer can wrap it with its own caption and label

Running `python papers/primers/generate_tables.py` regenerates all tables. This is run manually before compiling the primers, not as part of a CI pipeline.

## Build Process

To compile a primer:
```bash
cd papers/primers
python generate_tables.py          # regenerate tables from latest JSON
pdflatex nfxp.tex                  # compile individual primer
```

## First Primer: NFXP-NK

The first primer to be built establishes the template. It covers:
- Rust (1987) original formulation
- Iskhakov, Rust, and Schjerning (2016) SA-then-NK polyalgorithm
- BHHH optimization with analytical gradient via implicit function theorem
- Identification: MLE consistency under standard rank conditions on features
- Counterfactuals: policy simulation under parameter changes
- Simulation on the Rust bus engine (90 states, beta=0.95)
- Results from `examples/rust-bus-engine/mpec_inner_loop_showcase.py` (NFXP column) and any existing NFXP benchmark results

The NFXP primer source paper is `papers/foundational/1987_rust_optimal_replacement.md` (docling'd) supplemented by `papers/foundational/iskhakov_rust_schjerning_2016_mpec_comment.md` for the NK polyalgorithm.
