# Style guide for the econirl JSS paper

A loose collection of conventions we are following. Treat it as taste, not law. The four exemplar papers do not agree on everything, and neither do we.

## What we are imitating

The reference is Conlon and Gortmaker (2020) on PyBLP. It is the closest paper in shape to ours. A long-form software paper, a Python package, methodological contributions on the side, a journal with conventions close to JSS. When in doubt, read a section of PyBLP and write something with the same rhythm.

Secondary references when PyBLP does not cover the case. Rosseel (2012) on lavaan for inline-session listings. Gleave et al. (2022) on imitation for the related-software comparison. Raffin et al. (2021) on Stable-Baselines3 for the hero teaser.

## The principles, in order

**Problem first, package second.** Open with the substantive question and the audience that cares. The package shows up after the reader knows why they should keep reading. PyBLP does not name itself until the third sentence of the abstract, and we follow suit.

**Understate.** Do not call your contribution novel, principled, or state-of-the-art. Show the workflow, point at the comparison table, and let the reader draw the conclusion. The exemplars routinely admit that their results coincide with the existing literature on most things and differ on a few specific things. That is the register we want.

**One register per section.** Some sections are notation-heavy and look like applied math: the unified notation block, the per-estimator objectives, the algorithm pseudocode. Some sections are prose-heavy and look like a software README: the introduction, the related-software comparison, the worked illustrations. Do not force prose flavor into the math sections, and do not force equations into the prose sections. PyBLP does this without comment.

**Conversational where possible.** "We find," "By contrast," "A first example," "The main disadvantage" are good. Long passive constructions and stacked nominalizations are bad. Read your paragraph aloud. If it is unreadable, rewrite it in shorter sentences with the verbs in the right place.

**Punctuation is not a moral test.** Em dashes, semicolons, and colons are fine when they help the sentence. Do not stack them three to a sentence. Do not use them in place of a period when a period would do. The earlier rule that banned them outright produced stiff prose and we are relaxing it.

**Bullet lists belong outside the body.** PyBLP and lavaan use them for target-audience enumerations and for the section roadmap. Everywhere else they convert to prose. We do the same. A short comma list inside one sentence is not a bullet list and is fine.

**Show the workflow, not the script.** Worked examples use the lavaan inline-session format. Prompt, call, printed output, one paragraph of interpretation. Full scripts live in `code_snippets/` and are referenced by the caption.

```
>>> from econirl.datasets import load_rust_bus
>>> from econirl.environments.rust_bus import RustBusEnvironment
>>> from econirl.preferences.linear import LinearUtility
>>> from econirl.estimation import NFXP
>>> env = RustBusEnvironment(num_mileage_bins=90, discount_factor=0.9999)
>>> result = NFXP().estimate(panel=load_rust_bus(as_panel=True),
...     utility=LinearUtility.from_environment(env),
...     problem=env.problem_spec, transitions=env.transition_matrices)
>>> print(result.summary())
                          coef    std err        t    P>|t|
operating_cost          0.0010     0.0004     2.54    0.011
replacement_cost        3.0722     0.0747    41.11    0.000
```

**Bold package names. Code-font class and function names. Italicize terms of art on the first use.** "We provide **econirl**, a Python package for ..." "The `Estimator.estimate()` method." "We project the neural reward onto a *sieve basis*." Reset the italic for terms when the section changes register.

**First person plural in prose.** "We introduce econirl." "Our package implements." Reserve the impersonal third person for the math sections, where the agent does not matter and the equation does the talking.

**Interpret, do not summarize.** A table caption tells the reader what the table is. The paragraph that follows says what is interesting about the numbers in it. If the prose just restates the column headers, cut it.

## Section register, by section

| Section | Register | Notes |
|--|--|--|
| Abstract | Prose, problem first, install line at the end. | Three to four sentences. |
| Introduction | Prose. | One teaser code block at the end. No related work. |
| Models and methods | Math heavy. | Unified notation up front. Per-estimator objectives are the spine. Pseudocode only where the outer structure is non-trivial. |
| Related software | Prose plus comparison table. | Pulled out of the introduction so the reader can judge the comparison after reading the methods. |
| Software design | Prose, code-font heavy. | API contract, inference layer, JAX backbone. Listings are short. |
| Illustrations | Inline-session listings interleaved with interpretive prose. | Three worked examples on real datasets, then a benchmark table. |
| Computational details | Terse, factual. | Versions, hardware, OS. |
| Summary | Prose. | Roadmap and the two or three things we did not get to. |
| Appendix | Tables and pointers. | Reproducibility map. |

## Section ordering

Introduction, Models and methods, Related software, Software design, Illustrations, Computational details, Summary, Appendix. The earlier draft included a standalone "Why econirl" section. We dropped it because the motivation it carried lives naturally in the introduction.

## Useful phrases

Reuse these without apology.

- "The rest of this paper is organized as follows."
- "A first example."
- "The main disadvantage of X is."
- "A key advantage of X is."
- "By contrast."
- "We find that."

## What to keep an eye on

These are common mistakes, not absolute prohibitions.

- Self-applied superlatives like "novel," "state-of-the-art," "principled." The comparison table exists for a reason.
- Hedging like "we believe," "arguably," "it should be noted." Either claim it or drop it.
- Two ideas in one sentence joined by a semicolon when a period would do.
- Long lists in body prose. Short ones inside a sentence are fine. Bulleted lists in body prose are not.
- Captions that summarize the figure twice. The caption is a one-line label, the prose interprets.

## What to verify before submission

A short pass, not an audit.

- The abstract opens with the substantive problem and closes with `pip install econirl` plus the docs URL.
- Every estimator named in the paper appears in both the taxonomy table and the benchmark table.
- Every numbered figure and table caption names the script that produces it.
- Every numerical claim traces to a script in the appendix table.
- The paper compiles with no undefined references.
- The PDF fits the page limit.
