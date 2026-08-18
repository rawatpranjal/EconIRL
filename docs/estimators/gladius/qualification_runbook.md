# GLADIUS Qualification Runbook

Use this procedure when the estimator, paper-replication protocol, bootstrap,
serialization surface, or applied notebook changes. Run it from a clean release
branch. Do not publish a different commit from the one qualified here.

## 1. Record and Test the Candidate

```bash
git status --short
git rev-parse HEAD
uv sync --all-extras --dev
uv run ruff check \
  src/econirl/estimation/gladius.py \
  src/econirl/estimators/neural_gladius.py \
  tests/test_gladius.py \
  tests/test_gladius_010_contract.py \
  tests/test_gladius_bootstrap_calibration.py \
  tests/test_gladius_notebook.py \
  tests/test_gladius_paper_simple_cases.py \
  tests/test_gladius_paper_table2.py \
  tests/test_gladius_qualification_report.py \
  tests/test_gladius_serialization_check.py \
  tests/test_neural_gladius.py \
  tests/test_validation_evidence.py \
  validation/estimators/gladius/bootstrap_calibration.py \
  validation/estimators/gladius/paper_table2_mape.py \
  validation/estimators/gladius/qualification_report.py \
  validation/estimators/gladius/run.py \
  validation/estimators/gladius/serialization_check.py \
  examples/gladius/gladius_applied_workflow.ipynb
uv run pytest -q \
  tests/test_gladius.py \
  tests/test_gladius_010_contract.py \
  tests/test_gladius_bootstrap_calibration.py \
  tests/test_gladius_notebook.py \
  tests/test_gladius_paper_simple_cases.py \
  tests/test_gladius_paper_table2.py \
  tests/test_gladius_qualification_report.py \
  tests/test_gladius_serialization_check.py \
  tests/test_neural_gladius.py \
  tests/test_validation_evidence.py \
  tests/test_rtd_style_guide.py
```

The repository does not define project-wide Ruff or mypy release gates. Global
Ruff and mypy runs expose existing debt outside GLADIUS. The full pytest suite
also currently has unrelated TDCCP and SEES failures. Record those results
separately rather than presenting them as GLADIUS qualification requirements.

## 2. Cheap Paper-Objective Gates

Do not start Table 2 or bootstrap work until the literal `paper_minimax` path
recovers cardinal rewards in the one-state and three-state synthetic cases:

```bash
uv run pytest -q tests/test_gladius_paper_simple_cases.py
```

These tests use no bootstrap and at most 200 epochs. They guard raw reward
levels, policies, and known anchor rewards. Formula equality and finite output
are not substitutes for these gates.

The gate was added after the prior Table 2 driver was found to label receipts
as `shared_trunk` without setting `network_mode`, so it had actually trained
the lower-level `separate` default. The repaired recipe also matches the
checked-in author code's whole-trajectory batches, no batch shuffle,
Xavier-normal weights, zero hidden biases, -55 output bias, summed zeta loss,
unclipped zeta updates, and epoch learning-rate decay. Its per-Q-update anchor
level projection is a package repair beyond the author code; it uses only the
known anchor reward. Each projection is a common shift that leaves the current
action differences and policy unchanged, while changing the subsequent
optimization path. The qualification driver uses batches of two trajectories
at N=50 and five trajectories in larger cells, so Q-update count grows with
sample size. This batch rule is a disclosed package stabilization; the paper
does not publish the Table 2 batch size.

## 3. Oracle-Simulation Structural Checks

```bash
PYTHONPATH=src:. uv run python validation/estimators/gladius/run.py \
  --quiet-progress --enforce-gates
```

This must write `validation/results/gladius.json` with status
`strict_structural_counterfactual_pass` and 12 passing gates.

## 4. Prespecified Bootstrap Calibration (Deferred During Paper Diagnosis)

Do not rerun this expensive release gate until the staged paper checks in
Section 5 pass. The existing receipt remains useful for the unchanged public
bootstrap path; this section records the eventual release procedure.

The design is 20 panels, 19 whole-trajectory draws per panel. Each fit uses a
categorical encoder for the nominal four-state DGP, a 32-by-2 network, fixed
estimator seed 42, the calibrated public Bellman penalty 0.1, and at most 300
epochs with patience 50. Every base and bootstrap refit must converge.
Intervals are centered on the fitted functional with whole-trajectory
bootstrap standard errors. Shards are only a runtime device; the merge rejects
conflicting records. The receipt preserves the failed percentile and
reduced-capacity runs and records that the DGP, target functionals, coverage
thresholds, and tail thresholds did not change.

```bash
uv run python validation/estimators/gladius/bootstrap_calibration.py \
  --start-panel 0 --panels 5 --n-bootstrap 19 --skip-reproducibility \
  --output /tmp/gladius-bootstrap-0.json
uv run python validation/estimators/gladius/bootstrap_calibration.py \
  --start-panel 5 --panels 5 --n-bootstrap 19 --skip-reproducibility \
  --output /tmp/gladius-bootstrap-1.json
uv run python validation/estimators/gladius/bootstrap_calibration.py \
  --start-panel 10 --panels 5 --n-bootstrap 19 --skip-reproducibility \
  --output /tmp/gladius-bootstrap-2.json
uv run python validation/estimators/gladius/bootstrap_calibration.py \
  --start-panel 15 --panels 5 --n-bootstrap 19 --skip-reproducibility \
  --output /tmp/gladius-bootstrap-3.json

uv run python validation/estimators/gladius/bootstrap_calibration.py \
  --merge-shard /tmp/gladius-bootstrap-0.json \
  --merge-shard /tmp/gladius-bootstrap-1.json \
  --merge-shard /tmp/gladius-bootstrap-2.json \
  --merge-shard /tmp/gladius-bootstrap-3.json \
  --output validation/results/gladius_bootstrap_calibration.json
```

The merge also repeats one seeded panel twice and requires byte-for-byte equal
records.

## 5. Paper Table 2 Replication

Stage the paper check and stop at the first failed stage:

```bash
uv run python validation/estimators/gladius/paper_table2_mape.py \
  --sizes 50 --reps 7 --start-seed 0 --max-epochs 200 \
  --out /tmp/gladius-n50-seeds0-6.json
uv run python validation/estimators/gladius/paper_table2_mape.py \
  --sizes 250 --reps 1 --start-seed 0 --max-epochs 200 \
  --out /tmp/gladius-n250-seed0.json
uv run python validation/estimators/gladius/paper_table2_mape.py \
  --sizes 500 --reps 1 --start-seed 0 --max-epochs 400 \
  --out /tmp/gladius-n500-seed0.json
```

Stop at the first failed stage. On 2026-08-18 the adaptive protocol produced
4.323% mean MAPE for N=50 seeds 0-6, 0.484% for N=250 seed 0, and 0.892%
for N=500 seed 0. All three pass their prespecified bounds. The detailed diagnosis is
in `validation/estimators/gladius/paper_path_report.md`.

Run seeds 0 through 19 at all six paper sample sizes. Use 800 epochs in every
cell. Do not pass `--max-updates`: the old cap changed the number of data passes
with N. Do not pass `--batch-size`: the checked policy uses two trajectories at
N=50 and five in larger cells.

```bash
uv run python validation/estimators/gladius/paper_table2_mape.py \
  --sweep --reps 5 --start-seed 0 --max-epochs 800 \
  --out /tmp/gladius-table2-0.json
uv run python validation/estimators/gladius/paper_table2_mape.py \
  --sweep --reps 5 --start-seed 5 --max-epochs 800 \
  --out /tmp/gladius-table2-1.json
uv run python validation/estimators/gladius/paper_table2_mape.py \
  --sweep --reps 5 --start-seed 10 --max-epochs 800 \
  --out /tmp/gladius-table2-2.json
uv run python validation/estimators/gladius/paper_table2_mape.py \
  --sweep --reps 5 --start-seed 15 --max-epochs 800 \
  --out /tmp/gladius-table2-3.json

uv run python validation/estimators/gladius/paper_table2_mape.py \
  --merge-shard /tmp/gladius-table2-0.json \
  --merge-shard /tmp/gladius-table2-1.json \
  --merge-shard /tmp/gladius-table2-2.json \
  --merge-shard /tmp/gladius-table2-3.json \
  --out validation/results/gladius_paper_table2.json
```

The simulation-only best-true-MAPE epoch rule matches the checked-in author
experiment. It is deliberately isolated from the public fit path.

## 6. Build the Wheel and Exercise Installed Code

Build only after the candidate changes and tracked scientific receipts are
committed. The serialization receipt records the exact commit.

```bash
uv build
uvx twine check dist/*

gladius_release_tmp=$(mktemp -d)
uv venv "$gladius_release_tmp/venv"
uv pip install --python "$gladius_release_tmp/venv/bin/python" \
  dist/econirl-*.whl jupyter nbconvert

cd "$gladius_release_tmp"
"$gladius_release_tmp/venv/bin/python" \
  /absolute/path/to/econirl/validation/estimators/gladius/serialization_check.py \
  --expect-wheel \
  --output /absolute/path/to/econirl/validation/results/gladius_serialization.json

cp /absolute/path/to/econirl/examples/gladius/gladius_applied_workflow.ipynb \
  "$gladius_release_tmp/gladius_applied_workflow.ipynb"
"$gladius_release_tmp/venv/bin/jupyter" nbconvert \
  --to notebook --execute "$gladius_release_tmp/gladius_applied_workflow.ipynb" \
  --output "$gladius_release_tmp/gladius_applied_workflow.executed.ipynb" \
  --ExecutePreprocessor.timeout=600
cp "$gladius_release_tmp/gladius_applied_workflow.executed.ipynb" \
  /absolute/path/to/econirl/examples/gladius/gladius_applied_workflow.ipynb
```

Replace `/absolute/path/to/econirl` with the clean release worktree. Confirm
that the notebook prints `Installed package import: True`.

## 6. Fail-Closed Combined Gate

```bash
cd /absolute/path/to/econirl
PYTHONPATH=src:. uv run python validation/estimators/gladius/qualification_report.py
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
# Repeat the scoped Ruff and pytest commands from Section 1.
git status --short
```

Visually inspect the rendered GLADIUS landing, quick-start, validation,
counterfactual, high-state, and runbook pages. If any receipt, notebook output,
or source file changed after the wheel was built, rebuild and repeat the wheel
checks. Run the repository documentation-cleanup gate before any milestone PR,
release, tag, or protected-main push.
