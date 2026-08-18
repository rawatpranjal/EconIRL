# GLADIUS Qualification Runbook

Use this procedure when the estimator, paper-replication protocol, bootstrap,
serialization surface, or applied notebook changes. Run it from a clean release
branch. Do not publish a different commit from the one qualified here.

## 1. Record and Test the Candidate

```bash
git status --short
git rev-parse HEAD
uv sync --all-extras --dev
uv run ruff check .
uv run mypy src/econirl
uv run pytest -q
```

## 2. Known-Truth Structural Gates

```bash
PYTHONPATH=src:. uv run python validation/estimators/gladius/run.py \
  --quiet-progress --enforce-gates
```

This must write `validation/results/gladius.json` with status
`strict_structural_counterfactual_pass` and 12 passing gates.

## 3. Frozen Bootstrap Calibration

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

## 4. Paper Table 2 Replication

Run seeds 0 through 19 at all six paper sample sizes. The 128,000-update cap is
converted to whole epochs per sample size, subject to the 800-epoch ceiling,
and recorded in every shard.

```bash
uv run python validation/estimators/gladius/paper_table2_mape.py \
  --sweep --reps 5 --start-seed 0 --max-updates 128000 \
  --out /tmp/gladius-table2-0.json
uv run python validation/estimators/gladius/paper_table2_mape.py \
  --sweep --reps 5 --start-seed 5 --max-updates 128000 \
  --out /tmp/gladius-table2-1.json
uv run python validation/estimators/gladius/paper_table2_mape.py \
  --sweep --reps 5 --start-seed 10 --max-updates 128000 \
  --out /tmp/gladius-table2-2.json
uv run python validation/estimators/gladius/paper_table2_mape.py \
  --sweep --reps 5 --start-seed 15 --max-updates 128000 \
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

## 5. Build the Wheel and Exercise Installed Code

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
uv run pytest -q
git status --short
```

Visually inspect the rendered GLADIUS landing, quick-start, validation,
counterfactual, high-state, and runbook pages. If any receipt, notebook output,
or source file changed after the wheel was built, rebuild and repeat the wheel
checks. Run the repository documentation-cleanup gate before any milestone PR,
release, tag, or protected-main push.
