# Estimator Fix Blueprint

Use this blueprint when fixing one estimator from implementation through tests,
artifacts, documentation, and public Read the Docs verification.

The goal is one estimator that a reader can inspect from three angles. The code
must be linked. The generated results must be reproducible. The public docs must
explain what was checked and how to read the evidence.

## Scope

Pick exactly one estimator.

Name the estimator key used by the known-truth harness. Examples are `NFXP`,
`CCP`, `TD-CCP`, `MCE-IRL`, `AIRL`, `AIRL-Het`, `f-IRL`, `IQ-Learn`,
`GLADIUS`, `SEES`, and `NNES`.

Set the shell variables used by the commands below.

```bash
export EST=ccp
export KEY=CCP
```

Find the public wrapper, the lower-level estimator, the known-truth harness
entry, the primer generator, the generated artifacts, the estimator docs, and
the tests.

```bash
rg -n "${KEY}|run_estimator|run_cell_estimator" \
  src tests experiments papers docs
```

Keep unrelated dirty files out of scope. Read them only if they affect the
estimator being fixed.

## Code

Start from the estimator contract in `experiments/known_truth.py`. Check how
the estimator is constructed, what data object it receives, what summary object
it returns, and which gates are applied.

Fix the implementation where the bug lives. Prefer the local estimator API and
existing summary types. Do not add a parallel result format when the shared
interfaces already carry the needed fields.

For structural estimators, confirm transition orientation before debugging
numerics. The internal convention used by the validators is action, state, next
state.

For IRL estimators, confirm the reward normalization and identification claim
before changing thresholds. Do not turn a diagnostic failure into a success
claim by loosening gates without a source-backed reason.

## Generator

Each estimator needs a generator script at this shape.

```text
papers/econirl_package/primers/ESTIMATOR_SLUG/ESTIMATOR_SLUG_run.py
```

The generator must run from the repository root.

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/${EST}/${EST}_run.py --quiet-progress
```

The generator must write two repo-local artifacts.

```text
papers/econirl_package/primers/ESTIMATOR_SLUG/ESTIMATOR_SLUG_results.tex
papers/econirl_package/primers/ESTIMATOR_SLUG/ESTIMATOR_SLUG_results.json
```

The `.tex` file is the rendered table source. The `.json` file is the
machine-readable release artifact.

The JSON artifact should use this top-level shape.

```json
{
  "estimator": "ESTIMATOR_KEY",
  "paper_target": "plain target claim",
  "primary_cell_id": "known_truth_cell_id",
  "release_status": "release status",
  "result": {}
}
```

The `result` object should include diagnostics, summary, metrics, and gates
when the estimator supports the shared known-truth surface.

Write strict JSON. Do not publish `NaN`, `Infinity`, or `-Infinity`. Convert
non-finite diagnostic values to `null`.

If the generator imports shared primer display helpers, make sure those helpers
are tracked. A generator that runs locally through an untracked helper is not a
reproducible public link.

## Tests

Run the narrow checks first.

```bash
python -m py_compile \
  papers/econirl_package/primers/validation_display.py \
  papers/econirl_package/primers/${EST}/${EST}_run.py
```

Run the estimator generator and inspect the artifacts.

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/${EST}/${EST}_run.py --quiet-progress
```

Parse the JSON artifact with strict parsing.

```bash
python - <<'PY'
import json
import os
from pathlib import Path

def reject_constant(value):
    raise ValueError(value)

est = os.environ["EST"]
path = Path(f"papers/econirl_package/primers/{est}/{est}_results.json")
data = json.loads(path.read_text(), parse_constant=reject_constant)
required = {"estimator", "paper_target", "primary_cell_id", "release_status", "result"}
missing = required - set(data)
assert not missing, missing
for key in ["diagnostics", "summary", "metrics", "gates"]:
    assert key in data["result"], key
print("strict-json-ok")
PY
```

Run the estimator-specific tests. Include the known-truth tests when the harness
or gates changed.

```bash
PYTHONPATH=src:. python -m pytest tests/test_${EST}.py -q
PYTHONPATH=src:. python -m pytest tests/test_known_truth.py -q
```

If the repo has unrelated failing tests, report them as unrelated. Do not call a
repo-wide test run green unless the output is green.

## Documentation

Every estimator docs surface should have the same reader path.

For structural estimators, matching NFXP and CCP means matching the full docs
topology, not only the validation page. The parent page should use the same
section form.

```text
# ESTIMATOR
## Quick Decision
## Minimal Fit
## What Is Certified
## ESTIMATOR Guide
```

The guide toctree should use the same child-page order unless there is a
methodological reason to differ.

```text
ESTIMATOR_SLUG/context
ESTIMATOR_SLUG/quick_start
ESTIMATOR_SLUG/under_the_hood
ESTIMATOR_SLUG/pre_estimation
ESTIMATOR_SLUG/validation
ESTIMATOR_SLUG/counterfactuals
ESTIMATOR_SLUG/rust_bus
```

If an estimator has a real API difference from NFXP or CCP, keep the page slot
and document the boundary explicitly. Do not omit the page or pretend the
wrapper exists. For example, if the estimator only has a lower-level API, the
quick-start and Rust-bus pages should use that API and state the boundary.

The parent page should state what is certified and link the machine-readable
artifact.

```text
docs/estimators/ESTIMATOR_SLUG.md
```

The validation page should explain what the known-truth cell checks. It should
link the generator script, the `.tex` table source, and the `.json` artifact. It
should show the rerun command. It should show a compact harness snippet when the
script is too long to inline. It should explain how to read the result tables.

```text
docs/estimators/ESTIMATOR_SLUG/validation.md
```

The result-heavy child pages should link back to the validation page so the
reader can trace the tables.

```text
docs/estimators/ESTIMATOR_SLUG/pre_estimation.md
docs/estimators/ESTIMATOR_SLUG/counterfactuals.md
```

Copy numbers from the generated artifacts. Do not type new table values by hand
unless you also verify they match the regenerated JSON and `.tex` files.

Keep the page form and function aligned with the NFXP and CCP pages unless the
estimator has a real methodological reason to differ. If it differs, state the
reason on the relevant page.

## Local Checks

Run whitespace checks on the exact files you touched.

```bash
git diff --check -- \
  docs/estimators/${EST}.md \
  docs/estimators/${EST} \
  papers/econirl_package/primers/${EST}
```

Check that docs and artifacts agree on the values that readers see.

```bash
python - <<'PY'
import json
import os
from pathlib import Path

est = os.environ["EST"]
artifact = Path(f"papers/econirl_package/primers/{est}/{est}_results.json")
data = json.loads(artifact.read_text())
doc = Path(f"docs/estimators/{est}/validation.md").read_text()
tex = Path(f"papers/econirl_package/primers/{est}/{est}_results.tex").read_text()
time_text = f"{data['result']['summary']['estimation_time']:.2f} seconds"
assert time_text in doc, time_text
assert time_text in tex, time_text
print("doc-artifact-sync-ok")
PY
```

For paired or templated estimator docs, compare headings.

```bash
python - <<'PY'
import os
from pathlib import Path

est = os.environ["EST"]
for path in sorted(Path(f"docs/estimators/{est}").glob("*.md")):
    heads = [line for line in path.read_text().splitlines() if line.startswith("#")]
    print(path, heads)
PY
```

For structural estimators, compare the child-page set and toctree order against
NFXP and CCP. A single validation page is not enough when the docs claim to
follow that setup.

```bash
python - <<'PY'
import os
from pathlib import Path

est = os.environ["EST"]
expected = [
    "context",
    "quick_start",
    "under_the_hood",
    "pre_estimation",
    "validation",
    "counterfactuals",
    "rust_bus",
]

for slug in ["nfxp", "ccp", est]:
    actual_files = sorted(path.stem for path in Path(f"docs/estimators/{slug}").glob("*.md"))
    missing = sorted(set(expected) - set(actual_files))
    assert not missing, (slug, missing)

    parent = Path(f"docs/estimators/{slug}.md").read_text()
    cursor = -1
    for item in expected:
        needle = f"{slug}/{item}"
        position = parent.find(needle)
        assert position > cursor, needle
        cursor = position
print("doc-topology-ok")
PY
```

## Git

Stage only the estimator files that belong to this fix.

```bash
git status --short
git add \
  docs/estimators/${EST}.md \
  docs/estimators/${EST} \
  papers/econirl_package/primers/${EST}/${EST}_run.py \
  papers/econirl_package/primers/${EST}/${EST}_results.tex \
  papers/econirl_package/primers/${EST}/${EST}_results.json
git diff --cached --name-status
git diff --cached --check
```

If the generator depends on a shared helper that is not tracked, stage that
helper too.

Use a commit message that names the estimator and the validation surface.

```bash
git commit -m "docs: fix ${EST} validation provenance"
git push origin main
```

## Read the Docs

Do not build docs locally for this repo. Push first, then trigger RTD.

```bash
curl -sS -X POST \
  -H "Authorization: Token $RTD_TOKEN" \
  https://readthedocs.org/api/v3/projects/econirl/versions/latest/builds/
```

Poll the build until it finishes. Use the returned build id.

```bash
export BUILD_ID=33055730
```

```bash
curl -sS "https://app.readthedocs.org/api/v3/projects/econirl/builds/${BUILD_ID}/"
```

Verify that the build succeeded on the commit you pushed.

Then verify the public pages with a cache-busting query string.

```bash
curl -sS -L -o /dev/null -w "%{http_code}\n" \
  "https://econirl.readthedocs.io/en/latest/estimators/${EST}.html?v=${BUILD_ID}"

curl -sS -L -o /dev/null -w "%{http_code}\n" \
  "https://econirl.readthedocs.io/en/latest/estimators/${EST}/validation.html?v=${BUILD_ID}"
```

Check that the rendered validation page includes the generator, the artifacts,
and the new explanatory text.

```bash
curl -sS -L \
  "https://econirl.readthedocs.io/en/latest/estimators/${EST}/validation.html?v=${BUILD_ID}" \
  | rg "${EST}_run.py|${EST}_results.tex|${EST}_results.json|Read the tables"
```

Verify the GitHub artifact links.

```bash
for url in \
  "https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/${EST}/${EST}_run.py" \
  "https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/${EST}/${EST}_results.tex" \
  "https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/${EST}/${EST}_results.json"; do
  curl -sS -L -o /dev/null -w "%{http_code} ${url}\n" "$url"
done
```

Verify the raw JSON artifact again from GitHub.

```bash
python - <<'PY'
import json
import os
import subprocess

def reject_constant(value):
    raise ValueError(value)

est = os.environ["EST"]
url = f"https://raw.githubusercontent.com/rawatpranjal/EconIRL/main/papers/econirl_package/primers/{est}/{est}_results.json"
raw = subprocess.check_output(["curl", "-sS", url], text=True)
data = json.loads(raw, parse_constant=reject_constant)
print(data["estimator"], data["primary_cell_id"], data["release_status"])
PY
```

## Done

The estimator is done when the code path is fixed, the generator reruns, the
JSON artifact is strict and current, the docs link the code and artifacts, the
targeted tests pass, the commit is pushed, the RTD build succeeds on that
commit, and the public pages plus GitHub artifact links return `200`.

If any part fails, report the failing command and the exact reason. Do not call
the estimator fixed until the failing step has either passed or been explicitly
scoped as unrelated.
