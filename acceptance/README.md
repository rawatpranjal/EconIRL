# Joe acceptance program

Black-box QA before the package goes public. "Joe" is an opinionated Opus agent
playing an applied scientist at Uber. He `pip install econirl` from PyPI inside a
sealed Docker container, sees only the published package, the public docs, a paper,
and a dataset, and tries to recover a real medium-scale messy problem. He never
sees this repo, the author, or the true answer. The orchestrator holds the answer
key and grades him. What he finds drives the final cleanup.

The default roster is now the focused core set from `project/estimator_roster.md`:
NFXP, CCP, NNES, TD-CCP, MCE-IRL, RHIP, AIRL-1/AIRL, AIRL-2/AIRL-Het, and
GLADIUS. Other estimators remain available, but they are not default Joe chunks
unless the user explicitly promotes or names them.

Full design and the chunk list (J0..J12) live in `project/roadmap.md` under the
"Joe from Uber" stream.

## Exact paper replication profiles

Use these when the task is "match the paper number exactly, no cheating." These
profiles may use tighter tolerances or original-paper preprocessing that should
not automatically become package-wide defaults.

```bash
make rust-table-ix
```

This downloads Rust's official NFXP archive into `downloads/`, mirrors
`STORDAT.GPR` for the 1975 GMC A5308 group, runs the strict NFXP/BHHH Table IX
profile, writes machine receipts to `acceptance/loop/nfxp/table_ix/`, and writes
the private markdown receipt to `project/replications/nfxp_rust1987_table_ix.md`.

## Layout

```
synth.py       generate one estimator's matched problem; hold the answer key       (tracked)
grade.py       score Joe's recovered theta against the held-out truth              (tracked)
Dockerfile     Joe's sealed environment: a clean PyPI install, nothing from here   (tracked)
run.sh         generate -> build -> seal -> isolation check -> hand off to Joe      (tracked)
joe_agent.py   the Opus tool-use loop + Joe's persona                              (gitignored)
problems/      generated panel/features/problem + the held-back truth.json          (gitignored)
papers/        the paper handed to Joe per estimator                                (gitignored)
reports/       Joe's transcripts, findings.json, and the grades                     (gitignored)
```

## Running Joe

Two ways to drive Joe. Default is a Claude Code session in a sandbox folder (free, no API key).

**A. Claude Code in a sandbox (default).** Provision a folder outside the repo, then run Claude
Code in it; it reads `CLAUDE.md` and becomes Joe.

```
./provision_sandbox.sh nfxp ~/joe-sandbox
cd ~/joe-sandbox && source .venv/bin/activate && claude     # Joe works, writes findings.json
python grade.py --truth problems/nfxp/truth.json --joe ~/joe-sandbox/findings.json
```

Caveat: a Claude Code session still loads the user's global `~/.claude/CLAUDE.md`, so Joe is not a
perfectly clean stranger. The answer key is kept out of the sandbox, so the blindness that matters
holds. For a fully clean Joe, run Claude Code inside the container instead (path B).

**B. Sealed container (most isolated; needs an Anthropic key for the Opus loop).**

```
./run.sh nfxp          # generate, build + seal the container, prove isolation (free)
./run.sh nfxp joe      # launch the Opus Joe loop (needs ANTHROPIC_API_KEY + budget)
python grade.py --truth problems/nfxp/truth.json --joe reports/nfxp_findings.json
```

## Invariants (do not weaken)

- The answer key (`truth.json`) never enters the container. `run.sh` asserts this.
- No repo source is reachable from inside the container. `run.sh` asserts this.
- The matched problem is medium-scale and messy and is NOT diluted to make an
  estimator pass. A miss is a finding.
- The harness author never signs off on its own grade: a fresh agent verifies each
  recovery against the truth before a chunk closes.
