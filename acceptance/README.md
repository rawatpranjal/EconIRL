# Joe acceptance program

Black-box QA before the package goes public. "Joe" is an opinionated Opus agent
playing an applied scientist at Uber. He `pip install econirl` from PyPI inside a
sealed Docker container, sees only the published package, the public docs, a paper,
and a dataset, and tries to recover a real medium-scale messy problem. He never
sees this repo, the author, or the true answer. The orchestrator holds the answer
key and grades him. What he finds drives the final cleanup.

Full design and the chunk list (J0..J16) live in `project/roadmap.md` under the
"Joe from Uber" stream.

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

## The loop

```
./run.sh nfxp          # generate the problem, build + seal the container, prove isolation (free)
./run.sh nfxp joe      # also launch Joe (needs an Anthropic API key + budget)
python grade.py --truth problems/nfxp/truth.json --joe reports/nfxp_findings.json
```

## Invariants (do not weaken)

- The answer key (`truth.json`) never enters the container. `run.sh` asserts this.
- No repo source is reachable from inside the container. `run.sh` asserts this.
- The matched problem is medium-scale and messy and is NOT diluted to make an
  estimator pass. A miss is a finding.
- The harness author never signs off on its own grade: a fresh agent verifies each
  recovery against the truth before a chunk closes.
