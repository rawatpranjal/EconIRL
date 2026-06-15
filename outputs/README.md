# outputs/ — generated artifacts

Things produced *by* the repo: compiled paper PDFs, rendered notebooks, figures,
one-off experiment result dumps. The home and this README are tracked so the
convention is visible; the **contents are gitignored** (`/outputs/*`,
`!/outputs/README.md`) because they are regenerated, not authored.

## What belongs here (guidance, not law)

- compiled papers, slide decks, rendered notebooks.
- experiment runs and their result dumps (e.g. an autolab sweep, a JAX benchmark).
- scratch figures and tables for a draft.

## What does NOT belong here

- `validation/results/*.json` — that is **evidence the tests and public docs
  depend on**, not a throwaway output. It stays in `validation/`. The rule of
  thumb: if code imports it or a docs page cites it, it is validation, not output.
- source notebooks meant to be run as docs — those live with the docs.
