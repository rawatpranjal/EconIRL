# legacy/ - retired code and docs

Superseded or abandoned work, kept for reference but out of the active package.
The home and this README are tracked so the convention is visible; the
**contents are gitignored** (`/legacy/*`, `!/legacy/README.md`).

## What belongs here (guidance, not law)

- estimators, scripts, or pages that were replaced and are no longer maintained.
- old experiment harnesses kept only so a result can be reproduced later.
- pre-migration layouts or docs preserved for history.

## What does NOT belong here

- build output (`dist/`) - that is regenerated, leave it gitignored at the root.
- anything still imported or referenced. If it is live, it has a real home.

Prefer deleting over hoarding. `legacy/` is for the few things genuinely worth
keeping; git history already holds the rest.
