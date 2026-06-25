#!/usr/bin/env bash
# Provision Joe's sandbox so a human can run Claude Code in it AS Joe.
# Generates the matched problem (truth held back IN THIS REPO), copies only the
# observable inputs into a fresh folder outside the repo, installs the published
# econirl in a clean venv, and drops Joe's persona as that folder's CLAUDE.md.
#
#   ./provision_sandbox.sh nfxp ~/joe-sandbox
#
# Then:  cd ~/joe-sandbox && source .venv/bin/activate && claude
# Grade: python grade.py --truth problems/nfxp/truth.json --joe ~/joe-sandbox/findings.json
set -euo pipefail

EST="${1:-nfxp}"
SBX="${2:-$HOME/joe-sandbox}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
PROB="$ROOT/problems/$EST"

echo ">> generating matched problem ($EST), truth held back in this repo"
PYTHONPATH="$ROOT/../src" python "$ROOT/synth.py" --estimator "$EST" --out "$PROB"

echo ">> provisioning sandbox at $SBX (the answer key is NOT copied)"
mkdir -p "$SBX"
cp "$PROB/panel.csv" "$PROB/features.npy" "$PROB/problem.json" "$SBX/"
[ -f "$ROOT/papers/$EST.pdf" ] && cp "$ROOT/papers/$EST.pdf" "$SBX/paper.pdf" || true
sed "s/{ESTIMATOR}/$EST/g" "$ROOT/joe_CLAUDE.md" > "$SBX/CLAUDE.md"
# guard: the answer key must never land in the sandbox
test ! -e "$SBX/truth.json" && echo "   OK: truth.json is not in the sandbox"

echo ">> clean venv with the published package only"
uv venv "$SBX/.venv" >/dev/null
uv pip install --python "$SBX/.venv/bin/python" -q econirl==0.0.7 jupyter pandas numpy matplotlib

echo ""
echo ">> Joe's sandbox is ready. Launch him with:"
echo "     cd $SBX && source .venv/bin/activate && claude"
echo "   Joe reads CLAUDE.md as his brief and writes findings.json when done."
echo "   Grade afterwards:"
echo "     python $ROOT/grade.py --truth $PROB/truth.json --joe $SBX/findings.json"
