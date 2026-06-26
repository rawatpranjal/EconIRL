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
SRC="${3:-pypi}"   # pypi = econirl==0.0.7 (released); local = the fixed working tree
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

echo ">> clean venv"
uv venv "$SBX/.venv" >/dev/null
if [ "$SRC" = "local" ]; then
  echo "   installing the LOCAL fixed build from the working tree (not PyPI 0.0.7)"
  uv pip install --python "$SBX/.venv/bin/python" -q "$ROOT/.." jupyter pandas numpy matplotlib
else
  echo "   installing the published package econirl==0.0.7 (a real PyPI stranger)"
  uv pip install --python "$SBX/.venv/bin/python" -q econirl==0.0.7 jupyter pandas numpy matplotlib
fi
# Joe sees only the installed package; the repo source is not in his sandbox.
test ! -e "$SBX/src" && test ! -e "$SBX/.git" && echo "   OK: no repo source in the sandbox"

echo ""
echo ">> Joe's sandbox is ready. Launch him with:"
echo "     cd $SBX && source .venv/bin/activate && claude"
echo "   Joe reads CLAUDE.md as his brief and writes findings.json when done."
echo "   Grade afterwards:"
echo "     python $ROOT/grade.py --truth $PROB/truth.json --joe $SBX/findings.json"
