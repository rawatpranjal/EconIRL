#!/usr/bin/env bash
# Drive one estimator chunk: generate the matched problem (truth held back),
# seal Joe in a fresh-install container, prove isolation, hand off to Joe.
#
#   ./run.sh nfxp            generate + build + seal + isolation check (free, no API)
#   ./run.sh nfxp joe        also launch the Opus Joe loop (needs API key + budget)
set -euo pipefail

EST="${1:-nfxp}"
MODE="${2:-prep}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
PROB="$ROOT/problems/$EST"

echo ">> generating matched problem for $EST (truth held back)"
PYTHONPATH="$ROOT/../src" python "$ROOT/synth.py" --estimator "$EST" --out "$PROB"

echo ">> building Joe's sealed image"
docker build -q -t econirl-joe "$ROOT" >/dev/null

echo ">> sealing container (only the observable inputs go in)"
docker rm -f joe >/dev/null 2>&1 || true
docker run -d --name joe econirl-joe >/dev/null
docker cp "$PROB/panel.csv"    joe:/work/panel.csv
docker cp "$PROB/features.npy" joe:/work/features.npy
docker cp "$PROB/problem.json" joe:/work/problem.json
[ -f "$ROOT/papers/$EST.pdf" ] && docker cp "$ROOT/papers/$EST.pdf" joe:/work/paper.pdf || true

echo ">> isolation + fresh-install checks"
docker exec joe bash -lc 'test ! -e /work/truth.json' \
  && echo "   OK: answer key is not in the container"
docker exec joe bash -lc 'test ! -e /work/src && test ! -e /work/.git' \
  && echo "   OK: no repo source reachable"
docker exec joe bash -lc 'python -c "import econirl; print(\"   OK: econirl\", econirl.__version__, \"imports from a clean PyPI install\")"'

if [ "$MODE" = "joe" ]; then
  echo ">> launching Joe (Opus)"
  python "$ROOT/joe_agent.py" --container joe --estimator "$EST" --problem "$PROB"
else
  echo ">> container 'joe' is ready. Launch Joe with:  ./run.sh $EST joe"
fi
