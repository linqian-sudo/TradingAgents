#!/usr/bin/env bash
# Offline self-check: import, CLI help, symbol map, unconfigured run.
# Does not call a model and does not fetch market data. Pass the install prefix.
set -euo pipefail

PREFIX="${1:-${HOME}/.local/share/tradingagents-research}"
OUT="${2:-$PREFIX/var/self-check}"
BIN="$PREFIX/bin/tradingagents-research"
PY="$PREFIX/venv/bin/python"

if [[ ! -x "$BIN" || ! -x "$PY" ]]; then
  echo "install not found at $PREFIX" >&2
  exit 1
fi

mkdir -p "$OUT"
echo "import and version"
"$PY" - <<'PY'
import tradingagents
from research_entry.provenance import UPSTREAM_COMMIT, UPSTREAM_VERSION
assert tradingagents.__version__ == UPSTREAM_VERSION, tradingagents.__version__
print("import-ok", tradingagents.__version__, UPSTREAM_COMMIT)
PY

echo "upstream CLI help"
"$PREFIX/bin/tradingagents" --help > "$OUT/tradingagents-help.txt"

echo "research CLI help"
"$BIN" --help > "$OUT/tradingagents-research-help.txt"

echo "symbol map"
"$BIN" map --reference --output "$OUT/map"
AS_OF="$(date -u -d 'yesterday' +%F)"
echo "unconfigured run for $AS_OF (expect exit 2)"
set +e
"$BIN" run --symbol 0700.HK --market HK --date "$AS_OF" --output "$OUT/unconfigured"
code=$?
set -e
if [[ "$code" -ne 2 ]]; then
  echo "expected exit 2 for an unconfigured model, got $code" >&2
  exit 1
fi
echo "self-check-ok $OUT"
