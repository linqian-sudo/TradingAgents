#!/usr/bin/env bash
# Isolated install of the TradingAgents research bundle.
# Does not clone git, does not read a secret file, and does not start a service.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${HOME}/.local/share/tradingagents-research"
PYTHON="${PYTHON:-python3}"
MODE="locked"

usage() {
  echo "usage: install.sh [--prefix DIR] [--python PYTHON] [--unpinned]"
  echo "default prefix: ${HOME}/.local/share/tradingagents-research"
  echo "locked mode installs deploy/requirements.lock.txt then the local package."
  echo "--unpinned installs dependency ranges from pyproject.toml (less reproducible)."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --unpinned)
      MODE="unpinned"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "Python 3.11 or newer is required. Got:" >&2
  "$PYTHON" -V >&2 || true
  exit 1
fi

mkdir -p "$PREFIX"
echo "creating venv at $PREFIX/venv"
"$PYTHON" -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/python" -m pip install --upgrade pip

if [[ "$MODE" == "locked" ]]; then
  LOCK="$ROOT/deploy/requirements.lock.txt"
  if [[ ! -f "$LOCK" ]]; then
    echo "missing $LOCK" >&2
    echo "refuse to fall back to floating versions; pass --unpinned to allow that explicitly" >&2
    exit 1
  fi
  grep -Ev '^(tradingagents([= ].*)?|-e |#)' "$LOCK" > "$PREFIX/requirements.deps.txt"
  "$PREFIX/venv/bin/python" -m pip install -r "$PREFIX/requirements.deps.txt"
  "$PREFIX/venv/bin/python" -m pip install --no-deps "$ROOT"
else
  echo "unpinned install from pyproject ranges; this can resolve different versions than the verified lock" >&2
  "$PREFIX/venv/bin/python" -m pip install "$ROOT"
fi

mkdir -p "$PREFIX/bin" "$PREFIX/var/results" "$PREFIX/var/cache" "$PREFIX/var/memory"
cat > "$PREFIX/bin/tradingagents-research" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export TRADINGAGENTS_NO_UPDATE_CHECK=1
export TRADINGAGENTS_RESULTS_DIR="\${TRADINGAGENTS_RESULTS_DIR:-$PREFIX/var/results}"
export TRADINGAGENTS_CACHE_DIR="\${TRADINGAGENTS_CACHE_DIR:-$PREFIX/var/cache}"
export TRADINGAGENTS_MEMORY_LOG_PATH="\${TRADINGAGENTS_MEMORY_LOG_PATH:-$PREFIX/var/memory/trading_memory.md}"
exec "$PREFIX/venv/bin/tradingagents-research" "\$@"
EOF
cat > "$PREFIX/bin/tradingagents" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export TRADINGAGENTS_NO_UPDATE_CHECK=1
export TRADINGAGENTS_RESULTS_DIR="\${TRADINGAGENTS_RESULTS_DIR:-$PREFIX/var/results}"
export TRADINGAGENTS_CACHE_DIR="\${TRADINGAGENTS_CACHE_DIR:-$PREFIX/var/cache}"
export TRADINGAGENTS_MEMORY_LOG_PATH="\${TRADINGAGENTS_MEMORY_LOG_PATH:-$PREFIX/var/memory/trading_memory.md}"
exec "$PREFIX/venv/bin/tradingagents" "\$@"
EOF
chmod 755 "$PREFIX/bin/tradingagents-research" "$PREFIX/bin/tradingagents"

"$PREFIX/venv/bin/python" - <<PY
import json, platform, sys
from datetime import datetime, timezone
from importlib.metadata import version
import tradingagents
from research_entry.provenance import FORK_REPO, LICENSE_NAME, UPSTREAM_COMMIT, UPSTREAM_REPO
payload = {
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "distribution": "tradingagents",
    "version": tradingagents.__version__,
    "installed_distribution_version": version("tradingagents"),
    "upstream_repo": UPSTREAM_REPO,
    "upstream_commit": UPSTREAM_COMMIT,
    "fork_repo": FORK_REPO,
    "license": LICENSE_NAME,
    "install_prefix": "$PREFIX",
    "install_mode": "$MODE",
    "credentials_included": False,
}
open("$PREFIX/VERSION_MANIFEST.json", "w", encoding="utf-8").write(json.dumps(payload, indent=2) + "\n")
print("installed", tradingagents.__version__, "at", "$PREFIX")
PY

echo "console scripts:"
echo "  $PREFIX/bin/tradingagents-research"
echo "  $PREFIX/bin/tradingagents"
echo "manifest: $PREFIX/VERSION_MANIFEST.json"
