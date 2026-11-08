#!/usr/bin/env bash
# Remove an install created by deploy/install.sh.
# Keeps var/ (cache, results, memory) unless --purge-data is passed.
# Does not delete a home-directory .tradingagents folder unless --purge-home is passed.
set -euo pipefail

PREFIX="${HOME}/.local/share/tradingagents-research"
PURGE_DATA=0
PURGE_HOME=0

usage() {
  echo "usage: uninstall.sh [--prefix DIR] [--purge-data] [--purge-home]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --purge-data)
      PURGE_DATA=1
      shift
      ;;
    --purge-home)
      PURGE_HOME=1
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

if [[ ! -d "$PREFIX" ]]; then
  echo "nothing to remove: $PREFIX"
else
  rm -rf "$PREFIX/venv" "$PREFIX/bin" "$PREFIX/requirements.deps.txt" "$PREFIX/VERSION_MANIFEST.json"
  echo "removed venv, wrappers and manifest under $PREFIX"
  if [[ "$PURGE_DATA" -eq 1 ]]; then
    rm -rf "$PREFIX/var"
    echo "removed $PREFIX/var"
    rmdir "$PREFIX" 2>/dev/null || true
  else
    echo "kept $PREFIX/var (pass --purge-data to delete cache, results and the local memory log)"
  fi
fi

if [[ "$PURGE_HOME" -eq 1 ]]; then
  rm -rf "${HOME}/.tradingagents"
  echo "removed ${HOME}/.tradingagents"
else
  echo "kept ${HOME}/.tradingagents if it exists (pass --purge-home to delete it)"
fi
