#!/usr/bin/env bash
# Pack a source bundle without .git, virtualenvs, or secret files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${1:-/opt/cursor/artifacts/tradingagents-research}"
NAME="tradingagents-research-0.7.1-9fa1196"
STAGE="$(mktemp -d)"
mkdir -p "$DEST_DIR"

tar \
  --exclude-vcs \
  --exclude='.git' \
  --exclude='.github' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='*.pyc' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='.env' \
  --exclude='*.egg-info' \
  --exclude='dist' \
  --exclude='build' \
  --exclude='data' \
  --exclude='results' \
  -C "$ROOT" \
  -cf - . | tar -C "$STAGE" -xf -

rm -rf "$STAGE/.git" "$STAGE/.github"
if find "$STAGE" \( -name '.env' -o -name '*.pem' -o -name 'credentials.json' -o -name 'secrets.json' \) | grep -q .; then
  echo "refusing to pack a secret-like file" >&2
  find "$STAGE" \( -name '.env' -o -name '*.pem' -o -name 'credentials.json' -o -name 'secrets.json' \) >&2
  exit 1
fi

COPYFILE_DISABLE=1 tar -C "$STAGE" -czf "$DEST_DIR/$NAME.tar.gz" .
rm -rf "$STAGE"
echo "$DEST_DIR/$NAME.tar.gz"
