"""Durable partial records for a research run.

SIGTERM and ordinary exceptions flush status from the last checkpoint.
SIGKILL is not caught: only lines already fsynced remain, and the status
may still say the run is in progress.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

_TIME = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:?\d{2})?)?")
_UNAVAILABLE = ("NO_DATA_AVAILABLE", "DATA_UNAVAILABLE")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def quoted_times(text: str) -> list[str]:
    """Timestamps that already appear in the vendor text. Nothing is invented."""
    return _TIME.findall(text or "")[:12]


def install_source_log(log_path: Path, route):
    """Wrap the real vendor router and record both usable and failed calls."""

    def wrapped(method, *args, **kwargs):
        started = _now()
        try:
            result = route(method, *args, **kwargs)
        except Exception as exc:
            append_jsonl(log_path, {
                "method": method,
                "available": False,
                "outcome": "error",
                "observed_at": started,
                "finished_at": _now(),
                "source_times_quoted": [],
                "error_type": type(exc).__name__,
                "detail": str(exc)[:300],
            })
            raise
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        unavailable = any(flag in text for flag in _UNAVAILABLE)
        append_jsonl(log_path, {
            "method": method,
            "available": not unavailable,
            "outcome": "unavailable" if unavailable else "available",
            "observed_at": started,
            "finished_at": _now(),
            "source_times_quoted": [] if unavailable else quoted_times(text),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_head": text[:240],
        })
        return result

    return wrapped


def source_summary(log_path: Path) -> dict:
    rows = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return {
        "path": str(log_path.name),
        "calls": len(rows),
        "available": sum(1 for row in rows if row.get("available") is True),
        "unavailable": sum(1 for row in rows if row.get("outcome") == "unavailable"),
        "errors": sum(1 for row in rows if row.get("outcome") == "error"),
        "rows": rows,
    }


def write_role_files(role_dir: Path, state: dict) -> list[str]:
    written = []
    if not isinstance(state, dict):
        return written
    role_dir.mkdir(parents=True, exist_ok=True)
    for key, value in state.items():
        if isinstance(value, str) and value.strip():
            atomic_write(role_dir / f"{key}.md", value)
            written.append(key)
    return written


def write_response_checkpoint(output: Path, slot: int, text: str, finish_reason: str | None) -> None:
    body = text or ""
    atomic_write(output / "checkpoint" / f"slot-{slot}.md", body)
    append_jsonl(output / "checkpoint" / "responses.jsonl", {
        "at": _now(),
        "slot": slot,
        "finish_reason": finish_reason,
        "chars": len(body),
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    })
