"""Persistent, credential-free checkpoints for Codex coding trials.

The checkpoint lives below ``/logs/agent`` which is a host bind mount for
Docker trials.  It intentionally stores only repository state and Codex
session files.  Authentication is injected again by the caller on resume and
is never copied into the checkpoint manifest or workspace archive.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINT_SCHEMA_VERSION = 1
SENSITIVE_KEY_PARTS = ("token", "secret", "password", "credential", "api_key", "auth")
_CHECKPOINT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_FIXED_ARTIFACTS = {
    "workspace_patch": "workspace.patch",
    "untracked_archive": "untracked.tar.gz",
    "sessions_dir": "sessions",
    "events_file": "events.jsonl",
}


class CheckpointError(RuntimeError):
    """Base error for checkpoint validation and restoration."""


class CheckpointIncompatibleError(CheckpointError):
    """The checkpoint cannot be safely restored by this Pier build."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(part in normalized for part in SENSITIVE_KEY_PARTS):
                return True
            if _contains_sensitive_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_sensitive_key(child) for child in value)
    return False


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    """Atomically persist a private checkpoint manifest."""
    if _contains_sensitive_key(data):
        raise CheckpointError("checkpoint manifest contains a sensitive field name")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"invalid checkpoint manifest: {path}") from exc
    if not isinstance(data, dict):
        raise CheckpointError("checkpoint manifest must be an object")
    if data.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointIncompatibleError(
            f"unsupported checkpoint schema {data.get('schema_version')!r}"
        )
    if _contains_sensitive_key(data):
        raise CheckpointError("checkpoint manifest contains a sensitive field name")
    for required in ("checkpoint_id", "assignment_id", "phase", "created_at"):
        if not isinstance(data.get(required), str) or not data[required]:
            raise CheckpointError(f"checkpoint missing {required}")
    if not _CHECKPOINT_ID_RE.fullmatch(data["checkpoint_id"]):
        raise CheckpointError("checkpoint_id has an invalid format")
    generation = data.get("resume_generation", 0)
    if not isinstance(generation, int) or generation < 0:
        raise CheckpointError("checkpoint resume_generation is invalid")
    for key, expected in _FIXED_ARTIFACTS.items():
        if data.get(key, expected) != expected:
            raise CheckpointError(f"checkpoint {key} must be {expected!r}")
    return data


def safe_artifact_path(checkpoint_dir: Path, relative: str) -> Path:
    """Resolve a manifest artifact without allowing symlink/path escape."""
    root = checkpoint_dir.resolve()
    path = checkpoint_dir / relative
    if path.is_symlink():
        raise CheckpointError(f"checkpoint artifact is a symlink: {relative}")
    resolved = path.resolve()
    if root not in resolved.parents:
        raise CheckpointError(f"checkpoint artifact escapes its directory: {relative}")
    if path.is_dir() and any(child.is_symlink() for child in path.rglob("*")):
        raise CheckpointError(f"checkpoint artifact contains a symlink: {relative}")
    return path


def new_manifest(
    *,
    assignment_id: str,
    model: str | None,
    base_commit: str,
    task_id: str | None = None,
    effort: str | None = None,
    resume_generation: int = 0,
    checkpoint_id: str | None = None,
    resume_count: int = 0,
) -> dict[str, Any]:
    if not isinstance(resume_generation, int) or resume_generation < 0:
        raise CheckpointError("resume_generation must be a non-negative integer")
    now = utc_now()
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id or uuid.uuid4().hex,
        "assignment_id": assignment_id,
        "phase": "running",
        "created_at": now,
        "updated_at": now,
        "last_heartbeat": now,
        "model": model,
        "task_id": task_id,
        "effort": effort,
        "base_commit": base_commit,
        "resume_generation": resume_generation,
        "root_thread_id": None,
        "resume_count": resume_count,
        "workspace_patch": "workspace.patch",
        "untracked_archive": "untracked.tar.gz",
        "sessions_dir": "sessions",
        "events_file": "events.jsonl",
    }


def update_manifest(path: Path, **changes: Any) -> dict[str, Any]:
    data = load_manifest(path)
    data.update(changes)
    data["updated_at"] = utc_now()
    write_manifest(path, data)
    return data


def append_event(checkpoint_dir: Path, event: str, **detail: Any) -> None:
    payload = {"at": utc_now(), "event": event, "detail": detail}
    if _contains_sensitive_key(payload):
        raise CheckpointError("checkpoint event contains a sensitive field name")
    checkpoint_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = checkpoint_dir / "events.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def read_events(checkpoint_dir: Path) -> list[dict[str, Any]]:
    path = checkpoint_dir / "events.jsonl"
    if path.is_symlink() or not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and not _contains_sensitive_key(item):
            events.append(item)
    return events


def snapshot_script(
    *, workdir: str, checkpoint_dir: str, codex_home: str, interval_sec: int
) -> str:
    """Return a POSIX script that atomically snapshots repository progress.

    The patch is relative to the task's initial commit, so committed and
    uncommitted tracked changes are both retained.  Untracked source files are
    archived, with credential-like filenames excluded defensively.
    """
    interval = max(10, min(int(interval_sec), 300))
    return f"""#!/bin/sh
set -eu
umask 077
workdir={shlex.quote(workdir)}
checkpoint_dir={shlex.quote(checkpoint_dir)}
codex_home={shlex.quote(codex_home)}
interval={interval}
mkdir -p "$checkpoint_dir"
# High-precision credential shapes. A secret-bearing repository patch cannot
# be rewritten without corrupting it, so mark the checkpoint invalid instead.
# Session files are optional: omit them on a hit and let resume degrade to a
# fresh Codex session over the still-restored workspace.
secret_re='(sk-(ant-|proj-)?[A-Za-z0-9_-]{{16,}}|ghp_[A-Za-z0-9]{{20,}}|github_pat_[A-Za-z0-9_]{{20,}}|gAAAAA[A-Za-z0-9_-]{{40,}}|eyJ[A-Za-z0-9_-]{{10,}}[.][A-Za-z0-9_-]{{10,}}[.][A-Za-z0-9_-]{{10,}})'
snapshot_once() {{
  base=$(cat "$checkpoint_dir/base_commit")
  git -C "$workdir" diff --binary "$base" -- > "$checkpoint_dir/workspace.patch.tmp"
  if LC_ALL=C grep -aEq "$secret_re" "$checkpoint_dir/workspace.patch.tmp"; then
    rm -f "$checkpoint_dir/workspace.patch.tmp" "$checkpoint_dir/workspace.patch"
    printf 'credential-shaped content detected in workspace patch\n' > "$checkpoint_dir/invalid-secret"
  else
    mv "$checkpoint_dir/workspace.patch.tmp" "$checkpoint_dir/workspace.patch"
  fi
  git -C "$workdir" status --short > "$checkpoint_dir/progress-summary.txt.tmp"
  printf '\nChanged files:\n' >> "$checkpoint_dir/progress-summary.txt.tmp"
  git -C "$workdir" diff --stat "$base" -- >> "$checkpoint_dir/progress-summary.txt.tmp"
  mv "$checkpoint_dir/progress-summary.txt.tmp" "$checkpoint_dir/progress-summary.txt"
  git -C "$workdir" ls-files --others --exclude-standard -z | \
    tar -C "$workdir" --null -czf "$checkpoint_dir/untracked.tar.gz.tmp" \
      --exclude='.env' --exclude='.env.*' --exclude='auth.json' \
      --exclude='*.pem' --exclude='*.key' --exclude='credentials*' \
      --exclude='token*' --exclude='secret*' --exclude='password*' \
      --exclude='*.p12' --exclude='*.pfx' --files-from=-
  if tar -xOzf "$checkpoint_dir/untracked.tar.gz.tmp" 2>/dev/null | \
      LC_ALL=C grep -aEq "$secret_re"; then
    rm -f "$checkpoint_dir/untracked.tar.gz.tmp" "$checkpoint_dir/untracked.tar.gz"
    printf 'credential-shaped content detected in untracked files\n' > "$checkpoint_dir/invalid-secret"
  else
    mv "$checkpoint_dir/untracked.tar.gz.tmp" "$checkpoint_dir/untracked.tar.gz"
  fi
  if [ -d "$codex_home/sessions" ]; then
    if LC_ALL=C grep -aErq "$secret_re" "$codex_home/sessions"; then
      rm -rf "$checkpoint_dir/sessions" "$checkpoint_dir/sessions.tmp"
      printf 'session omitted because it contained credential-shaped content\n' \
        > "$checkpoint_dir/session-omitted-sensitive"
    else
      rm -rf "$checkpoint_dir/sessions.tmp"
      cp -R "$codex_home/sessions" "$checkpoint_dir/sessions.tmp"
      rm -rf "$checkpoint_dir/sessions"
      mv "$checkpoint_dir/sessions.tmp" "$checkpoint_dir/sessions"
      rm -f "$checkpoint_dir/session-omitted-sensitive"
    fi
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$checkpoint_dir/last_heartbeat.tmp"
  mv "$checkpoint_dir/last_heartbeat.tmp" "$checkpoint_dir/last_heartbeat"
}}
if [ "${{1:-}}" = "--once" ]; then
  snapshot_once
  exit 0
fi
while :; do
  snapshot_once || true
  sleep "$interval"
done
"""
