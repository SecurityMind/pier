import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pier.agents.installed.codex import Codex
from pier.agents.installed.codex_checkpoint import (
    CheckpointError,
    CheckpointIncompatibleError,
    append_event,
    load_manifest,
    new_manifest,
    read_events,
    safe_artifact_path,
    snapshot_script,
    write_manifest,
)


def test_manifest_is_atomic_private_and_rejects_sensitive_fields(tmp_path: Path):
    path = tmp_path / "checkpoint" / "checkpoint.json"
    manifest = new_manifest(
        assignment_id="a1", model="gpt-test", base_commit="abc123"
    )
    write_manifest(path, manifest)

    assert load_manifest(path) == manifest
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert not path.with_suffix(".json.tmp").exists()

    manifest["access_token"] = "must-not-land"
    with pytest.raises(CheckpointError, match="sensitive"):
        write_manifest(path, manifest)


def test_checkpoint_events_are_structured_and_ignore_corruption(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoint"
    append_event(checkpoint_dir, "paused", failure_type="ConnectionError")
    with (checkpoint_dir / "events.jsonl").open("a") as handle:
        handle.write("not-json\n")

    assert read_events(checkpoint_dir) == [
        {
            "at": read_events(checkpoint_dir)[0]["at"],
            "detail": {"failure_type": "ConnectionError"},
            "event": "paused",
        }
    ]


def test_checkpoint_artifacts_cannot_escape_through_paths_or_symlinks(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    outside = tmp_path / "auth.json"
    outside.write_text("must-not-read")
    (checkpoint / "sessions").symlink_to(outside)
    with pytest.raises(CheckpointError, match="symlink"):
        safe_artifact_path(checkpoint, "sessions")
    with pytest.raises(CheckpointError, match="escapes"):
        safe_artifact_path(checkpoint, "../auth.json")


def test_snapshot_preserves_repo_progress_sessions_and_excludes_credentials(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    subprocess.run(["git", "-C", str(work), "config", "user.email", "test@example.invalid"])
    subprocess.run(["git", "-C", str(work), "config", "user.name", "test"])
    (work / "tracked.txt").write_text("base\n")
    subprocess.run(["git", "-C", str(work), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(work), "commit", "-qm", "base"], check=True)
    base = subprocess.check_output(
        ["git", "-C", str(work), "rev-parse", "HEAD"], text=True
    ).strip()
    (work / "tracked.txt").write_text("changed\n")
    (work / "new.py").write_text("print('kept')\n")
    (work / ".env").write_text("TOKEN=must-not-land\n")

    codex_home = tmp_path / "codex-home"
    (codex_home / "sessions" / "2026").mkdir(parents=True)
    (codex_home / "sessions" / "2026" / "rollout.jsonl").write_text("{}\n")
    (codex_home / "auth.json").write_text('{"access_token":"must-not-land"}\n')
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "base_commit").write_text(base + "\n")
    script = checkpoint / "snapshot.sh"
    script.write_text(
        snapshot_script(
            workdir=str(work),
            checkpoint_dir=str(checkpoint),
            codex_home=str(codex_home),
            interval_sec=30,
        )
    )
    subprocess.run(["sh", str(script), "--once"], check=True)

    assert "tracked.txt" in (checkpoint / "workspace.patch").read_text()
    assert "tracked.txt" in (checkpoint / "progress-summary.txt").read_text()
    assert (checkpoint / "sessions" / "2026" / "rollout.jsonl").is_file()
    assert not (checkpoint / "auth.json").exists()
    assert "must-not-land" not in "".join(
        path.read_text(errors="ignore") for path in checkpoint.rglob("*") if path.is_file()
    )
    with tarfile.open(checkpoint / "untracked.tar.gz") as archive:
        names = {name.removeprefix("./") for name in archive.getnames()}
    assert "new.py" in names
    assert ".env" not in names


def test_snapshot_rejects_secret_patch_and_omits_secret_session(tmp_path: Path):
    work = tmp_path / "work with ' quote"
    work.mkdir()
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    subprocess.run(["git", "-C", str(work), "config", "user.email", "test@example.invalid"])
    subprocess.run(["git", "-C", str(work), "config", "user.name", "test"])
    (work / "tracked.txt").write_text("base\n")
    subprocess.run(["git", "-C", str(work), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(work), "commit", "-qm", "base"], check=True)
    base = subprocess.check_output(
        ["git", "-C", str(work), "rev-parse", "HEAD"], text=True
    ).strip()
    secret = "sk-proj-abc123def456ghi789xyz"
    (work / "tracked.txt").write_text(secret + "\n")
    codex_home = tmp_path / "codex-home"
    (codex_home / "sessions").mkdir(parents=True)
    (codex_home / "sessions" / "rollout.jsonl").write_text(secret + "\n")
    checkpoint = tmp_path / "checkpoint with ' quote"
    checkpoint.mkdir()
    (checkpoint / "base_commit").write_text(base + "\n")
    script = checkpoint / "snapshot.sh"
    script.write_text(snapshot_script(
        workdir=str(work), checkpoint_dir=str(checkpoint),
        codex_home=str(codex_home), interval_sec=30,
    ))
    subprocess.run(["sh", str(script), "--once"], check=True)

    assert (checkpoint / "invalid-secret").is_file()
    assert not (checkpoint / "workspace.patch").exists()
    assert (checkpoint / "session-omitted-sensitive").is_file()
    assert not (checkpoint / "sessions").exists()
    assert secret not in "".join(
        path.read_text(errors="ignore") for path in checkpoint.rglob("*") if path.is_file()
    )


@pytest.mark.asyncio
async def test_restore_rejects_a_changed_task_base(tmp_path: Path):
    previous_dir = tmp_path / "previous"
    previous_dir.mkdir()
    previous = new_manifest(
        assignment_id="a1", model="gpt-test", base_commit="old-base"
    )
    write_manifest(previous_dir / "checkpoint.json", previous)

    agent = Codex(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-test",
        checkpoint_enabled=True,
        checkpoint_assignment_id="a1",
    )
    agent.exec_as_agent = AsyncMock(return_value=SimpleNamespace(stdout=""))

    with pytest.raises(CheckpointIncompatibleError, match="task base changed"):
        await agent._restore_checkpoint(
            environment=object(),
            env={},
            previous_dir=previous_dir,
            previous=previous,
            base_commit="new-base",
        )


def test_root_thread_is_recovered_from_previous_output(tmp_path: Path):
    checkpoint = tmp_path / "agent" / "checkpoint"
    checkpoint.mkdir(parents=True)
    manifest = new_manifest(
        assignment_id="a1", model="gpt-test", base_commit="base"
    )
    write_manifest(checkpoint / "checkpoint.json", manifest)
    (checkpoint.parent / "codex.txt").write_text(
        json.dumps({"type": "thread.started", "thread_id": "root-thread"}) + "\n"
    )
    agent = Codex(logs_dir=checkpoint.parent, model_name="openai/gpt-test")

    assert agent._checkpoint_root_thread_id(checkpoint) == "root-thread"
