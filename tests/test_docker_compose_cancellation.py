import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from pier.environments.docker.docker import DockerEnvironment


class _TestDockerEnvironment(DockerEnvironment):
    @property
    def _docker_compose_paths(self) -> list[Path]:
        return []


class _FakeComposeProcess:
    def __init__(self, *, require_kill: bool = False) -> None:
        self.returncode = None
        self.require_kill = require_kill
        self.started = asyncio.Event()
        self.terminated = 0
        self.killed = 0
        self._calls = 0

    async def communicate(self):
        self._calls += 1
        if self._calls == 1:
            self.started.set()
            await asyncio.Event().wait()
        if self.require_kill and not self.killed:
            await asyncio.Event().wait()
        self.returncode = -9 if self.killed else -15
        return b"", b""

    def terminate(self) -> None:
        self.terminated += 1

    def kill(self) -> None:
        self.killed += 1


def _environment(tmp_path: Path) -> DockerEnvironment:
    environment = _TestDockerEnvironment.__new__(_TestDockerEnvironment)
    environment.session_id = "task__abcdefg"
    environment.environment_dir = tmp_path
    environment.environment_name = "task"
    environment._env_vars = SimpleNamespace(to_env_dict=lambda **_kwargs: {})
    environment._compose_task_env = {}
    environment._persistent_env = {}
    environment._windows_container_name = None
    return environment


@pytest.mark.asyncio
async def test_cancelled_compose_exec_is_reaped_before_cancellation_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeComposeProcess()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    task = asyncio.create_task(
        _environment(tmp_path)._run_docker_compose_command(["exec", "main", "agent"])
    )
    await process.started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated == 1
    assert process.killed == 0
    assert process.returncode == -15


@pytest.mark.asyncio
async def test_cancelled_compose_exec_escalates_when_term_does_not_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeComposeProcess(require_kill=True)

    async def create_process(*_args, **_kwargs):
        return process

    async def immediate_timeout(awaitable, timeout):
        if timeout == 5:
            awaitable.close()
            raise asyncio.TimeoutError
        return await awaitable

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(asyncio, "wait_for", immediate_timeout)
    task = asyncio.create_task(
        _environment(tmp_path)._run_docker_compose_command(["exec", "main", "agent"])
    )
    await process.started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated == 1
    assert process.killed == 1
    assert process.returncode == -9
