"""Headless cancellation contract and durable-boundary tests."""
from __future__ import annotations

import json
import sys
import threading
import time

import pytest

from spacr.cancellation import (
    CancellationToken,
    PipelineCancelled,
    cancellation_requested,
    checkpoint,
    current_token,
    installed_token,
)


def test_token_is_idempotent_and_preserves_first_reason():
    token = CancellationToken()
    assert not token.cancelled
    assert token.cancel("first reason")
    assert not token.cancel("second reason")
    assert token.cancelled
    assert token.reason == "first reason"
    with pytest.raises(PipelineCancelled, match="first reason"):
        token.checkpoint()


def test_installed_token_is_thread_local_and_restored():
    outer = CancellationToken("outer")
    inner = CancellationToken("inner")
    seen = []

    with installed_token(outer):
        assert current_token() is outer

        def inspect_other_thread():
            seen.append(current_token())
            seen.append(cancellation_requested())
            checkpoint()

        thread = threading.Thread(target=inspect_other_thread)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        with installed_token(inner):
            assert current_token() is inner
        assert current_token() is outer

    assert current_token() is None
    assert seen == [None, False]
    checkpoint()  # plain CLI/API calls remain a no-op


def test_checkpoint_mark_is_durable_before_cancellation(tmp_path):
    from spacr.checkpoint import CheckpointStore

    store = CheckpointStore(
        tmp_path / "state.json",
        workflow="audit",
        signature={"input": "plate-a"},
        boundary="field",
    )
    token = CancellationToken()
    token.cancel("stop after field")

    with installed_token(token):
        with pytest.raises(PipelineCancelled, match="stop after field"):
            store.mark("A01/F001", {"output": "complete"})

    document = json.loads((tmp_path / "state.json").read_text())
    assert document["completed"]["A01/F001"]["output"] == "complete"
    assert document["status"] == "running"


def test_batch_cancellation_leaves_unstarted_jobs_resumable(tmp_path):
    from spacr.batch import (
        Job,
        Queue,
        STATUS_NOT_RUN,
        STATUS_PENDING,
        run_queue,
    )

    token = CancellationToken()
    queue = Queue(name="cancel-audit")
    first = queue.add(Job(
        module="mask", settings={"src": str(tmp_path), "cell_channel": 0},
        id="first",
    ), validate=False)
    second = queue.add(Job(
        module="mask", settings={"src": str(tmp_path), "cell_channel": 0},
        id="second",
    ), validate=False)

    def runner(*_args):
        token.cancel("queue stop")
        return 0

    with installed_token(token):
        with pytest.raises(PipelineCancelled, match="queue stop"):
            run_queue(queue, runner=runner, force=True, echo=False)

    # The completed first job is settled; the second was never touched.
    assert first.status == "success"
    assert second.status in {STATUS_PENDING, STATUS_NOT_RUN}


def test_batch_subprocess_is_terminated_on_cancellation(
        tmp_path, monkeypatch):
    from spacr import batch

    token = CancellationToken()
    job = batch.Job(
        module="mask", settings={"src": str(tmp_path), "cell_channel": 0},
        id="slow",
    )
    monkeypatch.setattr(
        batch,
        "job_command",
        lambda *_args, **_kwargs: [
            sys.executable, "-c", "import time; time.sleep(30)"],
    )
    timer = threading.Timer(0.2, lambda: token.cancel("stop child"))
    timer.start()
    started = time.monotonic()
    try:
        with installed_token(token):
            with pytest.raises(PipelineCancelled, match="stop child"):
                batch.subprocess_runner(
                    job, "", str(tmp_path / "slow.log"))
    finally:
        timer.cancel()

    assert time.monotonic() - started < 5
    assert "child process stopped" in (
        tmp_path / "slow.log").read_text(encoding="utf-8")
