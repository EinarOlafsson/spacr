"""One run's life in ``spacr.runctx``: open, account for itself, close.

What :func:`run_context` hands the pipeline (id, seed, policy, ledgers,
streams), what it writes into its own JSONL log on the way in and out, and
the two diagnostics seams that must never cost the run: worker labelling for
resource accounting (:meth:`RunContext.register_worker`) and the GUI
performance-logging preference, which must not pull Qt into a headless job.

CPU-only and offline; per-run logs go to a scratch ``SPACR_LOG_DIR``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import types

import numpy as np
import pytest

from spacr import runctx
from spacr.errors import RunLedger
from spacr.runctx import RunContext, read_run_log, run_context, spacr_rng


@pytest.fixture
def scratch_logs(tmp_path, monkeypatch):
    """Per-run logs in a scratch folder, with the root logger open."""
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(runctx, "_ROOT_SEED_SEQUENCE",
                        runctx._ROOT_SEED_SEQUENCE)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    root = logging.getLogger()
    before = root.level
    root.setLevel(logging.DEBUG)
    yield tmp_path / "logs"
    root.setLevel(before)


def _skip_one(run, message="empty plate"):
    for attempt in run.policy.attempts_for("plate1", stage="plate"):
        with attempt:
            raise ValueError(message)


_ACCOUNT_KEYS = ["compute_path", "gpu_memory", "log_path", "module",
                 "on_error", "on_error_attempts", "resource_artifact_id",
                 "resource_log_path", "run_id", "seed", "seed_report",
                 "skipped", "started_utc"]


def test_run_context_hands_out_one_id_seed_and_policy(subtests, scratch_logs,
                                                      monkeypatch):
    """run_context: the RunContext's seed, streams, str, ledgers, dict and env."""
    with subtests.test(msg="a seeded skip run accounts for itself"):
        with run_context("mask", {"random_seed": 4, "on_error": "skip"}) as run:
            assert run.seed == 4
            assert run.random_state() == 4
            assert run.random_state(default=9) == 4
            assert np.array_equal(run.rng("fold0").random(3),
                                  spacr_rng("fold0", seed=4).random(3))
            assert str(run) == f"run {run.run_id} (mask) seed=4 on_error=skip"
            _skip_one(run)
            account = run.to_dict()
        assert sorted(account) == _ACCOUNT_KEYS
        assert (account["run_id"], account["module"], account["seed"],
                account["on_error"], account["on_error_attempts"]) == (
            run.run_id, "mask", 4, "skip", 3)
        skipped = dict(account["skipped"][0])
        skipped.pop("utc")
        assert account["skipped"][0]["utc"]
        assert skipped == {"unit": "plate1", "stage": "plate",
                           "reason": "ValueError: empty plate",
                           "exc_type": "ValueError", "message": "empty plate",
                           "attempts": 1, "run_id": run.run_id}
        assert [s.unit for s in run.skips] == ["plate1"]
        assert account["seed_report"]["seed"] == 4
        assert account["log_path"] == os.path.join(
            str(scratch_logs), "runs", f"{run.run_id}.jsonl")
        assert json.loads(json.dumps(account)) == account

    with subtests.test(msg="an unseeded run hands back the caller's default"):
        with run_context("mask", {"random_seed": None}) as run:
            assert run.seed is None
            assert run.seed_report is None
            assert run.random_state() is None
            assert run.random_state(default=5) == 5
            assert str(run) == f"run {run.run_id} (mask) seed=None on_error=stop"
            assert run.to_dict()["seed_report"] is None

    with subtests.test(msg="an anonymous unlogged run prints as spacr"):
        with run_context("", {}, log=False) as anonymous:
            assert str(anonymous) == (
                f"run {anonymous.run_id} (spacr) seed=42 on_error=stop")
            assert anonymous.log_path == ""
            assert anonymous.log.name == "spacr.run"

    with subtests.test(msg="every ledger the run makes carries its own id"):
        with run_context("mask", {}, run_id="  run-fixed  ") as run:
            assert run.run_id == "run-fixed"
            assert run.ledger.run_id == "run-fixed"
            made = run.new_ledger("segmentation")
            assert (made.name, made.run_id) == ("segmentation", "run-fixed")
            outside = RunLedger("built-elsewhere")
            assert outside.run_id != "run-fixed"
            assert run.adopt(outside) is outside
            assert outside.run_id == "run-fixed"
            assert run.log.name == "spacr.mask"

    with subtests.test(msg="the run id environment variable is put back"):
        monkeypatch.delenv(runctx.RUN_ID_ENV, raising=False)
        with run_context("mask", {}) as run:
            assert os.environ[runctx.RUN_ID_ENV] == run.run_id
            assert runctx.current_run_id() == run.run_id
        assert runctx.RUN_ID_ENV not in os.environ

        monkeypatch.setenv(runctx.RUN_ID_ENV, "an-outer-run")
        with run_context("mask", {}) as inner:
            assert os.environ[runctx.RUN_ID_ENV] == inner.run_id
            assert inner.run_id != "an-outer-run"
        assert os.environ[runctx.RUN_ID_ENV] == "an-outer-run"


def _messages(run_id):
    return [record["message"] for record in read_run_log(run_id)]


def test_run_context_writes_its_own_story_into_its_log(subtests, scratch_logs,
                                                       monkeypatch, caplog):
    """run_context: start, finish, skip summary, failure and traceback lines."""
    with subtests.test(msg="a clean run says it started and finished"):
        with run_context("mask", {"random_seed": 6}) as run:
            run_id = run.run_id
        messages = _messages(run_id)
        assert messages[0] == (
            f"run {run_id} started — module=mask seed=6 on_error=stop")
        assert re.fullmatch(rf"run {run_id} finished in \d+\.\ds — 0 skipped",
                            [m for m in messages if "finished in" in m][0])
        assert not any("NOT in the output" in m for m in messages)

    with subtests.test(msg="a run that skipped something warns on the way out"):
        with run_context("mask", {"on_error": "skip"}) as run:
            run_id = run.run_id
            _skip_one(run)
        messages = _messages(run_id)
        assert (f"run {run_id}: on_error=skip\n"
                "  SKIPPED  : 1 unit(s) — these are NOT in the output\n"
                "    plate1 (plate): ValueError: empty plate") in messages
        assert any(re.fullmatch(
            rf"run {run_id} finished in \d+\.\ds — 1 skipped", m)
            for m in messages)

    with subtests.test(msg="a run that raises says so before it propagates"):
        with pytest.raises(RuntimeError, match="the plate list was empty"):
            with run_context("mask", {}) as run:
                run_id = run.run_id
                raise RuntimeError("the plate list was empty")
        failed = [m for m in _messages(run_id) if "failed after" in m]
        assert len(failed) == 1
        assert re.fullmatch(
            rf"run {run_id} failed after \d+\.\ds — RuntimeError: "
            "the plate list was empty", failed[0])

    with subtests.test(msg="a traceback is carried into the run log"):
        with run_context("mask", {}, log=True) as run:
            try:
                raise ValueError("the plate is empty")
            except ValueError:
                run.log.exception("plate 1 failed")
            run_id = run.run_id
        carried = [r for r in read_run_log(run_id) if "traceback" in r]
        assert [r["message"] for r in carried] == ["plate 1 failed"]
        assert carried[0]["run_id"] == run_id
        assert carried[0]["logger"] == "spacr.mask"
        assert carried[0]["traceback"].endswith(
            "ValueError: the plate is empty\n")

    with subtests.test(msg="a run whose log cannot be opened still runs"):
        def refuse(_run_id):
            raise OSError("the log directory is read-only")

        caplog.clear()
        with monkeypatch.context() as patch:
            patch.setattr(runctx, "run_log_path", refuse)
            with caplog.at_level(logging.WARNING, logger="spacr.runctx"):
                with run_context("mask", {"random_seed": 5}) as run:
                    assert run.run_id
                    assert run.log_path == ""
                    assert run.seed == 5
                    assert run.policy.mode == "stop"
        assert [r.getMessage() for r in caplog.records
                if r.name == "spacr.runctx"] == [
            f"could not open the run log for {run.run_id}: "
            "the log directory is read-only"]


# ---------------------------------------------------------------------------
# RunContext.register_worker
# ---------------------------------------------------------------------------

class _RecordingSampler:
    """Stands in for ``fit_resources._ResourceSampler``'s worker seam."""

    def __init__(self, answer="stamped", explode=False):
        self.stamps = []
        self._answer = answer
        self._explode = explode

    def _register_worker(self, stamp):
        self.stamps.append(stamp)
        if self._explode:
            raise RuntimeError("sampler is not accepting workers")
        return self._answer


def test_register_worker_labels_a_process_or_shrugs(subtests, monkeypatch,
                                                    caplog):
    """register_worker: no sampler, a whole stamp, built stamps, refusals."""
    import psutil

    own_created = float(psutil.Process(os.getpid()).create_time())
    cases = [
        # (id, sampler answer/explode or None, args, kwargs, returned, stamp)
        ("no sampler labels nothing", None, ("cellpose", 3), {}, "", None),
        ("a kind and id are stamped around this process", ("1234:9.5", False),
         ("cellpose", 3), {}, "1234:9.5",
         {"pid": os.getpid(), "create_time": own_created,
          "worker_kind": "cellpose", "worker_id": "3"}),
        ("a missing worker id is stamped as None", ("x", False),
         ("saver",), {}, "x",
         {"pid": os.getpid(), "create_time": own_created,
          "worker_kind": "saver", "worker_id": "None"}),
        ("a supplied creation time is not looked up again", ("x", False),
         ("trial", 17), {"pid": 4242, "create_time": 111.5}, "x",
         {"pid": 4242, "create_time": 111.5, "worker_kind": "trial",
          "worker_id": "17"}),
        ("a pid psutil cannot read costs only the creation time",
         ("x", False), ("ghost", 1), {"pid": 2 ** 30}, "x",
         {"pid": 2 ** 30, "create_time": None, "worker_kind": "ghost",
          "worker_id": "1"}),
        ("a sampler that refuses the stamp costs the label", (None, True),
         ("cellpose", 1), {"pid": 7, "create_time": 1.0}, "",
         {"pid": 7, "create_time": 1.0, "worker_kind": "cellpose",
          "worker_id": "1"}),
    ]
    for case_id, sampler_spec, args, kwargs, returned, stamp in cases:
        with subtests.test(msg=case_id):
            sampler = (None if sampler_spec is None
                       else _RecordingSampler(*sampler_spec))
            run = RunContext(run_id="r7", module="mask",
                             _resource_sampler=sampler)
            caplog.clear()
            with monkeypatch.context() as patch:
                if "create_time" in kwargs:
                    def _refuse(_pid):
                        raise AssertionError("psutil must not be consulted")
                    patch.setattr(psutil, "Process", _refuse)
                with caplog.at_level(logging.DEBUG, logger="spacr.mask"):
                    assert run.register_worker(*args, **kwargs) == returned
            if sampler is not None:
                assert sampler.stamps == [stamp]
            refused = [r.getMessage() for r in caplog.records
                       if r.name == "spacr.mask"]
            assert refused == (["could not label resource worker"]
                               if sampler_spec == (None, True) else [])

    with subtests.test(msg="a worker's own stamp is passed through whole"):
        sampler = _RecordingSampler()
        run = RunContext(run_id="r7", module="mask", _resource_sampler=sampler)
        given = {"pid": 4242, "create_time": 111.5,
                 "worker_kind": "fastq-saver", "worker_id": "7"}
        assert run.register_worker(given) == "stamped"
        assert sampler.stamps == [given]
        assert sampler.stamps[0] is not given


# ---------------------------------------------------------------------------
# _performance_logging_preference
# ---------------------------------------------------------------------------

def test_performance_logging_preference_never_drags_qt_in(subtests,
                                                          monkeypatch):
    """_performance_logging_preference: setting, environment, Qt absent/present."""
    def _never():
        raise AssertionError("Qt preferences must not be imported")

    def _corrupt():
        raise RuntimeError("preferences file is not readable")

    cases = [
        # (id, settings, env value or None, Qt loaded, preference fn, expected)
        ("an explicit setting wins over everything", {"performance_logging":
                                                       "summary"},
         "detailed", True, _never, "summary"),
        ("an explicit None setting is answered as None",
         {"performance_logging": None}, None, True, _never, None),
        ("the environment declines to answer", {}, "detailed", True, _never,
         None),
        ("a headless run never imports Qt preferences", {}, None, False,
         _never, None),
        ("with Qt loaded the GUI preference is read", {}, None, True,
         lambda: "detailed", "detailed"),
        ("an unreadable GUI preference falls back to None", {}, None, True,
         _corrupt, None),
    ]
    for case_id, settings, env, qt_loaded, preference, expected in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            if env is None:
                patch.delenv("SPACR_PERFORMANCE_LOG", raising=False)
            else:
                patch.setenv("SPACR_PERFORMANCE_LOG", env)
            if qt_loaded:
                patch.setitem(sys.modules, "PySide6.QtCore",
                              types.ModuleType("PySide6.QtCore"))
            else:
                patch.delitem(sys.modules, "PySide6.QtCore", raising=False)
            store = types.SimpleNamespace(get_performance_logging=preference)
            patch.setitem(sys.modules, "spacr.qt",
                          types.SimpleNamespace(preferences=store))
            patch.setitem(sys.modules, "spacr.qt.preferences", store)
            assert runctx._performance_logging_preference(settings) == expected
