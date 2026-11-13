"""Table-driven tests of home-folder storage and secret files in ``spacr.run_journal``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: planning and clearing the log folders, pruning plans
that meet unreadable or undeletable entries, moving a disk cache (and every
refusal that guards the move), and writing the notification-secret file
atomically.  Every case gets its own temporary home, log folder and cache
folder through environment variables, so nothing real is read or deleted.
"""
from __future__ import annotations

import json
import logging
import os
import re
import types
from pathlib import Path

import pytest

from spacr import run_journal as RJ
from tests.refused_access import only_the_owner_can_read_the_file
from tests.test_home_storage_prune_caches_purge import _aged, home  # noqa: F401


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _case_dir(tmp_path, name):
    """A fresh per-case directory under ``tmp_path``."""
    d = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", name)
    d.mkdir()
    return d


def _fresh_home(mp, folder):
    """Point HOME, the cache root and the log folder at ``folder``."""
    mp.setenv("HOME", str(folder))
    mp.setenv("USERPROFILE", str(folder))
    mp.setenv("XDG_CACHE_HOME", str(folder / ".cache"))
    mp.setenv("SPACR_LOG_DIR", str(folder / ".spacr" / "logs"))
    mp.delenv("SPACR_RUN_ID", raising=False)
    mp.delenv("TORCH_HOME", raising=False)
    mp.delenv("SPACR_NEWS_CACHE", raising=False)
    return folder / ".spacr" / "logs"


def _empty_groups(**filled):
    """The ``groups`` mapping of a clear-logs plan, empty except ``filled``."""
    groups = {group: [] for group in RJ._LOG_GROUPS}
    groups.update(filled)
    return groups


@pytest.fixture
def open_log():
    """Attach a FileHandler for a path; detached and closed afterwards."""
    attached = []

    def attach(path):
        handler = logging.FileHandler(path)
        logger = logging.getLogger(f"spacr.test_storage_open_{len(attached)}")
        logger.addHandler(handler)
        attached.append((logger, handler))
        return logger, handler

    yield attach
    for logger, handler in attached:
        logger.removeHandler(handler)
        handler.close()


# ---------------------------------------------------------------------------
# _clear_logs_plan / _clear_logs
# ---------------------------------------------------------------------------

def test_clearing_logs_removes_closed_logs_and_refuses_everything_else(
        subtests, tmp_path, monkeypatch, open_log, home):  # noqa: F811
    """The plan lists each log once; the clear re-checks and names every refusal."""

    def closed_logs(logs, mp):
        files = {"daily": _aged(logs / "spacr-20200101.log", 60, size=100),
                 "crash": _aged(logs / "spacr-crash.log", 60, size=200),
                 "verbose": _aged(logs / "spacr-debug.log.1", 60, size=300),
                 "run_logs": _aged(logs / "runs" / "done123.jsonl", 60, size=400)}
        (logs / "keep.txt").write_text("not a log", encoding="utf-8")
        groups = _empty_groups(**{g: [(str(p), s, False)] for (g, p), s in zip(
            files.items(), (100, 200, 300, 400))})
        return groups, None, (4, 0, 1000, []), lambda: (
            not any(p.exists() for p in files.values())
            and (logs / "keep.txt").read_text(encoding="utf-8") == "not a log")

    def open_handler(logs, mp):
        active = _aged(logs / "spacr-debug.log", 60, size=300)
        logger, handler = open_log(active)

        def kept():
            if active.stat().st_size != 0:
                return False
            handler.emit(logging.LogRecord(logger.name, logging.INFO, __file__,
                                           0, "logger still writes", (), None))
            return active.stat().st_size > 0

        return (_empty_groups(verbose=[(str(active), 300, True)]), None,
                (0, 1, 300, []), kept)

    def open_handler_refused(logs, mp):
        active = _aged(logs / "spacr-debug.log", 60, size=300)
        open_log(active)

        def refuse(path, length):
            raise OSError("read-only log")

        return (_empty_groups(verbose=[(str(active), 300, True)]),
                lambda: mp.setattr(RJ.os, "truncate", refuse),
                (0, 0, 0, ["spacr-debug.log: read-only log"]),
                lambda: active.read_bytes() == b"x" * 300)

    def run_log_became_active(logs, mp):
        active = _aged(logs / "runs" / "live123.jsonl", 100, size=50)
        return (_empty_groups(run_logs=[(str(active), 50, False)]),
                lambda: mp.setenv("SPACR_RUN_ID", "live123"),
                (0, 0, 0, ["live123.jsonl: in use"]), active.exists)

    def busy(logs, mp):
        log = _aged(logs / "spacr-20200101.log", 400, size=10)
        real_unlink = Path.unlink

        def refuse(self, *args, **kwargs):
            if self == log:
                raise PermissionError("log file is busy")
            return real_unlink(self, *args, **kwargs)

        return (_empty_groups(daily=[(str(log), 10, False)]),
                lambda: mp.setattr(Path, "unlink", refuse),
                (0, 0, 0, ["spacr-20200101.log: log file is busy"]), log.exists)

    def replaced_by_link(logs, mp):
        log = _aged(logs / "spacr-20200101.log", 400, size=10)
        outside = logs.parent.parent / "precious.log"
        outside.write_text("keep me", encoding="utf-8")

        def swap():
            log.unlink()
            log.symlink_to(outside)

        return (_empty_groups(daily=[(str(log), 10, False)]), swap,
                (0, 0, 0, ["spacr-20200101.log: not a log any more"]),
                lambda: outside.read_text(encoding="utf-8") == "keep me")

    def vanished(logs, mp):
        log = _aged(logs / "spacr-20200101.log", 400, size=10)
        return (_empty_groups(daily=[(str(log), 10, False)]), log.unlink,
                (0, 0, 0, ["spacr-20200101.log: not a log any more"]),
                lambda: not log.exists())

    def unreadable_folder(logs, mp):
        mp.setattr(RJ, "_prune_root", lambda kind: logs.parent / "missing")
        return _empty_groups(), None, (0, 0, 0, []), lambda: True

    def unreadable_run_log_type(logs, mp):
        candidate = logs / "runs" / "r1.jsonl"
        candidate.parent.mkdir(parents=True)
        candidate.write_text("keep", encoding="utf-8")

        def unreadable(_kind, _entry):
            raise OSError("entry type unavailable")

        mp.setattr(RJ, "_is_prunable_name", unreadable)
        return (_empty_groups(), None, (0, 0, 0, []),
                lambda: candidate.read_text(encoding="utf-8") == "keep")

    cases = [
        ("recognised closed logs are deleted, other files kept", closed_logs),
        ("a log this session holds open is emptied, not deleted", open_handler),
        ("an open log that refuses truncation is named", open_handler_refused),
        ("a run log that became active is kept", run_log_became_active),
        ("a log that cannot be unlinked is named", busy),
        ("a log gone since the plan is refused", vanished),
        ("an unreadable log folder plans nothing", unreadable_folder),
        ("a run log whose type cannot be read is not planned",
         unreadable_run_log_type),
    ]
    if os.name != "nt":
        cases.append(("a log replaced by a link is refused", replaced_by_link))
    for case_id, build in cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as mp:
                logs = _fresh_home(mp, _case_dir(tmp_path, case_id))
                groups, between, result, kept = build(logs, mp)
                plan = RJ._clear_logs_plan()
                assert plan == {"groups": groups}
                if between is not None:
                    between()
                assert RJ._clear_logs(plan) == result
                assert kept()


# ---------------------------------------------------------------------------
# _prune_plan / _prune
# ---------------------------------------------------------------------------

def test_prune_plans_skip_unreadable_entries_and_name_failed_deletions(
        subtests, tmp_path, monkeypatch, home):  # noqa: F811
    """Unreadable roots and entries plan nothing; an undeletable entry is a refusal."""
    with subtests.test(msg="an unreadable storage root plans nothing"):
        with monkeypatch.context() as mp:
            mp.setattr(RJ, "_prune_root", lambda kind: tmp_path / "not-a-folder")
            plan = RJ._prune_plan("logs", 7, 0, now=1_000_000.0)
        assert plan == {"kind": "logs", "root": str(tmp_path / "not-a-folder"),
                        "count": 0, "total": 0,
                        "cutoff": 1_000_000.0 - 7 * 86400.0, "delete": []}

    with subtests.test(msg="an entry whose type cannot be read is skipped"):
        folder = _case_dir(tmp_path, "type")
        candidate = folder / "spacr-20200101.log"
        candidate.write_text("old log", encoding="utf-8")

        def unreadable(_kind, _entry):
            raise OSError("entry type unavailable")

        with monkeypatch.context() as mp:
            mp.setattr(RJ, "_prune_root", lambda _kind: folder)
            mp.setattr(RJ, "_protected_storage", set)
            mp.setattr(RJ, "_is_prunable_name", unreadable)
            plan = RJ._prune_plan("logs", 7, 0)
        assert (plan["count"], plan["delete"]) == (0, [])
        assert candidate.read_text(encoding="utf-8") == "old log"

    with subtests.test(msg="an entry that cannot be deleted is refused"):
        with monkeypatch.context() as mp:
            logs = _fresh_home(mp, _case_dir(tmp_path, "busy"))
            old = _aged(logs / "spacr-20200101.log", 400, size=10)
            plan = RJ._prune_plan("logs", 7, 0)
            assert plan["delete"] == [(str(old), 10)]
            real_unlink = Path.unlink

            def refuse(self, *a, **k):
                if self.name == old.name:
                    raise OSError("busy")
                return real_unlink(self, *a, **k)

            mp.setattr(Path, "unlink", refuse)
            assert RJ._prune(plan) == (0, 0, ["spacr-20200101.log: busy"])
        assert old.exists()


# ---------------------------------------------------------------------------
# _relocate_cache (and the refusals it relies on)
# ---------------------------------------------------------------------------

def test_relocating_a_cache_moves_it_or_refuses_with_the_reason(
        subtests, tmp_path, monkeypatch, home):  # noqa: F811
    """A cache moves into ``parent/spacr-<key>``; every guard raises ValueError and moves nothing."""

    def no_setting(h, mp):
        return "not-a-cache", h / "elsewhere", (
            "not-a-cache: this cache has no setting to move it with")

    def protected(h, mp):
        source = h / "current"
        source.mkdir()
        (source / "model.bin").write_bytes(b"model")
        mp.setattr(RJ, "_read_cache_locations", dict)
        mp.setattr(RJ, "_cache_folder", lambda _key: source)
        mp.setattr(RJ, "_cache_refusal", lambda _folder: "cache is in use")
        return "torch", h / "elsewhere", "cache is in use"

    def full_destination(h, mp):
        full = h / "elsewhere" / "spacr-news"
        full.mkdir(parents=True)
        (full / "keep").write_text("x", encoding="utf-8")
        return "news", h / "elsewhere", f"{full} already holds files"

    def inside_source(h, mp):
        source = h / ".cache" / "torch"
        source.mkdir(parents=True)
        (source / "weights.pt").write_bytes(b"model")
        return "torch", source / "nested", (
            f"{source / 'nested' / 'spacr-torch'} overlaps {source}")

    def set_outside(h, mp):
        mp.setenv("TORCH_HOME", str(h / "owned-outside-spacr"))
        return "torch", h / "elsewhere", (
            "TORCH_HOME is set outside spaCR; change it there")

    refusals = [
        ("a cache without a setting", no_setting),
        ("a protected cache", protected),
        ("a destination that already holds files", full_destination),
        ("a destination inside the source", inside_source),
        ("a cache whose variable was set outside spaCR", set_outside),
    ]
    for case_id, build in refusals:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as mp:
                h = _case_dir(tmp_path, case_id)
                _fresh_home(mp, h)
                key, parent, message = build(h, mp)
                before = sorted(p.relative_to(h) for p in h.rglob("*"))
                with pytest.raises(ValueError) as caught:
                    RJ._relocate_cache(key, parent)
                assert str(caught.value) == message
                assert sorted(p.relative_to(h) for p in h.rglob("*")) == before

    with subtests.test(msg="a cache that does not exist yet gets a fresh folder"):
        with monkeypatch.context() as mp:
            h = _case_dir(tmp_path, "fresh")
            _fresh_home(mp, h)
            moved = RJ._relocate_cache("torch", h / "fresh")
            assert moved == h / "fresh" / "spacr-torch"
            assert moved.is_dir() and list(moved.iterdir()) == []
            assert os.environ["TORCH_HOME"] == str(moved)
            assert json.loads((h / ".spacr" / "cache_locations.json").read_text(
                encoding="utf-8")) == {"TORCH_HOME": str(moved)}

    with subtests.test(msg="an existing cache replaces an empty destination"):
        with monkeypatch.context() as mp:
            h = _case_dir(tmp_path, "reuse")
            _fresh_home(mp, h)
            source = h / ".cache" / "torch"
            source.mkdir(parents=True)
            (source / "weights.pt").write_bytes(b"model")
            target = h / "bigdisk" / "spacr-torch"
            target.mkdir(parents=True)
            assert RJ._relocate_cache("torch", h / "bigdisk") == target
            assert (target / "weights.pt").read_bytes() == b"model"
            assert not source.exists()

    with subtests.test(msg="the refusal guards name what they protect"):
        with monkeypatch.context() as mp:
            h = _case_dir(tmp_path, "guards")
            _fresh_home(mp, h)
            spacr_home = RJ._spacr_home().resolve()
            assert RJ._cache_refusal(RJ._spacr_home()) == (
                f"{spacr_home}: refusing spaCR's own home folder")
            assert RJ._cache_refusal(h) == (
                f"{h.resolve()}: refusing a home or top-level folder")
            assert RJ._cache_refusal(h / ".cache" / "torch") is None
            assert RJ._clear_cache("news") == (0, [])


# ---------------------------------------------------------------------------
# _write_notify_secret_file / _load_notify_secret
# ---------------------------------------------------------------------------

def test_the_secret_file_is_written_whole_or_not_at_all(
        subtests, tmp_path, monkeypatch):
    """Only non-empty secrets are kept, atomically; a failed write leaves no temporary."""

    def raise_(exc):
        def _raise(*_a, **_k):
            raise exc
        return _raise

    def vanish(source, target):
        Path(source).unlink()
        raise OSError("disk went away")

    cases = [
        # (id, values, existing file?, patch, expected exception, expected content)
        ("non-empty values are written, empty ones dropped",
         {"a": "1", "b": ""}, False, None, None, {"a": "1"}),
        ("a new write replaces the old file", {"b": "2"}, True, None, None,
         {"b": "2"}),
        ("nothing left removes the file", {"a": ""}, True, None, None, None),
        ("nothing to write on a missing file is not an error", {}, False, None,
         None, None),
        ("a failed encode removes its temporary", {"a": "1"}, False,
         ("json", "dump", raise_(ValueError("cannot encode"))),
         (ValueError, "cannot encode"), None),
        ("a refused replace removes its temporary", {"a": "1"}, False,
         ("os", "replace", raise_(OSError("read-only"))),
         (OSError, "read-only"), None),
        ("a temporary already gone is not a second error", {"a": "1"}, False,
         ("os", "replace", vanish), (OSError, "disk went away"), None),
    ]
    for case_id, values, existing, patch, raises, content in cases:
        with subtests.test(msg=case_id):
            d = _case_dir(tmp_path, case_id)
            path = d / "secrets.json"
            if existing:
                path.write_text('{"old": "x"}', encoding="utf-8")
            with monkeypatch.context() as mp:
                if patch is not None:
                    mp.setattr(getattr(RJ, patch[0]), patch[1], patch[2])
                if raises is not None:
                    with pytest.raises(raises[0], match=raises[1]):
                        RJ._write_notify_secret_file(values, path)
                else:
                    RJ._write_notify_secret_file(values, path)
            if content is None:
                assert list(d.iterdir()) == []
            else:
                assert [p.name for p in d.iterdir()] == ["secrets.json"]
                assert json.loads(path.read_text(encoding="utf-8")) == content
                assert only_the_owner_can_read_the_file(path)

    with subtests.test(msg="an empty keyring answer falls back to the file"):
        d = _case_dir(tmp_path, "keyring")
        ring = types.SimpleNamespace(get_password=lambda service, name: "")
        with monkeypatch.context() as mp:
            mp.setattr(RJ, "_notify_keyring", lambda: ring)
            mp.setattr(RJ, "_notify_secrets_path", lambda: d / "s.json")
            RJ._write_notify_secret_file({"slack_webhook": "from-file"})
            assert RJ._load_notify_secret("slack_webhook") == "from-file"
            assert RJ._load_notify_secret("ntfy_topic") == ""
