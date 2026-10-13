"""Storage planning keeps its safety boundary when filesystem calls fail."""
from __future__ import annotations

import logging
import os
import runpy
import warnings
from pathlib import Path

import pytest

from spacr import logging_util
from spacr import run_journal as journal


def _unresolvable(monkeypatch, bad):
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == bad:
            raise OSError("path cannot be resolved")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)


def test_running_run_and_open_log_protection_survive_unreadable_paths(
        tmp_path, monkeypatch):
    bad = tmp_path / "bad.log"
    _unresolvable(monkeypatch, bad)

    def missing_run():
        raise OSError("run state disappeared")

    monkeypatch.setattr(journal, "current_run", missing_run)
    logger = logging.getLogger("spacr.test_ci_unreadable_log")
    handler = logging.NullHandler()
    handler.baseFilename = str(bad)
    logger.addHandler(handler)
    try:
        protected = journal._protected_storage()
    finally:
        logger.removeHandler(handler)
    assert str(bad) in protected
    assert journal._is_protected(bad, protected)


def test_prune_planning_skips_an_entry_whose_type_cannot_be_read(
        tmp_path, monkeypatch):
    candidate = tmp_path / "spacr-20200101.log"
    candidate.write_text("old log")
    monkeypatch.setattr(journal, "_prune_root", lambda _kind: tmp_path)
    monkeypatch.setattr(journal, "_protected_storage", set)

    def unreadable(_kind, _entry):
        raise OSError("entry type unavailable")

    monkeypatch.setattr(journal, "_is_prunable_name", unreadable)
    plan = journal._prune_plan("logs", 7, 0)
    assert plan["delete"] == []
    assert candidate.read_text() == "old log"


def test_prune_refuses_an_entry_that_cannot_be_resolved(
        tmp_path, monkeypatch):
    candidate = tmp_path / "spacr-20200101.log"
    candidate.write_text("keep")
    _unresolvable(monkeypatch, candidate)
    monkeypatch.setattr(journal, "_prune_root", lambda _kind: tmp_path)
    monkeypatch.setattr(journal, "_protected_storage", set)
    plan = {"kind": "logs", "cutoff": 2, "delete": [(str(candidate), 4)]}

    deleted, freed, refused = journal._prune(plan)

    assert (deleted, freed) == (0, 0)
    assert refused == [f"{candidate.name}: path cannot be resolved"]
    assert candidate.read_text() == "keep"


def test_failed_run_deletion_stays_a_refusal_in_the_prune_result(
        tmp_path, monkeypatch):
    candidate = tmp_path / "old-run"
    candidate.mkdir()
    monkeypatch.setattr(journal, "_prune_root", lambda _kind: tmp_path)
    monkeypatch.setattr(journal, "_protected_storage", set)
    monkeypatch.setattr(journal, "_tree_stats", lambda _entry: (4, 0))
    monkeypatch.setattr(journal, "delete_runs",
                        lambda _entries: (0, ["old-run: busy"]))
    plan = {"kind": "run_folders", "cutoff": 2,
            "delete": [(str(candidate), 4)]}

    assert journal._prune(plan) == (0, 0, ["old-run: busy"])
    assert candidate.is_dir()


def test_unreadable_log_types_are_excluded_from_clear_plans(
        tmp_path, monkeypatch):
    candidate = tmp_path / "r1.jsonl"
    candidate.write_text("keep")
    monkeypatch.setattr(journal, "_prune_root", lambda _kind: tmp_path)

    def unreadable(_kind, _entry):
        raise OSError("entry type unavailable")

    monkeypatch.setattr(journal, "_is_prunable_name", unreadable)
    plan = journal._clear_logs_plan()
    assert plan["groups"]["run_logs"] == []
    assert candidate.read_text() == "keep"


def test_a_log_that_becomes_unreadable_is_refused_during_clear(
        tmp_path, monkeypatch):
    candidate = tmp_path / "spacr-20200101.log"
    candidate.write_text("keep")
    monkeypatch.setattr(journal, "_prune_root", lambda _kind: tmp_path)
    monkeypatch.setattr(journal, "_protected_storage", set)
    original = Path.stat

    def unreadable(path, *args, **kwargs):
        if path == candidate:
            raise OSError("log cannot be read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", unreadable)
    plan = {"groups": {"daily": [(str(candidate), 4, False)]}}
    assert journal._log_group(candidate) is None
    monkeypatch.setattr(journal, "_log_group", lambda _entry: "daily")

    assert journal._clear_logs(plan) == (
        0, 0, 0, [f"{candidate.name}: log cannot be read"])
    assert candidate.read_text() == "keep"


def test_cache_clear_reports_one_busy_child_and_removes_a_free_one(
        tmp_path, monkeypatch):
    folder = tmp_path / "cache"
    folder.mkdir()
    busy, free = folder / "busy.bin", folder / "free.bin"
    busy.write_text("keep")
    free.write_text("remove")
    monkeypatch.setattr(journal, "_cache_folder", lambda _key: folder)
    monkeypatch.setattr(journal, "_cache_refusal", lambda _folder: None)
    original = Path.unlink

    def cannot_remove(path, *args, **kwargs):
        if path == busy:
            raise OSError("file is in use")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", cannot_remove)
    removed, refused = journal._clear_cache("torch")
    assert removed == 1 and refused == ["busy.bin: file is in use"]
    assert busy.read_text() == "keep" and not free.exists()


def test_a_protected_cache_cannot_be_relocated(tmp_path, monkeypatch):
    source = tmp_path / "current"
    source.mkdir()
    (source / "model.bin").write_bytes(b"model")
    monkeypatch.delenv("TORCH_HOME", raising=False)
    monkeypatch.setattr(journal, "_read_cache_locations", dict)
    monkeypatch.setattr(journal, "_cache_folder", lambda _key: source)
    monkeypatch.setattr(journal, "_cache_refusal",
                        lambda _folder: "cache is in use")

    with pytest.raises(ValueError, match="cache is in use"):
        journal._relocate_cache("torch", tmp_path / "elsewhere")
    assert (source / "model.bin").read_bytes() == b"model"
    assert not (tmp_path / "elsewhere").exists()


def test_an_unresolvable_cache_is_refused_before_any_removal(
        tmp_path, monkeypatch):
    folder = tmp_path / "cache"
    folder.mkdir()
    _unresolvable(monkeypatch, folder)
    assert journal._cache_refusal(folder) == "path cannot be resolved"
    assert folder.is_dir()


def test_reapplying_cache_locations_preserves_an_existing_override(monkeypatch):
    monkeypatch.setattr(journal, "_read_cache_locations", lambda: {
        "CI_CACHE_ONE": "/tmp/one", "CI_CACHE_TWO": "/tmp/two"})
    monkeypatch.setenv("CI_CACHE_ONE", "/tmp/owned-outside-spacr")
    monkeypatch.delenv("CI_CACHE_TWO", raising=False)

    assert journal._apply_cache_locations() == 1
    assert os.environ["CI_CACHE_ONE"] == "/tmp/owned-outside-spacr"
    assert os.environ["CI_CACHE_TWO"] == "/tmp/two"


def test_unreadable_cache_locations_do_not_stop_journal_import(
        monkeypatch, caplog):
    def unavailable():
        raise RuntimeError("cache location unavailable")

    monkeypatch.setattr(logging_util, "_spacr_home", unavailable)
    with warnings.catch_warnings(), caplog.at_level(
            logging.DEBUG, logger="spacr.run_journal"):
        warnings.simplefilter("ignore", RuntimeWarning)
        module = runpy.run_module("spacr.run_journal",
                                  run_name="spacr._run_journal_failure_probe",
                                  alter_sys=True)

    assert module["MANIFEST_SCHEMA_VERSION"] == journal.MANIFEST_SCHEMA_VERSION
    assert "Relocated caches could not be applied" in caplog.text
