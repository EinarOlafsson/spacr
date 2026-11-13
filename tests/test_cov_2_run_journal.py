"""The run journal's refusal to fail a run over its own record-keeping.

Every path here is one where the journal cannot do its job: git is not there,
a package list will not enumerate, a settings value is not a path, a file
cannot be hashed, the manifest cannot be written. The rule the whole module is
built on is that none of these may take down the pipeline that was actually
producing results -- and that none of them may be silent either, because a
manifest that quietly omits an output reads as a run that did not make one.

The other half is reading the journal back: a history scan walks folders other
processes are writing to and deleting, and it has to survive all of it.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from spacr import run_journal as RJ


# ---------------------------------------------------------------------------
# the environment snapshot
# ---------------------------------------------------------------------------

def test_a_checkout_git_cannot_read_reports_no_commit(monkeypatch):
    """A non-zero ``git rev-parse`` means there is no commit to record.

    An installed wheel is not a checkout, and a checkout with no commits is
    not one either. Recording the empty string as a commit hash would put a
    provenance field in the manifest that names nothing.
    """
    def _fail(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=128,
                                           stdout="", stderr="not a git repo")

    monkeypatch.setattr(RJ.subprocess, "run", _fail)
    assert RJ._git_hash() is None


def test_git_being_absent_altogether_reports_no_commit(monkeypatch):
    """No ``git`` on PATH is a machine, not a failure.

    Clusters routinely have no git. Letting the ``FileNotFoundError`` out
    would stop a run before it started, over a field nobody needs.
    """
    def _explode(*_args, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(RJ.subprocess, "run", _explode)
    assert RJ._git_hash() is None


def test_a_package_list_that_will_not_enumerate_is_empty_not_fatal(caplog):
    """A broken site-packages costs the package list, never the run.

    The environment snapshot is taken while the run is starting. A
    distribution with unreadable metadata is a real thing on shared machines,
    and it must degrade to "no packages recorded" with a line saying why.

    The list is cached for the whole process, so the cache is emptied on both
    sides of the swap: leaving an empty list in it would give every later run
    in this session an environment block with no packages in it.
    """
    real = importlib.metadata.distributions

    def _explode():
        raise OSError("site-packages is not readable")

    try:
        importlib.metadata.distributions = _explode
        RJ._installed_packages.cache_clear()
        with caplog.at_level("WARNING"):
            packages = RJ._installed_packages()
    finally:
        importlib.metadata.distributions = real
        RJ._installed_packages.cache_clear()

    assert packages == {}
    assert any("Could not enumerate installed packages" in r.message
               for r in caplog.records)
    assert RJ._installed_packages(), "the real list comes back afterwards"


# ---------------------------------------------------------------------------
# atomic writes
# ---------------------------------------------------------------------------

def test_the_temp_file_refusing_to_go_does_not_lose_the_write(tmp_path,
                                                              monkeypatch):
    """Cleanup of the scratch file must never undo a completed write.

    ``os.replace`` has already put the content where it belongs by the time
    the temporary name is unlinked. An error raised there would propagate out
    of the manifest writer and report a successful run as failed.
    """
    real_unlink = Path.unlink

    def _refuse(self, *args, **kwargs):
        if self.name.startswith("."):
            raise OSError(13, "Permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _refuse)

    target = tmp_path / "manifest.json"
    RJ._atomic_write_text(target, '{"ok": true}')

    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


# ---------------------------------------------------------------------------
# walking settings
# ---------------------------------------------------------------------------

def test_a_settings_dict_that_contains_itself_is_walked_once():
    """A cycle in the settings must not become an infinite recursion.

    Panels build nested settings by reference, and a dict that ends up
    holding itself is a bug in the caller, not a reason for the journal to
    hang while trying to record it.
    """
    inner = {"src": "/data"}
    inner["self"] = inner
    values = dict(RJ._walk_setting_values(inner))

    assert values["src"] == "/data"
    assert not any(key.count("self") > 1 for key in values), \
        "the cycle was followed more than once"


def test_a_settings_list_that_contains_itself_is_walked_once():
    """The same cycle guard, for the sequence branch.

    A list is walked by a different arm of the same function, so a guard on
    only one of them leaves the other able to recurse forever.
    """
    values = ["a"]
    values.append(values)
    walked = dict(RJ._walk_setting_values({"channels": values}))

    assert walked["channels[0]"] == "a"
    assert len(walked) < 5, f"the cycle was expanded: {walked}"


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------

def test_numpy_not_being_loaded_records_no_numpy_state(monkeypatch):
    """A run that never imported numpy has no numpy state to record.

    ``None`` is the honest answer, and it must be written: leaving the key
    out entirely makes a manifest from a numpy-free run look like one where
    the capture failed.
    """
    monkeypatch.setitem(sys.modules, "numpy", None)

    seeds = RJ.extract_seeds({})

    assert seeds["numpy_random_state_sha256"] is None
    assert "numpy_random_state_error" not in seeds


def test_a_random_state_that_will_not_be_read_is_reported_as_an_error(
        monkeypatch):
    """Both random-state captures record their own failure rather than raise.

    They run inside ``open_run``, before the pipeline does anything. An
    exception here would mean a library in a strange state stopped a run that
    would otherwise have worked.
    """
    numpy_stub = types.SimpleNamespace(
        random=types.SimpleNamespace(
            get_state=lambda: (_ for _ in ()).throw(RuntimeError("no state"))))
    torch_stub = types.SimpleNamespace(
        initial_seed=lambda: (_ for _ in ()).throw(RuntimeError("no seed")))
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    seeds = RJ.extract_seeds({})

    assert "no state" in seeds["numpy_random_state_error"]
    assert "no seed" in seeds["torch_seed_error"]
    assert "numpy_random_state_sha256" not in seeds


# ---------------------------------------------------------------------------
# file records
# ---------------------------------------------------------------------------

@pytest.mark.posix_only
def test_a_file_that_cannot_be_read_gets_no_record(tmp_path):
    """A file that stats but will not open produces no provenance record.

    A record with no digest in it would claim the file was fingerprinted.
    The caller turns the ``None`` into a warning that names the file.
    """
    blocked = tmp_path / "locked.bin"
    blocked.write_bytes(b"secret")
    os.chmod(blocked, 0o000)
    try:
        assert RJ._file_record(blocked) is None
    finally:
        os.chmod(blocked, 0o644)


def test_a_missing_file_has_no_inventory_signature(tmp_path):
    """The before/after signature of a file that is not there is ``None``.

    The baseline is taken over paths the settings named, which routinely do
    not exist yet -- an output folder is the ordinary case.
    """
    assert RJ._inventory_signature(tmp_path / "not-yet") is None
