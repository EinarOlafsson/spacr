"""Table-driven tests of reading the run journal back in ``spacr.run_journal``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: the run-folder listing, ``recent_runs``,
``search_runs`` (warnings, settings fallback, sorting, a folder vanishing
mid-scan), ``journal_totals`` (per-app tallies, distinct model digests in
both manifest shapes, the incremental cache), the settings readers and
``hash_file`` with the lock check built on it.  Every case lays out its own
run folders by hand under ``tmp_path`` with the journal root redirected there.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from spacr import run_journal as RJ
from tests.refused_access import refused

ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _case_dir(tmp_path, name):
    """A fresh per-case directory under ``tmp_path``."""
    d = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", name)
    d.mkdir()
    return d


def _write_run(root, name, manifest, *, settings="{}", settings_csv=None,
               log_text=None):
    """Lay out one run folder by hand, the way a finished run leaves one.

    ``manifest`` is a dict (written as JSON), a string (written verbatim) or
    ``None`` (no manifest at all).
    """
    directory = root / name
    directory.mkdir(parents=True)
    if isinstance(manifest, dict):
        (directory / "manifest.json").write_text(json.dumps(manifest),
                                                 encoding="utf-8")
    elif manifest is not None:
        (directory / "manifest.json").write_text(manifest, encoding="utf-8")
    if settings is not None:
        (directory / "settings.json").write_text(settings, encoding="utf-8")
    if settings_csv is not None:
        (directory / "settings.csv").write_text(settings_csv, encoding="utf-8")
    if log_text is not None:
        (directory / "log.txt").write_text(log_text, encoding="utf-8")
    return directory


def _run_name(index, app="mask"):
    """A run-folder name ``index`` minutes after 2026-01-01."""
    when = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
    return f"{when:%Y-%m-%d_%H%M%S}_tag{index:04d}__{app}"


def _started(index):
    """The ISO start time matching ``_run_name(index)``."""
    return (datetime(2026, 1, 1, tzinfo=timezone.utc)
            + timedelta(minutes=index)).isoformat()


def _totals(total=0, mask=0, measure=0, classify=0, models=0):
    """The exact dict ``journal_totals`` returns."""
    return {"total_runs": total, "mask_runs": mask, "measure_runs": measure,
            "classify_runs": classify, "models_recorded": models}


# ---------------------------------------------------------------------------
# _run_dir_names
# ---------------------------------------------------------------------------

def test_run_dir_names_lists_the_run_folders_and_never_raises(subtests, tmp_path):
    """Folders (and links to folders) are runs; files, absence and refusal give no names."""
    a, b = "2026-08-30_120000_aaaa__mask", "2026-08-30_130000_bbbb__measure"

    def two_folders(root):
        for name in (a, b):
            (root / name).mkdir()
        return root

    def loose_files(root):
        (root / a).mkdir()
        (root / ".journal_totals.json").write_text("{}", encoding="utf-8")
        (root / "notes.txt").write_text("hello", encoding="utf-8")
        return root

    def link(root):
        real = root / "elsewhere" / a
        real.mkdir(parents=True)
        journal = root / "journal"
        journal.mkdir()
        (journal / a).symlink_to(real, target_is_directory=True)
        return journal

    cases = [
        ("every folder is reported", two_folders, [a, b]),
        ("a loose file is not a run", loose_files, [a]),
        ("an empty journal is an empty list", lambda root: root, []),
        ("a journal that is not there is an empty list",
         lambda root: root / "no-such-directory", []),
        ("a file in place of the journal is an empty list",
         lambda root: (root / "f").write_text("x") and root / "f", []),
    ]
    try:
        probe = tmp_path / "probe"
        probe.symlink_to(tmp_path, target_is_directory=True)
        probe.unlink()
        cases.append(("a link to a folder still counts", link, [a]))
    except (OSError, NotImplementedError):
        pass
    for case_id, build, expected in cases:
        with subtests.test(msg=case_id):
            root = build(_case_dir(tmp_path, case_id))
            assert sorted(RJ._run_dir_names(root)) == expected

    if not ROOT:
        with subtests.test(msg="a journal that cannot be read is an empty list"):
            blocked = _case_dir(tmp_path, "blocked")
            (blocked / a).mkdir()
            with refused(blocked, 0o000, windows="read"):
                assert RJ._run_dir_names(blocked) == []


# ---------------------------------------------------------------------------
# recent_runs
# ---------------------------------------------------------------------------

def test_recent_runs_returns_the_newest_readable_runs_first(
        subtests, tmp_path, monkeypatch, caplog):
    """Runs are ordered by start time, bounded by ``limit``, and unreadable ones logged."""
    hourly = [f"2026-08-30_{h:02d}0000_x{h}__mask" for h in range(5)]

    def five_hourly(root):
        for h, name in enumerate(hourly):
            _write_run(root, name, {"app_key": "mask", "status": "done",
                                    "start_utc": f"2026-08-30T{h:02d}:00:00",
                                    "elapsed_s": 1.0})

    def twelve(root):
        for i in range(12):
            _write_run(root, _run_name(i), {"app_key": "mask",
                                            "status": "success",
                                            "start_utc": _started(i)})

    def good_and_bad(root):
        _write_run(root, "2026-08-31_010000_good__mask",
                   {"app_key": "mask", "status": "done",
                    "start_utc": "2026-08-31T01:00:00"})
        _write_run(root, "2026-08-31_020000_bad__mask", "{ this is not json")
        _write_run(root, "2026-08-31_030000_starting__mask", None)

    def same_second_by_start(root):
        _write_run(root, "2026-08-31_010000_b__measure",
                   {"app_key": "measure", "status": "failed",
                    "start_utc": "2026-08-31T01:00:00.200000"})
        _write_run(root, "2026-08-31_010000_a__mask",
                   {"app_key": "mask", "status": "success",
                    "start_utc": "2026-08-31T01:00:00.900000"})

    def one_start_unreadable(root):
        _write_run(root, "2026-08-31_010000_a__mask",
                   {"app_key": "mask", "status": "success",
                    "start_utc": "2026-08-31T01:00:00+00:00"})
        _write_run(root, "2026-08-31_020000_b__mask",
                   {"app_key": "mask", "status": "success",
                    "start_utc": "not a date"})
        _write_run(root, "2026-08-31_030000_c__mask",
                   {"app_key": "mask", "status": "success",
                    "start_utc": "2026-08-31T03:00:00+00:00"})

    cases = [
        # (id, layout, limit, expected folder names, expected log fragment)
        ("an unreadable start time sorts as the oldest", one_start_unreadable,
         5, ["2026-08-31_030000_c__mask", "2026-08-31_010000_a__mask",
             "2026-08-31_020000_b__mask"], None),
        ("newest first, bounded by the limit", five_hourly, 3,
         hourly[::-1][:3], None),
        ("limit None returns the whole journal", twelve, None,
         [_run_name(i) for i in range(11, -1, -1)], None),
        ("a limit of 10 returns ten", twelve, 10,
         [_run_name(i) for i in range(11, 1, -1)], None),
        ("an unreadable manifest is skipped and logged, a missing one quietly",
         good_and_bad, 10, ["2026-08-31_010000_good__mask"],
         "skipping run folder 2026-08-31_020000_bad__mask: its manifest.json "
         "could not be read"),
        ("the start time, not the folder name, orders the same second",
         same_second_by_start, 5,
         ["2026-08-31_010000_a__mask", "2026-08-31_010000_b__measure"], None),
        ("no journal gives no runs", None, 5, [], None),
    ]
    for case_id, layout, limit, expected, fragment in cases:
        with subtests.test(msg=case_id):
            root = _case_dir(tmp_path, case_id) / "runs"
            if layout is not None:
                root.mkdir()
                layout(root)
            caplog.clear()
            with monkeypatch.context() as mp, caplog.at_level("WARNING"):
                mp.setattr(RJ, "runs_root", lambda: root)
                got = RJ.recent_runs(limit)
            assert [r["dir"].name for r in got] == expected
            if fragment is not None:
                assert fragment in caplog.text

    with subtests.test(msg="each entry carries the manifest fields"):
        root = _case_dir(tmp_path, "fields")
        same_second_by_start(root)
        with monkeypatch.context() as mp:
            mp.setattr(RJ, "runs_root", lambda: root)
            got = RJ.recent_runs(1)
        manifest = {"app_key": "mask", "status": "success",
                    "start_utc": "2026-08-31T01:00:00.900000"}
        assert got == [{"dir": root / "2026-08-31_010000_a__mask",
                        "app_key": "mask", "status": "success",
                        "start_utc": "2026-08-31T01:00:00.900000",
                        "elapsed_s": None, "manifest": manifest}]


# ---------------------------------------------------------------------------
# search_runs
# ---------------------------------------------------------------------------

def test_search_runs_surfaces_every_warning_and_survives_damaged_runs(
        subtests, tmp_path, monkeypatch, caplog):
    """History records carry exact warnings, settings and order, whatever the folders hold."""
    base = {"app_key": "mask", "status": "success",
            "start_utc": "2026-01-01T00:00:00+00:00"}
    cases = [
        # (id, [(name, manifest, extra _write_run kwargs)], {run_id: warnings})
        ("a warnings string arrives whole beside the provenance list",
         [("r", dict(base, warnings="one thing went wrong",
                     provenance_warnings=["and one more"]), {})],
         {"r": ["one thing went wrong", "and one more"]}),
        ("empty warnings blocks contribute nothing",
         [("r", dict(base, warnings="", provenance_warnings=None), {})],
         {"r": []}),
        ("an empty warnings list contributes nothing",
         [("r", dict(base, warnings=[]), {})], {"r": []}),
        ("a non-string scalar warning is stringified",
         [("r", dict(base, warnings=3), {})], {"r": ["3"]}),
        ("repeated and blank list entries are kept once",
         [("r", dict(base, warnings=["a", "", "a"], provenance_warnings=["a", "b"]),
           {})], {"r": ["a", "b"]}),
        ("a legacy run has its warnings read out of the log",
         [("r", base, {"log_text": "starting up\nWARNING: two fields had no "
                                   "mask\nall done\nwarn: low memory  \n"})],
         {"r": ["WARNING: two fields had no mask", "warn: low memory"]}),
        ("a run with a warnings block does not mine its log",
         [("r", dict(base, warnings=[]), {"log_text": "WARNING: x\n"})],
         {"r": []}),
        ("a folder with no settings says so",
         [("r", dict(base, warnings=[]), {"settings": None})],
         {"r": ["no settings.json / settings.csv in run folder"]}),
        ("a manifest that is not an object says so",
         [("r", "[1, 2]", {})],
         {"r": ["manifest.json is list, not an object"]}),
    ]
    for case_id, runs, expected in cases:
        with subtests.test(msg=case_id):
            root = _case_dir(tmp_path, case_id)
            for name, manifest, extra in runs:
                _write_run(root, name, manifest, **extra)
            with monkeypatch.context() as mp:
                mp.setattr(RJ, "runs_root", lambda: root)
                got = {r["run_id"]: r["warnings"] for r in RJ.search_runs()}
            assert got == expected

    with subtests.test(msg="an unparsable settings.json falls back to its CSV twin"):
        root = _case_dir(tmp_path, "settings")
        manifest = dict(base, warnings=[])
        _write_run(root, _run_name(0), manifest, settings="{not json at all")
        _write_run(root, _run_name(1), manifest, settings="{not json at all",
                   settings_csv="Key,Value\nsrc,/data/plate1\n")
        _write_run(root, _run_name(2), manifest, settings="{not json at all",
                   settings_csv="Key,Value\nsrc,a\x00b\n")
        _write_run(root, _run_name(3), manifest, settings="[1, 2]")
        with monkeypatch.context() as mp:
            mp.setattr(RJ, "runs_root", lambda: root)
            got = {r["run_id"]: (r["settings"], r["warnings"])
                   for r in RJ.search_runs()}
        assert got == {
            _run_name(0): ({}, ["settings.json unreadable (JSONDecodeError)"]),
            _run_name(1): ({"src": "/data/plate1"},
                           ["settings.json unreadable (JSONDecodeError)",
                            "fell back to settings.csv"]),
            _run_name(2): ({}, ["settings.json unreadable (JSONDecodeError)",
                                "settings.csv unreadable (Error)"]),
            _run_name(3): ({}, ["settings is list, not a dict"]),
        }

    if not ROOT:
        with subtests.test(msg="a log that cannot be read is reported"):
            root = _case_dir(tmp_path, "locked-log")
            folder = _write_run(root, "r", base, log_text="WARNING: something\n")
            with monkeypatch.context() as mp, refused(folder / "log.txt", 0o000,
                                                      windows="read"):
                mp.setattr(RJ, "runs_root", lambda: root)
                record = RJ.search_runs()[0]
            assert record["warnings"] == ["log.txt unreadable (PermissionError)"]

    with subtests.test(msg="naive and aware start times sort together, newest first"):
        root = _case_dir(tmp_path, "sorting")
        _write_run(root, "run-naive", dict(base, start_utc="2026-01-01T00:00:00"))
        _write_run(root, "run-aware",
                   dict(base, start_utc="2026-02-01T00:00:00+00:00"))
        _write_run(root, "run-undated", dict(base, start_utc="not a date"))
        with monkeypatch.context() as mp:
            mp.setattr(RJ, "runs_root", lambda: root)
            assert [r["run_id"] for r in RJ.search_runs()] == [
                "run-aware", "run-naive", "run-undated"]
            assert [r["run_id"] for r in RJ.search_runs(limit=1)] == ["run-aware"]

    with subtests.test(msg="a run folder deleted mid-scan still appears"):
        root = _case_dir(tmp_path, "vanish")
        _write_run(root, "run-gone", base)
        real_read = RJ._read_run_record

        def read_then_vanish(reference):
            record = real_read(reference)
            shutil.rmtree(Path(reference), ignore_errors=True)
            return record

        with monkeypatch.context() as mp:
            mp.setattr(RJ, "runs_root", lambda: root)
            mp.setattr(RJ, "_read_run_record", read_then_vanish)
            assert [r["run_id"] for r in RJ.search_runs()] == ["run-gone"]
        assert not (root / "run-gone").exists()

    with subtests.test(msg="a journal that is not there is empty and logged"):
        missing = tmp_path / "never-made"
        caplog.clear()
        with monkeypatch.context() as mp, caplog.at_level("WARNING"):
            mp.setattr(RJ, "runs_root", lambda: missing)
            assert RJ.search_runs() == []
        assert f"Could not enumerate run history at {missing}: " in caplog.text


# ---------------------------------------------------------------------------
# journal_totals
# ---------------------------------------------------------------------------

def test_journal_totals_counts_runs_per_app_and_distinct_model_digests(
        subtests, tmp_path, monkeypatch, caplog):
    """Every readable manifest is a run; mask/measure/classify are tallied; digests counted once."""
    cases = [
        # (id, [(name, manifest)], expected totals, expected log fragment)
        ("mixed apps, shared digests, legacy list, a loose file and a corrupt run",
         [("r1__mask", {"app_key": "mask",
                        "model_hashes": {"cyto": "cyto.pth:abc123"}}),
          ("r2__mask", {"app_key": "mask",
                        "model_hashes": {"cyto": "cyto.pth:abc123",
                                         "nuc": "nuc.pth:def456"}}),
          ("r3__measure", {"app_key": "measure"}),
          ("r4__classify", {"app_key": "classify",
                            "models": [{"sha256": "legacy999"}]}),
          ("r5__mask", "{not json")],
         _totals(4, mask=2, measure=1, classify=1, models=3),
         "run folder r5__mask is not counted: its manifest.json could not be read"),
        ("a legacy model_hashes list is ignored, its models list honoured",
         [(_run_name(0), {"app_key": "mask", "model_hashes": ["cyto2:deadbeef"],
                          "models": [{"sha256": "aaa111"}]}),
          (_run_name(1), {"app_key": "mask", "model_hashes": {"cyto2": "bbb222"}})],
         _totals(2, mask=2, models=2), None),
        ("an empty digest is not a model",
         [(_run_name(0), {"app_key": "mask",
                          "model_hashes": {"unhashable": "",
                                           "cyto2": "cyto2.pth:abc123"}})],
         _totals(1, mask=1, models=1), None),
        ("legacy entries without a digest are not counted",
         [(_run_name(0), {"app_key": "measure",
                          "models": [{"sha256": ""}, {"name": "no digest here"},
                                     "not a dict at all", {"sha256": "ccc333"}]})],
         _totals(1, measure=1, models=1), None),
        ("another module counts only as a run",
         [("2026-08-31_010000_a__mask", {"app_key": "mask"}),
          ("2026-08-31_020000_b__regression", {"app_key": "regression"})],
         _totals(2, mask=1), None),
        ("a folder with no manifest is passed over quietly",
         [("2026-08-31_010000_good__mask", {"app_key": "mask"}),
          ("2026-08-31_030000_starting__mask", None)],
         _totals(1, mask=1), None),
    ]
    for case_id, runs, expected, fragment in cases:
        with subtests.test(msg=case_id):
            root = _case_dir(tmp_path, case_id)
            (root / "loose.txt").write_text("x", encoding="utf-8")
            for name, manifest in runs:
                _write_run(root, name, manifest)
            caplog.clear()
            with monkeypatch.context() as mp, caplog.at_level("WARNING"):
                mp.setattr(RJ, "runs_root", lambda: root)
                assert RJ.journal_totals() == expected
                assert RJ.journal_totals() == expected, "the cached answer agrees"
            if fragment is not None:
                assert fragment in caplog.text

    with subtests.test(msg="no journal gives zeros"):
        with monkeypatch.context() as mp:
            mp.setattr(RJ, "runs_root", lambda: tmp_path / "does_not_exist")
            assert RJ.journal_totals() == _totals()
        assert not (tmp_path / "does_not_exist").exists()

    with subtests.test(msg="the cache grows incrementally and a deletion forces a recount"):
        root = _case_dir(tmp_path, "incremental")
        a = _write_run(root, "2026-08-30_010000_a__mask",
                       {"app_key": "mask", "model_hashes": {"m": "m:1"}})
        with monkeypatch.context() as mp:
            mp.setattr(RJ, "runs_root", lambda: root)
            assert RJ.journal_totals() == _totals(1, mask=1, models=1)
            cache = json.loads((root / ".journal_totals.json").read_text(
                encoding="utf-8"))
            assert cache == {"version": 1, "totals": _totals(1, mask=1, models=1),
                             "counted": ["2026-08-30_010000_a__mask"],
                             "models": ["m:1"]}
            _write_run(root, "2026-08-30_020000_b__measure",
                       {"app_key": "measure", "model_hashes": {"m": "m:1"}})
            assert RJ.journal_totals() == _totals(2, mask=1, measure=1, models=1)
            shutil.rmtree(a)
            assert RJ.journal_totals() == _totals(1, measure=1, models=1)
            (root / ".journal_totals.json").write_text("{broken", encoding="utf-8")
            assert RJ.journal_totals() == _totals(1, measure=1, models=1)


# ---------------------------------------------------------------------------
# load_run_settings / _read_settings_csv
# ---------------------------------------------------------------------------

def test_run_settings_are_read_from_json_or_the_csv_twin(subtests, tmp_path):
    """settings.json wins, settings.csv is the fallback, and damage is refused."""
    cases = [
        # (id, settings.json text, settings.csv bytes, expected or exception)
        ("json is read as it was written",
         json.dumps({"src": "/data", "diam": 30}), None,
         {"src": "/data", "diam": 30}),
        ("json wins over the csv", json.dumps({"src": "/json"}),
         b"Key,Value\nsrc,/csv\n", {"src": "/json"}),
        ("the csv is the fallback and holds text",
         None, b"Key,Value\nsrc,/data\ndiam,30\n", {"src": "/data", "diam": "30"}),
        ("an ordinary csv keeps bracketed values whole",
         None, b"Key,Value\ndiameter,30\nchannels,[0 1]\n",
         {"diameter": "30", "channels": "[0 1]"}),
        ("a row with no value reads as empty", None, b"Key,Value\ndiameter\n",
         {"diameter": ""}),
        ("blank rows and a blank key are skipped", None,
         b"Key,Value\n\n,orphan\nsrc,x\n", {"src": "x"}),
        ("an embedded NUL is refused", None, b"Key,Value\ndiameter,3\x000\n",
         (csv.Error, "NUL")),
        ("no settings at all is FileNotFoundError", None, None,
         (FileNotFoundError, "no settings in")),
        ("broken json propagates", "{not json", None,
         (json.JSONDecodeError, "Expecting property name")),
    ]
    for case_id, as_json, as_csv, expected in cases:
        with subtests.test(msg=case_id):
            d = _case_dir(tmp_path, case_id)
            if as_json is not None:
                (d / "settings.json").write_text(as_json, encoding="utf-8")
            if as_csv is not None:
                (d / "settings.csv").write_bytes(as_csv)
            if isinstance(expected, tuple):
                with pytest.raises(expected[0], match=expected[1]):
                    RJ.load_run_settings(d)
            else:
                assert RJ.load_run_settings(str(d)) == expected
                if as_json is None:
                    assert RJ._read_settings_csv(d / "settings.csv") == expected


# ---------------------------------------------------------------------------
# hash_file and the lock check built on it
# ---------------------------------------------------------------------------

def test_hash_file_returns_a_short_or_full_sha256_or_none(subtests, tmp_path):
    """16 hex characters by default, 64 with ``full``, ``None`` when unreadable."""
    data = b"weights" * 1000
    model = tmp_path / "m.pt"
    model.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"")
    cases = [
        ("the default is the first 16 characters", (model,), {}, digest[:16]),
        ("full is all 64", (model,), {"full": True}, digest),
        ("the chunk size does not change the digest", (model, 7), {"full": True},
         digest),
        ("an empty file has the empty digest", (empty,), {"full": True},
         hashlib.sha256(b"").hexdigest()),
        ("a missing file is None", (tmp_path / "nope",), {}, None),
        ("a folder is None", (tmp_path,), {}, None),
    ]
    for case_id, args, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            assert RJ.hash_file(*args, **kwargs) == expected

    with subtests.test(msg="a locked file is a deviation only when it changed"):
        record = {"settings": {}, "files": {str(model): digest,
                                            str(tmp_path / "gone.pt"): "a" * 64}}
        deviations = [d for d in RJ._lock_deviations(record, {})
                      if d["key"].startswith("file:")]
        assert deviations == [{"key": f"file:{tmp_path / 'gone.pt'}",
                               "locked": "a" * 16, "now": "missing"}]
