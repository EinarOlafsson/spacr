"""Table-driven tests of what an open run records in ``spacr.run_journal``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: the distinct, bounded warning list; the stage recorder
and the log tail it writes; which settings values are offered as paths; the
file walker; the explicit model / input / output records of an open run; the
before/after output inventory; how a run closes when its own bookkeeping
fails; and the Cellpose checkpoint hook in ``spacr.pipeline_v2`` that feeds
``record_model``.  Every case runs in a fresh folder under ``tmp_path`` with
the journal root redirected there, so nothing touches the user's ``~/.spacr``.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import types
from pathlib import Path

import pytest

from spacr import pipeline_v2 as PV
from spacr import run_journal as RJ
from tests.refused_access import refused
from spacr.settings import SETTINGS_SCHEMA_VERSION

POSIX = os.name != "nt"
ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

def _case_dir(tmp_path, name):
    """A fresh per-case directory under ``tmp_path``."""
    d = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", name)
    d.mkdir()
    return d


def _sha(data: bytes) -> str:
    """Full SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _bare_run(folder):
    """A ``Run`` built directly, with no journal folder written."""
    return RJ.Run(app_key="mask", settings={}, dir=folder)


def _stage(identifier, label=None, state="pending", started=None, ended=None,
           duration=None, metrics=None):
    """The exact stage dict ``Run._record_stage`` stores."""
    return {"id": identifier, "label": identifier if label is None else label,
            "state": state, "started_at": started, "ended_at": ended,
            "duration_s": duration, "metrics": metrics or {}}


def _manifest(run):
    """The run's manifest.json, parsed."""
    return json.loads((run.dir / "manifest.json").read_text(encoding="utf-8"))


def _symlinks_work(folder):
    """Whether this filesystem makes symlinks."""
    probe = folder / ".probe-link"
    try:
        probe.symlink_to(folder)
    except (OSError, NotImplementedError):
        return False
    probe.unlink()
    return True


# ---------------------------------------------------------------------------
# Run.record_warning
# ---------------------------------------------------------------------------

def test_record_warning_keeps_distinct_stripped_warnings_up_to_five_hundred(
        subtests, tmp_path):
    """Warnings are stripped, blank ones dropped, repeats kept once, the first 500 kept."""
    fields = [f"field {i} had no objects" for i in range(600)]
    cases = [
        ("a repeat is kept once",
         ["class imbalance detected", "class imbalance detected"],
         ["class imbalance detected"]),
        ("an empty string is not a warning", [""], []),
        ("whitespace is not a warning", ["   "], []),
        ("None is not a warning", [None], []),
        ("text is stripped before it is compared",
         ["two fields had no objects", "  two fields had no objects\n"],
         ["two fields had no objects"]),
        ("a mixed stream keeps the distinct non-blank lines in order",
         ["cellpose: no masks found", "cellpose: no masks found", "   ", None,
          "torch: falling back to CPU"],
         ["cellpose: no masks found", "torch: falling back to CPU"]),
        ("600 distinct warnings keep the first 500", fields, fields[:500]),
        ("the cap keeps the first warning, not the last",
         ["the thing that actually went wrong"] + fields,
         ["the thing that actually went wrong"] + fields[:499]),
        ("the 501st distinct warning is dropped",
         fields[:500] + ["this one arrives over the ceiling"], fields[:500]),
        ("a non-string is stored as its text", [42, "42"], ["42"]),
    ]
    for case_id, messages, expected in cases:
        with subtests.test(msg=case_id):
            run = _bare_run(_case_dir(tmp_path, case_id))
            for message in messages:
                run.record_warning(message)
            assert run.run_warnings == expected


# ---------------------------------------------------------------------------
# Run._record_stage
# ---------------------------------------------------------------------------

class _HostileMetrics:
    """A metrics mapping whose ``items`` raises ``exc``."""

    def __init__(self, exc):
        self.exc = exc

    def items(self):
        raise self.exc


def test_record_stage_merges_observations_in_first_seen_order(
        subtests, tmp_path, monkeypatch):
    """Stages are created once, updated in place, timed, and never raise."""
    cases = [
        ("an empty id records nothing", [("", {})], []),
        ("a whitespace id records nothing", [("   ", {})], []),
        ("None is recorded under the id 'None'", [(None, {})], [_stage("None")]),
        ("the id is stripped", [("  segment  ", {})], [_stage("segment")]),
        ("a second label updates the first in place",
         [("segment", {"label": "Segmenting"}),
          ("segment", {"label": "Segmenting cells"})],
         [_stage("segment", "Segmenting cells")]),
        ("no label on a second call keeps the first",
         [("segment", {"label": "Segmenting"}), ("segment", {"state": "done"})],
         [_stage("segment", "Segmenting", state="done")]),
        ("first-seen order survives updates",
         [("a", {"label": "A"}), ("b", {"label": "B"}), ("a", {"state": "done"})],
         [_stage("a", "A", state="done"), _stage("b", "B")]),
        ("both ends give a duration",
         [("a", {"started_at": 10}), ("a", {"ended_at": "12.5"})],
         [_stage("a", started=10.0, ended=12.5, duration=2.5)]),
        ("one end gives no duration",
         [("a", {"started_at": 10.0})], [_stage("a", started=10.0)]),
        ("an end alone gives no duration",
         [("a", {"ended_at": 3})], [_stage("a", ended=3.0)]),
        ("labels and states are stored as text",
         [(7, {"label": 8, "state": 9})], [_stage("7", "8", state="9")]),
        ("metrics merge across calls with text keys",
         [("a", {"metrics": {"n": 3}}), ("a", {"metrics": {1: "x", "n": 4}})],
         [_stage("a", metrics={"n": 4, "1": "x"})]),
        ("empty metrics change nothing",
         [("a", {"metrics": {}})], [_stage("a")]),
        ("a KeyboardInterrupt while merging is swallowed after the stage exists",
         [("segment", {"metrics": _HostileMetrics(KeyboardInterrupt("ctrl-c"))})],
         [_stage("segment")]),
    ]
    for case_id, calls, expected in cases:
        with subtests.test(msg=case_id):
            run = _bare_run(_case_dir(tmp_path, case_id))
            for stage_id, kwargs in calls:
                run._record_stage(stage_id, **kwargs)
            assert run.stages == expected

    with subtests.test(msg="a logger that cannot log leaves the recorder usable"):
        run = _bare_run(_case_dir(tmp_path, "logger"))

        def refuse(*_a, **_k):
            raise RuntimeError("the log handler is gone too")

        with monkeypatch.context() as mp:
            mp.setattr(RJ.LOG, "debug", refuse)
            run._record_stage("segment",
                              metrics=_HostileMetrics(RuntimeError("bad metrics")))
            run._record_stage("measure")
        assert run.stages == [_stage("segment"), _stage("measure")]


# ---------------------------------------------------------------------------
# Run._snapshot_log_tail
# ---------------------------------------------------------------------------

def test_snapshot_log_tail_writes_the_log_tail_and_the_stage_lines(
        subtests, tmp_path, monkeypatch, caplog):
    """log.txt holds the last n log lines, a newline, then one line per stage."""
    from spacr import logging_util as LU

    stage_line = ("FlowView stage segment state=done started_at=1.0 "
                  "ended_at=3.0 duration_s=2.0 metrics={\"n\":2}\n")
    cases = [
        # (id, log text or None, stages?, n, expected log.txt or None)
        ("a tail with no final newline gains one before the stage line",
         "first line\nlast line without a newline", True, 10,
         "first line\nlast line without a newline\n" + stage_line),
        ("only the last n lines are copied", "1\n2\n3\n4\n", False, 2, "3\n4\n"),
        ("stages alone are written when there is no log", None, True, 10,
         stage_line),
        ("nothing to write writes no file at all", None, False, 10, None),
        ("an empty log and no stages writes no file", "", False, 10, None),
    ]
    for case_id, log_text, with_stage, n, expected in cases:
        with subtests.test(msg=case_id):
            folder = _case_dir(tmp_path, case_id)
            run = _bare_run(folder)
            src = folder / "app.log"
            if log_text is not None:
                src.write_text(log_text, encoding="utf-8")
            if with_stage:
                run._record_stage("segment", state="done", started_at=1,
                                  ended_at=3, metrics={"n": 2})
            with monkeypatch.context() as mp:
                mp.setattr(LU, "log_path", lambda: src)
                run._snapshot_log_tail(n=n)
            written = folder / "log.txt"
            if expected is None:
                assert not written.exists()
            else:
                assert written.read_text(encoding="utf-8") == expected

    with subtests.test(msg="a disk that refuses the snapshot is only warned about"):
        folder = _case_dir(tmp_path, "refused")
        run = _bare_run(folder)
        src = folder / "app.log"
        src.write_text("a line\n", encoding="utf-8")

        def refuse(self, *_a, **_k):
            raise OSError("no space left on device")

        caplog.clear()
        with monkeypatch.context() as mp, caplog.at_level("WARNING"):
            mp.setattr(LU, "log_path", lambda: src)
            mp.setattr(Path, "write_text", refuse)
            run._snapshot_log_tail(n=10)
        assert not (folder / "log.txt").exists()
        assert (f"could not write run log snapshot into {folder} "
                "(no space left on device)") in caplog.text


# ---------------------------------------------------------------------------
# _setting_path_candidates
# ---------------------------------------------------------------------------

def test_setting_path_candidates_offers_each_key_and_path_once(subtests, tmp_path):
    """Only path-like keys with usable values become (key, path, output_only)."""
    plate = tmp_path / "plate1"
    plate.mkdir()
    target = tmp_path / "input.tif"
    target.write_bytes(b"x" * 32)
    missing_out = tmp_path / "never-made" / "out.csv"
    cases = [
        ("a newline in a path setting is not a path",
         {"src": str(plate), "src_notes": "first line\nsecond line"},
         [("src", plate, False)]),
        ("an unexpandable home is skipped, the output still offered",
         {"src": "~no_such_user_zzz/plate1", "dst": str(missing_out)},
         [("dst", missing_out, True)]),
        ("a flattened key repeating a nested one is offered once per key",
         {"paths": {"src_file": str(target)}, "paths.src_file": str(target),
          "model_path": str(target)},
         [("paths.src_file", target, False), ("model_path", target, False)]),
        ("a missing input is dropped, a missing output kept",
         {"src": str(tmp_path / "absent"), "dst": str(missing_out)},
         [("dst", missing_out, True)]),
        ("blank, NUL and non-string values are passed over",
         {"src": "   ", "src_file": "a\x00b", "model_path": 5, "dst": None}, []),
        ("an existing value under a non-path key is not a path",
         {"diameter": str(plate)}, []),
    ]
    for case_id, settings, expected in cases:
        with subtests.test(msg=case_id):
            assert RJ._setting_path_candidates(settings) == expected


# ---------------------------------------------------------------------------
# _iter_files
# ---------------------------------------------------------------------------

def test_iter_files_yields_regular_files_outside_the_excluded_roots(
        subtests, tmp_path, monkeypatch):
    """The walk yields sorted regular files and never raises on a bad path."""
    keep = tmp_path / "keep"
    (keep / "sub").mkdir(parents=True)
    (keep / "b.txt").write_text("b", encoding="utf-8")
    (keep / "a.txt").write_text("a", encoding="utf-8")
    (keep / "sub" / "c.txt").write_text("c", encoding="utf-8")
    single = tmp_path / "single.txt"
    single.write_text("s", encoding="utf-8")
    project = tmp_path / "project"
    (project / "nested").mkdir(parents=True)
    (project / "nested" / "a.txt").write_text("a", encoding="utf-8")
    walked = [keep / "a.txt", keep / "b.txt", keep / "sub" / "c.txt"]
    cases = [
        ("a folder is walked in sorted order", keep, [], walked),
        ("an excluded folder yields nothing", keep, [keep], []),
        ("an excluded sub-folder is pruned", keep, [keep / "sub"], walked[:2]),
        ("a file yields itself", single, [], [single]),
        ("an excluded file yields nothing", single, [single], []),
        ("a path under an excluded root yields nothing",
         project / "nested", [project], []),
        ("an unresolvable exclusion still lets the walk run",
         keep, [Path("bad\x00root")], walked),
        ("an unresolvable root yields nothing", Path("nope\x00nope"), [], []),
        ("a missing path yields nothing", tmp_path / "not-here", [], []),
    ]
    for case_id, root, excluded, expected in cases:
        with subtests.test(msg=case_id):
            assert list(RJ._iter_files(root, excluded)) == expected

    with subtests.test(msg="an entry the walk may not stat is passed by"):
        real = Path.is_symlink

        def refusing(self):
            if self.name == "b.txt":
                raise PermissionError(13, "Access is denied")
            return real(self)

        with monkeypatch.context() as mp:
            mp.setattr(Path, "is_symlink", refusing)
            assert list(RJ._iter_files(keep, ())) == [keep / "a.txt",
                                                      keep / "sub" / "c.txt"]

    if _symlinks_work(tmp_path):
        with subtests.test(msg="a symlinked file or folder is not walked"):
            linked = tmp_path / "linked"
            linked.mkdir()
            (linked / "real.txt").write_text("r", encoding="utf-8")
            (linked / "link.txt").symlink_to(linked / "real.txt")
            (linked / "dirlink").symlink_to(keep, target_is_directory=True)
            assert list(RJ._iter_files(linked, ())) == [linked / "real.txt"]
            assert list(RJ._iter_files(linked / "link.txt", ())) == []


# ---------------------------------------------------------------------------
# Run.record_model / record_input / record_output / attach_output
# ---------------------------------------------------------------------------

def _rec(path, data, keys=None):
    """The exact file record the journal writes for ``path`` holding ``data``."""
    stat = Path(path).stat()
    record = {"sha256": _sha(data), "size_bytes": len(data),
              "mtime_ns": stat.st_mtime_ns}
    if keys is not None:
        record["setting_keys"] = keys
    return record


def test_an_open_run_records_models_inputs_and_outputs_without_raising(
        subtests, tmp_path, monkeypatch):
    """Each record method stores exact records, or a warning naming what failed."""

    def model_ok(run, d):
        ckpt = d / "cyto.pth"
        ckpt.write_bytes(b"model-weights")
        run.record_model("cellpose_cyto", str(ckpt))
        return ({"cellpose_cyto": f"cyto.pth:{_sha(b'model-weights')[:16]}"},
                {"cellpose_cyto": dict(_rec(ckpt, b"model-weights"),
                                       path=str(ckpt.resolve()))})

    def check_model_ok(run, d, expected):
        hashes, files = expected
        assert run.model_hashes == hashes
        assert run.model_files == files
        assert run.provenance_warnings == []
        assert _manifest(run)["model_hashes"] == hashes

    def model_missing(run, d):
        run.record_model("missing", "/nonexistent/path.pth")

    def check_model_missing(run, d, _):
        assert (run.model_hashes, run.model_files) == ({}, {})
        assert run.provenance_warnings == []

    def model_bad(run, d):
        run.record_model("cyto", "bad\x00path.pth")

    def check_model_bad(run, d, _):
        assert (run.model_hashes, run.model_files) == ({}, {})
        assert len(run.provenance_warnings) == 1
        assert run.provenance_warnings[0].startswith(
            "model 'cyto' could not be recorded: ")
        assert "embedded null" in run.provenance_warnings[0]
        assert _manifest(run)["provenance_warnings"] == run.provenance_warnings

    def attach_file(run, d):
        out = d / "result.csv"
        out.write_text("a,b\n1,2\n", encoding="utf-8")
        return run.attach_output(str(out))

    def check_attach_file(run, d, dst):
        assert dst == run.dir / "outputs" / "result.csv"
        assert dst.read_text(encoding="utf-8") == "a,b\n1,2\n"
        assert run.output_hashes == {}, "hashing is off by default"
        assert run.provenance_warnings == []

    def attach_dir(run, d):
        figs = d / "figs"
        figs.mkdir()
        (figs / "a.png").write_bytes(b"x")
        return run.attach_output(str(figs))

    def check_attach_dir(run, d, dst):
        assert dst == run.dir / "outputs" / "figs"
        assert (dst / "a.png").read_bytes() == b"x"

    def attach_missing(run, d):
        return run.attach_output("/nonexistent/result.csv")

    def check_attach_missing(run, d, dst):
        assert dst is None
        assert len(run.provenance_warnings) == 1
        assert run.provenance_warnings[0].startswith(
            "output '/nonexistent/result.csv' could not be attached: ")

    def output_hashed(run, d):
        out = d / "table.csv"
        out.write_bytes(b"a,b\n1,2\n")
        run.record_output(out, setting_key="dst")
        return out

    def check_output_hashed(run, d, out):
        assert run.output_hashes == {
            str(out.resolve()): _rec(out, b"a,b\n1,2\n", ["dst"])}

    def output_or_input_unhashed(run, d):
        out = d / "table.csv"
        out.write_bytes(b"a,b\n")
        run.record_output(out, setting_key="dst")
        run.record_input(out, setting_key="src")

    def check_nothing_hashed(run, d, _):
        assert (run.output_hashes, run.input_hashes) == ({}, {})

    def attach_hashed(run, d):
        out = d / "result.csv"
        out.write_bytes(b"r")
        run.attach_output(out)
        return out

    def check_attach_hashed(run, d, out):
        assert run.output_hashes == {
            str(out.resolve()): _rec(out, b"r", ["attach_output"])}

    def keyed_inputs(run, d):
        (d / "anonymous.tif").write_bytes(b"1" * 16)
        (d / "named.tif").write_bytes(b"2" * 16)
        run.record_input(d / "anonymous.tif")
        run.record_input(d / "named.tif", setting_key="src_file")
        run.record_input(d / "named.tif", setting_key="model_path")

    def check_keyed_inputs(run, d, _):
        assert run.input_hashes == {
            str(d / "anonymous.tif"): _rec(d / "anonymous.tif", b"1" * 16),
            str(d / "named.tif"): _rec(d / "named.tif", b"2" * 16,
                                       ["model_path", "src_file"]),
        }

    def empty_folder(run, d):
        (d / "empty").mkdir()
        run.record_input(d / "empty", setting_key="src")

    def check_empty_folder(run, d, _):
        assert run.input_hashes == {}
        assert run.provenance_warnings == [
            f"no regular provenance files found in {(d / 'empty').resolve()}"]

    def unexpandable(run, d):
        run.record_input("~no_such_user_zzz/plate", setting_key="src")

    def check_unexpandable(run, d, _):
        assert run.input_hashes == {}
        assert len(run.provenance_warnings) == 1
        assert run.provenance_warnings[0].startswith(
            "could not record provenance path ~no_such_user_zzz/plate: ")

    def locked_in_folder(run, d):
        plate = d / "plate"
        plate.mkdir()
        (plate / "good.tif").write_bytes(b"pixels")
        (plate / "locked.tif").write_bytes(b"pixels")
        with refused(plate / "locked.tif", 0o000, windows="read"):
            run.record_input(plate, setting_key="src")

    def check_locked_in_folder(run, d, _):
        plate = d / "plate"
        assert run.input_hashes == {
            str(plate / "good.tif"): _rec(plate / "good.tif", b"pixels", ["src"])}
        assert run.provenance_warnings == [
            f"could not hash provenance file {plate / 'locked.tif'}"]

    def symlinked(run, d):
        plate = d / "plate"
        plate.mkdir()
        (plate / "a.tif").write_bytes(b"a" * 64)
        (plate / "a_link.tif").symlink_to(plate / "a.tif")
        run.record_input(plate, setting_key="src_folder")

    def check_symlinked(run, d, _):
        assert list(run.input_hashes) == [str(d / "plate" / "a.tif")]

    cases = [
        ("a readable model is fingerprinted", {}, model_ok, check_model_ok),
        ("a missing model leaves no entry and no warning", {}, model_missing,
         check_model_missing),
        ("a model path that cannot be inspected is a warning", {}, model_bad,
         check_model_bad),
        ("a file is attached into outputs/", {}, attach_file, check_attach_file),
        ("a folder is attached into outputs/", {}, attach_dir, check_attach_dir),
        ("a missing attachment is a warning", {}, attach_missing,
         check_attach_missing),
        ("an attachment is hashed when hashing is on", {"hash_inputs": True},
         attach_hashed, check_attach_hashed),
        ("an output is hashed when hashing is on", {"hash_inputs": True},
         output_hashed, check_output_hashed),
        ("nothing is hashed when hashing is off", {}, output_or_input_unhashed,
         check_nothing_hashed),
        ("a record carries only the setting keys that named it",
         {"hash_inputs": True}, keyed_inputs, check_keyed_inputs),
        ("an empty input folder is reported", {"hash_inputs": True},
         empty_folder, check_empty_folder),
    ]
    if POSIX:
        cases.append(("a path that cannot be expanded is reported",
                      {"hash_inputs": True}, unexpandable, check_unexpandable))
    if not ROOT:
        cases.append(("a file in a folder that will not hash is named",
                      {"hash_inputs": True}, locked_in_folder,
                      check_locked_in_folder))
    if _symlinks_work(tmp_path):
        cases.append(("a symlinked file in a source folder is not hashed",
                      {"hash_inputs": True}, symlinked, check_symlinked))
    for case_id, settings, body, check in cases:
        with subtests.test(msg=case_id):
            d = _case_dir(tmp_path, case_id)
            runs = d / "runs"
            runs.mkdir()
            with monkeypatch.context() as mp:
                mp.setattr(RJ, "runs_root", lambda: runs)
                with RJ.open_run("mask", settings) as run:
                    result = body(run, d)
            check(run, d, result)


# ---------------------------------------------------------------------------
# the before/after output inventory
# ---------------------------------------------------------------------------

def _delete_once(mp, trigger_name, doomed):
    """Delete ``doomed`` the first time a signature is taken for ``trigger_name``."""
    real = RJ._inventory_signature
    state = {"done": False}

    def racing(path):
        if not state["done"] and Path(path).name == trigger_name:
            state["done"] = True
            Path(doomed).unlink()
        return real(path)

    mp.setattr(RJ, "_inventory_signature", racing)


def test_the_final_inventory_hashes_what_the_run_created_or_changed(
        subtests, tmp_path, monkeypatch):
    """Changed files under setting-derived roots become output records; failures are named."""

    def never_made(d, mp, stack):
        return {"dst": str(d / "never-made" / "out.csv"), "hash_inputs": True}, None

    def unhashable_output(d, mp, stack):
        out = d / "out"
        out.mkdir()

        def body():
            (out / "table.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (out / "locked.csv").write_text("secret", encoding="utf-8")
            stack.enter_context(refused(out / "locked.csv", 0o000, windows="read"))

        return {"dst": str(out / "table.csv"), "hash_inputs": True}, body

    def named_vanishes(d, mp, stack):
        target = d / "input.tif"
        target.write_bytes(b"before")
        (d / "keeper.tif").write_bytes(b"unchanged")
        _delete_once(mp, "input.tif", target)
        return ({"hash_inputs": True, "src_file": str(target),
                 "model_path": str(d / "keeper.tif")},
                lambda: target.write_bytes(b"written by the run"))

    def walked_vanishes(d, mp, stack):
        plate = d / "plate"
        plate.mkdir()
        (plate / "a.tif").write_bytes(b"a" * 32)
        doomed = plate / "b.tif"
        doomed.write_bytes(b"b" * 32)
        _delete_once(mp, "a.tif", doomed)
        return ({"hash_inputs": True, "src_folder": str(plate)},
                lambda: doomed.write_bytes(b"rebuilt by the run"))

    def changed_unreadable(d, mp, stack):
        readable, sealed = d / "readable.tif", d / "sealed.tif"
        readable.write_bytes(b"r" * 16)
        sealed.write_bytes(b"s" * 16)

        def body():
            readable.write_bytes(b"r" * 4096)
            sealed.write_bytes(b"s" * 4096)
            stack.enter_context(refused(sealed, 0o000, windows="read"))

        return ({"hash_inputs": True, "src_file": str(readable),
                 "model_path": str(sealed)}, body)

    def hashing_off(d, mp, stack):
        out = d / "out"
        out.mkdir()
        return ({"dst": str(out / "t.csv")},
                lambda: (out / "t.csv").write_text("x", encoding="utf-8"))

    cases = [
        # (id, setup, expected {name: setting_keys}, expected warnings)
        ("an output folder never created inventories nothing", never_made, {}, []),
        ("a named file missed by the baseline counts as created", named_vanishes,
         {"input.tif": ["src_file"]}, []),
        ("a walked file missed by the baseline counts as created",
         walked_vanishes, {"b.tif": ["src_folder"]}, []),
        ("with hashing off nothing is hashed at the end", hashing_off, {}, []),
    ]
    if not ROOT:
        cases += [
            ("a new output that cannot be hashed is named", unhashable_output,
             {"table.csv": ["dst"]}, [lambda d: f"could not hash changed output file {d / 'out' / 'locked.csv'}"]),
            ("a changed input that cannot be read is left out", changed_unreadable,
             {"readable.tif": ["src_file"]}, []),
        ]
    for case_id, setup, expected, warnings in cases:
        with subtests.test(msg=case_id):
            d = _case_dir(tmp_path, case_id)
            runs = d / "runs"
            runs.mkdir()
            with contextlib.ExitStack() as stack, monkeypatch.context() as mp:
                mp.setattr(RJ, "runs_root", lambda: runs)
                settings, body = setup(d, mp, stack)
                with RJ.open_run("mask", settings) as run:
                    if body is not None:
                        body()
            got = {Path(k).name: v["setting_keys"]
                   for k, v in run.output_hashes.items()}
            assert got == expected
            assert run.provenance_warnings == [w(d) for w in warnings]
            assert _manifest(run)["output_hashes"] == run.output_hashes


# ---------------------------------------------------------------------------
# open_run: closing the run
# ---------------------------------------------------------------------------

def test_open_run_always_closes_with_a_manifest(
        subtests, tmp_path, monkeypatch, caplog):
    """The manifest lands, the right status is stamped, and bookkeeping failures never mask results."""
    from spacr import logging_util, workspace

    def explode(message, exc=RuntimeError):
        def _raise(*_a, **_k):
            raise exc(message)
        return _raise

    def write_fails_on_close(mp, calls):
        real = RJ.Run._write_manifest

        def _write(self):
            calls.append(self)
            if len(calls) > 1:
                raise OSError("read-only filesystem")
            return real(self)

        mp.setattr(RJ.Run, "_write_manifest", _write)

    cases = [
        # (id, patch, body raises?, expected status, expected prov. warnings, log fragment)
        ("a clean run is stamped success", None, None, "success", [], None),
        ("a raising body is stamped failed and re-raised", None,
         ValueError("boom"), "failed", [], None),
        ("a log that cannot be copied costs only log.txt",
         lambda mp, calls: mp.setattr(logging_util, "log_path",
                                      explode("no log directory", OSError)),
         None, "success", [], "could not copy the last 200 log lines into {dir} "
                              "(no log directory)"),
        ("a failed final inventory is recorded in the manifest",
         lambda mp, calls: mp.setattr(RJ.Run, "_capture_final_provenance",
                                      explode("the disk went away")),
         None, "success", ["final output provenance failed: the disk went away"],
         None),
        ("a workspace that cannot be saved does not fail the run",
         lambda mp, calls: mp.setattr(workspace, "save_for_run",
                                      explode("no GUI to describe")),
         None, "success", [], "Could not save the workspace for {dir}"),
    ]
    for case_id, patch, raised, status, prov, fragment in cases:
        with subtests.test(msg=case_id):
            d = _case_dir(tmp_path, case_id)
            calls = []
            caplog.clear()
            with monkeypatch.context() as mp, caplog.at_level("WARNING"):
                mp.setattr(RJ, "runs_root", lambda: d)
                if patch is not None:
                    patch(mp, calls)
                with contextlib.ExitStack() as stack:
                    if raised is not None:
                        stack.enter_context(pytest.raises(type(raised),
                                                          match=str(raised)))
                    with RJ.open_run("measure", {"diameter": 30}) as run:
                        assert RJ.current_run() is run
                        run.record_warning("two fields had no objects")
                        if raised is not None:
                            raise raised
            assert RJ.current_run() is None
            manifest = _manifest(run)
            assert run.status == manifest["status"] == status
            assert manifest["provenance_warnings"] == prov
            assert manifest["warnings"] == ["two fields had no objects"]
            assert manifest["app_key"] == "measure"
            assert json.loads((run.dir / "settings.json").read_text(
                encoding="utf-8")) == {"diameter": 30,
                                       "spacr_settings_version": SETTINGS_SCHEMA_VERSION}
            if raised is not None:
                assert run.error_traceback.endswith(f"ValueError: {raised}\n")
                assert manifest["traceback"] == run.error_traceback
            else:
                assert manifest["traceback"] is None
            if fragment is not None:
                assert fragment.format(dir=run.dir) in caplog.text
            if case_id.startswith("a log that"):
                assert not (run.dir / "log.txt").exists()

    with subtests.test(msg="a status the body stamps is not overwritten"):
        d = _case_dir(tmp_path, "stamped")
        with monkeypatch.context() as mp:
            mp.setattr(RJ, "runs_root", lambda: d)
            with RJ.open_run("measure", {}) as run:
                run.set_status("cancelled")
        assert run.status == _manifest(run)["status"] == "cancelled"

    with subtests.test(msg="a closing manifest that cannot be written does not "
                           "mask the pipeline's error"):
        d = _case_dir(tmp_path, "close-fails")
        calls = []
        caplog.clear()
        with monkeypatch.context() as mp, caplog.at_level("ERROR"):
            mp.setattr(RJ, "runs_root", lambda: d)
            write_fails_on_close(mp, calls)
            with pytest.raises(ValueError, match="segmentation blew up"):
                with RJ.open_run("mask", {}) as run:
                    raise ValueError("segmentation blew up")
        assert len(calls) == 2
        assert f"Could not finalize run manifest in {run.dir}" in caplog.text
        assert _manifest(run)["status"] == "running", "only the opening manifest landed"


# ---------------------------------------------------------------------------
# spacr.pipeline_v2._record_cellpose_hash -> Run.record_model
# ---------------------------------------------------------------------------

def test_record_cellpose_hash_records_every_checkpoint_shape(
        subtests, tmp_path, monkeypatch):
    """Checkpoints on the model, as a list or one level down, are recorded under the model name."""
    cases = [
        ("no checkpoint records nothing", lambda p: types.SimpleNamespace(
            pretrained_model=None), None),
        ("a missing checkpoint file records nothing", lambda p: types.SimpleNamespace(
            pretrained_model=[str(p.with_name("absent.pth"))]), None),
        ("a top-level list is recorded", lambda p: types.SimpleNamespace(
            pretrained_model=[str(p)]), "model.pth"),
        ("a nested path is recorded", lambda p: types.SimpleNamespace(
            pretrained_model=None,
            cp=types.SimpleNamespace(pretrained_model=str(p))), "model.pth"),
        ("a nested list is recorded", lambda p: types.SimpleNamespace(
            pretrained_model=None,
            cp=types.SimpleNamespace(pretrained_model=[str(p)])), "model.pth"),
    ]
    for case_id, make_model, recorded in cases:
        with subtests.test(msg=case_id):
            d = _case_dir(tmp_path, case_id)
            ckpt = d / "model.pth"
            ckpt.write_bytes(b"weights")
            with monkeypatch.context() as mp:
                mp.setattr(RJ, "runs_root", lambda: d)
                with RJ.open_run("mask", {"src": "/x"}) as run:
                    PV._record_cellpose_hash(make_model(ckpt), "cyto")
            expected = ({} if recorded is None else
                        {"cyto": f"{recorded}:{_sha(b'weights')[:16]}"})
            assert _manifest(run)["model_hashes"] == expected
