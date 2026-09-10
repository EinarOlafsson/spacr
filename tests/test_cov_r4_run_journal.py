"""The journal's edge cases: duplicates, vanished files, and unreadable records.

Every path pinned here is one the journal meets on a real filesystem rather
than in a tidy fixture:

* a settings dict whose flattened key repeats a nested one;
* a symlink sitting in an otherwise ordinary source folder;
* a file that disappears between being listed and being stat'ed -- the
  inventory's own race, and the reason a file can be reported as an output
  the run created;
* an input the run modified and then made unreadable;
* a warning stream that repeats itself, or will not stop;
* manifests whose ``model_hashes`` is the wrong shape, or whose digests are
  empty, written by older spaCR versions;
* a ``settings.json`` that will not parse with no CSV twin to fall back to.

None of these may take down the pipeline that was producing results, and none
of them may be silent about what the record now omits.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from spacr import run_journal as RJ


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """A sandboxed runs root, so nothing here touches the user's journal."""
    folder = tmp_path / "runs"
    folder.mkdir()
    monkeypatch.setattr(RJ, "runs_root", lambda: folder)
    return folder


def _write_run(runs_dir, name, manifest, *, settings=None, settings_csv=None):
    """Lay out one run folder by hand, the way a finished run leaves one."""
    directory = runs_dir / name
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    if settings is not None:
        (directory / "settings.json").write_text(settings)
    if settings_csv is not None:
        (directory / "settings.csv").write_text(settings_csv)
    return directory


def _run_name(index, app="mask"):
    when = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
    return f"{when:%Y-%m-%d_%H%M%S}_tag{index:04d}__{app}"


# ---------------------------------------------------------------------------
# discovering paths in a settings dict
# ---------------------------------------------------------------------------

def test_a_flattened_key_that_repeats_a_nested_one_is_offered_once(tmp_path):
    """Settings reach the journal flattened by CSV *and* nested from Python.

    Round-tripping a nested ``{"paths": {"src_file": ...}}`` through a
    ``Key,Value`` CSV produces the literal key ``paths.src_file``, so a dict
    carrying both spellings of one setting is an ordinary merge, not a
    pathological input. Offering that path twice would make the initial
    inventory walk it twice on every run.

    The deduplication is invisible downstream -- both entries would collapse
    into the same ``input_hashes`` key -- so the candidate list is the only
    place the effect can be observed, hence the private call.
    """
    target = tmp_path / "input.tif"
    target.write_bytes(b"x" * 32)

    settings = {
        "paths": {"src_file": str(target)},
        "paths.src_file": str(target),
        "model_path": str(target),
    }

    candidates = RJ._setting_path_candidates(settings)

    keys = sorted(key for key, _path, _out in candidates)
    assert keys == ["model_path", "paths.src_file"], (
        "the repeated key was offered twice, or the distinct one was lost"
    )
    # The same file under a DIFFERENT key is still its own candidate: the
    # deduplication is per (key, path), not per path.
    assert len({path for _key, path, _out in candidates}) == 1


# ---------------------------------------------------------------------------
# walking a source tree
# ---------------------------------------------------------------------------

def test_a_symlinked_file_in_a_source_folder_is_not_hashed(runs_dir, tmp_path):
    """``followlinks=False`` is only half the protection.

    ``os.walk`` never descends a symlinked directory, but it still lists a
    symlinked FILE among a folder's names. Hashing it would record the same
    bytes twice under two paths and make a plate look twice its size.
    """
    folder = tmp_path / "plate"
    folder.mkdir()
    real = folder / "a.tif"
    real.write_bytes(b"a" * 64)
    link = folder / "a_link.tif"
    link.symlink_to(real)

    with RJ.open_run("mask", {"hash_inputs": True}) as run:
        run.record_input(folder, setting_key="src_folder")

    recorded = {Path(key).name for key in run.input_hashes}
    assert "a.tif" in recorded, "the real file was not hashed at all"
    assert "a_link.tif" not in recorded, "a symlink was hashed as its own file"


def test_a_recorded_file_only_carries_the_setting_that_named_it(runs_dir,
                                                                tmp_path):
    """``setting_key`` is optional, and an empty one must not be written.

    A record whose ``setting_keys`` is ``[""]`` reads, in the manifest, as a
    file some anonymous setting asked for. Callers that know nothing about
    which setting produced a path -- ``attach_output`` aside, most of them --
    leave it off, and their records must simply not have the field.
    """
    anonymous = tmp_path / "anonymous.tif"
    anonymous.write_bytes(b"1" * 16)
    named = tmp_path / "named.tif"
    named.write_bytes(b"2" * 16)

    with RJ.open_run("mask", {"hash_inputs": True}) as run:
        run.record_input(anonymous)
        run.record_input(named, setting_key="src_file")

    records = {Path(key).name: value for key, value in run.input_hashes.items()}
    assert "setting_keys" not in records["anonymous.tif"]
    assert records["named.tif"]["setting_keys"] == ["src_file"]


# ---------------------------------------------------------------------------
# warnings the pipeline hands back
# ---------------------------------------------------------------------------

def test_a_repeated_or_blank_warning_is_not_retained_twice(runs_dir):
    """The dashboard shows distinct warnings, not a transcript.

    A library that emits the same line once per image would otherwise fill
    the manifest with one warning repeated thousands of times, and blank
    output lines would appear as empty rows in the run-history panel.
    """
    with RJ.open_run("mask", {}) as run:
        run.record_warning("cellpose: no masks found")
        run.record_warning("cellpose: no masks found")
        run.record_warning("   ")
        run.record_warning(None)
        run.record_warning("torch: falling back to CPU")

    assert run.run_warnings == [
        "cellpose: no masks found",
        "torch: falling back to CPU",
    ]


def test_the_five_hundredth_warning_is_the_last_one_kept(runs_dir):
    """The manifest is bounded even when the warning text is never repeated.

    Deduplication alone does not bound it: a library that names the offending
    field in every line produces unlimited DISTINCT warnings, and a manifest
    that grows with the data is not a record anybody can read.
    """
    with RJ.open_run("mask", {}) as run:
        for index in range(500):
            run.record_warning(f"field {index} is out of range")
        run.record_warning("this one arrives over the ceiling")

    assert len(run.run_warnings) == 500
    assert "field 499 is out of range" in run.run_warnings, (
        "the ceiling was hit before 500 distinct warnings were kept"
    )
    assert "this one arrives over the ceiling" not in run.run_warnings


# ---------------------------------------------------------------------------
# the inventory's own race with the filesystem
# ---------------------------------------------------------------------------

def _delete_once(monkeypatch, trigger_name, doomed):
    """Delete ``doomed`` the first time a signature is taken for ``trigger_name``.

    The baseline inventory stats files a live pipeline directory is still
    changing, so a path listed at 09:00:00 can be gone microseconds later.
    That window is inside one function -- ``_setting_path_candidates`` and the
    stat loop are both in ``_capture_initial_provenance`` -- so there is no
    outside seam to drive it from. The real ``_inventory_signature`` still
    produces the ``None``; this only opens the window.
    """
    real = RJ._inventory_signature
    state = {"done": False}

    def racing(path):
        if not state["done"] and Path(path).name == trigger_name:
            state["done"] = True
            Path(doomed).unlink()
        return real(path)

    monkeypatch.setattr(RJ, "_inventory_signature", racing)
    return state


def test_a_named_file_that_vanishes_before_it_is_stated_counts_as_created(
        runs_dir, tmp_path, monkeypatch):
    """A file missed by the baseline is reported as one the run created.

    That is the honest answer, not a bug: the baseline exists to say which
    files were NOT there before, and a file the inventory could not stat was
    not observed. Reporting it is how a reviewer sees the gap; dropping it
    would make the manifest claim a completeness it does not have.
    """
    data = tmp_path / "data"
    data.mkdir()
    target = data / "input.tif"
    target.write_bytes(b"before")
    keeper = data / "keeper.tif"
    keeper.write_bytes(b"unchanged")

    _delete_once(monkeypatch, "input.tif", target)

    settings = {
        "hash_inputs": True,
        "src_file": str(target),
        "model_path": str(keeper),
    }
    with RJ.open_run("mask", settings) as run:
        target.write_bytes(b"written by the run")

    outputs = {Path(key).name for key in run.output_hashes}
    assert "input.tif" in outputs, (
        "the un-baselined file was not reported as created"
    )
    assert "keeper.tif" not in outputs, (
        "an untouched input was reported as an output, so the baseline is "
        "not being consulted at all"
    )


def test_a_walked_file_that_vanishes_mid_inventory_counts_as_created(
        runs_dir, tmp_path, monkeypatch):
    """The same race one level down, inside a directory walk.

    A source folder is inventoried file by file, and a pipeline writing into
    that folder can remove one between the listing and its stat. Skipping it
    must cost only that file: the rest of the folder is still baselined, or
    every file in it would be reported as new at the end of the run.
    """
    data = tmp_path / "plate"
    data.mkdir()
    first = data / "a.tif"
    first.write_bytes(b"a" * 32)
    doomed = data / "b.tif"
    doomed.write_bytes(b"b" * 32)

    _delete_once(monkeypatch, "a.tif", doomed)

    with RJ.open_run("mask", {"hash_inputs": True,
                              "src_folder": str(data)}) as run:
        doomed.write_bytes(b"rebuilt by the run")

    outputs = {Path(key).name for key in run.output_hashes}
    assert "b.tif" in outputs, "the skipped file was not reported as created"
    assert "a.tif" not in outputs, (
        "the rest of the folder lost its baseline when one file vanished"
    )


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root can read a mode-000 file, so nothing fails")
def test_a_changed_input_that_cannot_be_read_is_left_out_of_the_manifest(
        runs_dir, tmp_path):
    """A file whose bytes changed but cannot be hashed has no honest record.

    ``_file_record`` returns ``None`` rather than a record with a missing
    digest, and the entry is dropped -- an output entry without a SHA-256 is
    worse than no entry, because the manifest's whole promise is that a
    listed file can be verified.
    """
    readable = tmp_path / "readable.tif"
    readable.write_bytes(b"r" * 16)
    sealed = tmp_path / "sealed.tif"
    sealed.write_bytes(b"s" * 16)

    settings = {
        "hash_inputs": True,
        "src_file": str(readable),
        "model_path": str(sealed),
    }
    try:
        with RJ.open_run("mask", settings) as run:
            readable.write_bytes(b"r" * 4096)
            sealed.write_bytes(b"s" * 4096)
            os.chmod(sealed, 0o000)
    finally:
        os.chmod(sealed, 0o600)

    outputs = {Path(key).name for key in run.output_hashes}
    assert "readable.tif" in outputs, (
        "a changed, readable input was not recorded, so the absence below "
        "proves nothing"
    )
    assert "sealed.tif" not in outputs


# ---------------------------------------------------------------------------
# reading the journal back
# ---------------------------------------------------------------------------

def test_recent_runs_without_a_limit_returns_the_whole_journal(runs_dir):
    """``limit=None`` is how a caller asks for everything.

    The pre-truncation that keeps startup cheap is keyed on a non-negative
    limit; with none given there is no bound to apply, and applying the
    default one anyway would silently shorten an export of the journal.
    """
    for index in range(12):
        _write_run(runs_dir, _run_name(index),
                   {"app_key": "mask", "status": "success",
                    "start_utc": (datetime(2026, 1, 1, tzinfo=timezone.utc)
                                  + timedelta(minutes=index)).isoformat()})

    assert len(RJ.recent_runs(limit=None)) == 12
    assert len(RJ.recent_runs(limit=10)) == 10, (
        "a bounded call returned everything too, so the None case is not "
        "distinguishable"
    )


def test_a_legacy_model_hashes_list_does_not_break_the_model_tally(runs_dir):
    """Older manifests stored models as a list, not a ``{name: digest}`` dict.

    Home's "models recorded" counter reads ``model_hashes.values()``; asking a
    list for those raises, and the whole dashboard number would be lost over
    one old run folder. The legacy ``models`` list is honoured instead.
    """
    _write_run(runs_dir, _run_name(0), {
        "app_key": "mask",
        "model_hashes": ["cyto2:deadbeef"],
        "models": [{"sha256": "aaa111"}],
    })
    _write_run(runs_dir, _run_name(1), {
        "app_key": "mask",
        "model_hashes": {"cyto2": "bbb222"},
    })

    totals = RJ.journal_totals()

    assert totals["total_runs"] == 2
    assert totals["models_recorded"] == 2, (
        "expected the legacy list's sha256 and the modern dict's digest, "
        f"got {totals['models_recorded']}"
    )


def test_an_empty_model_digest_is_not_counted_as_a_model(runs_dir):
    """A name recorded with no digest names nothing.

    ``record_model`` writes an entry per model name, and a checkpoint it
    could not hash leaves the digest empty. Counting those would inflate the
    dashboard's model tally with models nobody can identify.
    """
    _write_run(runs_dir, _run_name(0), {
        "app_key": "mask",
        "model_hashes": {"unhashable": "", "cyto2": "cyto2.pth:abc123"},
    })

    totals = RJ.journal_totals()

    assert totals["models_recorded"] == 1, (
        "the empty digest was counted, or the real one was not"
    )


def test_a_legacy_models_entry_without_a_sha_is_not_counted(runs_dir):
    """The legacy list held free-form dicts, and not all carried a digest.

    Anything without one -- an empty string, a missing key, or an entry that
    is not a dict at all -- is skipped rather than counted or raised on.
    """
    _write_run(runs_dir, _run_name(0), {
        "app_key": "measure",
        "models": [
            {"sha256": ""},
            {"name": "no digest here"},
            "not a dict at all",
            {"sha256": "ccc333"},
        ],
    })

    totals = RJ.journal_totals()

    assert totals["measure_runs"] == 1
    assert totals["models_recorded"] == 1, (
        "the digest-less legacy entries were counted, or the real one was not"
    )


def test_an_unparsable_settings_json_with_no_csv_twin_says_so(runs_dir):
    """``settings.json`` and ``settings.csv`` are written together.

    So an unreadable JSON usually has a twin to recover from -- and when it
    does not, the history entry has to say the settings are gone rather than
    show an empty settings pane that looks like a run with no settings.
    """
    manifest = {"app_key": "mask", "status": "success", "warnings": []}
    _write_run(runs_dir, _run_name(0), manifest,
               settings="{not json at all")
    _write_run(runs_dir, _run_name(1), manifest,
               settings="{not json at all",
               settings_csv="Key,Value\nsrc,/data/plate1\n")

    by_id = {record["run_id"]: record for record in RJ.search_runs()}
    orphan = by_id[_run_name(0)]
    twinned = by_id[_run_name(1)]

    assert any("settings.json unreadable" in w for w in orphan["warnings"])
    assert not any("fell back to settings.csv" in w for w in orphan["warnings"])
    assert orphan["settings"] == {}

    # The same damage WITH the twin present recovers, which is what makes the
    # two assertions above a statement about the missing CSV.
    assert any("fell back to settings.csv" in w for w in twinned["warnings"])
    assert twinned["settings"] == {"src": "/data/plate1"}


# ---------------------------------------------------------------------------
# search_runs' warning collection: line 1142's false arc is unreachable
# ---------------------------------------------------------------------------
#
#   1139        values = manifest.get(key) or []
#   1140        if isinstance(values, (list, tuple)):
#   1141            warnings_list.extend(...)
#   1142        elif values:
#   1143            warnings_list.append(str(values))
#
# ``elif values:`` can never be false. Line 1139's ``or []`` replaces every
# falsy value with ``[]``, and ``[]`` is caught by the ``isinstance`` on 1140,
# so line 1142 is only ever reached with a truthy ``values``. ``manifest``
# comes from ``json.loads`` (line 1121 -> ``_read_run_record``), so the only
# non-list types it can hold are dict / str / int / float / bool, none of
# which can change truthiness between line 1140 and line 1142. This is the
# "defensive re-check after a call that already guarantees the condition"
# family; the test below pins the guarantee rather than contorting to reach
# an arc that has no input.

def test_a_scalar_warnings_field_is_still_surfaced(runs_dir):
    """Only a truthy non-list can reach the scalar branch, and it is kept.

    Legacy manifests sometimes wrote ``warnings`` as one string. Dropping it
    would lose the only warning an old run recorded.
    """
    _write_run(runs_dir, _run_name(0), {
        "app_key": "mask", "status": "success",
        "warnings": "cellpose: no masks found",
    }, settings=json.dumps({"src": "/data"}))
    _write_run(runs_dir, _run_name(1), {
        "app_key": "mask", "status": "success",
        "warnings": [],
    }, settings=json.dumps({"src": "/data"}))

    by_id = {record["run_id"]: record for record in RJ.search_runs()}

    assert by_id[_run_name(0)]["warnings"] == ["cellpose: no masks found"]
    assert by_id[_run_name(1)]["warnings"] == []
