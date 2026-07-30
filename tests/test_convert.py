"""
The Format converter / importer — :mod:`spacr.convert`.

Everything here runs on synthetic TIFFs and hand-built fake readers: no
sample file is downloaded, nothing touches a GPU, and every expectation
is about a number or a filename that was put there on purpose.

The properties the converter lives or dies by:

* the ``run1/wt/`` tree comes out as ``plate1_A01_T0001F001L01A01Z01C01.tif``
  — the exact name ``metadata_type='cellvoyager'`` parses — with a
  unique field id per field-set and a unique channel id per channel;
* the **map file round-trips**: every converted name resolves back to
  exactly one original file, which is the only thing that makes the
  renaming reversible;
* **two sources colliding on one output name is a plan-time error**
  naming both, not a last-writer-wins at convert time;
* **z is never silently flattened** — the plan says how many planes
  there are, the handling is a choice, and the choice is written into
  every row of the map;
* an **unreadable source is skipped, counted and named**, and the map is
  stamped partial so a later reader can tell;
* an **interrupted convert leaves no partial file**, and a **re-run does
  not overwrite**;
* a **missing optional reader produces the message and the pip command**,
  never an ImportError traceback.
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import convert as cv
from spacr.errors import ConfigurationError, read_run_status, run_is_complete


#: The regex ``spacr.utils._get_regex('cellvoyager', 'tif')`` builds.
#: Copied rather than imported because importing ``spacr.utils`` drags in
#: torch; if the two ever diverge, this test suite is the wrong place to
#: find out but the right place to have pinned the contract.
CELLVOYAGER_REGEX = (
    r"(?P<plateID>.*)_(?P<wellID>.*)_T(?P<timeID>.*)F(?P<fieldID>.*)"
    r"L(?P<laserID>..)A(?P<AID>..)Z(?P<sliceID>.*)C(?P<chanID>.*).tif")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path, value=1, shape=(8, 8), dtype=np.uint16, **kwargs):
    """Write a synthetic TIFF whose pixels encode which file it is."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, np.full(shape, value, dtype), **kwargs)
    return path


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _tifs(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith(".tif"))


@pytest.fixture
def run1(tmp_path):
    """The user's tree: ``run1/wt/`` with ten field-sets of two channels."""
    root = tmp_path / "src"
    for field in range(1, 11):
        for channel in (1, 2):
            _write(str(root / "run1" / "wt" / f"fov{field:02d}_C{channel}.tif"),
                   value=field * 10 + channel)
    return str(root)


# ---------------------------------------------------------------------------
# The pseudo-Yokogawa naming
# ---------------------------------------------------------------------------

def test_run1_wt_produces_exactly_the_specified_naming(run1, tmp_path):
    """``run1/wt/`` + 10 field-sets -> plate1 / A01 / F001..F010 / C01,C02."""
    sources = cv.scan(run1)
    assert len(sources) == 20
    plan = cv.plan(sources)
    assert plan.ok
    assert len(plan) == 20

    targets = sorted(m.target for m in plan.mappings)
    assert targets[0] == "plate1_A01_T0001F001L01A01Z01C01.tif"
    assert targets[1] == "plate1_A01_T0001F001L01A01Z01C02.tif"
    assert targets[-1] == "plate1_A01_T0001F010L01A01Z01C02.tif"

    # Field ids are unique per field-set and channel ids per channel.
    assert sorted({m.field for m in plan.mappings}) == list(range(1, 11))
    assert sorted({m.channel for m in plan.mappings}) == [1, 2]
    assert {m.plate for m in plan.mappings} == {"plate1"}
    assert {m.well for m in plan.mappings} == {"A01"}
    # 10 fields x 2 channels, each pair distinct.
    assert len({(m.field, m.channel) for m in plan.mappings}) == 20

    dst = str(tmp_path / "out")
    result = cv.convert(plan, dst)
    assert result.n_written == 20
    # Everything lands in ONE folder, as specified.
    assert _tifs(dst) == sorted(m.target for m in plan.mappings)


def test_every_target_parses_with_the_cellvoyager_regex(run1):
    """The output is only useful if spaCR's own regex reads it back."""
    pattern = re.compile(CELLVOYAGER_REGEX)
    plan = cv.plan(cv.scan(run1))
    for mapping in plan.mappings:
        match = pattern.match(mapping.target)
        assert match is not None, mapping.target
        assert match.group("plateID") == mapping.plate
        assert match.group("wellID") == mapping.well
        assert int(match.group("fieldID")) == mapping.field
        assert int(match.group("chanID")) == mapping.channel
        assert int(match.group("sliceID")) == mapping.z
        assert int(match.group("timeID")) == mapping.t


def test_target_name_is_the_documented_string():
    assert cv.target_name("plate1", "A01", 1, 1) == \
        "plate1_A01_T0001F001L01A01Z01C01.tif"
    assert cv.target_name("plate2", "H12", 137, 4, z=9, t=27) == \
        "plate2_H12_T0027F137L01A01Z09C04.tif"


def test_scanning_and_planning_write_nothing(run1, tmp_path):
    before = sorted(os.walk(run1))
    plan = cv.plan(cv.scan(run1))
    assert len(plan)
    assert sorted(os.walk(run1)) == before
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# Well naming — the deterministic, reversible rule
# ---------------------------------------------------------------------------

def test_a_canonical_well_name_keeps_its_own_address():
    assigned = cv.assign_wells(["B02", "a1", "H-12"])
    assert assigned == {"B02": "B02", "a1": "A01", "H-12": "H12"}


def test_non_canonical_names_are_assigned_in_natural_sorted_order():
    assigned = cv.assign_wells(["wt", "ko", "dmso"])
    assert assigned == {"dmso": "A01", "ko": "A02", "wt": "A03"}
    # Deterministic: the same names always give the same wells.
    assert cv.assign_wells(["wt", "dmso", "ko"]) == assigned


def test_assignment_skips_wells_a_canonical_name_already_claimed():
    assigned = cv.assign_wells(["A01", "wt"])
    assert assigned["A01"] == "A01"
    assert assigned["wt"] == "A02"


def test_more_wells_than_a_forced_384_plate_is_a_configuration_error():
    with pytest.raises(ConfigurationError) as excinfo:
        cv.assign_wells([f"sample{i}" for i in range(385)], n_wells=384)
    assert "384" in str(excinfo.value)


def test_normalise_well_respects_a_forced_384_plate_boundary():
    assert cv.normalise_well("A01") == "A01"
    assert cv.normalise_well("a1") == "A01"
    assert cv.normalise_well("P24") == "P24"
    assert cv.normalise_well("P25") == "P25"    # valid on a 1536 plate
    assert cv.normalise_well("Q01") == "Q01"    # valid on a 1536 plate
    assert cv.normalise_well("P25", n_wells=384) is None
    assert cv.normalise_well("Q01", n_wells=384) is None
    assert cv.normalise_well("wt") is None
    assert cv.normalise_well("") is None


def test_the_map_is_what_makes_a_synthetic_well_reversible(tmp_path):
    """``wt`` -> ``A01`` is only reversible because the map says so."""
    root = tmp_path / "src"
    for well in ("dmso", "wt"):
        _write(str(root / "run1" / well / "fov01_C1.tif"), value=1)
    plan = cv.plan(cv.scan(str(root)))
    result = cv.convert(plan, str(tmp_path / "out"))
    frame = cv.read_map(result.map_path)
    back = dict(zip(frame["well"], frame["source_well"]))
    assert back == {"A01": "dmso", "A02": "wt"}


def test_channel_ids_sort_naturally_so_c10_is_not_c2(tmp_path):
    root = tmp_path / "src"
    for channel in range(1, 11):
        _write(str(root / "plateA" / "wellA" / f"fov01_C{channel}.tif"),
               value=channel)
    plan = cv.plan(cv.scan(str(root)))
    lookup = {m.source_channel: m.channel for m in plan.mappings}
    assert lookup["C1"] == 1
    assert lookup["C2"] == 2
    assert lookup["C10"] == 10


def test_plate_naming_can_keep_the_folder_name(tmp_path):
    root = tmp_path / "src"
    _write(str(root / "my run 1" / "wt" / "fov01_C1.tif"))
    plan = cv.plan(cv.scan(str(root)), plate_naming="name")
    # Underscores are stripped: the cellvoyager regex splits plate from
    # well on '_', so a plate called "my_run" would move the split.
    assert plan.mappings[0].plate == "my-run-1"
    assert "_" not in plan.mappings[0].plate


def test_explicit_well_and_plate_overrides_win(run1):
    plan = cv.plan(cv.scan(run1), plate_naming="index",
                   plate_map={"run1": "screenX"}, well_map={"wt": "H12"})
    assert plan.mappings[0].plate == "screenX"
    assert plan.mappings[0].well == "H12"


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------

def test_auto_layout_detects_plate_well_well_and_flat(tmp_path):
    deep = tmp_path / "deep"
    _write(str(deep / "run1" / "wt" / "a.tif"))
    assert cv.scan(str(deep))[0].plate == "run1"
    assert cv.scan(str(deep))[0].well == "wt"

    mid = tmp_path / "mid"
    _write(str(mid / "wt" / "a.tif"))
    source = cv.scan(str(mid))[0]
    assert (source.plate, source.well) == ("mid", "wt")

    flat = tmp_path / "flat"
    _write(str(flat / "a.tif"))
    source = cv.scan(str(flat))[0]
    assert (source.plate, source.well) == ("flat", cv.DEFAULT_WELL)


def test_an_explicit_layout_overrides_the_guess(tmp_path):
    root = tmp_path / "src"
    _write(str(root / "run1" / "wt" / "a.tif"))
    source = cv.scan(str(root), layout="flat")[0]
    assert (source.plate, source.well) == ("src", cv.DEFAULT_WELL)
    source = cv.scan(str(root), layout="well")[0]
    assert (source.plate, source.well) == ("src", "run1")


def test_a_bad_source_or_layout_is_a_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError):
        cv.scan(str(tmp_path / "nope"))
    root = tmp_path / "src"
    _write(str(root / "a.tif"))
    with pytest.raises(ConfigurationError):
        cv.scan(str(root), layout="sideways")


def test_hidden_files_and_folders_are_skipped(tmp_path):
    root = tmp_path / "src"
    _write(str(root / "a.tif"))
    _write(str(root / ".hidden.tif"))
    _write(str(root / ".cache" / "b.tif"))
    assert [os.path.basename(s.path) for s in cv.scan(str(root))] == ["a.tif"]


# ---------------------------------------------------------------------------
# The map file
# ---------------------------------------------------------------------------

def test_the_map_round_trips_every_target_to_exactly_one_source(run1, tmp_path):
    plan = cv.plan(cv.scan(run1))
    result = cv.convert(plan, str(tmp_path / "out"))
    frame = cv.read_map(result.map_path)

    assert len(frame) == 20
    assert list(frame.columns) == list(cv.MAP_COLUMNS)
    assert frame["target"].is_unique
    assert frame.groupby("target")["source"].nunique().max() == 1
    # Every file on disk is described by the map, and vice versa.
    assert set(frame["target"]) == set(_tifs(str(tmp_path / "out")))
    for _, row in frame.iterrows():
        assert os.path.isfile(row["source"])
        assert os.path.isfile(row["target_path"])
        # The map alone is enough to walk the arrow backwards.
        expected = f"fov{int(row['field']):02d}_C{int(row['channel'])}.tif"
        assert os.path.basename(row["source"]) == expected


def test_the_map_carries_the_spacr_join_keys(run1, tmp_path):
    plan = cv.plan(cv.scan(run1))
    result = cv.convert(plan, str(tmp_path / "out"))
    frame = cv.read_map(result.map_path)
    row = frame[frame["target"] ==
                "plate1_A01_T0001F003L01A01Z01C02.tif"].iloc[0]
    assert row["plateID"] == "plate1"
    assert row["rowID"] == "r1"
    assert row["columnID"] == "c1"
    assert row["fieldID"] == "f3"
    assert row["prc"] == "plate1_r1_c1"
    assert row["prcf"] == "plate1_r1_c1_f3"


def test_read_map_refuses_a_file_that_is_not_a_conversion_map(tmp_path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(ConfigurationError) as excinfo:
        cv.read_map(str(missing))
    assert "does not exist" in str(excinfo.value)

    wrong = tmp_path / "other.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(wrong, index=False)
    with pytest.raises(ConfigurationError) as excinfo:
        cv.read_map(str(wrong))
    assert "not a spaCR conversion map" in str(excinfo.value)
    assert "target" in str(excinfo.value)


def test_read_map_refuses_a_file_that_is_not_a_csv_at_all(tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_bytes(b"\x00\x01\x02binary junk\x00")
    with pytest.raises(ConfigurationError):
        cv.read_map(str(broken))


def test_write_map_can_send_the_map_somewhere_else(run1, tmp_path):
    plan = cv.plan(cv.scan(run1))
    result = cv.convert(plan, str(tmp_path / "out"))
    elsewhere = tmp_path / "archive" / "map.csv"
    written = cv.write_map(result, str(elsewhere))
    assert written.is_file()
    assert len(cv.read_map(str(written))) == 20


# ---------------------------------------------------------------------------
# Collisions
# ---------------------------------------------------------------------------

def test_two_sources_on_one_target_is_a_plan_time_error_naming_both(tmp_path):
    root = tmp_path / "src"
    # Same field key, same channel key, different extension: both would
    # be written to plate1_A01_T0001F001L01A01Z01C01.tif.
    first = _write(str(root / "fov01_C1.tif"), value=1)
    second = str(root / "fov01_C1.tiff")
    _write(second, value=2)

    plan = cv.plan(cv.scan(str(root)))
    assert not plan.ok
    assert len(plan.errors) == 1
    message = plan.errors[0]
    assert "plate1_A01_T0001F001L01A01Z01C01.tif" in message
    assert first in message
    assert second in message

    # And it stays a plan-time error: convert refuses, writes nothing.
    dst = tmp_path / "out"
    with pytest.raises(ConfigurationError) as excinfo:
        cv.convert(plan, str(dst))
    assert "cannot be converted" in str(excinfo.value)
    assert not dst.exists()


def test_a_collision_is_reported_once_per_target_not_once_per_source(tmp_path):
    root = tmp_path / "src"
    for ext in (".tif", ".tiff", ".png"):
        path = root / f"fov01_C1{ext}"
        os.makedirs(root, exist_ok=True)
        if ext == ".png":
            from PIL import Image
            Image.fromarray(np.zeros((8, 8), np.uint8)).save(path)
        else:
            _write(str(path))
    plan = cv.plan(cv.scan(str(root)))
    assert len(plan.errors) == 1
    assert "3 sources" in plan.errors[0]


def test_a_channel_token_on_a_multi_channel_file_is_an_error(tmp_path):
    """Two numbering schemes for one axis: one of them would be lost."""
    root = tmp_path / "src"
    _write(str(root / "fov01_C1.tif"), shape=(3, 8, 8))
    plan = cv.plan(cv.scan(str(root)))
    assert not plan.ok
    assert "holds 3 channels" in plan.errors[0]


def test_a_z_token_on_a_multi_plane_file_is_an_error(tmp_path):
    root = tmp_path / "src"
    _write(str(root / "fov01_Z02.tif"), shape=(6, 8, 8))
    plan = cv.plan(cv.scan(str(root)))
    assert not plan.ok
    assert "Z2 token" in plan.errors[0]
    assert "6 z planes" in plan.errors[0]


def test_a_t_token_on_a_multi_timepoint_file_is_an_error(tmp_path):
    root = tmp_path / "src"
    _write(str(root / "fov01_T02.tif"), shape=(2, 6, 8, 8),
           metadata={"axes": "TZYX"})
    plan = cv.plan(cv.scan(str(root)))
    assert not plan.ok
    assert "T2 token" in plan.errors[0]


def test_per_plane_files_become_one_field_with_several_z(tmp_path):
    """``fov01_Z01_C1`` … is a stack spread over files, not five fields."""
    root = tmp_path / "src"
    for z in range(1, 6):
        _write(str(root / f"fov01_Z{z:02d}_C1.tif"), value=z)
    plan = cv.plan(cv.scan(str(root)))
    assert plan.ok
    assert {m.field for m in plan.mappings} == {1}
    assert sorted(m.z for m in plan.mappings) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Z and T handling
# ---------------------------------------------------------------------------

def test_a_z_stack_is_reported_in_the_plan_and_recorded_in_the_map(tmp_path):
    root = tmp_path / "src"
    stack = np.arange(5 * 8 * 8, dtype=np.uint16).reshape(5, 8, 8)
    os.makedirs(root, exist_ok=True)
    tifffile.imwrite(str(root / "fov01.tif"), stack, metadata={"axes": "ZYX"})

    keep = cv.plan(cv.scan(str(root)), z_handling=cv.Z_KEEP)
    assert any("5 in total" in note for note in keep.notes)
    assert len(keep) == 5
    result = cv.convert(keep, str(tmp_path / "keep"))
    frame = cv.read_map(result.map_path)
    assert list(frame["z"]) == [1, 2, 3, 4, 5]
    assert set(frame["z_handling"]) == {"keep"}
    assert set(frame["n_z_planes"]) == {5}
    # Every plane really is its own plane, not five copies of one.
    written = [tifffile.imread(p) for p in sorted(frame["target_path"])]
    assert [int(a[0, 0]) for a in written] == [int(stack[z, 0, 0])
                                              for z in range(5)]


def test_max_projection_is_never_the_silent_default(tmp_path):
    root = tmp_path / "src"
    stack = np.zeros((5, 8, 8), np.uint16)
    stack[3] = 900
    os.makedirs(root, exist_ok=True)
    tifffile.imwrite(str(root / "fov01.tif"), stack, metadata={"axes": "ZYX"})

    # The default keeps the planes.
    assert cv.plan(cv.scan(str(root))).z_handling == cv.Z_KEEP

    projected = cv.plan(cv.scan(str(root)), z_handling=cv.Z_MAX)
    assert len(projected) == 1
    warning = "\n".join(projected.warnings)
    assert "max-projects" in warning
    assert "will NOT be written" in warning
    assert "5 in total" in warning

    result = cv.convert(projected, str(tmp_path / "max"))
    frame = cv.read_map(result.map_path)
    assert list(frame["z_handling"]) == ["max"]
    assert list(frame["n_z_planes"]) == [5]
    assert list(frame["source_z"]) == ["max(1..5)"]
    assert int(tifffile.imread(frame["target_path"].iloc[0])[0, 0]) == 900


def test_first_plane_only_says_what_it_discards(tmp_path):
    root = tmp_path / "src"
    stack = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
    os.makedirs(root, exist_ok=True)
    tifffile.imwrite(str(root / "fov01.tif"), stack, metadata={"axes": "ZYX"})
    plan = cv.plan(cv.scan(str(root)), z_handling=cv.Z_FIRST)
    assert len(plan) == 1
    assert "3 plane(s) are discarded" in "\n".join(plan.warnings)
    result = cv.convert(plan, str(tmp_path / "first"))
    frame = cv.read_map(result.map_path)
    assert list(frame["z_handling"]) == ["first"]
    assert int(tifffile.imread(frame["target_path"].iloc[0])[0, 0]) == 0


def test_timepoints_are_kept_and_announced(tmp_path):
    root = tmp_path / "src"
    movie = np.zeros((3, 8, 8), np.uint16)
    for t in range(3):
        movie[t] = 100 + t
    os.makedirs(root, exist_ok=True)
    tifffile.imwrite(str(root / "fov01.tif"), movie, metadata={"axes": "TYX"})
    plan = cv.plan(cv.scan(str(root)))
    assert "multiple timepoints" in "\n".join(plan.notes)
    assert sorted(m.t for m in plan.mappings) == [1, 2, 3]
    result = cv.convert(plan, str(tmp_path / "out"))
    frame = cv.read_map(result.map_path).sort_values("t")
    assert [int(tifffile.imread(p)[0, 0]) for p in frame["target_path"]] == \
        [100, 101, 102]


def test_an_unknown_axes_guess_is_a_warning_not_a_silence(tmp_path):
    """A plain 3-D imwrite records no axes; the guess must be visible."""
    root = tmp_path / "src"
    _write(str(root / "fov01.tif"), shape=(6, 8, 8))
    sources = cv.scan(str(root))
    assert sources[0].z == 6
    assert "assumed Z=6" in sources[0].meta["axes_assumed"]
    plan = cv.plan(sources)
    assert any("axes not recorded" in w for w in plan.warnings)


def test_a_small_leading_axis_with_no_recorded_axes_reads_as_channels(tmp_path):
    root = tmp_path / "src"
    _write(str(root / "fov01.tif"), shape=(3, 8, 8))
    sources = cv.scan(str(root))
    assert (sources[0].n_channels, sources[0].z) == (3, 1)
    assert "read as channels" in sources[0].meta["axes_assumed"]


def test_an_unknown_z_handling_is_refused(run1):
    with pytest.raises(ConfigurationError) as excinfo:
        cv.plan(cv.scan(run1), z_handling="average")
    assert "z_handling" in str(excinfo.value)
    with pytest.raises(ConfigurationError):
        cv.plan(cv.scan(run1), plate_naming="whatever")


# ---------------------------------------------------------------------------
# Unreadable sources
# ---------------------------------------------------------------------------

def test_an_unreadable_source_is_skipped_counted_named_and_stamped(tmp_path):
    root = tmp_path / "src"
    _write(str(root / "good.tif"), value=7)
    (root / "corrupt.tif").write_bytes(b"this is not a TIFF at all")

    sources = cv.scan(str(root))
    broken = [s for s in sources if not s.readable]
    assert len(broken) == 1
    assert broken[0].error

    plan = cv.plan(sources)
    assert plan.ok                        # one bad file does not block the run
    assert len(plan) == 1
    # The preview shows it rather than quietly omitting it.
    frame = plan.to_frame()
    assert len(frame) == 2
    skipped_row = frame[frame["target"] == ""].iloc[0]
    assert skipped_row["status"].startswith("SKIP")

    dst = str(tmp_path / "out")
    result = cv.convert(plan, dst)
    assert result.n_written == 1
    assert result.n_skipped == 1
    assert not result.is_complete
    summary = result.summary()
    assert "Skipped 1 source(s)" in summary
    assert "corrupt.tif" in summary

    # And the artifact itself carries the verdict.
    assert not run_is_complete(result.map_path)
    stamped = read_run_status(result.map_path)
    assert stamped[-1]["n_failed"] == 1
    assert "corrupt.tif" in stamped[-1]["failures"][0]["item"]


def test_a_folder_with_no_images_plans_empty_without_raising(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "notes.txt").write_text("nothing to see")
    plan = cv.plan(cv.scan(str(root)))
    assert plan.ok
    assert len(plan) == 0
    assert "No readable images" in "\n".join(plan.notes)


# ---------------------------------------------------------------------------
# Optional readers
# ---------------------------------------------------------------------------

def test_module_available_answers_both_ways_for_real_modules():
    assert cv._module_available("json") is True
    assert cv._module_available("a_module_that_is_not_installed_xyz") is False


def test_module_available_is_true_for_a_module_already_imported(monkeypatch):
    """An injected module has ``__spec__ = None``; find_spec raises on it."""
    fake = types.ModuleType("spacr_fake_reader_probe")
    assert fake.__spec__ is None
    monkeypatch.setitem(sys.modules, "spacr_fake_reader_probe", fake)
    assert cv._module_available("spacr_fake_reader_probe") is True


@pytest.mark.parametrize("ext,module,command", [
    (".nd2", "nd2reader", "pip install nd2reader"),
    (".czi", "czifile", "pip install czifile"),
    (".lif", "readlif", "pip install readlif"),
])
def test_a_missing_optional_reader_gives_the_message_not_a_traceback(
        tmp_path, monkeypatch, ext, module, command):
    root = tmp_path / "src"
    root.mkdir()
    (root / f"movie{ext}").write_bytes(b"vendor bytes")

    real = cv._module_available
    monkeypatch.setattr(cv, "_module_available",
                        lambda name: False if name == module else real(name))

    # No exception anywhere on the path from scan to convert.
    sources = cv.scan(str(root))
    assert len(sources) == 1
    assert not sources[0].readable
    assert module in sources[0].error
    assert command in sources[0].error
    assert "Traceback" not in sources[0].error

    plan = cv.plan(sources)
    assert plan.ok and len(plan) == 0
    assert any(command in w for w in plan.warnings)

    result = cv.convert(plan, str(tmp_path / "out"))
    assert result.n_skipped == 1
    assert command in result.summary()


def test_the_installed_branch_of_the_probe_is_exercised():
    """Formats served by always-present dependencies need no probe."""
    assert cv.reader_requirement(".tif") is None
    assert cv.reader_available(".tif") is True
    assert cv.reader_requirement(".nd2") == ("nd2reader", "pip install nd2reader")
    assert "not a supported input format" in cv.missing_reader_message(".xyz")


def test_import_reader_raises_a_message_when_the_import_itself_fails(
        tmp_path, monkeypatch):
    monkeypatch.setattr(cv, "_module_available", lambda name: True)

    def _boom(name):
        raise RuntimeError("the wheel is broken")

    monkeypatch.setattr(cv.importlib, "import_module", _boom)
    with pytest.raises(ConfigurationError) as excinfo:
        cv._import_reader(".lif")
    assert "readlif" in str(excinfo.value)
    assert "the wheel is broken" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Mocked vendor readers
# ---------------------------------------------------------------------------

def _install_fake_nd2(monkeypatch, sizes):
    class FakeND2Reader:
        def __init__(self, path):
            self.sizes = dict(sizes)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_frame_2D(self, t=0, z=0, c=0, v=0):
            return np.full((6, 6), 1000 * v + 100 * t + 10 * z + c, np.uint16)

    module = types.ModuleType("nd2reader")
    module.ND2Reader = FakeND2Reader
    monkeypatch.setitem(sys.modules, "nd2reader", module)


def test_an_nd2_with_several_fields_of_view_becomes_several_fields(
        tmp_path, monkeypatch):
    _install_fake_nd2(monkeypatch,
                      {"x": 6, "y": 6, "c": 2, "t": 1, "z": 3, "v": 2})
    root = tmp_path / "src"
    root.mkdir()
    (root / "movie.nd2").write_bytes(b"nd2")

    sources = cv.scan(str(root))
    assert len(sources) == 2                     # one per field of view
    assert [s.field for s in sources] == ["movie#s1", "movie#s2"]
    assert all((s.z, s.t, s.n_channels) == (3, 1, 2) for s in sources)

    plan = cv.plan(sources)
    assert len(plan) == 2 * 3 * 2                # fields x z x channels
    result = cv.convert(plan, str(tmp_path / "out"))
    assert result.n_written == 12
    # v=1 (second FOV -> F002), z index 2 (-> Z03), c=1 (-> C02).
    value = tifffile.imread(
        str(tmp_path / "out" / "plate1_A01_T0001F002L01A01Z03C02.tif"))
    assert int(value[0, 0]) == 1000 * 1 + 10 * 2 + 1


def test_a_czi_is_read_through_czifile(tmp_path, monkeypatch):
    array = np.zeros((2, 1, 2, 4, 6, 6), np.uint16)   # S T C Z Y X
    for s in range(2):
        for c in range(2):
            for z in range(4):
                array[s, 0, c, z] = 100 * s + 10 * c + z

    class FakeCziFile:
        def __init__(self, path):
            self.shape = array.shape
            self.axes = "STCZYX"
            self.dtype = array.dtype

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def asarray(self):
            return array

    module = types.ModuleType("czifile")
    module.CziFile = FakeCziFile
    monkeypatch.setitem(sys.modules, "czifile", module)

    root = tmp_path / "src"
    root.mkdir()
    (root / "scan.czi").write_bytes(b"czi")
    sources = cv.scan(str(root))
    assert len(sources) == 2
    assert (sources[0].z, sources[0].n_channels, sources[0].t) == (4, 2, 1)

    plan = cv.plan(sources)
    result = cv.convert(plan, str(tmp_path / "out"))
    assert result.n_written == 2 * 4 * 2
    value = tifffile.imread(
        str(tmp_path / "out" / "plate1_A01_T0001F002L01A01Z04C02.tif"))
    assert int(value[0, 0]) == 100 * 1 + 10 * 1 + 3


def test_a_lif_is_read_through_readlif(tmp_path, monkeypatch):
    class FakeDims:
        x, y, z, t = 6, 6, 2, 1

    class FakeImage:
        def __init__(self, index):
            self.index = index
            self.dims = FakeDims()
            self.channels = 2

        def getFrame(self, z=0, t=0, c=0):
            return np.full((6, 6), 100 * self.index + 10 * z + c, np.uint16)

    class FakeReader:
        def __init__(self, path):
            self._images = [FakeImage(0), FakeImage(1), FakeImage(2)]

        def getIterImage(self):
            return iter(self._images)

    module = types.ModuleType("readlif")
    module.Reader = FakeReader
    monkeypatch.setitem(sys.modules, "readlif", module)

    root = tmp_path / "src"
    root.mkdir()
    (root / "experiment.lif").write_bytes(b"lif")
    sources = cv.scan(str(root))
    assert len(sources) == 3
    plan = cv.plan(sources)
    result = cv.convert(plan, str(tmp_path / "out"))
    assert result.n_written == 3 * 2 * 2
    value = tifffile.imread(
        str(tmp_path / "out" / "plate1_A01_T0001F003L01A01Z02C01.tif"))
    assert int(value[0, 0]) == 100 * 2 + 10 * 1 + 0


def test_a_vendor_file_that_raises_mid_read_is_ledgered_not_fatal(
        tmp_path, monkeypatch):
    _install_fake_nd2(monkeypatch,
                      {"x": 6, "y": 6, "c": 1, "t": 1, "z": 1, "v": 1})
    root = tmp_path / "src"
    root.mkdir()
    (root / "movie.nd2").write_bytes(b"nd2")
    _write(str(root / "fine.tif"), value=5)

    plan = cv.plan(cv.scan(str(root)))
    assert len(plan) == 2

    def _boom(source):
        raise OSError("the file went away")

    monkeypatch.setattr(cv, "_read_nd2", _boom)
    result = cv.convert(plan, str(tmp_path / "out"))
    assert result.n_written == 1
    assert result.n_skipped == 1
    assert len(result.failed) == 1
    assert "the file went away" in result.summary()
    assert not run_is_complete(result.map_path)


# ---------------------------------------------------------------------------
# Writing: atomic, never overwriting
# ---------------------------------------------------------------------------

def test_an_interrupted_convert_leaves_no_partial_file(tmp_path, monkeypatch):
    root = tmp_path / "src"
    _write(str(root / "fov01_C1.tif"), value=1)
    _write(str(root / "fov02_C1.tif"), value=2)
    _write(str(root / "fov03_C1.tif"), value=3)
    plan = cv.plan(cv.scan(str(root)))
    assert len(plan) == 3

    real = cv._imwrite
    calls = {"n": 0}

    def _flaky(path, array):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full halfway through the write")
        return real(path, array)

    monkeypatch.setattr(cv, "_imwrite", _flaky)
    dst = tmp_path / "out"
    result = cv.convert(plan, str(dst))

    assert result.n_written == 2
    assert result.n_skipped == 1
    # The failed target does not exist at all — not truncated, not empty.
    failed_target = result.failed[0].target
    assert not (dst / failed_target).exists()
    # And no temp file is left behind for the next run to trip over.
    leftovers = [f for f in os.listdir(dst) if f.startswith(cv._TMP_PREFIX)]
    assert leftovers == []
    # Everything that WAS written is a complete, readable TIFF.
    for mapping in result.written:
        assert tifffile.imread(str(dst / mapping.target)).shape == (8, 8)


def test_the_write_really_goes_through_a_temp_file(tmp_path, monkeypatch):
    """Not "it happens to work" — the rename is the mechanism."""
    seen = {}

    def _record(path, array):
        seen["path"] = path
        return np.save(path, array)

    monkeypatch.setattr(cv, "_imwrite", _record)
    target = tmp_path / "final.tif"
    cv._atomic_write(str(target), np.zeros((4, 4), np.uint16))
    assert os.path.basename(seen["path"]).startswith(cv._TMP_PREFIX)
    assert seen["path"] != str(target)
    assert target.exists()


def test_rerunning_over_an_existing_output_does_not_overwrite(run1, tmp_path):
    dst = str(tmp_path / "out")
    plan = cv.plan(cv.scan(run1))
    first = cv.convert(plan, dst)
    assert first.n_written == 20
    digests = {name: _digest(os.path.join(dst, name)) for name in _tifs(dst)}

    second = cv.convert(cv.plan(cv.scan(run1)), dst)
    assert second.n_written == 0
    assert len(second.existing) == 20
    assert {name: _digest(os.path.join(dst, name))
            for name in _tifs(dst)} == digests
    # A no-op re-run is still a complete run, and it says what it left alone.
    assert second.is_complete
    assert "20 target(s) already existed and were left untouched" in \
        second.summary()
    # The map still describes the whole folder, marked as pre-existing.
    frame = cv.read_map(second.map_path)
    assert set(frame["status"]) == {"existing"}
    assert len(frame) == 20


def test_resume_uses_complete_fields_and_repairs_a_corrupt_field(
        tmp_path, monkeypatch):
    root = tmp_path / "src"
    _write(str(root / "fov01_C1.tif"), value=1)
    _write(str(root / "fov02_C1.tif"), value=2)
    plan = cv.plan(cv.scan(str(root)))
    dst = tmp_path / "out"

    first = cv.convert(plan, str(dst))
    assert first.n_written == 2
    assert (dst / cv.CHECKPOINT_FILENAME).is_file()

    corrupt = next(
        mapping for mapping in plan.mappings if mapping.field == 2)
    (dst / corrupt.target).write_bytes(b"truncated")
    reads = []
    real_read = cv._read_source

    def _count(source):
        reads.append(source.path)
        return real_read(source)

    monkeypatch.setattr(cv, "_read_source", _count)
    resumed = cv.convert(
        cv.plan(cv.scan(str(root))), str(dst), resume=True)

    assert len(resumed.resumed_fields) == 1
    assert resumed.n_written == 1
    assert len(reads) == 1
    assert tifffile.imread(str(dst / corrupt.target)).shape == (8, 8)
    assert "Checkpoint repair" not in resumed.summary()
    assert "Resumed 1 completed field" in resumed.summary()


def test_resume_refuses_changed_conversion_inputs(tmp_path):
    root = tmp_path / "src"
    source = root / "fov01_C1.tif"
    _write(str(source), value=1)
    dst = tmp_path / "out"
    cv.convert(cv.plan(cv.scan(str(root))), str(dst))

    _write(str(source), value=9)
    with pytest.raises(ConfigurationError, match="does not match"):
        cv.convert(cv.plan(cv.scan(str(root))), str(dst), resume=True)


def test_overwrite_is_opt_in(run1, tmp_path):
    dst = str(tmp_path / "out")
    cv.convert(cv.plan(cv.scan(run1)), dst)
    again = cv.convert(cv.plan(cv.scan(run1)), dst, overwrite=True)
    assert again.n_written == 20
    assert again.existing == []


def test_converting_in_place_is_refused(tmp_path):
    root = tmp_path / "src"
    _write(str(root / "fov01_C1.tif"))
    plan = cv.plan(cv.scan(str(root)))
    with pytest.raises(ConfigurationError) as excinfo:
        cv.convert(plan, str(root))
    assert "originals stay" in str(excinfo.value)


def test_progress_is_reported_once_per_source(run1, tmp_path):
    seen = []
    plan = cv.plan(cv.scan(run1))
    cv.convert(plan, str(tmp_path / "out"),
               progress=lambda done, total, item: seen.append((done, total)))
    assert len(seen) == plan.n_sources == 20
    assert seen[0] == (1, 20)
    assert seen[-1] == (20, 20)


# ---------------------------------------------------------------------------
# The read-back into measurements.db
# ---------------------------------------------------------------------------

def test_populate_db_from_map_joins_the_originals_onto_measurements(
        run1, tmp_path):
    plan = cv.plan(cv.scan(run1))
    result = cv.convert(plan, str(tmp_path / "out"))

    # A measurements.db shaped the way spacr.measure writes one.
    db_path = str(tmp_path / "measurements.db")
    measurements = pd.DataFrame([
        {"plateID": "plate1", "rowID": "r1", "columnID": "c1",
         "fieldID": f"f{field}", "object_label": obj,
         "cell_area": 100.0 + field}
        for field in range(1, 11) for obj in (1, 2)])
    connection = sqlite3.connect(db_path)
    try:
        measurements.to_sql("cell", connection, index=False)
    finally:
        connection.close()

    written = cv.populate_db_from_map(db_path, result.map_path)
    assert written == 20

    connection = sqlite3.connect(db_path)
    try:
        joined = pd.read_sql_query(
            "SELECT c.fieldID, c.object_label, m.source, m.source_well, "
            "       m.source_field, m.source_channel "
            "FROM cell AS c JOIN conversion_map AS m "
            "  ON c.plateID = m.plateID AND c.rowID = m.rowID "
            " AND c.columnID = m.columnID AND c.fieldID = m.fieldID "
            "ORDER BY c.fieldID, c.object_label, m.source_channel",
            connection)
    finally:
        connection.close()

    # Two objects x two channels for each of ten fields.
    assert len(joined) == 40
    assert set(joined["source_well"]) == {"wt"}
    first = joined[joined["fieldID"] == "f1"].iloc[0]
    assert os.path.basename(first["source"]) == "fov01_C1.tif"
    assert first["source_field"] == "fov01"


def test_populating_twice_replaces_rather_than_duplicates(run1, tmp_path):
    plan = cv.plan(cv.scan(run1))
    result = cv.convert(plan, str(tmp_path / "out"))
    db_path = str(tmp_path / "measurements.db")
    assert cv.populate_db_from_map(db_path, result.map_path) == 20
    assert cv.populate_db_from_map(db_path, result.map_path) == 20
    connection = sqlite3.connect(db_path)
    try:
        count = connection.execute(
            f"SELECT COUNT(*) FROM {cv.CONVERSION_TABLE}").fetchone()[0]
        indexes = [r[0] for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index'")]
    finally:
        connection.close()
    assert count == 20
    assert f"idx_{cv.CONVERSION_TABLE}_prcf" in indexes


def test_populate_db_creates_the_database_if_it_is_missing(run1, tmp_path):
    plan = cv.plan(cv.scan(run1))
    result = cv.convert(plan, str(tmp_path / "out"))
    db_path = str(tmp_path / "deeper" / "new" / "measurements.db")
    assert cv.populate_db_from_map(db_path, result.map_path) == 20
    assert os.path.isfile(db_path)


def test_populate_db_refuses_a_map_that_is_not_one(tmp_path):
    wrong = tmp_path / "not_a_map.csv"
    pd.DataFrame({"x": [1]}).to_csv(wrong, index=False)
    with pytest.raises(ConfigurationError):
        cv.populate_db_from_map(str(tmp_path / "m.db"), str(wrong))


# ---------------------------------------------------------------------------
# Plain formats and small helpers
# ---------------------------------------------------------------------------

def test_a_png_is_converted_and_its_channels_split(tmp_path):
    from PIL import Image

    root = tmp_path / "src"
    root.mkdir()
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 10, 20, 30
    Image.fromarray(rgb).save(root / "fov01.png")

    sources = cv.scan(str(root))
    assert sources[0].n_channels == 3
    plan = cv.plan(sources)
    assert len(plan) == 3
    result = cv.convert(plan, str(tmp_path / "out"))
    values = [int(tifffile.imread(os.path.join(result.dst, m.target))[0, 0])
              for m in sorted(result.written, key=lambda m: m.channel)]
    assert values == [10, 20, 30]


def test_an_ome_tiff_extension_is_recognised(tmp_path):
    root = tmp_path / "src"
    _write(str(root / "fov01.ome.tif"), value=4)
    sources = cv.scan(str(root))
    assert len(sources) == 1
    assert sources[0].ext == ".ome.tif"
    assert sources[0].field == "fov01"


def test_the_natural_key_orders_digits_numerically():
    assert sorted(["C10", "C2", "C1"], key=cv._natural_key) == \
        ["C1", "C2", "C10"]
    assert cv._natural_key("") == ((10 ** 15, ""),)


def test_strip_tokens_pulls_channel_z_and_t_apart():
    assert cv._strip_tokens("fov01_C2") == ("fov01", "C2", None, None)
    assert cv._strip_tokens("fov01_Z03_C2") == ("fov01", "C2", 3, None)
    assert cv._strip_tokens("fov01_T0004_Z03_ch2") == ("fov01", "C2", 3, 4)
    # No separator in front: not a token.
    assert cv._strip_tokens("BC1") == ("BC1", None, None, None)
    # An already-Yokogawa name is left whole.
    assert cv._strip_tokens("plate1_A01_T0001F001L01A01Z01C01")[1] is None


def test_an_already_converted_looking_source_is_flagged(tmp_path):
    root = tmp_path / "src"
    _write(str(root / "plate1_A01_T0001F001L01A01Z01C01.tif"))
    plan = cv.plan(cv.scan(str(root)))
    assert any("already named like" in w for w in plan.warnings)


def test_the_plan_summary_carries_errors_warnings_and_notes(tmp_path):
    root = tmp_path / "src"
    _write(str(root / "fov01_C1.tif"))
    _write(str(root / "fov01_C1.tiff"))
    (root / "corrupt.tif").write_bytes(b"not a TIFF")
    text = cv.plan(cv.scan(str(root))).summary()
    assert "ERROR:" in text                       # the collision
    assert "WARNING:" in text                     # the unreadable file
    assert "corrupt.tif" in text
    assert "plate(s)" in text


def test_a_plan_with_no_sources_still_renders_a_frame():
    empty = cv.plan([])
    frame = empty.to_frame()
    assert list(frame.columns)[:2] == ["source", "target"]
    assert len(frame) == 0
    assert empty.ok
    assert len(empty) == 0


# ---------------------------------------------------------------------------
# The corners: every branch that only fires when something is wrong
# ---------------------------------------------------------------------------

def test_a_probe_that_raises_reads_as_not_installed(monkeypatch):
    """``find_spec`` throws for a namespace package with no parent."""
    def _boom(name):
        raise ValueError(f"{name}.__spec__ is not set")

    monkeypatch.setattr(cv.importlib.util, "find_spec", _boom)
    assert cv._module_available("some_module_not_in_sys_modules") is False


def test_import_reader_refuses_an_unsupported_extension():
    with pytest.raises(ConfigurationError) as excinfo:
        cv._import_reader(".xyz")
    assert "not a supported input format" in str(excinfo.value)


def test_import_reader_refuses_an_uninstalled_reader(monkeypatch):
    monkeypatch.setattr(cv, "_module_available", lambda name: False)
    with pytest.raises(ConfigurationError) as excinfo:
        cv._import_reader(".czi")
    assert "pip install czifile" in str(excinfo.value)


def test_describe_and_read_refuse_an_unsupported_extension():
    with pytest.raises(ConfigurationError):
        cv._describe("whatever.xyz", ".xyz")
    fake = cv.SourceImage(path="whatever.xyz", plate="p", well="w", field="f",
                          meta={"ext": ".xyz"})
    with pytest.raises(ConfigurationError):
        cv._read_source(fake)


def test_a_lif_with_no_images_is_reported_not_indexed(tmp_path, monkeypatch):
    class EmptyReader:
        def __init__(self, path):
            pass

        def getIterImage(self):
            return iter([])

    module = types.ModuleType("readlif")
    module.Reader = EmptyReader
    monkeypatch.setitem(sys.modules, "readlif", module)

    root = tmp_path / "src"
    root.mkdir()
    (root / "empty.lif").write_bytes(b"lif")
    sources = cv.scan(str(root))
    assert not sources[0].readable
    assert "contains no images" in sources[0].error


def test_well_ids_survive_a_well_that_is_not_a_well():
    assert cv._well_ids("A01") == ("r1", "c1")
    assert cv._well_ids("H12") == ("r8", "c12")
    assert cv._well_ids("12") == ("12", "12")       # no leading letter
    assert cv._well_ids("AX") == ("AX", "AX")       # letter, but no number


@pytest.mark.parametrize("shape,axes,expected,fragment", [
    ((3, 8, 8), "QYX", (1, 1, 3), "read as channels"),
    ((6, 8, 8), "QYX", (1, 6, 1), "assumed Z=6"),
    ((2, 3, 8, 8), "QQYX", (2, 3, 1), "assumed T=2, Z=3"),
    ((5, 3, 8, 8), "QZYX", (5, 3, 1), "assumed T=5"),
    ((2, 3, 4, 5, 8, 8), "QQQQYX", (2, 3, 4), "ignored=5"),
    ((2, 3, 8, 8), "ZCYX", (1, 2, 3), ""),
    ((3, 8, 8), "SYX", (1, 1, 3), "interleaved samples"),
])
def test_axes_are_resolved_and_every_guess_is_stated(shape, axes, expected,
                                                     fragment):
    n_t, n_z, n_c, note = cv._axes_dims(shape, axes)
    assert (n_t, n_z, n_c) == expected
    assert fragment in note


def test_to_5d_handles_recorded_axes_it_does_not_model():
    # A leading mosaic/block axis is collapsed to its first element.
    array = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 3, 4, 5, 6)
    out = cv._to_5d(array, "BZCYX", 1, 3, 4)
    assert out.shape == (1, 3, 4, 5, 6)
    assert np.array_equal(out[0], array[0])

    # 'S' next to a real 'C' is interleaved samples of that channel.
    array = np.arange(2 * 3 * 4 * 4, dtype=np.uint16).reshape(2, 3, 4, 4)
    out = cv._to_5d(array, "CSYX", 1, 1, 2)
    assert out.shape == (1, 1, 2, 4, 4)
    assert np.array_equal(out[0, 0, 1], array[1, 0])

    # 'S' on its own stands in for the channel axis.
    out = cv._to_5d(np.zeros((3, 4, 4), np.uint16), "SYX", 1, 1, 3)
    assert out.shape == (1, 1, 3, 4, 4)


def test_to_5d_falls_back_when_the_counts_do_not_multiply_out():
    """A count that disagrees with the array is not silently reshaped."""
    out = cv._to_5d(np.zeros((7, 4, 4), np.uint16), "QYX", 2, 2, 2)
    assert out.shape == (1, 1, 7, 4, 4)


def test_a_grayscale_png_converts_as_one_channel(tmp_path):
    from PIL import Image

    root = tmp_path / "src"
    root.mkdir()
    Image.fromarray(np.full((8, 8), 42, np.uint8)).save(root / "fov01.png")
    plan = cv.plan(cv.scan(str(root)))
    assert len(plan) == 1
    result = cv.convert(plan, str(tmp_path / "out"))
    written = tifffile.imread(os.path.join(result.dst, result.written[0].target))
    assert written.shape == (8, 8)
    assert int(written[0, 0]) == 42


def test_a_quirky_axes_tiff_converts_through_the_fallback(tmp_path):
    """A plain 3-D imwrite: no recorded axes, so the counts drive it."""
    root = tmp_path / "src"
    stack = np.arange(6 * 8 * 8, dtype=np.uint16).reshape(6, 8, 8)
    os.makedirs(root, exist_ok=True)
    tifffile.imwrite(str(root / "fov01.tif"), stack)
    plan = cv.plan(cv.scan(str(root)))
    assert len(plan) == 6
    result = cv.convert(plan, str(tmp_path / "out"))
    frame = cv.read_map(result.map_path).sort_values("z")
    assert [int(tifffile.imread(p)[0, 0]) for p in frame["target_path"]] == \
        [int(stack[z, 0, 0]) for z in range(6)]


def test_a_filename_t_token_becomes_the_output_timepoint(tmp_path):
    root = tmp_path / "src"
    for t in (1, 2, 3):
        _write(str(root / f"fov01_T{t:02d}_C1.tif"), value=t)
    plan = cv.plan(cv.scan(str(root)))
    assert sorted(m.t for m in plan.mappings) == [1, 2, 3]
    assert {m.field for m in plan.mappings} == {1}
    assert sorted(m.target for m in plan.mappings)[2].startswith(
        "plate1_A01_T0003F001")


def test_zero_based_z_and_t_tokens_become_one_based(tmp_path):
    root = tmp_path / "src"
    _write(str(root / "fov01_T00_Z00_C1.tif"))
    plan = cv.plan(cv.scan(str(root)))
    assert (plan.mappings[0].t, plan.mappings[0].z) == (1, 1)
    # The map still records what the filename actually said.
    assert (plan.mappings[0].source_t, plan.mappings[0].source_z) == ("0", "0")


def test_the_temp_file_is_cleaned_even_when_cleanup_itself_fails(
        tmp_path, monkeypatch):
    """Losing the temp file must not mask the real write error."""
    def _write_boom(path, array):
        raise OSError("no space left on device")

    def _unlink_boom(path):
        raise OSError("and the temp file cannot be removed either")

    monkeypatch.setattr(cv, "_imwrite", _write_boom)
    monkeypatch.setattr(cv.os, "unlink", _unlink_boom)
    with pytest.raises(OSError) as excinfo:
        cv._atomic_write(str(tmp_path / "out.tif"), np.zeros((4, 4), np.uint16))
    assert "no space left" in str(excinfo.value)


def test_a_plan_whose_sources_went_missing_is_a_setup_error(run1, tmp_path):
    """Not a per-item failure: every item would fail the same way."""
    plan = cv.plan(cv.scan(run1))
    plan.sources = []
    with pytest.raises(ConfigurationError) as excinfo:
        cv.convert(plan, str(tmp_path / "out"))
    assert "was not scanned" in str(excinfo.value)


def test_read_map_reports_an_unreadable_csv_as_a_configuration_error(
        tmp_path, monkeypatch):
    path = tmp_path / "map.csv"
    path.write_text("target,source\na,b\n")

    def _boom(*_a, **_k):
        raise MemoryError("the file is 40 GB")

    monkeypatch.setattr(cv.pd, "read_csv", _boom)
    with pytest.raises(ConfigurationError) as excinfo:
        cv.read_map(str(path))
    assert "could not be read as a conversion map" in str(excinfo.value)


def test_a_source_image_reports_its_extension_without_meta():
    bare = cv.SourceImage(path="/data/fov01.TIF", plate="p", well="w",
                          field="f")
    assert bare.ext == ".tif"
    assert bare.readable is True
    assert bare.error == ""


# ---------------------------------------------------------------------------
# The settings-dict entry point
# ---------------------------------------------------------------------------

def test_default_settings_fills_in_and_lets_the_caller_win():
    defaults = cv.default_settings()
    assert defaults["src"] is None
    assert defaults["z_handling"] == cv.Z_KEEP        # lossless by default
    assert defaults["overwrite"] is False
    assert defaults["map_name"] == cv.MAP_FILENAME
    assert cv.default_settings({"src": "/data", "overwrite": True})["src"] == \
        "/data"
    assert cv.default_settings({"overwrite": True})["overwrite"] is True


def test_convert_folder_runs_the_whole_thing_in_one_call(run1, tmp_path, capsys):
    dst = str(tmp_path / "out")
    db_path = str(tmp_path / "measurements.db")
    result = cv.convert_folder({"src": run1, "dst": dst, "db_path": db_path})

    assert result.n_written == 20
    assert os.path.isfile(result.map_path)
    assert len(cv.read_map(result.map_path)) == 20

    printed = capsys.readouterr().out
    # The plan is on the log before anything is written.
    assert "20 file(s) would be written" in printed
    assert "plate1_A01_T0001F001L01A01Z01C01.tif" in printed
    assert "and 0 more row(s)" not in printed
    assert "Converted 20 file(s)" in printed
    assert "conversion_map row(s)" in printed

    connection = sqlite3.connect(db_path)
    try:
        count = connection.execute(
            f"SELECT COUNT(*) FROM {cv.CONVERSION_TABLE}").fetchone()[0]
    finally:
        connection.close()
    assert count == 20


def test_convert_folder_defaults_the_destination_beside_the_source(run1):
    result = cv.convert_folder({"src": run1, "preview_only": True})
    assert result.dst == os.path.abspath(run1) + "_yokogawa"
    assert result.n_written == 0
    assert result.map_path == ""
    assert not os.path.exists(result.dst)


def test_convert_folder_preview_only_writes_nothing(run1, tmp_path, capsys):
    dst = tmp_path / "out"
    result = cv.convert_folder(src=run1, dst=str(dst), preview_only=True)
    assert result.n_written == 0
    assert not dst.exists()
    assert "nothing was written" in capsys.readouterr().out
    assert len(result.plan) == 20


def test_convert_folder_truncates_a_long_preview(run1, capsys):
    cv.convert_folder({"src": run1, "preview_only": True, "preview_rows": 5})
    printed = capsys.readouterr().out
    assert "… and 15 more row(s)." in printed


def test_convert_folder_needs_a_src():
    with pytest.raises(ConfigurationError) as excinfo:
        cv.convert_folder({})
    assert "'src'" in str(excinfo.value)


def test_convert_folder_refuses_a_plan_with_a_collision(tmp_path):
    root = tmp_path / "src"
    _write(str(root / "fov01_C1.tif"))
    _write(str(root / "fov01_C1.tiff"))
    with pytest.raises(ConfigurationError) as excinfo:
        cv.convert_folder({"src": str(root), "dst": str(tmp_path / "out")})
    assert "nothing was written" in str(excinfo.value)
    assert not (tmp_path / "out").exists()


def test_convert_folder_passes_z_handling_through(tmp_path):
    root = tmp_path / "src"
    os.makedirs(root)
    tifffile.imwrite(str(root / "fov01.tif"),
                     np.zeros((4, 8, 8), np.uint16), metadata={"axes": "ZYX"})
    result = cv.convert_folder({"src": str(root), "dst": str(tmp_path / "out"),
                                "z_handling": cv.Z_MAX})
    assert result.n_written == 1
    assert set(cv.read_map(result.map_path)["z_handling"]) == {"max"}


def test_the_result_summary_mentions_failed_planned_files(tmp_path, monkeypatch):
    root = tmp_path / "src"
    _write(str(root / "fov01_C1.tif"))
    _write(str(root / "fov01_C2.tif"))
    plan = cv.plan(cv.scan(str(root)))

    def _boom(path, array):
        raise OSError("write failed")

    monkeypatch.setattr(cv, "_imwrite", _boom)
    result = cv.convert(plan, str(tmp_path / "out"))
    assert result.n_written == 0
    assert len(result.failed) == 2
    assert "were not written because their source failed" in result.summary()
