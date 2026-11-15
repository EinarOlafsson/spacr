"""The preview selectors shared by every live preview, on the days nothing lines up.

:mod:`spacr.qt.widgets.preview_controls` builds the field-of-view and channel
dropdowns, the sample cap box and the folder enumeration behind them.  Each
mismatch a real session produces -- a stain name in the channel combo, a field
that left the sample, a naming dialect spaCR does not know, a ``utils.py`` it
cannot lift ``_get_regex`` from, an unreadable folder -- must degrade to
something the user can still work with.  Nothing here opens an image: the
enumeration reads file names, so the plates are empty files with real
acquisition names.  One general test per unit, each walking a case table with
pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox  # noqa: E402

from spacr.qt.widgets import preview_controls as pc  # noqa: E402

pytestmark = pytest.mark.qt

WELLS = ("A01", "A02", "B03")
FIELDS = (1, 2)
GROUPED = ["A01 f001 (2ch)", "A01 f002 (2ch)", "A02 f001 (2ch)",
           "A02 f002 (2ch)", "B03 f001 (2ch)", "B03 f002 (2ch)"]
CUSTOM = (r".*_(?P<wellID>[A-Z]\d+)_T\d+F(?P<fieldID>\d+)"
          r"L\d+A\d+Z\d+C(?P<chanID>\d+)")


@pytest.fixture(autouse=True)
def _fresh_regex_caches(qapp):
    """Both regex helpers are ``lru_cache``d process-wide; start and end clean."""
    pc._get_regex_callable.cache_clear()
    pc._acquisition_regex.cache_clear()
    yield
    pc._get_regex_callable.cache_clear()
    pc._acquisition_regex.cache_clear()


def _clear_caches():
    pc._get_regex_callable.cache_clear()
    pc._acquisition_regex.cache_clear()


def _plate(root: Path, extensions=("tif", "tif")) -> list:
    """An empty file per (well, field, channel) with Yokogawa names."""
    root.mkdir(parents=True, exist_ok=True)
    names = []
    for well in WELLS:
        for field in FIELDS:
            for chan, ext in zip((1, 2), extensions):
                name = f"plate1_{well}_T0001F{field:03d}L01A01Z01C{chan:02d}.{ext}"
                (root / name).write_bytes(b"")
                names.append(name)
    return sorted(names)


def _entries(combo: QComboBox) -> list:
    return [combo.itemText(i) for i in range(combo.count())]


def _flat_keys(names):
    return [("", "", n) for n in names]


def test_selected_channel_and_channel_view_read_only_a_real_plane(qtbot, subtests):
    """Only ``Ch <n>``/``ch<n>`` names a plane; anything else or out of range shows the source whole."""
    for text, expected in [("", None), (pc.ALL_CHANNELS, None), ("channel one", None),
                           ("ch", None), ("chX", None), ("ch 2 of 4", None),
                           ("DAPI", None), ("Merged", None), ("2", None),
                           ("Ch 2", 2), ("Ch 0", 0), ("ch0", 0), ("ch12", 12)]:
        with subtests.test(msg=f"selected_channel {text!r}"):
            combo = QComboBox()
            qtbot.addWidget(combo)
            combo.addItems(["placeholder", text])
            combo.setCurrentIndex(1)
            assert pc.selected_channel(combo) == expected

    flat = np.zeros((4, 5), dtype=np.uint16)
    stack = np.arange(40, dtype=np.uint16).reshape(4, 5, 2)
    for case_id, image, index, whole in [
            ("flat image", flat, 1, True), ("past the end", stack, 7, True),
            ("negative", stack, -1, True), ("not a number", stack, "not a number", True),
            ("no selection", stack, None, True), ("no image", None, 0, True)]:
        with subtests.test(msg=f"channel_view {case_id}"):
            assert pc.channel_view(image, index) is image
    with subtests.test(msg="channel_view plane 1"):
        np.testing.assert_array_equal(pc.channel_view(stack, 1), stack[..., 1])


def test_sibling_sources_lists_neighbours_and_keeps_the_open_field(tmp_path, subtests):
    """Neighbours are matching files (or folders) in sorted order, and the open one always stays."""
    with subtests.test(msg="nothing loaded"):
        assert pc.sibling_sources("", (".tif",)) == []
        assert pc.sibling_sources(None, (".tif",)) == []

    with subtests.test(msg="folder gone"):
        assert pc.sibling_sources(tmp_path / "gone" / "field.tif", (".tif",)) == []

    with subtests.test(msg="unlistable folder keeps the open file"):
        unreadable = tmp_path / "locked"
        unreadable.mkdir()
        target = unreadable / "field.tif"
        target.write_bytes(b"x")
        os.chmod(unreadable, 0o100)
        try:
            listed = pc.sibling_sources(target, (".tif",))
        finally:
            os.chmod(unreadable, 0o755)
        assert listed == [target]

    with subtests.test(msg="only matching suffixes"):
        root = tmp_path / "files"
        root.mkdir()
        for name in ("a.tif", "b.tif", "notes.txt"):
            (root / name).write_bytes(b"x")
        (root / "subfolder").mkdir()
        assert pc.sibling_sources(root / "b.tif", (".tif",)) == [root / "a.tif", root / "b.tif"]

    with subtests.test(msg="folders as fields"):
        root = tmp_path / "folders"
        root.mkdir()
        for name in ("fov1", "fov2"):
            (root / name).mkdir()
        (root / "stray.tif").write_bytes(b"x")
        assert pc.sibling_sources(root / "fov1", (".tif",), directories=True) == [
            root / "fov1", root / "fov2"]

    with subtests.test(msg="open file with an unwanted suffix"):
        root = tmp_path / "odd"
        root.mkdir()
        (root / "a.tif").write_bytes(b"x")
        odd = root / "loaded.czi"
        odd.write_bytes(b"x")
        assert pc.sibling_sources(odd, (".tif",)) == [root / "a.tif", odd]


def test_image_set_plane_paths_and_depth(tmp_path, subtests):
    """A set answers with its file per channel, every plane of a stack, and its z depth."""
    def image_set(planes=None):
        return pc.ImageSet(key=("plate1", "A01", "1"), directory=str(tmp_path),
                           channels={"1": "A01_C1_z1.tif", "2": "A01_C2_z1.tif"},
                           planes=planes or {})

    with subtests.test(msg="no planes recorded"):
        item = image_set()
        assert item.plane_paths() == [tmp_path / "A01_C1_z1.tif"]
        assert item.plane_paths("2") == [tmp_path / "A01_C2_z1.tif"]
        assert item.z_count == 1

    with subtests.test(msg="no channels"):
        assert pc.ImageSet(key=("", "", "x"), directory=str(tmp_path)).plane_paths() == []

    with subtests.test(msg="stack"):
        item = image_set({"1": ["A01_C1_z1.tif", "A01_C1_z2.tif", "A01_C1_z3.tif"],
                          "2": ["A01_C2_z1.tif"]})
        assert item.plane_paths("1") == [tmp_path / f"A01_C1_z{n}.tif" for n in (1, 2, 3)]
        assert item.plane_paths() == item.plane_paths("1")
        assert item.z_count == 3


def test_populate_fov_combo_labels_selects_and_never_blanks(qtbot, tmp_path, subtests):
    """Labels fall back to names, a missing field stays on row 0, an empty list on -1."""
    def combo():
        box = QComboBox()
        qtbot.addWidget(box)
        return box

    with subtests.test(msg="labels shorter than sources"):
        box = combo()
        sources = [tmp_path / "one.tif", tmp_path / "two.tif"]
        pc.populate_fov_combo(box, sources, current=sources[1], labels=["A01 f001"])
        assert _entries(box) == ["A01 f001", "two.tif"]
        assert box.currentIndex() == 1

    with subtests.test(msg="current outside the sample"):
        box = combo()
        sources = [tmp_path / "A01 f001.tif", tmp_path / "A02 f001.tif"]
        labels = ["A01 f001", "A02 f001"]
        pc.populate_fov_combo(box, sources, current=tmp_path / "B12 f009.tif", labels=labels)
        assert _entries(box) == labels
        assert [box.itemData(i) for i in range(box.count())] == [str(p) for p in sources]
        assert (box.currentIndex(), box.currentData()) == (0, str(sources[0]))
        pc.populate_fov_combo(box, sources, current=sources[1], labels=labels)
        assert (box.currentIndex(), box.currentData()) == (1, str(sources[1]))

    with subtests.test(msg="empty source list"):
        box = combo()
        loaded = tmp_path / "plate" / "A01 f001.tif"
        pc.populate_fov_combo(box, [], current=loaded)
        assert (_entries(box), box.currentIndex(), box.currentData()) == ([], -1, None)
        pc.populate_fov_combo(box, [loaded], current=loaded)
        assert _entries(box) == ["A01 f001.tif"]
        assert box.currentData() == str(loaded)


def test_apply_sample_to_combo_states_the_sample_with_or_without_a_cap_box(tmp_path, subtests):
    """The note always says what is shown; a cap box is clamped to the folder and fed back."""
    plate = tmp_path / "plate"
    _plate(plate)

    with subtests.test(msg="no cap box, sampled"):
        sampler = pc.ImageSetSampler(max_sets=2)
        sampler.enumerate(plate, [".tif"])
        assert sampler.total == 6
        combo = QComboBox()
        note = pc.apply_sample_to_combo(combo, None, sampler, None, tooltip="Image set")
        assert note == f"showing a random sample of 2 of 6 image sets (seed {sampler.seed:016x})"
        assert combo.toolTip() == f"Image set — {note}."
        every = [s.label for s in sampler.sets]
        assert len(_entries(combo)) == 2
        assert set(_entries(combo)) <= set(every)
        assert _entries(combo) == sorted(_entries(combo), key=every.index)
        assert sampler.max_sets == 2

    with subtests.test(msg="cap box clamped"):
        sampler = pc.ImageSetSampler(max_sets=2)
        sampler.enumerate(plate, [".tif"])
        box = pc.FlatSpinBox(value=50)
        combo = QComboBox()
        note = pc.apply_sample_to_combo(combo, box, sampler, None)
        assert (box.suffix(), box.maximum(), box.value(), box.isEnabled()) == (
            " of 6 sets", 6, 6, True)
        assert sampler.max_sets == 6
        assert note == "showing all 6 image sets"
        assert _entries(combo) == [s.label for s in sampler.sets]

    with subtests.test(msg="empty folder, no cap box"):
        empty = tmp_path / "empty"
        empty.mkdir()
        sampler = pc.ImageSetSampler(max_sets=2)
        sampler.enumerate(empty, [".tif"])
        combo = QComboBox()
        note = pc.apply_sample_to_combo(combo, None, sampler, None, tooltip="Image set")
        assert (sampler.total, note, _entries(combo)) == (0, "showing all 0 image sets", [])
        assert combo.toolTip() == "Image set — showing all 0 image sets."
        assert sampler.max_sets == 2


def test_get_regex_callable_lifts_only_a_real_top_level_def(tmp_path, monkeypatch, caplog, subtests):
    """The import-free lift returns spacr.utils' own helper, a lifted top-level def, or None."""
    root = tmp_path / "plate"
    names = _plate(root)

    def stub(name, text):
        path = tmp_path / name
        path.write_text(text, encoding="utf8")
        return SimpleNamespace(util=SimpleNamespace(
            find_spec=lambda _name: SimpleNamespace(origin=str(path))))

    with subtests.test(msg="loaded module is used"):
        import spacr.utils
        _clear_caches()
        assert pc._get_regex_callable() is spacr.utils._get_regex

    with subtests.test(msg="lifted from source without importing"), monkeypatch.context() as m:
        m.delitem(sys.modules, "spacr.utils", raising=False)
        _clear_caches()
        lifted = pc._get_regex_callable()
        assert callable(lifted)
        assert "spacr.utils" not in sys.modules
        assert isinstance(lifted("cellvoyager", "tif", custom_regex=None), str)
        grouped, channels = pc.enumerate_image_sets(root, [".tif"], "cellvoyager")
        assert ([s.label for s in grouped], channels) == (GROUPED, ["01", "02"])

    with subtests.test(msg="a stub top-level def is lifted"), monkeypatch.context() as m:
        m.delitem(sys.modules, "spacr.utils", raising=False)
        m.setattr(pc, "importlib", stub(
            "good_utils.py",
            "def _get_regex(metadata_type, img_format, custom_regex=None):\n"
            f"    return {CUSTOM!r}\n"))
        _clear_caches()
        assert callable(pc._get_regex_callable())
        grouped, channels = pc.enumerate_image_sets(root, [".tif"], "cellvoyager")
        assert ([s.label for s in grouped], channels) == (GROUPED, ["01", "02"])

    unusable = [
        ("no _get_regex in the source",
         lambda m: m.setattr(pc, "importlib", stub(
             "utils_without_the_helper.py",
             "def _something_else():\n    return 'not the regex'\n"))),
        ("async def and a method are not lifted",
         lambda m: m.setattr(pc, "importlib", stub(
             "utils_with_the_helper_moved.py",
             "async def _get_regex(metadata_type, img_format, custom_regex=None):\n"
             "    return 'never awaited'\n\n"
             "class Naming:\n"
             "    def _get_regex(self, metadata_type, img_format, custom_regex=None):\n"
             "        return 'not module level'\n"))),
        ("find_spec returns None",
         lambda m: m.setattr(pc, "importlib", SimpleNamespace(
             util=SimpleNamespace(find_spec=lambda name: None)))),
        ("find_spec raises",
         lambda m: m.setattr(__import__("importlib.util").util, "find_spec",
                             _raise_import_error)),
    ]
    for case_id, patch in unusable:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            m.delitem(sys.modules, "spacr.utils", raising=False)
            patch(m)
            _clear_caches()
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger="spacr.qt.preview_controls"):
                assert pc._get_regex_callable() is None
            assert pc._acquisition_regex("cellvoyager", "tif") is None
            flat, channels = pc.enumerate_image_sets(root, [".tif"], "cellvoyager")
            assert ([s.key for s in flat], channels) == (_flat_keys(names), [])
            if case_id == "find_spec returns None":
                assert any("Could not lift _get_regex" in r.message for r in caplog.records)
    _clear_caches()


def _raise_import_error(_name):
    raise ImportError("spacr.utils cannot be located")


def test_enumerate_image_sets_groups_known_names_and_lists_the_rest(tmp_path, monkeypatch, subtests):
    """Known dialects group by well and field; anything unusable lists one set per image file."""
    plate = tmp_path / "plate"
    names = _plate(plate)
    mixed = tmp_path / "mixed"
    mixed_names = _plate(mixed, extensions=("tif", "png"))

    grouping = [
        ("known dialect", plate, [".tif"], ("cellvoyager",), GROUPED, ["01", "02"]),
        ("closed custom regex", plate, [".tif"], ("custom", CUSTOM), GROUPED, ["01", "02"]),
        ("mixed suffixes, known dialect", mixed, [".tif", ".png"], ("cellvoyager",),
         GROUPED, ["01", "02"]),
    ]
    for case_id, root, suffixes, args, labels, channels in grouping:
        with subtests.test(msg=case_id):
            sets, found = pc.enumerate_image_sets(root, suffixes, *args)
            assert ([s.label for s in sets], found) == (labels, channels)

    flat_cases = [
        ("unknown dialect", plate, [".tif"], ("klingon",), names,
         [("klingon", "tif")]),
        ("unparsable custom regex", plate, [".tif"], ("custom", "(unclosed"), names,
         [("custom", "tif", "(unclosed")]),
        ("unknown dialect, every extension", mixed, [".tif", ".png"], ("klingon",),
         mixed_names, [("klingon", "tif"), ("klingon", "png")]),
    ]
    for case_id, root, suffixes, args, listed, regex_args in flat_cases:
        with subtests.test(msg=case_id):
            sets, found = pc.enumerate_image_sets(root, suffixes, *args)
            assert ([s.key for s in sets], found) == (_flat_keys(listed), [])
            if case_id == "unknown dialect":
                assert [s.label for s in sets] == names
            for regex_call in regex_args:
                assert pc._acquisition_regex(*regex_call) is None

    for case_id, source in [("not a path", object()), ("absent folder", tmp_path / "absent")]:
        with subtests.test(msg=case_id):
            assert pc.enumerate_image_sets(source, (".tif",)) == ([], [])

    with subtests.test(msg="only image files"):
        root = tmp_path / "only_images"
        root.mkdir()
        (root / "plate1_A01_f01_C1.tif").write_bytes(b"x")
        (root / "notes.txt").write_bytes(b"x")
        (root / "subfolder").mkdir()
        (root / "subfolder.tif").mkdir()
        sets, _channels = pc.enumerate_image_sets(root, (".tif",))
        assert {n for item in sets for n in item.channels.values()} == {"plate1_A01_f01_C1.tif"}

    with subtests.test(msg="an entry that cannot be stat-ed"), monkeypatch.context() as m:
        root = tmp_path / "vanishing"
        root.mkdir()
        good = root / "plate1_A01_f01_C1.tif"
        good.write_bytes(b"x")

        class Vanished:
            name = "plate1_A01_f02_C1.tif"

            def is_file(self):
                raise OSError("no such file")

        real_scandir = os.scandir

        class Listing:
            def __init__(self, directory):
                self._real = real_scandir(directory)

            def __enter__(self):
                return list(self._real) + [Vanished()]

            def __exit__(self, *_exc):
                self._real.close()
                return False

        m.setattr(pc.os, "scandir", Listing)
        sets, _channels = pc.enumerate_image_sets(root, (".tif",))
        assert {n for item in sets for n in item.channels.values()} == {good.name}
