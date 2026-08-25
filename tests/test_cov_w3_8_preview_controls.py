"""Preview plumbing: what the dropdowns read, and what a stale pick must not do.

These helpers run before any image is opened, so every one of them is
exercised against a real folder of real file names -- an empty folder, a
folder the process cannot read, names the acquisition regex does not
understand -- rather than against a stand-in.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox  # noqa: E402

from spacr.qt.widgets import preview_controls as pc  # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Which channel is selected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", pc.ALL_CHANNELS, "channel one",
                                  "ch", "chX", "ch 2 of 4"])
def test_a_dropdown_entry_that_names_no_index_means_every_channel(qtbot,
                                                                  text):
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItem(text)
    assert pc.selected_channel(combo) is None


def test_a_numbered_entry_names_its_channel(qtbot):
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItems(["ch0", "ch12"])
    assert pc.selected_channel(combo) == 0
    combo.setCurrentIndex(1)
    assert pc.selected_channel(combo) == 12


# ---------------------------------------------------------------------------
# Reducing an image to one channel
# ---------------------------------------------------------------------------

def test_a_flat_image_is_handed_back_whole():
    flat = np.zeros((4, 5), dtype=np.uint16)
    assert pc.channel_view(flat, 1) is flat


def test_a_channel_the_image_does_not_have_is_not_an_error():
    stack = np.zeros((4, 5, 2), dtype=np.uint16)
    assert pc.channel_view(stack, 7) is stack
    assert pc.channel_view(stack, -1) is stack
    assert pc.channel_view(stack, 1).shape == (4, 5)


def test_a_stale_selection_never_raises_while_a_field_is_loading():
    """The index arrives from a dropdown; it can be anything at all."""
    stack = np.zeros((4, 5, 2), dtype=np.uint16)
    assert pc.channel_view(stack, "not a number") is stack
    assert pc.channel_view(None, 0) is None
    assert pc.channel_view(stack, None) is stack


# ---------------------------------------------------------------------------
# Finding the neighbouring fields
# ---------------------------------------------------------------------------

def test_nothing_loaded_has_no_siblings():
    assert pc.sibling_sources("", (".tif",)) == []
    assert pc.sibling_sources(None, (".tif",)) == []


def test_a_folder_that_cannot_be_listed_offers_only_what_is_open(tmp_path):
    """Execute but not read: the open file still stats, the folder will not
    list. The dropdown must keep the field the user is looking at."""
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


def test_a_file_whose_folder_is_gone_lists_nothing(tmp_path):
    missing = tmp_path / "gone" / "field.tif"
    assert pc.sibling_sources(missing, (".tif",)) == []


def test_only_the_matching_suffixes_count_as_siblings(tmp_path):
    wanted = tmp_path / "b.tif"
    for name in ("a.tif", "b.tif", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "subfolder").mkdir()
    assert pc.sibling_sources(wanted, (".tif",)) == [
        tmp_path / "a.tif", tmp_path / "b.tif"]


def test_folders_are_listed_when_a_field_of_view_is_a_folder(tmp_path):
    for name in ("fov1", "fov2"):
        (tmp_path / name).mkdir()
    (tmp_path / "stray.tif").write_bytes(b"x")
    assert pc.sibling_sources(tmp_path / "fov1", (".tif",),
                              directories=True) == [tmp_path / "fov1",
                                                    tmp_path / "fov2"]


def test_the_open_file_is_listed_even_when_its_suffix_is_not_wanted(tmp_path):
    """Whatever is loaded stays reachable, or the dropdown loses the field."""
    (tmp_path / "a.tif").write_bytes(b"x")
    odd = tmp_path / "loaded.czi"
    odd.write_bytes(b"x")
    listed = pc.sibling_sources(odd, (".tif",))
    assert listed == [tmp_path / "a.tif", odd]


# ---------------------------------------------------------------------------
# Filling the dropdown
# ---------------------------------------------------------------------------

def test_sources_past_the_end_of_the_labels_fall_back_to_their_names(qtbot,
                                                                     tmp_path):
    combo = QComboBox()
    qtbot.addWidget(combo)
    sources = [tmp_path / "one.tif", tmp_path / "two.tif"]
    pc.populate_fov_combo(combo, sources, current=sources[1],
                          labels=["A01 f001"])
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "A01 f001", "two.tif"]
    assert combo.currentIndex() == 1


# ---------------------------------------------------------------------------
# One field of view
# ---------------------------------------------------------------------------

def _image_set(tmp_path, planes=None):
    return pc.ImageSet(
        key=("plate1", "A01", "1"), directory=str(tmp_path),
        channels={"1": "A01_C1_z1.tif", "2": "A01_C2_z1.tif"},
        planes=planes or {})


def test_a_field_with_no_planes_recorded_still_answers_with_its_file(
        tmp_path):
    item = _image_set(tmp_path)
    assert item.plane_paths() == [tmp_path / "A01_C1_z1.tif"]
    assert item.plane_paths("2") == [tmp_path / "A01_C2_z1.tif"]
    assert item.z_count == 1


def test_a_field_with_no_channels_at_all_has_no_planes(tmp_path):
    empty = pc.ImageSet(key=("", "", "x"), directory=str(tmp_path))
    assert empty.plane_paths() == []


def test_a_stack_reports_every_plane_of_the_asked_channel(tmp_path):
    item = _image_set(tmp_path, planes={
        "1": ["A01_C1_z1.tif", "A01_C1_z2.tif", "A01_C1_z3.tif"],
        "2": ["A01_C2_z1.tif"]})
    assert item.plane_paths("1") == [
        tmp_path / f"A01_C1_z{n}.tif" for n in (1, 2, 3)]
    assert item.plane_paths() == item.plane_paths("1")
    assert item.z_count == 3


# ---------------------------------------------------------------------------
# The acquisition regex
# ---------------------------------------------------------------------------

def test_the_pattern_is_lifted_from_the_source_when_spacr_utils_is_unloaded(
        monkeypatch):
    """The Qt layer must not import spacr.utils to learn a filename shape."""
    pc._get_regex_callable.cache_clear()
    monkeypatch.delitem(sys.modules, "spacr.utils", raising=False)
    try:
        lifted = pc._get_regex_callable()
        assert callable(lifted)
        assert "spacr.utils" not in sys.modules
        assert isinstance(lifted("cellvoyager", "tif", custom_regex=None), str)
    finally:
        pc._get_regex_callable.cache_clear()


def test_the_loaded_module_is_used_when_it_is_already_there():
    import spacr.utils  # noqa: F401 - the point is that it is imported

    pc._get_regex_callable.cache_clear()
    try:
        assert pc._get_regex_callable() is spacr.utils._get_regex
    finally:
        pc._get_regex_callable.cache_clear()


def test_a_source_file_that_cannot_be_read_lifts_nothing(monkeypatch):
    import importlib.util

    pc._get_regex_callable.cache_clear()
    monkeypatch.delitem(sys.modules, "spacr.utils", raising=False)

    def refuse(_name):
        raise ImportError("spacr.utils cannot be located")

    monkeypatch.setattr(importlib.util, "find_spec", refuse)
    try:
        assert pc._get_regex_callable() is None
    finally:
        pc._get_regex_callable.cache_clear()


def test_without_a_pattern_source_there_is_no_acquisition_regex(monkeypatch):
    pc._acquisition_regex.cache_clear()
    monkeypatch.setattr(pc, "_get_regex_callable", lambda: None)
    try:
        assert pc._acquisition_regex("cellvoyager", "tif", None) is None
    finally:
        pc._acquisition_regex.cache_clear()


def test_an_unparsable_custom_pattern_degrades_to_no_pattern():
    pc._acquisition_regex.cache_clear()
    try:
        assert pc._acquisition_regex("custom", "tif", "([unclosed") is None
    finally:
        pc._acquisition_regex.cache_clear()


# ---------------------------------------------------------------------------
# Enumerating a folder
# ---------------------------------------------------------------------------

def test_something_that_is_not_a_path_enumerates_nothing():
    assert pc.enumerate_image_sets(object(), (".tif",)) == ([], [])


def test_a_folder_that_is_not_there_enumerates_nothing(tmp_path):
    assert pc.enumerate_image_sets(tmp_path / "absent", (".tif",)) == ([], [])


def test_only_image_files_are_enumerated(tmp_path):
    (tmp_path / "plate1_A01_f01_C1.tif").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / "subfolder").mkdir()
    (tmp_path / "subfolder.tif").mkdir()
    sets, _channels = pc.enumerate_image_sets(tmp_path, (".tif",))
    names = {name for item in sets for name in item.channels.values()}
    assert names == {"plate1_A01_f01_C1.tif"}


def test_a_directory_entry_that_cannot_be_stat_ed_is_skipped(tmp_path,
                                                             monkeypatch):
    """A name that vanishes mid-scan must cost that name, not the folder."""
    good = tmp_path / "plate1_A01_f01_C1.tif"
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

    monkeypatch.setattr(pc.os, "scandir", Listing)
    sets, _channels = pc.enumerate_image_sets(tmp_path, (".tif",))
    names = {name for item in sets for name in item.channels.values()}
    assert names == {good.name}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sampler(tmp_path, item):
    sampler = pc.ImageSetSampler()
    sampler.adopt(str(tmp_path), [item], ["1", "2"])
    return sampler


def test_no_path_belongs_to_no_set(tmp_path):
    sampler = _sampler(tmp_path, _image_set(tmp_path))
    assert sampler.set_for_path(None) is None


def test_a_path_is_matched_to_the_set_that_holds_it(tmp_path):
    item = _image_set(tmp_path)
    sampler = _sampler(tmp_path, item)
    assert sampler.set_for_path(tmp_path / "A01_C2_z1.tif") is item
    assert sampler.set_for_path(tmp_path / "not_enumerated.tif") is None
