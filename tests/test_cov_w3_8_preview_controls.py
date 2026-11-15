"""Preview plumbing: what the dropdowns read, and what a stale pick must not do.

These helpers run before any image is opened, so every one of them is
exercised against a real folder of real file names -- an empty folder, a
folder the process cannot read, names the acquisition regex does not
understand -- rather than against a stand-in.
"""
from __future__ import annotations


import pytest

pytest.importorskip("PySide6")


from spacr.qt.widgets import preview_controls as pc  # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# One field of view
# ---------------------------------------------------------------------------

def _image_set(tmp_path, planes=None):
    return pc.ImageSet(
        key=("plate1", "A01", "1"), directory=str(tmp_path),
        channels={"1": "A01_C1_z1.tif", "2": "A01_C2_z1.tif"},
        planes=planes or {})


# ---------------------------------------------------------------------------
# The acquisition regex
# ---------------------------------------------------------------------------


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


def test_listing_cache_evicts_the_oldest_folder(tmp_path, monkeypatch):
    """Browsing many plates keeps recent listings within the cache bound."""
    monkeypatch.setattr(pc, "_LISTING_SETTLE_S", -1.0)
    pc._LISTINGS.clear()
    try:
        folders = []
        for index in range(pc._LISTINGS_KEPT + 1):
            folder = tmp_path / f"plate-{index}"
            folder.mkdir()
            (folder / f"field-{index}.tif").write_bytes(b"x")
            folders.append(folder)
            assert pc._file_names(folder) == (f"field-{index}.tif",)
        assert len(pc._LISTINGS) == pc._LISTINGS_KEPT
        assert str(folders[0]) not in pc._LISTINGS
        assert str(folders[-1]) in pc._LISTINGS
    finally:
        pc._LISTINGS.clear()


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
