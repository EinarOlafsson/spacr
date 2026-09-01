"""Annotate and Classify can fetch a working example, settings and all.

Both modules need what a Measure run produces -- crops, a measurements
database that indexes them, and LABELS. Without labels a "classify example" is
only a viewer demo, so the published set carries 88 real annotations.

THE PATHS ARE THE HARD PART. A measurements database stores absolute paths to
its crops, which name the machine that made it and resolve nowhere else. The
published copy stores them relative to the dataset root; the downloader turns
them back into paths that open, wherever it was unpacked.

Instruction 332's sibling, reported 2026-09-01: "when these test datasets are
downloaded they should come with a new settings files that should be downloaded
with the datasets and wired in to the spacr settings for the user so they can
essentially just click run".
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from spacr.qt.hf_download import (ANNOTATE_EXAMPLE_REPO, DATASET_PLACEHOLDER,
                                  make_the_example_paths_absolute)


@pytest.fixture
def unpacked(tmp_path):
    """A dataset as it arrives: relative paths throughout."""
    (tmp_path / "settings").mkdir()
    connection = sqlite3.connect(tmp_path / "measurements.db")
    connection.execute("create table png_list (png_path text, note text)")
    connection.executemany(
        "insert into png_list values (?, ?)",
        [("data/a/one.png", "a note, not a path"),
         ("data/a/two.png", "another"),
         ("/already/absolute.png", "left alone")])
    connection.commit()
    connection.close()
    (tmp_path / "settings" / "annotate_settings.csv").write_text(
        f"Key,Value\nsrc,{DATASET_PLACEHOLDER}/measurements.db\n")
    (tmp_path / "settings" / "classify_settings.csv").write_text(
        f"Key,Value\nsrc,{DATASET_PLACEHOLDER}\n")
    return tmp_path


def _paths(root):
    connection = sqlite3.connect(root / "measurements.db")
    try:
        return [r[0] for r in connection.execute("select png_path from png_list")]
    finally:
        connection.close()


def test_relative_crop_paths_become_absolute(unpacked):
    make_the_example_paths_absolute(unpacked)
    assert str(unpacked / "data/a/one.png") in _paths(unpacked)


def test_an_already_absolute_path_is_not_prefixed_twice(unpacked):
    """The failure this guards: /home/me/data//already/absolute.png."""
    make_the_example_paths_absolute(unpacked)
    assert "/already/absolute.png" in _paths(unpacked)


def test_running_it_twice_changes_nothing(unpacked):
    """A re-download over an existing copy must not double the prefix."""
    make_the_example_paths_absolute(unpacked)
    once = _paths(unpacked)
    make_the_example_paths_absolute(unpacked)
    assert _paths(unpacked) == once


def test_prose_columns_are_left_alone(unpacked):
    """Only values that look like our relative paths are touched."""
    make_the_example_paths_absolute(unpacked)
    connection = sqlite3.connect(unpacked / "measurements.db")
    notes = [r[0] for r in connection.execute("select note from png_list")]
    connection.close()
    assert notes == ["a note, not a path", "another", "left alone"]


def test_the_settings_placeholder_is_substituted(unpacked):
    """So the user can press Run without first editing a path."""
    make_the_example_paths_absolute(unpacked)
    text = (unpacked / "settings" / "annotate_settings.csv").read_text()
    assert DATASET_PLACEHOLDER not in text
    assert str(unpacked) in text


def test_both_modules_settings_are_rewritten(unpacked):
    make_the_example_paths_absolute(unpacked)
    for name in ("annotate_settings.csv", "classify_settings.csv"):
        text = (unpacked / "settings" / name).read_text()
        assert DATASET_PLACEHOLDER not in text, name


def test_a_missing_database_is_not_an_error(tmp_path):
    """A cancelled download leaves a folder with no database in it."""
    make_the_example_paths_absolute(tmp_path)


def test_it_reports_how_much_it_changed(unpacked):
    assert make_the_example_paths_absolute(unpacked) > 0


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_classify_offers_the_button_where_its_labels_are():
    from spacr.qt.screens.app_screen import EXAMPLE_DATA_SECTIONS

    assert EXAMPLE_DATA_SECTIONS["classify"] == "Labels & Classes"


def test_the_two_modules_share_one_download():
    """One dataset. Downloading it twice would cost 280 MB twice, and let the
    two copies drift. The annotate half lives on the settings dialog, which is
    where the source field it fills is."""
    from spacr.qt.screens.annotate import _SettingsDialog
    from spacr.qt.screens.app_screen import AppScreen

    # Called unbound on a bare stand-in: a QWidget cannot be built with
    # object.__new__, and neither method touches anything but `Path.home()`.
    class _Stand:
        pass

    annotate_dir = _SettingsDialog.example_destination(_Stand())
    classify_dir = AppScreen.annotate_example_destination(_Stand())
    assert annotate_dir == classify_dir


def test_the_settings_land_through_the_real_import_path():
    """A second reader would drift from "Import settings…", and the point of
    shipping settings is that they land exactly as a user's own file would."""
    source = Path(
        __import__("spacr.qt.screens.app_screen", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    assert "loaded = self._load_settings_csv(str(path))" in source
    assert "applied = self.apply_settings_dict(loaded)" in source


def test_the_cached_copy_is_reused(tmp_path):
    """The DATABASE is the test, not the folder: a cancelled download leaves
    the folder behind."""
    from spacr.qt.screens.annotate import _SettingsDialog

    class _Screen:
        example_destination = lambda self: tmp_path
        _use_the_example_data = lambda self, dest: "reused"
        _load_the_example_data = _SettingsDialog._load_the_example_data

    screen = _Screen()
    (tmp_path / "measurements.db").write_bytes(b"")

    result = screen._load_the_example_data(
        ask=lambda *a, **k: pytest.fail("it re-downloaded"))
    assert result == "reused"


def test_an_empty_folder_still_downloads(tmp_path):
    from spacr.qt.screens.annotate import _SettingsDialog

    class _Screen:
        _example_btn = None
        example_destination = lambda self: tmp_path
        _load_the_example_data = _SettingsDialog._load_the_example_data

    asked = []
    _Screen()._load_the_example_data(ask=lambda *a, **k: asked.append(True))
    assert asked


def test_the_repo_is_named_once():
    assert ANNOTATE_EXAMPLE_REPO == "einarolafsson/spacr-example-annotate"
