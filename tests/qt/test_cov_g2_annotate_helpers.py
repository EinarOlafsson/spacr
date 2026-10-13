"""Annotate screen helpers given blank, odd or unreadable input."""
from __future__ import annotations

import os
import pathlib

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import annotate as an  # noqa: E402


def test_blank_folders_are_neither_probed_nor_vouched_for(monkeypatch):
    probed = []
    monkeypatch.setattr(an.path_probe, "isdir", lambda text: probed.append(text))
    assert an._vouch_later("  ") is None
    assert an._probe_isdir("") is False
    assert an._ask_about_the_folder(None) is None
    assert probed == []


@pytest.mark.parametrize("verdict, value", [("odd", 1), (0, 1), (0, None)])
def test_an_unreadable_or_zero_verdict_counts_as_a_contradiction(verdict, value):
    assert an.verdict_contradicts(verdict, value) is True


def test_a_database_beside_the_plate_names_its_folder(tmp_path):
    database = tmp_path / "plate" / "measurements.db"
    assert an._plate_of_source(str(database)) == str(tmp_path / "plate")


def test_an_unreadable_settings_file_reads_as_empty(tmp_path, monkeypatch):
    path = tmp_path / "annotate_settings.csv"
    path.write_text("Key,Value\nsrc,/x\n")

    def broken(self, *a, **k):
        raise OSError("locked")

    monkeypatch.setattr(pathlib.Path, "open", broken)
    assert an._read_example_settings(path) == {}


def test_a_crop_path_without_a_file_name_maps_only_its_full_path():
    lookup = an._blind_lookup({"folder" + os.sep: "C1"}, "")
    assert "" not in lookup and lookup["folder" + os.sep] == "C1"


def test_a_field_view_of_an_empty_folder_says_so(qtbot, tmp_path):
    from spacr.image_quality import REPORT

    report = tmp_path / REPORT
    report.parent.mkdir(parents=True)
    report.write_text("{not json")
    dialog = an._FieldQCDialog()
    qtbot.addWidget(dialog)
    assert dialog.load_folder(str(tmp_path)) == 0
    dialog.show_field(0)
    assert "No .npy fields" in dialog._caption.text()
    assert dialog.save() == ""
    assert "Nothing labelled" in dialog._status.text()
