"""Item 448: the keep/discard CSV, driven at the engine.

The four columns are the maintainer's, in his order, and the layout is the
one he set on 2026-09-20: masks at ``<images>/masks`` and the curation CSV
at ``<images>/csv``.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from spacr.qt import mask_engine as engine


def _rows(folder):
    with open(engine.curation_csv_path(folder), encoding="utf-8",
              newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_csv_lives_in_a_csv_folder_beside_the_images(tmp_path):
    folder = str(tmp_path / "images")
    assert engine.curation_folder(folder) == os.path.join(folder, "csv")
    assert engine.curation_csv_path(folder) == os.path.join(
        folder, "csv", "keep_discard.csv")


def test_the_masks_stay_with_the_images(tmp_path):
    """The other half of the same layout."""
    folder = str(tmp_path / "images")
    assert engine.masks_folder(folder) == os.path.join(folder, "masks")
    assert engine.mask_save_path(folder, "field.tif") == os.path.join(
        folder, "masks", "field.tif")


def test_a_verdict_is_written_with_the_four_columns_asked_for(tmp_path):
    folder = str(tmp_path / "images")
    os.makedirs(folder)
    image = os.path.join(folder, "field.tif")
    mask = engine.mask_save_path(folder, "field.tif")
    written = engine.record_curation(folder, image, mask, 17, True)
    assert written == engine.curation_csv_path(folder)
    rows = _rows(folder)
    assert len(rows) == 1
    assert list(rows[0]) == list(engine.CURATION_COLUMNS)
    assert rows[0]["image path"] == image
    assert rows[0]["mask path"] == mask
    assert rows[0]["object count"] == "17"
    assert rows[0]["keep"] == "true"


def test_keep_then_discard_leaves_one_row_with_the_later_verdict(tmp_path):
    """Two rows that disagree are worse than no file."""
    folder = str(tmp_path / "images")
    os.makedirs(folder)
    image = os.path.join(folder, "field.tif")
    mask = engine.mask_save_path(folder, "field.tif")
    engine.record_curation(folder, image, mask, 17, True)
    engine.record_curation(folder, image, mask, 12, False)
    rows = _rows(folder)
    assert len(rows) == 1
    assert rows[0]["keep"] == "false"
    assert rows[0]["object count"] == "12"
    assert engine.curation_verdict(folder, image) is False


def test_two_fields_keep_a_row_each(tmp_path):
    folder = str(tmp_path / "images")
    os.makedirs(folder)
    for name, keep in (("a.tif", True), ("b.tif", False)):
        engine.record_curation(folder, os.path.join(folder, name),
                               engine.mask_save_path(folder, name), 3, keep)
    rows = _rows(folder)
    assert [row["keep"] for row in rows] == ["true", "false"]


def test_no_file_means_no_verdict_rather_than_an_error(tmp_path):
    folder = str(tmp_path / "images")
    os.makedirs(folder)
    assert engine.read_curation(folder) == {}
    assert engine.curation_verdict(folder, "anything.tif") is None


def test_a_csv_somebody_broke_by_hand_costs_the_verdicts_not_the_folder(
        tmp_path):
    """A curation aid must not be able to stop somebody opening a folder."""
    folder = str(tmp_path / "images")
    os.makedirs(engine.curation_folder(folder))
    with open(engine.curation_csv_path(folder), "w", encoding="utf-8") as h:
        h.write("this is not a csv\x00\x00\n")
    assert engine.read_curation(folder) == {}
    engine.record_curation(folder, os.path.join(folder, "a.tif"),
                           engine.mask_save_path(folder, "a.tif"), 1, True)
    assert len(_rows(folder)) == 1


def test_the_file_is_never_seen_half_written(tmp_path, monkeypatch):
    """Written to a dot-name and renamed over, like the bundles are."""
    folder = str(tmp_path / "images")
    os.makedirs(folder)
    image = os.path.join(folder, "field.tif")
    engine.record_curation(folder, image,
                           engine.mask_save_path(folder, "field.tif"), 2, True)
    seen = []
    original = os.replace

    def _watch(source, target):
        seen.append((os.path.basename(source), os.path.basename(target)))
        return original(source, target)

    monkeypatch.setattr(os, "replace", _watch)
    engine.record_curation(folder, image,
                           engine.mask_save_path(folder, "field.tif"), 5, False)
    assert seen, "the rewrite did not go through os.replace"
    source, target = seen[-1]
    assert source.startswith(".") and target == engine.CURATION_CSV_NAME
    assert not any(name.startswith(".") and name.endswith(".tmp")
                   for name in os.listdir(engine.curation_folder(folder)))


def test_the_count_recorded_is_the_one_it_was_given(tmp_path):
    """The screen reads the count when the button is pressed; the engine
    must not second-guess it."""
    folder = str(tmp_path / "images")
    os.makedirs(folder)
    labels = np.zeros((8, 8), dtype=np.uint16)
    labels[0, 0], labels[2, 2], labels[4, 4] = 1, 5, 9
    from spacr.qt.screens.make_masks import _object_count

    count = _object_count(labels)
    assert count == 3
    engine.record_curation(folder, os.path.join(folder, "f.tif"),
                           engine.mask_save_path(folder, "f.tif"), count, True)
    assert _rows(folder)[0]["object count"] == "3"
