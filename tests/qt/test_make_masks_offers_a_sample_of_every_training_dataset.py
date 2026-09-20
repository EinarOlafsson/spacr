"""Item 450: a sample of each model's training dataset, one dialog away.

Asked for on 2026-09-20: "there should be multiple datasets for the user to choose
from, a small sample of each dataset a model is trained on where I have the data on my
huggingface."

NOTHING HERE TOUCHES THE NETWORK. The picker and the fetch are both injectable for
exactly that reason -- a test that needs Hugging Face is a test that fails on a train.
What is held is the part that can be wrong without anyone noticing: which two folders a
repository keeps its images and masks in, that the sample is the same ten fields on
every machine, and that a half-downloaded sample is treated as absent.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import make_masks_datasets as md


def test_every_dataset_names_both_of_its_folders():
    assert md.MASK_DATASETS
    for dataset in md.MASK_DATASETS:
        assert dataset.images and dataset.masks
        assert dataset.repo.startswith("einarolafsson/")
        assert dataset.model


def test_the_cellmask_set_keeps_its_masks_under_masks_pv():
    """The one that would have silently produced images and no labels.

    cross-channel-toxoplasma-from-cellmask calls the folder masks_pv. Assuming "masks"
    for every repo gives ten images, no masks, and no error to explain it.
    """
    dataset = md.DATASETS_BY_KEY["toxoplasma_from_cellmask"]
    assert dataset.masks == "masks_pv"
    assert all(d.masks == "masks" for d in md.MASK_DATASETS
               if d.key != "toxoplasma_from_cellmask")


def test_the_sample_pairs_on_the_stem_and_is_the_same_ten_everywhere():
    dataset = md.DATASETS_BY_KEY["toxoplasma_pv"]
    listing = ["README.md", "fields.csv"]
    listing += [f"images/f{i:03d}.tif" for i in range(30)]
    listing += [f"masks/f{i:03d}.tif" for i in range(30)]
    picked = md.choose_sample(dataset, listing)
    assert len(picked) == md.SAMPLE_SIZE
    assert picked[0] == ("images/f000.tif", "masks/f000.tif")
    assert picked == md.choose_sample(dataset, list(reversed(listing))), (
        "the sample must not depend on the order the repository lists its files")


def test_a_field_with_no_mask_is_not_sampled():
    dataset = md.DATASETS_BY_KEY["toxoplasma_pv"]
    listing = [f"images/f{i:03d}.tif" for i in range(12)]
    listing += [f"masks/f{i:03d}.tif" for i in range(4)]
    picked = md.choose_sample(dataset, listing)
    assert len(picked) == 4


def test_the_masks_pv_folder_is_actually_used(tmp_path):
    dataset = md.DATASETS_BY_KEY["toxoplasma_from_cellmask"]
    listing = [f"images/f{i}.tif" for i in range(3)] + \
              [f"masks_pv/f{i}.tif" for i in range(3)]
    picked = md.choose_sample(dataset, listing)
    assert picked == [(f"images/f{i}.tif", f"masks_pv/f{i}.tif") for i in range(3)]
    assert md.choose_sample(md.DATASETS_BY_KEY["toxoplasma_pv"], listing) == []


def _make_sample(folder, n):
    (folder / "masks").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"f{i}.tif").write_bytes(b"x")
        (folder / "masks" / f"f{i}.tif").write_bytes(b"x")


def test_a_half_downloaded_sample_reads_as_absent(tmp_path):
    _make_sample(tmp_path, 4)
    assert not md.is_present(tmp_path)
    _make_sample(tmp_path, md.SAMPLE_SIZE)
    assert md.is_present(tmp_path)


def test_a_cached_sample_opens_without_fetching(tmp_path):
    dataset = md.MASK_DATASETS[0]
    folder = md.sample_folder(tmp_path, dataset)
    _make_sample(folder, md.SAMPLE_SIZE)

    opened, fetched = [], []

    class Screen:
        _status_label = None

        def _open_folder(self, path):
            opened.append(path)
            return True

    ok = md.open_a_training_dataset(
        Screen(), pick=lambda _s: dataset,
        fetch=lambda *a: fetched.append(a), root=tmp_path)
    assert ok is True
    assert opened == [str(folder)]
    assert fetched == [], "a cached sample must not ask the network"


def test_choosing_nothing_does_nothing(tmp_path):
    fetched = []
    ok = md.open_a_training_dataset(
        object(), pick=lambda _s: None,
        fetch=lambda *a: fetched.append(a), root=tmp_path)
    assert ok is False and fetched == []


def test_an_absent_sample_is_fetched_then_opened(tmp_path):
    dataset = md.MASK_DATASETS[0]
    folder = md.sample_folder(tmp_path, dataset)
    opened = []

    class Screen:
        _status_label = None

        def _open_folder(self, path):
            opened.append(path)
            return True

    screen = Screen()
    captured = {}

    def fetch(_screen, _dataset, where, on_done):
        captured["where"] = where
        on_done(str(where), "")

    ok = md.open_a_training_dataset(screen, pick=lambda _s: dataset,
                                    fetch=fetch, root=tmp_path)
    assert ok is False, "a download opens later, not from this call"
    assert captured["where"] == folder
    assert opened == [str(folder)]


def test_a_failed_fetch_says_why_and_opens_nothing(tmp_path):
    dataset = md.MASK_DATASETS[0]
    said = []

    class Screen:
        class _Label:
            def setText(self, text):
                said.append(text)
        _status_label = _Label()

        def _open_folder(self, path):
            raise AssertionError("nothing should be opened after a failure")

    md.open_a_training_dataset(
        Screen(), pick=lambda _s: dataset,
        fetch=lambda _s, _d, _w, done: done(None, "no network"),
        root=tmp_path)
    assert any("no network" in text for text in said)


def test_the_button_is_on_the_make_masks_screen(qtbot, qt_theme_applied):
    from spacr.qt.screens.make_masks import MakeMasksScreen

    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    assert getattr(screen, "_btn_training_datasets", None) is not None
    assert getattr(screen, "_btn_test_data", None) is not None, (
        "item 412's button must survive; the two answer different questions")
    screen._magnifier.close()
    screen.close_folded()
