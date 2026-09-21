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


def test_every_dataset_names_its_folders_and_its_model():
    assert md.MASK_DATASETS
    for dataset in md.MASK_DATASETS:
        assert dataset.images
        assert dataset.repo.startswith("einarolafsson/")
        assert dataset.model
        assert dataset.apps
    with_masks = [d for d in md.MASK_DATASETS if d.masks]
    assert len(with_masks) == len(md.MASK_DATASETS) - 1, (
        "exactly one set is images-only: the plaque figures the pipeline takes")


def test_the_cellmask_set_keeps_its_masks_under_masks_pv():
    """The one that would have silently produced images and no labels.

    cross-channel-toxoplasma-from-cellmask calls the folder masks_pv. Assuming "masks"
    for every repo gives ten images, no masks, and no error to explain it.
    """
    dataset = md.DATASETS_BY_KEY["toxoplasma_from_cellmask"]
    assert dataset.masks == "masks_pv"
    assert all(d.masks == "masks" for d in md.MASK_DATASETS
               if d.key not in ("toxoplasma_from_cellmask", "plaque_figures"))


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


def test_each_module_is_offered_only_its_own_sets():
    """Item 451: the plaque module gained its own two, and Make Masks kept its five."""
    mask = md.datasets_for("mask")
    plaque = md.datasets_for("analyze_plaques")
    assert len(mask) == 5 and len(plaque) == 2
    assert {d.key for d in plaque} == {"toxoplasma_plaque", "plaque_figures"}
    assert "plaque_figures" not in {d.key for d in mask}, (
        "the figures have no masks; Make Masks would open ten blank fields")
    assert "toxoplasma_plaque" in {d.key for d in mask}, (
        "the segmented plaque fields belong to both: curated in one, run in the other")


def test_a_dataset_with_no_masks_folder_samples_images_alone():
    dataset = md.DATASETS_BY_KEY["plaque_figures"]
    assert dataset.masks == ""
    listing = [f"images/p{i:02d}.jpg" for i in range(20)] + \
              [f"labels/p{i:02d}.txt" for i in range(20)]
    picked = md.choose_sample(dataset, listing)
    assert len(picked) == md.SAMPLE_SIZE
    assert all(mask == "" for _image, mask in picked), (
        "labels/ holds YOLO boxes, and a box must never be handed over as a mask")


def test_an_images_only_sample_counts_as_present_without_a_masks_folder(tmp_path):
    for i in range(md.SAMPLE_SIZE):
        (tmp_path / f"p{i}.jpg").write_bytes(b"x")
    assert md.is_present(tmp_path)


def test_the_module_screen_points_src_at_the_sample_instead_of_opening_it(tmp_path):
    """A module does not open a folder, it runs on one."""
    dataset = md.DATASETS_BY_KEY["plaque_figures"]
    folder = md.sample_folder(tmp_path, dataset)
    folder.mkdir(parents=True)
    for i in range(md.SAMPLE_SIZE):
        (folder / f"p{i}.jpg").write_bytes(b"x")
    used = []

    class Screen:
        _status_label = None

        def _open_folder(self, path):
            raise AssertionError("a module screen has no _open_folder")

    ok = md.open_a_training_dataset(
        Screen(), pick=lambda _s: dataset, root=tmp_path,
        app_key="analyze_plaques", use=lambda p: (used.append(p), True)[1])
    assert ok is True and used == [folder]


def test_the_plaque_module_has_a_test_data_button_at_all(qtbot, qt_theme_applied):
    """It had none: analyze_plaques was missing from EXAMPLE_DATA_SECTIONS, so the
    dispatch that installs these buttons never reached it and nothing was built."""
    from spacr.qt.screens.app_screen import AppScreen, EXAMPLE_DATA_SECTIONS

    assert EXAMPLE_DATA_SECTIONS.get("analyze_plaques") == "Input & Channels"
    screen = AppScreen("analyze_plaques")
    qtbot.addWidget(screen)
    button = getattr(screen, "_plaque_example_button", None)
    assert button is not None
    assert button.text() == "Load test data…"
    assert hasattr(screen, "point_src_at")


def _plaque_listing():
    """A listing shaped like toxoplasma-plaque-dataset v5, domain by domain."""
    counts = {"literature": 298, "malnio": 96, "patrick": 67, "bigbean": 27}
    listing = []
    for domain, n in counts.items():
        for i in range(n):
            stem = f"{domain}__field{i:03d}"
            listing += [f"images/{stem}.tif", f"masks/{stem}.tif"]
    return listing


def test_the_plaque_sample_is_twenty_fields_split_by_domain():
    """2026-09-21, the maintainer: "20 images total ... 40% patrick, 20% bigbean,
    20 % malnio, 20% literature". The first ten by name were all bigbean."""
    dataset = md.DATASETS_BY_KEY["toxoplasma_plaque"]
    picked = md.choose_sample(dataset, _plaque_listing())
    domains = [md.Path(image).name.split("__")[0] for image, _mask in picked]
    assert dataset.size == len(picked) == 20
    assert {d: domains.count(d) for d in set(domains)} == {
        "patrick": 8, "bigbean": 4, "malnio": 4, "literature": 4}
    assert all(mask for _image, mask in picked)


def test_a_domain_is_sampled_across_it_and_not_from_its_start():
    dataset = md.DATASETS_BY_KEY["toxoplasma_plaque"]
    picked = md.choose_sample(dataset, _plaque_listing())
    patrick = sorted(image for image, _m in picked if "patrick__" in image)
    assert patrick[0] != "images/patrick__field000.tif"
    assert patrick[-1] >= "images/patrick__field050.tif"


def test_the_old_ten_field_plaque_cache_is_not_reopened(tmp_path):
    dataset = md.DATASETS_BY_KEY["toxoplasma_plaque"]
    old = tmp_path / "mask_datasets" / "toxoplasma_plaque"
    assert md.sample_folder(tmp_path, dataset) != old
    for i in range(10):
        (old / "masks").mkdir(parents=True, exist_ok=True)
        (old / f"bigbean__{i}.tif").write_bytes(b"x")
        (old / "masks" / f"bigbean__{i}.tif").write_bytes(b"x")
    assert not md.is_present(md.sample_folder(tmp_path, dataset), dataset.size)
