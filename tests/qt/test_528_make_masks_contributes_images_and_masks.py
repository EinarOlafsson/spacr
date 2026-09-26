"""Item 528: Make Masks sends images and their masks to a community dataset.

Two ways in, both driven offscreen with the upload faked (nothing leaves the
machine): an images folder plus a masks folder matched by file name, and
one click from curation that sends the image on screen with its mask. The
user names the dataset, which becomes ``einarolafsson/community_<name>``;
a missing or misnamed mask blocks the upload and names the files; the 523
consent and conscience text apply.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import model_share  # noqa: E402
from spacr.qt.widgets import model_share_dialog as msd  # noqa: E402


class _Prefs:
    def __init__(self):
        self.store = {}

    def value(self, key, default=None):
        return self.store.get(key, default)

    def setValue(self, key, value):
        self.store[key] = value


class _Upload:
    def __init__(self):
        self.calls = []

    def __call__(self, folder, target):
        self.calls.append((Path(folder), target))
        repo = model_share.community_repo(target)
        return f"https://huggingface.co/datasets/{repo}/discussions/1"


@pytest.fixture
def prefs(monkeypatch):
    store = _Prefs()
    monkeypatch.setattr(msd, "_preferences", lambda: store)
    return store


def _agree(dialog):
    asked = []

    def ask():
        asked.append(1)
        return True

    dialog.ask_consent = ask
    return asked


def _pair_folders(tmp_path, names=("a", "b", "c"), mask_names=None):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    rng = np.random.default_rng(0)
    for name in names:
        imageio.imwrite(images / f"{name}.png",
                        rng.integers(0, 255, (40, 50), dtype=np.uint8))
    for name in (names if mask_names is None else mask_names):
        mask = np.zeros((40, 50), np.uint16)
        mask[5:15, 5:15] = 1
        mask[20:30, 25:40] = 2
        imageio.imwrite(masks / f"{name}.tif", mask)
    return images, masks


def test_the_name_maps_to_its_own_community_dataset():
    assert model_share.masks_dataset_target("Toxoplasma vacuoles, GFP") == \
        "community_toxoplasma_vacuoles_gfp"
    assert model_share.community_repo(
        model_share.masks_dataset_target("Toxoplasma vacuoles, GFP")) == \
        "einarolafsson/community_toxoplasma_vacuoles_gfp"
    assert model_share.community_repo(
        model_share.masks_dataset_target("plaques")) == \
        "einarolafsson/community_plaques"
    assert model_share.masks_dataset_target("community_hela") == \
        "community_hela"
    with pytest.raises(ValueError):
        model_share.masks_dataset_target("  ?? ")


def test_matched_folders_upload_to_community_name(qtbot, prefs, tmp_path):
    import tifffile

    images, masks = _pair_folders(tmp_path)
    upload = _Upload()
    dialog = msd.ContributeMasksDialog(upload=upload, threaded=False)
    qtbot.addWidget(dialog)
    assert dialog.source() == msd.FOLDERS_SOURCE
    assert not dialog.upload_button.isEnabled()
    dialog.images_edit.setText(str(images))
    assert not dialog.upload_button.isEnabled()
    dialog.masks_edit.setText(str(masks))
    assert not dialog.upload_button.isEnabled()
    dialog.name_edit.setText("HeLa nuclei DAPI")
    assert dialog.upload_button.isEnabled()
    assert "einarolafsson/community_hela_nuclei_dapi" in \
        dialog.target_label.text()
    dialog.notes_edit.setPlainText("20x widefield")
    asked = _agree(dialog)

    assert dialog.upload() is True
    assert asked == [1]
    assert upload.calls == [(dialog.contribution_folder,
                             "community_hela_nuclei_dapi")]
    folder = dialog.contribution_folder
    record = json.loads((folder / "contribution.json").read_text())
    assert record["repo"] == "einarolafsson/community_hela_nuclei_dapi"
    assert record["images"] == 3 and record["objects"] == 6
    assert record["notes"] == "20x widefield"
    assert sorted(p.name for p in (folder / "images").iterdir()) == \
        ["a.png", "b.png", "c.png"]
    assert sorted(p.name for p in (folder / "masks").iterdir()) == \
        ["a.tif", "b.tif", "c.tif"]
    assert (folder / "images" / "a.png").read_bytes() == \
        (images / "a.png").read_bytes()
    assert tifffile.imread(str(folder / "masks" / "b.tif")).max() == 2
    meta = json.loads((folder / "meta" / "a.json").read_text())
    assert meta["dataset"] == "HeLa nuclei DAPI"
    assert meta["mask_file"] == "a.tif"
    assert "discussions/1" in dialog.status.text()
    assert prefs.store[msd.LAST_MASKS_DATASET_KEY] == "HeLa nuclei DAPI"


@pytest.mark.parametrize("mask_names, blocked", [
    (("a", "b"), ["c.png"]),
    (("a", "b", "x"), ["c.png", "x.tif"]),
])
def test_a_missing_or_misnamed_mask_blocks_and_names_the_files(
        qtbot, prefs, tmp_path, mask_names, blocked):
    images, masks = _pair_folders(tmp_path, mask_names=mask_names)
    upload = _Upload()
    dialog = msd.ContributeMasksDialog(images_dir=str(images),
                                       masks_dir=str(masks), name="hela",
                                       upload=upload, threaded=False)
    qtbot.addWidget(dialog)
    _agree(dialog)
    assert not dialog.upload_button.isEnabled()
    for name in blocked:
        assert name in dialog.check.text()
    assert dialog.upload() is False
    assert upload.calls == []
    assert dialog.contribution_folder is None
    for name in blocked:
        assert name in dialog.status.text()


def test_same_folder_and_empty_masks_are_refused(qtbot, prefs, tmp_path):
    images, masks = _pair_folders(tmp_path)
    upload = _Upload()
    dialog = msd.ContributeMasksDialog(images_dir=str(images),
                                       masks_dir=str(images), name="hela",
                                       upload=upload, threaded=False)
    qtbot.addWidget(dialog)
    _agree(dialog)
    assert not dialog.upload_button.isEnabled()
    imageio.imwrite(masks / "b.tif", np.zeros((40, 50), np.uint16))
    dialog.masks_edit.setText(str(masks))
    assert dialog.upload_button.isEnabled()
    assert dialog.upload() is False
    assert "b.png" in dialog.status.text()
    assert upload.calls == []


def test_a_mask_of_another_size_is_refused_by_name(qtbot, prefs, tmp_path):
    import tifffile

    images, masks = _pair_folders(tmp_path)
    small = np.zeros((30, 50), np.uint16)
    small[5:15, 5:15] = 1
    imageio.imwrite(masks / "b.tif", small)
    (images / "c.png").unlink()
    tifffile.imwrite(str(images / "c.tif"),
                     np.zeros((2, 40, 50), np.uint16), metadata={"axes": "CYX"})
    upload = _Upload()
    dialog = msd.ContributeMasksDialog(images_dir=str(images),
                                       masks_dir=str(masks), name="hela",
                                       upload=upload, threaded=False)
    qtbot.addWidget(dialog)
    _agree(dialog)
    assert dialog.upload_button.isEnabled()
    assert dialog.upload() is False
    text = dialog.status.text()
    assert "not the same size" in text
    assert "b.png: image 50 x 40, mask 50 x 30" in text
    assert "a.png" not in text and "c.tif" not in text
    assert upload.calls == [] and dialog.contribution_folder is None

    imageio.imwrite(masks / "b.tif", np.pad(small, ((0, 10), (0, 0))))
    assert dialog.upload() is True
    assert len(upload.calls) == 1


def test_the_size_check_guards_every_masks_contribution(tmp_path):
    source = tmp_path / "field.png"
    imageio.imwrite(source, np.zeros((40, 50), np.uint8))
    labels = np.zeros((50, 40), np.uint16)
    labels[1:5, 1:5] = 1
    item = {"name": "field.png", "source": str(source), "labels": labels}
    with pytest.raises(ValueError, match=r"field\.png: image 50 x 40, "
                                         r"mask 40 x 50"):
        model_share.write_contribution("community_x", [item],
                                       tmp_path / "out", consent={})
    assert not (tmp_path / "out").exists()
    unknown = tmp_path / "field.xyz"
    unknown.write_bytes(b"not an image header")
    folder = model_share.write_contribution(
        "community_x", [dict(item, source=str(unknown), name="field.xyz")],
        tmp_path / "out", consent={})
    assert (folder / "masks" / "field.tif").is_file()


def test_the_conscience_and_the_licence_are_shown(qtbot, prefs):
    dialog = msd.ContributeMasksDialog(threaded=False)
    qtbot.addWidget(dialog)
    assert dialog.conscience.text() == model_share.MASK_CONSCIENCE
    assert "we'll train one for you" in msd.contribute_masks_tooltip()
    assert model_share.COMMUNITY_LICENCE in msd.contribute_masks_tooltip()


def test_a_refused_licence_sends_nothing(qtbot, prefs, tmp_path):
    images, masks = _pair_folders(tmp_path)
    upload = _Upload()
    dialog = msd.ContributeMasksDialog(images_dir=str(images),
                                       masks_dir=str(masks), name="hela",
                                       upload=upload, threaded=False)
    qtbot.addWidget(dialog)
    dialog.ask_consent = lambda: False
    assert dialog.upload() is False
    assert upload.calls == []


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path):
    from spacr.qt.screens.make_masks import MakeMasksScreen

    folder = tmp_path / "field"
    (folder / "masks").mkdir(parents=True)
    rng = np.random.default_rng(3)
    for i in range(3):
        imageio.imwrite(folder / f"f{i}.tif",
                        rng.integers(0, 65535, (48, 48), dtype=np.uint16))
    saved = np.zeros((48, 48), np.uint16)
    saved[4:12, 4:12] = 1
    imageio.imwrite(folder / "masks" / "f2.tif", saved)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget._open_folder(str(folder))
    return widget


def test_the_button_is_in_make_masks_with_the_maintainers_tooltip(screen):
    button = screen._btn_contribute
    assert button.text() == "Contribute images and masks…"
    assert button.isEnabled()
    from html import unescape

    tip = unescape(button.toolTip())
    assert "If you can't train a model yourself" in tip
    assert "we'll train one for you" in tip
    assert "Hugging Face" in tip and model_share.COMMUNITY_LICENCE in tip


def test_curation_one_click_sends_the_image_on_screen_and_its_mask(
        screen, qtbot, prefs):
    import tifffile

    mask = np.zeros_like(screen._canvas.mask)
    mask[10:20, 10:20] = 1
    mask[30:40, 5:15] = 2
    screen._canvas.mask = mask
    upload = _Upload()
    dialog = screen.contribute_images_and_masks(show=False, upload=upload,
                                                threaded=False)
    assert dialog.source() == msd.CURRENT_SOURCE
    assert "f0.tif" in dialog.source_buttons[msd.CURRENT_SOURCE].text()
    dialog.name_edit.setText("toxo vacuoles")
    assert dialog.upload_button.isEnabled()
    _agree(dialog)
    assert dialog.upload() is True
    assert upload.calls[0][1] == "community_toxo_vacuoles"
    folder = dialog.contribution_folder
    assert [p.name for p in (folder / "images").iterdir()] == ["f0.tif"]
    sent = tifffile.imread(str(folder / "masks" / "f0.tif"))
    assert np.array_equal(sent, mask)
    assert (folder / "images" / "f0.tif").read_bytes() == \
        Path(screen._folder, "f0.tif").read_bytes()


def test_the_whole_curated_queue_is_one_choice_away(screen, qtbot, prefs):
    mask = np.zeros_like(screen._canvas.mask)
    mask[10:20, 10:20] = 1
    screen._canvas.mask = mask
    upload = _Upload()
    dialog = screen.contribute_images_and_masks(show=False, upload=upload,
                                                threaded=False)
    dialog.use(msd.CURATED_SOURCE)
    assert "(2)" in dialog.source_buttons[msd.CURATED_SOURCE].text()
    dialog.name_edit.setText("toxo vacuoles")
    _agree(dialog)
    assert dialog.upload() is True
    folder = dialog.contribution_folder
    assert sorted(p.name for p in (folder / "images").iterdir()) == \
        ["f0.tif", "f2.tif"]


def test_no_mask_on_screen_no_upload(screen, qtbot, prefs):
    screen._canvas.mask = np.zeros_like(screen._canvas.mask)
    upload = _Upload()
    dialog = screen.contribute_images_and_masks(show=False, upload=upload,
                                                threaded=False)
    dialog.name_edit.setText("toxo vacuoles")
    assert not dialog.upload_button.isEnabled()
    assert "No mask, no upload" in dialog.check.text()
    _agree(dialog)
    assert dialog.upload() is False
    assert upload.calls == []
