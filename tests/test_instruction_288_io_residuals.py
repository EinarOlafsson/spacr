"""Small executable decisions that close instruction 288 for ``spacr.io``."""

from __future__ import annotations

import io as byte_io
import tarfile

import numpy as np
from PIL import Image

from spacr import io


def test_load_images_and_labels_accepts_two_empty_lists(capsys):
    loaded = io._load_images_and_labels([], [])

    assert loaded == ([], [], [], [])
    assert "Loaded 0 images and 0 labels from None and None" in \
        capsys.readouterr().out


def test_no_class_dataset_applies_a_supplied_transform(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (3, 2), (11, 22, 33)).save(image_path)
    seen = []

    def transform(image):
        seen.append((image.mode, image.size))
        return "transformed"

    dataset = io.NoClassDataset(
        str(tmp_path), transform=transform, shuffle=False)
    image, path = dataset[0]

    assert image == "transformed"
    assert path == str(image_path)
    assert seen == [("RGB", (3, 2))]


def test_tar_image_dataset_applies_a_supplied_transform(tmp_path):
    payload = byte_io.BytesIO()
    Image.new("RGB", (4, 3), (7, 8, 9)).save(payload, format="PNG")
    tar_path = tmp_path / "images.tar"
    with tarfile.open(tar_path, "w") as archive:
        member = tarfile.TarInfo("sample.png")
        member.size = len(payload.getvalue())
        archive.addfile(member, byte_io.BytesIO(payload.getvalue()))

    dataset = io.TarImageDataset(
        str(tar_path), transform=lambda image: (image.mode, image.size))

    assert dataset[0] == (("RGB", (4, 3)), "sample.png")


def test_invalid_mask_before_a_valid_mask_reports_and_keeps_iterating(capsys):
    invalid = np.ones((2, 2, 2, 2), dtype=np.int32)
    valid = np.zeros((5, 5), dtype=np.int32)
    valid[1:4, 1:4] = 1

    count, area = io._get_avg_object_size([invalid, valid])

    assert count == 0.5
    assert area == 9
    assert "Mask 0 has invalid dimension: 4" in capsys.readouterr().out


def test_merge_with_an_empty_reference_stack_is_a_clean_no_op(tmp_path):
    root = tmp_path / "plate"
    (root / "stack").mkdir(parents=True)

    result = io._load_and_concatenate_arrays(
        str(root), channels=[0, 1], cell_chann_dim=None,
        nucleus_chann_dim=None, pathogen_chann_dim=None,
        organelle_chann_dim=None)

    assert result is None
    assert (root / "merged").is_dir()
    assert list((root / "merged").iterdir()) == []


def test_cv_group_ids_verbose_report_names_the_grouping(capsys):
    names = [
        "plate1_A01_1_1_cell_1.png",
        "plate1_A01_2_1_cell_2.png",
        "plate1_B01_1_1_cell_3.png",
    ]

    groups, offset = io._cv_group_ids(names, "well", verbose=True)

    assert groups == ["plate1_A01", "plate1_A01", "plate1_B01"]
    assert offset == 0
    assert "Grouping folds by well: 2 distinct well(s) across 3 crops" in \
        capsys.readouterr().out


def test_cv_group_ids_can_suppress_the_report(capsys):
    groups, offset = io._cv_group_ids(
        ["plate1_A01_1_1_cell_1.png"], "well", verbose=False)

    assert groups == ["plate1_A01"]
    assert offset == 0
    assert capsys.readouterr().out == ""


def test_yokogawa_conversion_skips_an_unsupported_sidecar(tmp_path):
    Image.new("I;16", (4, 4), 17).save(tmp_path / "image.tif")
    (tmp_path / "notes.txt").write_text("keep me", encoding="utf-8")

    ledger = io.convert_to_yokogawa(str(tmp_path))

    assert ledger.is_complete
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "keep me"
    assert len(list(tmp_path.glob("plate*.tif"))) == 1
