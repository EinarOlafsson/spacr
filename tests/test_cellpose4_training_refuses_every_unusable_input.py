"""Item 288: Cellpose 4 fine-tuning refuses each unusable input by name.

``train_cellpose`` fine-tunes Cellpose-SAM on paired images and label
masks. Every refusal below happens BEFORE a model is built -- minutes of
loading and a GPU claimed for a run that was never going to start -- and
each names what is wrong, so the user can fix the one thing rather than
guess. They are grouped as the code checks them: the settings, the folder
pairing, then each image and its mask.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from spacr import submodules as sub
from tests.test_cellpose4_training_inputs import paired


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(sub.cp_models, "CellposeModel", lambda **kw: pytest.fail(
        "a model was built before the inputs were checked"))
    monkeypatch.setattr("spacr.utils.save_settings", lambda *a, **k: None)


@pytest.mark.parametrize("change, match", [
    ({"src": "  "}, "Choose the training image source folder"),
    ({"from_scratch": True}, "fine-tunes pretrained weights"),
    ({"n_epochs": 0}, "n_epochs must be a positive integer"),
    ({"batch_size": True}, "batch_size must be a positive integer"),
    ({"nimg_per_epoch": 0}, "nimg_per_epoch must be a positive integer or"),
    ({"learning_rate": 0.0}, "learning_rate must be positive"),
    ({"weight_decay": float("nan")}, "weight_decay nonnegative"),
    ({"min_train_masks": -1}, "min_train_masks must be a nonnegative"),
    ({"scale_range": 3.0}, "scale_range must be between 0 and 2"),
    ({"percentiles": [99, 1]}, "percentiles must contain two increasing"),
])
def test_an_unusable_setting_is_refused_by_name(tmp_path, change, match):
    paired(tmp_path / "images")
    settings = dict(src=str(tmp_path / "images"))
    settings.update(change)
    with pytest.raises(ValueError, match=match):
        sub.train_cellpose(settings)


def test_a_validation_mask_folder_needs_validation_images(tmp_path):
    paired(tmp_path / "images")
    with pytest.raises(ValueError, match="also requires a validation image"):
        sub.train_cellpose(dict(src=str(tmp_path / "images"),
                                test_mask_src=str(tmp_path / "images")))


def test_a_model_name_that_is_a_path_is_refused(tmp_path):
    paired(tmp_path / "images")
    with pytest.raises(ValueError, match="model_name must be a filename"):
        sub.train_cellpose(dict(src=str(tmp_path / "images"),
                                model_name="../elsewhere/model"))


def test_folders_that_are_not_there_are_named(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        sub._cellpose_training_pairs(tmp_path / "nowhere")
    assert "Choose an image folder and a mask folder" in str(excinfo.value)
    assert "nowhere" in str(excinfo.value)


def test_a_folder_with_no_images_has_no_pairs(tmp_path):
    images = tmp_path / "images"
    (images / "masks").mkdir(parents=True)
    (images / "masks" / "README.txt").write_text("labels go here")
    with pytest.raises(ValueError, match="No paired training images"):
        sub._cellpose_training_pairs(images)


def _pair(tmp_path, image, label, name="field"):
    folder = tmp_path / "pairs"
    folder.mkdir(exist_ok=True)
    image_path = folder / f"{name}.tif"
    label_path = folder / f"{name}_masks.tif"
    tifffile.imwrite(image_path, image, photometric="minisblack")
    tifffile.imwrite(label_path, label)
    return [(str(image_path), str(label_path))]


LABEL = np.zeros((6, 7), np.uint16)
LABEL[1:3, 1:3] = 1


@pytest.mark.parametrize("channels", [[0, 0], [], [0, 1, 2, 3], [-1], ["0"]])
def test_channels_must_be_one_to_three_distinct_indices(tmp_path, channels):
    pairs = _pair(tmp_path, np.zeros((6, 7), np.uint16), LABEL)
    with pytest.raises(ValueError, match="one to three distinct"):
        sub._cellpose_training_arrays(pairs, {"channels": channels})


@pytest.mark.parametrize("image, label, settings, match", [
    (np.zeros((6, 7), np.uint16), np.zeros((6, 7), np.float32), {},
     "2-D nonnegative integer object labels"),
    (np.zeros((6, 8), np.uint16), LABEL, {}, "image and mask dimensions"),
    (np.zeros((6, 7), np.uint16), LABEL, {"channels": [1]},
     "a grayscale image only has channel 0"),
    (np.zeros((7, 7, 7), np.uint16), np.zeros((7, 7), np.uint16), {},
     "set channel_axis explicitly"),
    (np.zeros((6, 7, 2), np.uint16), LABEL, {"channel_axis": 5},
     "channel_axis must be"),
    (np.zeros((6, 7, 2), np.uint16), LABEL, {"channel_axis": 0},
     "dimensions differ for channel_axis=0"),
    (np.zeros((6, 7, 2), np.uint16), LABEL, {"channels": [2]},
     "a selected channel is outside the image"),
    (np.zeros((2, 2, 6, 7), np.uint16), LABEL, {}, "not a Z stack"),
    (np.full((6, 7), np.nan, np.float32), LABEL, {},
     "the image contains nonfinite values"),
])
def test_an_image_and_mask_that_cannot_train_are_refused(
        tmp_path, image, label, settings, match):
    pairs = _pair(tmp_path, image, label)
    with pytest.raises(ValueError, match=match):
        sub._cellpose_training_arrays(pairs, settings)
