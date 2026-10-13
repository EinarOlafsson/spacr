"""Small Make Masks recovery paths without model downloads or GPU work."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, QRect, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QDialog  # noqa: E402

from spacr import drop_classification, spacr_cellpose  # noqa: E402
from spacr.qt import preferences  # noqa: E402
from spacr.qt.screens import make_masks as mm  # noqa: E402

pytestmark = pytest.mark.qt


def test_an_unblinded_status_line_is_shown_and_reported(qtbot):
    label = mm._StatusLabel()
    qtbot.addWidget(label)
    said = []
    label.said.connect(said.append)

    label.setText("Ready\nsecond line")

    assert label.text() == "Ready\nsecond line"
    assert said == ["Ready\nsecond line"]


def test_cellpose_flows_with_the_wrong_vector_shape_are_not_used(monkeypatch):
    labels = np.ones((2, 2), dtype=np.int32)
    monkeypatch.setattr(spacr_cellpose, "parse_cellpose4_output",
                        lambda _output: ([labels], [None],
                                         [np.zeros((1, 2, 2))], [None], [None]))
    monkeypatch.setattr(mm, "cellpose_intermediates",
                        lambda _flows: (np.zeros((2, 2)),
                                        np.zeros((2, 2, 3), dtype=np.uint8)))
    model = SimpleNamespace(eval=lambda fields, **_kwargs: fields)

    found, _probability, _rgb, vectors = mm._cellpose_detect_with_vectors(
        np.zeros((2, 2), dtype=np.uint16), model)

    np.testing.assert_array_equal(found, labels)
    assert vectors is None


def test_a_small_prompt_move_keeps_the_press_without_starting_a_box():
    prompt = SimpleNamespace(_press=(Qt.LeftButton, QPointF(0, 0), (0, 0)),
                             _drag=None)

    assert mm._PromptSession.move(prompt, QPointF(1, 1)) is True
    assert prompt._drag is None


def test_a_lens_intersection_without_an_image_pixel_is_empty():
    canvas = SimpleNamespace(rect=lambda: QRect(0, 0, 100, 100))
    magnifier = SimpleNamespace(canvas=canvas)

    part, painted = mm._LiveMagnifier._visible_part(
        magnifier, (0, 0, 1, 1), QRectF(-1, 0, 2, 2), 0.1)

    assert part is None and painted is None


def _image_result():
    return SimpleNamespace(
        labels=np.ones((4, 4), dtype=np.int32),
        overlay=np.zeros((4, 4, 4), dtype=np.uint8),
        request=SimpleNamespace(colour=(255, 0, 0)),
    )


def test_a_cached_magnifier_slice_is_reused():
    result = _image_result()
    picture = QImage(2, 2, QImage.Format_RGBA8888)
    owner = SimpleNamespace(
        _image_result=result, _cursor=(0, 0), overlap="replace",
        _mask_token=7,
        _image_view=(result, ((0, 0, 2, 2), 1, "replace", 0), picture),
    )

    assert mm._LiveMagnifier._image_slice(owner, (0, 0, 2, 2)) is picture


def test_an_object_outside_the_slice_does_not_change_its_picture(monkeypatch):
    result = _image_result()
    owner = SimpleNamespace(
        _image_result=result, _cursor=(0, 0), overlap="replace",
        _mask_token=0, _image_view=None,
    )
    monkeypatch.setattr(mm, "_object_window", lambda *_args: (3, 3, 4, 4))

    picture = mm._LiveMagnifier._image_slice(owner, (0, 0, 2, 2))

    assert (picture.width(), picture.height()) == (2, 2)


def test_a_visible_organize_dialog_can_be_accepted(monkeypatch):
    monkeypatch.setattr(mm, "is_headless", lambda: False)
    dialog = SimpleNamespace(exec=lambda: QDialog.Accepted)
    assert mm.MakeMasksScreen._run_organize(None, dialog) is True


def test_a_channel_folder_drop_hands_off_to_organize(monkeypatch):
    found = SimpleNamespace(
        kind="nested", folders=["/data"],
        channel_like=True, channel_folders=["/data/DAPI", "/data/GFP"],
        unrecognised=[], description="",
    )
    monkeypatch.setattr(drop_classification, "classify_drop",
                        lambda _paths: found)
    offered = []
    owner = SimpleNamespace(
        _masks_console=SimpleNamespace(post=lambda *_args: None),
        _open_organize=lambda folder, folders: offered.append(
            (folder, folders)) or True,
    )

    assert mm.MakeMasksScreen.open_paths(owner, ["/data"]) is True
    assert offered == [("/data", ["/data/DAPI", "/data/GFP"])]


def test_masks_already_beside_the_images_open_without_a_second_mask_dir():
    image = "/project/one.png"
    found = SimpleNamespace(images=[image], masks={
        image: "/project/masks/one.tif"}, unpaired_masks=[])
    opened = []
    owner = SimpleNamespace(
        _masks_console=SimpleNamespace(post=lambda *_args: None),
        _open_folder=lambda *args, **kwargs: opened.append(
            (args, kwargs)) or True,
    )

    assert mm.MakeMasksScreen._open_with_masks(owner, found) is True
    assert opened == [(('/project',), {'files': ['one.png'],
                                      'masks_dir': None})]


def test_one_uncopyable_dropped_mask_does_not_block_the_image_queue(
        monkeypatch):
    first, second = "/a/one.png", "/b/two.png"
    found = SimpleNamespace(images=[first, second], masks={
        first: "/drop/one.tif", second: "/drop/two.tif"},
        unpaired_masks=[])
    posted, queued = [], []
    owner = SimpleNamespace(
        _masks_console=SimpleNamespace(post=lambda text, *_args:
                                       posted.append(text)),
        _confirm=lambda *_args: True,
        _open_queue=lambda images: queued.extend(images) or True,
    )

    def cannot_copy(*_args):
        raise OSError("read-only destination")

    monkeypatch.setattr(mm, "_copy_mask_as_tiff", cannot_copy)
    assert mm.MakeMasksScreen._open_with_masks(owner, found) is True
    assert queued == [first, second]
    assert any("read-only destination" in line for line in posted)


def test_alpha_visibility_tolerates_a_missing_optional_ensemble(monkeypatch):
    class EmptyCombo:
        def count(self):
            return 0

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: False)
    monkeypatch.setattr(preferences, "_apply_alpha_widgets", lambda _owner: None)
    refreshed = []
    owner = SimpleNamespace(
        _mag_mode=EmptyCombo(), _cp_model=EmptyCombo(),
        _uncertainty_ensemble=None,
        _fold_screens={"legacy": SimpleNamespace()},
        _resync_magnifier_modes=lambda: refreshed.append(True),
    )

    mm.MakeMasksScreen._refresh_alpha_visibility(owner)
    assert refreshed == [True]
