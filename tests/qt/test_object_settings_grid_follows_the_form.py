"""The object settings grid redraws its diameter row when the form's inputs change.

`_follow_pixel_size`, `_follow_calculation` and `_follow_magnification` store
the new value, redraw the diameter row once (a repeat changes nothing), redraw
nothing when there is no diameter row, and still report a redraw when the
model has no columns to signal over.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.object_settings_grid import ObjectSettingsGrid  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def grid(qtbot, qt_theme_applied):
    """A grid over two objects with blank diameters and no pixel size."""
    widget = ObjectSettingsGrid()
    qtbot.addWidget(widget)
    widget.set_settings({'cell_channel': 0, 'cell_diameter': None,
                         'nucleus_channel': 1, 'nucleus_diameter': None,
                         'magnification': 20, 'pixel_size_um': None,
                         'calculate_diameters': False, 'object_filters': {}})
    return widget


_FOLLOWERS = (
    ("pixel size", "_follow_pixel_size", "_pixel_size", "pixel_size_um", 0.65),
    ("calculation", "_follow_calculation", "_calculating",
     "calculate_diameters", True),
    ("magnification", "_follow_magnification", "_magnification",
     "magnification", 40),
)


def test_following_the_form_redraws_the_diameter_row_once(grid, qtbot,
                                                          monkeypatch, subtests):
    """Each follower: first change redraws and stores, a repeat returns False."""
    for case, method, model_attr, key, value in _FOLLOWERS:
        with subtests.test(msg=f"{case}: redrawn once, then a repeat changes nothing"):
            follow = getattr(grid, method)
            with qtbot.waitSignal(grid._model.dataChanged, timeout=1000):
                assert follow(value) is True
            assert getattr(grid._model, model_attr) == value
            assert grid._base[key] == value
            assert follow(value) is False

    with subtests.test(msg="no diameter row: nothing is redrawn"):
        with monkeypatch.context() as patch:
            patch.setattr(grid, 'questions', lambda: ['channel'])
            assert grid._follow_pixel_size(0.3) is False
            assert grid._follow_calculation(False) is False
            assert grid._follow_magnification(10) is False
        assert (grid._model._pixel_size, grid._model._calculating,
                grid._model._magnification) == (0.3, False, 10)

    with subtests.test(msg="no columns: still reported as a redraw"):
        with monkeypatch.context() as patch:
            patch.setattr(grid._model, 'columnCount', lambda *a: 0)
            assert grid._follow_pixel_size(0.9) is True
            assert grid._follow_calculation(True) is True
            assert grid._follow_magnification(63) is True
