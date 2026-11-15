"""Displayed-crop export from the Measure preview: refusals and cancellation."""
from __future__ import annotations


import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import measure_preview as mp  # noqa: E402


def test_crop_pixels_must_be_nonempty_rgb():
    with pytest.raises(ValueError):
        mp._preview_crop_pixels(np.zeros((4, 4), np.uint8), "rgb")
