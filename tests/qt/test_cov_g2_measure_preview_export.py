"""Displayed-crop export from the Measure preview: refusals and cancellation."""
from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import measure_preview as mp  # noqa: E402


def _crop(**extra):
    entry = {"crop": np.zeros((4, 4, 3), np.uint8), "source_path": "/a.npy",
             "object_key": ("plate1_r1_c1_f1", "cell", 1), "label": 1,
             "area": 9, "bbox": [0, 0, 4, 4]}
    entry.update(extra)
    return entry


def test_crop_pixels_must_be_nonempty_rgb():
    with pytest.raises(ValueError):
        mp._preview_crop_pixels(np.zeros((4, 4), np.uint8), "rgb")


def test_an_empty_preview_is_not_exported(tmp_path):
    result = mp._write_preview_crop_export(tmp_path / "out", [], {}, "rgb",
                                           threading.Event())
    assert "crop preview" in result["error"]


def test_a_cancelled_export_writes_nothing(tmp_path):
    cancelled = threading.Event()
    cancelled.set()
    result = mp._write_preview_crop_export(tmp_path / "out", [_crop()], {},
                                           "rgb", cancelled)
    assert result == {"cancelled": True}
    assert list(tmp_path.iterdir()) == []


def test_a_cancel_during_writing_removes_the_stage(tmp_path, monkeypatch):
    cancelled = threading.Event()
    real = mp._preview_crop_pixels

    def cancel_after_first(crop, primaries):
        cancelled.set()
        return real(crop, primaries)

    monkeypatch.setattr(mp, "_preview_crop_pixels", cancel_after_first)
    result = mp._write_preview_crop_export(tmp_path / "out", [_crop()], {},
                                           "rgb", cancelled)
    assert result == {"cancelled": True}
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".spacr-")]


def test_a_cancel_between_crops_removes_the_stage(tmp_path, monkeypatch):
    cancelled = threading.Event()
    calls = []
    real = mp._preview_crop_pixels

    def cancel_on_second(crop, primaries):
        calls.append(1)
        if len(calls) == 1:
            cancelled.set()
        return real(crop, primaries)

    monkeypatch.setattr(mp, "_preview_crop_pixels", cancel_on_second)
    result = mp._write_preview_crop_export(tmp_path / "out", [_crop(), _crop()],
                                           {}, "rgb", cancelled)
    assert result == {"cancelled": True} and len(calls) == 1
