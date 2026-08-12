"""The crop preview must show the colours the run writes.

Instruction 77. Measured, not reasoned about: ``MeasurePreviewPanel``'s RGB
control was handed straight to ``crop_objects_from_array``, whose
``channels`` argument IS RGB order, while a real run resolves
``png_channel_mapping``, which ships as ``{'r': 2, 'g': 1, 'b': 0}``. With
the control defaulting to "0,1,2" the two were exact mirrors -- on a
standard 405/488/555 stack the preview showed the nuclear stain red and the
run wrote it blue.

The control also propagated ``png_dims``, which
``resolve_png_channel_mapping`` ignores whenever a mapping is set, and
Measure sets one by default. So the value was discarded by the run it was
tuning, which is why the disagreement could not be corrected by using the
control.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _qapp(qapp):
    return qapp


class TestTheCropPreviewAgreesWithTheRun:
    """The colours on screen must be the colours written to disk.

    Measured, not reasoned about: the panel's RGB control was handed
    straight to ``crop_objects_from_array``, whose ``channels`` argument is
    RGB order, while the run resolves ``png_channel_mapping``, which ships
    as ``{'r': 2, 'g': 1, 'b': 0}``. With the control defaulting to "0,1,2"
    the two were exact mirrors of each other -- on a standard 405/488/555
    stack the preview showed the nuclear stain red and the run wrote it
    blue.
    """

    def _array(self):
        data = np.zeros((32, 32, 5), np.float32)
        data[..., 0] = 10.0
        data[..., 1] = 100.0
        data[..., 2] = 200.0
        mask = np.zeros((32, 32), np.int32)
        mask[8:24, 8:24] = 1
        data[..., 4] = mask
        return data

    def _run_rgb(self, data, settings):
        """What the RUN writes, through the run's own two functions."""
        from spacr.crops import build_png_channels, resolve_png_channel_mapping
        from spacr.measure import crop_objects_from_array
        raw = crop_objects_from_array(
            data, mask_dim=4, channels=[0, 1, 2, 3], to_rgb=False,
            normalize=False)[0]["crop"]
        png = build_png_channels(raw, resolve_png_channel_mapping(settings))
        return [int(png[8, 8, i]) for i in range(3)]

    def _preview_channels(self, panel):
        from spacr.qt.widgets.measure_preview import _parse_channels
        return _parse_channels(panel._png_dims.text())

    def test_the_default_preview_matches_the_default_run(self, qtbot):
        from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
        from spacr.settings import get_measure_crop_settings
        from spacr.crops import resolve_png_channel_mapping

        p = MeasurePreviewPanel(threaded=False)
        qtbot.addWidget(p)
        run_mapping = resolve_png_channel_mapping(get_measure_crop_settings({}))

        assert self._preview_channels(p) == [
            run_mapping["r"], run_mapping["g"], run_mapping["b"]]

    def test_the_channel_order_drawn_is_the_channel_order_written(self, qtbot):
        """Ordering, end to end, against the real cropper."""
        from spacr.measure import crop_objects_from_array
        from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
        from spacr.settings import get_measure_crop_settings

        p = MeasurePreviewPanel(threaded=False)
        qtbot.addWidget(p)
        data = self._array()

        shown = crop_objects_from_array(
            data, mask_dim=4, channels=self._preview_channels(p),
            normalize=False)[0]["crop"]
        preview_rgb = [int(shown[8, 8, i]) for i in range(3)]
        run_rgb = self._run_rgb(data, get_measure_crop_settings({}))

        # Both are uint8 rescalings of the same three source channels, so
        # compare the ORDER -- which channel is brightest in which plane --
        # rather than the exact levels.
        assert np.argsort(preview_rgb).tolist() == np.argsort(run_rgb).tolist()

    def test_the_control_propagates_the_key_the_run_reads(self, qtbot):
        """``png_dims`` is ignored whenever a mapping is set, and one is."""
        from spacr.qt.widgets.measure_preview import MeasurePreviewPanel

        p = MeasurePreviewPanel(threaded=False)
        qtbot.addWidget(p)
        p._png_dims.setText("1,0,2")

        out = p.settings_for_propagation()
        assert out["png_channel_mapping"] == {"r": 1, "g": 0, "b": 2}
        assert "png_dims" not in out

    def test_a_legacy_png_dims_settings_file_still_seeds_the_panel(self, qtbot):
        """The legacy list reads entry 0 as BLUE, and must keep doing so.

        Every settings CSV in the wild holds one. Seeding it as RGB order
        would silently re-colour crops for exactly the users who have been
        running longest.
        """
        from spacr.qt.widgets.measure_preview import MeasurePreviewPanel

        p = MeasurePreviewPanel(threaded=False)
        qtbot.addWidget(p)
        p.apply_settings({"png_dims": [0, 1, 2]})

        assert p.settings_for_propagation()["png_channel_mapping"] == {
            "r": 2, "g": 1, "b": 0}

    def test_an_explicit_mapping_beats_the_legacy_list(self, qtbot):
        from spacr.qt.widgets.measure_preview import MeasurePreviewPanel

        p = MeasurePreviewPanel(threaded=False)
        qtbot.addWidget(p)
        p.apply_settings({"png_dims": [0, 1, 2],
                          "png_channel_mapping": {"r": 1, "g": 1, "b": 1}})

        assert p.settings_for_propagation()["png_channel_mapping"] == {
            "r": 1, "g": 1, "b": 1}
