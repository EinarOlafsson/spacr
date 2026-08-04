"""Every object type you can select has a channel you can set.

`OBJECT_TYPES` has offered `pathogen` and `organelle` all along, and both
have their own settings panel in the Live Settings dialog. Neither had a
channel control. `_build_request` built its channel map from the cell and
nucleus spin boxes only, so `channels.get(obj, 0)` fell back to 0 and
selecting "pathogen" quietly segmented the cell channel -- a preview that
looked like it worked and was answering a different question.
"""

from __future__ import annotations

import numpy as np
import pytest

from PySide6.QtWidgets import QFormLayout, QLabel


@pytest.fixture()
def lp():
    return pytest.importorskip("spacr.qt.widgets.live_preview")


@pytest.mark.parametrize("obj,attr,channel", [
    ("cell", "_cell_channel", 0),
    ("nucleus", "_nucleus_channel", 1),
    ("pathogen", "_pathogen_channel", 2),
    ("organelle", "_organelle_channel", 3),
])
def test_the_request_carries_the_channel_the_user_set(lp, qapp, obj, attr,
                                                      channel):
    panel = lp.LivePreviewPanel()
    try:
        panel._object_box.setCurrentText(obj)
        getattr(panel, attr).setValue(channel)
        panel._image = np.zeros((8, 8, 4), dtype=np.uint16)
        request = panel._build_request()
        assert request.channels.get(obj) == channel, (
            f"{obj} segmented channel {request.channels.get(obj)}; a missing "
            f"entry falls back to 0, which is the cell channel")
    finally:
        panel.deleteLater()


def test_every_object_type_has_a_channel_entry(lp, qapp):
    """No object type may rely on the `.get(obj, 0)` fallback."""
    panel = lp.LivePreviewPanel()
    try:
        panel._image = np.zeros((8, 8, 4), dtype=np.uint16)
        channels = panel._build_request().channels
        singles = {o for o in lp.OBJECT_TYPES if "+" not in o}
        missing = sorted(singles - set(channels))
        assert not missing, f"selectable but with no channel of their own: {missing}"
    finally:
        panel.deleteLater()


def test_the_dialog_shows_a_row_for_each_channel(lp, qapp):
    panel = lp.LivePreviewPanel()
    dialog = lp.LiveSettingsDialog(panel)
    try:
        labels = []
        for form in dialog.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.LabelRole)
                if item and isinstance(item.widget(), QLabel):
                    labels.append(item.widget().text())
        for expected in ("Cell channel", "Nucleus channel",
                         "Pathogen channel", "Organelle channel"):
            assert expected in labels, f"{expected} is not on the form"
    finally:
        dialog.deleteLater()
        panel.deleteLater()


def test_the_new_channels_are_propagated_to_the_main_panel(lp, qapp):
    """Otherwise tuning one here is lost when the dialog closes."""
    panel = lp.LivePreviewPanel()
    try:
        panel._pathogen_channel.setValue(2)
        panel._organelle_channel.setValue(3)
        out = panel.settings_for_propagation()
        assert out["pathogen_channel"] == 2
        assert out["organelle_channel"] == 3
    finally:
        panel.deleteLater()
