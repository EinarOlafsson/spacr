"""Which plane the preview shows, read two ways.

The channel dropdown's captions are translated -- ``All channels`` reads
``Alla kanaler`` on a Swedish screen -- so
:meth:`~spacr.qt.widgets.live_preview.LivePreviewPanel.display_channel`
prefers the untranslated entry kept in the item's data.

That data is written by the panel's localisation pass, and the pass is a
separate step from filling the box:
:func:`~spacr.qt.widgets.preview_controls.populate_channel_combo` -- the
shared refill every preview module uses -- adds captions and no data at all.
Between those two calls, and for any caller that only does the first, the
reader has to fall back to the caption or the panel would show every channel
while the dropdown said ``Ch 1``.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot):
    from spacr.qt.widgets import live_preview as LP

    widget = LP.LivePreviewPanel()
    qtbot.addWidget(widget)
    return widget


def test_a_localised_dropdown_is_read_from_its_entry_not_its_caption(panel):
    """The panel builds its own box and localises it, so the data is there."""
    box = panel._channel_box

    assert box.currentData() == "All channels"
    assert panel.display_channel() is None


def test_a_bare_refill_leaves_no_entry_and_the_caption_is_read_instead(panel):
    """``populate_channel_combo`` writes captions only -- no item data.

    A panel that insisted on the entry would report "all channels" for a box
    whose visible selection is a single plane.
    """
    from spacr.qt.widgets.preview_controls import populate_channel_combo

    box = panel._channel_box
    populate_channel_combo(box, 3)
    box.setCurrentIndex(2)

    assert box.currentData() is None, "the shared refill stores no entry"
    assert box.currentText() == "Ch 1"
    assert panel.display_channel() == 1


def test_a_bare_refill_with_all_channels_selected_still_means_all(panel):
    from spacr.qt.widgets.preview_controls import populate_channel_combo

    box = panel._channel_box
    populate_channel_combo(box, 3)
    box.setCurrentIndex(0)

    assert box.currentData() is None
    assert panel.display_channel() is None
