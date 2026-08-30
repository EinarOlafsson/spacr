"""``B15`` — the four corners of the orthogonal view nothing else reaches.

:mod:`tests.qt.test_ortho_view` drives the widget the way a user does, and
:mod:`tests.test_cov_14_ortho_view` drives the painting. What is left are the
places the view has to survive something it did not build itself:

* a caller that put a stretch in :attr:`OrthoView.slider_box` or in one of its
  rows, which ``set_stack`` must take out again rather than pile a second set
  of sliders on top of;
* a labels layer with no :class:`~spacr.layers.FieldKey`, where a click can
  say "label 17" but has no measurement-table name to publish, so the shared
  selection must be left alone rather than told a wrong one;
* a mask whose array no longer contains the label its own ``labels()`` reports
  — the guard that stops a linked selection from moving the crosshair to the
  mean of nothing, which is ``nan`` and takes the snap arithmetic with it.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider

from spacr.layers import FieldKey, LayerStack, Spacing
from spacr.qt import ortho_view as ov
from spacr.qt.linked_selection import LinkedSelection, Selection

pytestmark = pytest.mark.qt

CONFOCAL = Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65}, units="um")
PLATE_KEY_VALUES = ("plate1", "A", "1", "1")
#: The mask in :func:`_volume` sits at slice 5 of a 10-slice 2 µm stack.
OBJECT_KEY = "plate1_A_1_1_17"
OBJECT_SLICE = 5


def _field_key():
    return FieldKey(values=dict(zip(FieldKey.columns(), PLATE_KEY_VALUES)))


def _volume(shape=(10, 64, 64), *, with_mask=False, field="key"):
    """A confocal-shaped stack, optionally with label 17 on the middle slice."""
    stack = LayerStack()
    data = np.zeros(shape, np.uint16)
    data[shape[0] // 2, 30:34, 30:34] = 4000
    stack.add_image(data, name="volume", spacing=CONFOCAL,
                    contrast_limits=(0.0, 4000.0))
    if with_mask:
        mask = np.zeros(shape, np.int32)
        mask[shape[0] // 2, 30:34, 30:34] = 17
        stack.add_labels(mask, name="mask", spacing=CONFOCAL,
                         field=_field_key() if field == "key" else None)
    return stack


def _view(qtbot, stack=None, **kwargs):
    view = ov.OrthoView(stack if stack is not None else _volume(), **kwargs)
    qtbot.addWidget(view)
    view.resize(520, 520)
    return view


def _object_pixel(view):
    """The (row, column) of label 17 in the top panel, with z on its slice."""
    view.move_to(z=OBJECT_SLICE * 2.0)
    return view.views.xy.pixel_at({"y": 31 * 0.65, "x": 31 * 0.65})


# ---------------------------------------------------------------------------
# Rebuilding the slider box
# ---------------------------------------------------------------------------

def test_a_stretch_left_in_the_slider_box_does_not_survive_a_new_volume(
        qtbot, qt_theme_applied):
    """A second volume must replace the sliders, not queue up behind them.

    ``slider_box`` is a public layout, and a screen embedding this view can
    reasonably push its rows to the top with a stretch. ``_build_sliders``
    drains that box every time ``set_stack`` is called; an item that is
    neither a widget nor a nested row — a spacer is exactly that — falls
    through both arms of the drain. If the drain stopped there the box would
    still be emptied, but if it ever stopped taking items the loop would spin
    forever, and the user would get the sliders of the old stack sitting
    above the sliders of the new one, both live, both moving a crosshair in
    a volume only one of them describes.
    """
    view = _view(qtbot, width=128)
    box = view.slider_box
    assert box.count() == 3, "one row per axis to start with"

    box.addStretch(1)
    assert box.itemAt(3).spacerItem() is not None, \
        "the stretch this test relies on was not added as a spacer item"

    view.set_stack(_volume(shape=(6, 32, 32)))

    assert box.count() == 3, "the stretch was left behind, or a row was"
    assert [box.itemAt(i).spacerItem() for i in range(box.count())] == \
        [None, None, None]
    assert sorted(view._sliders) == ["x", "y", "z"]
    assert view.views.n_slices("z") == 6, \
        "the rebuilt sliders describe the old volume"
    assert len({id(slider) for slider in view._sliders.values()}) == 3, \
        "two axes ended up sharing one slider"


def test_a_stretch_inside_a_slider_row_still_leaves_the_row_empty(
        qtbot, qt_theme_applied):
    """Every widget of a discarded row has to go, or the view leaks a slider.

    A dropped row whose caption and slider are still alive is not merely
    memory: a live ``QSlider`` still parented to the view is still connected
    to ``_on_slider_moved``, so a stray keyboard focus or a programmatic
    ``setValue`` would move the crosshair of a volume that is no longer
    shown. ``_drain`` walks the row taking items, and a spacer among them
    must be stepped over rather than treated as a widget — ``deleteLater``
    on ``None`` is an ``AttributeError`` inside a rebuild that has already
    half-emptied the layout.
    """
    view = _view(qtbot, width=128)
    row = view.slider_box.itemAt(0).layout()
    caption = row.itemAt(0).widget()
    slider = row.itemAt(1).widget()
    row.addStretch(1)
    assert row.count() == 4 and isinstance(slider, QSlider)

    view.set_stack(_volume(shape=(4, 32, 32)))

    assert row.count() == 0, "the drained row still holds items"
    qt_theme_applied.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    from shiboken6 import Shiboken
    assert not Shiboken.isValid(caption), "the old caption is still alive"
    assert not Shiboken.isValid(slider), "the old slider is still alive"
    assert view.views.n_slices("z") == 4
    assert view._sliders["z"].value() == pytest.approx(500, abs=250), \
        "the new z slider did not start near the middle of the new volume"


def test_draining_a_hand_built_row_takes_the_widgets_and_the_spacer(
        qtbot, qt_theme_applied):
    """``_drain`` is the one place rows are emptied, so it must empty any row.

    It is a static helper and the next slider row somebody adds — a checkbox
    beside the caption, a stretch to keep the readout right-aligned — goes
    through it. Asserted on a row built here rather than on one the view
    built, so the helper's contract is pinned independently of what
    ``_add_slider`` happens to put in a row today.
    """
    row = QHBoxLayout()
    caption = QLabel("z")
    slider = QSlider(Qt.Horizontal)
    row.addWidget(caption)
    row.addStretch(1)
    row.addWidget(slider)
    assert row.count() == 3

    ov.OrthoView._drain(row)

    assert row.count() == 0
    qt_theme_applied.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    from shiboken6 import Shiboken
    assert not Shiboken.isValid(caption)
    assert not Shiboken.isValid(slider)


# ---------------------------------------------------------------------------
# Clicking a mask that cannot name its objects
# ---------------------------------------------------------------------------

def test_a_mask_with_no_field_key_publishes_nothing_until_it_has_one(
        qtbot, qt_theme_applied):
    """A click on an unnamed object must not select the wrong cell elsewhere.

    Without a :class:`~spacr.layers.FieldKey` the layer can say "label 17"
    and nothing more — 17 is a number local to one field, and there is a
    label 17 in every field on the plate. Publishing it would highlight an
    arbitrary object in the UMAP and the annotation grid, which is worse
    than highlighting none. The same click on the same pixel, once the layer
    knows which field it segments, is the whole point of the mechanism, so
    both halves are driven here.
    """
    stack = _volume(with_mask=True, field=None)
    view = _view(qtbot, stack, width=128)
    link = LinkedSelection()
    view.link_selection(ov.ORTHO_LINK_SOURCE, link=link)
    picked = []
    view.object_picked.connect(picked.append)
    row, column = _object_pixel(view)

    view._on_panel_clicked("xy", row, column)

    assert picked == [], "an unnamed object reached the shared selection"
    assert link.selection.keys is None
    assert stack["mask"].selected_label == 0, \
        "the layer was marked selected for a key it could not name"
    assert view.slice_index("z") == OBJECT_SLICE, \
        "the click did not even move the crosshair"

    stack["mask"].field = _field_key()
    view._on_panel_clicked("xy", row, column)

    assert picked == [OBJECT_KEY]
    assert list(link.selection.keys) == [OBJECT_KEY]
    assert stack["mask"].selected_label == 17


# ---------------------------------------------------------------------------
# A selection naming a label the array no longer holds
# ---------------------------------------------------------------------------

def test_a_selection_for_a_label_the_array_does_not_hold_moves_nothing(
        qtbot, qt_theme_applied):
    """The crosshair must not be sent to the mean of an empty index array.

    ``on_linked_selection_changed`` finds the label whose object key another
    view selected and moves the crosshair to that object's centre of mass.
    The centre comes from ``np.argwhere(data == label)``, and when that is
    empty ``mean(axis=0)`` is ``nan`` — which reaches ``move_to`` and dies in
    ``int(round(nan))`` inside ``_snap``, i.e. a ``ValueError`` raised out of
    a signal handler on the selection bus, taking down whichever view
    published the selection.

    A mask can hold labels its own array does not: ``LabelsLayer.labels()``
    returns the raw unique values, while this loop compares ``data ==
    int(label)``. Replace a mask's data with a float array — the setter
    allows it, unlike the constructor — and 17.5 is a label whose key is
    ``…_17`` and whose pixels are nowhere. The first half of this test shows
    the same selection moving the crosshair when the array is intact, so the
    absence asserted in the second half is a real difference and not a
    selection that never arrived.
    """
    stack = _volume(with_mask=True)
    view = _view(qtbot, stack, width=128)
    mask = stack["mask"]
    view.move_to(z=0.0)

    view.on_linked_selection_changed(Selection(keys=[OBJECT_KEY],
                                               source="umap"))

    assert view.slice_index("z") == OBJECT_SLICE, \
        "an intact mask did not draw the crosshair onto its object"
    assert mask.selected_label == 17

    mask.selected_label = 0
    view.move_to(z=0.0)
    corrupt = np.asarray(mask.data, dtype=np.float64)
    corrupt[corrupt > 0] += 0.5
    mask.data = corrupt
    assert mask.labels().tolist() == [17.5], \
        "the mask no longer reports the label this test needs"

    view.on_linked_selection_changed(Selection(keys=[OBJECT_KEY],
                                               source="umap"))

    assert mask.selected_label == 17, \
        "the loop never matched the key, so the guard was never reached"
    assert view.slice_index("z") == 0, \
        "the crosshair moved to the centre of an object with no pixels"
    assert not math.isnan(view.views.point["z"])
    assert view.views.point["z"] == pytest.approx(0.0), \
        "the crosshair left the slice it was parked on"
