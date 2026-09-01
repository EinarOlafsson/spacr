"""The loaded path must not decide the layout of the row it sits in.

Reported 2026-09-01: "in mask live view there is a field in the top left that
overlaps with the image path".

A QHBoxLayout does not overlap on its own. What it does is honour sizeHints,
and the path label's hint was the FULL path -- routinely longer than the panel
is wide -- so the MIP toggle, both spin boxes and the Choose button were pushed
past the right edge. On screen that reads as the top-left field overlapping the
path.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def panel(qapp):
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    widget = LivePreviewPanel()
    widget.resize(900, 600)
    yield widget
    widget.deleteLater()


def test_a_very_long_path_does_not_inflate_the_label(panel):
    """The property that keeps the row intact."""
    long_path = "/" + "/".join(f"a_rather_long_directory_name_{i}"
                               for i in range(20)) + "/field_001.tif"
    panel._path_full = long_path
    panel._show_elided_path()

    shown = panel._path_label.text()
    assert len(shown) < len(long_path), "the path was not elided"
    assert panel._path_label.sizeHint().width() <= panel.width(), (
        "the label still claims more width than the panel has, which is what "
        "pushed its row-mates off the edge")


def test_the_full_path_is_still_reachable(panel):
    """Eliding must not lose information, only stop it setting the layout."""
    long_path = "/data/plate1/" + "x" * 300 + "/field_001.tif"
    panel._path_full = long_path
    panel._show_elided_path()

    assert panel._path_label.toolTip() == long_path


def test_the_middle_is_elided_so_both_ends_survive(panel):
    """The two ends identify an image: the plate folder and the file name.
    A tail-elided path is a column of identical prefixes.

    Asserted on SHAPE rather than on a fixed prefix: how many characters
    survive depends on the label's width, and in a headless test that is
    whatever the unlaid-out widget reports. What must hold at any width is
    that the cut is in the middle -- something of the head, something of the
    tail, and the ellipsis between them.
    """
    long_path = "/PLATEFOLDER/" + "m" * 300 + "/FIELDNAME.tif"
    panel._path_full = long_path
    panel._show_elided_path()

    shown = panel._path_label.text()
    assert "\u2026" in shown, f"nothing was elided: {shown}"
    head, _, tail = shown.partition("\u2026")
    assert head and long_path.startswith(head), (
        f"the head is not the start of the path: {head!r}")
    assert tail and long_path.endswith(tail), (
        f"the tail is not the end of the path: {tail!r}")
    assert "m" * 50 not in shown, "the middle survived instead of the ends"


def test_the_label_can_be_squeezed_below_its_text(panel):
    """A minimum width of zero is what lets the layout shrink it at all.

    VACUOUS AS FIRST WRITTEN, and kept only with that said: QLabel's default
    minimumWidth is already 0, so this assertion passed with the explicit call
    removed. A mutation caught it. The eliding is what actually does the work
    -- see the first test, which does fail without it -- and the explicit call
    survives as a statement of intent rather than as behaviour.
    """
    assert panel._path_label.minimumWidth() == 0


def test_the_live_model_row_offers_the_zoo(panel, monkeypatch):
    """Reported 2026-09-01: the live settings offered cpsam and nothing else.

    A zoo model could be chosen for the RUN and not for the PREVIEW, which is
    the preview showing a different model than the run will use while the user
    tunes against it.
    """
    import spacr.qt.widgets.model_zoo_picker as picker

    monkeypatch.setattr(picker, "choose_model",
                        lambda *a, **k: "/models/cpsam_v2_toxo_r2")
    panel._choose_a_preview_model()

    assert panel._model_box.currentText() == "/models/cpsam_v2_toxo_r2", (
        "the chosen model was not selected in the combo")


def test_a_model_not_in_the_menu_is_added_rather_than_ignored(panel,
                                                              monkeypatch):
    """The menu lists what was on disk when the panel was BUILT. Selecting an
    item the combo does not hold would silently do nothing."""
    import spacr.qt.widgets.model_zoo_picker as picker

    before = panel._model_box.count()
    monkeypatch.setattr(picker, "choose_model",
                        lambda *a, **k: "/somewhere/new_checkpoint")
    panel._choose_a_preview_model()

    assert panel._model_box.count() == before + 1
    assert panel._model_box.currentText() == "/somewhere/new_checkpoint"


def test_cancelling_the_picker_leaves_the_model_alone(panel, monkeypatch):
    import spacr.qt.widgets.model_zoo_picker as picker

    chosen = panel._model_box.currentText()
    monkeypatch.setattr(picker, "choose_model", lambda *a, **k: None)
    panel._choose_a_preview_model()

    assert panel._model_box.currentText() == chosen
