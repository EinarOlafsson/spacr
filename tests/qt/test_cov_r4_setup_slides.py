"""The two seams of the terms slide that only open when the gate is shut.

The reading gate has one wire in and one sentence out, and neither is
exercised by the ordinary pass through the slide. The wire is the scroll
bar: the gate is connected to one where the viewport has one, and the page
still has to be a whole page on a viewport that does not. The sentence is
the refusal, which says one thing about an unticked switch and two about a
switch that cannot be ticked yet.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QScrollArea      # noqa: E402

from spacr.qt.widgets.setup_slides import SetupSlides, _say  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def slides(app):
    """A setup screen. ``_isolated_qsettings_store`` in the root conftest is
    autouse, so the preferences it reads are this test's own."""
    return SetupSlides()


def test_the_terms_page_is_whole_with_or_without_a_scroll_bar(slides,
                                                              monkeypatch):
    """The gate is opened by the scroll bar's own signals, so a viewport
    with no bar to connect to has to leave a page that is still a page --
    the licence, the switch and the reason it is greyed -- rather than a
    half-built one that raises on the way past the connection."""
    page = slides._terms_page()
    assert page is not None
    bar = slides._terms_scroll.verticalScrollBar()
    assert bar is not None, "an ordinary viewport has a bar to listen to"
    assert slides._agree.isEnabled() is False, "the gate is drawn shut"
    assert slides._scroll_hint.isHidden() is False, "and says why"
    # THE BAR IS THE WIRE. Growing its range is the viewport reporting how
    # much document there is, which is one of the two ways the gate is
    # re-read; if the connection at 1024 were not made, nothing would move.
    bar.setRange(0, 50)
    assert slides._agree.isEnabled() is True, "the bar opened the gate"
    assert slides._scroll_hint.isHidden() is True, "and took the reason away"

    class _NoBar(QScrollArea):
        """A viewport that reports no vertical scroll bar at all."""

        def verticalScrollBar(self):        # noqa: N802 - Qt naming
            return None

    # PATCHED AT THE IMPORT SITE: `_terms_page` imports QScrollArea when it
    # runs, so replacing the name in QtWidgets is what the page picks up.
    monkeypatch.setattr("PySide6.QtWidgets.QScrollArea", _NoBar)
    bare = slides._terms_page()
    assert bare is not None
    assert slides._terms_scroll.verticalScrollBar() is None
    from spacr.qt import terms as terms_module
    assert terms_module.terms_text()[:60] in slides._terms_body.text(), \
        "the licence is still on the page"
    assert slides._agree.text() == _say(terms_module.AGREE_LABEL)
    assert slides._agree.isEnabled() is False, "and the gate is still shut"
    assert slides._terms_read is False, "with no bar to open it"


def test_the_refusal_names_the_gate_as_well_as_the_switch(slides,
                                                          monkeypatch):
    """Two things can be missing on this slide and they need two different
    sentences. "Tick the box above" is not actionable advice about a box
    that will not take a tick, so a gate that is still shut is named first
    and the switch second."""
    from spacr.qt import terms as terms_module

    slides._agree_note.setVisible(False)
    assert slides._refuse_to_leave_the_terms() == slides.slide(), "stays put"
    only_the_switch = slides._agree_note.text()
    assert only_the_switch == _say(terms_module.WHY_NOT_YET)
    assert slides._agree_note.isHidden() is False, "and it is said out loud"

    # `terms_were_read` is hard-coded True since the scroll gate was
    # removed, so the second sentence is only reachable through a gate that
    # answers False -- which is the state the method is written for and the
    # one a re-introduced gate would put it in.
    monkeypatch.setattr(slides, "terms_were_read", lambda: False)
    assert slides._refuse_to_leave_the_terms() == slides.slide()
    both = slides._agree_note.text()
    assert both != only_the_switch, "a shut gate is a second thing to say"
    assert both.startswith(_say(terms_module.SCROLL_HINT))
    assert both.endswith(only_the_switch)
