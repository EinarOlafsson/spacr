"""The terms are accepted by choosing to, not by dragging a scroll bar."""
from __future__ import annotations

from spacr.qt.widgets import setup_slides


def test_acceptance_is_not_gated_on_scrolling(qtbot):
    """Asked for 2026-08-28: the startup window must not require scrolling."""
    dialog = setup_slides.SetupSlides()
    qtbot.addWidget(dialog)
    # Never scrolled, and on some paths never even shown.
    assert dialog.terms_were_read() is True


def test_the_gate_is_open_before_the_page_is_even_shown():
    """The old gate answered False for a page that had never been laid out."""
    owner = _find_owner()
    assert owner is not None, "no class defines terms_were_read any more"
    instance = owner.__new__(owner)
    # No scroll area, no layout, nothing shown -- and still accepted.
    assert owner.terms_were_read(instance) is True


def _find_owner():
    """The class that answers the gate question."""
    import inspect

    for _name, obj in vars(setup_slides).items():
        if inspect.isclass(obj) and "terms_were_read" in vars(obj):
            return obj
    return None


def test_the_acceptance_itself_is_still_required():
    """Removing the gate must not remove the agreement."""
    from spacr.qt import terms

    assert hasattr(terms, "record_agreement")
    assert terms.TERMS_VERSION
    # The document is still there in full.
    assert len(terms.terms_text()) > 2000
