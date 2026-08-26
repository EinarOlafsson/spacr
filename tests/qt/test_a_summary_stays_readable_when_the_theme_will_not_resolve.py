"""The reading surface behind a folded summary, including when it has none.

Summary text sits over the animated backdrop, and fully transparent put that
animation directly behind the type. The panel asks the theme for the
``surface_alt`` colour and for the alpha solved for legibility over it, and
paints an ``rgba(...)`` behind each section.

Neither of those can be assumed to answer. Preferences may not have been
written yet, a palette may not carry the role, and a theme name may not
resolve at all -- none of which is a reason to lose the summary. The panel
falls back to transparency, which is what it had before the surface existed,
and leaves a line in the debug log rather than a traceback on the screen.
"""
from __future__ import annotations

import logging
import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.qt.widgets.folding_summary import FoldingSummaryView   # noqa: E402


_RGBA = re.compile(r"^rgba\((\d+), (\d+), (\d+), ([01]\.\d+)\)$")

SUMMARY = """THE ANSWER
----------
  n cells                4,120
  D'Agostino K2 p        4.96e-157 (REJECTED at 0.05)
"""


def _view(qtbot):
    view = FoldingSummaryView()
    qtbot.addWidget(view)
    return view


def test_the_surface_is_an_rgba_of_the_themes_own_colour(qtbot):
    """The control: a theme that resolves gives a real, bounded colour."""
    view = _view(qtbot)

    surface = view._reading_surface()

    match = _RGBA.match(surface)
    assert match, surface
    red, green, blue, alpha = match.groups()
    assert all(0 <= int(channel) <= 255 for channel in (red, green, blue))
    assert 0.0 <= float(alpha) <= 1.0


def test_a_theme_that_cannot_be_resolved_falls_back_to_transparency(
        qtbot, monkeypatch):
    """Transparency is what the panel had before the surface existed."""
    from spacr.qt import theme

    def _no_alpha(*args, **kwargs):
        raise KeyError("surface_alt")

    monkeypatch.setattr(theme, "panel_alpha", _no_alpha)
    view = _view(qtbot)

    assert view._reading_surface() == "transparent"


def test_the_fallback_says_so_in_the_debug_log(qtbot, monkeypatch, caplog):
    """A panel that quietly stops painting its surface must leave a trace."""
    from spacr.qt import theme

    monkeypatch.setattr(theme, "palette_for",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("no palette for that theme")))
    view = _view(qtbot)

    with caplog.at_level(logging.DEBUG,
                         logger="spacr.qt.widgets.folding_summary"):
        assert view._reading_surface() == "transparent"

    assert any("reading surface" in record.message
               for record in caplog.records)


def test_a_summary_still_renders_its_rows_without_a_surface(qtbot,
                                                            monkeypatch):
    """Losing the surface may not cost the summary itself."""
    from spacr.qt import theme

    monkeypatch.setattr(theme, "panel_alpha",
                        lambda *a, **k: (_ for _ in ()).throw(
                            KeyError("surface_alt")))
    view = _view(qtbot)

    view.setPlainText(SUMMARY)

    assert "n cells" in view.toPlainText()
    assert view.section_titles() == ("THE ANSWER",)
    assert view.is_section_expanded("THE ANSWER") is True
