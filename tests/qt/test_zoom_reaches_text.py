"""``Z1`` — the Zoom preference has to reach every text surface.

Zoom ("font scale", default 150 %) scales :data:`spacr.qt.theme.FONT_SIZE`
inside the application stylesheet, so anything styled *by* that sheet grows
for free. Two kinds of surface are not:

* a widget that sets its **own** stylesheet — in Qt a sheet set on a widget
  beats the application sheet whatever the selector says, so a literal
  ``font-size: 13px`` in it pins that widget at 13 px for ever;
* a widget that paints text with a ``QPainter``.

Four were reported: the tab strips, the right-hand Home panel, the hover
tooltip, and the text buttons — "Live" and "AI". Each had a hard-coded
pixel number; each now goes through :func:`spacr.qt.theme.font_px`.

**Measured as rendered geometry.** A stylesheet string containing the right
number proves only that the string was built — not that Qt applied it, not
that the widget it was set on is the one that draws the text, and not that a
per-widget sheet did not override it afterwards. Everything here reads
``QFontMetrics`` off the **widget's resolved font**, and the toggle is
additionally measured by counting rows of ink in a real render.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QApplication, QLabel, QTabWidget, QWidget

from spacr.qt import preferences as prefs

#: 100 % and the shipped default. The ratio between them is what every
#: assertion below is really about.
SMALL, LARGE = 1.0, 1.5

#: Three apps is enough for Home to build its tabs and its aside.
APPS = [
    ("mask", "Mask", "Segment cells", "Core"),
    ("measure", "Measure", "Measure objects", "Core"),
    ("map_barcodes", "Map Barcodes", "Sequencing", "Core"),
]


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, tmp_path):
    """Never write to the developer's real preferences."""
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    assert str(tmp_path) in store.fileName(), (
        "QSettings isolation failed; refusing to write to real preferences")
    return store


@pytest.fixture
def app_theme_restored(qt_theme_applied):
    """Put the shared QApplication back at 100 % afterwards.

    ``apply_preferences_to_app`` restyles the whole application; leaving it
    at 150 % would move every later test that measures a pixel.
    """
    yield
    from spacr.qt.theme import apply_qpalette, stylesheet
    apply_qpalette(qt_theme_applied)
    qt_theme_applied.setStyleSheet(stylesheet())


def _at(scale: float):
    """Apply ``scale`` to the running application and settle the events."""
    prefs.set_theme("dark")
    prefs.set_font_scale(scale)
    prefs.apply_preferences_to_app(QApplication.instance())
    QApplication.processEvents()


def _text_height(widget) -> int:
    """Rendered line height of ``widget``'s resolved font, in pixels."""
    return QFontMetrics(widget.font()).height()


def _ink_rows(widget) -> int:
    """Pixel rows of ``widget``'s render that are not the background.

    The only measurement here that does not trust Qt's font metrics at all:
    it grabs the widget and counts the rows the glyphs actually mark.
    """
    from collections import Counter
    image = widget.grab().toImage()
    if image.width() == 0 or image.height() == 0:
        return 0
    counts = Counter()
    for y in range(image.height()):
        for x in range(image.width()):
            counts[image.pixel(x, y)] += 1
    background = counts.most_common(1)[0][0]
    return sum(1 for y in range(image.height())
               if any(image.pixel(x, y) != background
                      for x in range(image.width())))


def _grew(small: float, large: float, what: str) -> None:
    """Assert ``large`` is bigger than ``small`` by roughly the scale ratio."""
    assert large > small, (
        f"{what} renders at {small} px at {SMALL:.0%} and {large} px at "
        f"{LARGE:.0%} — Zoom does not reach it")
    ratio = large / small
    assert 1.2 <= ratio <= 1.8, (
        f"{what} scaled by x{ratio:.2f} between {SMALL:.0%} and {LARGE:.0%}, "
        f"which is not the x{LARGE / SMALL:.2f} the preference asked for")


# ---------------------------------------------------------------------------
# Tab text
# ---------------------------------------------------------------------------

def test_zoom_reaches_a_plain_tab_strip(qtbot, app_theme_restored):
    """The application sheet's own ``QTabBar::tab`` states its size."""
    tabs = QTabWidget()
    tabs.addTab(QWidget(), "Runs found")
    qtbot.addWidget(tabs)
    tabs.show()

    _at(SMALL)
    small = _text_height(tabs.tabBar())
    _at(LARGE)
    large = _text_height(tabs.tabBar())
    _grew(small, large, "tab text")


def test_zoom_reaches_the_home_tab_strip(qtbot, app_theme_restored):
    """Home's tab strip sets its own sheet, so it needs its own scaling.

    This is the one the report was about: a per-widget ``setStyleSheet``
    outranks the application sheet, so ``font-size: 13px`` in ``_tab_qss``
    pinned the category tabs at 13 px at every Zoom setting.
    """
    from spacr.qt.widgets.home import HomePage

    _at(SMALL)
    page = HomePage(APPS, lambda key: None)
    qtbot.addWidget(page)
    page.show()
    QApplication.processEvents()
    small = _text_height(page._tabs.tabBar())

    _at(LARGE)
    page_large = HomePage(APPS, lambda key: None)
    qtbot.addWidget(page_large)
    page_large.show()
    QApplication.processEvents()
    large = _text_height(page_large._tabs.tabBar())

    _grew(small, large, "Home tab text")


# ---------------------------------------------------------------------------
# The right-hand Home panel
# ---------------------------------------------------------------------------

def _aside_labels(page):
    """Every label with text in Home's fixed-width right-hand column."""
    width = prefs.scaled_px(type(page).ASIDE_W)
    for widget in page.findChildren(QWidget):
        if widget.minimumWidth() == widget.maximumWidth() == width:
            labels = [lab for lab in widget.findChildren(QLabel)
                      if lab.text().strip()]
            if labels:
                return labels
    raise AssertionError("could not find Home's aside column")


def test_zoom_reaches_the_right_hand_home_panel(qtbot, app_theme_restored):
    """Its labels each carried a literal 10–14 px in an inline sheet."""
    from spacr.qt.widgets.home import HomePage

    _at(SMALL)
    page = HomePage(APPS, lambda key: None)
    qtbot.addWidget(page)
    page.show()
    QApplication.processEvents()
    small = [_text_height(lab) for lab in _aside_labels(page)]

    _at(LARGE)
    page_large = HomePage(APPS, lambda key: None)
    qtbot.addWidget(page_large)
    page_large.show()
    QApplication.processEvents()
    large = [_text_height(lab) for lab in _aside_labels(page_large)]

    assert small and len(small) == len(large), (
        "the aside built a different number of labels at the two scales")
    stuck = [(a, b) for a, b in zip(small, large) if b <= a]
    assert not stuck, (
        f"{len(stuck)} of {len(small)} labels in Home's right-hand panel "
        f"render at the same height at {SMALL:.0%} and {LARGE:.0%}: {stuck}")
    _grew(sum(small), sum(large), "Home aside text")


# ---------------------------------------------------------------------------
# Tooltips
# ---------------------------------------------------------------------------

def test_zoom_reaches_the_hover_tooltip(qtbot, app_theme_restored):
    """The popup is a separate top-level window with an inline sheet."""
    from spacr.qt.widgets.hover_tooltip import HoverTooltip

    anchor = QLabel("anchor")
    qtbot.addWidget(anchor)
    anchor.show()

    def measure(scale):
        _at(scale)
        tip = HoverTooltip.instance()
        tip.show_for(anchor, "A description that wraps a little.", None)
        QApplication.processEvents()
        labels = [lab for lab in tip.findChildren(QLabel)
                  if lab.text().strip()]
        assert labels, "the tooltip showed no text"
        height = _text_height(labels[0])
        tip.hide()
        return height

    _grew(measure(SMALL), measure(LARGE), "hover tooltip text")


def test_zoom_reaches_the_native_tooltip_sheet(app_theme_restored):
    """``QToolTip`` is styled by the application sheet, which scales."""
    import re
    from spacr.qt.theme import stylesheet

    def size(scale):
        block = re.search(r"QToolTip \{(.*?)\}",
                          stylesheet(font_scale=scale), re.S)
        assert block, "the stylesheet has no QToolTip block"
        found = re.search(r"font-size:\s*(\d+)px", block.group(1))
        assert found, "the QToolTip block states no font size"
        return int(found.group(1))

    _grew(size(SMALL), size(LARGE), "native tooltip text")


# ---------------------------------------------------------------------------
# Text buttons — Live and AI
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Live", "AI"])
def test_zoom_reaches_the_text_toggles(text, qtbot, app_theme_restored):
    """Measured twice: by font metrics, and by counting rows of ink.

    ``AiToggleLabel`` built its sheet once, in ``__init__``, from
    ``FONT_SIZE['body']`` — so it was pinned at 13 px *and* would not have
    picked a new size up on a preferences save even if the number had been
    right. Both halves are asserted: the size, and that saving preferences
    with the widget already on screen moves it.
    """
    from spacr.qt.widgets.ai_toggle_label import AiToggleLabel

    _at(SMALL)
    toggle = AiToggleLabel(text=text)
    qtbot.addWidget(toggle)
    toggle.show()
    QApplication.processEvents()
    small_height, small_ink = _text_height(toggle), _ink_rows(toggle)

    # The SAME widget, still on screen — this is the live-update half.
    _at(LARGE)
    QApplication.processEvents()
    large_height, large_ink = _text_height(toggle), _ink_rows(toggle)

    _grew(small_height, large_height, f"the {text} toggle's text")
    assert large_ink > small_ink, (
        f"the {text} toggle renders {small_ink} rows of ink at {SMALL:.0%} "
        f"and {large_ink} at {LARGE:.0%} — the glyphs did not actually grow")


def test_the_flat_preview_controls_match_the_toggles(app_theme_restored):
    """The controls beside Live/AI share its look, so they share its size."""
    import re
    from spacr.qt.widgets.preview_controls import _flat_qss

    def size(scale):
        _at(scale)
        found = re.search(r"font-size:\s*(\d+)px", _flat_qss("QPushButton"))
        assert found, "the flat control QSS states no font size"
        return int(found.group(1))

    _grew(size(SMALL), size(LARGE), "the flat preview controls")
