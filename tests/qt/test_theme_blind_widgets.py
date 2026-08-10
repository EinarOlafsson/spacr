"""The light theme must not be rendered with the dark theme's colours.

``spacr.qt.theme`` used to expose a module-level ``PALETTE`` holding the
**dark** palette, and nothing ever updated it. Any widget that did::

    from ..theme import PALETTE
    box.setStyleSheet(f"background: {PALETTE['surface_alt']};")

therefore painted ``#161719`` — near-black — on every theme. On the
light theme that puts a black box on a ``#fafafa`` page, and any text
the app stylesheet inks (light ``fg`` = ``#0d0e10``) lands on it at
**1.08:1**. Black on black.

That is what these tests measure, off a real render, rather than
asserting on structure — the whole point of the bug is that the widget
tree was perfectly correct and the pixels were not.

Three layers, because they fail for different reasons:

1. :func:`spacr.qt.theme.contrast_failures` proves the *palettes*
   are fine. If a page is illegible with a clean palette, the page mixed
   two of them.
2. The featured/news surface is rendered in each theme and the ink the
   app stylesheet supplies is measured against the fill the widget
   paints. This is the 1.08:1 case, through shipped public API
   (:meth:`HomePage.set_reserved_content`).
3. A static sweep pins the modules that still import the frozen dark
   palette. It may only ever shrink.

Layer 2 grew a second case on 2026-08-10: a *filled* danger surface,
where the ink is ``bg`` rather than ``fg``. The sweep in layer 2 flagged
``style_as_danger`` hard-coding ``#ffffff`` for the force-quit button's
hover state — right on light, 2.04:1 on glass — which is the same
mistake as importing the wrong palette, only spelled as a literal.
"""
from __future__ import annotations

import ast
import os
import pathlib
import re

import pytest

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from spacr.qt import bridge, preferences, theme
from spacr.qt.app import make_home_page

#: AA for body text (WCAG 1.4.3) — the same number
#: :data:`spacr.qt.theme.CONTRAST_RULES` demands of `fg` on a surface.
AA_BODY = 4.5

QT_ROOT = pathlib.Path(theme.__file__).resolve().parent

_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")

#: A QSS comment. Qt's parser throws these away before it resolves a
#: single rule, so a hex inside one cannot colour a pixel.
_QSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _painted(sheet: str) -> str:
    """``sheet`` reduced to the part Qt actually parses.

    Added 2026-08-10. The QSS blocks in ``spacr/qt`` document the bug they
    fix *inside the stylesheet*: ``screens/qc_dashboard.py`` gained one on
    2026-08-06 (9cebd643) explaining, in a ``/* ... */``, that an unstyled
    QLabel "paints the WINDOW colour -- #000000 on dark". Sweeping the raw
    text charged that prose with inlining a dark-only colour on every
    non-dark theme. The prose is the point of those blocks and must not
    have to launder its own hexes, so the sweep discards comments exactly
    as Qt does.
    """
    return _QSS_COMMENT.sub(" ", sheet)


@pytest.fixture(autouse=True)
def _empty_registry():
    """The run registry is process-wide; never leak a job between tests."""
    bridge.registry().clear()
    yield
    bridge.registry().clear()


@pytest.fixture
def _empty_journal(tmp_path, monkeypatch):
    from spacr import run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    yield tmp_path


@pytest.fixture
def themed_home(qtbot, qapp, monkeypatch, _empty_journal):
    """Build a Home page rendered *as* the given theme, and hand it back.

    The stylesheet goes on the page rather than on the QApplication: a
    global ``setStyleSheet`` re-polishes every widget any other test left
    behind, which is slow and a good way to crash on a stale one.
    Everything that resolves the theme itself goes through
    ``preferences.get_theme``, which is patched, so a page built here is
    in exactly the state the user's chosen theme would put it in.
    """
    def build(theme_name: str) -> QWidget:
        monkeypatch.setattr(preferences, "get_theme", lambda: theme_name)
        page = make_home_page()  # the page MainWindow ships
        qtbot.addWidget(page)
        page.setStyleSheet(theme.stylesheet(theme_name))
        page.resize(1400, 900)
        page.show()
        qtbot.waitExposed(page)
        qapp.processEvents()
        return page

    return build


def _fill_behind(widget: QWidget) -> str:
    """The colour actually painted behind ``widget``, off a real render."""
    image = widget.grab().toImage()
    assert image.width() > 4 and image.height() > 4, "widget never laid out"
    return QColor(image.pixel(image.width() // 2, 2)).name()


def _ink(label: QLabel) -> str:
    """The colour the label will draw its text in, after styling."""
    label.ensurePolished()
    return label.palette().color(QPalette.WindowText).name()


# ===========================================================================
# 1. The palettes themselves are clean
# ===========================================================================

@pytest.mark.parametrize("theme_name", theme.THEMES)
def test_every_palette_clears_every_contrast_rule(theme_name):
    """The premise. A page that is illegible mixed two palettes."""
    assert theme.contrast_failures(theme_name) == []


def test_the_light_palette_shares_no_surface_with_the_dark_one():
    """Why mixing them is fatal rather than merely ugly.

    `surface_alt` is ``#161719`` dark and ``#f2f4f7`` light. Inlining
    the first while the stylesheet inks for the second is not a near
    miss — the two are at opposite ends of the range.
    """
    dark = theme.DARK_PALETTE
    light = theme.LIGHT_PALETTE
    for role in ("bg", "surface", "surface_alt", "surface_hi"):
        assert theme.contrast_ratio(dark[role], light["fg"]) < 1.5, (
            f"{role}: the dark surface is no longer black-on-black under "
            "the light theme's ink — this test's premise moved")
        assert theme.contrast_ratio(light[role], light["fg"]) >= AA_BODY


# ===========================================================================
# 2. The rendered page
# ===========================================================================

@pytest.mark.parametrize("theme_name", theme.THEMES)
def test_the_featured_surface_is_legible_in_every_theme(themed_home,
                                                        theme_name):
    """Drop content into the reserved surface and measure it on screen.

    ``set_reserved_content`` is the documented escape hatch for a news
    feed / featured panel, and a plain ``QLabel`` is what a caller would
    hand it — no inline colour, so the app stylesheet inks it. That ink
    has to be readable on whatever the panel painted itself.

    Measured on the page this one replaced: **1.08:1** on light
    (``#0d0e10`` on ``#161719``) and 17.94:1 on dark, because the panel
    fill was inlined from the frozen dark palette.
    """
    page = themed_home(theme_name)
    marker = QLabel("Featured content lands here")
    page.set_reserved_content(marker)
    page.window().update()

    ink = _ink(marker)
    fill = _fill_behind(marker.parentWidget())
    ratio = theme.contrast_ratio(ink, fill)
    assert ratio >= AA_BODY, (
        f"{theme_name}: featured content is {ratio:.2f}:1 — ink {ink} on "
        f"a panel painted {fill}")


@pytest.mark.parametrize("theme_name", theme.THEMES)
def test_every_home_panel_is_legible_in_every_theme(themed_home, theme_name):
    """Every captioned box in the right-hand column, ink vs its own fill.

    Panels are where the bug showed: they are the widgets that paint a
    surface colour themselves instead of letting the app stylesheet do
    it.
    """
    page = themed_home(theme_name)
    panels = [w for w in page.findChildren(QWidget)
              if w.objectName() == "HomePanelBox"]
    assert panels, "no Home panels rendered"

    failures = []
    for panel in panels:
        fill = _fill_behind(panel)
        for label in panel.findChildren(QLabel):
            if not label.text().strip():
                continue
            ink = _ink(label)
            ratio = theme.contrast_ratio(ink, fill)
            if ratio < AA_BODY:
                failures.append(
                    f"{label.text()[:24]!r}: {ink} on {fill} = {ratio:.2f}:1")
    assert not failures, f"{theme_name}: {failures}"


@pytest.mark.parametrize("theme_name", theme.THEMES)
def test_home_inlines_no_colour_from_another_theme(themed_home, theme_name):
    """The cause, not the symptom: no widget hard-codes a dark-only hex.

    Only colours that are in :data:`DARK_PALETTE` and *not* in the theme
    on screen are flagged. That deliberately excuses
    :data:`CONSTANT_ROLES` (``button_accent`` and friends are the same
    hex in every theme, on purpose) and anything from outside the
    palettes entirely — this test is about importing the wrong palette,
    not about every literal in the tree.

    Scanned through :func:`_painted`, which drops QSS comments (see its
    docstring). The page's own sheet is the whole application stylesheet,
    put there by the fixture; it is swept anyway rather than excused,
    because a composed sheet that carried a dark-only hex into the light
    theme would be the same bug one level up.
    """
    page = themed_home(theme_name)
    live = {value.lower() for value in theme.palette_for(theme_name).values()}
    dark_only = {value.lower() for value in theme.DARK_PALETTE.values()
                 if value.lower() not in live}
    role_of = {value.lower(): role
               for role, value in theme.DARK_PALETTE.items()}

    offenders = []
    for widget in [page] + page.findChildren(QWidget):
        sheet = _painted(widget.styleSheet())
        if not sheet.strip():
            continue
        for found in {m.group(0).lower() for m in _HEX.finditer(sheet)}:
            if found in dark_only:
                offenders.append(
                    f"{widget.objectName() or type(widget).__name__} "
                    f"inlines {found} (dark {role_of[found]})")
    assert not offenders, f"{theme_name}: {sorted(set(offenders))}"


def test_the_comment_strip_only_removes_comments():
    """Control for :func:`_painted`, so the sweep above cannot go vacuous.

    A strip that ate the rules as well as the comments would make
    ``test_home_inlines_no_colour_from_another_theme`` pass forever
    without looking at anything, which is the failure mode of every
    "ignore the false positive" fix. Both halves are asserted: the hex in
    the comment is gone, the hex in the rule survives, and the surviving
    one is still recognised by the same :data:`_HEX` the sweep uses.
    """
    sheet = ("/* a QLabel with no background paints the WINDOW colour --\n"
             "   #000000 on dark -- behind its own text. */\n"
             "QLabel { color: #161719; background: transparent; }")
    painted = _painted(sheet)
    assert {m.group(0) for m in _HEX.finditer(painted)} == {"#161719"}
    assert "QLabel { color:" in painted


# ===========================================================================
# 2b. Filled danger surfaces
# ===========================================================================

@pytest.mark.parametrize("theme_name", theme.THEMES)
def test_the_force_quit_button_is_legible_while_hovered(qtbot, theme_name):
    """Hovering the force-quit button fills it; the ink has to survive that.

    Found 2026-08-10 by the sweep above, which reported ``DangerButton
    inlines #ffffff`` on glass. :func:`spacr.qt.shutdown.style_as_danger`
    resolved the fill from the live palette but hard-coded the hover ink
    as white, which is only right on light — ``error`` is a *pale* red on
    cell and glass, so the ink on the one control that force-quits a run
    measured **2.20:1** and **2.04:1**, below AA-large.

    Pinned against ``bg`` rather than a literal because that is the role
    the theme already reserves for this: ``CONTRAST_RULES`` carries
    ``("bg", "error", 4.5)`` under the comment "`bg` is the ink on filled
    accent/danger surfaces … DangerButton on hover", and the application
    sheet's own ``#DangerButton:pressed`` rule inks with it.
    """
    from spacr.qt.shutdown import style_as_danger

    palette = theme.palette_for(theme_name)
    button = QPushButton("Force quit")
    qtbot.addWidget(button)
    style_as_danger(button, palette)

    hover = _painted(button.styleSheet()).split(":hover", 1)[1]
    fill = re.search(r"(?<![-\w])background:\s*(#[0-9a-fA-F]{6})",
                     hover).group(1)
    ink = re.search(r"(?<![-\w])color:\s*(#[0-9a-fA-F]{6})", hover).group(1)

    assert fill.lower() == palette["error"].lower(), (
        "the hover fill stopped following the live palette")
    assert ink.lower() == palette["bg"].lower(), (
        f"{theme_name}: hover ink is {ink}, not the palette's bg "
        f"{palette['bg']} — a literal here is right on at most one theme")
    ratio = theme.contrast_ratio(ink, fill)
    assert ratio >= AA_BODY, (
        f"{theme_name}: force-quit reads {ratio:.2f}:1 while hovered — "
        f"ink {ink} on a fill of {fill}")


# ===========================================================================
# 3. The name that was the trap
# ===========================================================================

def test_the_dark_palette_says_it_is_dark():
    """``PALETTE`` is not a module global any more."""
    assert "DARK_PALETTE" in vars(theme)
    assert "PALETTE" not in vars(theme), (
        "PALETTE is back as a real global — the name reads as 'the "
        "palette' and gets imported as one")
    assert theme.DARK_PALETTE["surface_alt"] == "#161719"


def test_reading_the_old_name_warns_and_cannot_be_mutated():
    with pytest.deprecated_call():
        frozen = theme.PALETTE
    assert frozen == theme.DARK_PALETTE
    with pytest.raises(TypeError):
        frozen["fg"] = "#ff0000"
    with pytest.raises(AttributeError):
        theme.NO_SUCH_PALETTE


def test_active_palette_follows_the_preference(monkeypatch):
    for theme_name in theme.THEMES:
        monkeypatch.setattr(preferences, "get_theme", lambda t=theme_name: t)
        assert theme.active_palette() == theme.palette_for(theme_name)


def test_active_palette_falls_back_to_dark_when_preferences_explode(
        monkeypatch):
    def boom():
        raise RuntimeError("no settings backend")
    monkeypatch.setattr(preferences, "resolve_effective_theme", boom)
    assert theme.active_palette() == theme.palette_for("dark")


# ===========================================================================
# 4. The sweep, pinned
# ===========================================================================

#: Every module under ``spacr/qt`` that still reads the frozen dark
#: palette. Each entry is a light-theme rendering bug — the colours it
#: inlines do not change when the theme does.
#:
#: This list may only ever SHRINK. It is asserted as an upper bound, not
#: an equality, so fixing one of them does not fail the suite; adding a
#: new one does.
#:
#: ``spacr/qt/screens/startup.py`` used to head this list with 24 sites.
#: It was deleted, not fixed: ``qt/widgets/home.py`` had already replaced
#: it and nothing in the app imported it.
STILL_READS_THE_DARK_PALETTE = {
    "app.py",                       # unused import
    "screens/agreement.py",
    "screens/align.py",             # incl. paintEvent
    "screens/app_screen.py",
    "screens/batch.py",
    "screens/convert.py",
    "screens/db_browser.py",
    "screens/foreign.py",
    "screens/make_masks.py",        # incl. paintEvent
    "screens/model_compare.py",
    "screens/model_zoo.py",
    "screens/plate_view.py",        # incl. paintEvent
    "screens/report.py",
    "screens/train_compare.py",
    "widgets/ai_chat_panel.py",
    "widgets/ai_toggle_label.py",   # incl. paintEvent
    "widgets/console_panel.py",     # module-level COLOR_* constants
    "widgets/empty_state.py",       # unused import
    "widgets/hover_tooltip.py",
    "widgets/measure_preview.py",
    "widgets/metadata_table.py",
    "widgets/toggle.py",            # incl. paintEvent
}


def _modules_reading_the_dark_palette() -> set:
    """Every ``spacr/qt`` module that names ``PALETTE``, by AST.

    Parsed rather than grepped so a module that only *mentions* the name
    in a docstring — ``widgets/home.py`` explains the bug in its own —
    is not counted as a consumer.
    """
    found = set()
    for path in sorted(QT_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            imported = (isinstance(node, ast.ImportFrom)
                        and (node.module or "").endswith("theme")
                        and any(a.name == "PALETTE" for a in node.names))
            attribute = (isinstance(node, ast.Attribute)
                         and node.attr == "PALETTE")
            if imported or attribute:
                found.add(path.relative_to(QT_ROOT).as_posix())
                break
    return found


def test_no_new_module_reads_the_frozen_dark_palette():
    found = _modules_reading_the_dark_palette()
    new = found - STILL_READS_THE_DARK_PALETTE
    assert not new, (
        f"{sorted(new)} import spacr.qt.theme.PALETTE. That is the DARK "
        "palette and nothing updates it, so these render dark chrome on "
        "the light theme. Use theme.active_palette().")


def test_the_deleted_home_screen_is_gone_and_unreferenced():
    """``screens/startup.py`` was replaced by ``widgets/home.py``.

    Asserted rather than assumed: the app's only Home is the one
    ``MainWindow`` installs, and a second one drifting alongside it is
    how the light-theme bug survived being fixed once already.

    Scoped to ``spacr/qt`` — the *application*. The home-screen variant
    generator under ``spacr/resources/home/versions/_generators`` is a
    review artefact, not shipped UI, and imports its subject lazily
    inside one variant function.
    """
    assert not (QT_ROOT / "screens" / "startup.py").exists()
    offenders = []
    for path in sorted(QT_ROOT.rglob("*.py")):
        text = path.read_text()
        if "screens.startup" in text or "screens import startup" in text:
            offenders.append(os.fspath(path))
    assert not offenders, f"still import the deleted Home screen: {offenders}"


def test_home_is_the_only_home_the_window_installs():
    """Was a grep of ``_install_startup_page`` alone for "HomePage".

    That method now calls ``make_home_page``, the single constructor the
    window and the suite share: the two groupings, the notes and the
    icon provider are four arguments that have to agree, and a test that
    assembled its own HomePage was exercising a page nobody ships. The
    indirection is followed rather than trusted.
    """
    import inspect
    from spacr.qt import app as qt_app
    from spacr.qt.widgets.home import HomePage as _HomePage
    source = inspect.getsource(qt_app.MainWindow._install_startup_page)
    source += inspect.getsource(qt_app.make_home_page)
    assert "HomePage" in source
    assert "StartupPage" not in source
    page = qt_app.make_home_page()
    assert isinstance(page, _HomePage)
    page.deleteLater()
