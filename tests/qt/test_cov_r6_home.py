"""Round-6 coverage for ``spacr.qt.widgets.home``: the Home screen with a
piece missing.

Every branch pinned here is one the page takes when something it usually
has is absent -- a banner with no job bound to it, a panel body holding a
spacer instead of a widget, a build with no logo artwork, a grouping that
names an app the registry no longer has, a hover over something that is not
a tile, and the surface sweep run against a page that has not been built.

The one branch left open is ``_clear_page_surfaces``' ``page is not None``
check inside ``for i in range(tabs.count())``: ``self._tabs`` is a
``QTabWidget``, whose ``count()`` is the number of pages it holds, so
``widget(i)`` for ``i < count()`` is never ``None``. That is written up in
the round report rather than faked with a stand-in tab widget.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from spacr.qt.widgets import home as home_mod
from spacr.qt.widgets.home import (
    HomePage,
    QueuedPanel,
    RecentRunsPanel,
    RunningBanner,
    SystemPanel,
    TotalsPanel,
    _find_logo_pixmap,
)

APPS = [
    ("mask", "Mask", "Segment a plate", "Core"),
    ("measure", "Measure", "Measure objects", "Core"),
    ("classify", "Classify", "Train a classifier", "Analysis"),
]


@pytest.fixture
def _empty_journal(tmp_path, monkeypatch):
    from spacr import run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    yield tmp_path


# ---------------------------------------------------------------------------
# _find_logo_pixmap
# ---------------------------------------------------------------------------

def test_a_logo_that_will_not_load_is_not_a_logo(monkeypatch):
    """home.py:145 -- ``if not pix.isNull()`` False, on to the next candidate.

    ``themed_pixmap`` re-inks the artwork for the theme and can hand back a
    null pixmap on a build whose resources did not ship. A null pixmap is
    still truthy in PySide, so ``themed_pixmap(path) or QPixmap(path)``
    cannot catch it -- the ``isNull`` test is what does, and with every
    candidate null the answer is ``None`` rather than an invisible mark.
    """
    from spacr.qt import iconset

    real = _find_logo_pixmap()
    assert real is not None and not real.isNull(), \
        "the bundled logo must be found"

    # Artwork that is on disk but will not decode: the re-inker declines and
    # the raw QPixmap comes back null.
    monkeypatch.setattr(iconset, "themed_pixmap", lambda *a, **k: None)
    monkeypatch.setattr(home_mod, "QPixmap", lambda *a, **k: QPixmap())
    assert _find_logo_pixmap() is None


# ---------------------------------------------------------------------------
# RunningBanner
# ---------------------------------------------------------------------------

class _Handle:
    app_key = "measure"
    supports_pause = False
    last_line = "Progress: 3/9"

    def __init__(self):
        self.done = 3
        self.total = 9
        self.paused = False


def test_an_unbound_banner_opens_nothing(qtbot):
    """home.py:647 -- ``if self._handle is not None`` False, returning.

    The banner is constructed before any job exists, so a click that
    arrives before ``bind`` must emit nothing at all -- an ``open_requested``
    carrying a stale or empty key would navigate the window somewhere the
    user did not ask for.
    """
    banner = RunningBanner(lambda key: None, {"measure": "Measure"})
    qtbot.addWidget(banner)

    seen = []
    banner.open_requested.connect(seen.append)

    banner._on_open()
    assert seen == []

    banner._handle = _Handle()
    banner._on_open()
    assert seen == ["measure"]


# ---------------------------------------------------------------------------
# Panel.refresh -- the "take everything out of the body" loops
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [
    QueuedPanel, RecentRunsPanel, SystemPanel, TotalsPanel])
def test_a_spacer_in_a_panel_body_is_cleared_without_being_deleted(
        qtbot, factory, _empty_journal):
    """home.py:698 / 766 / 831 / 934 -- ``if widget is not None`` False.

    ``body_layout.takeAt(0)`` returns a layout ITEM, and a stretch is an
    item with no widget behind it. ``deleteLater`` on ``None`` is a crash,
    so the clear loop steps over it -- and the widget beside the spacer is
    still taken out, which is what the loop is for.
    """
    panel = factory()
    qtbot.addWidget(panel)

    marker = QLabel("STALE ROW")
    panel.body_layout.addWidget(marker)
    panel.body_layout.addStretch(1)
    before = panel.body_layout.count()
    assert before >= 2

    panel.refresh()

    # The stale row is gone from the layout (deleteLater is queued, so the
    # widget object may still exist -- what matters is that it is no longer
    # in the body).
    remaining = [panel.body_layout.itemAt(i).widget()
                 for i in range(panel.body_layout.count())]
    assert marker not in remaining


# ---------------------------------------------------------------------------
# HomePage._grouping
# ---------------------------------------------------------------------------

def test_a_group_naming_only_dead_apps_loses_its_tab(qtbot, _empty_journal):
    """home.py:1485 -- ``if entries:`` False, so the group is dropped.

    A grouping is a *view* of the registry. One that names an app which no
    longer exists should lose the tile, and a group that names nothing else
    should lose the whole tab -- not raise, and not draw an empty tab.
    """
    page = HomePage(APPS, lambda key: None,
                    categories=[("Real", ["mask", "gone"]),
                                ("Ghosts", ["gone", "also_gone"])])
    qtbot.addWidget(page)

    titles = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert any(t.startswith("Real") for t in titles), titles
    assert not any(t.startswith("Ghosts") for t in titles), titles
    # The dead key cost its tile, not the tab: "Real" holds only "mask".
    assert any(t.startswith("Real  (1)") for t in titles), titles


# ---------------------------------------------------------------------------
# HomePage._build_hero
# ---------------------------------------------------------------------------

def test_a_build_without_the_logo_still_has_a_masthead(qtbot, monkeypatch,
                                                       _empty_journal):
    """home.py:1541 -- ``if logo is not None`` False, so no mark is hung.

    The wordmark and the subtitle are the masthead; the mark is artwork
    that a build may not carry. Losing the artwork must cost the mark and
    nothing else.
    """
    with_logo = HomePage(APPS, lambda key: None)
    qtbot.addWidget(with_logo)
    assert with_logo.findChild(QLabel, "HeroMark") is not None

    monkeypatch.setattr(home_mod, "_find_logo_pixmap", lambda: None)
    without = HomePage(APPS, lambda key: None)
    qtbot.addWidget(without)

    assert without.findChild(QLabel, "HeroMark") is None
    assert without._hero_mark is None
    texts = [w.text() for w in without.findChildren(QLabel)]
    assert "spaCR" in texts


# ---------------------------------------------------------------------------
# HomePage.eventFilter
# ---------------------------------------------------------------------------

def test_hovering_something_that_is_not_a_tile_leaves_the_hint_alone(
        qtbot, _empty_journal):
    """home.py:2003 -- ``if hint:`` False, straight to the base handler.

    ``_tile_hints`` is keyed on the tiles the page installed itself on. An
    Enter from anything else -- a panel, a label, a scroll area -- must not
    rewrite the hint bar, while an Enter from a real tile must.
    """
    page = HomePage(APPS, lambda key: None)
    qtbot.addWidget(page)

    default = page._hint_bar.text()
    stranger = QWidget(page)

    page.eventFilter(stranger, QEvent(QEvent.Enter))
    assert page._hint_bar.text() == default

    tile = next(iter(page._tile_hints))
    page.eventFilter(tile, QEvent(QEvent.Enter))
    assert page._hint_bar.text() != default
    assert page._hint_bar.text()


# ---------------------------------------------------------------------------
# HomePage._clear_page_surfaces
# ---------------------------------------------------------------------------

def test_the_surface_sweep_finds_what_a_built_page_has_and_survives_a_bare_one(
        qtbot, _empty_journal):
    """home.py:1417 / 1430 / 1443 / 1446 -- every surface absent.

    The sweep is written entirely in terms of ``findChild``, ``getattr`` and
    ``layout()``, because it also runs on a theme change that can arrive
    while the page is still being assembled. Run against a host with no
    Hero, no ``_tabs`` and no layout it must clear nothing and raise
    nothing; against a real HomePage it must clear the masthead's labels,
    which is the black band it exists to remove.
    """
    page = HomePage(APPS, lambda key: None)
    qtbot.addWidget(page)
    hero = page.findChild(QWidget, "Hero")
    assert hero is not None
    hero_labels = hero.findChildren(QLabel)
    assert hero_labels
    from spacr.qt.theme import TRANSPARENT_PROPERTY
    for label in hero_labels:
        label.setProperty(TRANSPARENT_PROPERTY, None)
    assert not any(lbl.property(TRANSPARENT_PROPERTY)
                   for lbl in hero_labels)

    page._clear_page_surfaces()
    assert all(lbl.property(TRANSPARENT_PROPERTY) for lbl in hero_labels)

    # The same sweep against a host that has none of those surfaces. It is
    # called unbound because there is no way to build a HomePage without a
    # Hero and a tab widget -- and the point of the guards is precisely the
    # moment before those exist.
    bare = QWidget()
    qtbot.addWidget(bare)
    assert bare.layout() is None
    HomePage._clear_page_surfaces(bare)
    assert bare.findChild(QWidget, "Hero") is None

    # ...and one whose first layout item is a spacer rather than a widget.
    spaced = QWidget()
    qtbot.addWidget(spaced)
    layout = QVBoxLayout(spaced)
    layout.addStretch(1)
    HomePage._clear_page_surfaces(spaced)
    assert layout.itemAt(0).widget() is None
