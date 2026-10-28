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
        qtbot, _empty_journal, immediate_hover_help):
    """home.py:2003 -- ``if hint:`` False, straight to the base handler.

    ``_tile_hints`` is keyed on the tiles the page installed itself on. An
    Enter from anything else -- a panel, a label, a scroll area -- must not
    rewrite the hint bar, while an Enter from a real tile must.
    """
    page = HomePage(APPS, lambda key: None)
    qtbot.addWidget(page)

    page.resize(1200, 860)
    page.show()
    qtbot.waitExposed(page)
    default = page._hint_bar.text()
    stranger = QWidget(page)

    page.eventFilter(stranger, QEvent(QEvent.Enter))
    assert page._hint_bar.text() == default

    tile = next(iter(page._tile_hints))
    immediate_hover_help(tile, lambda: page._hint_bar.text() != default)
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


@pytest.fixture
def classic_preferences(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    settings = QSettings(str(tmp_path / 'classic-home.ini'), QSettings.IniFormat)
    monkeypatch.setattr(preferences, '_settings', lambda: settings)
    return preferences


@pytest.fixture
def classic_journal(_empty_journal):
    import json

    manifests = []
    for index, module in enumerate(('mask', 'measure')):
        folder = _empty_journal / f'2020-01-01_12000{index}_{module}'
        folder.mkdir()
        path = folder / 'manifest.json'
        path.write_text(json.dumps({'app_key': module, 'status': 'success',
            'start_utc': f'2020-01-01T12:00:0{index}+00:00', 'elapsed_s': 3}), encoding='utf-8')
        manifests.append(path)
    return {path: path.read_bytes() for path in manifests}


def test_registry_failure_keeps_real_recent_rows_named_and_navigable(qtbot, classic_preferences, classic_journal):
    from PySide6.QtWidgets import QPushButton

    def unavailable():
        raise RuntimeError('application registry unavailable')

    panel = RecentRunsPanel(known_keys=unavailable, read_now=False)
    qtbot.addWidget(panel)
    rows = panel.read()
    assert {row['app_key'] for row in rows} == {'mask', 'measure'}
    assert panel.name_for('measure') == 'measure'
    panel.refresh(rows)
    names = {label.text() for label in panel.findChildren(QLabel)}
    assert {'mask', 'measure'} <= names
    chosen = []
    panel.run_clicked.connect(chosen.append)
    buttons = [panel.body_layout.itemAt(index).widget() for index in range(panel.body_layout.count())]
    assert all(isinstance(button, QPushButton) for button in buttons)
    for button in buttons:
        button.click()
    assert chosen == [row['app_key'] for row in rows]
    assert 'Run History' in panel._clear.toolTip()
    assert all(path.read_bytes() == payload for path, payload in classic_journal.items())


def test_recent_run_import_failure_shows_a_safe_empty_panel(qtbot, monkeypatch, classic_preferences):
    import builtins

    panel = RecentRunsPanel(read_now=False)
    qtbot.addWidget(panel)
    original = builtins.__import__

    def unavailable(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'spacr.run_journal' and 'recent_runs' in (fromlist or ()):
            raise ImportError('journal service unavailable')
        return original(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, '__import__', unavailable)
        assert panel.read() == []
        panel.refresh()
    assert any(label.text() == 'No runs yet.' for label in panel.findChildren(QLabel))
    assert not panel._clear.isEnabled()


def test_dashboard_reset_signals_move_only_watermarks_and_keep_manifests(qtbot, classic_preferences, classic_journal):
    recent = RecentRunsPanel(known_keys={'mask': 'Segment', 'measure': 'Measure'}, read_now=False)
    totals = TotalsPanel(read_now=False)
    qtbot.addWidget(recent)
    qtbot.addWidget(totals)
    assert totals.read()['total_runs'] == 2
    recent.refresh()
    totals.refresh()
    assert len(recent.read()) == 2
    cleared, reset = [], []
    recent.cleared.connect(lambda: cleared.append(True))
    totals.reset_requested.connect(lambda: reset.append(True))
    recent._clear.click()
    totals._reset.click()
    assert cleared == [True] and reset == [True]
    assert classic_preferences.get_dashboard_watermark('runs')
    assert classic_preferences.get_dashboard_watermark('totals')
    assert recent.read() == []
    assert totals.read()['total_runs'] == 0
    assert not recent._clear.isEnabled() and not totals._reset.isEnabled()
    assert all(path.read_bytes() == payload for path, payload in classic_journal.items())


@pytest.mark.parametrize('failed_service', ['watermark', 'journal'])
def test_totals_service_failures_preserve_the_real_journal(qtbot, monkeypatch, classic_preferences, classic_journal, failed_service):
    from spacr import run_journal

    panel = TotalsPanel(read_now=False)
    qtbot.addWidget(panel)

    def unavailable(*args, **kwargs):
        raise OSError('optional dashboard service unavailable')

    if failed_service == 'watermark':
        monkeypatch.setattr(classic_preferences, 'get_dashboard_watermark', unavailable)
        assert panel.read() == dict(total_runs=2, mask_runs=1, measure_runs=1,
                                    classify_runs=0, models_recorded=0)
    else:
        monkeypatch.setattr(run_journal, 'journal_totals', unavailable)
        assert panel.read() == dict(total_runs=0, mask_runs=0, measure_runs=0, models_recorded=0)
        panel.refresh()
        assert not panel._reset.isEnabled()
    assert all(path.read_bytes() == payload for path, payload in classic_journal.items())


@pytest.mark.parametrize('resource', ['missing', 'invalid_json', 'wrong_shape'])
def test_unreadable_bundled_news_keeps_a_real_placeholder(qtbot, tmp_path, monkeypatch, classic_preferences, resource):
    import importlib.resources

    folder = tmp_path / 'package'
    (folder / 'resources').mkdir(parents=True)
    path = folder / 'resources' / 'release_notes.json'
    if resource != 'missing':
        path.write_text('{broken' if resource == 'invalid_json' else '[]', encoding='utf-8')
    with monkeypatch.context() as patch:
        patch.setattr(importlib.resources, 'files', lambda package: folder)
        assert home_mod.NewsPanel.read_releases() == []
        panel = home_mod.NewsPanel()
    qtbot.addWidget(panel)
    assert panel.releases == []
    assert 'No release notes bundled' in panel._placeholder.text()
    assert panel._placeholder.parentWidget() is panel._notes.widget()
    assert panel._notes_column.indexOf(panel._placeholder) >= 0


def test_bad_fetched_news_preserves_the_existing_release_widgets(qtbot, monkeypatch, classic_preferences):
    releases = [{'tag': 'v1.2', 'name': 'Existing release', 'body': 'Preserved notes.'}]
    monkeypatch.setattr(home_mod.NewsPanel, 'read_releases', staticmethod(lambda: releases))
    panel = home_mod.NewsPanel()
    qtbot.addWidget(panel)
    headlines = tuple(panel._headline_labels)
    texts = [label.text() for label in headlines]
    panel.apply_releases(42)
    assert panel.releases == releases
    assert tuple(panel._headline_labels) == headlines
    assert [label.text() for label in headlines] == texts
    block = panel._release_block({'tag': 'v1.3', 'body': '   '})
    labels = block.findChildren(QLabel)
    assert len(labels) == 1 and labels[0].text() == 'v1.3'
    assert labels[0].focusPolicy() == Qt.NoFocus
    assert panel.render_body('  \n', '#123456') == ''
    block.setParent(panel)


def test_news_title_wraps_while_a_long_plain_date_can_scroll(qtbot, monkeypatch, classic_preferences):
    from PySide6.QtWidgets import QFrame

    name = 'A readable release heading with enough words to require several lines on a narrow panel'
    date = 'Wednesday September 30 2026 23:59:59 UTC, localized publication date'
    monkeypatch.setattr(home_mod.NewsPanel, 'read_releases', staticmethod(lambda: [
        {'tag': 'v2', 'name': name, 'published': date, 'body': ''}]))
    panel = home_mod.NewsPanel()
    qtbot.addWidget(panel)
    title, stamp = panel._headline_labels
    title.setFrameStyle(QFrame.Box)
    title.setMargin(4)
    title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
    panel.resize(580, 360)
    panel.show()
    qtbot.waitExposed(panel)
    panel._news_title_fit._fit()
    wide_height = title.property('minimumSize').height()
    panel.resize(280, 360)
    qtbot.waitUntil(lambda: panel._notes.viewport().property('size').width() < 400)
    panel._news_title_fit._fit()
    width = panel._notes.viewport().property('size').width()
    assert title.property('minimumSize').width() == width
    assert title.property('maximumSize').width() == width
    assert title.property('minimumSize').height() > wide_height
    assert title.text() == name and title.accessibleName() == name
    assert stamp.text() == date and not stamp.wordWrap()
    assert stamp.property('maximumSize').width() > width


@pytest.mark.parametrize('state', ['zero_allocation', 'retired_viewport'])
def test_news_title_fitter_declines_unusable_viewports(qtbot, classic_preferences, state):
    from PySide6.QtWidgets import QWidget
    from shiboken6 import isValid

    panel = home_mod.NewsPanel()
    qtbot.addWidget(panel)
    fitter = panel._news_title_fit
    titles = tuple(panel._headline_labels)
    if state == 'retired_viewport':
        previous = fitter._viewport
        content = panel._notes.takeWidget()
        panel._notes.setViewport(QWidget())
        panel._notes.setWidget(content)
        assert not isValid(previous)
    else:
        fitter._viewport.resize(0, 0)
        assert fitter._viewport.property('size').width() == 0
    fitter._fit()
    assert tuple(panel._headline_labels) == titles
    assert isValid(panel) and all(isValid(title) for title in titles)


def test_retired_news_panel_drops_a_late_refresh_request(classic_preferences):
    from shiboken6 import delete

    panel = home_mod.NewsPanel()
    requests = []
    panel.refresh_requested.connect(lambda: requests.append(True))
    panel._ask_for_newer_releases()
    assert requests == [True]
    delete(panel)
    panel._ask_for_newer_releases()
    assert requests == [True]


def test_news_height_is_remembered_in_unscaled_units(qtbot, monkeypatch, classic_preferences):
    panel = home_mod.NewsPanel()
    qtbot.addWidget(panel)
    with monkeypatch.context() as patch:
        patch.setattr(classic_preferences, 'scaled_px', lambda pixels: pixels * 2)
        panel._remember_height(642)
        assert classic_preferences.get_news_height() == 321
        assert panel._stored_height() == 642
