"""Navigator retains actual Home panels rather than omitting their headings."""
from PySide6.QtCore import Qt

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def _page(qtbot, private_ai_profile):
    from spacr.qt.app import make_home_page
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_show_alpha(True)
    page = make_home_page()
    qtbot.addWidget(page)
    page.resize(1280, 900)
    page.show()
    qtbot.waitUntil(lambda: page._navigator_aside.width() > 0)
    return page


def test_startup_has_visible_native_headings_with_collapsed_bodies(qtbot, private_ai_profile):
    page = _page(qtbot, private_ai_profile)
    try:
        aside = page._navigator_aside
        assert set(aside._navigator_folders) == {'Queued', 'Recent runs', 'News', 'Totals', 'System'}
        for folder in aside._navigator_folders.values():
            assert folder.shut and folder.heading.isVisible() and not folder.body.isVisible()
            point = folder.heading.mapTo(page, folder.heading.rect().center())
            assert page.rect().contains(point)
        assert not aside._journal_jobs.is_busy()
        assert aside._ambient is None and not hasattr(aside, '_ticker')
        assert not hasattr(aside, '_tabs')
    finally:
        page.close()
    assert aside._closing and not aside.active_jobs()


def test_explicit_expansion_survives_recreation_without_changing_normal_home(qtbot, private_ai_profile):
    from spacr.qt.app import make_home_page
    page = _page(qtbot, private_ai_profile)
    folder = page._navigator_aside._navigator_folders['System']
    qtbot.mouseClick(folder.heading, Qt.LeftButton)
    assert not folder.shut and folder.body.isVisible()
    page.close()
    other = _page(qtbot, private_ai_profile)
    try:
        assert not other._navigator_aside._navigator_folders['System'].shut
        assert other._navigator_aside._navigator_folders['News'].shut
        preferences, _, _ = private_ai_profile
        preferences.set_easy_mode(False)
        normal = make_home_page()
        qtbot.addWidget(normal)
        assert not hasattr(normal, '_navigator_aside')
        assert not normal._system._box.isHidden()
        normal.close()
    finally:
        other.close()


def test_workflow_does_not_cover_headers_after_resize_or_splitter_drag(qtbot, private_ai_profile):
    page = _page(qtbot, private_ai_profile)
    try:
        page.field.set_cards([('mask', 'Mask', '', '')])
        page.field.cards[0].activated.connect(page._select_suggestion)
        page._select_suggestion('mask', '')
        qtbot.waitUntil(lambda: page._workflow is not None)
        for width in (1280, 1000):
            page.resize(width, 900)
            qtbot.wait(20)
            assert page._workflow.geometry().right() < page._navigator_aside.geometry().left()
            assert all(folder.heading.isVisible()
                       for folder in page._navigator_aside._navigator_folders.values())
        split = page._navigator_aside_split
        split.setSizes([600, 350])
        split.splitterMoved.emit(600, 1)
        assert page._workflow.width() == page._suggestions_scroll.width()
    finally:
        page.close()


def test_native_panel_signals_forward_without_duplicate_connections(qtbot, private_ai_profile):
    page = _page(qtbot, private_ai_profile)
    calls = []
    page.tile_clicked.connect(lambda key: calls.append(key))
    page._navigator_aside._on_run_clicked('mask')
    page.refresh()
    page._navigator_aside._on_run_clicked('mask')
    assert calls == ['mask', 'mask']
    page.close()


def test_journal_panels_load_only_after_expansion_and_stop_on_close(
        qtbot, private_ai_profile, monkeypatch):
    from spacr.qt.widgets.home import RecentRunsPanel, TotalsPanel
    reads = []
    monkeypatch.setattr(RecentRunsPanel, 'read', lambda self: reads.append('recent') or [])
    monkeypatch.setattr(TotalsPanel, 'read', lambda self: reads.append('totals') or {})
    page = _page(qtbot, private_ai_profile)
    aside = page._navigator_aside
    try:
        assert reads == []
        qtbot.mouseClick(aside._navigator_folders['Recent runs'].heading, Qt.LeftButton)
        qtbot.waitUntil(lambda: len(reads) == 2 and not aside.active_jobs())
        assert reads == ['recent', 'totals']
        assert aside._recent._box.isVisible()
    finally:
        page.close()
    assert aside._closing and not aside.active_jobs()
