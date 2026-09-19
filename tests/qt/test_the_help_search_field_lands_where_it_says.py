"""Type into the field beside Help, press Return, and see where you end up.

Instruction 422, decided by the maintainer on 2026-09-19: "Build as
recommended" -- fuzzy matching over names AND descriptions; one result row per
module where a setting appears; offline, show the local docstring and say the
web page isn't reachable (no dead links); opening a result never discards the
current module's settings; a result opens its module with only the matching
category expanded and the setting highlighted; API entries open the right API
page; preferences open the right Preferences tab.

Every test here drives the real control: it puts text in the box the way a
user does and then asks the WINDOW what happened, not the search module what
it thinks it did. The index is built once for the whole file and handed to the
field -- building it costs about a second, and it is the same list the field's
own worker would have produced.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest


@pytest.fixture(scope="module")
def help_index_rows():
    """The real index, built once. About a second; 11,000 rows."""
    from spacr.qt.help_index import build_index

    return build_index()


@pytest.fixture
def window(qtbot, qt_theme_applied):
    """A real main window with the field installed, as a user gets it."""
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1400, 900)
    win.show()
    qtbot.waitExposed(win)
    return win


@pytest.fixture
def field(window, help_index_rows):
    """The installed field, loaded with the index."""
    from spacr.qt.help_search import field_of

    found = field_of(window)
    assert found is not None, "no search field was installed beside Help"
    found.set_index(help_index_rows)
    return found


@pytest.fixture(autouse=True)
def _the_documentation_probe_never_reaches_the_network(monkeypatch):
    """No test here asks the internet anything; each says what it is.

    The probe is a process singleton, so it is replaced rather than poked,
    and the real one is put back afterwards.
    """
    from spacr.qt import help_search

    monkeypatch.setattr(help_search, "_REACH", help_search._DocsReach())
    monkeypatch.setattr(help_search._DocsReach, "start", lambda self: None)
    yield


def test_the_field_sits_beside_the_help_menu(window):
    """Beside, not inside: it is in the strip that follows the last menu.

    Help is the last menu, and the menu bar's top-right corner widget is
    what comes after it -- the same strip the minimise and close marks are
    in, with the field in front of them.
    """
    from spacr.qt.help_search import FIELD_NAME, field_of

    corner = window.menuBar().cornerWidget(Qt.Corner.TopRightCorner)
    assert corner is not None
    found = field_of(window)
    assert found is not None
    assert found.parent() is corner
    assert found.objectName() == FIELD_NAME
    row = corner.layout()
    order = [row.itemAt(i).widget().objectName() for i in range(row.count())]
    assert order[0] == FIELD_NAME, order
    assert "CloseWindow" in order


def test_a_setting_returns_one_row_per_module_that_shows_it(field):
    """The maintainer's answer to the open question the instruction left.

    One row per module, each naming its module and category, rather than one
    row that then asks which module was meant.
    """
    results = field.type_and_search("cell_min_size")
    settings = [e for e in results if e.kind == "setting"
                and e.title == "cell_min_size"]
    modules = [e.payload["app"] for e in settings]
    assert len(modules) == len(set(modules)), modules
    assert len(modules) >= 2, modules
    for row in settings:
        assert row.subtitle.startswith(
            dict((k, n) for k, n, _d, _s in __import__(
                "spacr.qt.app", fromlist=["APPS"]).APPS)[row.payload["app"]])


def test_a_setting_also_offers_the_api_entry_of_what_reads_it(field):
    """The other half of a setting result, as the instruction describes it."""
    results = field.type_and_search("cell_diameter")
    reading = [e for e in results
               if e.kind == "api" and "cell_diameter" in e.payload.get("reads", "")]
    assert reading, [e.title for e in results]
    assert any(e.title == "spacr.core.preprocess_generate_masks"
               for e in reading), [e.title for e in reading]


def test_return_opens_the_module_with_only_that_category_open(qtbot, window,
                                                              field):
    """The failure mode this feature exists to prevent, pinned.

    "Dumping the user into a fully expanded settings panel and leaving them
    to find the row is the failure mode this exists to prevent."
    """
    results = field.type_and_search("cell_diameter")
    row = next(e for e in results
               if e.kind == "setting" and e.title == "cell_diameter")
    field._list.setCurrentRow(results.index(row))
    QTest.keyClick(field, Qt.Key_Return)
    qtbot.wait(50)

    screen = window._screens.get(row.payload["app"])
    assert screen is not None
    assert window._stack.currentWidget() is screen
    bar = screen._settings_search
    assert bar.revealed_key() == "cell_diameter"
    open_sections = [s for s in screen._settings_sections
                     if hasattr(s, "is_expanded") and s.is_expanded()]
    assert len(open_sections) == 1, [s.title for s in open_sections]
    assert bar.section_of("cell_diameter") is open_sections[0]


def test_the_revealed_row_is_marked_and_the_module_is_not_filtered(qtbot,
                                                                   window,
                                                                   field):
    """Revealed, not filtered: every other setting is still on the form.

    Filtering would hide the neighbouring rows, so a user who arrived from
    the search and then wanted to look around would have to work out what
    had happened to the panel.
    """
    from spacr.qt.help_search import reveal_setting

    assert reveal_setting(window, "mask", "cell_diameter")
    qtbot.wait(50)
    bar = window._screens["mask"]._settings_search
    assert bar.query() == ""
    assert not bar.modified_only()
    assert len(bar.visible_keys()) > 50, len(bar.visible_keys())
    section, widget = bar._index["cell_diameter"]
    assert widget.property("spacrRevealed") is True


def test_opening_a_result_never_discards_the_settings_already_typed(qtbot,
                                                                    window,
                                                                    field):
    """The maintainer's fourth answer, and the one worth a regression test.

    "opening a result never discards the current module's settings (they stay
    in memory — verify that is true and make it true if not)". It WAS true:
    ``_on_nav_selected`` keeps every screen it builds and switches the stack
    rather than rebuilding. This is what stops that being taken away by
    accident later.
    """
    from spacr.qt.help_search import reveal_setting

    assert reveal_setting(window, "mask", "cell_diameter")
    qtbot.wait(20)
    mask = window._screens["mask"]
    mask._settings_model._widgets["cell_diameter"].set_value(77)

    results = field.type_and_search("Measure")
    module = next(e for e in results if e.kind == "module")
    field._list.setCurrentRow(results.index(module))
    QTest.keyClick(field, Qt.Key_Return)
    qtbot.wait(50)
    assert window._stack.currentWidget() is not mask

    assert reveal_setting(window, "mask", "cell_diameter")
    qtbot.wait(20)
    assert window._screens["mask"] is mask
    assert str(
        mask._settings_model._widgets["cell_diameter"].get_value()) == "77"


def test_a_preference_result_names_the_tab_it_opens(field):
    """Preferences open the right Preferences tab."""
    results = field.type_and_search("png resolution")
    row = next(e for e in results if e.kind == "preference")
    assert row.title == "PNG resolution"
    assert row.payload["tab"] == "PreferencesTabFigures"
    assert row.subtitle.endswith("Figures")


def test_the_named_tab_is_the_one_that_comes_up(qtbot, window):
    """And the row on it is scrolled to and marked.

    The dialog is navigated rather than exec'd: ``exec`` blocks, and what is
    under test is where the dialog lands, not that Qt can run a modal loop.
    """
    from spacr.qt import preferences_navigation as navigation
    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog(window)
    qtbot.addWidget(dialog)
    assert navigation.show_tab(dialog, "PreferencesTabFigures",
                               "PNG resolution")
    tabs = navigation.tab_widget(dialog)
    current = tabs.widget(tabs.currentIndex())
    page = navigation.page_named(dialog, "PreferencesTabFigures")
    assert page is current or page in current.findChildren(type(page))
    qtbot.wait(20)
    widget = navigation.row_field(page, "PNG resolution")
    assert widget is not None
    assert widget.property("spacrRevealed") is True


def test_every_offered_preference_is_a_row_the_dialog_really_has(qtbot,
                                                                 window):
    """The drift guard: a generated list still has to match what is built.

    This is the check the instruction's central worry asks for -- "a map that
    is hand-maintained will drift the first time a setting is renamed" -- and
    it is made against the dialog that is actually built, not against another
    list.

    The index is what the field OFFERS, which is not quite the generated
    table: the Fractal page exists only while the fractal backdrop is on, so
    its rows are offered only then. The generator turned the backdrop on to
    see them; this asks the same question this process can answer.
    """
    from spacr.qt import preferences_navigation as navigation
    from spacr.qt.help_index import preference_entries
    from spacr.qt.preferences import PreferencesDialog
    from spacr.qt.theme import spaceout_enabled

    offered = preference_entries()
    assert len(offered) > 100
    if not spaceout_enabled():
        assert not any(e.payload["tab"] == "PreferencesTabFractal"
                       for e in offered)

    dialog = PreferencesDialog(window)
    qtbot.addWidget(dialog)
    missing = []
    for row in offered:
        page = navigation.page_named(dialog, row.payload["tab"])
        if page is None or navigation.row_field(page, row.title) is None:
            missing.append((row.title, row.payload["tab"]))
    assert missing == [], missing


def test_an_api_result_opens_the_page_when_the_page_answers(field,
                                                            monkeypatch):
    """API entries open the right API page."""
    from spacr.qt import help_search

    opened = []
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    help_search.docs_reach().set_state("reachable")

    results = field.type_and_search("preprocess_generate_masks")
    row = next(e for e in results if e.kind == "api")
    field._list.setCurrentRow(results.index(row))
    QTest.keyClick(field, Qt.Key_Return)

    assert opened == [
        "https://einarolafsson.github.io/spacr/api/spacr/core/index.html"
        "#spacr.core.preprocess_generate_masks"]


def test_offline_it_shows_the_local_docstring_and_says_so(qtbot, field,
                                                          monkeypatch):
    """No dead links. A result that opens a 404 is worse than no result."""
    from PySide6.QtWidgets import QTextBrowser
    from spacr.qt import help_search

    opened = []
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    help_search.docs_reach().set_state("unreachable")

    results = field.type_and_search("preprocess_generate_masks")
    row = next(e for e in results if e.kind == "api")
    field._list.setCurrentRow(results.index(row))
    QTest.keyClick(field, Qt.Key_Return)
    qtbot.wait(20)

    assert opened == []
    dialog = field.window().findChild(
        help_search.ApiEntryDialog, help_search.DIALOG_NAME)
    if dialog is None:
        from PySide6.QtWidgets import QApplication

        dialog = next(w for w in QApplication.topLevelWidgets()
                      if isinstance(w, help_search.ApiEntryDialog))
    qtbot.addWidget(dialog)
    body = dialog.findChild(QTextBrowser)
    assert "Cellpose" in body.toPlainText()
    assert "not reachable" in dialog._note.text()
    assert not dialog._open.isEnabled()
    dialog.close()


def test_the_local_docstring_is_read_without_importing_the_module():
    """The offline path must not drag the scientific stack into the process."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    probe = (
        "import sys;"
        "from spacr.qt.help_search import local_docstring;"
        "text = local_docstring('spacr.core.preprocess_generate_masks');"
        "print(len(text) > 200, 'spacr.core' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=str(root),
                         capture_output=True, timeout=300)
    assert out.stdout.decode().strip().endswith("True False"), (
        out.stdout.decode(), out.stderr.decode())


def test_the_keyboard_alone_gets_there(qtbot, window, field):
    """Focus shortcut, arrows, Return, Esc — no mouse anywhere."""
    from spacr.qt.help_search import focus_field

    assert focus_field(window)
    assert field.hasFocus()

    QTest.keyClicks(field, "Measure")
    field._debounce.stop()
    field._refresh()
    assert field.popup().isVisible()
    assert field._list.currentRow() == 0

    QTest.keyClick(field, Qt.Key_Down)
    assert field._list.currentRow() == 1

    QTest.keyClick(field, Qt.Key_Up)
    assert field._list.currentRow() == 0
    chosen = field.results()[0]

    with qtbot.waitSignal(field.opened, timeout=2000) as caught:
        QTest.keyClick(field, Qt.Key_Return)
    assert caught.args[0] is chosen
    assert not field.popup().isVisible()


def test_escape_closes_the_list_and_then_clears_the_box(field):
    """Two presses, two meanings, and neither is "lose what I typed"."""
    field.type_and_search("mask")
    assert field.popup().isVisible()
    QTest.keyClick(field, Qt.Key_Escape)
    assert not field.popup().isVisible()
    assert field.text() == "mask"
    QTest.keyClick(field, Qt.Key_Escape)
    assert field.text() == ""


def test_typing_before_the_index_lands_says_so_rather_than_nothing(window,
                                                                   qtbot):
    """An empty list would read as "there is no such thing"."""
    from spacr.qt.help_search import field_of

    fresh = field_of(window)
    fresh._loader.start = lambda: None
    fresh.setText("mask")
    fresh._on_text_edited("mask")
    fresh._debounce.stop()
    fresh._refresh()
    assert "index" in fresh.note().lower()
    assert fresh.results() == []


def test_the_shortcut_is_bound_and_is_not_one_of_the_others():
    """Ctrl+F already means "find a setting on this module"."""
    from spacr.qt.shortcuts import SHORTCUTS, installed

    keys = [s.keys for s in SHORTCUTS]
    assert keys.count("Ctrl+Shift+H") == 1
    assert "Ctrl+Shift+H" in [s.keys for s in installed()]
    assert keys.count("Ctrl+F") == 1


def test_arriving_here_does_not_rewrite_the_essentials_choice(qtbot, window):
    """A lookup must not quietly change a preference the user set.

    The module remembers whether it opens on Essentials or on All settings.
    Revealing a row that Essentials already shows leaves that choice alone;
    only a row Essentials is HIDING is worth raising the level for, and then
    the strip says so on screen.
    """
    from spacr.qt.help_search import reveal_setting
    from spacr.qt.settings_search import ALL, ESSENTIALS

    assert reveal_setting(window, "mask", "cell_diameter")
    bar = window._screens["mask"]._settings_search
    essential = set(bar._model.essential_keys())
    shown = [k for k in bar.indexed_keys() if k in essential]
    hidden = [k for k in bar.indexed_keys() if k not in essential]
    assert shown and hidden

    bar.set_level(ESSENTIALS)
    assert reveal_setting(window, "mask", shown[0])
    qtbot.wait(20)
    assert bar.level() == ESSENTIALS, shown[0]
    assert bar.revealed_key() == shown[0]

    assert reveal_setting(window, "mask", hidden[0])
    qtbot.wait(20)
    assert bar.level() == ALL, hidden[0]
    assert hidden[0] in bar.visible_keys()
