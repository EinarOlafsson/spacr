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


def test_enabled_object_helpers_have_search_rows_local_prose_and_exact_links():
    """Publishing a private nested helper also makes its real page findable."""
    from spacr.qt.help_api_index import API_ENTRIES
    from spacr.qt.help_search import api_url, local_docstring

    entries = dict(API_ENTRIES)
    keys = (
        "spacr.object._cellpose_z_segment_fn._segment",
        "spacr.object.merge_split_filter_masks._progress",
        "spacr.object.merge_split_filter_masks._run_one",
    )
    for key in keys:
        assert key in entries
        assert local_docstring(key).startswith(entries[key])
        assert api_url(key, "en").endswith(f"/spacr/object/index.html#{key}")
    assert "spacr.object._cellpose_z_segment_fn" not in entries


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

    It yields the REAL :meth:`_DocsReach.start`, for the one test that is
    about the probe itself: stubbing the method out everywhere is what left
    the thread it starts unrun by anything. That test puts this back and
    stubs the socket instead, so the promise above still holds.
    """
    from spacr.qt import help_search

    real_start = help_search._DocsReach.start
    monkeypatch.setattr(help_search, "_REACH", help_search._DocsReach())
    monkeypatch.setattr(help_search._DocsReach, "start", lambda self: None)
    yield real_start


def test_the_field_sits_directly_right_of_the_help_menu(window):
    """Directly right of Help, and NOT in the window-marks strip.

    It was first installed in the menu bar's top-right corner widget, which
    is right-aligned, so the field sat against the minimise, full screen and
    close marks at the far edge of the window. The maintainer asked on
    2026-09-19 for it "directly to the right of help, not on the side as the
    minimize, expand, close", so it is a plain child of the bar, placed after
    the last menu and re-placed whenever the bar changes shape or language.
    """
    from spacr.qt.help_search import FIELD_NAME, field_of

    bar = window.menuBar()
    found = field_of(window)
    assert found is not None
    assert found.objectName() == FIELD_NAME
    assert found.parent() is bar

    corner = bar.cornerWidget(Qt.Corner.TopRightCorner)
    if corner is not None:
        row = corner.layout()
        if row is not None:
            marks = [row.itemAt(i).widget().objectName()
                     for i in range(row.count())]
            assert FIELD_NAME not in marks, marks
            assert "CloseWindow" in marks

    menus = [bar.actionGeometry(a) for a in bar.actions()
             if not a.isSeparator() and a.isVisible()]
    assert menus, "the bar has no menus to sit beside"
    last_menu_right = max(rect.right() for rect in menus)
    assert found.geometry().left() >= last_menu_right
    assert found.geometry().left() - last_menu_right <= 24, (
        found.geometry(), last_menu_right)
    if corner is not None and corner.width():
        assert found.geometry().right() < bar.width() - corner.width()


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


@pytest.mark.parametrize("spaceout", [False, True])
def test_every_offered_preference_is_a_row_the_dialog_really_has(
    qtbot, window, monkeypatch, spaceout,
):
    """The drift guard: a generated list still has to match what is built.

    This is the check the instruction's central worry asks for -- "a map that
    is hand-maintained will drift the first time a setting is renamed" -- and
    it is made against the dialog that is actually built, not against another
    list.

    The index is what the field OFFERS, which is not quite the generated
    table: the Fractal and Sound pages exist only in spaceout mode, so
    their rows are offered only then. The generator turned the backdrop on to
    see them; this asks the same question this process can answer.
    """
    from spacr.qt import preferences_navigation as navigation
    from spacr.qt.help_index import preference_entries
    from spacr.qt.preferences import PreferencesDialog
    from spacr.qt import theme

    monkeypatch.setattr(theme, "_SPACEOUT", spaceout)

    offered = preference_entries()
    assert len(offered) > 100
    for name in ("PreferencesTabFractal", "PreferencesTabSound"):
        assert any(e.payload["tab"] == name for e in offered) is spaceout

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
    only a row Essentials is HIDING is worth raising the level for, and even
    then the raise is what the FORM shows, not what the store keeps: most
    settings are not essentials, so persisting it would move a module out of
    Essentials for good the first time anybody looked one up.
    """
    from spacr.qt.help_search import reveal_setting
    from spacr.qt.settings_search import ALL, ESSENTIALS, disclosure_for

    assert reveal_setting(window, "mask", "cell_diameter")
    bar = window._screens["mask"]._settings_search
    essential = set(bar._model.essential_keys())
    shown = [k for k in bar.indexed_keys() if k in essential]
    hidden = [k for k in bar.indexed_keys() if k not in essential]
    assert shown and hidden

    bar.set_level(ESSENTIALS)
    assert disclosure_for("mask") == ESSENTIALS
    assert reveal_setting(window, "mask", shown[0])
    qtbot.wait(20)
    assert bar.level() == ESSENTIALS, shown[0]
    assert bar.revealed_key() == shown[0]
    assert disclosure_for("mask") == ESSENTIALS

    assert reveal_setting(window, "mask", hidden[0])
    qtbot.wait(20)
    assert bar.level() == ALL, hidden[0]
    assert hidden[0] in bar.visible_keys()
    assert disclosure_for("mask") == ESSENTIALS, (
        "a lookup raised the level on the form and wrote it to the store")

    bar.set_level(ESSENTIALS)
    qtbot.wait(20)
    assert disclosure_for("mask") == ESSENTIALS
    bar.set_level(ALL)
    qtbot.wait(20)
    assert disclosure_for("mask") == ALL, "clicking the switch still remembers"


def test_the_field_leaves_its_english_where_a_language_change_reads_it(field):
    """A caption put on screen in French must not become the English source.

    The strip is built from ``stack.currentChanged``, which fires AFTER the
    window's one language pass, so the field captions itself and is already
    rendered in the user's language when it first appears.
    :func:`spacr.qt.i18n.retranslate_widget_tree` reads the English back out
    of two Qt properties; a caption that left them empty would have the next
    pass adopt whatever is on screen as the canonical English and translate
    that, which freezes the control in the language it was born in.
    """
    for source, prop in (
        ("Search spaCR…", "_spacr_i18n_placeholder"),
        ("Search spaCR", "_spacr_i18n_accessible_name"),
    ):
        assert field.property(prop) == source, prop
        assert field.property(f"{prop}_last_rendered") is not None, prop
    # The tooltip was removed on 2026-09-19 at the maintainer's request
    # ("remove the tooltip for the search"), so there is no English left for
    # a language pass to read -- and nothing on screen for it to adopt.
    assert not field.toolTip()
    assert field.property("_spacr_i18n_tooltip") is None


def test_a_field_born_in_another_language_comes_back(qtbot, window,
                                                     monkeypatch):
    """The round trip the missing property broke, driven end to end.

    A field built while the catalogs answer in another language renders its
    captions in that language. One pass back to English has to restore them,
    and it can only do that from the source property.
    """
    from spacr.qt import help_search
    from spacr.qt.i18n import retranslate_widget_tree

    monkeypatch.setattr(help_search, "tr",
                        lambda text, **values: "fr: " + str(text))
    born = help_search.HelpSearchField(window, window)
    qtbot.addWidget(born)
    assert born.placeholderText() == "fr: Search spaCR…"
    assert born.property("_spacr_i18n_placeholder") == "Search spaCR…"

    monkeypatch.undo()
    retranslate_widget_tree(born, "en")
    assert born.placeholderText() == "Search spaCR…"
    assert born.accessibleName() == "Search spaCR"


def test_a_result_row_says_where_it_lives_in_the_readers_language(field,
                                                                  monkeypatch):
    """"API reference" is the subtitle of ten thousand rows, so it is chrome.

    The index stays English because the query is matched against it. What is
    drawn is rendered from a template at the moment the row is built, so a
    language change is picked up by the next keystroke.
    """
    from spacr.qt import help_search
    from spacr.qt.help_index import SUBTITLE_API, SUBTITLE_MODULE

    def stub(text, **values):
        rendered = str(text)
        return "[" + (rendered.format(**values) if values else rendered) + "]"

    monkeypatch.setattr(help_search, "tr", stub)
    results = field.type_and_search("mask")
    rows = [field._list.item(i).text() for i in range(field._list.count())]
    assert rows, "no results to draw"
    assert {e.kind for e in results} >= {"module", "api"}, results
    for entry, row in zip(results, rows):
        where = help_search.rendered_subtitle(entry, stub)
        assert row == f"{entry.title}    {where}".rstrip(), row
        if entry.kind == "module":
            assert row.endswith("[" + SUBTITLE_MODULE + "]"), row
        if entry.kind == "api" and not entry.payload.get("reads"):
            assert row.endswith("[" + SUBTITLE_API + "]"), row


def test_the_index_is_built_off_the_gui_thread_and_arrives_on_it(qtbot,
                                                                 monkeypatch):
    """The loader every real session uses, which no test drove.

    Every other test here hands the field a list. This one drives the
    loader: it builds on a worker, hands the result back over a signal, and
    a second ask does not build a second time.
    """
    import threading

    from spacr.qt import help_search

    seen = {}

    def build():
        seen["thread"] = threading.current_thread().name
        seen["builds"] = seen.get("builds", 0) + 1
        return [help_search.HelpEntry(kind="module", title="Probe")]

    monkeypatch.setattr(help_search, "build_index", build)
    loader = help_search._IndexLoader()
    assert loader.started() is False
    with qtbot.waitSignal(loader.ready, timeout=10000) as caught:
        loader.start()
        assert loader.started() is True
        loader.start()
    assert [e.title for e in caught.args[0]] == ["Probe"]
    assert seen["builds"] == 1
    assert seen["thread"] != threading.current_thread().name


def test_an_index_that_cannot_be_built_costs_an_empty_list(qtbot, monkeypatch):
    """A provider taking the window down with it is the one thing it may not do."""
    from spacr.qt import help_search

    def explode():
        raise RuntimeError("no registry today")

    monkeypatch.setattr(help_search, "build_index", explode)
    loader = help_search._IndexLoader()
    with qtbot.waitSignal(loader.ready, timeout=10000) as caught:
        loader.start()
    assert caught.args[0] == []


def test_the_reachability_probe_asks_the_docs_root_off_the_gui_thread(
        qtbot, monkeypatch, _the_documentation_probe_never_reaches_the_network):
    """The daemon thread behind "no dead links", with the socket stubbed.

    It answers ``reachable`` only for a real answer, it asks on a thread of
    its own, and once it has an answer it never asks again -- the probe must
    not become a network call per keystroke.
    """
    import threading
    import urllib.request

    from spacr.qt import help_search
    from spacr.qt.screens.settings_model import DOCS_SITE_BASE

    monkeypatch.setattr(help_search._DocsReach, "start",
                        _the_documentation_probe_never_reaches_the_network)
    asked = {}

    class _Answer:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_urlopen(url, timeout=None):
        asked["url"] = url
        asked["timeout"] = timeout
        asked["thread"] = threading.current_thread().name
        return _Answer()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    reach = help_search._DocsReach()
    with qtbot.waitSignal(reach.settled, timeout=10000) as caught:
        reach.start()
    assert caught.args[0] == "reachable"
    assert reach.state() == "reachable"
    assert asked["url"] == DOCS_SITE_BASE
    assert asked["timeout"] == help_search._DocsReach.TIMEOUT_S
    assert asked["thread"] != threading.current_thread().name

    asked.clear()
    reach.start()
    qtbot.wait(50)
    assert asked == {}, "the probe asked twice"


def test_a_site_that_does_not_answer_is_said_to_be_unreachable(
        qtbot, monkeypatch, _the_documentation_probe_never_reaches_the_network):
    """An exception on the socket is an answer, not a crash on a daemon thread."""
    import urllib.request

    from spacr.qt import help_search

    monkeypatch.setattr(help_search._DocsReach, "start",
                        _the_documentation_probe_never_reaches_the_network)

    def refuse(url, timeout=None):
        raise OSError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    reach = help_search._DocsReach()
    with qtbot.waitSignal(reach.settled, timeout=10000) as caught:
        reach.start()
    assert caught.args[0] == "unreachable"


# ---------------------------------------------------------------------------
# what happens when a result cannot be honoured
#
# Every test above is a result that lands. These are the ones that cannot:
# the module refuses to open, the tab has gone, the browser will not start.
# A search result is believed, so what it SAYS when it did not arrive is as
# much a part of the feature as where it goes when it did.
# ---------------------------------------------------------------------------

def test_a_preference_result_opens_preferences_on_its_own_tab(window, field,
                                                              monkeypatch):
    """Driven from the field, not from the navigation helper underneath it."""
    from spacr.qt.help_search import open_entry

    asked = {}

    def show(tab="", label=""):
        asked["tab"] = tab
        asked["label"] = label
        return True

    monkeypatch.setattr(window, "show_preferences_on", show, raising=False)
    results = field.type_and_search("png resolution")
    row = next(e for e in results if e.kind == "preference")
    said = open_entry(window, row)
    assert asked == {"tab": "PreferencesTabFigures", "label": "PNG resolution"}
    assert said.endswith("Preferences ▸ Figures"), said


def test_a_preference_whose_tab_has_gone_is_not_claimed_to_be_there(window,
                                                                    field,
                                                                    monkeypatch):
    """A row built at start-up can name a page that is no longer built.

    The Fractal page exists only while that backdrop is on. Saying "in
    Preferences ▸ Fractal" over a dialog that opened somewhere else is the
    search reporting something it did not do.
    """
    from spacr.qt.help_search import open_entry

    monkeypatch.setattr(window, "show_preferences_on",
                        lambda tab="", label="": False, raising=False)
    results = field.type_and_search("png resolution")
    row = next(e for e in results if e.kind == "preference")
    assert "not in Preferences" in open_entry(window, row)

    monkeypatch.delattr(window, "show_preferences_on", raising=False)
    monkeypatch.setattr(window, "show_preferences_on", None, raising=False)
    assert open_entry(window, row).startswith("Could not open")


def test_an_opener_that_cannot_do_its_job_says_so(window):
    """A kind with no opener does nothing quietly; one that raises says so."""
    from spacr.qt import help_search
    from spacr.qt.help_index import HelpEntry

    unknown = HelpEntry(kind="_no_such_kind", title="Nowhere")
    assert help_search.open_entry(window, unknown) == ""
    assert "_no_such_kind" not in help_search.opener_kinds()

    def explode(_window, _entry):
        raise RuntimeError("the module would not open")

    help_search.register_opener("_explodes_for_the_test", explode)
    try:
        assert "_explodes_for_the_test" in help_search.opener_kinds()
        said = help_search.open_entry(
            window, HelpEntry(kind="_explodes_for_the_test", title="Boom"))
        assert said == "Could not open Boom."
    finally:
        help_search._OPENERS.pop("_explodes_for_the_test", None)


def test_revealing_a_setting_survives_every_way_it_can_fail(window,
                                                            monkeypatch):
    """Four refusals, none of which may reach the user as a traceback."""
    from spacr.qt.help_search import reveal_setting

    def explode(_key):
        raise RuntimeError("that module will not build")

    monkeypatch.setattr(window, "open_module", explode, raising=False)
    assert reveal_setting(window, "mask", "cell_diameter") is False

    monkeypatch.setattr(window, "open_module", lambda key: "not_a_module",
                        raising=False)
    assert reveal_setting(window, "mask", "cell_diameter") is False

    monkeypatch.undo()
    assert reveal_setting(window, "mask", "cell_diameter") is True
    screen = window._screens["mask"]
    bar = screen._settings_search
    monkeypatch.setattr(screen, "_settings_search", object(), raising=False)
    assert reveal_setting(window, "mask", "cell_diameter") is False

    class _Refuses:
        def reveal(self, _key):
            raise RuntimeError("the form went away")

    monkeypatch.setattr(screen, "_settings_search", _Refuses(), raising=False)
    assert reveal_setting(window, "mask", "cell_diameter") is False
    monkeypatch.setattr(screen, "_settings_search", bar, raising=False)


def test_an_api_result_opens_the_browser_when_the_site_answers(window, field,
                                                               monkeypatch):
    """The reachable half of "no dead links"."""
    import webbrowser

    from spacr.qt import help_search

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    help_search.docs_reach().set_state("reachable")
    results = field.type_and_search("preprocess_generate_masks")
    row = next(e for e in results if e.kind == "api")
    said = help_search.open_entry(window, row)
    assert opened and opened[0].endswith("#" + row.title)
    assert said.startswith("Opened ")

    def refuse(_url):
        raise RuntimeError("no browser on this machine")

    monkeypatch.setattr(webbrowser, "open", refuse)
    assert help_search.open_entry(window, row).startswith("Could not open ")


def test_the_offline_dialog_says_only_what_it_knows(qtbot, window,
                                                    monkeypatch):
    """Three states, three sentences, and the button live for one of them."""
    import webbrowser

    from spacr.qt.help_search import ApiEntryDialog

    dialog = ApiEntryDialog("spacr.core.preprocess_generate_masks",
                            "https://example.invalid/page.html", "", window)
    qtbot.addWidget(dialog)
    dialog._on_settled("unknown")
    assert "Checking whether" in dialog._note.text()
    assert dialog._open.isEnabled() is False
    dialog._on_settled("unreachable")
    assert "not reachable" in dialog._note.text()
    assert dialog._open.isEnabled() is False
    dialog._on_settled("reachable")
    assert "reachable" in dialog._note.text()
    assert dialog._open.isEnabled() is True

    went = []
    monkeypatch.setattr(webbrowser, "open", lambda url: went.append(url))
    dialog._open_the_page()
    assert went == ["https://example.invalid/page.html"]

    def refuse(_url):
        raise RuntimeError("no browser")

    monkeypatch.setattr(webbrowser, "open", refuse)
    dialog._open_the_page()


def test_the_page_of_a_symbol_is_addressed_in_the_readers_language(monkeypatch):
    """The URL half of an API result, including the parts English never sees."""
    from spacr.qt import help_search
    from spacr.qt.screens import settings_model

    symbol = "spacr.core.preprocess_generate_masks"
    assert help_search.api_url(symbol, "en").endswith("#" + symbol)
    korean = help_search.api_url(symbol, "ko")
    assert "?lang=ko" in korean and korean.endswith("#" + symbol)
    module_only = help_search.api_url("spacr.core", "ko")
    assert module_only.endswith("?lang=ko")
    assert help_search.api_url("numpy.zeros", "en").endswith("/index.html")

    def explode(_language=None):
        raise RuntimeError("no language store")

    monkeypatch.setattr(settings_model, "_language_code", explode)
    assert help_search.api_url(symbol).endswith("#" + symbol)


def test_the_local_docstring_is_read_off_the_disk_or_not_at_all(monkeypatch):
    """What the offline dialog shows, and what it shows when there is nothing."""
    from pathlib import Path

    from spacr.qt import help_search

    assert help_search.local_docstring(
        "spacr.qt.help_search.local_docstring").startswith(
            "The docstring of")
    assert help_search.local_docstring("spacr.core.no_such_function") == ""
    assert help_search.local_docstring("numpy.zeros") == ""
    assert help_search._module_file("numpy") is None

    def refuse(_self):
        raise OSError("the file system went away")

    monkeypatch.setattr(Path, "is_file", refuse)
    assert help_search._module_file("spacr.core") is None


def test_a_source_that_cannot_be_parsed_is_no_docstring(monkeypatch, tmp_path):
    """A half-written file on disk is not a reason to fail a lookup."""
    from spacr.qt import help_search

    broken = tmp_path / "broken.py"
    broken.write_text("def half(:\n", encoding="utf-8")
    monkeypatch.setattr(help_search, "_module_file", lambda _d: broken)
    assert help_search.local_docstring("spacr.core.anything") == ""


# ---------------------------------------------------------------------------
# the box itself, at its edges
# ---------------------------------------------------------------------------

def test_the_box_answers_before_the_index_lands_and_after_it_does(qtbot,
                                                                  window,
                                                                  help_index_rows):
    """Typing while the worker is still building says so, then catches up."""
    from spacr.qt.help_search import field_of

    found = field_of(window)
    found._index = None
    found.type_and_search("cell_diameter")
    assert "Building the index" in found.note()
    assert found.index() is None

    found._on_index_ready(help_index_rows)
    assert found.index() is not None
    assert found.results(), "the search did not run again once the index came"

    found.set_index(help_index_rows)
    assert found.results()

    found.setText("")
    found._refresh()
    assert found.results() == []
    assert found.popup().isVisible() is False


def test_a_query_that_matches_nothing_says_so(field):
    """The empty answer is an answer, and it names what was asked."""
    field.type_and_search("zzzzzznotathing")
    assert field.results() == []
    assert "zzzzzznotathing" in field.note()


def test_the_list_is_only_placed_when_there_is_somewhere_to_put_it(qtbot,
                                                                   window,
                                                                   field):
    """A hidden field, or a popup with no window, must not be a crash."""
    field.hide()
    field._place_popup()
    assert field.popup().isVisible() is False
    field.show()
    orphan = field.popup()
    parent = orphan.parentWidget()
    orphan.setParent(None)
    field._place_popup()
    orphan.setParent(parent)


def test_return_on_nothing_opens_nothing(qtbot, window, field):
    """Return with an empty list, and a row carrying no entry, do nothing."""
    from PySide6.QtWidgets import QListWidgetItem

    opened = []
    field.opened.connect(opened.append)
    assert field.type_and_search("zzzzzznotathing") == []
    field._activate_current()
    empty = QListWidgetItem("a row nobody filled in")
    field._list.addItem(empty)
    field._on_activated(empty)
    assert opened == []


def test_a_window_with_no_status_bar_is_still_told_where_it_went(qtbot,
                                                                 window,
                                                                 field,
                                                                 monkeypatch):
    """The status line is where the message goes, not whether it is opened."""
    results = field.type_and_search("cell_diameter")
    row = next(e for e in results if e.kind == "setting")

    def no_bar():
        raise RuntimeError("this window has no status bar")

    monkeypatch.setattr(window, "statusBar", no_bar, raising=False)
    field._list.setCurrentRow(results.index(row))
    with qtbot.waitSignal(field.opened, timeout=2000) as blocker:
        field._activate_current()
    assert blocker.args == [row]


def test_paging_keys_with_an_empty_list_fall_through(qtbot, window, field):
    """PageDown before anything has been typed is an ordinary key press."""
    field.setText("")
    field._list.clear()
    QTest.keyClick(field, Qt.Key_PageDown)
    QTest.keyClick(field, Qt.Key_Down)
    assert field.popup().isVisible() is False


def test_focusing_a_window_that_has_no_field_is_not_a_crash(qtbot):
    """The shortcut is installed window-wide; a window may have no field."""
    from PySide6.QtWidgets import QMainWindow

    from spacr.qt.help_search import field_of, focus_field, install

    bare = QMainWindow()
    qtbot.addWidget(bare)
    assert focus_field(bare) is False
    assert field_of(bare) is None

    made = install(bare)
    assert made is not None, "a plain window has a menu bar to hang it on"
    assert install(bare) is made, "installing twice makes two fields"
    assert focus_field(bare) is True


def test_a_window_with_no_menu_bar_gets_no_field():
    """Installing is best-effort: a window without a menu bar keeps none."""
    from spacr.qt.help_search import install

    class _NoMenuBar:
        def menuBar(self):
            raise RuntimeError("this window has no menu bar")

    class _EmptyMenuBar:
        _help_search = None

        def menuBar(self):
            return None

    assert install(_NoMenuBar()) is None
    assert install(_EmptyMenuBar()) is None


def test_a_caption_that_cannot_be_translated_is_still_a_caption(qtbot,
                                                                monkeypatch):
    """The i18n seam must not be able to leave a control with no text.

    Three ways it can go wrong: a widget with no such setter, a catalog that
    raises, and an object that cannot hold a Qt property. None of them may
    cost the caption itself.
    """
    from PySide6.QtWidgets import QLabel

    from spacr.qt import help_search

    class _NoSetter:
        pass

    help_search._localize(_NoSetter(), "setText", "_spacr_i18n_text", "Hello")

    def explode(_text, **_values):
        raise RuntimeError("no catalog")

    label = QLabel()
    qtbot.addWidget(label)
    monkeypatch.setattr(help_search, "tr", explode)
    help_search._localize(label, "setText", "_spacr_i18n_text", "Hello")
    assert label.text() == "Hello"
    monkeypatch.undo()

    class _Plain:
        def setText(self, text):
            self.said = text

    plain = _Plain()
    help_search._localize(plain, "setText", "_spacr_i18n_text", "Hello")
    assert plain.said == "Hello"


def test_a_docstring_that_cannot_be_read_is_no_docstring(monkeypatch):
    """``ast`` answering with an exception is an empty description, not a stop."""
    import ast

    from spacr.qt import help_search

    def explode(_node):
        raise RuntimeError("that tree is not a tree")

    monkeypatch.setattr(ast, "get_docstring", explode)
    assert help_search.local_docstring("spacr.qt.help_search.api_url") == ""


def test_a_site_that_answers_with_an_error_is_not_reachable(
        qtbot, monkeypatch, _the_documentation_probe_never_reaches_the_network):
    """502 is an answer, and it is not "the page is there"."""
    import urllib.request

    from spacr.qt import help_search

    monkeypatch.setattr(help_search._DocsReach, "start",
                        _the_documentation_probe_never_reaches_the_network)

    class _Broken:
        status = 502

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=None: _Broken())
    reach = help_search._DocsReach()
    with qtbot.waitSignal(reach.settled, timeout=10000) as caught:
        reach.start()
    assert caught.args[0] == "unreachable"


def test_the_probe_is_made_once_and_shared(monkeypatch):
    """One probe per process: the answer is about the machine, not the click."""
    from spacr.qt import help_search

    monkeypatch.setattr(help_search, "_REACH", None)
    first = help_search.docs_reach()
    assert first is help_search.docs_reach()


def test_a_row_with_nothing_to_say_carries_no_tooltip(field):
    """A tooltip is what the row is FOR; an empty one would be a grey box."""
    from spacr.qt.help_index import HelpEntry

    field.set_index([HelpEntry(kind="module", title="Zzzprobe",
                               subtitle="Module", payload={"app": "mask"})])
    field.type_and_search("Zzzprobe")
    assert field._list.count() == 1
    assert field._list.item(0).toolTip() == ""


def test_a_result_that_says_nothing_still_counts_as_opened(qtbot, window,
                                                           field):
    """The signal is the record of the click, whatever the status bar shows."""
    from PySide6.QtWidgets import QListWidgetItem

    from spacr.qt.help_index import HelpEntry
    from spacr.qt.help_search import ENTRY_ROLE

    silent = HelpEntry(kind="_no_such_kind", title="Nowhere")
    item = QListWidgetItem("Nowhere")
    item.setData(ENTRY_ROLE, silent)
    field._list.addItem(item)
    with qtbot.waitSignal(field.opened, timeout=2000) as caught:
        field._on_activated(item)
    assert caught.args[0] is silent
