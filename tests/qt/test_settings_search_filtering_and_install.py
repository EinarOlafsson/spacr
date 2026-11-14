"""The settings search strip: its remembered disclosure level, the index it
builds from a rendered form, how the query, the Modified filter and the
Essentials/All level compose, how sections open and close around a filter,
row visibility on any Qt, and installing the strip into a screen and a
window's stack.

The screens are real widgets carrying the attributes a settings screen
exposes, with real ``QFormLayout`` rows, so row visibility is Qt's own.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import (QFormLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QSplitter, QStackedWidget,
                               QVBoxLayout, QWidget)

from spacr.qt import settings_search as ss
from spacr.qt.settings_search import (ALL, ESSENTIALS, SettingsSearchBar,
                                      _row_is_visible, _set_row_visible,
                                      _StackWatcher, disclosure_for,
                                      forget_disclosure, install,
                                      install_window_hooks,
                                      remember_disclosure)

_PROBE = "b12_search_probe"
_OTHER = "b12_search_other"
_NO_MATCH = ("No setting matches. Clear the search box, or switch to All "
             "settings.")


class _Section(QWidget):
    """A settings section: a form, and optionally the collapse protocol."""

    def __init__(self, parent=None, *, with_form=True):
        super().__init__(parent)
        self._expanded = True
        if with_form:
            self._form = QFormLayout(self)
        else:
            QVBoxLayout(self)

    def is_expanded(self):
        return self._expanded

    def set_expanded(self, on):
        self._expanded = bool(on)

    def add_row(self, label, field):
        self._form.addRow(QLabel(label), field)


class _PlainSection(QWidget):
    """A section with a form but no collapse protocol at all."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._form = QFormLayout(self)

    def add_row(self, label, field):
        self._form.addRow(QLabel(label), field)


class _Model:
    """The search, modified and essential queries the strip asks a model."""

    def __init__(self, widgets, *, modified=(), essential=(), raises=()):
        self._widgets = dict(widgets)
        self._modified = list(modified)
        self._essential = list(essential)
        self._raises = set(raises)

    def keys_matching(self, query):
        if "keys_matching" in self._raises:
            raise RuntimeError("the search index is not built")
        return [k for k in self._widgets if query.lower() in k.lower()]

    def modified_keys(self):
        if "modified_keys" in self._raises:
            raise RuntimeError("no defaults to compare against")
        return list(self._modified)

    def essential_keys(self):
        if "essential_keys" in self._raises:
            raise RuntimeError("this module declares no essentials")
        return list(self._essential)


def _screen_with(sections_and_keys, *, section_type=_Section, **model_kwargs):
    """A screen whose sections hold one QLineEdit row per key."""
    screen = QWidget()
    screen.app_key = _PROBE
    fields, sections = {}, []
    for keys in sections_and_keys:
        section = section_type(screen)
        for key in keys:
            field = QLineEdit(section)
            section.add_row(key, field)
            fields[key] = field
        sections.append(section)
    screen._settings_sections = list(sections)
    screen._settings_model = _Model(fields, **model_kwargs)
    screen._sections = tuple(sections)
    return screen


@pytest.fixture
def form_screen(qapp):
    """Two sections, four settings, one modified, two essential."""
    return _screen_with([("cell_diameter", "cell_cellprob_threshold"),
                         ("plot_dpi", "verbose")],
                        modified=["plot_dpi"],
                        essential=["cell_diameter", "plot_dpi"])


@pytest.fixture
def normal_mode():
    """Easy mode off, so the strip honours the remembered disclosure level."""
    from spacr.qt import preferences

    previous = preferences.get_easy_mode()
    try:
        preferences.set_easy_mode(False)
        yield
    finally:
        preferences.set_easy_mode(previous)


@pytest.fixture(autouse=True)
def _forget_the_probe_choices():
    """The disclosure level is persisted, so clear it around each test."""
    forget_disclosure(_PROBE)
    forget_disclosure(_OTHER)
    yield
    forget_disclosure(_PROBE)
    forget_disclosure(_OTHER)


def _splitter_screen(*, with_left=False):
    """A screen whose settings scroll area sits in a splitter."""
    screen = QWidget()
    screen.app_key = _PROBE
    splitter = QSplitter(screen)
    if with_left:
        splitter.addWidget(QWidget(splitter))
    scroll = QScrollArea(splitter)
    splitter.addWidget(scroll)
    section = _Section(screen)
    field = QLineEdit(section)
    section.add_row("a_setting", field)
    screen._settings_scroll = scroll
    screen._settings_model = _Model({"a_setting": field})
    screen._settings_sections = [section]
    return screen, splitter, scroll


def test_the_disclosure_level_is_remembered_per_module(qapp, subtests):
    """Essentials until told otherwise; any non-All level is Essentials; one
    module or every module can be forgotten."""
    with subtests.test(msg="default is essentials"):
        assert disclosure_for(_PROBE) == ESSENTIALS

    for level, expected in ((ALL, ALL), (ESSENTIALS, ESSENTIALS),
                            ("everything", ESSENTIALS), (ALL, ALL)):
        with subtests.test(msg=f"remember {level!r}"):
            remember_disclosure(_PROBE, level)
            assert disclosure_for(_PROBE) == expected

    with subtests.test(msg="forgetting one module leaves the others"):
        remember_disclosure(_PROBE, ALL)
        remember_disclosure(_OTHER, ALL)
        forget_disclosure(_PROBE)
        assert (disclosure_for(_PROBE), disclosure_for(_OTHER)) == (ESSENTIALS, ALL)

    with subtests.test(msg="forgetting every module at once"):
        remember_disclosure(_PROBE, ALL)
        remember_disclosure(_OTHER, ALL)
        forget_disclosure()
        assert (disclosure_for(_PROBE), disclosure_for(_OTHER)) == (ESSENTIALS, ESSENTIALS)


def test_the_index_is_read_back_from_the_rendered_form(qapp, subtests):
    """Only form rows holding a model widget are indexed: no model, a
    formless section, spanning rows and nested layouts contribute nothing."""
    with subtests.test(msg="two sections, four settings"):
        screen = _screen_with([("cell_diameter", "cell_cellprob_threshold"),
                               ("plot_dpi", "verbose")])
        bar = SettingsSearchBar(screen)
        assert bar.indexed_keys() == ["cell_diameter", "cell_cellprob_threshold",
                                      "plot_dpi", "verbose"]
        first, second = screen._sections
        assert [bar.section_of(k) is s for k, s in (("cell_diameter", first),
                                                    ("verbose", second))] == [True, True]

    with subtests.test(msg="no model and no sections"):
        empty = QWidget()
        empty.app_key = _PROBE
        empty._settings_sections = []
        empty._settings_model = None
        bar = SettingsSearchBar(empty)
        assert (bar.indexed_keys(), bar.visible_keys(), bar.count_text()) == ([], [], "")
        bar.set_query("anything")
        assert bar.count_text() == ""

    with subtests.test(msg="a section with no form"):
        screen = QWidget()
        screen.app_key = _PROBE
        screen._settings_sections = [_Section(screen, with_form=False)]
        screen._settings_model = _Model({})
        bar = SettingsSearchBar(screen)
        assert (bar.indexed_keys(), bar.count_text()) == ([], "")

    with subtests.test(msg="spanning rows and nested layouts are skipped"):
        screen = QWidget()
        screen.app_key = _PROBE
        section = _Section(screen)
        section._form.addRow(QLabel("a heading"))
        section._form.addRow(QLabel("a pair"), QVBoxLayout())
        stray = QLineEdit(section)
        section.add_row("not_a_model_widget", stray)
        field = QLineEdit(section)
        section.add_row("real_setting", field)
        screen._settings_sections = [section]
        screen._settings_model = _Model({"real_setting": field})
        bar = SettingsSearchBar(screen)
        assert bar.indexed_keys() == ["real_setting"]

    with subtests.test(msg="a field found through its child widget"):
        screen = QWidget()
        screen.app_key = _PROBE
        section = _Section(screen)
        holder = QWidget(section)
        inner = QLineEdit(holder)
        section.add_row("wrapped", holder)
        screen._settings_sections = [section]
        screen._settings_model = _Model({"wrapped": inner})
        bar = SettingsSearchBar(screen)
        assert bar.indexed_keys() == ["wrapped"]
        assert bar.section_of("wrapped") is section


def test_query_modified_and_level_compose_into_one_view(qapp, form_screen,
                                                        subtests):
    """Each (level, Modified, query) combination shows exactly the rows the
    three filters agree on and says so in the count line."""
    bar = SettingsSearchBar(form_screen)
    with subtests.test(msg="starts on essentials, unfiltered"):
        assert (bar.level(), bar.modified_only(), bar.query()) == (ESSENTIALS, False, "")

    # (level, modified only, query, visible keys, count line)
    cases = [
        (ESSENTIALS, False, "", ["cell_diameter", "plot_dpi"],
         "Showing 2 of 4 settings — 2 more under All settings."),
        (ESSENTIALS, True, "", ["plot_dpi"],
         "Showing 1 of 4 settings — 2 more under All settings — modified only."),
        (ESSENTIALS, False, "zzzz-no-such-setting", [], _NO_MATCH),
        (ALL, False, "", ["cell_cellprob_threshold", "cell_diameter", "plot_dpi",
                          "verbose"], "4 settings."),
        (ALL, False, "cell", ["cell_cellprob_threshold", "cell_diameter"],
         "Showing 2 of 4 settings."),
        (ALL, False, "  CELL ", ["cell_cellprob_threshold", "cell_diameter"],
         "Showing 2 of 4 settings."),
        (ALL, True, "", ["plot_dpi"], "Showing 1 of 4 settings — modified only."),
        (ALL, True, "cell", [], _NO_MATCH),
        (ALL, False, "verbose", ["verbose"], "Showing 1 of 4 settings."),
    ]
    for level, modified, query, visible, count in cases:
        with subtests.test(msg=f"{level}/{modified}/{query!r}"):
            bar.set_level(level)
            bar.set_modified_only(modified)
            bar.set_query(query)
            assert (bar.level(), bar.modified_only(), bar.query()) == (level, modified, query)
            assert sorted(bar.visible_keys()) == visible
            assert bar.count_text() == count
            assert sorted(bar.keys_it_hides()) == sorted(
                set(bar.indexed_keys()) - set(visible))

    with subtests.test(msg="every setting essential"):
        screen = _screen_with([("a_setting", "b_setting")],
                              essential=["a_setting", "b_setting"])
        whole = SettingsSearchBar(screen)
        assert whole.level() == ESSENTIALS
        assert sorted(whole.visible_keys()) == ["a_setting", "b_setting"]
        assert whole.count_text() == "Showing all 2 settings."
        whole.set_level(ALL)
        assert whole.count_text() == "2 settings."

    with subtests.test(msg="a model whose every query raises shows everything"):
        screen = _screen_with([("a_setting", "b_setting")],
                              raises={"keys_matching", "modified_keys",
                                      "essential_keys"})
        broken = SettingsSearchBar(screen)
        for action in (lambda: None, lambda: broken.set_query("a_"),
                       lambda: broken.set_modified_only(True)):
            action()
            assert sorted(broken.visible_keys()) == ["a_setting", "b_setting"]
        assert broken.count_text() == "2 settings."


def test_sections_open_while_filtered_and_return_to_the_users_form(
        qapp, form_screen, normal_mode, subtests):
    """A narrowing filter opens what it kept and hides emptied sections;
    releasing it restores what the user had and hands maturity back."""
    calls = []
    form_screen.refresh_maturity_visibility = lambda: calls.append("refresh")
    remember_disclosure(_PROBE, ALL)
    bar = SettingsSearchBar(form_screen)
    first, second = form_screen._sections

    with subtests.test(msg="an unfiltered start hands maturity to the screen"):
        assert bar.level() == ALL
        assert calls == ["refresh"]

    with subtests.test(msg="narrowing opens kept sections and hides empty ones"):
        first.set_expanded(False)
        second.set_expanded(False)
        bar.set_query("cell")
        assert (first.is_expanded(), first.isHidden()) == (True, False)
        assert (second.is_expanded(), second.isHidden()) == (False, True)
        assert calls == ["refresh"]

    with subtests.test(msg="releasing restores the collapsed form"):
        bar.set_query("")
        assert (first.is_expanded(), second.is_expanded()) == (False, False)
        assert (first.isHidden(), second.isHidden()) == (False, False)
        assert calls == ["refresh", "refresh"]

    with subtests.test(msg="a maturity hook that raises does not break it"):
        def _explode():
            calls.append("exploded")
            raise RuntimeError("the screen is being torn down")

        form_screen.refresh_maturity_visibility = _explode
        bar.set_query("plot")
        bar.set_query("")
        assert calls[-1] == "exploded"
        assert sorted(bar.visible_keys()) == sorted(bar.indexed_keys())

    with subtests.test(msg="an empty section is left alone while unfiltered"):
        empty = _Section(form_screen)
        empty.setVisible(True)
        form_screen._settings_sections.append(empty)
        fresh = SettingsSearchBar(form_screen)
        assert (fresh.level(), fresh.query()) == (ALL, "")
        assert empty.isHidden() is False
        fresh.set_query("cell")
        assert empty.isHidden() is True
        fresh.set_query("")
        assert empty.isHidden() is False

    with subtests.test(msg="a section without the collapse protocol"):
        screen = _screen_with([("a_setting", "b_setting")],
                              section_type=_PlainSection)
        plain = SettingsSearchBar(screen)
        plain.set_level(ALL)
        section = screen._sections[0]
        plain.set_query("a_")
        assert (plain.visible_keys(), section.isHidden()) == (["a_setting"], False)
        plain.set_query("nothing matches this")
        assert (plain.visible_keys(), section.isHidden()) == ([], True)
        plain.set_query("")
        assert (sorted(plain.visible_keys()), section.isHidden()) == (
            ["a_setting", "b_setting"], False)


def test_row_visibility_works_whatever_qt_offers(qapp, monkeypatch, subtests):
    """A form row is hidden through the form, a private form is found by
    search, and with no form or no setRowVisible the field itself hides."""
    def _no_form():
        section = QWidget()
        QVBoxLayout(section)
        return section, QLineEdit(section), None

    def _private_form():
        section = QWidget()
        form = QFormLayout(section)
        field = QLineEdit(section)
        form.addRow(QLabel("a_setting"), field)
        return section, field, form

    def _attribute_form():
        section = _Section()
        field = QLineEdit(section)
        section.add_row("a_setting", field)
        return section, field, section._form

    for case, build in (("no form", _no_form), ("private form", _private_form),
                        ("form attribute", _attribute_form)):
        with subtests.test(msg=case):
            section, field, form = build()
            section.show()
            assert ss._form_of(section) is form
            for visible in (False, True, False):
                _set_row_visible(section, field, visible)
                assert _row_is_visible(section, field) is visible
                assert field.isVisible() is visible
                if form is not None:
                    assert form.isRowVisible(field) is visible
            section.close()

    with subtests.test(msg="a Qt without row visibility"):
        section, field, form = _attribute_form()
        section.show()

        def _gone(*_args, **_kwargs):
            raise AttributeError("Qt < 6.4 has no setRowVisible")

        monkeypatch.setattr(type(form), "setRowVisible", _gone, raising=False)
        monkeypatch.setattr(type(form), "isRowVisible", _gone, raising=False)
        for visible in (False, True):
            _set_row_visible(section, field, visible)
            assert field.isVisible() is visible
            assert _row_is_visible(section, field) is visible
        section.close()


def test_install_puts_the_strip_in_the_splitter_or_declines(qapp, monkeypatch,
                                                           subtests):
    """Screens without a full settings form or without a splitter get None;
    a real one gets one strip in the splitter slot, idempotently, even when
    translation fails; a failing build leaves no strip."""
    with subtests.test(msg="not a settings screen"):
        assert install(QWidget()) is None

    with subtests.test(msg="a screen missing its model and sections"):
        partial = QWidget()
        partial._settings_scroll = QScrollArea(partial)
        partial._settings_model = None
        partial._settings_sections = []
        assert install(partial) is None
        assert getattr(partial, "_settings_search", None) is None

    with subtests.test(msg="a scroll area outside a splitter"):
        screen = QWidget()
        host = QWidget(screen)
        QVBoxLayout(host)
        screen._settings_scroll = QScrollArea(host)
        screen._settings_model = _Model({})
        screen._settings_sections = [_Section(screen)]
        assert install(screen) is None
        assert getattr(screen, "_settings_search", None) is None

    with subtests.test(msg="the strip takes the splitter slot"):
        screen, splitter, scroll = _splitter_screen(with_left=True)
        button = QPushButton("Import settings")
        screen._settings_file_buttons = [button]
        bar = install(screen)
        assert type(bar) is SettingsSearchBar
        assert screen._settings_search is bar
        assert splitter.count() == 2
        container = splitter.widget(1)
        assert container.objectName() == ss.PANE_NAME
        assert scroll.parentWidget() is container
        assert bar.parentWidget() is container
        assert button.parent() is bar
        assert bar.indexed_keys() == ["a_setting"]
        assert install(screen) is bar

    with subtests.test(msg="a strip that cannot be translated is installed"):
        import spacr.qt.i18n as i18n

        def _no_catalog(_widget):
            raise RuntimeError("no catalog loaded")

        monkeypatch.setattr(i18n, "retranslate_widget_tree", _no_catalog)
        screen, splitter, _scroll = _splitter_screen()
        bar = install(screen)
        assert type(bar) is SettingsSearchBar
        assert screen._settings_search is bar
        assert splitter.widget(0).objectName() == ss.PANE_NAME
        monkeypatch.undo()

    with subtests.test(msg="a build that throws leaves no strip"):
        def _explode(*_args, **_kwargs):
            raise RuntimeError("the strip could not be built")

        monkeypatch.setattr(ss, "SettingsSearchBar", _explode)
        screen, splitter, scroll = _splitter_screen()
        assert install(screen) is None
        assert getattr(screen, "_settings_search", None) is None
        assert (splitter.count(), splitter.widget(0)) == (1, scroll)
        monkeypatch.undo()


def test_install_window_hooks_follows_a_real_stack_once(qapp, qtbot, subtests):
    """No stack or an unconnectable signal yields no watcher; a real stack is
    followed by one watcher that the window keeps and hands back again."""
    with subtests.test(msg="no stack"):
        window = QWidget()
        assert install_window_hooks(window) is None
        assert getattr(window, "_settings_search_watcher", None) is None

    with subtests.test(msg="a stack whose signal cannot connect"):
        class _Signalless:
            currentChanged = None

        window = QWidget()
        window._stack = _Signalless()
        assert install_window_hooks(window) is None
        assert getattr(window, "_settings_search_watcher", None) is None

    with subtests.test(msg="a real stack"):
        window = QWidget()
        stack = QStackedWidget(window)
        stack.addWidget(QWidget())
        window._stack = stack
        watcher = install_window_hooks(window)
        assert type(watcher) is _StackWatcher
        assert window._settings_search_watcher is watcher
        assert watcher.parent() is window
        assert install_window_hooks(window) is watcher
        assert watcher.install_current() is None

    with subtests.test(msg="switching to a settings screen installs the strip"):
        screen, _splitter, _scroll = _splitter_screen()
        stack.addWidget(screen)
        stack.setCurrentWidget(screen)
        assert type(screen._settings_search) is SettingsSearchBar
        assert watcher.install_current() is screen._settings_search
