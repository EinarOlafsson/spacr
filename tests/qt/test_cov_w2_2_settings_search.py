"""Filtering a settings form, and the four ways the form can refuse.

The strip's promise is that the query, the Modified filter and the
disclosure level are one path, so they can never disagree about what is on
screen. Everything below that promise is degradation: a model whose search
raises, a section with no form in it, a Qt too old for ``setRowVisible``, a
screen that is not a settings screen at all. Each of those has to leave a
usable panel rather than a blank one -- "the filter excluded that setting"
and "the panel will not draw" look identical to a user for about a second,
and then only one of them recovers.

The screens here are real widgets carrying the same four attributes a real
settings screen exposes, with real ``QFormLayout`` rows in them, so the
row-visibility calls are the Qt ones and the index is really built from a
rendered form.
"""

import pytest
from PySide6.QtWidgets import (QFormLayout, QLabel, QLineEdit, QScrollArea,
                               QSplitter, QVBoxLayout, QWidget)

from spacr.qt import settings_search as ss
from spacr.qt.settings_search import (ALL, ESSENTIALS, SettingsSearchBar,
                                      _row_is_visible, _set_row_visible,
                                      _StackWatcher, disclosure_for,
                                      forget_disclosure, install,
                                      install_window_hooks,
                                      remember_disclosure)


# ---------------------------------------------------------------------------
# stand-ins with the real screen's shape
# ---------------------------------------------------------------------------

class _Section(QWidget):
    """A settings section: a form, and the collapse protocol the strip uses."""

    def __init__(self, parent=None, *, with_form=True, collapsible=True):
        super().__init__(parent)
        self._expanded = True
        self._collapsible = collapsible
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
    """The four methods the strip asks a settings model for."""

    def __init__(self, widgets, *, matching=None, modified=(), essential=(),
                 raises=()):
        self._widgets = dict(widgets)
        self._matching = matching
        self._modified = list(modified)
        self._essential = list(essential)
        self._raises = set(raises)

    def keys_matching(self, query):
        if "keys_matching" in self._raises:
            raise RuntimeError("the search index is not built")
        if self._matching is not None:
            return list(self._matching)
        return [k for k in self._widgets if query.lower() in k.lower()]

    def modified_keys(self):
        if "modified_keys" in self._raises:
            raise RuntimeError("no defaults to compare against")
        return list(self._modified)

    def essential_keys(self):
        if "essential_keys" in self._raises:
            raise RuntimeError("this module declares no essentials")
        return list(self._essential)


def _screen(qapp, *, sections=None, model=None, app_key="w2_2_probe"):
    screen = QWidget()
    screen.app_key = app_key
    screen._settings_sections = list(sections or [])
    screen._settings_model = model
    return screen


@pytest.fixture
def form_screen(qapp):
    """A screen with two sections, four settings, and a model over them."""
    screen = QWidget()
    screen.app_key = "w2_2_probe"
    first = _Section(screen)
    second = _Section(screen)
    fields = {}
    for section, keys in ((first, ("cell_diameter", "cell_CP_prob")),
                          (second, ("plot_dpi", "verbose"))):
        for key in keys:
            field = QLineEdit(section)
            section.add_row(key, field)
            fields[key] = field
    screen._settings_sections = [first, second]
    screen._settings_model = _Model(fields, modified=["plot_dpi"],
                                    essential=["cell_diameter", "plot_dpi"])
    screen._sections = (first, second)
    return screen


@pytest.fixture(autouse=True)
def forget_this_modules_choice():
    """The disclosure level is persisted, so clear it around each test."""
    forget_disclosure("w2_2_probe")
    yield
    forget_disclosure("w2_2_probe")


# ---------------------------------------------------------------------------
# the remembered disclosure level
# ---------------------------------------------------------------------------

def test_a_module_starts_on_essentials_and_remembers_being_told_otherwise(
        qapp):
    """A user meets a few settings at a time until they say otherwise."""
    assert disclosure_for("w2_2_probe") == ESSENTIALS

    remember_disclosure("w2_2_probe", ALL)
    assert disclosure_for("w2_2_probe") == ALL

    forget_disclosure("w2_2_probe")
    assert disclosure_for("w2_2_probe") == ESSENTIALS


def test_forgetting_one_module_leaves_the_others_alone(qapp):
    """Per-module, because a user's answer for Mask is not their answer for
    Measure."""
    remember_disclosure("w2_2_probe", ALL)
    remember_disclosure("w2_2_other", ALL)
    try:
        forget_disclosure("w2_2_probe")
        assert disclosure_for("w2_2_probe") == ESSENTIALS
        assert disclosure_for("w2_2_other") == ALL
    finally:
        forget_disclosure("w2_2_other")


# ---------------------------------------------------------------------------
# filtering a real form
# ---------------------------------------------------------------------------

def test_a_strip_with_no_form_behind_it_says_nothing(qapp):
    """No model and no sections is an empty count line, not a crash.

    The strip is installed from a stack signal and can meet a screen that
    has no settings at all.
    """
    bar = SettingsSearchBar(_screen(qapp))
    assert bar.indexed_keys() == []
    assert bar.count_text() == ""
    bar.set_query("anything")
    assert bar.count_text() == ""


def test_the_index_is_built_from_the_rendered_form(qapp, form_screen):
    """Which key went where is the screen's decision, read back, not redone."""
    bar = SettingsSearchBar(form_screen)
    assert sorted(bar.indexed_keys()) == ["cell_CP_prob", "cell_diameter",
                                          "plot_dpi", "verbose"]


def test_a_query_shows_the_rows_that_match_and_hides_the_rest(qapp,
                                                              form_screen):
    """Row visibility, not widget visibility: a collapsed heading is not a
    filter."""
    bar = SettingsSearchBar(form_screen)
    bar.set_level(ALL)

    bar.set_query("cell")
    assert sorted(bar.visible_keys()) == ["cell_CP_prob", "cell_diameter"]

    bar.set_query("")
    assert sorted(bar.visible_keys()) == sorted(bar.indexed_keys())


def test_the_modified_filter_narrows_to_what_no_longer_holds_its_default(
        qapp, form_screen):
    """The three controls are one path, so they compose rather than fight."""
    bar = SettingsSearchBar(form_screen)
    bar.set_level(ALL)
    assert bar.modified_only() is False

    bar.set_modified_only(True)
    assert bar.modified_only() is True
    assert bar.visible_keys() == ["plot_dpi"]

    bar.set_query("cell")
    assert bar.visible_keys() == []

    bar.set_modified_only(False)
    assert sorted(bar.visible_keys()) == ["cell_CP_prob", "cell_diameter"]


def test_essentials_hides_the_rest_until_the_user_asks_for_all(qapp,
                                                               form_screen):
    """The default level shows the module's declared essentials only."""
    bar = SettingsSearchBar(form_screen)
    assert bar.level() == ESSENTIALS
    assert sorted(bar.visible_keys()) == ["cell_diameter", "plot_dpi"]

    bar.set_level(ALL)
    assert bar.level() == ALL
    assert sorted(bar.visible_keys()) == sorted(bar.indexed_keys())


def test_a_model_whose_search_raises_leaves_the_panel_usable(qapp):
    """A broken index costs the filter, never the form."""
    screen = QWidget()
    screen.app_key = "w2_2_probe"
    section = _Section(screen)
    fields = {}
    for key in ("a_setting", "b_setting"):
        field = QLineEdit(section)
        section.add_row(key, field)
        fields[key] = field
    screen._settings_sections = [section]
    screen._settings_model = _Model(
        fields, raises={"keys_matching", "modified_keys", "essential_keys"})

    bar = SettingsSearchBar(screen)          # essential_keys raises at build
    assert sorted(bar.visible_keys()) == ["a_setting", "b_setting"]

    bar.set_query("a_")                      # keys_matching raises
    assert sorted(bar.visible_keys()) == ["a_setting", "b_setting"]

    bar.set_modified_only(True)              # modified_keys raises
    assert sorted(bar.visible_keys()) == ["a_setting", "b_setting"]


def test_a_narrowing_filter_opens_what_it_kept_and_puts_it_back_after(
        qapp, form_screen):
    """A filter that leaves every section collapsed is worse than no filter.

    And releasing it restores the form the user had, rather than one the
    filter invented.
    """
    bar = SettingsSearchBar(form_screen)
    bar.set_level(ALL)
    first, second = form_screen._sections
    first.set_expanded(False)
    second.set_expanded(False)

    bar.set_query("cell")
    assert first.is_expanded() is True
    assert second.isVisible() is False

    bar.set_query("")
    assert first.is_expanded() is False, "the filter kept the form splayed"
    assert second.is_expanded() is False


def test_a_section_that_cannot_collapse_is_still_shown_and_hidden(qapp):
    """Not every section carries the collapse protocol; visibility still works."""
    screen = QWidget()
    screen.app_key = "w2_2_probe"
    section = _PlainSection(screen)
    fields = {}
    for key in ("a_setting", "b_setting"):
        field = QLineEdit(section)
        section.add_row(key, field)
        fields[key] = field
    screen._settings_sections = [section]
    screen._settings_model = _Model(fields)

    bar = SettingsSearchBar(screen)
    bar.set_level(ALL)
    bar.set_query("a_")
    assert bar.visible_keys() == ["a_setting"]
    assert section.isHidden() is False

    bar.set_query("nothing matches this")
    assert bar.visible_keys() == []


def test_releasing_a_filter_hands_maturity_back_to_the_screen(qapp,
                                                              form_screen):
    """Only the screen knows why a section was hidden in the first place."""
    called = []
    form_screen.refresh_maturity_visibility = lambda: called.append(1)

    bar = SettingsSearchBar(form_screen)
    bar.set_level(ALL)
    bar.set_query("cell")
    bar.set_query("")
    assert called, "the screen was never asked to restore maturity visibility"


def test_a_screen_that_cannot_restore_maturity_does_not_break_the_filter(
        qapp, form_screen):
    """An exception in the screen's own hook is logged, not raised."""
    def explode():
        raise RuntimeError("the screen is being torn down")

    form_screen.refresh_maturity_visibility = explode

    bar = SettingsSearchBar(form_screen)
    bar.set_level(ALL)
    bar.set_query("cell")
    bar.set_query("")                        # must not raise
    assert sorted(bar.visible_keys()) == sorted(bar.indexed_keys())


def test_a_section_with_no_form_contributes_nothing_to_the_index(qapp):
    """A section that is not a settings form is skipped rather than guessed at."""
    screen = QWidget()
    screen.app_key = "w2_2_probe"
    formless = _Section(screen, with_form=False)
    screen._settings_sections = [formless]
    screen._settings_model = _Model({})

    bar = SettingsSearchBar(screen)
    assert bar.indexed_keys() == []


def test_a_form_row_whose_field_is_a_layout_is_skipped(qapp):
    """A row holding a nested layout has no field widget, so it is no setting."""
    screen = QWidget()
    screen.app_key = "w2_2_probe"
    section = _Section(screen)
    section._form.addRow(QLabel("a heading"))          # spanning: no field
    section._form.addRow(QLabel("a pair"), QVBoxLayout())
    field = QLineEdit(section)
    section.add_row("real_setting", field)
    screen._settings_sections = [section]
    screen._settings_model = _Model({"real_setting": field})

    bar = SettingsSearchBar(screen)
    assert bar.indexed_keys() == ["real_setting"]


def test_a_trailing_control_joins_the_strip_rather_than_a_second_one(
        qapp, form_screen):
    """The seam other modules use to put a settings-scoped control here."""
    bar = SettingsSearchBar(form_screen)
    extra = QLabel("Live preview")
    bar.add_trailing_widget(extra)
    assert extra.parent() is bar


# ---------------------------------------------------------------------------
# row visibility, whatever Qt offers
# ---------------------------------------------------------------------------

def test_a_field_outside_a_form_is_hidden_directly(qapp):
    """With no form to ask, the field itself is the row."""
    section = QWidget()
    QVBoxLayout(section)
    field = QLineEdit(section)
    section.show()

    _set_row_visible(section, field, False)
    assert field.isVisible() is False
    assert _row_is_visible(section, field) is False

    _set_row_visible(section, field, True)
    assert _row_is_visible(section, field) is True


def test_a_qt_without_row_visibility_hides_the_field_instead(qapp,
                                                             monkeypatch):
    """A stranded label is a far smaller problem than a panel that will not
    draw."""
    section = _Section()
    field = QLineEdit(section)
    section.add_row("a_setting", field)
    section.show()

    def gone(*_args, **_kwargs):
        raise AttributeError("Qt < 6.4 has no setRowVisible")

    monkeypatch.setattr(type(section._form), "setRowVisible", gone,
                        raising=False)
    monkeypatch.setattr(type(section._form), "isRowVisible", gone,
                        raising=False)

    _set_row_visible(section, field, False)
    assert field.isVisible() is False
    assert _row_is_visible(section, field) is False


def test_a_form_found_by_search_is_used_when_there_is_no_form_attribute(qapp):
    """A section that keeps its form privately is still reachable."""
    section = QWidget()
    form = QFormLayout(section)
    field = QLineEdit(section)
    form.addRow(QLabel("a_setting"), field)
    section.show()

    assert ss._form_of(section) is form
    _set_row_visible(section, field, False)
    assert _row_is_visible(section, field) is False


# ---------------------------------------------------------------------------
# installing the strip
# ---------------------------------------------------------------------------

def test_a_screen_that_is_not_a_settings_screen_gets_no_strip(qapp):
    """Missing any of the four attributes means there is nothing to filter."""
    assert install(QWidget()) is None

    partial = QWidget()
    partial._settings_scroll = QScrollArea(partial)
    partial._settings_model = None
    partial._settings_sections = []
    assert install(partial) is None


def test_a_scroll_area_that_is_not_in_a_splitter_gets_no_strip(qapp):
    """The strip takes the splitter's slot; without one there is nowhere."""
    screen = QWidget()
    host = QWidget(screen)
    QVBoxLayout(host)
    scroll = QScrollArea(host)
    screen._settings_scroll = scroll
    screen._settings_model = _Model({})
    screen._settings_sections = [_Section(screen)]
    assert install(screen) is None


def test_the_strip_takes_the_splitter_slot_and_keeps_the_pane_sizes(qapp):
    """The scroll area moves into a container that takes its place."""
    screen = QWidget()
    screen.app_key = "w2_2_probe"
    splitter = QSplitter(screen)
    left = QWidget(splitter)
    scroll = QScrollArea(splitter)
    splitter.addWidget(left)
    splitter.addWidget(scroll)
    section = _Section(screen)
    field = QLineEdit(section)
    section.add_row("a_setting", field)
    screen._settings_scroll = scroll
    screen._settings_model = _Model({"a_setting": field})
    screen._settings_sections = [section]

    bar = install(screen)
    assert bar is not None
    assert screen._settings_search is bar
    assert splitter.count() == 2
    container = splitter.widget(1)
    assert container.objectName() == ss.PANE_NAME
    assert scroll.parentWidget() is container
    # asked twice, the same strip comes back
    assert install(screen) is bar


def test_an_installation_that_throws_leaves_the_screen_without_a_strip(
        qapp, monkeypatch):
    """A failure here costs the filter, not the settings panel."""
    screen = QWidget()
    screen.app_key = "w2_2_probe"
    splitter = QSplitter(screen)
    scroll = QScrollArea(splitter)
    splitter.addWidget(scroll)
    screen._settings_scroll = scroll
    screen._settings_model = _Model({})
    screen._settings_sections = [_Section(screen)]

    def explode(*_args, **_kwargs):
        raise RuntimeError("the strip could not be built")

    monkeypatch.setattr(ss, "SettingsSearchBar", explode)
    assert install(screen) is None
    assert getattr(screen, "_settings_search", None) is None


def test_a_strip_that_cannot_be_translated_is_still_installed(qapp,
                                                              monkeypatch):
    """The language pass is idempotent housekeeping, not a precondition."""
    import spacr.qt.i18n as i18n

    screen = QWidget()
    screen.app_key = "w2_2_probe"
    splitter = QSplitter(screen)
    scroll = QScrollArea(splitter)
    splitter.addWidget(scroll)
    section = _Section(screen)
    field = QLineEdit(section)
    section.add_row("a_setting", field)
    screen._settings_scroll = scroll
    screen._settings_model = _Model({"a_setting": field})
    screen._settings_sections = [section]

    def explode(_widget):
        raise RuntimeError("no catalog loaded")

    monkeypatch.setattr(i18n, "retranslate_widget_tree", explode)
    assert install(screen) is not None


# ---------------------------------------------------------------------------
# following the stack
# ---------------------------------------------------------------------------

def test_a_window_with_no_stack_is_not_followed(qapp):
    """Nothing to follow returns None rather than raising."""
    assert install_window_hooks(QWidget()) is None


def test_a_watcher_ignores_a_stack_it_cannot_read(qapp):
    """A window being torn down is not a screen to install into."""
    class _Hostile(QWidget):
        @property
        def _stack(self):
            raise RuntimeError("the window is going away")

    watcher = _StackWatcher(QWidget())
    watcher._window = _Hostile()
    assert watcher.install_current() is None


def test_an_empty_stack_installs_nothing(qapp):
    """A stack showing no widget has no settings form on it."""
    class _Stack(QWidget):
        def currentWidget(self):
            return None

    window = QWidget()
    window._stack = _Stack()
    watcher = _StackWatcher(window)
    assert watcher.install_current() is None
    watcher.on_current_changed(0)


def test_a_stack_whose_signal_cannot_be_connected_is_given_up_on(qapp):
    """No `currentChanged` leaves no half-installed watcher on the window."""
    class _Signalless:
        currentChanged = None

    window = QWidget()
    window._stack = _Signalless()
    assert install_window_hooks(window) is None
    assert getattr(window, "_settings_search_watcher", None) is None


def test_a_real_stack_is_followed_once(qapp, qtbot):
    """The hook connects, installs into what is already there, and is not
    doubled."""
    from PySide6.QtWidgets import QStackedWidget

    window = QWidget()
    stack = QStackedWidget(window)
    stack.addWidget(QWidget())
    window._stack = stack

    watcher = install_window_hooks(window)
    assert watcher is not None
    assert install_window_hooks(window) is watcher


def test_forgetting_every_module_at_once_is_possible(qapp):
    """One call for the whole preference, for a reset or a test teardown."""
    remember_disclosure("w2_2_probe", ALL)
    remember_disclosure("w2_2_other", ALL)
    forget_disclosure()
    assert disclosure_for("w2_2_probe") == ESSENTIALS
    assert disclosure_for("w2_2_other") == ESSENTIALS


def test_the_search_box_reports_what_was_typed_into_it(qapp, form_screen):
    """`query` is how a caller reads the box rather than reaching for the
    widget."""
    bar = SettingsSearchBar(form_screen)
    assert bar.query() == ""
    bar.set_query("  cell ")
    assert bar.query() == "  cell "


def test_an_empty_section_is_left_alone_while_nothing_is_filtered(qapp,
                                                                  form_screen):
    """A section with no rows was already hidden for its own reasons.

    Maturity decides that, and a filter that is not filtering has no business
    overruling it.
    """
    empty = _Section(form_screen)
    empty.setVisible(True)
    form_screen._settings_sections.append(empty)
    # start at All, so the very first pass is not a narrowing one
    remember_disclosure("w2_2_probe", ALL)

    bar = SettingsSearchBar(form_screen)
    assert bar.level() == ALL
    assert bar.query() == ""
    assert empty.isHidden() is False, "an unfiltered view hid an empty section"


def test_a_module_whose_settings_are_all_essential_says_so(qapp):
    """Showing everything at the Essentials level is a different sentence."""
    screen = QWidget()
    screen.app_key = "w2_2_probe"
    section = _Section(screen)
    fields = {}
    for key in ("a_setting", "b_setting"):
        field = QLineEdit(section)
        section.add_row(key, field)
        fields[key] = field
    screen._settings_sections = [section]
    screen._settings_model = _Model(fields, essential=list(fields))

    bar = SettingsSearchBar(screen)
    assert bar.level() == ESSENTIALS
    assert sorted(bar.visible_keys()) == ["a_setting", "b_setting"]
    assert "2" in bar.count_text()

    bar.set_level(ALL)
    assert "2 settings." in bar.count_text()


def test_a_query_that_matches_nothing_says_how_to_get_back(qapp, form_screen):
    """An empty form with no explanation reads as a broken panel."""
    bar = SettingsSearchBar(form_screen)
    bar.set_query("zzzz-no-such-setting")
    assert bar.visible_keys() == []
    assert "No setting matches" in bar.count_text()
    assert "All settings" in bar.count_text()


def test_a_partial_view_counts_what_it_hid(qapp, form_screen):
    """The line under the controls says how many of how many are showing."""
    bar = SettingsSearchBar(form_screen)
    bar.set_level(ALL)
    bar.set_modified_only(True)
    text = bar.count_text()
    assert "1" in text and "4" in text
    assert "modified only" in text


# ---------------------------------------------------------------------------
# the strip's own styling
# ---------------------------------------------------------------------------

def test_the_strip_paints_nothing_behind_itself(qapp):
    """It is type and controls sitting on the page, not a card.

    A painted strip composites the pane's own surface onto whatever is behind
    it, which is the "the container is not subject to the opacity setting"
    report.
    """
    from spacr.qt.theme import palette_for

    palette = dict(palette_for("dark"), theme="dark", font_scale=1.0)
    qss = ss._bar_qss(palette, 1.0)
    assert f"QWidget#{ss.PANE_NAME}" in qss
    assert f"QWidget#{ss.BAR_NAME}" in qss
    assert "background: transparent" in qss
    assert f"QLineEdit#{ss.INPUT_NAME}" in qss
