"""The module screen's refusals, its lazy containers and its late rows.

Nothing here draws a finished panel. Every path below is one the screen takes
when a collaborator is absent, half-built or already destroyed: a plugin table
that will not enumerate at import, a backdrop that refuses to stop for a run, a
settings model that cannot say which objects the run has, a heading whose rows
were deliberately left uncaptioned until something asked for them, and the
handful of results-view helpers that are asked to show a page that is not
there.

The shared rule is that the screen does the part it can do, says what it could
not, and never lets the failure reach the GUI thread -- and the shared risk is
that a silent refusal looks exactly like success. So each test drives both
arms: the one where the collaborator answers and the one where it does not.
"""
from __future__ import annotations

import gc
import logging
import os
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (                                  # noqa: E402
    QLabel, QLineEdit, QWidget,
)

from spacr.qt.screens import app_screen as aps                   # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


def _console_text(console) -> str:
    """Concatenate every stdout block a ConsolePanel has rendered."""
    from spacr.qt.widgets.console_panel import _StdoutBlock

    return "\n".join(block.text()
                     for block in console.findChildren(_StdoutBlock))


@pytest.fixture
def screen(qtbot):
    """A regression screen: results, figures and the live volcano."""
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


# -- the plugin apps a screen picks up while it is being imported -------------

def _run_the_module_body_again(path: Path):
    """Execute ``app_screen``'s own module body into a throwaway namespace.

    The plugin absorption at the top of the module runs exactly once, at
    import, and by the time any test can patch the plugin table it has already
    happened. Re-running the body under the real filename is the only way to
    drive it; it is executed into a fresh module object so the imported
    ``spacr.qt.screens.app_screen`` every other test holds is untouched.
    """
    module = types.ModuleType("spacr.qt.screens.app_screen")
    module.__file__ = str(path)
    module.__package__ = "spacr.qt.screens"
    module.__loader__ = aps.__loader__
    module.__spec__ = aps.__spec__
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"),
         module.__dict__)
    return module


class TestAPluginAppGetsAHeaderAndABlurb:

    @pytest.fixture(autouse=True)
    def _keep_the_qss_registry(self):
        """The module body re-registers the settings-column QSS block."""
        from spacr.qt import theme

        saved = theme._WIDGET_QSS.get(aps.SETTINGS_PANEL_NAME)
        yield
        if saved is not None:
            theme._WIDGET_QSS[aps.SETTINGS_PANEL_NAME] = saved

    def test_a_registered_plugin_names_itself_in_both_tables(self,
                                                             monkeypatch):
        """Otherwise a plugin screen opens with no header and no intro."""
        import spacr.plugins

        contribution = types.SimpleNamespace(
            key="ghost_module",
            name="Ghost Module",
            description="Counts what is not there.")
        monkeypatch.setattr(spacr.plugins, "plugin_apps",
                            lambda: (contribution,))

        module = _run_the_module_body_again(Path(aps.__file__))

        assert module.APP_TITLES["ghost_module"] == "Ghost Module"
        assert module.APP_INTROS["ghost_module"] == "Counts what is not there."

    def test_a_plugin_table_that_explodes_still_lets_the_screen_import(
            self, monkeypatch):
        """Discovery records its own failures; the class must still exist.

        The comparison is the point: the same table, enumerable, puts the key
        in -- so the empty table below is the failure being swallowed and not
        a plugin that was never offered.
        """
        import spacr.plugins

        contribution = types.SimpleNamespace(
            key="ghost_module",
            name="Ghost Module",
            description="Counts what is not there.")

        monkeypatch.setattr(spacr.plugins, "plugin_apps",
                            lambda: (contribution,))
        working = _run_the_module_body_again(Path(aps.__file__))
        assert "ghost_module" in working.APP_TITLES

        def refusing():
            raise RuntimeError("the plugin index is corrupt")

        monkeypatch.setattr(spacr.plugins, "plugin_apps", refusing)
        broken = _run_the_module_body_again(Path(aps.__file__))

        assert "ghost_module" not in broken.APP_TITLES
        assert issubclass(broken.AppScreen, QWidget)


# -- an explainer whose screen has gone --------------------------------------

class TestTheExplainerAfterItsScreenIsGone:

    def test_a_locale_change_on_an_orphaned_browser_renders_nothing(self,
                                                                    qtbot):
        """The browser holds a weak method so it cannot keep a screen alive.

        Which means the screen can be collected while the browser is still
        being told the language changed, and the only correct answer then is
        to render nothing rather than to raise inside Qt's retranslate walk.
        """
        rendered = []

        class Owner:
            def refresh(self, language):
                rendered.append(language)

        owner = Owner()
        browser = aps._ExplainerBrowser(owner.refresh)
        qtbot.addWidget(browser)

        browser.retranslate_dynamic_content("is")
        assert rendered == ["is"]

        del owner
        gc.collect()

        browser.retranslate_dynamic_content("de")

        assert rendered == ["is"]


# -- the two containers that build the rest of the panel when read ------------

class TestReadingAPanelBuildsTheRestOfIt:

    def test_walking_the_caption_index_hands_over_every_caption(self):
        """A check that walks the index must not see a fifth of the panel."""
        built = []

        def caption_the_rest():
            built.append(1)
            index["late_key"] = "late help"

        index = aps._CaptionsBuiltWhenTheyAreAskedFor(caption_the_rest)
        index["early_key"] = "early help"

        assert set(index.keys()) == {"early_key", "late_key"}
        assert sorted(index.values()) == ["early help", "late help"]
        # Idempotent: the second read does not build a second time.
        assert built == [1]

    def test_a_lookup_alone_builds_nothing(self):
        """A pointer crossing the panel must not rebuild it several times a
        second, so the difference between reading one entry and walking the
        index is the whole reason the class exists."""
        built = []
        index = aps._CaptionsBuiltWhenTheyAreAskedFor(
            lambda: built.append(1))

        assert index.get("anything") is None
        assert built == []

        list(index.keys())

        assert len(built) == 1

    def test_counting_containing_and_printing_a_heading_builds_its_rows(self):
        """Every reader of ``Section._row_widgets`` gets the whole heading."""
        late = ("late label", object())

        def build_the_rest():
            rows.append(late)

        rows = aps._RowsBuiltWhenTheyAreAskedFor(build_the_rest)
        early = ("early label", object())
        rows.append(early)

        assert len(rows) == 2
        assert late in rows
        assert "late label" in repr(rows)


# -- the fractal backdrop, wound down for the duration of a run ---------------

def _backdrop(**answers):
    """A stand-in for the spaceout fractal: `pause`/`resume`/`backend_name`."""
    return types.SimpleNamespace(backend_name="numba", **answers)


def _screen_showing(*backdrops):
    """A screen-shaped object whose window holds exactly ``backdrops``."""
    window = types.SimpleNamespace(findChildren=lambda _cls: list(backdrops))
    return types.SimpleNamespace(window=lambda: window)


class TestFindingTheBackdrop:

    def test_a_screen_with_no_window_yet_yields_no_backdrop(self):
        """Built before it is parented, a screen's window is not there.

        The same screen, once it has one, finds the backdrop -- which is what
        makes the empty answer a missing window rather than a missing search.
        """
        one = _backdrop(pause=lambda: True)

        assert list(aps._each_fractal_backdrop(_screen_showing(one))) == [one]

        unparented = types.SimpleNamespace(window=lambda: None)

        assert list(aps._each_fractal_backdrop(unparented)) == []


class TestPausingAndResumingTheFractal:

    def test_a_window_that_cannot_be_walked_pauses_nothing_and_says_so(
            self, caplog):
        def gone():
            raise RuntimeError("wrapped C/C++ object has been deleted")

        caplog.set_level(logging.DEBUG, logger=aps.LOG.name)

        assert aps._pause_the_fractal(
            types.SimpleNamespace(window=gone)) == 0
        assert "could not reach the fractal" in caplog.text

    def test_a_backdrop_that_will_not_resume_does_not_stop_the_others(
            self, caplog):
        """One refusing backdrop must not leave the rest of them frozen."""
        def refusing():
            raise RuntimeError("the GPU context is gone")

        caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
        already_running = _backdrop(pause=lambda: True, resume=lambda: False)
        broken = _backdrop(pause=lambda: True, resume=refusing)
        good = _backdrop(pause=lambda: True, resume=lambda: True)

        resumed = aps._resume_the_fractal(
            _screen_showing(already_running, broken, good))

        assert resumed == 1
        assert "could not resume the fractal" in caplog.text

    def test_a_window_that_cannot_be_walked_resumes_nothing_and_says_so(
            self, caplog):
        def gone():
            raise RuntimeError("wrapped C/C++ object has been deleted")

        caplog.set_level(logging.DEBUG, logger=aps.LOG.name)

        assert aps._resume_the_fractal(
            types.SimpleNamespace(window=gone)) == 0
        assert "could not reach the fractal" in caplog.text


# -- the two backdrops a screen may be given ---------------------------------

class TestTheSequencingRain:

    def test_a_rain_that_will_not_install_leaves_the_screen_without_one(
            self, qtbot, monkeypatch):
        """A decoration that cannot be built must not cost the module.

        ``map_barcodes`` is the one key in :data:`DNA_RAIN_APPS`, so it is the
        only screen that takes this branch at all. Both arms are driven here:
        an installer that answers gives the screen its rain, and one that
        raises leaves the attribute explicitly empty rather than unset.
        """
        from spacr.qt.widgets import dna_rain

        assert "map_barcodes" in aps.DNA_RAIN_APPS

        installed = object()
        monkeypatch.setattr(dna_rain, "install_dna_rain",
                            lambda *a, **k: installed)
        working = AppScreen("map_barcodes")
        qtbot.addWidget(working)
        assert working._dna_rain is installed

        def refusing(*_a, **_k):
            raise RuntimeError("no OpenGL surface for the rain")

        monkeypatch.setattr(dna_rain, "install_dna_rain", refusing)
        broken = AppScreen("map_barcodes")
        qtbot.addWidget(broken)

        assert broken._dna_rain is None


class TestTheAmbientBackdrop:

    def _prefer(self, monkeypatch, theme, palette):
        from spacr.qt import preferences

        monkeypatch.setattr(preferences, "get_ambient_enabled", lambda: True)
        monkeypatch.setattr(preferences, "get_ambient_theme", lambda: theme)
        monkeypatch.setattr(preferences, "get_ambient_palette",
                            lambda: palette)

    def test_a_backdrop_that_refuses_the_new_theme_keeps_the_old_one(
            self, screen, monkeypatch):
        """A refusal must not record a theme the widget is not painting.

        ``_ambient_applied`` is what the next call compares against, so
        recording a theme the backdrop rejected would freeze it on the old
        one for good.
        """
        assert aps.uses_ambient_background(screen.app_key)
        self._prefer(monkeypatch, "blobs", "cool")

        applied = []
        screen._ambient = types.SimpleNamespace(
            set_theme=applied.append,
            set_palette=applied.append)
        screen._ambient_applied = None
        screen.refresh_ambient_background()
        assert applied == ["blobs", "cool"]
        assert screen._ambient_applied == ("blobs", "cool")

        def refusing(_value):
            raise RuntimeError("the backdrop has been deleted")

        self._prefer(monkeypatch, "rain", "warm")
        screen._ambient = types.SimpleNamespace(set_theme=refusing,
                                                set_palette=refusing)
        screen.refresh_ambient_background()

        assert screen._ambient_applied == ("blobs", "cool")


# -- following the settings that decide what the form holds ------------------

class _RecordingSignal:
    """A signal that records its connections, or refuses them."""

    def __init__(self, refuse=False):
        self.connected = []
        self._refuse = refuse

    def connect(self, slot):
        if self._refuse:
            raise RuntimeError("this signal has no receiver list")
        self.connected.append(slot)


class TestWatchingTheFormShapingSettings:

    def _model_with(self, widget):
        return types.SimpleNamespace(
            _widgets={"number_of_organelles": widget})

    def test_a_field_commit_is_what_asks_for_a_rebuild(self, screen):
        """One connection per shaping key, on the commit signal it has."""
        widget = types.SimpleNamespace(editingFinished=_RecordingSignal())
        screen._settings_model = self._model_with(widget)

        screen._watch_the_settings_that_decide_the_form()

        assert widget.editingFinished.connected == [screen._rebuild_the_form]

    def test_with_no_model_there_is_nothing_to_follow(self, screen):
        """A screen whose panel failed to build must not raise here.

        The model that IS there connects, above; this one has nothing to read
        ``_widgets`` off, and the loop must never run at all.
        """
        screen._settings_model = None

        screen._watch_the_settings_that_decide_the_form()

        assert screen._settings_model is None

    def test_a_commit_signal_that_refuses_falls_through_to_the_value_signal(
            self, screen):
        """A spin box whose `editingFinished` will not take a receiver."""
        widget = types.SimpleNamespace(
            editingFinished=_RecordingSignal(refuse=True),
            valueChanged=_RecordingSignal())
        screen._settings_model = self._model_with(widget)

        screen._watch_the_settings_that_decide_the_form()

        assert widget.valueChanged.connected == [screen._rebuild_the_form]

    def test_a_widget_whose_every_signal_refuses_is_simply_skipped(self,
                                                                   screen):
        """And the pass carries on to the next shaping key.

        Nothing on this widget can be connected to, so the only visible
        outcome is that the run of connections is empty and the sweep still
        returns -- which is why the working widget above is the comparison.
        """
        widget = types.SimpleNamespace(
            editingFinished=_RecordingSignal(refuse=True),
            valueChanged=_RecordingSignal(refuse=True),
            currentIndexChanged=_RecordingSignal(refuse=True))
        screen._settings_model = self._model_with(widget)

        screen._watch_the_settings_that_decide_the_form()

        assert widget.valueChanged.connected == []
        assert widget.currentIndexChanged.connected == []


class TestRebuildingTheForm:

    def _window_that_rebuilds(self, screen, monkeypatch):
        asked = []
        window = screen.window()
        monkeypatch.setattr(
            window, "rebuild_app_screen",
            lambda key, keep: asked.append((key, dict(keep))),
            raising=False)
        return asked

    def test_a_shape_that_has_not_changed_costs_no_rebuild(self, screen,
                                                            monkeypatch):
        """Two signals for one edit must not build the screen twice."""
        asked = self._window_that_rebuilds(screen, monkeypatch)
        screen._form_shape_on_screen = screen._form_shape()
        screen._deferred_form_values = None

        screen._rebuild_the_form()
        assert asked == []

        screen._form_shape_on_screen = (("number_of_organelles", "9"),)
        screen._rebuild_the_form()

        assert [key for key, _keep in asked] == [screen.app_key]

    def test_a_rebuild_already_under_way_does_not_start_a_second(self,
                                                                 screen,
                                                                 monkeypatch):
        """Re-entry here is a screen replacing the screen replacing it."""
        asked = self._window_that_rebuilds(screen, monkeypatch)
        screen._form_shape_on_screen = (("number_of_organelles", "9"),)

        screen._rebuilding_the_form = True
        screen._rebuild_the_form()
        assert asked == []

        screen._rebuilding_the_form = False
        screen._rebuild_the_form()

        assert len(asked) == 1


# -- the settings the run has no object for, and the rows that wait ----------

class TestWhichObjectsTheRunHas:

    def test_a_model_that_cannot_answer_hides_nothing(self, screen, caplog):
        """A panel must never be built with every gated row hidden.

        The working model below names one key, so the empty answer that
        follows it is the refusal being absorbed rather than a run that
        happens to have every object.
        """
        screen._run_has_no_object_for = None
        screen._settings_model = types.SimpleNamespace(
            keys_whose_object_the_run_lacks=lambda: {"pathogen_channel"})
        assert screen._keys_the_run_has_no_object_for() == {"pathogen_channel"}

        def refusing():
            raise RuntimeError("the organelle slots are not built")

        caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
        screen._run_has_no_object_for = None
        screen._settings_model = types.SimpleNamespace(
            keys_whose_object_the_run_lacks=refusing)

        assert screen._keys_the_run_has_no_object_for() == set()
        assert "could not decide which objects the run has" in caplog.text

    def test_with_no_model_at_all_the_answer_is_still_a_set(self, screen):
        screen._run_has_no_object_for = None
        screen._settings_model = None

        assert screen._keys_the_run_has_no_object_for() == set()

    def test_the_answer_is_taken_once_and_kept_for_the_build(self, screen):
        """Every heading asks, and the answer cannot change mid-build."""
        asked = []

        def once():
            asked.append(1)
            return {"nucleus_channel"}

        screen._run_has_no_object_for = None
        screen._settings_model = types.SimpleNamespace(
            keys_whose_object_the_run_lacks=once)

        first = screen._keys_the_run_has_no_object_for()

        assert screen._keys_the_run_has_no_object_for() is first
        assert asked == [1]


def _a_heading_with_one_waiting_row(screen):
    """Build a heading holding one row the object rule has decided to hide.

    Through :meth:`AppScreen._build_settings_section`, which is the only
    thing that creates a waiting row, so the section, its declared-row record
    and ``_rows_awaiting_layout`` are left exactly as a panel build leaves
    them.

    NOT a fixture: pytest-qt spins the event loop between setup and the test
    body, and the settings model's queued object-visibility pass captions
    every waiting row the rule does not hide -- so a heading built in a
    fixture arrives already captioned.
    """
    field = QLineEdit()
    screen._settings_model._widgets["ghost_setting"] = field
    screen._widget_key_stamp = None
    screen._run_has_no_object_for = {"ghost_setting"}
    section = screen._build_settings_section(
        ("Ghost", [("Ghost setting", field)]))
    return section, field


def _rows_of(section):
    """The heading's rows -- which is what asks for the waiting ones."""
    return [widget for _label, widget in section.__dict__["_row_widgets"]]


class TestABuiltHeading:

    def test_a_model_that_cannot_remember_its_rows_still_gets_a_heading(
            self, screen):
        """An older settings model has no ``remember_section_rows``.

        The heading is what the user sees, and losing a whole category
        because the model could not be told what it holds would take its
        settings off the panel entirely.
        """
        field = QLineEdit()
        screen._run_has_no_object_for = set()
        screen._settings_model = types.SimpleNamespace(_widgets={})

        section = screen._build_settings_section(
            ("Old model", [("A setting", field)]))

        assert section.title().lower() == "old model"
        assert _rows_of(section) == [field]

    def test_a_hidden_row_is_on_the_form_but_not_yet_captioned(self, screen):
        """The field spans its row; the caption is what is deferred."""
        section, field = _a_heading_with_one_waiting_row(screen)

        assert screen._rows_awaiting_layout == {"ghost_setting": section}
        assert section._spacr_declared_rows == (
            ("ghost_setting", "Ghost setting", field),)
        # The caption is what is missing: `_lay_out_setting_row` is what
        # binds one to its field, and it has not run for this row yet.
        assert not hasattr(field, "_spacr_setting_label")

    def test_reading_the_heading_back_hands_over_the_caption(self, screen):
        """`_RowsBuiltWhenTheyAreAskedFor` is what makes the deferral safe."""
        section, field = _a_heading_with_one_waiting_row(screen)
        rows = section.__dict__["_row_widgets"]
        assert isinstance(rows, aps._RowsBuiltWhenTheyAreAskedFor)

        assert [widget for _label, widget in rows] == [field]
        assert screen._rows_awaiting_layout == {}


class TestCaptioningTheRowsThatCameBack:

    def test_a_key_the_rule_still_hides_stays_uncaptioned(self, screen):
        """The rule calls this BEFORE deciding visibility, with what it hides.

        Handing it the same key back must leave the row where it is, while a
        call that does not name it captions it -- the pair is what shows the
        filter is what decided.
        """
        section, field = _a_heading_with_one_waiting_row(screen)

        screen._lay_out_the_rows_that_are_back({"ghost_setting"})
        assert screen._rows_awaiting_layout == {"ghost_setting": section}

        screen._lay_out_the_rows_that_are_back(())

        assert screen._rows_awaiting_layout == {}
        assert _rows_of(section) == [field]

    def test_with_nothing_waiting_there_is_nothing_to_lay_out(self, screen):
        screen._rows_awaiting_layout = {}

        screen._lay_out_the_rows_that_are_back(())

        assert screen._rows_awaiting_layout == {}

    def test_a_heading_asks_only_for_its_own_rows(self, screen):
        """Reading one heading must not caption the whole panel."""
        from spacr.qt.widgets.section import Section

        section, field = _a_heading_with_one_waiting_row(screen)

        screen._lay_out_every_waiting_row(Section("Elsewhere"))
        assert screen._rows_awaiting_layout == {"ghost_setting": section}

        screen._lay_out_every_waiting_row(section)

        assert screen._rows_awaiting_layout == {}
        assert _rows_of(section) == [field]

    def test_a_heading_asked_twice_has_nothing_left_to_hand_over(self,
                                                                 screen):
        section, _field = _a_heading_with_one_waiting_row(screen)
        screen._lay_out_every_waiting_row(section)
        assert screen._rows_awaiting_layout == {}

        screen._lay_out_every_waiting_row(section)

        assert screen._rows_awaiting_layout == {}

    def test_the_coarse_pass_captions_every_heading_at_once(self, screen):
        """What the checks that walk the panel's captions ask for."""
        _section, field = _a_heading_with_one_waiting_row(screen)

        screen._caption_every_waiting_row()

        assert screen._rows_awaiting_layout == {}
        assert field._spacr_setting_label.text() == "Ghost setting"


class TestCaptioningOneWaitingRow:

    def test_a_key_nothing_is_waiting_for_is_left_alone(self, screen):
        section, _field = _a_heading_with_one_waiting_row(screen)

        screen._lay_out_one_waiting_row("not_a_waiting_key")
        assert screen._rows_awaiting_layout == {"ghost_setting": section}

        screen._lay_out_one_waiting_row("ghost_setting")

        assert screen._rows_awaiting_layout == {}

    def test_a_heading_that_no_longer_declares_the_row_captions_nothing(
            self, screen):
        """The declared-row record carries the label and the field."""
        section, _field = _a_heading_with_one_waiting_row(screen)
        section._spacr_declared_rows = ()

        screen._lay_out_one_waiting_row("ghost_setting")

        assert screen._rows_awaiting_layout == {}
        assert list(section.__dict__["_row_widgets"]) == []

    def test_a_heading_with_no_form_captions_nothing(self, screen):
        """A heading rebuilt without its QFormLayout has nowhere to put it."""
        section, _field = _a_heading_with_one_waiting_row(screen)
        section._form = None

        screen._lay_out_one_waiting_row("ghost_setting")

        assert screen._rows_awaiting_layout == {}
        assert list(section.__dict__["_row_widgets"]) == []

    def test_a_field_that_is_not_on_the_form_is_captioned_at_the_end(
            self, screen):
        """`getWidgetPosition` answers -1, so there is no row to take out."""
        section, field = _a_heading_with_one_waiting_row(screen)
        section._form.takeRow(field)

        screen._lay_out_one_waiting_row("ghost_setting")

        assert _rows_of(section) == [field]
        assert field._spacr_setting_label.text() == "Ghost setting"

    def test_a_heading_destroyed_under_the_pass_is_only_logged(self, screen,
                                                               caplog):
        """The screen that owned it went away between the two calls."""
        section, _field = _a_heading_with_one_waiting_row(screen)
        caplog.set_level(logging.DEBUG, logger=aps.LOG.name)

        def gone(_widget):
            raise RuntimeError("Internal C++ object already deleted.")

        section._form.getWidgetPosition = gone

        screen._lay_out_one_waiting_row("ghost_setting")

        assert "no heading left to caption" in caplog.text
        assert screen._rows_awaiting_layout == {}

    def test_the_first_late_caption_opens_the_record_of_them(self, screen):
        """`_captioned_late` is what the language pass is handed afterwards."""
        section, _field = _a_heading_with_one_waiting_row(screen)
        screen._captioned_late = None

        screen._lay_out_one_waiting_row("ghost_setting")

        assert screen._captioned_late == {section}


# -- putting the panel's row-shaped answers back in step ---------------------

class _Heading:
    """The parts of a built heading ``_the_rows_moved`` reads."""

    def __init__(self, rows, declared):
        self._row_widgets = list(rows)
        self._spacr_declared_rows = tuple(declared)


class TestTheRowsMoved:

    def test_a_heading_goes_back_into_the_order_it_was_declared_in(
            self, screen):
        """A late row is appended, and appended is not where it belongs.

        The second heading holds something the declared record does not name
        -- a field inside a holder with a button beside it -- and guessing its
        place would be worse than the order it already has.
        """
        first, second = QLineEdit(), QLineEdit()
        holder = QWidget()
        placeable = _Heading(
            rows=[("Second", second), ("First", first)],
            declared=[("a", "First", first), ("b", "Second", second)])
        unplaceable = _Heading(
            rows=[("Second", second), ("Held", holder)],
            declared=[("a", "First", first), ("b", "Second", second)])
        screen._settings_sections = [placeable, unplaceable]

        screen._the_rows_moved(judge_them=False)

        assert placeable._row_widgets == [("First", first), ("Second", second)]
        assert unplaceable._row_widgets == [("Second", second),
                                            ("Held", holder)]

    def test_the_rule_that_asked_for_the_rows_is_not_run_again(self, screen):
        """It decides every gated row itself the moment this returns."""
        judged = []
        screen._settings_sections = []
        screen._settings_model = types.SimpleNamespace(
            refresh_object_visibility=lambda: judged.append(1))

        screen._the_rows_moved(judge_them=False)
        assert judged == []

        screen._the_rows_moved(judge_them=True)

        assert judged == [1]

    def test_with_no_model_there_is_no_object_rule_to_run(self, screen):
        screen._settings_sections = []
        screen._settings_model = None

        screen._the_rows_moved(judge_them=True)

        assert screen._captioned_late == set()

    def test_an_object_rule_that_throws_does_not_abandon_the_rest(
            self, screen, caplog):
        """The dimension switches and the search index still get their pass."""
        def refusing():
            raise RuntimeError("the organelle slots are half-built")

        caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
        screen._settings_sections = []
        screen._settings_model = types.SimpleNamespace(
            refresh_object_visibility=refusing)
        indexed = []
        screen._settings_search = types.SimpleNamespace(
            _build_index=lambda: indexed.append("index"),
            apply=lambda: indexed.append("apply"))

        screen._the_rows_moved(judge_them=True)

        assert "could not re-decide the object rows" in caplog.text
        assert indexed == ["index", "apply"]

    def test_dimension_switches_that_throw_are_logged_and_stepped_over(
            self, screen, caplog, monkeypatch):
        def refusing():
            raise RuntimeError("the 3D switch has been deleted")

        caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
        screen._settings_sections = []
        monkeypatch.setattr(screen, "_apply_dimension_visibility", refusing)
        indexed = []
        screen._settings_search = types.SimpleNamespace(
            _build_index=lambda: indexed.append("index"),
            apply=lambda: indexed.append("apply"))

        screen._the_rows_moved(judge_them=False)

        assert "could not re-apply the dimension switches" in caplog.text
        assert indexed == ["index", "apply"]

    def test_a_caption_that_arrived_late_is_translated_where_it_stands(
            self, screen, monkeypatch):
        """Otherwise it sits in English inside a translated window."""
        from spacr.qt import i18n

        walked = []
        monkeypatch.setattr(i18n, "retranslate_widget_tree", walked.append)
        screen._settings_sections = []

        screen._captioned_late = set()
        screen._the_rows_moved(judge_them=False)
        assert walked == []

        heading = QWidget()
        screen._captioned_late = {heading}
        screen._the_rows_moved(judge_them=False)

        assert walked == [heading]
        assert screen._captioned_late == set()

    def test_a_language_pass_that_throws_leaves_the_caption_in_english(
            self, screen, monkeypatch, caplog):
        from spacr.qt import i18n

        def refusing(_widget):
            raise ValueError("no catalog for this locale")

        caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
        monkeypatch.setattr(i18n, "retranslate_widget_tree", refusing)
        screen._settings_sections = []
        screen._captioned_late = {QWidget()}

        screen._the_rows_moved(judge_them=False)

        assert "could not translate a caption that arrived late" in caplog.text

    def test_a_search_strip_that_will_not_re_index_is_only_logged(
            self, screen, caplog):
        """A row that has just been captioned may hold a different widget."""
        def refusing():
            raise RuntimeError("the strip's model is gone")

        caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
        screen._settings_sections = []
        screen._settings_search = types.SimpleNamespace(
            _build_index=refusing, apply=lambda: None)

        screen._the_rows_moved(judge_them=False)

        assert "could not re-index the settings search" in caplog.text

    def test_a_screen_with_no_search_strip_indexes_nothing(self, screen):
        screen._settings_sections = []
        screen._settings_search = None

        screen._the_rows_moved(judge_them=False)

        assert screen._settings_search is None


# -- the help that moves onto the setting's name ------------------------------

class TestTheNameCarriesTheHelp:

    def test_help_that_will_not_move_is_a_blemish_not_a_closed_module(
            self, screen, monkeypatch):
        """A module that would not open because a tooltip failed is worse."""
        from spacr.qt.screens import settings_model

        monkeypatch.setattr(settings_model, "retarget_field_tooltips",
                            lambda _widget: 7)
        assert screen.the_name_carries_the_help() == 7

        def refusing(_widget):
            raise RuntimeError("a label was destroyed mid-sweep")

        monkeypatch.setattr(settings_model, "retarget_field_tooltips",
                            refusing)

        assert screen.the_name_carries_the_help() == 0


# -- opening the umbrella above a heading ------------------------------------

class TestOpeningTheHeadingsAbove:

    def test_an_umbrella_that_is_already_open_is_left_alone(self):
        """Reasserting it would fight a user who had just collapsed it."""
        opened = []
        already = types.SimpleNamespace(is_expanded=lambda: True,
                                        set_expanded=opened.append)
        shut = types.SimpleNamespace(is_expanded=lambda: False,
                                     set_expanded=opened.append)

        AppScreen._open_the_headings_above(already, True)
        assert opened == []

        AppScreen._open_the_headings_above(shut, True)

        assert opened == [True]


# -- the two spellings the advisor reads its tables out of --------------------

class TestTheTablesTheAdvisorReads:

    def test_a_plate_row_naming_only_a_score_contributes_only_a_score(self):
        """Half-filled rows are ordinary: the count arrives next."""
        counts, scores = AppScreen._tables_for_the_advisor({
            "paired_data": [
                {"score": "plate1_dv.csv"},
                {"count": "plate2_counts.csv", "score": "plate2_dv.csv"},
                "not a row at all",
            ],
        })

        assert counts == ["plate2_counts.csv"]
        assert scores == ["plate1_dv.csv", "plate2_dv.csv"]

    def test_a_legacy_settings_file_naming_one_path_as_a_string(self):
        counts, scores = AppScreen._tables_for_the_advisor({
            "count_data": "old_counts.csv",
            "score_data": ["old_scores.csv", "old_scores.csv"],
        })

        assert counts == ["old_counts.csv"]
        assert scores == ["old_scores.csv"]


# -- folding the console away -------------------------------------------------

class _OnePaneSplitter:
    """A splitter holding the console and nothing else.

    A real QSplitter with one child pins that child to its own height, so
    the sizes it was ASKED for -- which is what the fold computes -- cannot
    be read back off it. This records them instead, and having exactly one
    pane is the case under test: there is no neighbour above to hand the
    freed height to.
    """

    def __init__(self, height):
        self._sizes = [height]
        self.asked_for = []

    def indexOf(self, _widget):
        return 0

    def sizes(self):
        return list(self._sizes)

    def setSizes(self, sizes):
        self._sizes = list(sizes)
        self.asked_for.append(list(sizes))


class TestFoldingTheConsole:

    def _console_in(self, screen):
        wrap = QLabel("spaCR output")
        splitter = _OnePaneSplitter(400)
        screen._console_wrap = wrap
        screen._console_splitter = splitter
        return splitter, wrap

    def test_with_nothing_above_it_the_freed_height_goes_nowhere(self,
                                                                 screen):
        """The pane still shrinks to its heading and the height is kept.

        There is simply no pane above to grow by what it gave up, and the
        fold must not hand the freed pixels to a pane that is not there.
        """
        splitter, wrap = self._console_in(screen)

        screen._console_folded(True)

        assert screen._console_height == 400
        assert wrap.minimumHeight() == 0
        assert splitter.asked_for == [[wrap.sizeHint().height()]]

    def test_a_console_that_was_never_folded_is_given_no_height_back(
            self, screen):
        """`_console_height` is what a fold records.

        Unfolding without one must leave the splitter as the user left it,
        which is the difference between the two halves below.
        """
        splitter, wrap = self._console_in(screen)
        screen._console_height = 0

        screen._console_folded(False)
        assert splitter.asked_for == [[400]]
        assert wrap.minimumHeight() == 180

        screen._console_height = 260
        screen._console_folded(False)

        assert splitter.asked_for[-1] == [260]


# -- the plate map beside a wells field ---------------------------------------

class TestPickingWellsFromTheMap:

    def test_a_field_with_no_setText_still_reports_what_was_chosen(
            self, screen, monkeypatch):
        """The picker's answer is returned even when nothing can hold it.

        A field that CAN take it is written, which is what makes the pair a
        difference in the field rather than in the picker.
        """
        from spacr.qt.widgets import plate_map_picker

        class Picker:
            _layout_size = 96

            def __init__(self, before, parent=None):
                self.before = before

            def exec(self):
                return True

            def value(self):
                return "A1,B2"

        monkeypatch.setattr(plate_map_picker, "PlateMapPicker", Picker)

        field = QLineEdit()
        assert screen.pick_wells_for(field, "control_wells") == "A1,B2"
        assert field.text() == "A1,B2"

        readonly = types.SimpleNamespace(text=lambda: "")

        assert screen.pick_wells_for(readonly, "control_wells") == "A1,B2"
        assert "A1,B2 chosen from the 96-well map" in _console_text(
            screen._console)


# -- a rebuild the window refuses ---------------------------------------------

class TestARebuildThatFails:

    def test_a_window_that_throws_leaves_the_screen_usable(self, screen,
                                                            monkeypatch,
                                                            caplog):
        """The re-entry guard has to be cleared however the rebuild ends.

        A screen left with ``_rebuilding_the_form`` set would refuse every
        later shaping edit for the rest of the session, silently.
        """
        caplog.set_level(logging.ERROR, logger=aps.LOG.name)
        window = screen.window()

        def refusing(_key, _keep):
            raise RuntimeError("the module registry is not loaded")

        monkeypatch.setattr(window, "rebuild_app_screen", refusing,
                            raising=False)
        screen._form_shape_on_screen = (("number_of_organelles", "9"),)

        screen._rebuild_the_form()

        assert "could not rebuild the settings form" in caplog.text
        assert screen._rebuilding_the_form is False


# -- the box that explains the model, and what it follows ---------------------

class TestTheModelExplainersFollowers:

    def test_only_the_settings_that_are_on_the_panel_are_followed(
            self, screen, monkeypatch):
        """`level` is new and may not be built; a plain field has no signal.

        Both are ordinary. The checkbox is the comparison: a setting that HAS
        a commit signal is connected, and the box re-renders when it flips.
        """
        from PySide6.QtWidgets import QCheckBox
        from spacr.qt.widgets.section import Section

        rendered = []
        monkeypatch.setattr(screen, "_refresh_model_explainer",
                            lambda *_a: rendered.append(1))

        signal_less = QWidget()
        screen._settings_model = types.SimpleNamespace(
            _widgets={"regression_type": signal_less})
        screen._install_section_explainer(Section("Model & Inference"),
                                          "Model & Inference")
        assert rendered == [1]

        box = QCheckBox()
        screen._settings_model = types.SimpleNamespace(
            _widgets={"regression_type": box})
        screen._install_section_explainer(Section("Model & Inference"),
                                          "Model & Inference")
        assert rendered == [1, 1]

        box.setChecked(True)

        assert len(rendered) == 3


# -- the example plate, fetched or already on disk ---------------------------

class _Fetched:
    """What ``download_toxo_mito_demo`` hands its callback."""

    def __init__(self, dataset_path, settings_path):
        self.dataset_path = dataset_path
        self.settings_path = settings_path


class TestLoadingTheExampleImages:

    def _point_at(self, screen, monkeypatch, tmp_path):
        monkeypatch.setattr(screen, "example_images_destination",
                            lambda: tmp_path)

    def test_a_plate_already_on_disk_is_reused_rather_than_downloaded(
            self, screen, monkeypatch, tmp_path):
        """The dataset is about 400 MB, so this is the branch that matters."""
        # The destination IS the plate folder now, and it is shared with the
        # other example sets -- so an image, not a non-empty folder, is what
        # says this one has already been fetched.
        self._point_at(screen, monkeypatch, tmp_path)
        (tmp_path / "field.tif").write_bytes(b"")

        asked = []
        placed = screen.load_the_example_images(
            ask=lambda *a: asked.append(a))

        assert asked == []
        assert placed["src"] == str(tmp_path)
        assert placed["settings"] == ""

    def test_a_download_that_fails_says_so_and_restores_the_button(
            self, screen, monkeypatch, tmp_path):
        from PySide6.QtWidgets import QPushButton

        self._point_at(screen, monkeypatch, tmp_path)
        button = QPushButton("Load the example images…")
        screen._example_images_button = button

        def download(_screen, _destination, done):
            assert not button.isEnabled()
            assert "Fetching" in button.text()
            done(None, "no route to huggingface.co")

        assert screen.load_the_example_images(ask=download) == {}
        assert button.isEnabled()
        # RESTORED TO THE CANONICAL CAPTION, not to whatever the button
        # happened to be created with: the screen writes "Fetching…" over
        # the name while the download runs and writes its own name back
        # afterwards. This asserted the literal "Load the example images",
        # which the button has not said since the three example-data buttons
        # became one "Load test data…" -- so the test was pinning a name that
        # had moved rather than the behaviour it is about, which is that a
        # failed download leaves a button the user can press again.
        assert "Fetching" not in button.text()
        assert button.text().strip()
        assert "no route to huggingface.co" in _console_text(screen._console)

    def test_a_finished_download_fills_src_and_names_the_settings(
            self, screen, monkeypatch, tmp_path):
        """Both halves of the answer: the folder AND the settings beside it."""
        self._point_at(screen, monkeypatch, tmp_path)
        screen._example_images_button = None
        images = tmp_path / "toxo_mito"
        images.mkdir()
        settings = tmp_path / "toxo_mito.csv"
        settings.write_text("k,v\n", encoding="utf-8")

        def download(_screen, _destination, done):
            done(_Fetched(images, settings), None)

        placed = screen.load_the_example_images(ask=download)

        assert placed == {"src": str(images), "settings": str(settings)}
        assert screen._settings_model._widgets["src"].text() == str(images)
        text = _console_text(screen._console)
        assert f"Source directory (src): {images}" in text
        assert f"Compatible example settings: {settings}" in text

    def test_with_no_downloader_named_the_real_one_is_reached_for(
            self, screen, monkeypatch, tmp_path):
        """``ask`` exists for tests; the button passes nothing."""
        from spacr.qt import hf_download

        self._point_at(screen, monkeypatch, tmp_path)
        screen._example_images_button = None
        called = []
        # The screen now asks for the TAR worker rather than taking the
        # shared default, so the real function is reached with a
        # `worker_factory` keyword. Accepting it here rather than swallowing
        # every keyword: the request for the tar is the point of the call,
        # and a stub that ignored it could not notice the screen dropping it.
        monkeypatch.setattr(
            hf_download, "download_toxo_mito_demo",
            lambda screen_, destination, done, worker_factory=None:
                called.append((destination, worker_factory)))

        assert screen.load_the_example_images() == {}
        assert len(called) == 1
        destination, worker = called[0]
        assert destination == str(tmp_path)
        assert worker is hf_download._MaskTarWorker, (
            "the mask demo must ask for the archive, not fetch 212 files")

    def test_a_screen_with_no_src_control_still_reports_the_folder(
            self, screen, tmp_path):
        """A module without ``src`` is told where the images went anyway."""
        screen._settings_model = types.SimpleNamespace(_widgets={})

        placed = screen._put_the_example_images_in_place(tmp_path, None)

        assert placed == {"src": str(tmp_path), "settings": ""}
        assert f"Source directory (src): {tmp_path}" in _console_text(
            screen._console)


class TestLoadingTheExampleScreen:

    def test_a_screen_with_no_fetch_button_still_pairs_the_tables(
            self, screen, monkeypatch, tmp_path):
        """The button is the progress display, not part of the transaction."""
        from spacr import example_data

        got = types.SimpleNamespace(
            counts=["p1_counts.csv"], scores=["p1_dv.csv"],
            folder=str(tmp_path), note=lambda: "Cached.")
        # `kind` is accepted because the screen now fetches counts and
        # scores separately; a stub without it hides that call.
        monkeypatch.setattr(example_data, "missing",
                            lambda folder=None, kind=None: [])
        monkeypatch.setattr(example_data, "fetch",
                            lambda **_kwargs: got)
        screen._example_data_button = None
        added = []
        screen._settings_model._widgets["paired_data"] = types.SimpleNamespace(
            add_paths_for_side=lambda paths, side: (
                added.append((tuple(paths), side)) or len(paths)))

        answer = screen.load_the_example_screen(download=False)

        assert answer["applied"] == 2
        assert added == [(("p1_dv.csv",), "score"),
                         (("p1_counts.csv",), "count")]


# -- the usage poll following the page ----------------------------------------

class TestTheUsagePollFollowsTheScreen:

    def test_a_screen_with_no_poll_timer_still_shows_and_hides(self, screen):
        """The timer is optional; the generation bump that invalidates an
        in-flight sample is not."""
        from PySide6.QtGui import QHideEvent, QShowEvent

        timer = screen._usage_timer
        assert timer is not None
        timer.stop()
        screen.showEvent(QShowEvent())
        assert timer.isActive()

        screen._usage_timer = None
        before = screen._usage_generation
        screen.showEvent(QShowEvent())
        screen.hideEvent(QHideEvent())

        assert screen._usage_generation == before + 1

    def test_every_core_bar_is_painted_from_one_sample(self, screen):
        painted = []
        screen._per_core_bars = [
            types.SimpleNamespace(set_value=lambda v: painted.append(v))
            for _ in range(3)]

        screen._apply_usage({"per_core": [11, 22, 33]})

        assert painted == [11, 22, 33]


# -- what the run banner says about a permutation test ------------------------

class TestAnnouncingTheFit:

    def test_a_permutation_run_gets_its_own_banner_and_is_never_slow(
            self, screen):
        """It fits no model, so the model/level/backend sentence is wrong.

        The ordinary settings below print that sentence, which is what makes
        the permutation banner a different branch rather than a missing one.
        """
        screen._announce_the_fit({"regression_type": "ols", "level": "gene"})
        assert "Model: ols" in _console_text(screen._console)
        assert screen._slow_fit is False

        screen._announce_the_fit({"inference": "nonparametric",
                                  "guide_permutations": 10000})

        text = _console_text(screen._console)
        assert "permutation" in text.lower()
        assert screen._slow_fit is False


# -- the cards the action-row switches open -----------------------------------

class TestTheSweepCard:

    def test_a_card_with_no_panel_behind_it_is_shown_and_not_seeded(
            self, screen):
        """The seeding needs both halves; showing the card needs neither."""
        card = QWidget()
        screen._sweep_card = card
        seeded = []
        screen._sweep = types.SimpleNamespace(
            apply_settings=lambda values: seeded.append(values))
        screen._on_sweep_switch(True)
        assert len(seeded) == 1

        screen._sweep = None
        screen._on_sweep_switch(False)
        screen._on_sweep_switch(True)

        assert card.isVisible() or card.isVisibleTo(screen)
        assert len(seeded) == 1

    def test_a_panel_that_refuses_the_seed_still_opens_the_card(
            self, screen, caplog):
        def refusing(_values):
            raise RuntimeError("the sweep grid is not built")

        caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
        screen._sweep_card = QWidget()
        screen._sweep = types.SimpleNamespace(apply_settings=refusing)

        screen._on_sweep_switch(True)

        assert "could not seed the parameter sweep" in caplog.text


# -- the figure that arrives while the interactive view is open ---------------

class _RecordingQueue:
    def __init__(self):
        self.added = []
        self.shown = 0

    def add_figure(self, fig, prerendered_png=None):
        self.added.append(fig)

    def show(self):
        self.shown += 1


class TestAFigureArrivingMidRun:

    def _switch(self, checked):
        return types.SimpleNamespace(isChecked=lambda: checked)

    def test_the_grid_does_not_steal_the_open_interactive_view(self, screen,
                                                                monkeypatch):
        """A figure arriving must not throw the user out of the explorer."""
        from matplotlib.figure import Figure

        monkeypatch.setattr(screen, "_queue_figure_grid_refresh",
                            lambda: None)
        screen._umap_explorer = object()
        screen._umap_payload_ready = True

        queue = screen._figure_queue = _RecordingQueue()
        screen._interactive_switch = self._switch(False)
        screen._on_figure_ready(Figure())
        assert queue.shown == 1

        screen._interactive_switch = self._switch(True)
        screen._on_figure_ready(Figure())

        assert len(queue.added) == 2
        assert queue.shown == 1


# -- closing the screen -------------------------------------------------------

class TestClosingTheScreen:

    def test_a_job_pool_that_was_never_built_is_stepped_over(self, screen):
        """Both pools are named; a screen may only ever have made one."""
        from PySide6.QtGui import QCloseEvent

        drained = []
        screen._usage_jobs = None
        screen._jobs = types.SimpleNamespace(
            shutdown=lambda: drained.append("jobs"))

        screen.closeEvent(QCloseEvent())

        assert drained == ["jobs"]

    def test_menus_that_cannot_be_retired_do_not_stop_the_close(
            self, screen, monkeypatch):
        """A parentless pyqtgraph menu is a leak, not a reason to stay open."""
        from PySide6.QtGui import QCloseEvent
        from spacr.qt import widget_cleanup

        def refusing(_widget):
            raise RuntimeError("the graphics scene is already gone")

        monkeypatch.setattr(widget_cleanup, "retire_pyqtgraph_menus", refusing)
        event = QCloseEvent()
        event.ignore()

        screen.closeEvent(event)

        assert event.isAccepted()


# -- the right-click menus on the results tiles -------------------------------

class TestTheTileMenus:

    def _panel_with_a_volcano(self, opened, fail=False):
        def build_style_menu():
            if fail:
                raise RuntimeError("the plot item is gone")
            return types.SimpleNamespace(exec=opened.append)

        return types.SimpleNamespace(
            volcano=types.SimpleNamespace(
                build_style_menu=build_style_menu))

    def test_the_pinned_tile_is_left_to_its_own_signal(self, screen,
                                                        monkeypatch):
        """`pinned_menu_requested` already carries the pinned tile."""
        from PySide6.QtCore import QPoint
        from spacr.qt.widgets.figure_grid_view import PINNED_KEY

        monkeypatch.setattr(screen, "_pin_regression_graph", lambda: None)
        opened = []
        panel = self._panel_with_a_volcano(opened)
        panel.qq = panel.volcano
        screen._results_panel = panel
        assert AppScreen._LIVE_TILE_WIDGETS["qq"] == "qq"

        screen._live_tile_menu(PINNED_KEY, QPoint(3, 4))
        assert opened == []

        screen._live_tile_menu("qq", QPoint(3, 4))

        assert opened == [QPoint(3, 4)]

    def test_the_pinned_menu_is_the_graphs_own_menu(self, screen,
                                                     monkeypatch):
        from PySide6.QtCore import QPoint

        pinned = []
        monkeypatch.setattr(screen, "_pin_regression_graph",
                            lambda: pinned.append(1))
        opened = []
        screen._results_panel = self._panel_with_a_volcano(opened)

        screen._pinned_menu(QPoint(1, 2))

        assert opened == [QPoint(1, 2)]
        assert pinned == [1]

    def test_a_menu_that_will_not_build_still_retakes_the_photograph(
            self, screen, monkeypatch, caplog):
        """The tile is a picture of the graph, so it is retaken either way."""
        from PySide6.QtCore import QPoint

        caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
        pinned = []
        monkeypatch.setattr(screen, "_pin_regression_graph",
                            lambda: pinned.append(1))
        screen._results_panel = self._panel_with_a_volcano([], fail=True)

        screen._pinned_menu(QPoint(1, 2))

        assert "could not open the live tile's menu" in caplog.text
        assert pinned == [1]


# -- where a merged measurements table is written -----------------------------

class TestWhereTheMergedMeasurementsGo:

    def test_a_row_naming_no_database_is_skipped_for_one_that_does(
            self, screen, monkeypatch):
        """A half-filled input row is ordinary while a project is assembled."""
        screen._settings_model = None
        monkeypatch.setattr(
            screen, "_attached_database_rows",
            lambda: [{"database": ""},
                     {"database":
                      "/proj/plate1/measurements/measurements.db"}])

        assert screen._measurements_destination() == os.path.join(
            "/proj/plate1", "measurements")


# -- a column's own fit, and its row in the Runs tab --------------------------

class TestAColumnFit:

    def test_a_fit_the_runs_tab_would_not_record_gets_no_handle(
            self, screen, monkeypatch):
        """Every outcome is written through the handle, so a missing one must
        not be stored as if it were a row."""
        screen._column_run_handles = {}
        monkeypatch.setattr(screen, "_record_run_in_runs_tab",
                            lambda *_a: "handle-7")
        screen._on_column_fit_started("response_1", {})
        assert screen._column_run_handles == {"response_1": "handle-7"}

        monkeypatch.setattr(screen, "_record_run_in_runs_tab",
                            lambda *_a: None)
        screen._on_column_fit_started("response_2", {})

        assert screen._column_run_handles == {"response_1": "handle-7"}


# -- a run removed from the Runs tab ------------------------------------------

class TestForgettingARemovedRun:

    def test_a_record_with_no_run_label_drops_no_figures(self, screen,
                                                          monkeypatch):
        """The queue sections its tiles BY LABEL, so an unlabelled record has
        nothing to name -- while a labelled one takes its section with it."""
        forgotten = []
        screen._results_panel = types.SimpleNamespace(
            forget_run=lambda folder: forgotten.append(folder))
        screen._compare_panel = None
        dropped = []
        screen._figure_queue = types.SimpleNamespace(
            forget_run=lambda label: dropped.append(label) or True)
        monkeypatch.setattr(screen, "_queue_figure_grid_refresh",
                            lambda: None)

        screen._on_runs_removed([{"folder": "/data/run_1", "run": ""},
                                 {"folder": "/data/run_2", "run": "run 2"},
                                 {"folder": ""}])

        assert forgotten == ["/data/run_1", "/data/run_2"]
        assert dropped == ["run 2"]


# -- the sweep's effects grid -------------------------------------------------

class TestTheEffectsGrid:

    def test_with_nowhere_to_write_it_the_grid_is_simply_not_written(
            self, screen, monkeypatch, tmp_path):
        """A sweep that produced its answer has not failed because of this."""
        import pandas as pd
        from spacr import cell_montage

        written = []
        monkeypatch.setattr(cell_montage, "write_effects_grid",
                            lambda effects, folder: written.append(folder))
        result = types.SimpleNamespace(
            effects=pd.DataFrame({"gene": ["A"], "effect": [1.0]}))

        monkeypatch.setattr(screen, "_results_source_path",
                            lambda: str(tmp_path))
        screen._keep_the_effects_grid(result)
        assert written == [str(tmp_path)]

        monkeypatch.setattr(screen, "_results_source_path", lambda: "")
        screen._keep_the_effects_grid(result)

        assert written == [str(tmp_path)]


# -- a trial opened from the Runs tab -----------------------------------------

class TestOpeningATrial:

    def test_a_runs_tab_with_no_undo_to_spend_is_simply_not_told(
            self, screen, monkeypatch, tmp_path):
        """An older Runs tab has no ``the_load_succeeded``; the figures and
        the sentence still have to arrive."""
        monkeypatch.setattr(screen, "_load_trial_figures", lambda _f: 0)
        screen._pending_trial = ("trial 3", str(tmp_path))
        screen._sweep_runs = None

        screen._on_trial_loaded(True)

        text = _console_text(screen._console)
        assert "trial 3 saved a results table but no figures" in text
        assert "trial 3 is loaded" in text
        assert screen._pending_trial is None


# -- the screen folders a run's shared figures sit in -------------------------

class TestTheScreenFoldersAboveARun:

    def test_a_results_folder_with_no_screen_above_it_names_only_itself(self):
        """A relative ``results/run_1`` has no directory above ``results``."""
        assert AppScreen._screen_folders_above(
            os.path.join("results", "run_1")) == ["results"]
        assert AppScreen._screen_folders_above(
            os.path.join("/data", "screen", "results", "run_1")) == [
                os.path.join("/data", "screen", "results"),
                os.path.join("/data", "screen")]


# -- the publication sheet ----------------------------------------------------

class _Sheet:
    def __init__(self, figure, skipped):
        self.figure = figure
        self.skipped = skipped

    def legend(self):
        return "Panels a-c: coefficients, p-values, QC."


class TestThePublicationSheet:

    def _ready(self, screen, monkeypatch, skipped):
        import pandas as pd
        from matplotlib.figure import Figure
        from spacr import figures

        screen._results_panel = types.SimpleNamespace(
            results_frame=lambda: pd.DataFrame({"gene": ["A"],
                                                "coefficient": [1.0]}))
        figure = Figure()
        monkeypatch.setattr(figures, "build_sheet",
                            lambda frame, width="double": _Sheet(figure,
                                                                 skipped))
        shown = []
        screen._figure_queue = types.SimpleNamespace(
            add_figure=shown.append,
            count=lambda: 1,
            show_index=lambda index: shown.append(index))
        return shown, figure

    def test_the_sheet_goes_into_the_ordinary_figure_queue(self, screen,
                                                            monkeypatch):
        """A bespoke viewer for one figure is a second set of the same bugs."""
        shown, figure = self._ready(screen, monkeypatch, skipped=())
        screen._figures_stack = None

        screen._show_publication_sheet()

        assert shown == [figure, 0]
        assert "Panels a-c" in _console_text(screen._console)

    def test_a_panel_the_sheet_could_not_draw_is_named(self, screen,
                                                        monkeypatch):
        """A missing panel is a fact worth reading, not a silent gap."""
        shown, _figure = self._ready(screen, monkeypatch, skipped=(
            types.SimpleNamespace(title="QQ", reason="no residuals"),))
        detail = QWidget()
        moved = []
        screen._figures_stack = types.SimpleNamespace(
            setCurrentWidget=moved.append)
        screen._figure_detail = detail

        screen._show_publication_sheet()

        assert moved == [detail]
        assert "QQ not shown (no residuals)" in _console_text(screen._console)


# -- the grid of a trial's saved figures --------------------------------------

def _write_png(path, colour="red"):
    from PySide6.QtGui import QColor, QPixmap

    pixmap = QPixmap(8, 8)
    pixmap.fill(QColor(colour))
    assert pixmap.save(str(path), "PNG")


class TestLoadingATrialsFigures:

    def test_the_screens_own_figures_are_shown_under_their_own_heading(
            self, screen, tmp_path):
        """They are NOT this run's, and a grid that mixed them in silently
        would caption a preprocessing figure as a result of the run."""
        run = tmp_path / "results" / "run_1"
        run.mkdir(parents=True)
        _write_png(run / "volcano.png")
        _write_png(tmp_path / "results" / "fraction_threshold.png", "blue")

        laid_out = {}
        screen._figure_grid = types.SimpleNamespace(
            set_figures=lambda pixmaps, titles, sections=(): laid_out.update(
                pixmaps=pixmaps, titles=titles, sections=list(sections)))
        screen._figures_stack = None

        assert screen._load_trial_figures(str(run)) == 2
        assert laid_out["titles"] == ["volcano", "fraction_threshold"]
        assert laid_out["sections"][-1] == (
            "the screen's own figures — not this run's", 1, 1)

    def test_a_run_whose_screen_folder_is_empty_gets_no_extra_section(
            self, screen, tmp_path):
        run = tmp_path / "results" / "run_1"
        run.mkdir(parents=True)
        _write_png(run / "volcano.png")

        laid_out = {}
        screen._figure_grid = types.SimpleNamespace(
            set_figures=lambda pixmaps, titles, sections=(): laid_out.update(
                titles=titles, sections=list(sections)))
        screen._figures_stack = None

        assert screen._load_trial_figures(str(run)) == 1
        assert laid_out["titles"] == ["volcano"]
        assert [name for name, _at, _count in laid_out["sections"]] == [
            "run_1"]


class TestTheGridTileWidth:

    def test_a_width_chosen_before_the_grid_exists_is_still_remembered(
            self, screen, monkeypatch):
        """It is a reading preference, so it has to outlive this screen."""
        from spacr.qt import preferences

        stored = []
        monkeypatch.setattr(preferences, "set_figure_grid_size", stored.append)
        widths = []
        screen._figure_grid = types.SimpleNamespace(
            set_target_cell_width=widths.append)

        screen._on_figure_size(320)
        assert widths == [320] and stored == [320]

        screen._figure_grid = None
        screen._on_figure_size(240)

        assert widths == [320]
        assert stored == [320, 240]


# -- which page of the results container is on screen -------------------------

class TestWhichPageIsShowing:

    def test_the_results_tab_answers_for_the_panel_it_holds(self, screen):
        """`showing_the_results` is what the run's end asks before saying
        anything about a table the user cannot see."""
        page = QWidget()
        screen._results_panel = page
        screen._results_tabs = types.SimpleNamespace(
            currentWidget=lambda: page)
        assert screen.showing_the_results() is True

        screen._results_tabs = types.SimpleNamespace(
            currentWidget=lambda: QWidget())

        assert screen.showing_the_results() is False

    def test_a_screen_with_no_figure_stack_shows_no_page(self, screen,
                                                          monkeypatch):
        """Every one of these is called from a signal that outlives the
        widgets, so each has to answer on a screen that is half torn down."""
        refreshed = []
        monkeypatch.setattr(screen, "_refresh_figure_grid",
                            lambda: refreshed.append(1))
        moved = []
        stack = types.SimpleNamespace(setCurrentIndex=moved.append,
                                      setCurrentWidget=moved.append)

        screen._figures_stack = stack
        screen._show_figure_grid()
        assert refreshed == [1] and moved == [0]

        screen._figures_stack = None
        screen._show_figure_grid()

        assert refreshed == [1] and moved == [0]

    def test_the_volcano_page_needs_both_a_stack_and_a_page(self, screen):
        moved = []
        page = QWidget()
        screen._figures_stack = types.SimpleNamespace(
            setCurrentWidget=moved.append)
        screen._volcano_page = page
        screen._show_regression_graph()
        assert moved == [page]

        screen._volcano_page = None
        screen._show_regression_graph()

        assert moved == [page]

    def test_a_gene_tile_that_is_already_open_is_not_reopened(self, screen,
                                                               monkeypatch):
        """Reasserting a size would fight anyone who had dragged it."""
        monkeypatch.setattr(screen, "_show_regression_graph", lambda: None)
        sized = []
        screen._gene_split = types.SimpleNamespace(
            sizes=lambda: [600, 0], setSizes=sized.append,
            height=lambda: 600)
        screen._gene_opened = False
        screen._on_guide_selected("gene_A")
        assert sized == [[360, 240]]

        screen._on_guide_selected("gene_A")

        assert sized == [[360, 240]]

    def test_a_pressed_tile_fills_the_container_with_that_figure(self,
                                                                  screen):
        moved = []
        detail = QWidget()
        screen._figure_detail = detail
        opened = []
        screen._figure_queue = types.SimpleNamespace(
            show_index=opened.append)

        screen._figures_stack = types.SimpleNamespace(
            setCurrentWidget=moved.append)
        screen._open_figure_from_grid(2)
        assert opened == [2] and moved == [detail]

        screen._figures_stack = None
        screen._open_figure_from_grid(3)

        assert opened == [2, 3]
        assert moved == [detail]


# -- the table a finished run hands back --------------------------------------

class TestTheRunsOwnTable:

    def test_a_panel_that_refuses_the_frame_gets_no_settings_after_it(
            self, screen, monkeypatch):
        """The run's settings are handed over only if the frame landed.

        Pushing them onto a panel still showing the previous run's table
        would describe that table with this run's model.
        """
        import pandas as pd

        monkeypatch.setattr(screen, "_update_run_in_runs_tab",
                            lambda **_kwargs: None)
        monkeypatch.setattr(screen, "_say_the_qc_verdict", lambda _p: None)
        taken = []
        screen._results_panel = types.SimpleNamespace(
            set_frame=lambda frame, source="": taken.append(source) and False)

        screen._on_pipeline_result({
            "results": pd.DataFrame({"gene": ["A"], "coefficient": [1.0]}),
            "res_folder": "/data/run_9"})

        assert taken == ["/data/run_9"]
        assert screen._last_run_folder == "/data/run_9"


# -- a settings file written before the Classes editor existed ----------------


class TestMigratingTheControlWells:

    def _classify_shaped(self, screen):
        screen._settings_model = types.SimpleNamespace(
            _widgets={"classes": QWidget()})

    def test_a_file_that_already_names_its_classes_is_left_alone(self,
                                                                 screen):
        """The trio is the OLD spelling; a file carrying both means the new."""
        self._classify_shaped(screen)
        already = {"location_column": "col", "positive_control": "A1",
                   "classes": [["A1"], ["B2"]]}

        assert screen._migrate_control_wells(already) is already

    def test_a_translation_that_fails_leaves_the_settings_untouched(
            self, screen, monkeypatch, caplog):
        """Rewriting them wrongly is worse than leaving the file as written."""
        from spacr import classify_classes

        self._classify_shaped(screen)
        old = {"location_column": "col", "positive_control": "A1",
               "negative_control": "B2"}

        monkeypatch.setattr(
            classify_classes, "normalize_settings",
            lambda settings: dict(settings, classes=[["A1"], ["B2"]]))
        migrated = screen._migrate_control_wells(old)
        assert migrated["classes"] == [["A1"], ["B2"]]

        def refusing(_settings):
            raise ValueError("a well name that is not a well")

        caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
        monkeypatch.setattr(classify_classes, "normalize_settings", refusing)

        assert screen._migrate_control_wells(old) is old
        assert "could not migrate the control wells" in caplog.text

    def test_a_trio_that_translates_to_nothing_is_not_written(self, screen,
                                                               monkeypatch):
        from spacr import classify_classes

        self._classify_shaped(screen)
        old = {"location_column": "col", "positive_control": "",
               "negative_control": ""}
        monkeypatch.setattr(classify_classes, "normalize_settings",
                            lambda settings: {"classes": []})

        assert screen._migrate_control_wells(old) is old


# -- the first count table a settings mapping names ---------------------------

class TestTheFirstCountFile:

    def test_a_row_with_an_empty_count_defers_to_the_next_row(self):
        assert aps._first_count_file({
            "paired_data": [{"count": "  ", "score": "p1_dv.csv"},
                            {"count": "p2_counts.csv"}],
        }) == "p2_counts.csv"

    def test_with_no_paired_row_naming_one_the_legacy_list_is_read(self):
        assert aps._first_count_file({
            "paired_data": [{"count": ""}],
            "count_data": ["legacy_counts.csv"],
        }) == "legacy_counts.csv"


# -- what a run is handed, and what it survives -------------------------------

class TestWhatTheRunIsHanded:

    def _entry(self, monkeypatch, fn):
        monkeypatch.setattr(
            "spacr.qt.screens.app_screen.resolve_pipeline_entry",
            lambda _key: fn)

    def test_a_umap_run_is_told_the_colours_on_the_gui_thread(
            self, qtbot, monkeypatch):
        """The worker must never read QApplication or QSettings.

        A pipeline module that imports PySide6 is unimportable on a cluster,
        so the canvas colours are resolved here and passed as plain strings.
        """
        from spacr.qt.theme import active_palette

        seen = {}
        self._entry(monkeypatch, lambda settings: seen.update(settings))
        screen = AppScreen("umap")
        qtbot.addWidget(screen)

        screen._on_run()
        qtbot.waitUntil(lambda: screen._btn_run.isEnabled(), timeout=20000)

        palette = active_palette()
        assert seen["_plot_theme"] == {
            "background": palette["surface_alt"],
            "foreground": palette["fg"],
            "border": palette["fg"],
        }

    def test_a_grid_that_will_not_mark_the_run_does_not_stop_it(
            self, qtbot, monkeypatch, caplog):
        """The run is marked on the grid at its START, so this fails before
        a single line of output has arrived -- and losing the run over a
        heading would be losing the run."""
        from spacr.qt import preferences

        caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
        seen = []
        self._entry(monkeypatch, lambda settings: seen.append(dict(settings)))
        screen = AppScreen("mask")
        qtbot.addWidget(screen)

        monkeypatch.setattr(preferences, "get_hash_inputs", lambda: True)
        screen._on_run()
        qtbot.waitUntil(lambda: screen._btn_run.isEnabled(), timeout=20000)
        assert seen[0]["hash_inputs"] is True

        def refusing(*_args, **_kwargs):
            raise RuntimeError("the grid has been destroyed")

        monkeypatch.setattr(screen._figure_queue, "mark_run", refusing)
        monkeypatch.setattr(preferences, "get_hash_inputs", refusing)

        screen._on_run()
        qtbot.waitUntil(lambda: len(seen) == 2, timeout=20000)
        qtbot.waitUntil(lambda: screen._btn_run.isEnabled(), timeout=20000)

        assert "could not mark the run on the figure grid" in caplog.text
        assert "could not read the hashing preference" in caplog.text
        assert "hash_inputs" not in seen[1]


# -- one truthful GPU state across the two UMAP pipelines ---------------------

class _Switch:
    def __init__(self, checked=False):
        self._checked = checked
        self.blocked = []

    def isChecked(self):
        return self._checked

    def setChecked(self, on):
        self._checked = bool(on)

    def blockSignals(self, on):
        self.blocked.append(bool(on))


class TestTheUmapGpuSwitch:

    def test_a_screen_with_no_settings_model_still_follows_the_panel(
            self, qtbot):
        """The panel is what decides; the model is only where it is recorded.

        The recorded half is driven first, so the silent half below is a
        missing model rather than a switch that never moved.
        """
        screen = AppScreen("umap")
        qtbot.addWidget(screen)
        screen._hyperparam = types.SimpleNamespace(
            request_gpu_enabled=lambda on, anchor=None: True)
        switch = screen._gpu_switch = _Switch(checked=False)

        recorded = []
        screen._settings_model = types.SimpleNamespace(
            set_hidden_value=lambda key, value: recorded.append((key, value)))
        screen._on_umap_gpu_switch(True)
        assert recorded == [("gpu", True)]
        assert switch.isChecked() is True

        screen._settings_model = None
        switch.setChecked(False)
        screen._on_umap_gpu_switch(True)

        assert switch.isChecked() is True
        assert recorded == [("gpu", True)]


# -- five branches nothing can drive, and why ---------------------------------
#
# Each of these is a defensive re-check placed after a call that has already
# guaranteed the condition, so no input reaches the other arm. They are left
# in the source unmarked: none is dead weight, and a `# pragma: no cover` on
# any of them would hide the day one of the guarantees below stops holding.
#
# 1. `_lay_out_one_waiting_row`, "the taken row has no field" and "the taken
#    row has no label" (`if field_item is None` / `if label_item is None`).
#    The row it takes back is `form.takeRow(form.rowCount() - 1)`, and the
#    statement immediately before it is `self._lay_out_setting_row(...)`,
#    whose only exit is `Section.add_row(lbl_widget, widget, wrap_label=True)`
#    -> `QFormLayout.addRow(form_label, widget)` -- a two-item row appended at
#    the end. `_lay_out_setting_row` contains no `return`, no `raise` and no
#    `try`, and `_attach_column_picker`, the last thing it calls, only
#    `replaceWidget`s the field in place. So the last row is always a labelled
#    row holding a field.
#
# 2. `_build_runtime_panel`, "this module's preview card is missing"
#    (`if getattr(self, card_attr, None) is not None`). Every key in
#    `preview_controls` -- mask, timelapse, motility, measure -- has an
#    `elif self.app_key == ...` arm earlier in the SAME method that assigns
#    the card and passes it straight to `QSplitter.addWidget`, which raises on
#    None. A screen that reaches the check therefore has its card.
#
# 3. `_build_runtime_panel`, "the figure queue cannot be clicked"
#    (`if queue is not None and hasattr(queue, "figure_clicked")`).
#    `self._figure_queue = FigureQueue(...)` runs earlier in the same method,
#    and `figure_clicked` is a class-level `Signal` on `FigureQueue`.
#
# 4. `apply_settings_dict`, "there is no model to mark as applying"
#    (`if model is not None: model._applying_settings = True`). The false arm
#    falls into `_apply_each_setting`, which dereferences
#    `self._settings_model._widgets` and calls
#    `self._settings_model._refresh_contextual_widgets()` unconditionally --
#    so a screen with no model raises `AttributeError` on the next statement
#    rather than completing the apply. Nothing can take that arm and return.
#
# 5. `_show_publication_sheet`, the `from spacr.figures import build_sheet`
#    fallback. The line it guards is `from ...figures import build_sheet`, and
#    a relative import is resolved against this module's `__package__`
#    ("spacr.qt.screens"), so both spellings name `spacr.figures`. The
#    fallback can only run when the first raises, and then it raises
#    identically.
