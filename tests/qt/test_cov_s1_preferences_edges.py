"""A preferences store that has been hand-edited, and callers that pass junk.

``~/.config/spacr/qt.conf`` is a plain INI a user can open, and spaCR has to
survive what comes back out of it: a number typed as a word, a JSON blob with
a missing brace, a NaN. The rule this file pins is the same one throughout the
module -- **a preference that cannot be read is a preference at its default**,
never an exception and never a number that quietly means something else.

The setters follow the same rule from the other side: a value that cannot be
made into a number is clamped or replaced, because refusing to store one would
leave a dialog unable to close.
"""
from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings

from spacr.qt import preferences as prefs

pytestmark = pytest.mark.qt


@pytest.fixture
def store(qapp, tmp_path, monkeypatch):
    """A throwaway preferences store, so no test touches the real one."""
    settings = QSettings(str(tmp_path / "spacr-qt.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    resolved = prefs._settings().fileName()
    assert str(tmp_path) in resolved, resolved
    return settings


# ---------------------------------------------------------------------------
# JSON that will not parse
# ---------------------------------------------------------------------------

def test_a_truncated_folded_panel_blob_folds_nothing(store):
    """The fold state is one JSON string covering every module's panels.

    A half-written blob is not a reason to raise on the first screen the user
    opens; it is a reason to open every panel.
    """
    store.setValue(prefs._KEY_FOLDED, '{"mask/console": true')

    assert prefs.get_folded_panels() == {}

    store.setValue(prefs._KEY_FOLDED, '["mask/console"]')
    assert prefs.get_folded_panels() == {}


def test_a_panel_with_no_name_is_not_recorded_as_folded(store):
    """The key is ``"<module>/<panel>"`` and a caller with neither has nothing
    to remember -- storing it under ``""`` would fold an unnamed panel on
    every module at once."""
    prefs.set_folded_panel("mask/console", True)
    prefs.set_folded_panel("   ", True)

    assert prefs.get_folded_panels() == {"mask/console": True}


def test_a_truncated_figure_style_leaves_matplotlib_alone(store):
    store.setValue(prefs._KEY_FIG_STYLE, '{"font.size": 9')
    store.setValue(prefs._KEY_FIG_STYLE_PER_GRAPH, '{"volcano": ')

    assert prefs.get_figure_style() == {}
    assert prefs.get_figure_style_per_graph() == {}


def test_a_section_layout_that_is_not_a_mapping_is_discarded(store):
    """The layout store is keyed by panel. A blob that is a list carries no
    panel at all, and reading one out of it would raise on a screen that is
    only trying to restore its splitter."""
    store.setValue(prefs._KEY_SECTION_LAYOUT, '["folded"]')
    assert prefs.get_section_layout("mask") == {}

    prefs.set_section_layout("mask", ["Console"], [200, 100])
    assert prefs.get_section_layout("mask")["folded"] == ["Console"]

    store.setValue(prefs._KEY_SECTION_LAYOUT, "{not json")
    prefs.set_section_layout("measure", ["Figures"], [10])
    assert prefs.get_section_layout("measure") == {
        "folded": ["Figures"], "sizes": [10]}


# ---------------------------------------------------------------------------
# Numbers typed as words
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key, getter, expected", [
    ("_KEY_FIG_LIVE_CACHE", "get_figure_live_cache", "DEFAULT_FIG_LIVE_CACHE"),
    ("_KEY_FIG_PNG_DPI", "get_figure_png_dpi", "DEFAULT_PNG_DPI"),
    ("_KEY_MONTAGE_COLUMNS", "get_montage_columns", "DEFAULT_MONTAGE_COLUMNS"),
    ("_KEY_PANE_OPACITY", "get_pane_opacity", None),
    ("_KEY_RIM_PERIOD", "get_rim_period", "DEFAULT_RIM_PERIOD"),
])
def test_a_setting_typed_as_a_word_reads_as_its_default(store, key, getter,
                                                        expected):
    store.setValue(getattr(prefs, key), "lots")

    value = getattr(prefs, getter)()

    if expected is None:                       # pane opacity is a fraction
        assert value == pytest.approx(prefs.DEFAULT_PANE_OPACITY_PCT / 100.0)
    else:
        assert value == getattr(prefs, expected)


def test_a_figure_text_size_that_is_not_a_number_leaves_matplotlib_alone(
        store):
    """Zero is the "say nothing" value, and it is what an unreadable stored
    size has to become -- any other fallback would restyle every figure."""
    store.setValue(prefs._KEY_FIG_TEXT_SIZE, "large")

    assert prefs.get_figure_text_size() == 0


def test_an_unreadable_workspace_limit_reads_as_the_shipped_ceiling(store):
    from spacr.workspace import DEFAULT_COPY_LIMIT_MB

    store.setValue(prefs._KEY_WORKSPACE_COPY_LIMIT, "big")
    assert prefs.get_workspace_copy_limit_mb() == float(DEFAULT_COPY_LIMIT_MB)

    assert prefs.set_workspace_copy_limit_mb("big") == float(
        DEFAULT_COPY_LIMIT_MB)


def test_a_spinner_delay_of_nan_is_the_shipped_delay(store):
    """NaN survives a float() round trip through an INI, and every comparison
    against it is False -- so the clamp below would let it through and the
    spinner would never appear."""
    store.setValue(prefs._KEY_SPINNER_DELAY, "nan")
    assert prefs.get_spinner_delay() == prefs.DEFAULT_SPINNER_DELAY

    store.setValue(prefs._KEY_SPINNER_DELAY, "soon")
    assert prefs.get_spinner_delay() == prefs.DEFAULT_SPINNER_DELAY

    prefs.set_spinner_delay("soon")
    assert prefs.get_spinner_delay() == prefs.DEFAULT_SPINNER_DELAY
    prefs.set_spinner_delay(float("nan"))
    assert prefs.get_spinner_delay() == prefs.DEFAULT_SPINNER_DELAY


def test_an_ambient_multiplier_of_nan_is_the_engine_s_default(store):
    (low, _high), default = prefs._ambient_ranges()[0]
    store.setValue(prefs._KEY_AMBIENT_SCALE, prefs.AMBIENT_MOTION_SCALE)
    store.setValue(prefs._KEY_AMBIENT_BLUR, "nan")

    assert prefs.get_ambient_blur() == default

    prefs.set_ambient_blur("thick")
    assert prefs.get_ambient_blur() == max(low, min(_high, default))
    prefs.set_ambient_blur(float("nan"))
    assert prefs.get_ambient_blur() == max(low, min(_high, default))


@pytest.mark.parametrize("setter, getter, default", [
    ("set_rim_length", "get_rim_length", "DEFAULT_RIM_LENGTH"),
    ("set_rim_lag", "get_rim_lag", "DEFAULT_RIM_LAG"),
    ("set_rim_period", "get_rim_period", "DEFAULT_RIM_PERIOD"),
])
def test_a_rim_setting_given_a_word_falls_back_rather_than_refusing(
        store, setter, getter, default):
    """The rim controls are spin boxes in a dialog. A setter that raised
    would leave the dialog unable to close on a value the user cannot see."""
    stored = getattr(prefs, setter)("as fast as possible")

    assert stored == getattr(prefs, default)
    assert getattr(prefs, getter)() == getattr(prefs, default)


# ---------------------------------------------------------------------------
# Booleans that arrive as INI strings
# ---------------------------------------------------------------------------

def test_a_boolean_written_as_a_word_still_reads_as_a_boolean(store):
    """QSettings hands back strings from an INI file, so ``"true"`` has to
    mean True -- ``bool("false")`` is True and would invert the setting."""
    store.setValue(prefs._KEY_FIG_DYNAMIC, "yes")
    assert prefs.get_figure_dynamic() is True
    store.setValue(prefs._KEY_FIG_DYNAMIC, "off")
    assert prefs.get_figure_dynamic() is False

    store.setValue(prefs._KEY_VERBOSE_LOG, "true")
    assert prefs.get_verbose_logging() is True
    store.setValue(prefs._KEY_VERBOSE_LOG, "no")
    assert prefs.get_verbose_logging() is False


def test_a_log_level_list_and_a_trailing_comma_both_read(store):
    """QSettings splits a comma-separated INI value into a list on some
    platforms and hands back the raw string on others; both spellings, and a
    stray separator in either, name the same levels."""
    store.setValue(prefs._KEY_LOG_FILE_LEVELS, ["INFO", "WARNING"])
    assert prefs.get_log_file_levels() == prefs._parse_levels(
        "INFO,WARNING", ())

    store.setValue(prefs._KEY_LOG_FILE_LEVELS, "INFO,,WARNING,")
    assert prefs.get_log_file_levels() == prefs._parse_levels(
        "INFO,WARNING", ())


# ---------------------------------------------------------------------------
# Values a caller may not choose at all
# ---------------------------------------------------------------------------

def test_an_unsupported_figure_format_is_refused_by_name(store):
    """The format decides which writer runs. Storing an unknown one would
    fail later, inside a save, with the figure already drawn."""
    with pytest.raises(ValueError) as excinfo:
        prefs.set_figure_format("tiff")

    assert "unknown figure format 'tiff'" in str(excinfo.value)
    assert str(prefs.VALID_FIG_FORMATS) in str(excinfo.value)


def test_an_unknown_theme_choice_lists_the_ones_that_exist(store):
    with pytest.raises(ValueError) as excinfo:
        prefs.set_theme_choice("neon")

    assert "unknown theme choice 'neon'" in str(excinfo.value)


def test_a_theme_with_variants_round_trips_through_its_composite_token(store):
    """``cell`` is a family, not a theme. The dropdown stores one token, and
    reading it back has to name the variant again or the dialog opens on the
    wrong row."""
    tokens = [token for _label, token in prefs.theme_choices()]
    cells = [t for t in tokens if t.startswith("cell:")]
    assert cells, "the cell family should offer at least one wallpaper"

    prefs.set_theme_choice(cells[-1])

    assert prefs.get_theme() == "cell"
    assert prefs.get_theme_choice() == cells[-1]


def test_an_explicit_line_colour_is_not_resolved_through_the_theme(store):
    """"auto" means the text ink; anything else is the user's own answer and
    must survive a theme change untouched."""
    store.setValue(prefs._KEY_FIG_LINE, "#ff00ff")

    assert prefs.get_figure_line_colour() == "#ff00ff"


# ---------------------------------------------------------------------------
# One-time migrations, over a store that will not cooperate
# ---------------------------------------------------------------------------

def test_a_scale_marker_typed_by_hand_does_not_block_a_migration(store):
    """The marker is the "already done" flag, and it is a plain INI number.

    Reading a word there has to be treated as "not migrated" rather than
    raising on the first figure the user draws.
    """
    store.setValue(prefs._KEY_FIG_COLOR_SCALE, "done")
    store.setValue(prefs._KEY_FIG_BG, "#000000")
    store.setValue(prefs._KEY_FIG_FG, "#ffffff")

    prefs._migrate_frozen_figure_colors()

    assert prefs._settings().value(prefs._KEY_FIG_BG) == \
        prefs.AUTO_FIGURE_COLOR
    assert int(store.value(prefs._KEY_FIG_COLOR_SCALE)) == \
        prefs.FIGURE_COLOR_SCALE


def test_an_ambient_scale_marker_typed_by_hand_does_not_block_a_migration(
        store):
    store.setValue(prefs._KEY_AMBIENT_SCALE, "done")
    store.setValue(prefs._KEY_AMBIENT_BLUR, 2.0)

    prefs._migrate_ambient_motion()

    assert int(store.value(prefs._KEY_AMBIENT_SCALE)) == \
        prefs.AMBIENT_MOTION_SCALE
    assert float(store.value(prefs._KEY_AMBIENT_BLUR)) == pytest.approx(1.0)


def test_a_store_that_refuses_to_be_written_is_a_cosmetic_loss(store,
                                                               monkeypatch):
    """A read-only configuration directory is a real state, and the ambient
    migration is decoration. It may not take a launch down with it."""
    def _refuses(*_args, **_kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(store, "setValue", _refuses)

    prefs._migrate_ambient_motion()      # must not raise


# ---------------------------------------------------------------------------
# Figure colours pinned on another theme
# ---------------------------------------------------------------------------

def test_a_text_colour_no_auto_theme_produces_is_left_alone(store, capsys):
    """The repair only ever undoes spaCR's own frozen pairs.

    A user who typed their own text colour and a background that happens to
    be black must keep it -- handing that back to the theme would silently
    discard a deliberate choice.
    """
    store.setValue(prefs._KEY_FIG_BG, "#000000")
    store.setValue(prefs._KEY_FIG_FG, "#3366cc")

    prefs._unfreeze_figure_colors_that_fight_the_theme()

    assert store.value(prefs._KEY_FIG_FG) == "#3366cc"
    assert store.value(prefs._KEY_FIG_BG) == "#000000"
    assert capsys.readouterr().out == ""


def test_a_preferences_read_that_raises_leaves_the_colours_where_they_are(
        store, monkeypatch, capsys):
    """This runs on the render path. A cosmetic repair that raised there
    would stop a figure being drawn at all."""
    def _broken():
        raise RuntimeError("the settings backend is gone")

    monkeypatch.setattr(prefs, "_settings", _broken)

    prefs._unfreeze_figure_colors_that_fight_the_theme()   # must not raise

    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Asking the desktop what colour it is
# ---------------------------------------------------------------------------

def test_a_desktop_that_cannot_be_asked_gets_the_dark_theme(store,
                                                            monkeypatch):
    """"Follow system" has to resolve to something. Qt's palette hint is not
    available on every desktop, and a raise there would leave the app with no
    theme at all."""
    from PySide6.QtWidgets import QApplication

    prefs.set_theme("system")

    def _no_palette(_self):
        raise AttributeError("no colour scheme hint on this platform")

    monkeypatch.setattr(QApplication, "palette", _no_palette)

    assert prefs.resolve_effective_theme() == "dark"


# ---------------------------------------------------------------------------
# The backdrops, and the widgets that have already gone
# ---------------------------------------------------------------------------

def test_a_backdrop_whose_c_half_is_gone_does_not_stop_the_others(
        store, qtbot, monkeypatch):
    """One dead widget in the tree used to skip every widget after it.

    The loop is per-widget precisely so that a screen closed while
    Preferences was open cannot cost every other backdrop its settings.
    """
    from spacr.qt.widgets.ambient import AmbientWidget

    class _Gone(AmbientWidget):
        def setVisible(self, on):
            raise RuntimeError("Internal C++ object already deleted.")

    class _Fine(AmbientWidget):
        def __init__(self):
            super().__init__()
            self.told = []

        def set_animating(self, on):
            self.told.append(bool(on))

    dead, alive = _Gone(), _Fine()
    qtbot.addWidget(dead)
    qtbot.addWidget(alive)
    prefs.set_ambient_enabled(True)

    prefs.apply_ambient_preferences()

    assert alive.told and alive.told[-1] is True


# ---------------------------------------------------------------------------
# Minimising and restoring the visuals
# ---------------------------------------------------------------------------

def test_an_animation_that_cannot_be_switched_off_still_minimises_the_rest(
        store, monkeypatch):
    """Performance mode is a promise about frames. If the animation key
    cannot be written, the detail and density reductions still have to
    land."""
    ranges = prefs._ambient_ranges()

    def _refuses(_key):
        raise ValueError("no such animation")

    monkeypatch.setattr(prefs, "set_ambient_animation", _refuses)

    prefs._minimise_visuals()

    assert prefs.get_ambient_resolution() == ranges[3][0][0]
    assert prefs.get_ambient_density() == ranges[4][0][0]
    assert prefs.get_field_fade_enabled() is False


def test_a_stash_that_cannot_be_put_back_says_it_did_not(store, monkeypatch):
    """The user keeps the minimums they can see and change, which is better
    than being handed somebody's idea of a default."""
    store.setValue(prefs._KEY_MODE_VISUAL_STASH,
                   json.dumps({"ambient_animation": "aurora"}))

    def _refuses(_value):
        raise ValueError("that animation is not installed")

    monkeypatch.setattr(prefs, "set_ambient_animation", _refuses)

    assert prefs._restore_visuals() is False
    assert not store.value(prefs._KEY_MODE_VISUAL_STASH, "")


def test_turning_the_field_fade_off_headless_is_not_an_error(store,
                                                             monkeypatch):
    """The paint hook's cache cannot be stale if it was never built, so a
    module that will not import is not a reason to refuse the setting."""
    import spacr.qt.widgets.field_fade as ff

    def _refuses():
        raise RuntimeError("no painter in this process")

    monkeypatch.setattr(ff, "invalidate_field_fade", _refuses)

    prefs.set_field_fade_enabled(False)

    assert prefs.get_field_fade_enabled() is False


# ---------------------------------------------------------------------------
# Pushing everything onto a live application
# ---------------------------------------------------------------------------

def test_applying_preferences_with_no_application_does_nothing(store,
                                                               monkeypatch):
    """``apply_preferences_to_app`` is called from module import paths that
    run before (and after) there is an app."""
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))

    prefs.apply_preferences_to_app()      # must not raise


def test_every_optional_step_of_a_preferences_save_may_fail_on_its_own(
        store, qapp, monkeypatch, caplog):
    """A save re-applies half a dozen independent things.

    Each is guarded separately because a workspace push, a field-fade
    install, a console zoom or a logging policy that fails must still leave
    the user with the theme and font they just chose.
    """
    import spacr.qt.widgets.field_fade as ff
    import spacr.logging_util as logging_util

    def _refuses(*_args, **_kwargs):
        raise RuntimeError("this step is unavailable")

    monkeypatch.setattr(prefs, "apply_workspace_preference", _refuses)
    monkeypatch.setattr(ff, "install_field_fade", _refuses)
    monkeypatch.setattr(ff, "repaint_fields", _refuses)
    monkeypatch.setattr(logging_util, "apply_level_policy", _refuses)
    monkeypatch.setattr(prefs, "get_font_scale", lambda: 1.0)
    import spacr.qt.widgets.console_panel as console_panel
    monkeypatch.delattr(console_panel, "ConsolePanel")

    with caplog.at_level(logging.DEBUG, logger="spacr.qt.preferences"):
        prefs.apply_preferences_to_app(qapp)

    assert qapp.property("spacrLanguage") == prefs.get_language()
    assert qapp.styleSheet()


# ---------------------------------------------------------------------------
# The two resource dialogs
# ---------------------------------------------------------------------------

def test_a_resource_action_is_confirmed_by_a_button_that_names_it(store,
                                                                  qtbot,
                                                                  monkeypatch):
    """"Are you sure?" is not a question anybody can answer.

    The accept button carries the action's own title, Cancel is the default
    so a stray Return does nothing, and the informative text says what the
    action cannot do before it happens.
    """
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt import resource_cleanup

    seen = {}

    def _press(self, which):
        seen["object"] = self.objectName()
        seen["informative"] = self.informativeText()
        seen["default"] = self.defaultButton().text()
        for button in self.buttons():
            if self.buttonRole(button) == which:
                button.click()
                return 0
        return 0

    monkeypatch.setattr(QMessageBox, "exec",
                        lambda self: _press(self, QMessageBox.AcceptRole))
    assert prefs.confirm_resource_action("ram") is True
    assert seen["object"] == "ResourceActionConfirm"
    assert seen["informative"] == resource_cleanup.confirmation_text("ram")
    assert seen["default"] == "Cancel"

    monkeypatch.setattr(QMessageBox, "exec",
                        lambda self: _press(self, QMessageBox.RejectRole))
    assert prefs.confirm_resource_action("ram") is False


def test_the_result_dialog_reports_what_actually_happened(store, qtbot,
                                                          monkeypatch):
    """Freeing nothing is a real outcome and has to be shown as one, with
    the per-item detail available but not shouted."""
    from PySide6.QtWidgets import QMessageBox

    from spacr.qt import resource_cleanup

    shown = {}

    def _capture(self):
        shown["object"] = self.objectName()
        shown["text"] = self.text()
        shown["details"] = self.detailedText()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", _capture)
    result = resource_cleanup.DiskReport(note="No project folder is known yet.")

    prefs._show_resource_result("disk", result)

    assert shown["object"] == "ResourceActionResult"
    assert shown["text"] == result.summary()
    assert shown["details"] == ""

    class _PerItem:
        details = ("plate1/merged: 2.1 GB", "plate2/merged: 0.4 GB")

        def summary(self):
            return "Freed 2.5 GB across two projects."

    prefs._show_resource_result("disk", _PerItem())

    assert shown["text"] == "Freed 2.5 GB across two projects."
    assert shown["details"] == ("plate1/merged: 2.1 GB\n"
                                "plate2/merged: 0.4 GB")


# ---------------------------------------------------------------------------
# Explaining the rows, and telling the cards
# ---------------------------------------------------------------------------

def test_a_form_row_whose_label_is_not_a_label_is_skipped(store, qtbot):
    """A form can hold a spanning row or a widget used as its own label.

    Reading ``.text()`` off one would raise while the Preferences dialog is
    being built, which is the one moment the user cannot recover from.
    """
    from PySide6.QtWidgets import (QCheckBox, QFormLayout, QLabel, QWidget)

    holder = QWidget()
    qtbot.addWidget(holder)
    form = QFormLayout(holder)
    form.addRow(QCheckBox("not a label"), QCheckBox("field"))
    form.addRow(QLabel("Theme"), QCheckBox("field"))

    explained = prefs.explain_every_row(holder)

    assert isinstance(explained, int)
    assert form.rowCount() == 2


def test_nothing_is_told_about_the_rim_when_there_is_no_application(
        store, monkeypatch):
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))

    assert prefs._tell_the_cards_the_rim_changed() == 0


def test_a_card_module_that_will_not_import_tells_nobody(store, monkeypatch):
    """The rim preference is decoration on one widget type. A build without
    that widget must still be able to save preferences."""
    import builtins

    real_import = builtins.__import__

    def _no_setup_card(name, globals=None, locals=None, fromlist=(), level=0):
        if "setup_card" in name or "setup_card" in (fromlist or ()):
            raise ImportError("no setup card in this build")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _no_setup_card)

    assert prefs._tell_the_cards_the_rim_changed() == 0
