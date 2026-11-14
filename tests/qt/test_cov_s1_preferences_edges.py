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

import os

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
    assert str(tmp_path) in os.path.normpath(resolved), resolved
    return settings


# ---------------------------------------------------------------------------
# JSON that will not parse
# ---------------------------------------------------------------------------


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


def test_an_explicit_line_colour_is_not_resolved_through_the_theme(store):
    """"auto" means the text ink; anything else is the user's own answer and
    must survive a theme change untouched."""
    store.setValue(prefs._KEY_FIG_LINE, "#ff00ff")

    assert prefs.get_figure_line_colour() == "#ff00ff"


# ---------------------------------------------------------------------------
# One-time migrations, over a store that will not cooperate
# ---------------------------------------------------------------------------


def test_a_store_that_refuses_to_be_written_is_a_cosmetic_loss(store,
                                                               monkeypatch):
    """A read-only configuration directory is a real state, and the ambient
    migration is decoration. It may not take a launch down with it."""
    attempted = []

    def _refuses(*args, **_kwargs):
        attempted.append(args)
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(store, "setValue", _refuses)

    result = prefs._migrate_ambient_motion()

    assert result is None
    assert attempted == [
        (prefs._KEY_AMBIENT_SCALE, prefs.AMBIENT_MOTION_SCALE)
    ], "the migration reached its version-marker write before degrading"


# ---------------------------------------------------------------------------
# Figure colours pinned on another theme
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Asking the desktop what colour it is
# ---------------------------------------------------------------------------


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

    instance_lookups = []
    monkeypatch.setattr(
        QApplication, "instance",
        staticmethod(lambda: instance_lookups.append(True) or None))

    result = prefs.apply_preferences_to_app()

    assert result is None
    assert instance_lookups == [True], "the fallback application was consulted"


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
    from spacr.qt import theme as _theme
    # THE SHEET IS ON THE WINDOWS, NOT ON THE APPLICATION, since 380's
    # last lever moved it there -- a preference save with four modules
    # open went from 10.2 s to 3.1 s. `qapp.styleSheet()` is empty by
    # design now.
    assert _theme.window_stylesheet(qapp)


# ---------------------------------------------------------------------------
# The two resource dialogs
# ---------------------------------------------------------------------------


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
